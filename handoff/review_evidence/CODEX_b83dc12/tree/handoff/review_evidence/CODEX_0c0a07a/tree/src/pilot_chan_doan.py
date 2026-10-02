"""
pilot_chan_doan.py -- Do chan doan cho PILOT 2 epoch (vong 37; nguoi dung duyet: YOLOv8n + YOLO11n, moi model 1 luot,
dataset smoke da khoa). Dung boi `train_detector.py --che-do pilot`.

Diem do (cung MOT quy trinh cho moi diem):
  - TRUOC HOC: callback `on_train_start` -- model 5 lop trainer da dung xong (doi nc, nap / remap trong so), optimizer,
    scheduler, EMA da tao, CHUA co forward/backward nao. Ghi bang chung: ten + thu tu 5 lop; tensor nao chep nguyen tu
    checkpoint COCO, tensor nao remap hang theo ten lop, tensor nao khoi tao moi (doi chieu doc lap voi checkpoint);
    optimizer chua co trang thai, so lan cap nhat (ema.updates) = 0, gradient chua co.
  - SAU MOI EPOCH: callback `on_fit_epoch_end` TRONG vong lap (da validate + luu checkpoint) -- do model HIEN TAI
    (trong so tho) va EMA (ban ultralytics dung de validate va ghi checkpoint), tach ro hai nguon.
    Lan goi `on_fit_epoch_end` thu hai trong `final_eval` (validate best.pt) KHONG phai epoch: chi ghi thoi diem.
  - SAU TRAIN: checkpoint last.pt / best.pt nap tu dia (EMA FP16 da luu).

Moi phep do:
  1. DANH GIA CHINH (giu nguyen quy tac): ban sao FP32 cua module dang do luu vao file tam trong thu muc luot, roi goi
     NGUYEN `eval_detector_5lop.du_doan_ultralytics` + `danh_gia` (imgsz 640, conf 0,001, NMS IoU 0,7 theo lop,
     max_det 300, FP32) tren 84 anh dev smoke -> so detection, phan bo confidence, AP theo lop co GT.
  2. CHAN DOAN DIEM TRUOC NGUONG (KHONG phai AP, KHONG ha nguong danh gia): cung predictor ultralytics (cung letterbox,
     tien xu ly, mo hinh da fuse), ghi sigmoid diem lop LON NHAT tren moi anchor cua dau ra tho Detect -- truoc NMS va
     truoc loc conf -- cho tung anh va tung lop.
  3. BAT BIEN TRANG THAI TRAINING: chup trang thai (tham so + buffer gom thong ke BatchNorm, co train/eval tung module,
     gradient, requires_grad, EMA + so lan cap nhat, optimizer, scheduler, scaler, bien dem trainer, RNG python / numpy /
     torch / cuda, co cudnn / deterministic / grad mode) TRUOC va SAU phep do; RNG va co duoc luu + khoi phuc. Ghi ca
     thay doi TRUOC khi khoi phuc (de thay phep do co cham vao gi khong); con khac SAU khoi phuc -> LoiPilot (dung train).
Ghi them theo batch: loss, LR + momentum tung nhom tham so, accumulate, ema.updates (tang 1 moi lan optimizer cap
nhat), thoi diem; theo epoch: trang thai mosaic thuc te, so batch, so lan cap nhat, dinh bo nho torch; so buoc warmup
tinh bang CHINH ham `_get_warmup_iterations` cua phien ban dang chay (khong suy tu so epoch).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

TU_KHOA_LOG = ("Overriding model.yaml", "Remapped", "Transferred", "Freezing layer", "Closing dataloader mosaic",
               "optimizer:", "EarlyStopping", "WARNING")
BIN_CONF = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.000001]


class LoiPilot(RuntimeError):
    """Phep do chan doan lam thay doi trang thai training (hoac khong do duoc) -> dung truoc khi train tiep."""


# =========================================================================== tien ich
def _bam_tensor(t) -> str | None:
    import torch
    if t is None:
        return None
    a = t.detach()
    if a.is_cuda:
        a = a.cpu()
    a = a.contiguous()
    if a.dtype == torch.bfloat16:
        a = a.float()
    h = hashlib.sha256(a.numpy().tobytes())
    h.update(f"{a.dtype}|{tuple(a.shape)}".encode())
    return h.hexdigest()[:24]


def _bam_obj(o) -> str:
    return hashlib.sha256(repr(o).encode()).hexdigest()[:24]


def _unwrap(m):
    try:
        from ultralytics.utils.torch_utils import unwrap_model
        return unwrap_model(m)
    except ImportError:
        return getattr(m, "module", m)


def luu_rng() -> dict:
    import torch
    s = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def khoi_phuc_rng(s: dict) -> None:
    import torch
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s:
        torch.cuda.set_rng_state_all(s["cuda"])


def luu_co() -> dict:
    import torch
    return {"cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_enabled": torch.backends.cudnn.enabled, "det_alg": torch.are_deterministic_algorithms_enabled(),
            "det_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}


def khoi_phuc_co(c: dict) -> None:
    import torch
    torch.backends.cudnn.benchmark = c["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = c["cudnn_deterministic"]
    torch.backends.cudnn.enabled = c["cudnn_enabled"]
    torch.use_deterministic_algorithms(c["det_alg"], warn_only=c["det_warn_only"])


def chup_trang_thai(tr) -> dict:
    """Anh chup (hash) moi trang thai co the anh huong cac buoc train sau. Chi DOC, khong sua gi."""
    import torch
    m = _unwrap(tr.model)
    s = {"model_tham_so_buffer": {k: _bam_tensor(v) for k, v in m.state_dict().items()},
         "model_che_do_train": _bam_obj([x.training for x in m.modules()]),
         "model_gradient": {n: _bam_tensor(p.grad) for n, p in m.named_parameters()},
         "model_requires_grad": _bam_obj([p.requires_grad for p in m.parameters()])}
    ema = getattr(tr, "ema", None)
    if ema is not None:
        s["ema_tham_so_buffer"] = {k: _bam_tensor(v) for k, v in ema.ema.state_dict().items()}
        s["ema_che_do_train"] = _bam_obj([x.training for x in ema.ema.modules()])
        s["ema_updates"] = int(ema.updates)
    opt = getattr(tr, "optimizer", None)
    if opt is not None:
        sd = opt.state_dict()
        s["optimizer_state"] = {f"{i}.{k}": (_bam_tensor(v) if torch.is_tensor(v) else repr(v))
                                for i, st in sd["state"].items() for k, v in st.items()}
        s["optimizer_param_groups"] = _bam_obj([{k: v for k, v in g.items()} for g in sd["param_groups"]])
    sch = getattr(tr, "scheduler", None)
    if sch is not None:
        s["scheduler"] = _bam_obj({k: v for k, v in sch.state_dict().items() if k != "lr_lambdas"})
    sc = getattr(tr, "scaler", None)
    if sc is not None:
        s["scaler"] = _bam_obj(sc.state_dict())
    s["trainer_bien"] = _bam_obj([getattr(tr, k, None) for k in ("accumulate", "epoch", "best_fitness", "stop")])
    s["rng_python"] = _bam_obj(random.getstate())
    s["rng_numpy"] = _bam_obj([x.tolist() if hasattr(x, "tolist") else x for x in np.random.get_state()])
    s["rng_torch"] = _bam_tensor(torch.get_rng_state())
    if torch.cuda.is_available():
        s["rng_cuda"] = [_bam_tensor(x) for x in torch.cuda.get_rng_state_all()]
    s["co"] = _bam_obj({**luu_co(), "grad": torch.is_grad_enabled(), "inference": torch.is_inference_mode_enabled()})
    return s


def so_sanh_trang_thai(a: dict, b: dict) -> list[str]:
    khac = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            kk = sorted(x for x in set(va) | set(vb) if va.get(x) != vb.get(x))
            if kk:
                khac.append(f"{k}: {len(kk)} muc khac (vd {kk[:3]})")
        elif va != vb:
            khac.append(k)
    return khac


def kiem_nap_trong_so(model, ckpt_khoi_tao: str | Path) -> dict:
    """Doi chieu DOC LAP model 5 lop (truoc buoc hoc) voi checkpoint COCO khoi tao: tensor nao chep nguyen, tensor nao
    remap hang theo ten lop (nhanh lop cuoi cua Detect, nhu `_remap_cls_by_names`), tensor nao khoi tao moi."""
    import torch
    from ultralytics.nn.modules import Detect
    ck = torch.load(str(ckpt_khoi_tao), map_location="cpu", weights_only=False)
    src = (ck.get("ema") or ck["model"]).float()
    csd = src.state_dict()
    m = _unwrap(model)
    msd = m.state_dict()
    src_names, tgt_names = dict(src.names), dict(m.names)
    norm = lambda s: str(s).strip().lower()  # noqa: E731
    lookup = {norm(v): int(k) for k, v in src_names.items()}
    idx = [lookup.get(norm(tgt_names[k]), -1) for k in range(len(tgt_names))]
    nc = len(tgt_names)
    cls_keys = {f"{name}.{attr}.{i}.{len(seq) - 1}.{p}"
                for name, mm in m.named_modules() if isinstance(mm, Detect)
                for attr in ("cv3", "one2one_cv3") for i, seq in enumerate(getattr(mm, attr, ()))
                if getattr(seq[-1], "out_channels", None) == nc for p in ("weight", "bias")}
    chep, remap, khac_gia_tri, moi = [], [], [], []
    for k, v in msd.items():
        v = v.detach().cpu()
        if k in cls_keys and k in csd and csd[k].shape[1:] == v.shape[1:]:
            ok = all(torch.equal(v[j], csd[k][idx[j]].to(v.dtype)) for j in range(nc) if idx[j] >= 0)
            (remap if ok else khac_gia_tri).append(k)
        elif k in csd and csd[k].shape == v.shape:
            (chep if torch.equal(csd[k].to(v.dtype), v) else khac_gia_tri).append(k)
        else:
            moi.append({"ten": k, "shape_model": list(v.shape),
                        "shape_checkpoint": list(csd[k].shape) if k in csd else None})
    return {"checkpoint": str(ckpt_khoi_tao), "so_lop_checkpoint": len(src_names), "ten_lop_model": tgt_names,
            "chi_so_lop_checkpoint_theo_ten": {tgt_names[j]: idx[j] for j in range(nc)},
            "tong_tensor_model": len(msd), "chep_nguyen": len(chep), "remap_hang_theo_ten": len(remap),
            "tensor_remap": sorted(remap), "cung_shape_khac_gia_tri": sorted(khac_gia_tri),
            "khoi_tao_moi": len(moi), "tensor_khoi_tao_moi": moi,
            "doi_chieu_log_ultralytics": f"'Transferred N/M' cua ultralytics = chep_nguyen + remap = {len(chep) + len(remap)}"
                                        f"/{len(msd)}",
            "ghi_chu": "khoi_tao_moi = khong chep duoc vi lech shape hoac khong co trong checkpoint (vd nhanh cv3 khi so "
                       "lop doi lam doi do rong an); remap_hang_theo_ten so tung hang lop dich voi hang lop COCO cung ten"}


def luu_ban_sao_tam(module, path: Path, train_args) -> str:
    """Ban sao FP32 cua module (khong cham module goc) theo dinh dang checkpoint ultralytics nap duoc; tra sha256."""
    import torch
    m = deepcopy(_unwrap(module)).cpu().float()
    if hasattr(m, "criterion"):
        m.criterion = None
    for p in m.parameters():
        p.grad = None
        p.requires_grad_(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    args = dict(vars(train_args)) if not isinstance(train_args, dict) else dict(train_args)
    torch.save({"model": m, "train_args": args, "ghi_chu": "ban sao chan doan pilot (FP32), KHONG phai checkpoint train"},
               str(path))
    del m
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diem_truoc_nguong(weights: str, anh: list[Path], qt: dict, device, batch: int = 16) -> tuple[np.ndarray, dict]:
    """Diem lop lon nhat (sigmoid) tren moi anchor cua dau ra tho Detect, truoc NMS va truoc loc conf; (n_anh, nc).
    Cung predictor + tien xu ly voi `eval_detector_5lop.du_doan_ultralytics` (cung imgsz / half / classes)."""
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionPredictor

    from detector_classes import anh_xa_model_sang_lop5
    ghi, hinh = [], set()

    class _PredictorGhiDiem(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            p = preds[0] if isinstance(preds, (list, tuple)) else preds
            ghi.append(p[:, 4:, :].float().amax(dim=2).cpu().numpy())
            hinh.add((tuple(p.shape[1:]), tuple(img.shape[1:])))
            return super().postprocess(preds, img, orig_imgs, **kwargs)

    y = YOLO(str(weights))
    keep = sorted(anh_xa_model_sang_lop5(y.names))
    for i in range(0, len(anh), batch):
        y.predict([str(p) for p in anh[i:i + batch]], imgsz=qt["imgsz"], conf=qt["conf"], iou=qt["iou_nms"],
                  max_det=qt["max_det"], classes=keep, agnostic_nms=qt["agnostic_nms"], half=qt["half"],
                  device=device, verbose=False, predictor=_PredictorGhiDiem)
    args = y.predictor.args            # ultralytics 8.4.141: 'half' da doi thanh 'quantize' (None = FP32), khong co .half
    info = {"imgsz": args.imgsz, "rect_letterbox": bool(args.rect), "quantize": getattr(args, "quantize", None),
            "batch_predict": args.batch, "hinh_dau_ra_va_dau_vao": sorted([list(a), list(b)] for a, b in hinh)}
    del y
    return np.concatenate(ghi, axis=0) if ghi else np.zeros((0, 0)), info


def _phan_vi(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n": 0}
    q = np.quantile(x, [0.1, 0.5, 0.9])
    return {"n": int(len(x)), "min": float(x.min()), "p10": float(q[0]), "trung_vi": float(q[1]), "p90": float(q[2]),
            "max": float(x.max())}


def do_mo_hinh(weights: str | Path, anh: list[Path], gt: list[dict], qt: dict, device) -> dict:
    """Danh gia chinh (nguyen eval_detector_5lop, conf qt['conf']) + chan doan diem truoc nguong cho MOT file weights."""
    import eval_detector_5lop as ev
    from detector_classes import LOP_5
    t0 = time.time()
    dd, mt = ev.du_doan_ultralytics(str(weights), anh, qt, device=device)
    r = ev.danh_gia(gt, dd)
    t1 = time.time()
    conf = np.concatenate([d["conf"] for d in dd]) if dd else np.zeros(0)
    cls = np.concatenate([d["cls"] for d in dd]) if dd else np.zeros(0, int)
    hist, _ = np.histogram(conf, bins=BIN_CONF)
    tho, info = diem_truoc_nguong(str(weights), anh, qt, device)
    t2 = time.time()
    max_anh = tho.max(axis=1) if tho.size else np.zeros(0)
    return {
        "weights": str(weights), "sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(), "anh_xa_lop": mt,
        "quy_tac_du_doan": qt, "so_anh": len(anh),
        "danh_gia_chinh": {"theo_lop": {k: {x: v[x] for x in ("n_gt", "n_det_conf_thap", "AP50", "AP50_95")} |
                                        {"van_hanh": v["van_hanh"]} for k, v in r["theo_lop"].items()},
                           "macro": r["macro"], "van_hanh": r["van_hanh"]},
        "detection": {"tong": int(len(conf)), "theo_lop": {c: int((cls == i).sum()) for i, c in enumerate(LOP_5)},
                      "so_anh_co_detection": int(sum(len(d["conf"]) > 0 for d in dd)),
                      "confidence": _phan_vi(conf),
                      "histogram": {f"[{BIN_CONF[i]:g}, {min(BIN_CONF[i + 1], 1.0):g}{']' if i == len(BIN_CONF) - 2 else ')'}":
                                    int(hist[i]) for i in range(len(hist))}},
        "diem_truoc_nguong": {
            "cach_tinh": "sigmoid(logit lop) cua dau ra tho Detect, TRUOC NMS va TRUOC loc conf; lay max tren moi anchor "
                         "cua tung anh (va tung lop); cung predictor, letterbox, fuse, FP32 nhu danh gia chinh. CHI de chan "
                         "doan, KHONG dung tinh AP, KHONG thay nguong danh gia",
            "tien_xu_ly": info,
            "max_moi_anh": _phan_vi(max_anh),
            "so_anh_max_ge_0_001": int((max_anh >= 0.001).sum()), "so_anh_max_ge_0_25": int((max_anh >= 0.25).sum()),
            "theo_lop_trung_vi_max_moi_anh": {c: float(np.median(tho[:, i])) for i, c in enumerate(LOP_5)}
            if tho.size else {},
            "theo_lop_max": {c: float(tho[:, i].max()) for i, c in enumerate(LOP_5)} if tho.size else {}},
        "thoi_gian_s": {"danh_gia_chinh": round(t1 - t0, 2), "diem_truoc_nguong": round(t2 - t1, 2)},
    }


class _BatLog(logging.Handler):
    def __init__(self, dich: list):
        super().__init__(level=logging.INFO)
        self.dich = dich

    def emit(self, record):
        try:
            msg = record.getMessage()
            if any(k in msg for k in TU_KHOA_LOG):
                self.dich.append({"t": round(time.time(), 3), "dong": msg.strip()[:300]})
        except Exception:  # noqa: BLE001
            pass


# =========================================================================== ghi + do trong luc train
class GhiPilot:
    def __init__(self, rec: dict, data_yaml: Path, device, ckpt_khoi_tao: str, do_chan_doan: bool = True):
        import eval_detector_5lop as ev
        self.rec, self.device, self.ckpt, self.do_cd = rec, device, ckpt_khoi_tao, do_chan_doan
        self.qt = dict(ev.QUY_TAC["du_doan"])
        t0 = time.time()
        self.dt_dev, self.anh, self.gt = ev.chuan_bi_danh_gia(Path(data_yaml), "val")     # hop dong dev, 1 lan
        self.p = rec.setdefault("pilot", {})
        self.p.update({
            "do_chan_doan": bool(do_chan_doan),
            "tap_do": {"split": "dev (val trong data.yaml)", "so_anh": len(self.anh), "anh_dau": self.anh[0].name,
                       "anh_cuoi": self.anh[-1].name, "so_gt_theo_lop": {}, "kiem_hop_dong_s": round(time.time() - t0, 2),
                       "ghi_chu": "dev smoke KHONG co motorcycle/bicycle -> AP hai lop nay khong danh gia duoc"},
            "quy_tac_danh_gia_chinh": self.qt, "log_ultralytics": [], "moc": [], "epoch": [], "do": {}, "loi": None})
        from detector_classes import LOP_5
        self.p["tap_do"]["so_gt_theo_lop"] = {c: int(sum(int((g["cls"] == i).sum()) for g in self.gt))
                                             for i, c in enumerate(LOP_5)}
        self.batch: list[dict] = []
        self._i = 0
        self._n_het_train = 0
        self._n_fit_trong_vong = 0
        self._ep: dict | None = None
        self._dir: Path | None = None
        self._log = _BatLog(self.p["log_ultralytics"])

    # ------------------------------------------------------------------ dang ky
    def gan(self, model) -> None:
        logging.getLogger("ultralytics").addHandler(self._log)
        for ev_name, fn in (("on_train_start", self.bat_dau), ("on_train_epoch_start", self.bat_dau_epoch),
                            ("on_train_batch_start", self.bat_dau_batch), ("on_train_batch_end", self.het_batch),
                            ("on_train_epoch_end", self.het_epoch_train), ("on_val_start", self.bat_dau_val),
                            ("on_val_end", self.het_val), ("on_model_save", self.luu_model),
                            ("on_fit_epoch_end", self.het_epoch_fit), ("on_train_end", self.het_train)):
            model.add_callback(ev_name, self._bat_loi(fn))

    def go(self) -> None:
        logging.getLogger("ultralytics").removeHandler(self._log)

    def _bat_loi(self, fn):
        def boc(x):
            try:
                fn(x)
            except Exception as e:
                tr = x if hasattr(x, "ema") else getattr(x, "trainer", None)
                self.p["loi"] = {"thoi_diem": round(time.time(), 3), "ham": fn.__name__,
                                 "loi": f"{type(e).__name__}: {e}"[:500], "so_batch_da_chay": len(self.batch),
                                 "ema_updates": int(tr.ema.updates) if tr is not None and getattr(tr, "ema", None) else None}
                raise
        boc.__name__ = fn.__name__
        return boc

    def _moc(self, ten: str, **them) -> float:
        t = time.time()
        self.p["moc"].append({"ten": ten, "t": round(t, 3), **them})
        return t

    # ------------------------------------------------------------------ do
    def _do_trong_luc_train(self, tr, module, nhan: str) -> dict:
        import torch
        t0 = time.time()
        truoc = chup_trang_thai(tr)
        rng, co = luu_rng(), luu_co()
        mem0 = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        try:
            path = self._dir / f"{nhan}.pt"
            sha_tam = luu_ban_sao_tam(module, path, tr.args)
            kq = do_mo_hinh(path, self.anh, self.gt, self.qt, self.device)
        finally:
            sau_tho = chup_trang_thai(tr)
            khoi_phuc_rng(rng)
            khoi_phuc_co(co)
        sau = chup_trang_thai(tr)
        thay_doi = so_sanh_trang_thai(truoc, sau_tho)
        khac = so_sanh_trang_thai(truoc, sau)
        kq.update({
            "nguon": nhan, "file_tam_sha256": sha_tam,
            "bat_bien": {"khong_doi_sau_khoi_phuc": not khac, "khac_sau_khoi_phuc": khac,
                         "thay_doi_truoc_khi_khoi_phuc_rng_co": thay_doi, "hang_muc": sorted(truoc),
                         "cach": "hash tung tensor / trang thai truoc va sau phep do; RNG + co cudnn/deterministic duoc luu "
                                 "va khoi phuc"},
            "torch_max_allocated_trong_do_mib": round(max(0, (torch.cuda.max_memory_allocated() - mem0)) / 2 ** 20)
            if torch.cuda.is_available() else None,
            "thoi_gian_tong_s": round(time.time() - t0, 2)})
        if khac:
            raise LoiPilot(f"phep do '{nhan}' lam doi trang thai training sau khi khoi phuc: {khac[:5]}")
        return kq

    def _thong_tin_optimizer(self, tr) -> dict:
        opt = tr.optimizer
        return {"ten": type(opt).__name__, "so_trang_thai": len(opt.state),
                "nhom": [{"param_group": g.get("param_group"), "lr": g["lr"], "initial_lr": g.get("initial_lr"),
                          "momentum": g.get("momentum"), "weight_decay": g.get("weight_decay"),
                          "nesterov": g.get("nesterov"), "so_tensor": len(g["params"])} for g in opt.param_groups],
                "ema_updates": int(tr.ema.updates), "accumulate": int(tr.accumulate),
                "scheduler_last_epoch": int(tr.scheduler.last_epoch)}

    # ------------------------------------------------------------------ callback
    def _kiem_ca_callback(self, tr, fn, nhan: str) -> dict:
        """Chay fn (thu bang chung + phep do) roi kiem TOAN BO callback khong doi trang thai training (sau khi khoi phuc
        RNG + co); khac -> LoiPilot."""
        truoc = chup_trang_thai(tr)
        rng, co = luu_rng(), luu_co()
        try:
            fn()
        finally:
            khoi_phuc_rng(rng)
            khoi_phuc_co(co)
        khac = so_sanh_trang_thai(truoc, chup_trang_thai(tr))
        if khac:
            raise LoiPilot(f"callback '{nhan}' lam doi trang thai training: {khac[:5]}")
        return {"khong_doi_sau_khoi_phuc": True, "khac_sau_khoi_phuc": []}

    def bat_dau(self, tr) -> None:
        import torch
        t = self._moc("on_train_start")
        self.p["truoc_hoc"] = {"thoi_diem": round(t, 3)}
        self.p["truoc_hoc"]["bat_bien_ca_callback"] = self._kiem_ca_callback(tr, lambda: self._bat_dau(tr),
                                                                             "on_train_start")
        self.rec["_t_epoch"] = self._moc("het_do_truoc_hoc")              # train_s epoch 1 KHONG gom phep do
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _bat_dau(self, tr) -> None:
        import torch
        m = _unwrap(tr.model)
        nb = len(tr.train_loader)
        src = inspect.getsource(type(tr)._get_warmup_iterations)
        self.p["warmup_theo_ma"] = {
            "so_batch_moi_epoch": nb, "so_buoc_warmup_nw": int(tr._get_warmup_iterations(nb)),
            "ham": "BaseTrainer._get_warmup_iterations (goi tren trainer dang chay)", "ma_nguon": src,
            "ma_nguon_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "args": {k: getattr(tr.args, k) for k in ("epochs", "warmup_epochs", "warmup_bias_lr", "warmup_momentum",
                                                       "momentum", "lr0", "lrf", "nbs", "batch", "cos_lr")},
            "accumulate_khoi_dau": int(tr.accumulate),
            "epoch_tat_mosaic_theo_ma": int(tr.epochs - tr.args.close_mosaic) if tr.args.close_mosaic else None}
        self._dir = Path(tr.save_dir) / "chan_doan"
        grads = [p.grad is not None for p in m.parameters()]
        ema_giong = all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), tr.ema.ema.state_dict().values()))
        bc = {"ten_lop_model": {int(k): v for k, v in m.names.items()},
              "ten_lop_data": {int(k): v for k, v in tr.data["names"].items()}, "nc": int(m.nc),
              "nap_trong_so": kiem_nap_trong_so(m, self.ckpt), "optimizer": self._thong_tin_optimizer(tr),
              "so_tham_so_co_gradient": int(sum(grads)), "so_tham_so": len(grads),
              "model_che_do_train": bool(m.training), "model_va_ema_giong_het": bool(ema_giong),
              "so_tensor_model": len(m.state_dict())}
        self.p["truoc_hoc"]["bang_chung_khoi_tao"] = bc
        if bc["optimizer"]["ema_updates"] != 0 or bc["optimizer"]["so_trang_thai"] != 0 or any(grads):
            raise LoiPilot(f"diem do 'truoc hoc' khong con o truoc buoc cap nhat dau: {bc['optimizer']}")
        if self.do_cd:
            self.p["do"]["truoc_hoc_model"] = self._do_trong_luc_train(tr, tr.model, "truoc_hoc_model")
            self.p["do"]["truoc_hoc_model"]["ghi_chu"] = ("model trainer da dung (5 lop, da nap/remap), truoc buoc cap "
                                                          "nhat dau; EMA giong het model tai thoi diem nay")

    def bat_dau_epoch(self, tr) -> None:
        import torch
        self._i = 0
        self._ep = {"epoch": int(tr.epoch) + 1, "t_bat_dau": self._moc("on_train_epoch_start", epoch=int(tr.epoch) + 1),
                    "ema_updates_dau": int(tr.ema.updates)}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def bat_dau_batch(self, tr) -> None:
        if self._i == 0 and self._ep is not None:
            ds = tr.train_loader.dataset
            self._ep["mosaic_thuc_te"] = bool(getattr(ds, "mosaic", None))
            self._ep["t_batch_dau"] = round(time.time(), 3)

    def het_batch(self, tr) -> None:
        import torch
        li = tr.loss_items
        loss = {k: float(v) for k, v in li.items()} if isinstance(li, dict) else \
            {f"loss{i}": float(x) for i, x in enumerate(li.tolist() if hasattr(li, "tolist") else [li])}
        self.batch.append({
            "epoch": int(tr.epoch) + 1, "i": self._i, "ni": self._i + len(tr.train_loader) * int(tr.epoch),
            "t": round(time.time(), 3), "loss_tong": float(tr.loss.detach()) if tr.loss is not None else None, **loss,
            "loss_huu_han": all(math.isfinite(v) for v in loss.values()),
            **{f"lr_{g.get('param_group', i)}": g["lr"] for i, g in enumerate(tr.optimizer.param_groups)},
            **{f"momentum_{g.get('param_group', i)}": g.get("momentum") for i, g in enumerate(tr.optimizer.param_groups)},
            "accumulate": int(tr.accumulate), "ema_updates": int(tr.ema.updates),
            "torch_allocated_mib": round(torch.cuda.memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None})
        self._i += 1

    def het_epoch_train(self, tr) -> None:
        import torch
        self._n_het_train += 1
        ep = self._ep or {"epoch": int(tr.epoch) + 1}
        t = self._moc("on_train_epoch_end", epoch=ep["epoch"])
        rows = [b for b in self.batch if b["epoch"] == ep["epoch"]]
        cap = [r["ni"] for k, r in enumerate(rows) if r["ema_updates"] > (rows[k - 1]["ema_updates"] if k else
                                                                          ep["ema_updates_dau"])]
        ep.update({"t_het_train": round(t, 3), "so_batch": len(rows), "ema_updates_cuoi": int(tr.ema.updates),
                   "so_lan_cap_nhat": int(tr.ema.updates) - ep["ema_updates_dau"], "ni_co_cap_nhat": cap,
                   "loss_huu_han": all(r["loss_huu_han"] for r in rows),
                   "lr_cuoi": {f"{g.get('param_group')}": g["lr"] for g in tr.optimizer.param_groups},
                   "torch_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20)
                   if torch.cuda.is_available() else None,
                   "torch_max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20)
                   if torch.cuda.is_available() else None, "val": []})

    def bat_dau_val(self, v) -> None:
        self._moc("on_val_start")

    def het_val(self, v) -> None:
        self._moc("on_val_end")

    def luu_model(self, tr) -> None:
        self._moc("on_model_save", epoch=int(tr.epoch) + 1)

    def het_epoch_fit(self, tr) -> None:
        if self._n_fit_trong_vong >= self._n_het_train:                 # lan goi trong final_eval (validate best.pt)
            self._moc("on_fit_epoch_end_final_eval", trainer_epoch_tam=int(tr.epoch))
            self.p["final_eval_ultralytics"] = {"metrics": {k: float(v) for k, v in (tr.metrics or {}).items()},
                                                "ghi_chu": "ultralytics final_eval: validate best.pt nap tu dia "
                                                           "(goi lai on_fit_epoch_end voi epoch tam +1)"}
            return
        self._n_fit_trong_vong += 1
        ep = self._ep
        t = self._moc("on_fit_epoch_end", epoch=ep["epoch"])
        ep.update({"t_het_fit": round(t, 3), "fitness": float(tr.fitness) if tr.fitness is not None else None,
                   "metrics_val_ultralytics": {k: float(v) for k, v in (tr.metrics or {}).items()},
                   "optimizer_sau_epoch": self._thong_tin_optimizer(tr)})
        if self.do_cd:
            k = ep["epoch"]

            def _do():
                self.p["do"][f"sau_epoch{k}_model"] = self._do_trong_luc_train(tr, tr.model, f"sau_epoch{k}_model")
                self.p["do"][f"sau_epoch{k}_model"]["ghi_chu"] = "model HIEN TAI (trong so tho) sau epoch, khong phai EMA"
                self.p["do"][f"sau_epoch{k}_ema"] = self._do_trong_luc_train(tr, tr.ema.ema, f"sau_epoch{k}_ema")
                self.p["do"][f"sau_epoch{k}_ema"]["ghi_chu"] = ("EMA (ultralytics dung de validate va ghi vao "
                                                                "checkpoint; o day FP32 trong bo nho)")
            ep["bat_bien_ca_callback"] = self._kiem_ca_callback(tr, _do, f"on_fit_epoch_end epoch {k}")
        self.rec["_t_epoch"] = self._moc("het_do_sau_epoch", epoch=ep["epoch"])    # train_s epoch sau KHONG gom do
        self.p["epoch"].append(ep)

    def het_train(self, tr) -> None:
        self._moc("on_train_end")

    # ------------------------------------------------------------------ sau train
    def do_checkpoint(self, path: Path, nhan: str) -> None:
        if not self.do_cd or not Path(path).is_file():
            return
        t0 = time.time()
        kq = do_mo_hinh(path, self.anh, self.gt, self.qt, self.device)
        kq.update({"nguon": nhan, "ghi_chu": "checkpoint nap tu dia (EMA FP16 ultralytics da luu)",
                   "thoi_gian_tong_s": round(time.time() - t0, 2)})
        self.p["do"][nhan] = kq

    def ghi_csv_batch(self, path: Path) -> None:
        import csv
        if not self.batch:
            return
        cot = list(dict.fromkeys(k for r in self.batch for k in r))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cot)
            w.writeheader()
            w.writerows(self.batch)

    def tom_tat(self, so_epoch_mong_doi: int) -> dict:
        """Tieu chi hoan chinh cua pilot (KHONG dat dieu kien AP)."""
        do = self.p.get("do", {})
        can = ["truoc_hoc_model"] + [f"sau_epoch{k}_{s}" for k in range(1, so_epoch_mong_doi + 1) for s in ("model", "ema")]
        eps = self.p.get("epoch", [])
        dk = {
            "du_epoch": len(eps) == so_epoch_mong_doi,
            "co_cap_nhat_trong_so_moi_epoch": bool(eps) and all(e.get("so_lan_cap_nhat", 0) > 0 for e in eps),
            "loss_huu_han_moi_batch": bool(self.batch) and all(b["loss_huu_han"] for b in self.batch),
            "du_phep_do": (not self.do_cd) or all(k in do for k in can),
            "phep_do_khong_doi_trang_thai": (not self.do_cd) or all(
                do[k]["bat_bien"]["khong_doi_sau_khoi_phuc"] for k in can if k in do),
            "khong_loi": self.p.get("loi") is None}
        return {"dieu_kien": dk, "dat": all(dk.values())}


def ghi_json(p: Path, o) -> None:
    p.write_text(json.dumps(o, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
