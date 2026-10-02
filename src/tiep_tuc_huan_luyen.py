"""
tiep_tuc_huan_luyen.py -- Nhat ky luot, diem khoi phuc va TIEP TUC (resume) luot huan luyen day du (vong 39; nguoi dung
duyet them resume truoc luot dai). Dung boi `train_detector.py --che-do day-du [--tiep-tuc]`.

RANH GIOI HO TRO: tiep tuc SAU MOT EPOCH DA HOAN TAT. Ultralytics ghi `weights/last.pt` cuoi moi epoch, sau validate.
Phan epoch dang chay luc bi ngat (ca validate cua no) phai chay lai tu dau epoch; KHONG tiep tuc giua epoch.

CO CHE: dung NGUYEN resume cua ultralytics 8.4.141 (`BaseTrainer.check_resume` / `resume_training`), goi
`YOLO(<checkpoint>).train(resume=<DUONG DAN checkpoint>)`. Khong dung `resume=True`: voi gia tri do, ultralytics chi canh
bao roi TU KHOI DONG LUOT MOI khi checkpoint thieu epoch/optimizer. Runner chan truoc moi truong hop khong hop le.
Khong viet vong train khac.

Ultralytics khoi phuc tu checkpoint (doc ma 8.4.141, kiem lai luc chay trong `GhiLuotDayDu._kiem_khoi_phuc`):
  - epoch -> start_epoch = epoch + 1; TONG epoch lay tu train_args cua checkpoint (khong cong them);
  - optimizer.state_dict: momentum SGD luu FP16 -> nap lai FP32; param_groups (lr, initial_lr, momentum, weight_decay);
  - EMA (luu FP16 -> nap .float()) + ema.updates; scaler (AMP tat -> rong); best_fitness;
  - scheduler: last_epoch = start_epoch - 1 -> lr epoch k = lr0 * lf(k) nhu luot lien tuc. Warmup tinh theo
    ni = i + nb * epoch (tuyet doi) nen khong chay lai khi start_epoch * nb >= nw;
  - MODEL DANG HOC duoc dung tu TRONG SO EMA cua checkpoint: ultralytics khong luu model tho ("model": None).
Ultralytics KHONG khoi phuc (gioi han; ghi vao ket qua):
  - RNG python/numpy/torch: init_seeds(seed + 1) lai tu dau. DataLoader tao generator moi cung seed, nen thu tu du lieu
    va random augmentation sau resume KHONG giong luot lien tuc;
  - gradient tich luy do dang cua cac batch cuoi epoch truoc buoc toi uu cuoi (bi bo); last_opt_step dat lai -1, nen
    buoc toi uu dau tien roi vao batch dau sau resume (chi mang gradient cua 1 batch) va ranh gioi cap nhat lech;
  - accumulate lay lai round(nbs / batch) (= 8). Gia tri nay chi khac gia tri cuoi warmup khi nw < 15; luot dai co
    nw = 6.311 nen bang nhau (8);
  - EarlyStopping duoc tao moi. Runner KHOI PHUC best_fitness / best_epoch / possible_stop tu nhat ky luot (ghi moi
    epoch) de dem patience nhu luot lien tuc.
Hau qua: tiep tuc KHONG trung tung bit voi luot lien tuc. Epoch, tong epoch, lich LR, optimizer, EMA, updates va
best_fitness duoc khoi phuc va kiem.

NHAT KY `<run_dir>/nhat_ky_luot.jsonl` (append + fsync), cac su kien:
  - bat_dau: moi | tiep_tuc, kem provenance;
  - epoch_da_luu: moi epoch -- sha256 last.pt va ban sao, fitness, EarlyStopping, ema.updates, lr, thu tu du lieu,
    trang thai mosaic thuc te (doc o batch dau epoch);
  - ket_thuc.
DIEM KHOI PHUC:
  - `<run_dir>/khoi_phuc/last_eNNN.pt` cho MOI epoch, chep sau khi kiem nap duoc. Mot lan ghi last.pt bi ngat giua chung
    khong lam mat diem khoi phuc truoc do;
  - truoc moi lan tiep tuc: chep checkpoint nguon thanh `khoi_phuc/nguon_tiep_tuc_N.pt` (khong bao gio bi ghi de), luu
    args.yaml / results.csv / theo_batch.csv / last.pt / best.pt cu vao `khoi_phuc/truoc_tiep_tuc_N/`.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

TEN_NHAT_KY = "nhat_ky_luot.jsonl"
THU_MUC_KP = "khoi_phuc"
TEN_BATCH_CSV = "theo_batch.csv"
#: tham so trainer duoc phep khac checkpoint khi tiep tuc (ultralytics tu dat resume/model = duong dan checkpoint)
KHOA_ARGS_DUOC_KHAC = ("resume", "model")
#: danh tinh dataset (tu hop dong kiem_hop_dong_dataset) phai trung luc tiep tuc
TRUONG_VAN_TAY_DATASET = ("hop_dong", "phien_ban_manifest", "ten", "tap_con", "sha256_data_yaml", "sha256_tom_tat",
                          "manifest_sha256", "sha256_video_json", "split_da_khoa", "so_anh", "so_video",
                          "so_nhan_rong_hop_le", "so_anh_co_file_ignore", "so_file_da_bam")
COT_BATCH = ["lan", "epoch", "i", "ni", "t", "loss_tong", "box_loss", "cls_loss", "dfl_loss", "loss_huu_han",
             "lr_weight", "lr_bn", "lr_bias", "momentum_weight", "momentum_bn", "momentum_bias", "accumulate",
             "ema_updates", "batch_size", "thu_tu_batch"]


class LoiTiepTuc(RuntimeError):
    """Khong tiep tuc duoc (hoac trang thai khoi phuc sai) -- dung TRUOC vong train, khong tu khoi dong lai tu dau."""


# =========================================================================== tien ich
def sha256_file(p: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (int, str, bool)) or o is None:
        return o
    if hasattr(o, "item"):
        try:
            return json_safe(o.item())
        except (ValueError, TypeError):
            pass
    return str(o)


def chep_nguyen_tu(src: Path, dst: Path) -> str:
    """Chep qua file tam roi os.replace (nguyen tu tren cung o dia); kiem sha256 ban chep == nguon; tra sha256."""
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".dang_chep")
    shutil.copyfile(src, tmp)
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    s_nguon, s_tam = sha256_file(src), sha256_file(tmp)
    if s_nguon != s_tam:
        tmp.unlink(missing_ok=True)
        raise LoiTiepTuc(f"chep {src} -> {dst}: sha256 lech ({s_tam[:12]} != {s_nguon[:12]})")
    os.replace(tmp, dst)
    return s_nguon


def van_tay_dataset(ds: dict) -> dict:
    return {k: ds.get(k) for k in TRUONG_VAN_TAY_DATASET}


def _bang_duong_dan(a, b) -> bool:
    try:
        return Path(str(a)).resolve() == Path(str(b)).resolve()
    except (OSError, ValueError):
        return False


# =========================================================================== nhat ky
def ghi_nhat_ky(run_dir: Path, loai: str, **du_lieu) -> dict:
    rec = {"loai": loai, "t": round(time.time(), 3),
           "t_iso": dt.datetime.now().astimezone().isoformat(timespec="seconds"), **du_lieu}
    dong = json.dumps(json_safe(rec), ensure_ascii=False)
    with open(Path(run_dir) / TEN_NHAT_KY, "a", encoding="utf-8", newline="\n") as f:
        f.write(dong + "\n")
        f.flush()
        os.fsync(f.fileno())
    return rec


def doc_nhat_ky(run_dir: Path) -> tuple[list[dict], list[str]]:
    """Tra (su kien, canh bao). Dong CUOI ghi do (bi ngat luc ghi) -> bo qua + canh bao; dong hong o GIUA -> LoiTiepTuc."""
    p = Path(run_dir) / TEN_NHAT_KY
    if not p.is_file():
        return [], [f"khong co {TEN_NHAT_KY}"]
    dong = p.read_bytes().decode("utf-8", "replace").split("\n")
    su_kien, canh_bao = [], []
    for i, d in enumerate(dong):
        if not d.strip():
            continue
        try:
            su_kien.append(json.loads(d))
        except ValueError:
            if all(not x.strip() for x in dong[i + 1:]):
                canh_bao.append(f"dong cuoi {i + 1} cua nhat ky ghi do (bi ngat luc ghi) -- bo qua")
            else:
                raise LoiTiepTuc(f"{p}: dong {i + 1} hong o GIUA nhat ky -- khong tin duoc nhat ky")
    return su_kien, canh_bao


# =========================================================================== checkpoint
def doc_checkpoint(path: Path) -> dict:
    """torch.load(weights_only=False). CHI goi tren file da doi chieu sha256 voi nhat ky luot (file cua chinh luot)."""
    import torch
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise LoiTiepTuc(f"{path}: checkpoint khong phai dict")
    return ck


def tom_tat_checkpoint(ck: dict) -> dict:
    import torch
    opt = ck.get("optimizer")
    ema = ck.get("ema")
    return {"epoch": ck.get("epoch"), "updates": ck.get("updates"), "best_fitness": json_safe(ck.get("best_fitness")),
            "co_optimizer": isinstance(opt, dict) and bool(opt.get("state")),
            "so_trang_thai_optimizer": len(opt.get("state", {})) if isinstance(opt, dict) else 0,
            "so_nhom_optimizer": len(opt.get("param_groups", [])) if isinstance(opt, dict) else 0,
            "co_ema": isinstance(ema, torch.nn.Module), "co_khoa_scaler": "scaler" in ck,
            "scaler": "rong (AMP tat)" if ck.get("scaler") == {} else ("co" if ck.get("scaler") else ck.get("scaler")),
            "version": ck.get("version"), "date": ck.get("date"),
            "ten_lop": {int(k): v for k, v in (getattr(ema, "names", None) or {}).items()} if ema is not None else None}


def kiem_ung_vien(path: Path, ban_ghi_theo_sha: dict, run_dir: Path, ten: str) -> dict:
    """Mot ung vien checkpoint tiep tuc. CHI torch.load khi sha256 co trong nhat ky luot (khong unpickle file la)."""
    p = Path(path)
    kq = {"duong_dan": str(p), "hop_le": False, "ly_do": []}
    if not p.is_file():
        kq["ly_do"].append("khong ton tai")
        return kq
    kq["bytes"] = p.stat().st_size
    kq["sha256"] = sha256_file(p)
    rec = ban_ghi_theo_sha.get(kq["sha256"])
    if rec is None:
        kq["ly_do"].append("sha256 khong co trong nhat ky luot (khong chung minh thuoc luot da khai bao, hoac file ghi do)")
        return kq
    kq["epoch_nhat_ky"] = rec["epoch"]
    try:
        ck = doc_checkpoint(p)
    except Exception as e:  # noqa: BLE001
        kq["ly_do"].append(f"khong nap duoc: {type(e).__name__}: {e}"[:300])
        return kq
    t = tom_tat_checkpoint(ck)
    kq["tom_tat"] = t
    ta = ck.get("train_args") or {}
    kq["train_args"] = json_safe(ta)
    if not isinstance(t["epoch"], int) or t["epoch"] < 0:
        kq["ly_do"].append(f"epoch {t['epoch']} (checkpoint da strip optimizer = luot da ket thuc)")
    if not t["co_optimizer"]:
        kq["ly_do"].append("thieu trang thai optimizer")
    if not t["co_ema"] or not isinstance(t["updates"], int):
        kq["ly_do"].append("thieu EMA / updates")
    if not t["co_khoa_scaler"]:
        kq["ly_do"].append("thieu khoa scaler")
    if not _bang_duong_dan(ta.get("save_dir", ""), run_dir):
        kq["ly_do"].append(f"train_args.save_dir {ta.get('save_dir')} khac thu muc luot {run_dir}")
    if ta.get("name") != ten:
        kq["ly_do"].append(f"train_args.name {ta.get('name')!r} khac luot {ten!r}")
    if isinstance(t["epoch"], int) and t["epoch"] >= 0 and (
            t["epoch"] != rec["epoch"] - 1 or t["updates"] != rec["ema_updates"]
            or t["best_fitness"] != rec["best_fitness"]):
        kq["ly_do"].append(f"noi dung lech nhat ky: epoch {t['epoch']} / updates {t['updates']} / best {t['best_fitness']}"
                           f" vs nhat ky epoch {rec['epoch'] - 1} / {rec['ema_updates']} / {rec['best_fitness']}")
    kq["hop_le"] = not kq["ly_do"]
    return kq


def kiem_tuong_thich(train_args_ck: dict, ts: dict) -> list[str]:
    """Moi tham so khai bao cua cau hinh hien tai (tru `model`) phai TRUNG train_args cua checkpoint."""
    khac = []
    for k, v in ts.items():
        if k == "model":
            continue
        vk = train_args_ck.get(k, "<thieu>")
        if k in ("data", "project"):
            if not _bang_duong_dan(vk, v):
                khac.append(f"{k}: checkpoint {vk!r} != cau hinh {v!r}")
        elif k == "device":
            if str(vk) != str(v):
                khac.append(f"device: checkpoint {vk!r} != {v!r}")
        elif vk != v:
            khac.append(f"{k}: checkpoint {vk!r} != cau hinh {v!r}")
    return khac


def so_lan_tiep_tuc_da_co(run_dir: Path) -> int:
    """So thu tu lan tiep tuc LON NHAT da dung (nhat ky: moi su kien co lan > 0; file khoi_phuc/nguon_tiep_tuc_N.pt).
    Lay max ca hai: mot lan tiep tuc hong truoc `bat_dau` van giu so cua no -> lan sau khong trung ten file."""
    su_kien, _ = doc_nhat_ky(run_dir)
    lan = [int(s["lan"]) for s in su_kien if isinstance(s.get("lan"), int) and s["lan"] > 0]
    for f in (Path(run_dir) / THU_MUC_KP).glob("nguon_tiep_tuc_*.pt"):
        try:
            lan.append(int(f.stem.rsplit("_", 1)[1]))
        except ValueError:
            pass
    return max(lan, default=0)


def chuan_bi_tiep_tuc(run_dir: Path, ten: str, ts: dict, ds: dict, wt: dict, moi_truong: dict) -> dict:
    """Kiem MOI dieu kien tiep tuc (khong ghi gi). Sai -> LoiTiepTuc (runner exit 2, khong goi train)."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise LoiTiepTuc(f"--tiep-tuc: thu muc luot {run_dir} khong ton tai -- chua co luot nao de tiep tuc")
    su_kien, canh_bao = doc_nhat_ky(run_dir)
    if not su_kien:
        raise LoiTiepTuc(f"{run_dir}: khong co nhat ky luot ({TEN_NHAT_KY}) -- khong phai luot day du cua runner nay "
                         f"(vd luot smoke/pilot cu, checkpoint da strip optimizer) -> khong tiep tuc; khong tao metadata "
                         f"hoi to")
    goc = [s for s in su_kien if s.get("loai") == "bat_dau" and s.get("che_do_luot") == "moi"]
    if len(goc) != 1:
        raise LoiTiepTuc(f"{run_dir}: nhat ky co {len(goc)} su kien bat_dau 'moi' (can dung 1)")
    goc = goc[0]
    if goc.get("ten") != ten:
        raise LoiTiepTuc(f"nhat ky thuoc luot {goc.get('ten')!r}, khong phai {ten!r}")
    ket = [s for s in su_kien if s.get("loai") == "ket_thuc"]
    if any(s.get("trang_thai") == "hoan_tat" for s in ket):
        raise LoiTiepTuc("luot da HOAN TAT (nhat ky ket_thuc hoan_tat) -> khong tiep tuc")
    luu = [s for s in su_kien if s.get("loai") == "epoch_da_luu"]
    if not luu:
        raise LoiTiepTuc("chua co epoch nao luu checkpoint (bi ngat trong epoch 1) -> khong co diem khoi phuc; chay luot "
                         "MOI voi --ten khac")
    # moi truong / ma: thay doi phien ban ultralytics / torch co the doi ngu nghia resume va so hoc -> tu choi
    mt_goc = goc.get("moi_truong", {})
    for k in ("ultralytics", "torch"):
        if mt_goc.get(k) != moi_truong.get(k):
            raise LoiTiepTuc(f"moi truong khac luot goc: {k} {mt_goc.get(k)} -> {moi_truong.get(k)}")
    if goc.get("ultralytics_trainer_sha256") != moi_truong.get("ultralytics_trainer_sha256"):
        raise LoiTiepTuc("ultralytics/engine/trainer.py khac luot goc (sha256) -- ngu nghia resume co the doi")
    # dataset: hop dong vua chay (ds) phai trung van tay luc bat dau luot goc
    vt_goc, vt_nay = goc.get("dataset_van_tay", {}), json_safe(van_tay_dataset(ds))
    lech = sorted(k for k in set(vt_goc) | set(vt_nay) if vt_goc.get(k) != vt_nay.get(k))
    if lech:
        raise LoiTiepTuc(f"van tay dataset khac luot goc o {lech} -- khong tiep tuc tren dataset khac")
    if goc.get("checkpoint_khoi_tao", {}).get("sha256") != wt.get("sha256"):
        raise LoiTiepTuc("checkpoint pretrained khoi tao cua cau hinh hien tai khac luot goc (sai model?)")
    # chon checkpoint hop le gan nhat
    theo_sha = {}
    for s in luu:
        theo_sha[s["last_sha256"]] = s
        if s.get("ban_sao_sha256"):
            theo_sha[s["ban_sao_sha256"]] = s
    kp = run_dir / THU_MUC_KP
    ung_vien = [run_dir / "weights" / "last.pt"] + sorted(kp.glob("last_e*.pt"), reverse=True)
    kq_uv = [kiem_ung_vien(p, theo_sha, run_dir, ten) for p in ung_vien]
    hop_le = [k for k in kq_uv if k["hop_le"]]
    bao_cao_uv = [{k: v for k, v in u.items() if k not in ("train_args", "tom_tat")} for u in kq_uv]
    if not hop_le:
        raise LoiTiepTuc("khong co checkpoint hop le de tiep tuc: "
                         + "; ".join(f"{Path(k['duong_dan']).name}: {k['ly_do']}" for k in kq_uv))
    chon = max(hop_le, key=lambda k: (k["epoch_nhat_ky"], Path(k["duong_dan"]).name != "last.pt"))
    rec = theo_sha[chon["sha256"]]
    E = int(ts["epochs"])
    if rec.get("dung"):
        raise LoiTiepTuc(f"epoch {rec['epoch']} da luu kem quyet dinh DUNG (early stopping hoac epoch cuoi) -> chi con "
                         f"danh gia cuoi, khong train tiep")
    if rec["epoch"] >= E:
        raise LoiTiepTuc(f"da huan luyen du {rec['epoch']}/{E} epoch -> khong tiep tuc")
    khac = kiem_tuong_thich(chon["train_args"], ts)
    if khac:
        raise LoiTiepTuc("cau hinh hien tai KHONG tuong thich checkpoint (resume khong doi tham so, khong cong them "
                         "epoch): " + "; ".join(khac[:8]))
    if int(chon["train_args"].get("epochs", -1)) != E:
        raise LoiTiepTuc(f"tong epoch checkpoint {chon['train_args'].get('epochs')} != cau hinh {E}")
    epoch_moi_nhat = max(s["epoch"] for s in luu)
    return {
        "luot_goc": {"ten": ten, "bat_dau": goc.get("t_iso"), "pid": goc.get("pid"), "cau_hinh_sha256": goc.get("cau_hinh",
                                                                                                                {}).get("sha256")},
        "lan": so_lan_tiep_tuc_da_co(run_dir) + 1,
        "chon": {k: chon[k] for k in ("duong_dan", "sha256", "bytes", "epoch_nhat_ky")} | {"tom_tat": chon["tom_tat"]},
        "ban_ghi_epoch": rec,
        "epoch_da_luu": rec["epoch"], "epoch_bat_dau_tiep": rec["epoch"] + 1, "tong_epoch": E,
        "epoch_moi_nhat_trong_nhat_ky": epoch_moi_nhat,
        "epoch_phai_chay_lai_vi_checkpoint_moi_hon_hong": list(range(rec["epoch"] + 1, epoch_moi_nhat + 1)),
        "ung_vien": bao_cao_uv, "canh_bao_nhat_ky": canh_bao,
        "lich_su": [{k: s.get(k) for k in ("loai", "t_iso", "che_do_luot", "lan", "epoch", "trang_thai", "loi")}
                    for s in su_kien if s.get("loai") in ("bat_dau", "ket_thuc")],
    }


def sao_luu_truoc_tiep_tuc(run_dir: Path, nguon: Path, lan: int) -> dict:
    """Bao toan truoc khi ultralytics ghi tiep: chep checkpoint nguon thanh khoi_phuc/nguon_tiep_tuc_N.pt (duong dan nay
    duoc dua cho ultralytics, khong bao gio bi ghi de) + luu file cu cua luot vao khoi_phuc/truoc_tiep_tuc_N/."""
    run_dir = Path(run_dir)
    kp = run_dir / THU_MUC_KP
    kp.mkdir(exist_ok=True)
    dich = kp / f"nguon_tiep_tuc_{lan}.pt"
    thu_muc = kp / f"truoc_tiep_tuc_{lan}"
    if dich.exists() or thu_muc.exists():
        raise LoiTiepTuc(f"{dich.name} / {thu_muc.name} da ton tai -- khong ghi de diem khoi phuc cu")
    sha = chep_nguyen_tu(nguon, dich)
    thu_muc.mkdir()
    da_luu = {}
    for rel in ("args.yaml", "results.csv", TEN_BATCH_CSV, "weights/last.pt", "weights/best.pt"):
        f = run_dir / rel
        if f.is_file():
            da_luu[rel] = chep_nguyen_tu(f, thu_muc / rel.replace("/", "__"))
    return {"nguon": str(dich), "nguon_sha256": sha, "thu_muc_luu_cu": str(thu_muc), "file_cu": da_luu}


# =========================================================================== callback luot day du
class GhiLuotDayDu:
    """Callback cho luot day du (moi hoac tiep tuc): kiem trang thai khoi phuc, khoi phuc EarlyStopping, chot batch,
    ghi theo batch (CSV), nhat ky + ban sao khoi phuc moi epoch. KHONG boc an_toan: loi o day phai DUNG train."""

    def __init__(self, rec: dict, run_dir: Path, ts: dict, che_do_luot: str, lan: int, ctx: dict | None,
                 provenance: dict):
        self.rec, self.run_dir, self.ts = rec, Path(run_dir), ts
        self.che_do_luot, self.lan, self.ctx, self.provenance = che_do_luot, lan, ctx or {}, provenance
        self._rows: list[dict] = []
        self._i = 0
        self._thu_tu_batch = None
        self._thu_tu_epoch = hashlib.sha256()
        self._mosaic = None
        self.kiem_khoi_phuc: dict | None = None
        self.epoch_da_luu: list[int] = []

    def gan(self, model) -> None:
        for su_kien, fn in (("on_pretrain_routine_end", self.sau_chuan_bi), ("on_train_epoch_start", self.bat_dau_epoch),
                            ("on_train_batch_start", self.bat_dau_batch), ("on_train_batch_end", self.het_batch),
                            ("on_train_epoch_end", self.het_epoch_train), ("on_model_save", self.luu_model)):
            model.add_callback(su_kien, fn)

    # ------------------------------------------------------------------ truoc vong train
    def sau_chuan_bi(self, tr) -> None:
        if Path(tr.save_dir).resolve() != self.run_dir.resolve():
            raise LoiTiepTuc(f"ultralytics luu vao {tr.save_dir}, khong phai thu muc luot {self.run_dir}")
        if self.che_do_luot == "tiep_tuc":
            self.kiem_khoi_phuc = self._kiem_khoi_phuc(tr)
            self._khoi_phuc_dung_som(tr)
        else:
            if tr.resume or tr.start_epoch != 0:
                raise LoiTiepTuc(f"luot MOI nhung trainer o che do resume (start_epoch {tr.start_epoch})")
            if (self.run_dir / TEN_NHAT_KY).exists():
                raise LoiTiepTuc(f"luot MOI nhung {TEN_NHAT_KY} da ton tai trong {self.run_dir}")
        goc = tr.preprocess_batch

        def boc(batch):                                   # chi QUAN SAT thu tu anh (im_file), khong doi batch
            try:
                s = "\n".join(map(str, batch.get("im_file", []))).encode()
                self._thu_tu_batch = hashlib.sha256(s).hexdigest()[:16]
                self._thu_tu_epoch.update(s + b"\n")
            except Exception:  # noqa: BLE001
                self._thu_tu_batch = None
            return goc(batch)
        tr.preprocess_batch = boc
        ghi_nhat_ky(self.run_dir, "bat_dau", che_do_luot=self.che_do_luot, lan=self.lan, ten=self.rec.get("ten"),
                    pid=os.getpid(), start_epoch=tr.start_epoch, tong_epoch=tr.epochs, **self.provenance,
                    kiem_khoi_phuc=self.kiem_khoi_phuc,
                    tiep_tuc={k: self.ctx.get(k) for k in ("lan", "chon", "epoch_da_luu", "epoch_bat_dau_tiep",
                                                            "sao_luu", "epoch_phai_chay_lai_vi_checkpoint_moi_hon_hong")}
                    if self.che_do_luot == "tiep_tuc" else None)

    def _kiem_khoi_phuc(self, tr) -> dict:
        """So trang thai trainer SAU khi ultralytics resume voi CHINH checkpoint nguon (nap lai doc lap)."""
        import torch
        from ultralytics.utils.torch_utils import unwrap_model
        nguon = Path(self.ctx["sao_luu"]["nguon"])
        if sha256_file(nguon) != self.ctx["sao_luu"]["nguon_sha256"]:
            raise LoiTiepTuc(f"{nguon} doi sha256 sau khi sao luu")
        ck = doc_checkpoint(nguon)
        loi, kq = [], {}

        def kiem(ten, dk, chi_tiet):
            kq[ten] = {"dat": bool(dk), "chi_tiet": chi_tiet}
            if not dk:
                loi.append(f"{ten}: {chi_tiet}")
        kiem("resume_dung_checkpoint", bool(tr.resume) and _bang_duong_dan(tr.args.resume, nguon), str(tr.args.resume))
        kiem("start_epoch", tr.start_epoch == ck["epoch"] + 1, f"{tr.start_epoch} (checkpoint epoch {ck['epoch']})")
        kiem("tong_epoch", tr.epochs == int(self.ts["epochs"]), f"{tr.epochs} vs cau hinh {self.ts['epochs']}")
        kiem("ema_updates", tr.ema.updates == ck["updates"], f"{tr.ema.updates} vs {ck['updates']}")
        kiem("best_fitness", json_safe(tr.best_fitness) == json_safe(ck["best_fitness"]),
             f"{json_safe(tr.best_fitness)} vs {json_safe(ck['best_fitness'])}")
        kiem("scheduler_last_epoch", tr.scheduler.last_epoch == tr.start_epoch - 1, str(tr.scheduler.last_epoch))
        sd_opt, ck_opt = tr.optimizer.state_dict(), ck["optimizer"]
        lech_m = 0.0
        thieu = 0
        for i, st in ck_opt["state"].items():
            a = sd_opt["state"].get(i, {}).get("momentum_buffer")
            b = st.get("momentum_buffer")
            if a is None or b is None:
                thieu += 1
                continue
            lech_m = max(lech_m, float((a.detach().cpu().float() - b.float()).abs().max()))
        kiem("optimizer_momentum", thieu == 0 and lech_m == 0.0 and len(sd_opt["state"]) == len(ck_opt["state"]),
             f"{len(sd_opt['state'])} trang thai, thieu {thieu}, |lech| max {lech_m} (luu FP16 -> FP32)")
        khoa_nhom = ("lr", "initial_lr", "momentum", "weight_decay", "nesterov", "dampening", "param_group")
        nhom_lech = [(i, k) for i, (ga, gb) in enumerate(zip(sd_opt["param_groups"], ck_opt["param_groups"]))
                     for k in khoa_nhom if ga.get(k) != gb.get(k)]
        kiem("optimizer_param_groups", not nhom_lech and len(sd_opt["param_groups"]) == len(ck_opt["param_groups"]),
             f"lech {nhom_lech[:4]}; momentum {[g.get('momentum') for g in sd_opt['param_groups']]}")
        ema_ck = ck["ema"].float().state_dict()

        def lech_sd(sd):
            n_khac, khoa_thieu = 0, [k for k in ema_ck if k not in sd]
            for k, v in ema_ck.items():
                if k in sd and not torch.equal(sd[k].detach().cpu().to(v.dtype), v):
                    n_khac += 1
            return n_khac, khoa_thieu
        n_ema, thieu_ema = lech_sd(tr.ema.ema.state_dict())
        kiem("ema_trong_so", n_ema == 0 and not thieu_ema, f"{n_ema} tensor khac, thieu {thieu_ema[:3]}")
        n_mod, thieu_mod = lech_sd(unwrap_model(tr.model).state_dict())
        kiem("model_dang_hoc_bang_ema_checkpoint", n_mod == 0 and not thieu_mod,
             f"{n_mod} tensor khac (ultralytics dung model dang hoc tu EMA cua checkpoint; khong luu model tho)")
        ta = ck.get("train_args") or {}
        args_lech = sorted(k for k in set(ta) | set(vars(tr.args)) if k not in KHOA_ARGS_DUOC_KHAC
                           and json_safe(ta.get(k)) != json_safe(getattr(tr.args, k, None)))
        kiem("tham_so_trainer_trung_checkpoint", not args_lech,
             f"lech {[(k, ta.get(k), getattr(tr.args, k, None)) for k in args_lech[:5]]}")
        kiem("batch_size", tr.batch_size == int(self.ts["batch"]), str(tr.batch_size))
        if loi:
            raise LoiTiepTuc("trang thai sau resume KHONG khop checkpoint -- dung truoc vong train: " + "; ".join(loi))
        return kq

    def _khoi_phuc_dung_som(self, tr) -> None:
        dsm = self.ctx["ban_ghi_epoch"].get("dung_som") or {}
        if not dsm or json_safe(dsm.get("patience")) != json_safe(tr.stopper.patience) \
                or json_safe(tr.stopper.patience) != json_safe(self.ts["patience"]):
            raise LoiTiepTuc(f"nhat ky thieu / lech trang thai EarlyStopping: {dsm} (patience trainer "
                             f"{json_safe(tr.stopper.patience)}, cau hinh {self.ts['patience']})")
        truoc = {"best_fitness": tr.stopper.best_fitness, "best_epoch": tr.stopper.best_epoch,
                 "possible_stop": tr.stopper.possible_stop}
        tr.stopper.best_fitness = dsm["best_fitness"]
        tr.stopper.best_epoch = dsm["best_epoch"]
        tr.stopper.possible_stop = dsm["possible_stop"]
        self.kiem_khoi_phuc["dung_som_khoi_phuc"] = {"dat": True, "chi_tiet": {"ultralytics_tao_moi": json_safe(truoc),
                                                                              "khoi_phuc_tu_nhat_ky": dsm}}

    # ------------------------------------------------------------------ trong vong train
    def bat_dau_epoch(self, tr) -> None:
        if int(tr.batch_size) != int(self.ts["batch"]):      # vd ultralytics tu giam batch khi OOM o epoch dau
            raise LoiTiepTuc(f"batch hieu luc {tr.batch_size} != cau hinh {self.ts['batch']} (ultralytics tu doi batch, "
                             f"vd sau OOM?) -- dung thay vi train voi batch khac")
        self._rows, self._i = [], 0
        self._thu_tu_epoch = hashlib.sha256()
        self._mosaic = None

    def bat_dau_batch(self, tr) -> None:
        # ultralytics tat mosaic SAU on_train_epoch_start (epoch == epochs - close_mosaic) hoac trong resume_training
        # (start_epoch > epochs - close_mosaic) -> doc trang thai thuc te o batch dau cua epoch
        if self._i == 0:
            self._mosaic = bool(getattr(tr.train_loader.dataset, "mosaic", None))

    def het_batch(self, tr) -> None:
        li = tr.loss_items
        loss = {k: float(v) for k, v in li.items()} if isinstance(li, dict) else {}
        nhom = {g.get("param_group", str(j)): g for j, g in enumerate(tr.optimizer.param_groups)}
        nb = len(tr.train_loader)
        self._rows.append({
            "lan": self.lan, "epoch": int(tr.epoch) + 1, "i": self._i, "ni": self._i + nb * int(tr.epoch),
            "t": round(time.time(), 3), "loss_tong": float(tr.loss.detach()) if tr.loss is not None else "",
            **{k: loss.get(k, "") for k in ("box_loss", "cls_loss", "dfl_loss")},
            "loss_huu_han": all(math.isfinite(v) for v in loss.values()) and bool(loss),
            **{f"lr_{g}": nhom[g]["lr"] if g in nhom else "" for g in ("weight", "bn", "bias")},
            **{f"momentum_{g}": nhom[g].get("momentum", "") if g in nhom else "" for g in ("weight", "bn", "bias")},
            "accumulate": int(tr.accumulate), "ema_updates": int(tr.ema.updates), "batch_size": int(tr.batch_size),
            "thu_tu_batch": self._thu_tu_batch or ""})
        self._i += 1

    def het_epoch_train(self, tr) -> None:
        p = self.run_dir / TEN_BATCH_CSV
        moi = not p.exists()
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COT_BATCH, restval="")
            if moi:
                w.writeheader()
            w.writerows(self._rows)
            f.flush()
            os.fsync(f.fileno())

    def luu_model(self, tr) -> None:
        """Sau khi ultralytics ghi last.pt (+ best.pt) cuoi epoch: kiem nap duoc, chep ban sao khoi phuc, ghi nhat ky."""
        ep = int(tr.epoch) + 1
        last = Path(tr.last)
        ck = doc_checkpoint(last)
        if ck.get("epoch") != int(tr.epoch) or ck.get("optimizer") is None:
            raise LoiTiepTuc(f"{last} vua ghi khong hop le (epoch {ck.get('epoch')}, optimizer "
                             f"{'co' if ck.get('optimizer') is not None else 'KHONG'})")
        del ck
        kp = self.run_dir / THU_MUC_KP
        kp.mkdir(exist_ok=True)
        ban_sao = kp / f"last_e{ep:03d}.pt"
        if ban_sao.exists():                                  # epoch chay lai sau khi tiep tuc: giu ban cu, them hau to
            ban_sao = kp / f"last_e{ep:03d}_lan{self.lan}.pt"
        sha_bs = chep_nguyen_tu(last, ban_sao)
        best_ghi = tr.best_fitness == tr.fitness and Path(tr.best).is_file()
        nhom = {g.get("param_group", str(j)): g for j, g in enumerate(tr.optimizer.param_groups)}
        rec = ghi_nhat_ky(
            self.run_dir, "epoch_da_luu", lan=self.lan, epoch=ep, epoch_0=int(tr.epoch), last_sha256=sha_bs,
            last_bytes=last.stat().st_size, ban_sao=str(ban_sao.relative_to(self.run_dir)).replace("\\", "/"),
            ban_sao_sha256=sha_bs, best_ghi_epoch_nay=bool(best_ghi),
            best_sha256=sha256_file(tr.best) if best_ghi else None,
            fitness=json_safe(tr.fitness), best_fitness=json_safe(tr.best_fitness),
            dung_som={"best_fitness": json_safe(tr.stopper.best_fitness), "best_epoch": int(tr.stopper.best_epoch),
                      "possible_stop": bool(tr.stopper.possible_stop), "patience": json_safe(tr.stopper.patience)},
            dung=bool(tr.stop), ema_updates=int(tr.ema.updates), accumulate=int(tr.accumulate),
            lr={g: nhom[g]["lr"] for g in nhom}, momentum={g: nhom[g].get("momentum") for g in nhom},
            thu_tu_du_lieu_epoch=self._thu_tu_epoch.hexdigest(), so_batch=self._i, mosaic_thuc_te=self._mosaic,
            metrics=json_safe(dict(tr.metrics or {})))
        self.epoch_da_luu.append(ep)
        self.rec.setdefault("nhat_ky_epoch", []).append({k: rec[k] for k in ("epoch", "last_sha256", "fitness",
                                                                             "best_fitness", "ema_updates", "dung")})
