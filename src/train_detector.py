"""
train_detector.py -- Fine-tune detector 5 lop (YOLOv8n, YOLO11n) tren dataset BDD100K MOT train (plan da chot, vong 27).

Ban cu co `--dry-run` THUC SU train 1 epoch va `exist_ok=True` (luot moi ghi de luot cu) -- ca hai da bo. Ba che do:

  --che-do kiem-cau-hinh   KHONG train, KHONG tao thu muc luot: kiem data.yaml (5 lop dung thu tu, val = dev noi bo),
                           sha256 manifest dataset, sha256 checkpoint, thu muc luot chua ton tai; in tham so da
                           resolve (get_cfg) + gia tri suy ra (accumulate, weight_decay hieu luc, so iteration).
                           `--dry-run` = che do nay.
  --che-do smoke           1 epoch tren dataset smoke (tap con co manifest): loss huu han, checkpoint ghi + nap lai
                           duoc (dung 5 lop), danh gia dev chay duoc, ghi cau hinh hieu luc / phien ban / thoi gian / VRAM.
                           Ket qua vao results/phase2/detector_smoke/ -- KHONG phai ket qua nghien cuu.
  --che-do day-du          training dai; bat buoc them --xac-nhan-train-dai. Vong 39: luot day du ghi nhat ky luot
                           (`<run_dir>/nhat_ky_luot.jsonl`), theo batch (`theo_batch.csv`) va ban sao khoi phuc moi epoch
                           (`<run_dir>/khoi_phuc/last_eNNN.pt`) -- src/tiep_tuc_huan_luyen.py.
  --che-do day-du --tiep-tuc
                           (vong 39) TIEP TUC luot day du da khai bao (cung --ten / ten mac dinh) tu checkpoint HOP LE
                           gan nhat, SAU mot epoch da hoan tat (khong tiep tuc giua epoch). Kiem truoc khi train: nhat ky
                           luot, sha256 checkpoint, du trang thai (epoch, optimizer, EMA, scaler), luot chua hoan tat,
                           cau hinh hieu luc trung checkpoint (khong cong them epoch, khong doi batch/LR/warmup/aug),
                           hop dong + van tay dataset trung luot goc, cung phien ban ultralytics/torch. Sai -> exit 2,
                           KHONG goi train, KHONG tu khoi dong lai tu dau. Dung resume cua ultralytics voi DUONG DAN
                           checkpoint (khong resume=True). Ket qua: `<ten>__tiep_tuc_<N>.json` (khong ghi de).
                           `--dry-run --tiep-tuc` = chi kiem dieu kien tiep tuc, in ke hoach, khong ghi gi.
Ctrl+C trong luc train (vong 39): ghi ket qua do dang + nhat ky ket_thuc 'bi_ngat', exit 130.
  --che-do kiem-pilot      (vong 37) KHONG train: khoa cau hinh pilot -- cau hinh pilot KE THUA NGUYEN cau hinh goc
                           (kiem sha256) va CHI duoc doi `epochs`; in tham so hieu luc, khac biet so voi smoke (phai chi
                           co epochs), checkpoint khoi tao + sha256, dataset smoke da xac minh, so buoc warmup theo ma.
  --che-do pilot           (vong 37) pilot chan doan tren dataset SMOKE; bat buoc --xac-nhan-pilot. Ngoai ghi nhan nhu
                           smoke: do model 5 lop TRUOC buoc hoc va SAU tung epoch (model hien tai + EMA) va checkpoint
                           bang CUNG quy trinh danh gia chinh (conf 0,001) + chan doan diem truoc nguong; ghi loss / LR /
                           momentum / accumulate / so lan cap nhat theo batch (src/pilot_chan_doan.py). Ket qua vao
                           results/phase2/detector_pilot/.
Vong 37 sua `on_fit_epoch_end`: ultralytics `final_eval` (validate best.pt) goi LAI callback nay sau vong lap; ban cu
ghi de `val_s` + `metrics` cua epoch cuoi (val_s smoke vong 35 vi vay gom ca luu checkpoint + final_eval). Nay lan goi do
duoc nhan ra (dem so lan het epoch train) va ghi rieng vao `danh_gia_cuoi_ultralytics`.

    python src/train_detector.py --config configs/phase2/detector/train_bdd5.yaml --model yolov8n --che-do kiem-cau-hinh
    python src/train_detector.py --config configs/phase2/detector/train_bdd5.yaml --model yolo11n --che-do smoke

Khong ghi de: `<project>/<ten>` va file ket qua phai CHUA ton tai (truyen exist_ok=False va tu kiem truoc, vi ultralytics
voi exist_ok=False se tu doi ten thanh <ten>2 -- runner coi do la loi).

Chot dau vao (vong 28, R27-F1/F2): moi che do goi `kiem_dataset()` = HOP DONG DUNG CHUNG voi exporter
(`bdd_yolo_export.kiem_hop_dong_dataset`) TRUOC khi nap checkpoint/model va truoc moi mkdir: dung duong dan ultralytics
resolve tu data.yaml, train/dev dung split da khoa, anh/nhan/ignore thuc te khop manifest ca ve NOI DUNG (sha256).
Khi train: callback `on_pretrain_routine_end` doi chieu anh + nhan ultralytics THUC SU nap (ke ca tu file .cache, vi
cache ultralytics chi kiem kich thuoc + duong dan) voi dataset da xac minh; khac -> dung truoc vong train.
Cache ANH (vong 29, R28-F1): chi ho tro `cache: false`; `load_image` cua ultralytics doc `<anh>.npy` THAY JPEG ke ca khi
cache=False, nen hop dong (truoc nap model) va callback (truoc vong train) deu tu choi .npy canh anh train/dev.
Hai model dung CUNG khoi `huan_luyen` trong cau hinh; chi `weights` khac.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import tiep_tuc_huan_luyen as ttl  # noqa: E402
from detector_classes import LOP_5, loai_checkpoint  # noqa: E402

KHOA_CAM_AUTO = ("optimizer", "batch", "amp", "workers", "imgsz", "epochs", "lr0", "momentum", "weight_decay")


class LoiCauHinh(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def p_abs(s: str | Path) -> Path:
    p = Path(s)
    return p if p.is_absolute() else config.PROJECT_ROOT / p


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


# =========================================================================== kiem dau vao
KHOA_PILOT_DUOC_DOI = ("epochs",)


def doc_cfg_pilot(p: Path, y: dict) -> dict:
    """Cau hinh pilot (vong 37): `ke_thua` + `ke_thua_sha256` (cau hinh goc, kiem nguyen byte) + khoi `pilot` CHI gom
    cac khoa trong KHOA_PILOT_DUOC_DOI. Moi khoa khac (huan_luyen, smoke, model, dataset ...) lay NGUYEN tu cau hinh goc."""
    cho_phep = {"ke_thua", "ke_thua_sha256", "pilot", "mo_ta"}
    la = sorted(set(y) - cho_phep)
    if la:
        raise LoiCauHinh(f"{p}: cau hinh pilot chi duoc co {sorted(cho_phep)}, gap them {la} (moi khoa khac phai ke thua)")
    if not isinstance(y.get("pilot"), dict) or not y["pilot"]:
        raise LoiCauHinh(f"{p}: thieu khoi 'pilot'")
    la = sorted(set(y["pilot"]) - set(KHOA_PILOT_DUOC_DOI))
    if la:
        raise LoiCauHinh(f"{p}: khoi 'pilot' chi duoc doi {list(KHOA_PILOT_DUOC_DOI)}, gap {la}")
    goc = p_abs(y["ke_thua"])
    if not goc.is_file():
        raise LoiCauHinh(f"{p}: khong thay cau hinh goc {goc}")
    sha = hashlib.sha256(goc.read_bytes().replace(b"\r\n", b"\n")).hexdigest()   # noi dung chuan hoa LF (= blob git)
    if sha != y.get("ke_thua_sha256"):
        raise LoiCauHinh(f"{p}: cau hinh goc {goc} sha256 (LF) {sha[:16]} khac khai bao "
                         f"{str(y.get('ke_thua_sha256'))[:16]}")
    cfg = doc_cfg(goc)
    if "pilot" in cfg:
        raise LoiCauHinh(f"{goc}: cau hinh goc khong duoc tu co khoi 'pilot'")
    cfg["pilot"] = dict(y["pilot"])
    cfg["_ke_thua"] = {"duong_dan": str(goc), "sha256_noi_dung_lf": sha, "sha256_byte_tho": cfg["_sha256"]}
    cfg["_duong_dan"], cfg["_sha256"] = str(p), sha256_file(Path(p))
    return cfg


def doc_cfg(p: Path) -> dict:
    import yaml
    cfg = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
    if isinstance(cfg, dict) and "ke_thua" in cfg:
        return doc_cfg_pilot(Path(p), cfg)
    for k in ("ten", "model", "dataset", "huan_luyen"):
        if k not in cfg:
            raise LoiCauHinh(f"{p} thieu khoa {k!r}")
    for k in KHOA_CAM_AUTO:
        v = cfg["huan_luyen"].get(k)
        if v is None or str(v).lower() == "auto" or (k == "batch" and isinstance(v, (int, float)) and v < 1):
            raise LoiCauHinh(f"huan_luyen.{k} = {v!r}: phai khai bao gia tri tuong minh (khong 'auto'/-1)")
    if cfg["huan_luyen"].get("cache") is not False:
        # Chinh sach cache ANH hien tai (R28-F1): cache: false -> pixel nap = JPEG da khoa sha256. cache ram/disk/True
        # dung .npy / RAM ma hop dong chua kiem chung noi dung (disk con giu .npy cu) -> chua ho tro, dung ro rang.
        raise LoiCauHinh(f"huan_luyen.cache = {cfg['huan_luyen'].get('cache')!r}: chinh sach cache anh hien tai chi cho "
                         f"phep cache: false (pixel nap phai la JPEG da khoa trong manifest)")
    cfg["_duong_dan"], cfg["_sha256"] = str(p), sha256_file(Path(p))
    return cfg


class LoiDauVaoDaNap(RuntimeError):
    """Du lieu ultralytics THUC SU nap (anh + nhan, ke ca tu cache) khac dataset da xac minh -> dung truoc vong train."""


def kiem_dataset(data_yaml: Path, split_noi_dung=("train", "dev")) -> dict:
    """Chot dau vao cua runner = HOP DONG DUNG CHUNG voi exporter (`bdd_yolo_export.kiem_hop_dong_dataset`): kiem dung
    duong dan ultralytics resolve tu `data_yaml` NGUYEN VAN, danh sach anh/nhan/ignore thuc te va noi dung (sha256).
    Goi TRUOC khi nap checkpoint/model va TRUOC khi tao thu muc luot (R27-F1, R27-F2)."""
    import bdd_yolo_export as bx
    data_yaml = Path(data_yaml)
    if not data_yaml.is_file():
        raise LoiCauHinh(f"chua co dataset {data_yaml} -- chay prepare_mixed_yolo_dataset.py --config ... --buoc xuat")
    try:
        from ultralytics.data.utils import IMG_FORMATS
    except ImportError as e:
        raise LoiCauHinh(f"khong import duoc ultralytics.data.utils ({e})") from e
    if {s.lower() for s in IMG_FORMATS} != set(bx.IMG_FORMATS_ULTRALYTICS):
        raise LoiCauHinh(f"ultralytics dang chay nhan dinh dang anh {sorted(IMG_FORMATS)} khac hop dong "
                         f"{sorted(bx.IMG_FORMATS_ULTRALYTICS)} -> cach liet ke anh cua hop dong khong con dung nhu trainer")
    try:
        return bx.kiem_hop_dong_dataset(data_yaml, split_noi_dung=split_noi_dung)
    except bx.LoiDuLieu as e:
        raise LoiCauHinh(f"dataset KHONG dat hop dong: {e}") from e


def nhan_mong_doi(data_yaml: Path) -> dict[str, dict[str, np.ndarray]]:
    """{split: {duong dan anh (normcase): mang (n, 5) [lop, cx, cy, w, h]}} doc tu file nhan DA xac minh."""
    import bdd_yolo_export as bx
    goc = Path(data_yaml).resolve().parent
    out: dict[str, dict[str, np.ndarray]] = {"train": {}, "dev": {}}
    with open(goc / "manifest" / "anh.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dong, _ = bx.doc_dong_nhan((goc / r["nhan_ra"]).read_bytes(), r["nhan_ra"])
            out[r["split"]][bx._nc(goc / r["anh_ra"])] = np.array(dong, float).reshape(-1, 5)
    return out


def _cung_tap_nhan(a: np.ndarray, b: np.ndarray, atol: float = 1e-5) -> bool:
    """So hai tap nhan (n, 5) khong phu thuoc thu tu; ultralytics bo dong trung lap chinh xac nen bo o ca hai phia."""
    a = np.unique(a, axis=0) if len(a) else a.reshape(0, 5)
    b = np.unique(b, axis=0) if len(b) else b.reshape(0, 5)
    if a.shape != b.shape:
        return False
    con = list(range(len(b)))
    for row in a:
        k = next((j for j in con if row[0] == b[j][0] and np.all(np.abs(row[1:] - b[j][1:]) <= atol)), None)
        if k is None:
            return False
        con.remove(k)
    return True


def kiem_dau_vao_da_nap(trainer, mong_doi: dict[str, dict[str, np.ndarray]]) -> dict:
    """So anh + nhan ultralytics THUC SU nap (`train_loader` / `test_loader`, gom ca nhan lay tu file .cache -- cache
    ultralytics chi kiem tong kich thuoc + duong dan file, khong kiem noi dung) voi dataset da xac minh. Khac -> loi.

    NGUON PIXEL (vong 29, R28-F1): dataset khong duoc cache anh (`ds.cache` phai None = cache=False) va khong duoc co
    `<anh>.npy` (`ds.npy_files`) -- `load_image` nap .npy THAY JPEG da xac minh ke ca khi cache=False. Kiem lai o day
    (ngoai hop dong file) de chan .npy xuat hien SAU khi hop dong da chay."""
    import bdd_yolo_export as bx
    ket = {}
    for s, loader in (("train", trainer.train_loader), ("dev", trainer.test_loader)):
        ds = loader.dataset
        che_do_cache = getattr(ds, "cache", None)
        if che_do_cache not in (None, False):
            raise LoiDauVaoDaNap(f"{s}: dataset ultralytics dang cache anh ({che_do_cache!r}); chinh sach hien tai cache: "
                                 f"false -- pixel phai doc tu JPEG da xac minh; dung truoc vong train")
        npy = [str(p) for p in getattr(ds, "npy_files", []) if Path(p).exists()]
        if npy:
            raise LoiDauVaoDaNap(f"{s}: {len(npy)} file cache anh .npy canh anh ultralytics se nap THAY JPEG da xac minh "
                                 f"(vd {npy[:2]}); dung truoc vong train -- di chuyen/xoa tay file .npy")
        nap = {bx._nc(lb["im_file"]): np.column_stack([np.asarray(lb["cls"], float).reshape(-1),
                                                       np.asarray(lb["bboxes"], float).reshape(-1, 4)])
               for lb in ds.labels}
        md = mong_doi[s]
        thieu, thua = sorted(set(md) - set(nap)), sorted(set(nap) - set(md))
        khac = [k for k in md.keys() & nap.keys() if not _cung_tap_nhan(md[k], nap[k])]
        ket[s] = {"so_anh_nap": len(nap), "so_anh_da_xac_minh": len(md), "thieu": len(thieu), "thua": len(thua),
                  "nhan_khac": len(khac), "cache_anh": che_do_cache, "npy_canh_anh": 0}
        if thieu or thua or khac:
            raise LoiDauVaoDaNap(f"{s}: ultralytics nap {len(nap)} anh / da xac minh {len(md)}; thieu {len(thieu)}, thua "
                                 f"{len(thua)}, nhan khac {len(khac)} (vd {[Path(x).name for x in (thieu + thua + khac)[:3]]}) "
                                 f"-- co the do cache .cache cu hoac anh bi ultralytics loai; dung truoc vong train")
    return ket


def kiem_weights(cfg: dict, model: str) -> dict:
    m = cfg["model"].get(model)
    if not m:
        raise LoiCauHinh(f"cau hinh khong co model {model!r} (co {sorted(cfg['model'])})")
    w = p_abs(m["weights"])
    if not w.is_file():
        raise LoiCauHinh(f"thieu checkpoint khoi tao {w}")
    sha = sha256_file(w)
    if sha != m["sha256"]:
        raise LoiCauHinh(f"{w}: sha256 {sha[:16]} khac cau hinh {m['sha256'][:16]}")
    from ultralytics import YOLO
    names = YOLO(str(w)).names
    loai = loai_checkpoint(names)
    if loai != "coco80":
        raise LoiCauHinh(f"{w}: checkpoint khoi tao phai la COCO 80 lop (gap {loai})")
    return {"weights": str(w), "sha256": sha, "bytes": w.stat().st_size, "loai_checkpoint": loai, "so_lop": len(names)}


def tham_so(cfg: dict, che_do: str, data_yaml: Path, project: Path, ten: str, batch_ghi_de: int | None,
            device: str, workers_ghi_de: int | None = None) -> dict:
    ts = dict(cfg["huan_luyen"])
    if che_do in ("smoke", "pilot", "kiem-pilot"):
        ts.update(cfg.get("smoke", {}))
    if che_do in ("pilot", "kiem-pilot"):
        ts.update(cfg["pilot"])                     # CHI cac khoa trong KHOA_PILOT_DUOC_DOI (doc_cfg_pilot da kiem)
    if batch_ghi_de:
        ts["batch"] = int(batch_ghi_de)
    if workers_ghi_de is not None:
        ts["workers"] = int(workers_ghi_de)
    ts.update({"data": str(data_yaml), "project": str(project), "name": ten, "exist_ok": False, "device": device})
    return ts


def kiem_pilot_chi_doi_epoch(cfg: dict, data_yaml: Path, project: Path, ten: str, device: str) -> dict:
    """Tham so hieu luc cua pilot so voi smoke (cung cau hinh goc, cung data/project/ten/device): khac biet phai DUNG
    bang KHOA_PILOT_DUOC_DOI (va chi khi gia tri thuc su doi). Khac them -> LoiCauHinh."""
    a = tham_so(cfg, "smoke", data_yaml, project, ten, None, device)
    b = tham_so(cfg, "pilot", data_yaml, project, ten, None, device)
    khac = {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    if set(khac) - set(KHOA_PILOT_DUOC_DOI) or not khac:
        raise LoiCauHinh(f"pilot phai CHI khac smoke o {list(KHOA_PILOT_DUOC_DOI)}; khac thuc te: {khac}")
    return khac


def resolve_tinh(ts: dict) -> dict:
    """Tham so sau get_cfg cua ultralytics (hop le hoa kieu, gop mac dinh) -- khong tao thu muc, khong train."""
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    c = get_cfg(DEFAULT_CFG, {k: v for k, v in ts.items()})
    return json_safe(vars(c))


def suy_ra(ts: dict, n_train: int) -> dict:
    b, nbs = int(ts["batch"]), int(ts.get("nbs", 64))
    acc = max(round(nbs / b), 1)
    it_epoch = math.ceil(n_train / b)
    return {"accumulate": acc, "weight_decay_hieu_luc": ts["weight_decay"] * b * acc / nbs,
            "batch_moi_epoch": it_epoch, "buoc_toi_uu_moi_epoch": math.ceil(it_epoch / acc),
            "tong_batch": it_epoch * int(ts["epochs"]),
            "ghi_chu": "accumulate = max(round(nbs/batch), 1); weight_decay hieu luc = wd*batch*accumulate/nbs (ultralytics). "
                       "buoc_toi_uu_moi_epoch la UOC TINH voi accumulate co dinh, CHUA tinh warmup (accumulate tang dan "
                       "tu 1) va ranh gioi cap nhat giua cac epoch -- khong phai so buoc thuc chay (R38-N1); tong_batch "
                       "chi dung neu chay du epochs (patience co the dung som)"}


# =========================================================================== moi truong + VRAM
def moi_truong() -> dict:
    import numpy
    import torch
    import ultralytics
    import ultralytics.engine.trainer as ut
    out = {"python": sys.version.split()[0], "os": platform.platform(), "ultralytics": ultralytics.__version__,
           "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "numpy": numpy.__version__, "ultralytics_trainer_sha256": sha256_file(Path(ut.__file__))}
    try:
        import cv2
        out["opencv"] = cv2.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["gpu"] = {"ten": p.name, "vram_mib": round(p.total_memory / 2 ** 20)}
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            drv, used = (x.strip() for x in r.stdout.strip().splitlines()[0].split(","))
            out["gpu"] |= {"driver": drv, "vram_dang_dung_truoc_khi_chay_mib": int(used)}
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
    return out


class DoVram(threading.Thread):
    """Lay mau `nvidia-smi memory.used` (toan GPU, gom ca tien trinh khac) + RAM he thong con trong, moi `chu_ky` s."""

    def __init__(self, chu_ky: float = 1.0):
        super().__init__(daemon=True)
        self.chu_ky, self.dung, self.mau, self.ram, self.gpu = chu_ky, threading.Event(), [], [], []
        self.tho: list[tuple] = []          # vong 37: mau tho (t, nvidia-smi memory.used MiB, util %, RAM con trong MiB)

    def run(self):
        try:
            import psutil
        except ImportError:
            psutil = None
        while not self.dung.is_set():
            t, mem, util, ram = time.time(), None, None, None
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                mem, util = (int(x) for x in r.stdout.strip().splitlines()[0].split(","))
                self.mau.append(mem)
                self.gpu.append(util)
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            if psutil is not None:
                ram = psutil.virtual_memory().available / 2 ** 20
                self.ram.append(ram)
            self.tho.append((round(t, 3), mem, util, round(ram) if ram is not None else None))
            self.dung.wait(self.chu_ky)

    def ghi_tho(self, path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_unix", "nvidia_smi_memory_used_mib_ca_gpu", "gpu_util_pct", "ram_he_thong_con_trong_mib"])
            w.writerows(self.tho)

    def ket_qua(self) -> dict:
        return {"so_mau": len(self.mau), "max_mib": max(self.mau) if self.mau else None,
                "chu_ky_s": self.chu_ky, "pham_vi": "nvidia-smi memory.used cua ca GPU (gom desktop + tien trinh khac)",
                "ram_con_trong_min_mib": round(min(self.ram)) if self.ram else None,
                "gpu_util_trung_vi_pct": sorted(self.gpu)[len(self.gpu) // 2] if self.gpu else None,
                "gpu_util_ghi_chu": "utilization.gpu cua nvidia-smi, lay mau suot luot (gom khoi dong + val)"}


# =========================================================================== train
def gan_callback(model, rec: dict) -> None:
    import torch

    def pg(o):
        out = []
        for g in o.param_groups:
            out.append({k: json_safe(g.get(k)) for k in ("lr", "initial_lr", "momentum", "betas", "weight_decay",
                                                           "nesterov", "param_group")} | {"so_tensor": len(g["params"])})
        return out

    def bat_dau(tr):
        rec["hieu_luc"] = {
            "amp": bool(tr.amp), "batch_size": int(tr.batch_size), "accumulate": int(tr.accumulate),
            "optimizer": type(tr.optimizer).__name__, "param_groups": pg(tr.optimizer),
            "scheduler": type(tr.scheduler).__name__, "workers_dataloader": int(tr.train_loader.num_workers),
            "n_anh_train": len(tr.train_loader.dataset), "n_anh_val": len(tr.test_loader.dataset),
            "batch_moi_epoch": len(tr.train_loader), "nc": int(tr.data["nc"]), "names": json_safe(tr.data["names"]),
            "save_dir": str(tr.save_dir), "args": json_safe(vars(tr.args)),
        }
        rec["_t_epoch"] = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def het_batch(tr):
        rec.setdefault("_t_batch", []).append(time.time())

    def het_epoch_train(tr):
        now = time.time()
        tl = tr.tloss                                   # ultralytics 8.4: dict {ten: gia tri}; ban cu: tensor
        if isinstance(tl, dict):
            ten_loss, loss = list(tl), [float(v) for v in tl.values()]
        else:
            loss = [float(x) for x in (tl.tolist() if hasattr(tl, "tolist") else [tl])]
            ten_loss = list(getattr(tr, "loss_names", [])) or [f"loss{i}" for i in range(len(loss))]
        tb = rec.pop("_t_batch", [])
        kc = sorted(b - a for a, b in zip(tb[5:], tb[6:]))              # bo 5 batch dau (khoi dong worker, cudnn)
        e = {"epoch": int(tr.epoch) + 1, "train_s": round(now - rec["_t_epoch"], 2),
             "loss": dict(zip(ten_loss, loss)), "loss_huu_han": bool(loss) and all(math.isfinite(x) for x in loss),
             "lr": [json_safe(g["lr"]) for g in tr.optimizer.param_groups],
             "so_batch": len(tb),
             "on_dinh": ({"so_khoang": len(kc), "s_moi_batch_trung_vi": round(kc[len(kc) // 2], 4),
                          "anh_moi_s_trung_vi": round(int(tr.batch_size) / kc[len(kc) // 2], 2),
                          "ghi_chu": "trung vi khoang giua cac batch, bo 5 batch dau"} if len(kc) >= 5 else None)}
        if torch.cuda.is_available():
            e["torch_max_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
            e["torch_max_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2 ** 20)
        rec.setdefault("epoch", []).append(e)
        rec["_t_val"] = now
        rec["_n_het_train"] = rec.get("_n_het_train", 0) + 1

    def het_epoch_fit(tr):
        now = time.time()
        if rec.get("_n_het_fit", 0) >= rec.get("_n_het_train", 0):
            # Vong 37: ultralytics final_eval (validate best.pt nap tu dia) goi LAI on_fit_epoch_end sau vong lap, voi
            # epoch tam +1. Khong phai epoch moi -> KHONG ghi de val_s / metrics cua epoch cuoi (ban cu ghi de).
            rec["danh_gia_cuoi_ultralytics"] = {
                "metrics": json_safe(dict(tr.metrics or {})), "trainer_epoch_tam": int(tr.epoch),
                "thoi_gian_tu_het_train_epoch_cuoi_s": round(now - rec.get("_t_val", now), 2),
                "ghi_chu": "ultralytics final_eval: validate best.pt (FP16 tu dia) sau vong lap; gom ca luu checkpoint "
                           "va don dep cuoi epoch"}
            return
        rec["_n_het_fit"] = rec.get("_n_het_fit", 0) + 1
        e = rec["epoch"][-1]
        e["val_s"] = round(now - rec.get("_t_val", now), 2)
        e["metrics"] = json_safe(dict(tr.metrics))
        e["fitness"] = json_safe(tr.fitness)
        rec["_t_epoch"] = now

    def an_toan(fn):
        """Loi trong phan GHI NHAN khong duoc lam hong luot train: ghi vao rec['loi_callback'] va chay tiep."""
        def boc(tr):
            try:
                fn(tr)
            except Exception as e:  # noqa: BLE001
                rec.setdefault("loi_callback", []).append(f"{fn.__name__}: {type(e).__name__}: {e}")
        return boc

    model.add_callback("on_train_start", an_toan(bat_dau))
    model.add_callback("on_train_batch_end", an_toan(het_batch))
    model.add_callback("on_train_epoch_end", an_toan(het_epoch_train))
    model.add_callback("on_fit_epoch_end", an_toan(het_epoch_fit))


def kiem_checkpoint(p: Path, ts: dict) -> dict:
    from ultralytics import YOLO
    m = YOLO(str(p))
    names = m.names
    ok_names = [names[i] for i in range(len(names))] == list(LOP_5)
    ta = (getattr(m, "ckpt", None) or {}).get("train_args", {}) or {}
    lech = {k: (ts[k], ta.get(k)) for k in ("epochs", "batch", "imgsz", "optimizer", "lr0", "seed", "amp")
            if k in ta and ta.get(k) != ts[k]}
    return {"duong_dan": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size, "nap_lai": True,
            "names": json_safe(names), "dung_5_lop": ok_names, "train_args_lech_cau_hinh": lech}


CHE_DO_PILOT = ("pilot", "kiem-pilot")


def du_doan_warmup_theo_ma(ts: dict, n_train: int) -> dict:
    """Goi CHINH ham `BaseTrainer._get_warmup_iterations` cua ultralytics dang cai voi tham so hieu luc (khong suy tu tay);
    so batch/epoch = ceil(n_train / batch) (DataLoader khong drop_last). Ban chay that ghi lai gia tri trainer tinh."""
    import inspect
    from types import SimpleNamespace

    from ultralytics.engine.trainer import BaseTrainer
    nb = math.ceil(n_train / int(ts["batch"]))
    gia = SimpleNamespace(args=SimpleNamespace(warmup_epochs=float(ts["warmup_epochs"])), epochs=int(ts["epochs"]))
    src = inspect.getsource(BaseTrainer._get_warmup_iterations)
    cm = int(ts.get("close_mosaic") or 0)
    return {"so_batch_moi_epoch": nb, "so_buoc_warmup_nw": int(BaseTrainer._get_warmup_iterations(gia, nb)),
            "ma_nguon": src, "ma_nguon_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "epoch_tat_mosaic_theo_ma": (int(ts["epochs"]) - cm) if cm else None,
            "ghi_chu": "du doan truoc khi chay bang chinh ham cua phien ban dang cai; luot pilot ghi lai gia tri trainer "
                       "that tinh va dien bien LR/accumulate tung batch"}


def _git_head() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                           cwd=config.PROJECT_ROOT)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance_luot(a, cfg: dict, ts: dict, ds: dict, wt: dict, mt: dict) -> dict:
    """Provenance ghi vao su kien `bat_dau` cua nhat ky luot day du (moi va moi lan tiep tuc)."""
    src = Path(__file__).resolve().parent
    return {"lenh": [sys.executable, *sys.argv], "cau_hinh": {"duong_dan": a.config, "sha256": cfg["_sha256"]},
            "tham_so_khai_bao": json_safe(ts), "dataset_van_tay": json_safe(ttl.van_tay_dataset(ds)),
            "checkpoint_khoi_tao": wt, "moi_truong": mt, "ultralytics_trainer_sha256": mt.get("ultralytics_trainer_sha256"),
            "ma_nguon_sha256": {f: sha256_file(src / f) for f in ("train_detector.py", "tiep_tuc_huan_luyen.py",
                                                                  "bdd_yolo_export.py", "detector_classes.py")},
            "git_head": _git_head()}


GIOI_HAN_TIEP_TUC = [
    "chi tiep tuc SAU epoch da hoan tat; phan epoch dang chay luc bi ngat (ca validate) phai chay lai",
    "model dang hoc duoc dung tu trong so EMA cua checkpoint (ultralytics khong luu model tho)",
    "momentum SGD va EMA luu FP16 trong checkpoint -> nap lai FP32 (lam tron)",
    "RNG python/numpy/torch va generator DataLoader tao lai tu seed -> thu tu du lieu / augmentation khac luot lien tuc",
    "gradient tich luy do dang cuoi epoch truoc bi bo; last_opt_step dat lai -1 -> buoc toi uu dau tien roi vao batch "
    "dau sau resume (chi gradient cua 1 batch) va ranh gioi buoc toi uu lech",
    "accumulate lay lai round(nbs/batch); chi khac gia tri cuoi warmup khi nw < 15 (luot dai nw = 6.311 -> bang nhau)",
    "EarlyStopping duoc runner khoi phuc tu nhat ky luot (ultralytics tao moi)",
    "ket qua tiep tuc KHONG trung tung bit voi luot lien tuc",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default=str(config.CONFIGS_DIR / "phase2" / "detector" / "train_bdd5.yaml"))
    ap.add_argument("--model", required=True, help="khoa trong cau hinh: yolov8n | yolo11n")
    ap.add_argument("--che-do", choices=["kiem-cau-hinh", "smoke", "day-du", "kiem-pilot", "pilot"], default=None)
    ap.add_argument("--dry-run", action="store_true", help="= --che-do kiem-cau-hinh (KHONG train)")
    ap.add_argument("--xac-nhan-train-dai", action="store_true", help="bat buoc voi --che-do day-du")
    ap.add_argument("--tiep-tuc", action="store_true",
                    help="tiep tuc luot day du da khai bao tu checkpoint hop le gan nhat (sau epoch da xong); di voi "
                         "--che-do day-du --xac-nhan-train-dai, hoac --dry-run de chi kiem")
    ap.add_argument("--xac-nhan-pilot", action="store_true", help="bat buoc voi --che-do pilot (pilot da duoc duyet)")
    ap.add_argument("--khong-do-chan-doan", action="store_true",
                    help="pilot: BO cac phep do chan doan -- CHI dung cho test A/B bat bien, khong phai luot pilot")
    ap.add_argument("--data", default=None, help="ghi de data.yaml (vd dataset fixture); ghi vao ket qua")
    ap.add_argument("--ten", default=None, help="ten luot (mac dinh <cfg.ten>_<model>_<che do>)")
    ap.add_argument("--project", default=str(config.MODELS_DIR / "finetune_phase2"))
    ap.add_argument("--ket-qua-dir", default=None)
    ap.add_argument("--batch", type=int, default=None, help="ghi de batch -- CHI cho smoke/kiem VRAM, ghi vao ket qua")
    ap.add_argument("--workers", type=int, default=None, help="ghi de so worker -- CHI cho smoke/kiem, ghi vao ket qua")
    ap.add_argument("--device", default="0")
    ap.add_argument("--khong-danh-gia", action="store_true", help="smoke/pilot: bo danh gia dev sau train")
    a = ap.parse_args()
    che_do = "kiem-cau-hinh" if a.dry_run else a.che_do
    if che_do is None:
        print("[SAI CACH DUNG] can --che-do {kiem-cau-hinh,smoke,day-du,kiem-pilot,pilot} (hoac --dry-run)",
              file=sys.stderr)
        return 1
    if che_do == "day-du" and not a.xac_nhan_train_dai:
        print("[DUNG] --che-do day-du can --xac-nhan-train-dai (training dai chua duoc duyet trong vong nay)",
              file=sys.stderr)
        return 1
    if a.tiep_tuc and che_do not in ("day-du", "kiem-cau-hinh"):
        print("[DUNG] --tiep-tuc chi dung voi --che-do day-du (hoac --dry-run de kiem truoc)", file=sys.stderr)
        return 1
    if che_do == "pilot" and not a.xac_nhan_pilot:
        print("[DUNG] --che-do pilot can --xac-nhan-pilot", file=sys.stderr)
        return 1
    if (a.batch or a.workers is not None) and (che_do in ("day-du", *CHE_DO_PILOT) or a.tiep_tuc):
        print("[DUNG] --batch/--workers chi ghi de duoc o smoke/kiem-cau-hinh; training dai, tiep tuc va pilot dung cau "
              "hinh", file=sys.stderr)
        return 1
    if a.khong_do_chan_doan and che_do != "pilot":
        print("[DUNG] --khong-do-chan-doan chi dung voi --che-do pilot (test A/B)", file=sys.stderr)
        return 1
    kt = None
    try:
        cfg = doc_cfg(p_abs(a.config))
        if che_do in CHE_DO_PILOT and "pilot" not in cfg:
            raise LoiCauHinh(f"--che-do {che_do} can cau hinh pilot (ke_thua + pilot), gap {a.config}")
        if che_do not in CHE_DO_PILOT and "pilot" in cfg:
            raise LoiCauHinh(f"{a.config} la cau hinh pilot -- chi dung voi --che-do pilot / kiem-pilot")
        khoa_ds = "smoke" if che_do in ("smoke", *CHE_DO_PILOT) else "day_du"
        data_yaml = p_abs(a.data or cfg["dataset"][khoa_ds])
        hau_to = ("pilot" if che_do in CHE_DO_PILOT else
                  "day_du" if a.tiep_tuc else che_do.replace("-", "_"))
        ten = a.ten or f"{cfg['ten']}_{a.model}_{hau_to}"
        project = p_abs(a.project)
        run_dir = project / ten
        thu_muc_kq = {"day-du": "detector_train", "pilot": "detector_pilot", "kiem-pilot": "detector_pilot"}
        kq_dir = p_abs(a.ket_qua_dir or (config.RESULTS_DIR / "phase2" / thu_muc_kq.get(che_do, "detector_smoke")))
        kq_file = kq_dir / f"{ten}.json"
        if run_dir.exists() and not a.tiep_tuc:
            raise LoiCauHinh(f"{run_dir} da ton tai -- khong ghi de luot cu (doi --ten, hoac --tiep-tuc de tiep tuc "
                             f"luot day du bi ngat)")
        if a.tiep_tuc and not run_dir.is_dir():
            raise LoiCauHinh(f"--tiep-tuc: thu muc luot {run_dir} khong ton tai -- chua co luot nao de tiep tuc")
        if kq_file.exists() and che_do not in ("kiem-cau-hinh", "kiem-pilot") and not a.tiep_tuc:
            raise LoiCauHinh(f"{kq_file} da ton tai -- khong ghi de ket qua cu")
        ds = kiem_dataset(data_yaml)          # HOP DONG dataset: TRUOC khi nap checkpoint / model, TRUOC moi mkdir
        wt = kiem_weights(cfg, a.model)
        khac_smoke = kiem_pilot_chi_doi_epoch(cfg, data_yaml, project, ten, a.device) if che_do in CHE_DO_PILOT else None
        ts = tham_so(cfg, che_do, data_yaml, project, ten, a.batch, a.device, a.workers)
        ts["model"] = wt["weights"]
        rs = resolve_tinh(ts)
        mt = moi_truong()
        if a.tiep_tuc:
            kt = ttl.chuan_bi_tiep_tuc(run_dir, ten, ts, ds, wt, mt)     # LoiTiepTuc -> exit 2, khong goi train
            kq_file = kq_dir / f"{ten}__tiep_tuc_{kt['lan']}.json"
            if kq_file.exists() and che_do != "kiem-cau-hinh":
                raise LoiCauHinh(f"{kq_file} da ton tai -- khong ghi de ket qua cu")
        rec = {"loai": {"kiem-cau-hinh": "KIEM CAU HINH (khong train)" + (" + KIEM TIEP TUC" if a.tiep_tuc else ""),
                        "smoke": "SMOKE -- chi kiem duong chay, KHONG dung de so sanh model hay chon tham so",
                        "day-du": "training day du" + (" -- TIEP TUC" if a.tiep_tuc else ""),
                        "kiem-pilot": "KIEM CAU HINH PILOT (khong train)",
                        "pilot": ("PILOT CHAN DOAN -- KHONG dung de so sanh / chon model" if not a.khong_do_chan_doan
                                  else "PILOT KHONG DO (CHI cho test A/B bat bien)")}[che_do],
               "che_do": che_do, "ten": ten, "model": a.model, "tao_luc": dt.datetime.now().isoformat(timespec="seconds"),
               "cau_hinh": {"duong_dan": a.config, "sha256": cfg["_sha256"], "ke_thua": cfg.get("_ke_thua")},
               "pilot_khac_smoke": khac_smoke, "dataset": ds, "checkpoint_khoi_tao": wt,
               "tham_so_khai_bao": json_safe(ts), "batch_ghi_de": a.batch, "workers_ghi_de": a.workers,
               "tham_so_resolve_tinh": rs, "suy_ra": suy_ra(ts, ds["so_anh"]["train"]), "moi_truong": mt,
               "run_dir": str(run_dir)}
        if che_do in CHE_DO_PILOT:
            rec["warmup_du_doan_theo_ma"] = du_doan_warmup_theo_ma(ts, ds["so_anh"]["train"])
        if kt is not None:
            rec["tiep_tuc"] = kt
            rec["gioi_han_tiep_tuc"] = GIOI_HAN_TIEP_TUC
    except (LoiCauHinh, ttl.LoiTiepTuc) as e:
        print(f"[LOI CAU HINH] {e}", file=sys.stderr)
        return 2
    if che_do == "kiem-cau-hinh":
        print(json.dumps({k: rec[k] for k in ("loai", "ten", "dataset", "checkpoint_khoi_tao", "tham_so_khai_bao",
                                              "suy_ra", "run_dir")}, ensure_ascii=False, indent=1))
        if kt is not None:
            print(json.dumps(json_safe({k: kt[k] for k in ("lan", "chon", "epoch_da_luu", "epoch_bat_dau_tiep",
                                                            "tong_epoch", "epoch_phai_chay_lai_vi_checkpoint_moi_hon_hong",
                                                            "ung_vien", "canh_bao_nhat_ky", "lich_su")}),
                             ensure_ascii=False, indent=1))
            print(f"[OK] kiem tiep tuc xong: se tiep tuc tu epoch {kt['epoch_bat_dau_tiep']}/{kt['tong_epoch']} "
                  f"(checkpoint {Path(kt['chon']['duong_dan']).name}); KHONG train, KHONG ghi gi")
            return 0
        print(f"[OK] kiem cau hinh xong, KHONG train, KHONG tao {run_dir}")
        return 0
    if che_do == "kiem-pilot":
        rec["ultralytics_trainer_py_sha256"] = mt["ultralytics_trainer_sha256"]
        rec["lenh_pilot"] = (f"python -u src/train_detector.py --config {a.config} --model {a.model} --che-do pilot "
                             f"--xac-nhan-pilot")
        print(json.dumps(json_safe(rec), ensure_ascii=False, indent=1))
        print(f"[OK] kiem cau hinh pilot xong, KHONG train, KHONG tao {run_dir}")
        return 0

    from ultralytics import YOLO
    project.mkdir(parents=True, exist_ok=True)
    gi = project / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n", encoding="utf-8")
    kq_dir.mkdir(parents=True, exist_ok=True)
    train_kwargs = {k: v for k, v in ts.items() if k != "model"}
    if kt is not None:
        try:
            kt["sao_luu"] = ttl.sao_luu_truoc_tiep_tuc(run_dir, Path(kt["chon"]["duong_dan"]), kt["lan"])
        except ttl.LoiTiepTuc as e:
            print(f"[LOI CAU HINH] {e}", file=sys.stderr)
            return 2
        # resume cua ultralytics voi DUONG DAN tuong minh: moi tham so lay tu checkpoint (da kiem trung cau hinh)
        model = YOLO(kt["sao_luu"]["nguon"])
        train_kwargs = {"resume": kt["sao_luu"]["nguon"], "data": str(data_yaml), "device": a.device}
    else:
        model = YOLO(wt["weights"])
    rec["train_kwargs"] = json_safe(train_kwargs)
    gan_callback(model, rec)
    mong_doi = nhan_mong_doi(data_yaml)

    def kiem_nap(tr):                              # KHONG boc an_toan: sai lech phai DUNG truoc vong train
        rec["dau_vao_da_nap"] = kiem_dau_vao_da_nap(tr, mong_doi)
    model.add_callback("on_pretrain_routine_end", kiem_nap)
    gl = None
    if che_do == "day-du":
        gl = ttl.GhiLuotDayDu(rec, run_dir, ts, "tiep_tuc" if kt is not None else "moi", kt["lan"] if kt else 0, kt,
                              provenance_luot(a, cfg, ts, ds, wt, mt))
        gl.gan(model)                              # SAU kiem_nap: chi ghi bat_dau khi du lieu thuc nap da dat
    gp = None
    if che_do == "pilot":
        import pilot_chan_doan as pcd
        gp = pcd.GhiPilot(rec, data_yaml, a.device, wt["weights"], do_chan_doan=not a.khong_do_chan_doan)
        gp.gan(model)                              # SAU callback runner: phep do khong tinh vao train_s / val_s
    vram = DoVram()
    vram.start()
    t0 = time.time()
    loi, bi_ngat = None, False
    try:
        res = model.train(**train_kwargs)
    except KeyboardInterrupt:                      # Ctrl+C: ghi ket qua do dang (checkpoint epoch da xong van con)
        loi, bi_ngat, res = "KeyboardInterrupt: bi ngat (Ctrl+C) trong luc train", True, None
    except Exception as e:                         # ghi lai loi (vd het VRAM) roi tra ma loi
        loi = f"{type(e).__name__}: {e}"
        res = None
    finally:
        vram.dung.set()
        vram.join(timeout=5)
        if gp is not None:
            gp.go()
    rec["thoi_gian_s"] = round(time.time() - t0, 1)
    rec["vram_nvidia_smi"] = vram.ket_qua()
    for k in ("_t_epoch", "_t_val", "_t_batch", "_n_het_train", "_n_het_fit"):
        rec.pop(k, None)
    rec["loi"] = loi
    if res is not None:
        sd = Path(res.save_dir)
        rec["save_dir"] = str(sd)
        if sd.resolve() != run_dir.resolve():
            rec["loi"] = f"ultralytics luu vao {sd} thay vi {run_dir} (tu doi ten?)"
        rec["checkpoint"] = {n: kiem_checkpoint(sd / "weights" / f"{n}.pt", ts)
                             for n in ("last", "best") if (sd / "weights" / f"{n}.pt").is_file()}
        rec["loss_huu_han"] = all(e.get("loss_huu_han") for e in rec.get("epoch", [])) and bool(rec.get("epoch"))
        if che_do in ("smoke", "pilot") and not a.khong_danh_gia and "best" in rec["checkpoint"]:
            import eval_detector_5lop as ev
            t1 = time.time()
            try:
                dt_dev, anh, gt = ev.chuan_bi_danh_gia(data_yaml, "val")    # kiem lai hop dong (noi dung dev)
                dd, mt_ = ev.du_doan_ultralytics(str(sd / "weights" / "best.pt"), anh, ev.QUY_TAC["du_doan"],
                                                 device=a.device)
                rec["danh_gia_dev_smoke"] = {"ghi_chu": "best.pt tren dev smoke; chi kiem duong chay; KHONG so sanh model",
                                             "dataset_da_xac_minh": dt_dev, "quy_tac": ev.QUY_TAC, "anh_xa_lop": mt_,
                                             "thoi_gian_s": round(time.time() - t1, 1), **ev.danh_gia(gt, dd)}
            except Exception as e:  # noqa: BLE001
                rec["loi"] = rec["loi"] or f"danh gia dev sau train: {type(e).__name__}: {e}"
        if gp is not None:
            for n in ("last", "best"):
                try:
                    gp.do_checkpoint(sd / "weights" / f"{n}.pt", f"checkpoint_{n}")
                except Exception as e:  # noqa: BLE001
                    rec["loi"] = rec["loi"] or f"do checkpoint {n}: {type(e).__name__}: {e}"
    if gp is not None:
        rec["pilot"]["tom_tat"] = gp.tom_tat(int(ts["epochs"]))
        f_batch, f_tn = kq_dir / f"{ten}_batch.csv", kq_dir / f"{ten}_tai_nguyen_nvidia_smi.csv"
        gp.ghi_csv_batch(f_batch)
        vram.ghi_tho(f_tn)
        rec["pilot"]["file"] = {"batch_csv": str(f_batch), "tai_nguyen_csv": str(f_tn),
                                "cach_lay_mau_tai_nguyen": "luong DoVram trong runner moi ~1 s: nvidia-smi memory.used + "
                                                           "utilization.gpu (CA GPU) va psutil.virtual_memory().available"}
    ok = rec["loi"] is None and not rec.get("loi_callback") and rec.get("loss_huu_han") and all(
        c["dung_5_lop"] and not c["train_args_lech_cau_hinh"] for c in rec.get("checkpoint", {}).values()) \
        and {"last", "best"} <= set(rec.get("checkpoint", {}))
    if gp is not None:
        ok = bool(ok) and rec["pilot"]["tom_tat"]["dat"]
    if gl is not None:
        f_tn = kq_file.with_name(kq_file.stem + "_tai_nguyen_nvidia_smi.csv")
        vram.ghi_tho(f_tn)
        rec["kiem_khoi_phuc"] = gl.kiem_khoi_phuc
        rec["luot_day_du"] = {"nhat_ky": str(run_dir / ttl.TEN_NHAT_KY), "theo_batch": str(run_dir / ttl.TEN_BATCH_CSV),
                              "khoi_phuc": str(run_dir / ttl.THU_MUC_KP), "epoch_da_luu_lan_nay": gl.epoch_da_luu,
                              "tai_nguyen_csv": str(f_tn)}
        if (run_dir / ttl.TEN_NHAT_KY).is_file():  # chi khi luot da qua buoc bat_dau (nhat ky da co)
            trang_thai = "bi_ngat" if bi_ngat else ("hoan_tat" if res is not None else "loi")
            ttl.ghi_nhat_ky(run_dir, "ket_thuc", lan=gl.lan, trang_thai=trang_thai, loi=rec["loi"],
                            kiem_sau_train_dat=bool(ok), epoch_da_luu_lan_nay=gl.epoch_da_luu,
                            exit_code=130 if bi_ngat else (0 if ok else 3), ket_qua=str(kq_file))
    kq_file.write_text(json.dumps(json_safe(rec), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[ket qua] {kq_file}")
    print(f"[{'DAT' if ok else ('BI NGAT' if bi_ngat else 'LOI')}] {che_do} {a.model}: loi={rec['loi']} "
          f"loi_callback={rec.get('loi_callback')} loss_huu_han={rec.get('loss_huu_han')} "
          f"checkpoint={sorted(rec.get('checkpoint', {}))} thoi_gian={rec['thoi_gian_s']} s "
          f"vram_max={rec['vram_nvidia_smi']['max_mib']} MiB"
          + (f" pilot={rec['pilot']['tom_tat']}" if gp is not None else "")
          + (f" epoch_da_luu={gl.epoch_da_luu}" if gl is not None else ""))
    return 130 if bi_ngat else (0 if ok else 3)


if __name__ == "__main__":
    raise SystemExit(main())
