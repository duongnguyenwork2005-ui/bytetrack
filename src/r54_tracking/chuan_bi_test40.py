"""
chuan_bi_test40.py -- R54: khoa giao thuc tracking test40 va smoke nho (detector -> cache -> ByteTrack
CV/EKF/UKF/PF500 -> TrackEval). Chi la WRAPPER dieu phoi: moi phep tinh (to xam, suy luan, cache, loc
luc load, tracker, evaluator) goi ham HIEN CO cua repo; phan moi chi xu ly anh xa lop, duong dan,
provenance va dieu phoi.

CAC BUOC (khong co buoc full; khong tu khoi chay luot dai)
    tien-kiem  chi doc: manifest 40 video, checkpoint (sha256 + names, khong suy luan), bang hash anh
               (stat moi anh, tu bam lai 256 anh mau), GT / seqinfo / ignored_regions, timeline, D4,
               cau hinh tracker hieu luc. Ghi chuan_bi/<phien>/. Lech -> exit 3, chan khoa/smoke.
    khoa       khoa SHA256 (ma src + Ultralytics + TrackEval thuc nap - ke ca module DetectionPredictor.__init__
               nap luoi, khoi tao khong setup_model/predict -, RECORD thu vien, cau hinh resolve,
               backend, dau vao) sau tien-kiem dat va test fixture dat tren DUNG ma hien tai; dau vao / ma /
               cau hinh phai khop bang chung tien-kiem (lech -> exit 2, khong khoa). Ghi chuan_bi/<phien>/.
    kiem-khoa  bam lai moi file trong khoa moi nhat co khoa.json (hoac --khoa). Khop -> exit 0, lech -> 2.
               Mac dinh CHI DOC; --luu chuan_bi|smoke moi ghi bang chung vao phien moi cua vung do.
    smoke      (R54B) kiem-khoa -> kiem truoc suy luan (backend, ma nap, cau hinh, thu vien) -> 3 cache moi
               tren DUNG 2 x 128 anh da khoa -> 36 to hop x 2 video -> cham -> lap PF -> cham lai ->
               kiem-khoa. Ghi smoke/<phien>/. Che do fixture (R54_GOC_KET_QUA_TEST) dung truoc suy luan, exit 5.
    hau-kiem   (R54B) doi chieu lai mot phien smoke (bam lai cache/GT smoke/MOT/metric/lap PF/cham lai)
               + kiem-khoa. Ghi smoke/<phien>/.

Mau smoke co dinh trong ma (MAU_SMOKE) va trong giao thuc; CLI khong co tham so mo rong mau.
Moi phien la thu muc MOI (ton tai -> exit 1). Thu vien (ultralytics, matplotlib) chi ghi cau hinh vao
<phien>/_cau_hinh_thu_vien, che do offline; dat TRUOC khi import ma du an.

CACH DUNG
    C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r54_tracking/chuan_bi_test40.py --buoc tien-kiem
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import configparser  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata as ilm  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

GOC = Path(__file__).resolve().parents[2]
SRC = GOC / "src"
VUNG_R54 = GOC / "results" / "phase2" / "r54_tracking"
GIAO_THUC = GOC / "configs" / "phase2" / "r54_tracking" / "giao_thuc_test40.json"
TAI_LIEU = GOC / "handoff" / "logs" / "review54" / "01_khoa_giao_thuc_test40.md"
FILE_TEST = Path(__file__).resolve().parent / "test_chuan_bi_test40.py"
BUOC = ("tien-kiem", "khoa", "kiem-khoa", "smoke", "hau-kiem")
#: Bien moi truong CHI cho fixture: doi goc vung ghi vao trong results/phase2/r54_tracking/tests/.
ENV_TEST = "R54_GOC_KET_QUA_TEST"

DETECTORS = ("yolov8n_coco", "yolov8n_mixed5", "yolo11n_mixed5")
DIEU_KIEN = {"c25": 0.25, "c10": 0.10}
BO_LOC = ("cv", "ekf", "ukf", "pf500")
SEEDS_PF = (0, 1, 2)
SO_RUN = 36
SO_VIDEO = 40
SO_ANH = 56340
SO_ANH_D4 = 173
#: Mau smoke chot TRUOC suy luan (R54A); cung 256 anh cho ca ba detector.
MAU_SMOKE = {"videos": ("MVI_39031", "MVI_39051"), "frame_dau": 1, "frame_cuoi": 128}
SPLIT_SMOKE = "DETRAC-smoke128"
FPS = 25.0
TRACK_BUFFER_SEC = 1.2
TRACK_BUFFER_BUOC = 30
IMG_HW = (540, 960)
TEN_HOP_LE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


# ---------------------------------------------------------------------------
# Loi + tien ich
# ---------------------------------------------------------------------------
class LoiR54(Exception):
    ma = 2


class LoiDauVao(LoiR54):
    ma = 2


class LoiGhiDe(LoiR54):
    ma = 1


class LoiTienKiem(LoiR54):
    ma = 3


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_ds(rows) -> str:
    """Nhu danh_gia_detector_bdd_val.sha256_ds: sha256 cac dong '<rel>\\t<sha256>\\n' da sap xep."""
    h = hashlib.sha256()
    for rel, s in sorted(rows):
        h.update(f"{rel}\t{s}\n".encode())
    return h.hexdigest()


def bay_gio() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ten_phien(buoc: str) -> str:
    return f"{buoc.replace('-', '_')}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}_pid{os.getpid()}"


def rel_goc(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(GOC).as_posix()
    except ValueError:
        return str(p)


def trong(p: Path, base: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


def kiem_trong_vung(p: Path, vung: list[Path]) -> Path:
    """Duong dan tuyet doi SAU resolve phai nam trong mot vung cho phep, neu khong -> LoiDauVao."""
    r = Path(p).resolve()
    if not any(trong(r, v) for v in vung):
        raise LoiDauVao(f"duong dan ghi {r} nam ngoai vung cho phep {[str(v) for v in vung]}")
    return r


def vung_ghi() -> dict[str, Path]:
    """Vung ghi theo buoc. Che do fixture (ENV_TEST) chi duoc tro VAO BEN TRONG tests/."""
    goc = VUNG_R54
    tests = VUNG_R54 / "tests"
    t = os.environ.get(ENV_TEST)
    if t:
        p = Path(t).resolve()
        if not trong(p, tests) or p == tests.resolve():
            raise LoiDauVao(f"{ENV_TEST}={t} phai nam ben trong {tests}")
        goc = p
        tests = p / "tests"
    return {"chuan_bi": goc / "chuan_bi", "smoke": goc / "smoke", "tests": tests}


def tao_phien(vung: Path, ten: str | None, buoc: str) -> Path:
    ten = ten or ten_phien(buoc)
    if not TEN_HOP_LE.match(ten) or ".." in ten:
        raise LoiDauVao(f"ten phien {ten!r} khong hop le (chi [A-Za-z0-9_.-], khong '..')")
    p = vung / ten
    kiem_trong_vung(p, [vung])
    if p.exists():
        raise LoiGhiDe(f"phien {p} da ton tai - tu choi ghi de")
    vung.mkdir(parents=True, exist_ok=True)
    p.mkdir(exist_ok=False)
    return p


def ghi_json(p: Path, obj, vung: list[Path], ghi_de: bool = False) -> str:
    p = kiem_trong_vung(p, vung)
    if p.exists() and not ghi_de:
        raise LoiGhiDe(f"{p} da ton tai - tu choi ghi de")
    p.parent.mkdir(parents=True, exist_ok=True)
    b = (json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n").encode("utf-8")
    p.write_bytes(b)
    return sha256_bytes(b)


def dat_moi_truong(phien: Path) -> dict:
    """Cau hinh thu vien chi trong <phien>/_cau_hinh_thu_vien, offline. Goi TRUOC import ultralytics."""
    cfg = phien / "_cau_hinh_thu_vien"
    (cfg / "ultralytics").mkdir(parents=True, exist_ok=True)
    (cfg / "matplotlib").mkdir(parents=True, exist_ok=True)
    env = {"YOLO_CONFIG_DIR": str(cfg / "ultralytics"), "MPLCONFIGDIR": str(cfg / "matplotlib"),
           "YOLO_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "YOLO_VERBOSE": "False"}
    if "ultralytics" in sys.modules and os.environ.get("YOLO_CONFIG_DIR") is None:
        raise LoiR54("ultralytics da duoc import truoc khi dat YOLO_CONFIG_DIR")
    for k, v in env.items():
        if k == "YOLO_CONFIG_DIR" and "ultralytics" in sys.modules:
            continue           # fixture: mot tien trinh, thu vien da nap voi cau hinh trong tests/
        os.environ[k] = v
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    return env


# ---------------------------------------------------------------------------
# Giao thuc
# ---------------------------------------------------------------------------
def doc_giao_thuc(path: Path = GIAO_THUC) -> dict:
    if not path.is_file():
        raise LoiDauVao(f"khong co giao thuc {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ten_run(det: str, cond: str, loc: str, seed: int | None) -> str:
    return f"r54-{det}-{cond}-{loc}" + (f"-s{seed}" if seed is not None else "")


def ma_tran(gt: dict) -> list[dict]:
    """36 to hop: detector x dieu kien x (cv, ekf, ukf, pf500-s0/s1/s2)."""
    rows = []
    for det in gt["detector"]:
        for cond, cmin in gt["tai"]["dieu_kien"].items():
            for bl in gt["tracker"]["bo_loc"]:
                for s in (bl.get("seeds") or [None]):
                    rows.append({"detector": det, "cond": cond, "conf_min": float(cmin), "bo_loc": bl["ten"],
                                 "yaml": bl["yaml"], "seed": s, "run": ten_run(det, cond, bl["ten"], s)})
    return rows


def kiem_giao_thuc(gt: dict) -> list[str]:
    """Giao thuc phai dung pham vi da chot (khong them nhanh). Tra ve danh sach loi."""
    e = []
    if tuple(gt.get("detector", {})) != DETECTORS:
        e.append(f"detector {list(gt.get('detector', {}))} != {list(DETECTORS)}")
    if gt.get("tai", {}).get("dieu_kien") != DIEU_KIEN:
        e.append("dieu_kien khac c25=0.25 / c10=0.10")
    if gt.get("tai", {}).get("classes") != [2, 3, 5, 7] or gt.get("tai", {}).get("max_det") != 300:
        e.append("loc luc load phai classes [2,3,5,7], max_det 300")
    bl = gt.get("tracker", {}).get("bo_loc", [])
    if [b.get("ten") for b in bl] != list(BO_LOC):
        e.append(f"bo_loc {[b.get('ten') for b in bl]} != {list(BO_LOC)}")
    for b in bl:
        want = list(SEEDS_PF) if b.get("ten") == "pf500" else None
        if b.get("seeds") != want:
            e.append(f"seeds cua {b.get('ten')} = {b.get('seeds')} != {want}")
    tr = gt.get("tracker", {})
    if (tr.get("fps"), tr.get("track_buffer_sec"), tr.get("track_buffer_buoc"), tr.get("elapsed_mode")) != (
            FPS, TRACK_BUFFER_SEC, TRACK_BUFFER_BUOC, "skip"):
        e.append("tracker phai fps 25, buffer 1.2 s = 30 buoc, elapsed_mode skip")
    c = gt.get("cache", {})
    want_c = {"imgsz": 640, "conf": 0.05, "iou": 0.7, "max_det": 3000, "half": False, "agnostic_nms": False,
              "augment": False, "classes": None, "mask_ignored_regions": True, "batch": 1}
    for k, v in want_c.items():
        if c.get(k, "THIEU") != v:
            e.append(f"cache.{k} = {c.get(k, 'THIEU')} != {v}")
    m = ma_tran(gt) if not e else []
    if not e and (len(m) != SO_RUN or len({r["run"] for r in m}) != SO_RUN):
        e.append(f"ma tran co {len(m)} run, can {SO_RUN}")
    mt = gt.get("ma_tran", {})
    if not e and (mt.get("so_run") != SO_RUN or mt.get("danh_sach") != [r["run"] for r in m]
                  or mt.get("so_video_moi_run") != SO_VIDEO):
        e.append("ma_tran.danh_sach / so_run / so_video_moi_run khong khop ma_tran() sinh tu giao thuc")
    td = gt.get("tap_du_lieu", {})
    if (td.get("so_video"), td.get("so_anh"), td.get("so_anh_D4")) != (SO_VIDEO, SO_ANH, SO_ANH_D4):
        e.append("tap_du_lieu phai 40 video / 56340 anh / 173 anh D4")
    sm = gt.get("smoke", {})
    if (tuple(sm.get("videos", ())), sm.get("frame_dau"), sm.get("frame_cuoi")) != (
            MAU_SMOKE["videos"], MAU_SMOKE["frame_dau"], MAU_SMOKE["frame_cuoi"]):
        e.append("mau smoke khac MAU_SMOKE trong ma")
    dg = gt.get("danh_gia", {})
    if dg.get("evaluator") != "MotChallenge2DBox" or dg.get("metrics") != ["HOTA", "CLEAR", "Identity"]:
        e.append("evaluator phai MotChallenge2DBox voi HOTA/CLEAR/Identity")
    return e


class Nguon:
    """Duong dan dau vao (that, hoac fixture voi `goc` khac)."""

    def __init__(self, gt: dict, goc: Path = GOC):
        n = gt["nguon"]
        self.goc = Path(goc)
        self.manifest = self.goc / n["manifest"]["rel"]
        self.fingerprint = self.goc / n["fingerprint"]["rel"]
        self.bang_hash = self.goc / n["bang_hash"]["rel"]
        self.loai_d4 = self.goc / n["loai_d4"]["rel"]
        self.khoa_r53 = self.goc / n["khoa_r53"]["rel"]
        self.sha256sums_r53 = self.goc / n["sha256sums_r53"]["rel"]
        self.thu_muc_anh = self.goc / n["thu_muc_anh"]
        self.thu_muc_gt = self.goc / n["thu_muc_gt"]
        self.sha_ky_vong = {k: v.get("sha256") for k, v in n.items() if isinstance(v, dict)}
        self.checkpoint = {d: self.goc / v["checkpoint"] for d, v in gt["detector"].items()}

    def gt_txt(self, v): return self.thu_muc_gt / v / "gt" / "gt.txt"
    def seqinfo(self, v): return self.thu_muc_gt / v / "seqinfo.ini"
    def vung_bo_qua(self, v): return self.thu_muc_gt / v / "gt" / "ignored_regions.txt"


# ---------------------------------------------------------------------------
# Anh xa lop (phan moi duy nhat cham vao detection)
# ---------------------------------------------------------------------------
def bang_anh_xa_coco(names, loai_ky_vong: str) -> dict | None:
    """ID lop model -> ID COCO, theo TEN qua detector_classes. coco80 -> None (giu ID COCO goc).

    Thieu/trung ten trong 5 lop -> LoiAnhXaLop (tu `anh_xa_model_sang_lop5`). Loai checkpoint khac
    loai da khoa -> LoiDauVao.
    """
    import detector_classes as dc
    m = dc.anh_xa_model_sang_lop5(names)              # nem LoiAnhXaLop neu thieu / trung
    loai = dc.loai_checkpoint(names)
    if loai != loai_ky_vong:
        raise LoiDauVao(f"loai checkpoint {loai!r} != {loai_ky_vong!r} da khoa")
    if loai == "coco80":
        return None
    if loai == "lop5":
        return {int(k): dc.sang_id_coco(v) for k, v in m.items()}
    raise LoiDauVao(f"loai checkpoint {loai!r} khong ho tro")


def doi_id_sang_coco(data, bang: dict | None):
    """Doi cot cls (cot 5) theo `bang`; KHONG doi box/conf/thu tu. bang None -> ban sao nguyen."""
    import numpy as np
    a = np.array(data, dtype=np.float32, copy=True).reshape(-1, 6)
    if bang is None or not len(a):
        return a
    goc = a[:, 5].astype(np.int64)
    if not np.all(a[:, 5] == goc):
        raise LoiDauVao("cot cls khong phai so nguyen")
    thieu = sorted(set(goc.tolist()) - set(bang))
    if thieu:
        raise LoiDauVao(f"cls {thieu} khong co trong bang anh xa {bang}")
    a[:, 5] = np.array([bang[int(c)] for c in goc], dtype=np.float32)
    return a


# ---------------------------------------------------------------------------
# Doc dau vao
# ---------------------------------------------------------------------------
def doc_bang_hash(p: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["rel"].split("/")[0], []).append(r)
    return out


def doc_seqinfo(p: Path) -> dict:
    cp = configparser.ConfigParser()
    cp.optionxform = str
    cp.read(p, encoding="utf-8")
    return dict(cp["Sequence"])


def doc_gt(p: Path) -> tuple[list[tuple[int, list[str]]], list[str]]:
    """(danh sach (frame, cot), loi dinh dang)."""
    rows, loi = [], []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        c = line.split(",")
        if len(c) != 9:
            loi.append(f"{p.name}:{i} co {len(c)} cot")
            continue
        try:
            rows.append((int(c[0]), c))
        except ValueError:
            loi.append(f"{p.name}:{i} frame khong phai so")
    return rows, loi


def doc_vung(p: Path) -> list[tuple]:
    """Cung cach doc nhu baseline_track.load_ignored_regions / DetracDataset.ignored_regions."""
    import numpy as np
    if not p.exists() or p.stat().st_size == 0:
        return []
    return [tuple(float(v) for v in row) for row in np.loadtxt(p, delimiter=",", ndmin=2)]


def doc_names_checkpoint(p: Path) -> dict:
    """model.names doc tu checkpoint qua ultralytics (CPU, KHONG suy luan)."""
    from ultralytics import YOLO
    import detector_classes as dc
    return dc.chuan_hoa_names(YOLO(str(p)).names)


# ---------------------------------------------------------------------------
# Cau hinh tracker hieu luc
# ---------------------------------------------------------------------------
def yaml_arg(y: str) -> str:
    """Duong dan YAML trong repo -> tuyet doi (khong phu thuoc cwd); ten tran (bytetrack.yaml) giu nguyen
    de resolver hien hanh (run_tracking.resolve_tracker_yaml) tim trong ultralytics nhu expA."""
    return str(GOC / y) if (GOC / y).is_file() else y


def cau_hinh_hieu_luc(gt: dict) -> tuple[dict, list[str]]:
    import run_tracking as rt
    import ul_backend
    from time_units import track_buffer_frames
    loi, out = [], {}
    if ul_backend.backend_name() != gt["tracker"]["backend"]:
        loi.append(f"backend tracker {ul_backend.backend_name()!r} != {gt['tracker']['backend']!r}")
    tb = track_buffer_frames(TRACK_BUFFER_SEC, FPS)
    if tb != TRACK_BUFFER_BUOC:
        loi.append(f"track_buffer_frames(1.2, 25) = {tb} != 30")
    reg = rt.tracker_registry()
    for bl in gt["tracker"]["bo_loc"]:
        res = Path(rt.resolve_tracker_yaml(yaml_arg(bl["yaml"])))
        cfg = rt.load_tracker_cfg(yaml_arg(bl["yaml"]), {"track_buffer": tb})
        d = rt.cfg_to_dict(cfg)
        rec = {"yaml": bl["yaml"], "yaml_resolve": str(res), "yaml_sha256": sha256_file(res),
               "tracker_type": cfg.tracker_type, "lop": f"{reg[cfg.tracker_type].__module__}."
                                                          f"{reg[cfg.tracker_type].__name__}",
               "cfg_hieu_luc": d, "override": {"track_buffer": tb}, "ngau_nhien": cfg.tracker_type in rt.STOCHASTIC_TRACKERS,
               "seeds": bl.get("seeds")}
        if cfg.tracker_type != bl["tracker_type"]:
            loi.append(f"{bl['ten']}: tracker_type {cfg.tracker_type} != {bl['tracker_type']}")
        if int(cfg.track_buffer) != TRACK_BUFFER_BUOC:
            loi.append(f"{bl['ten']}: track_buffer {cfg.track_buffer}")
        if str(d.get("lost_recover", "none")).lower() not in ("none", "false", "") or d.get("lost_no_fuse"):
            loi.append(f"{bl['ten']}: lost_recover bat (khong phai association goc)")
        if bl["ten"] == "pf500":
            if int(d.get("pf_num_particles", -1)) != 500:
                loi.append("pf500: pf_num_particles != 500")
            if str(d.get("pf_optimal_proposal", False)).strip().lower() in ("1", "true", "yes"):
                loi.append("pf500: K1 (pf_optimal_proposal) dang bat")
            if float(d.get("pf_roughening", 0.0)) != 0.0:
                loi.append("pf500: K2 (pf_roughening) khac 0")
        out[bl["ten"]] = rec
    return {"backend_tracker": ul_backend.backend_name(), "track_buffer_buoc": tb, "bo_loc": out}, loi


# ---------------------------------------------------------------------------
# Tien kiem
# ---------------------------------------------------------------------------
def tien_kiem(gt: dict, nguon: Nguon, doc_names=doc_names_checkpoint, mau: dict | None = None,
              kiem_cau_hinh: bool = True) -> dict:
    """Chi doc. Tra ve dict ket qua (ket_luan 'dat' khi khong co lech)."""
    mau = mau or MAU_SMOKE
    td = gt["tap_du_lieu"]
    lech: list[str] = []
    nguon_hash = {}
    dau_vao_truoc = dau_vao_sha(gt, nguon)
    # 1. File dau vao tong + doi chieu khoa R53
    for k, p in (("manifest", nguon.manifest), ("fingerprint", nguon.fingerprint), ("bang_hash", nguon.bang_hash),
                 ("loai_d4", nguon.loai_d4), ("khoa_r53", nguon.khoa_r53), ("sha256sums_r53", nguon.sha256sums_r53)):
        if not p.is_file():
            raise LoiTienKiem(f"thieu file {k}: {p}")
        s = sha256_file(p)
        nguon_hash[k] = {"rel": rel_goc(p), "sha256": s}
        want = nguon.sha_ky_vong.get(k)
        if want and s != want:
            lech.append(f"sha256 {k} {s} != khoa {want}")
    khoa53 = json.loads(nguon.khoa_r53.read_text(encoding="utf-8"))
    sums53 = {ln.split("  ", 1)[1].strip(): ln.split("  ", 1)[0]
              for ln in nguon.sha256sums_r53.read_text(encoding="utf-8").splitlines() if "  " in ln}
    rel53 = gt["nguon"]["khoa_r53"]["rel"]
    if sums53.get(rel53) != nguon_hash["khoa_r53"]["sha256"]:
        lech.append("khoa R53 khong khop SHA256SUMS R53")
    d53 = khoa53.get("du_lieu", {}).get("detrac_test", {}).get("file", {})
    for k53, k in (("manifest", "manifest"), ("fingerprint", "fingerprint"), ("hash_anh", "bang_hash"),
                   ("loai_d4", "loai_d4")):
        want = (d53.get(k53) or {}).get("sha256")
        if want != nguon_hash[k]["sha256"] or (d53.get(k53) or {}).get("duong_dan") != gt["nguon"][k]["rel"]:
            lech.append(f"{k}: sha256/duong dan {nguon_hash[k]['sha256']} != khoa R53 {want}")
    # 2. Checkpoint
    ckpt = {}
    for det, spec in gt["detector"].items():
        p = nguon.checkpoint[det]
        if not p.is_file():
            raise LoiTienKiem(f"thieu checkpoint {det}: {p}")
        s = sha256_file(p)
        k53 = khoa53.get("mo_hinh", {}).get(det, {})
        rec = {"checkpoint": spec["checkpoint"], "sha256": s, "sha256_giao_thuc": spec["sha256"],
               "sha256_khoa_r53": k53.get("sha256"), "checkpoint_khoa_r53": k53.get("checkpoint")}
        if not (s == spec["sha256"] == k53.get("sha256")) or k53.get("checkpoint") != spec["checkpoint"]:
            lech.append(f"checkpoint {det}: sha256/duong dan khong khop khoa R53")
        try:
            names = doc_names(p)
            bang = bang_anh_xa_coco(names, spec["loai_checkpoint"])
            import detector_classes as dc
            rec.update({"loai_checkpoint": dc.loai_checkpoint(names), "anh_xa": dc.mo_ta_anh_xa(names),
                        "model_id_sang_coco": None if bang is None else {str(k): v for k, v in sorted(bang.items())}})
            if spec["loai_checkpoint"] != k53.get("loai_checkpoint_ky_vong"):
                lech.append(f"{det}: loai checkpoint giao thuc != khoa R53")
        except Exception as e:  # noqa: BLE001
            lech.append(f"checkpoint {det}: anh xa lop loi {type(e).__name__}: {e}")
        ckpt[det] = rec
    # 3. Manifest / fingerprint / bang hash / D4
    man = json.loads(nguon.manifest.read_text(encoding="utf-8"))
    videos = list(man["videos"])
    if len(videos) != td["so_video"] or len(set(videos)) != len(videos):
        lech.append(f"manifest co {len(videos)} video (trung?) != {td['so_video']}")
    fpj = json.loads(nguon.fingerprint.read_text(encoding="utf-8"))
    vt = fpj["van_tay"]
    if vt["manifest"]["videos"] != videos:
        lech.append("danh sach video fingerprint != manifest")
    xml = {x["video"]: x for x in vt["xml"]}
    anh_fp = {x["video"]: x for x in vt["anh"]["theo_video"]}
    bang = doc_bang_hash(nguon.bang_hash)
    if list(bang) != videos:
        lech.append("thu tu video trong bang hash != manifest")
    d4: dict[str, set[int]] = {}
    with open(nguon.loai_d4, newline="", encoding="utf-8") as f:
        n_d4 = 0
        for r in csv.DictReader(f):
            n_d4 += 1
            d4.setdefault(r["video"], set()).add(int(r["frame"]))
            if r["rel"] != f"{r['video']}/img{int(r['frame']):05d}.jpg":
                lech.append(f"D4 rel {r['rel']} khong khop video/frame")
    if n_d4 != td["so_anh_D4"] or sum(len(s) for s in d4.values()) != td["so_anh_D4"]:
        lech.append(f"D4 co {n_d4} dong != {td['so_anh_D4']}")
    tat_ca = [r for v in bang.values() for r in v]
    if len(tat_ca) != td["so_anh"]:
        lech.append(f"bang hash {len(tat_ca)} dong != {td['so_anh']}")
    th = vt["tong_hop"]
    if sha256_bytes("".join(f"{r['rel']}\n" for r in tat_ca).encode()) != th["danh_sach_anh_co_thu_tu_sha256"]:
        lech.append("danh_sach_anh_co_thu_tu_sha256 tinh lai != fingerprint")
    if sha256_ds((r["rel"], r["sha256"]) for r in tat_ca) != th["noi_dung_anh_sha256_tong_hop"]:
        lech.append("noi_dung_anh_sha256_tong_hop tinh lai != fingerprint")
    # 4. Tung video
    mau_rels = {v: [f"{v}/img{i:05d}.jpg" for i in range(mau["frame_dau"], mau["frame_cuoi"] + 1)]
                for v in mau["videos"]}
    bang_rows, anh_tu_bam = [], []
    tong = {"anh": 0, "anh_d4": 0, "frame_co_gt": 0, "dong_gt": 0, "vung": 0, "seq_length": 0,
            "anh_stat_dung_size": 0}
    for v in videos:
        rows = bang.get(v, [])
        lv: list[str] = []
        n = len(rows)
        for i, r in enumerate(rows, 1):
            if r["rel"] != f"{v}/img{i:05d}.jpg":
                lv.append(f"timeline: dong {i} = {r['rel']} (khong lien tuc tu 1)")
                break
        if any((r["w"], r["h"]) != ("960", "540") for r in rows):
            lv.append("kich thuoc anh khac 960x540 trong bang hash")
        if n >= 100000:
            lv.append("so anh >= 100000: thu tu glob sorted khong con bang thu tu so")
        af = anh_fp.get(v, {})
        if (af.get("so_anh"), af.get("frame_dau"), af.get("frame_cuoi")) != (n, 1, n):
            lv.append(f"bang hash ({n}) != fingerprint anh ({af.get('so_anh')}, {af.get('frame_dau')}-{af.get('frame_cuoi')})")
        # anh: stat (ke thua hash) + tu bam lai mau smoke
        n_ok = 0
        for r in rows:
            p = nguon.thu_muc_anh / r["rel"]
            try:
                if p.stat().st_size == int(r["size"]):
                    n_ok += 1
            except OSError:
                pass
        if n_ok != n:
            lv.append(f"{n - n_ok} anh thieu / sai size so voi bang hash")
        tong["anh_stat_dung_size"] += n_ok
        if v in mau_rels:
            by_rel = {r["rel"]: r for r in rows}
            for rel in mau_rels[v]:
                r = by_rel.get(rel)
                p = nguon.thu_muc_anh / rel
                s = sha256_file(p) if p.is_file() else "THIEU"
                ok = r is not None and s == r["sha256"]
                anh_tu_bam.append({"rel": rel, "sha256_bang": r["sha256"] if r else None, "sha256_tu_bam": s, "khop": ok})
                if not ok:
                    lv.append(f"anh mau {rel}: sha256 tu bam != bang hash")
        # seqinfo
        si_p, gt_p, vb_p = nguon.seqinfo(v), nguon.gt_txt(v), nguon.vung_bo_qua(v)
        for nm, p in (("seqinfo", si_p), ("gt", gt_p), ("ignored_regions", vb_p)):
            if not p.is_file():
                lv.append(f"thieu {nm}: {rel_goc(p)}")
        if lv and any(x.startswith("thieu ") for x in lv):
            lech += [f"{v}: {x}" for x in lv]
            bang_rows.append({"video": v, "so_anh": n, "ket_luan": "khong_dat", "lech": " | ".join(lv)})
            continue
        si = doc_seqinfo(si_p)
        sl = int(si.get("seqLength", -1))
        if si.get("name") != v or sl != n or int(si.get("frameRate", -1)) != int(FPS) \
                or (si.get("imWidth"), si.get("imHeight")) != ("960", "540"):
            lv.append(f"seqinfo name/seqLength({sl})/frameRate/kich thuoc khong khop (so anh {n})")
        # gt
        gt_rows, loi_gt = doc_gt(gt_p)
        lv += loi_gt
        frames_gt = {f for f, _ in gt_rows}
        if any(f < 1 or f > n for f in frames_gt):
            lv.append("GT co frame ngoai 1..so anh")
        if any(c[6] != "1" or c[7] != "1" for _, c in gt_rows):
            lv.append("GT co dong conf != 1 hoac class != 1 (vehicle gop)")
        khong_gt = set(range(1, n + 1)) - frames_gt
        d4v = d4.get(v, set())
        if not d4v <= khong_gt:
            lv.append(f"{len(d4v - khong_gt)} anh D4 lai co dong GT")
        xv = xml.get(v, {})
        if xv.get("so_frame") != n - len(d4v):
            lv.append(f"so <frame> XML {xv.get('so_frame')} != so anh - D4 ({n - len(d4v)})")
        if (af.get("so_anh_loai_D4") or 0) != len(d4v):
            lv.append("so anh D4 fingerprint != file D4")
        vung = doc_vung(vb_p)
        if len(vung) != xv.get("so_vung_bo_qua"):
            lv.append(f"ignored_regions {len(vung)} dong != XML {xv.get('so_vung_bo_qua')}")
        tong["anh"] += n
        tong["anh_d4"] += len(d4v)
        tong["frame_co_gt"] += len(frames_gt)
        tong["dong_gt"] += len(gt_rows)
        tong["vung"] += len(vung)
        tong["seq_length"] += sl
        import run_fingerprint as rf
        bang_rows.append({
            "video": v, "so_anh": n, "seq_length": sl, "frame_dau": 1, "frame_cuoi": n,
            "so_anh_D4": len(d4v), "so_frame_xml": xv.get("so_frame"), "so_frame_co_gt": len(frames_gt),
            "so_frame_khong_gt_ngoai_D4": len(khong_gt - d4v), "so_dong_gt": len(gt_rows),
            "so_box_xml": xv.get("so_box"), "so_vung_bo_qua": len(vung), "anh_stat_dung_size": n_ok,
            "anh_tu_bam_lai": sum(1 for a in anh_tu_bam if a["rel"].startswith(v + "/")),
            "gt_sha256": sha256_file(gt_p), "gt_sha256_lf": rf.sha256_file(gt_p),
            "seqinfo_sha256": sha256_file(si_p), "seqinfo_sha256_lf": rf.sha256_file(si_p),
            "ignored_regions_sha256": sha256_file(vb_p),
            "ket_luan": "dat" if not lv else "khong_dat", "lech": " | ".join(lv)})
        lech += [f"{v}: {x}" for x in lv]
    for k, want in (("anh", td["so_anh"]), ("seq_length", td["so_anh"]), ("anh_d4", td["so_anh_D4"])):
        if tong[k] != want:
            lech.append(f"tong {k} = {tong[k]} != {want}")
    if tong["anh"] - tong["anh_d4"] != td.get("so_anh_co_frame_xml", tong["anh"] - tong["anh_d4"]):
        lech.append("tong anh - D4 != so_anh_co_frame_xml")
    if len(anh_tu_bam) != sum(len(x) for x in mau_rels.values()):
        lech.append("mau smoke khong du anh trong bang hash")
    cfg, loi_cfg = ({}, []) if not kiem_cau_hinh else cau_hinh_hieu_luc(gt)
    lech += loi_cfg
    dau_vao = dau_vao_sha(gt, nguon)
    doi = sorted(k for k in dau_vao if dau_vao[k] != dau_vao_truoc.get(k))
    if doi:
        lech.append(f"dau vao doi trong luc tien-kiem: {doi[:5]}")
    lech += [f"thieu file dau vao {k}" for k, s in dau_vao.items() if s == "THIEU"]
    return {"ket_luan": "dat" if not lech else "khong_dat", "lech": lech, "nguon": nguon_hash,
            "dau_vao_sha256": dau_vao,
            "checkpoint": ckpt, "tong": tong, "bang": bang_rows, "anh_tu_bam_lai": anh_tu_bam,
            "cau_hinh_hieu_luc": cfg,
            "mau_smoke": {"videos": list(mau["videos"]), "frame_dau": mau["frame_dau"], "frame_cuoi": mau["frame_cuoi"],
                          "so_anh": len(anh_tu_bam), "rel": [a["rel"] for a in anh_tu_bam],
                          "danh_sach_sha256": sha256_bytes("".join(f"{a['rel']}\t{a['sha256_bang']}\n"
                                                                   for a in anh_tu_bam).encode())}}


GIOI_HAN_TIEN_KIEM = [
    "Anh: chi stat ton tai + size theo bang hash R47 (hash KE THUA, khong bam lai); chi 256 anh mau smoke duoc tu bam lai.",
    "GT/seqinfo/ignored_regions: khong co hash tham chieu ke thua cho test40 (lock_moc2.json khong ghi); tien kiem ghi hash hien tai, khoa R54 co dinh tu day.",
    "Doi chieu GT voi XML qua so dem trong fingerprint R47 (so <frame>, so vung bo qua); khong doc lai XML, khong kiem toa do box.",
    "names checkpoint doc qua ultralytics tren CPU; khong suy luan, khong GPU.",
    "Khong kiem pixel to xam tren anh that o moc A (chi fixture).",
]


def ghi_bang_csv(p: Path, rows: list[dict], vung: list[Path]) -> str:
    p = kiem_trong_vung(p, vung)
    if p.exists():
        raise LoiGhiDe(f"{p} da ton tai")
    cols = list(rows[0]) if rows else ["video"]
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return sha256_file(p)


def buoc_tien_kiem(gt: dict, vung: dict, args) -> int:
    phien = tao_phien(vung["chuan_bi"], args.lan, "tien-kiem")
    dat_moi_truong(phien)
    t0 = time.time()
    kq = tien_kiem(gt, Nguon(gt))
    v = [phien]
    sha_bang = ghi_bang_csv(phien / "bang_tien_kiem.csv", kq.pop("bang"), v)
    sha_mau = ghi_json(phien / "mau_smoke.json", kq["mau_smoke"] | {"anh": kq.pop("anh_tu_bam_lai")}, v)
    ma = 0 if kq["ket_luan"] == "dat" else LoiTienKiem.ma
    kq.update({"buoc": "tien-kiem", "phien": rel_goc(phien), "luc": bay_gio(), "thoi_gian_s": round(time.time() - t0, 1),
               "giao_thuc_sha256": sha256_file(GIAO_THUC), "wrapper_sha256": sha256_file(Path(__file__)),
               "backend_env": os.environ.get("BYTETRACK_BACKEND"),
               "ma_da_nap_sha256": {rel_goc(p): sha256_file(p) for p in file_module_da_nap()},
               "bang_tien_kiem_sha256": sha_bang, "mau_smoke_sha256": sha_mau, "gioi_han": GIOI_HAN_TIEN_KIEM,
               "exit": ma})
    ghi_json(phien / "tien_kiem.json", kq, v)
    print(f"[tien-kiem] {kq['ket_luan']} ({len(kq['lech'])} lech) -> {rel_goc(phien)}  exit {ma}")
    for x in kq["lech"][:30]:
        print("   -", x)
    return ma


# ---------------------------------------------------------------------------
# Khoa / kiem khoa
# ---------------------------------------------------------------------------
MODULE_SMOKE = ("config", "det_cache", "detector_classes", "baseline_track", "run_tracking", "run_trackeval",
                "run_fingerprint", "run_state", "tracker_ctrv", "time_units", "ul_backend", "assoc_log",
                "boot_stats", "mot_datasets", "mot_datasets.detrac")
#: Ma doi chieu (khong import truc tiep nhung la nguon cua quy uoc duoc lap lai)
MA_DOI_CHIEU = ("src/run_experiment.py",)


#: R54A-F2: module Ultralytics THUC THI detector (YOLO/predict/AutoBackend PyTorch/NMS/letterbox) va nen
#: ByteTrack (byte_tracker, basetrack, matching, KalmanFilterXYAH). Nhieu module duoc ultralytics import LUOI
#: trong predict(); nap truoc o day de chung vao khoa va duoc kiem TRUOC suy luan. Import loi -> khoa dung.
MODULE_ULTRALYTICS = ("ultralytics", "ultralytics.engine.model", "ultralytics.engine.predictor",
                      "ultralytics.engine.results", "ultralytics.engine.exporter",
                      "ultralytics.models.yolo.detect.predict", "ultralytics.nn.tasks", "ultralytics.nn.autobackend",
                      "ultralytics.nn.backends.base", "ultralytics.nn.backends.pytorch", "ultralytics.utils.nms",
                      "ultralytics.utils.ops", "ultralytics.utils.checks", "ultralytics.data.augment",
                      "ultralytics.data.loaders", "ultralytics.data.build", "ultralytics.trackers",
                      "ultralytics.trackers.byte_tracker", "ultralytics.trackers.basetrack",
                      "ultralytics.trackers.utils.matching", "ultralytics.trackers.utils.kalman_filter",
                      "ultralytics.utils.files", "torchvision")
#: R54A-F2 (vong 2): nap LUOI trong BasePredictor.__init__ (callbacks.add_integration_callbacks -> utils.events).
#: KHONG import truoc: de khoi_tao_predictor ghi nhan module moi do chinh predictor nap; import lai sau do (no-op).
MODULE_ULTRALYTICS_SAU_PREDICTOR = ("ultralytics.utils.events",)
TEN_PREDICTOR_KHONG_GHI = "_predictor_khong_ghi"
MODULE_TRACKEVAL = ("trackeval", "trackeval.eval", "trackeval.datasets.mot_challenge_2d_box", "trackeval.metrics.hota",
                    "trackeval.metrics.clear", "trackeval.metrics.identity")
#: Goi nhi phan / thu vien ngoai: chi khoa phien ban + SHA256 file RECORD cua dist-info (khong bam tung file).
GOI_THU_VIEN = ("ultralytics", "torch", "torchvision", "trackeval", "numpy", "pandas", "opencv-python", "scipy",
                "lap", "fastparquet", "filterpy")


def tuyet_doi(rel: str) -> Path:
    """rel trong khoa (tuong doi GOC, hoac tuyet doi khi ngoai repo) -> duong dan tuyet doi da resolve."""
    return (Path(rel) if Path(rel).is_absolute() else GOC / rel).resolve()


def chuan_json(x):
    return json.loads(json.dumps(x, sort_keys=True, ensure_ascii=False, default=str))


def goc_ma_ngoai() -> dict[str, Path]:
    import trackeval
    import ultralytics
    return {"ultralytics": Path(ultralytics.__file__).resolve().parent,
            "trackeval": Path(trackeval.__file__).resolve().parent}


def file_module_da_nap() -> list[Path]:
    """File nguon cua MOI module dang nap thuoc src/, goi ultralytics, goi trackeval."""
    goc = [SRC] + [Path(sys.modules[g].__file__).resolve().parent for g in ("ultralytics", "trackeval")
                   if getattr(sys.modules.get(g), "__file__", None)]
    return sorted({Path(m.__file__).resolve() for m in list(sys.modules.values())
                   if getattr(m, "__file__", None) and any(trong(Path(m.__file__), g) for g in goc)})


def kiem_env_backend(gt: dict) -> str | None:
    """BYTETRACK_BACKEND ke thua tu moi truong phai trong hoac = backend da khoa (goi TRUOC import tracker)."""
    env = os.environ.get("BYTETRACK_BACKEND")
    if env is not None and env != gt["tracker"]["backend"]:
        raise LoiDauVao(f"BYTETRACK_BACKEND={env!r} != backend da khoa {gt['tracker']['backend']!r}")
    return env


def thong_tin_backend(gt: dict) -> dict:
    """Backend ByteTrack THUC NAP (ul_backend._load) + file nguon cua tung lop nen."""
    import ul_backend
    d = ul_backend._load()
    if d["backend"] != gt["tracker"]["backend"]:
        raise LoiDauVao(f"backend tracker thuc nap {d['backend']!r} != {gt['tracker']['backend']!r}")
    f = {k: rel_goc(Path(sys.modules[d[k].__module__].__file__)) for k in ("BYTETracker", "STrack", "BaseTrack",
                                                                            "KalmanFilterXYAH")}
    f["matching"] = rel_goc(Path(d["matching"].__file__))
    return {"backend": d["backend"], "file": f, "fingerprint": ul_backend.fingerprint()}


def khoi_tao_predictor(gt: dict, det: str, model, ckpt: Path, thu_muc: Path) -> dict:
    """R54A-F2 (vong 2): dung predictor Y NHU Model.predict luc chua co predictor (ultralytics engine/model.py:
    args = {**overrides, **custom, **kw}; _smart_load("predictor")(overrides=args, _callbacks=model.callbacks)),
    kw lay tu tao_spec nhu det_cache.detect_frame. __init__ nap luoi utils.events (add_integration_callbacks) va
    utils.files (get_save_dir). KHONG setup_model / predict / forward / warmup / chon thiet bi.
    project = thu_muc (tuyet doi, trong VUNG_R54); thu muc luu cua predictor KHONG duoc tao."""
    import torch
    thu_muc = kiem_trong_vung(thu_muc, [VUNG_R54])
    s = tao_spec(gt, det, Path(ckpt), "khong_dung")
    kw = {"conf": s.conf, "iou": s.iou, "imgsz": s.imgsz, "classes": s.classes, "max_det": s.max_det,
          "agnostic_nms": s.agnostic_nms, "augment": s.augment, "device": s.device, "verbose": False}
    if s.half:
        kw["half"] = True
    custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict", "rect": True, "embed": None}
    args = {**model.overrides, **custom, **kw, "project": str(thu_muc), "name": TEN_PREDICTOR_KHONG_GHI}
    dich = thu_muc / TEN_PREDICTOR_KHONG_GHI
    if dich.exists():
        raise LoiGhiDe(f"{dich} da ton tai")
    truoc = set(file_module_da_nap())
    p = model._smart_load("predictor")(overrides=args, _callbacks=model.callbacks)
    moi = sorted(set(file_module_da_nap()) - truoc)
    from ultralytics.utils import events as ul_events
    info = {"lop": f"{type(p).__module__}.{type(p).__name__}", "setup_model": p.model is not None,
            "save_dir": str(p.save_dir), "save_dir_trong_vung": trong(p.save_dir, thu_muc),
            "tao_thu_muc": dich.exists(), "save": bool(p.args.save), "device_args": str(p.args.device),
            "events_callback": ul_events.on_predict_end in p.callbacks.get("on_predict_end", []),
            "events_bat": bool(ul_events.events.enabled), "cuda_da_khoi_tao": bool(torch.cuda.is_initialized()),
            "module_moi": [rel_goc(x) for x in moi]}
    loi = [k for k, ok in (("lop != DetectionPredictor", type(p).__name__ == "DetectionPredictor"),
                           ("setup_model da chay", not info["setup_model"]),
                           ("save_dir ngoai thu_muc", info["save_dir_trong_vung"]),
                           ("predictor da tao thu muc", not info["tao_thu_muc"]),
                           ("save bat", not info["save"]),
                           ("events analytics BAT (khong offline)", not info["events_bat"]),
                           ("CUDA da khoi tao", not info["cuda_da_khoi_tao"])) if not ok]
    if loi:
        raise LoiDauVao(f"khoi tao predictor {det}: {loi}")
    return info


def phu_thuoc_thuc_dung(gt: dict, nguon: Nguon, thu_muc_predictor: Path) -> tuple[list[Path], dict]:
    """Nap moi module ma smoke dung (src, Ultralytics detector + nen ByteTrack, TrackEval), lop nn cua 3
    checkpoint (YOLO(ckpt) tren CPU) va KHOI TAO DetectionPredictor cua tung checkpoint (khoi_tao_predictor),
    KHONG suy luan. Tra ve (file nguon da nap - xem file_module_da_nap, thong tin predictor theo detector)."""
    import importlib
    kiem_env_backend(gt)

    def nap(ds):
        for m in ds:
            try:
                importlib.import_module(m)
            except ImportError as e:
                raise LoiDauVao(f"khong nap duoc module phu thuoc {m}: {e}") from e
    nap(MODULE_SMOKE + MODULE_ULTRALYTICS + MODULE_TRACKEVAL)
    import run_tracking as rt
    rt.tracker_registry()
    thong_tin_backend(gt)
    from ultralytics import YOLO
    pred = {}
    for d in gt["detector"]:
        pred[d] = khoi_tao_predictor(gt, d, YOLO(str(nguon.checkpoint[d])), nguon.checkpoint[d], thu_muc_predictor)
    nap(MODULE_ULTRALYTICS_SAU_PREDICTOR)
    return file_module_da_nap(), pred


def file_record_thu_vien() -> dict[str, Path]:
    out = {}
    for pk in GOI_THU_VIEN:
        try:
            dist = ilm.distribution(pk)
        except ilm.PackageNotFoundError:
            continue
        rec = [f for f in (dist.files or []) if f.name == "RECORD"]
        if rec and Path(dist.locate_file(rec[0])).is_file():
            out[pk] = Path(dist.locate_file(rec[0])).resolve()
    return out


def phien_moi_nhat(vung: Path, tien_to: str, can_file: str | None = None) -> Path | None:
    """Phien moi nhat theo ten; `can_file` -> chi tinh phien CO file do (bo qua phien that bai)."""
    if not vung.is_dir():
        return None
    ds = sorted(p for p in vung.iterdir() if p.is_dir() and p.name.startswith(tien_to)
                and (can_file is None or (p / can_file).is_file()))
    return ds[-1] if ds else None


def dau_vao_sha(gt: dict, nguon: Nguon) -> dict[str, str]:
    """R54A-F4: SHA256 dau vao KHONG can import thu vien: giao thuc, wrapper, file nguon, 3 checkpoint,
    GT/seqinfo/ignored_regions cua moi video trong manifest. Ghi o tien-kiem, doi chieu lai o khoa."""
    ps = [GIAO_THUC, Path(__file__), nguon.manifest, nguon.fingerprint, nguon.bang_hash, nguon.loai_d4,
          nguon.khoa_r53, nguon.sha256sums_r53] + [nguon.checkpoint[d] for d in gt["detector"]]
    if nguon.manifest.is_file():
        for v in json.loads(nguon.manifest.read_text(encoding="utf-8"))["videos"]:
            ps += [nguon.gt_txt(v), nguon.seqinfo(v), nguon.vung_bo_qua(v)]
    return {rel_goc(p): (sha256_file(p) if p.is_file() else "THIEU") for p in ps}


def doi_chieu_tien_kiem(tkj: dict, hien_tai: dict[str, str], nhom=("dau_vao_sha256",),
                        tk_dir: Path | None = None) -> list[str]:
    """R54A-F4: bang chung tien-kiem (dat) phai khop trang thai hien tai TRUOC khi khoa."""
    lech = []
    if tkj.get("ket_luan") != "dat" or tkj.get("exit") != 0:
        lech.append("tien-kiem khong dat")
    if tk_dir is not None:
        for f, k in (("bang_tien_kiem.csv", "bang_tien_kiem_sha256"), ("mau_smoke.json", "mau_smoke_sha256")):
            p = tk_dir / f
            s = sha256_file(p) if p.is_file() else "THIEU"
            if s != tkj.get(k):
                lech.append(f"{f}: {s} != tien_kiem.json {tkj.get(k)}")
    for nh in nhom:
        m = tkj.get(nh)
        if not m:
            lech.append(f"tien_kiem.json thieu {nh}")
            continue
        for rel, s in m.items():
            if rel not in hien_tai:
                lech.append(f"{rel}: co trong tien-kiem ({nh}) nhung khong co trong khoa")
            elif hien_tai[rel] != s:
                lech.append(f"{rel}: {hien_tai[rel]} != tien-kiem {s} (doi sau tien-kiem)")
    return lech


def test_dat_moi_nhat(vung_tests: Path) -> tuple[Path | None, dict | None]:
    """Ket qua fixture moi nhat co tat_ca_dat va sha256 wrapper/test = ban hien tai."""
    if not vung_tests.is_dir():
        return None, None
    sw, st = sha256_file(Path(__file__)), sha256_file(FILE_TEST)
    for p in sorted(vung_tests.glob("lan_*/ket_qua_test.json"), reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("tat_ca_dat") and d.get("wrapper_sha256") == sw and d.get("test_sha256") == st:
            return p, d
    return None, None


def vai_tro_ma(p: Path, goc_ngoai: dict[str, Path]) -> str:
    if trong(p, SRC):
        return "ma_phu_thuoc"
    for ten, g in goc_ngoai.items():
        if trong(p, g):
            return f"ma_{ten}"
    return "ma_khac"


def danh_sach_khoa(gt: dict, nguon: Nguon, tk_phien: Path, test_json: Path,
                   thu_muc_predictor: Path) -> tuple[list[dict], dict, dict, dict]:
    """(file khoa, cau hinh tracker hieu luc, backend thuc nap, thong tin khoi tao predictor)."""
    nap, pred = phu_thuoc_thuc_dung(gt, nguon, thu_muc_predictor)
    bk = thong_tin_backend(gt)
    goc_ngoai = goc_ma_ngoai()
    cfg, loi = cau_hinh_hieu_luc(gt)
    if loi:
        raise LoiDauVao(f"cau hinh hieu luc lech: {loi}")
    man = json.loads(nguon.manifest.read_text(encoding="utf-8"))
    items: list[tuple[Path, str]] = [
        (GIAO_THUC, "giao_thuc"), (TAI_LIEU, "tai_lieu_01"), (Path(__file__).resolve(), "wrapper"),
        (FILE_TEST, "test_fixture"), (test_json, "ket_qua_test_fixture"),
        (tk_phien / "tien_kiem.json", "tien_kiem"), (tk_phien / "bang_tien_kiem.csv", "tien_kiem"),
        (tk_phien / "mau_smoke.json", "tien_kiem"),
        (nguon.manifest, "du_lieu"), (nguon.fingerprint, "du_lieu"), (nguon.bang_hash, "du_lieu"),
        (nguon.loai_d4, "du_lieu"), (nguon.khoa_r53, "khoa_r53"), (nguon.sha256sums_r53, "khoa_r53")]
    items += [(nguon.checkpoint[d], "checkpoint") for d in gt["detector"]]
    for v in man["videos"]:
        items += [(nguon.gt_txt(v), "gt"), (nguon.seqinfo(v), "seqinfo"), (nguon.vung_bo_qua(v), "ignored_regions")]
    items += [(Path(r["yaml_resolve"]), "yaml_tracker") for r in cfg["bo_loc"].values()]
    items += [(p, vai_tro_ma(p, goc_ngoai)) for p in nap] + [(GOC / r, "ma_doi_chieu") for r in MA_DOI_CHIEU]
    items += [(p, f"thu_vien_record:{pk}") for pk, p in file_record_thu_vien().items()]
    for k, rel in bk["file"].items():
        if tuyet_doi(rel) not in nap:
            raise LoiDauVao(f"file nen ByteTrack {k} ({rel}) khong nam trong module da nap")
    seen, out = set(), []
    for p, vai in items:
        p = Path(p).resolve()
        if p in seen:
            continue
        seen.add(p)
        if not p.is_file():
            raise LoiDauVao(f"file can khoa khong ton tai: {p}")
        out.append({"rel": rel_goc(p), "vai_tro": vai, "size": p.stat().st_size, "sha256": sha256_file(p)})
    return out, cfg, bk, pred


def phien_ban_thu_vien() -> dict:
    out = {"python": sys.version.split()[0]}
    for pk in GOI_THU_VIEN:
        try:
            out[pk] = ilm.version(pk)
        except ilm.PackageNotFoundError:
            out[pk] = None
    return out


def buoc_khoa(gt: dict, vung: dict, args) -> int:
    tk = (vung["chuan_bi"] / args.tien_kiem) if args.tien_kiem else \
        phien_moi_nhat(vung["chuan_bi"], "tien_kiem_", "tien_kiem.json")
    if tk is None or not (tk / "tien_kiem.json").is_file():
        raise LoiDauVao("chua co phien tien-kiem")
    kiem_trong_vung(tk, [vung["chuan_bi"]])
    tkj = json.loads((tk / "tien_kiem.json").read_text(encoding="utf-8"))
    if tkj.get("ket_luan") != "dat" or tkj.get("exit") != 0:
        raise LoiDauVao(f"tien-kiem {tk.name} khong dat - khong khoa")
    if tkj.get("giao_thuc_sha256") != sha256_file(GIAO_THUC):
        raise LoiDauVao("giao thuc da doi sau tien-kiem - chay lai tien-kiem")
    tj, td = test_dat_moi_nhat(vung["tests"])
    if tj is None:
        raise LoiDauVao("chua co ket qua fixture dat tren DUNG wrapper/test hien tai")
    if not TAI_LIEU.is_file():
        raise LoiDauVao(f"thieu tai lieu {TAI_LIEU}")
    nguon = Nguon(gt)
    # F4 (1): dau vao du lieu/checkpoint/wrapper/giao thuc + bang tien-kiem phai khop bang chung tien-kiem,
    # kiem TRUOC khi tao phien (khong can import thu vien) -> lech thi khong ghi gi.
    lech = doi_chieu_tien_kiem(tkj, dau_vao_sha(gt, nguon), ("dau_vao_sha256",), tk)
    if lech:
        raise LoiDauVao(f"dau vao khac bang chung tien-kiem {tk.name} ({len(lech)} lech): {lech[:5]}")
    phien = tao_phien(vung["chuan_bi"], args.lan, "khoa")
    dat_moi_truong(phien)
    files, cfg, bk, pred = danh_sach_khoa(gt, nguon, tk, tj, phien)
    # F4 (2): ma da nap luc tien-kiem va cau hinh tracker hieu luc phai khop trang thai dang khoa.
    hien_tai = {f["rel"]: f["sha256"] for f in files}
    lech = doi_chieu_tien_kiem(tkj, hien_tai, ("dau_vao_sha256", "ma_da_nap_sha256"), tk)
    if chuan_json(cfg) != chuan_json(tkj.get("cau_hinh_hieu_luc")):
        lech.append("cau hinh tracker hieu luc khac tien-kiem")
    if lech:
        ghi_json(phien / "khoa_that_bai.json", {"buoc": "khoa", "tien_kiem": rel_goc(tk), "luc": bay_gio(),
                                                "lech": lech, "exit": 2}, [phien])
        print(f"[khoa] KHONG khoa: {len(lech)} lech so voi tien-kiem -> {rel_goc(phien)}/khoa_that_bai.json  exit 2",
              file=sys.stderr)
        for x in lech[:20]:
            print("   -", x, file=sys.stderr)
        return 2
    khoa = {"phien_ban": 3, "buoc": "R54A_LOCK", "tao_luc": bay_gio(), "phien": rel_goc(phien),
            "tien_kiem": rel_goc(tk), "ket_qua_test": rel_goc(tj), "so_test": td.get("so_test"),
            "ma_tran": [r["run"] for r in ma_tran(gt)], "mau_smoke": tkj["mau_smoke"],
            "backend_tracker": bk, "cau_hinh_hieu_luc": cfg, "predictor_khoi_tao": pred, "goc_ma_ngoai": {k: str(v) for k, v in goc_ma_ngoai().items()},
            "thu_vien": phien_ban_thu_vien(), "so_file": len(files), "file": files,
            "doi_chieu_tien_kiem": {"dau_vao": len(tkj["dau_vao_sha256"]), "ma_da_nap": len(tkj["ma_da_nap_sha256"]),
                                    "lech": 0},
            "ghi_chu": "khoa bang SHA256 day du + thoi diem; KHONG commit; sua sau khoa -> phien khoa moi, giu phien cu"}
    sha = ghi_json(phien / "khoa.json", khoa, [phien])
    (phien / "SHA256SUMS").write_text(f"{sha}  khoa.json\n", encoding="utf-8")
    ghi_json(phien / "thoi_diem_khoa.json", {"tao_luc": khoa["tao_luc"], "khoa_sha256": sha,
                                             "time_ns": time.time_ns()}, [phien])
    print(f"[khoa] {len(files)} file -> {rel_goc(phien)}  khoa.json {sha}")
    return 0


def kiem_khoa(vung: dict, ten: str | None) -> dict:
    """Bam lai moi file cua phien khoa. CHI DOC, khong ghi gi."""
    k = (vung["chuan_bi"] / ten) if ten else phien_moi_nhat(vung["chuan_bi"], "khoa_", "khoa.json")
    if k is None or not (k / "khoa.json").is_file():
        raise LoiDauVao("chua co phien khoa")
    kiem_trong_vung(k, [vung["chuan_bi"]])
    lech = []
    sk = sha256_file(k / "khoa.json")
    sums = (k / "SHA256SUMS").read_text(encoding="utf-8").split() if (k / "SHA256SUMS").is_file() else []
    if sums[:2] != [sk, "khoa.json"]:
        lech.append(f"khoa.json {sk} != SHA256SUMS {sums[:1]}")
    khoa = json.loads((k / "khoa.json").read_text(encoding="utf-8"))
    for f in khoa["file"]:
        p = tuyet_doi(f["rel"])
        s = sha256_file(p) if p.is_file() else "THIEU"
        if s != f["sha256"]:
            lech.append(f"{f['rel']}: {s} != {f['sha256']}")
    return {"khoa": rel_goc(k), "khoa_sha256": sk, "so_file": len(khoa["file"]), "lech": lech,
            "dat": not lech, "luc": bay_gio(), "_khoa": khoa, "_dir": k}


def buoc_kiem_khoa(gt: dict, vung: dict, args) -> int:
    """R54A-F3: mac dinh CHI DOC (in ket qua + exit, khong tao phien). `--luu chuan_bi|smoke` moi ghi bang chung
    vao phien MOI cua vung do (R54A: chuan_bi; R54B: smoke - dung write_paths tung buoc)."""
    if args.lan and not args.luu:
        raise LoiDauVao("--lan chi dung kem --luu (kiem-khoa mac dinh khong ghi)")
    kq = kiem_khoa(vung, args.khoa)
    ma = 0 if kq["dat"] else 2
    print(f"[kiem-khoa] {kq['khoa']} (khoa.json {kq['khoa_sha256']}): {kq['so_file']} file, "
          f"{len(kq['lech'])} lech -> exit {ma}")
    for x in kq["lech"][:20]:
        print("   -", x)
    if args.luu:
        phien = tao_phien(vung[args.luu], args.lan, "kiem-khoa")
        ghi_json(phien / "ket_qua.json", {k: v for k, v in kq.items() if not k.startswith("_")}
                 | {"exit": ma, "luu": args.luu}, [phien])
        print(f"   ket qua -> {rel_goc(phien)}/ket_qua.json")
    return ma


# ---------------------------------------------------------------------------
# Cache / tracking / danh gia (goi ham hien co)
# ---------------------------------------------------------------------------
def tao_spec(gt: dict, det: str, ckpt: Path, sha: str):
    import det_cache
    c = gt["cache"]
    return det_cache.DetectorSpec(backend="ultralytics", weights=str(ckpt), weights_sha256=sha, imgsz=c["imgsz"],
                                  conf=c["conf"], iou=c["iou"], classes=c["classes"], max_det=c["max_det"],
                                  half=c["half"], agnostic_nms=c["agnostic_nms"], augment=c["augment"],
                                  batch=c["batch"], device=str(c["device"]),
                                  mask_ignored_regions=c["mask_ignored_regions"], dataset="detrac", part="test")


def khung_mau(nguon: Nguon, video: str, mau: dict, bang_video: list[dict], so_khop_dataset: bool):
    """FrameRef cho frame mau (chi so 1-based). time_s tinh Y HET DetracDataset.frames: (i-1) * dt voi
    dt = dt_nominal = 1.0 / fps (R54A-F1: (i-1)/fps lech 1 ulp o 13/128 frame, vd frame 36)."""
    from mot_datasets import FrameRef
    from mot_datasets.detrac import DetracDataset
    if float(DetracDataset.fps) != FPS:
        raise LoiDauVao(f"DetracDataset.fps {DetracDataset.fps} != {FPS}")
    dt = 1.0 / float(DetracDataset.fps)        # = DatasetAdapter.dt_nominal
    by = {r["rel"]: r for r in bang_video}
    frames, hashes = [], []
    for i in range(mau["frame_dau"], mau["frame_cuoi"] + 1):
        rel = f"{video}/img{i:05d}.jpg"
        r = by.get(rel)
        p = nguon.thu_muc_anh / rel
        s = sha256_file(p) if p.is_file() else "THIEU"
        if r is None or s != r["sha256"]:
            raise LoiDauVao(f"anh mau {rel}: sha256 {s} != bang hash")
        frames.append(FrameRef(index=i, path=p, time_s=(i - 1) * dt, name=p.name))
        hashes.append((rel, s))
    if so_khop_dataset:       # duong that: khop DetracDataset.frames (glob sorted) tren cung thu muc anh
        from mot_datasets.detrac import DetracDataset
        ref = DetracDataset(img_root=nguon.thu_muc_anh).frames(video, "test")[mau["frame_dau"] - 1:mau["frame_cuoi"]]
        if [(f.index, Path(f.path).resolve(), f.time_s) for f in ref] != \
                [(f.index, Path(f.path).resolve(), f.time_s) for f in frames]:
            raise LoiDauVao(f"{video}: khung mau khac DetracDataset.frames")
    return frames, hashes


def tao_cache(gt, nguon, det, model, spec, cache_dir, mau, bang, so_khop_dataset) -> dict:
    """det_cache.build_video_yolo (to xam + detect_frame) -> doi ID sang COCO -> det_cache.save_video."""
    import numpy as np
    import det_cache
    bang_coco = bang_anh_xa_coco(model.names, gt["detector"][det]["loai_checkpoint"])
    det_cache.write_meta(cache_dir, spec, mode="create")
    out = {"anh_xa": None if bang_coco is None else {str(k): v for k, v in sorted(bang_coco.items())},
           "video": {}}
    for v in mau["videos"]:
        frames, hashes = khung_mau(nguon, v, mau, bang[v], so_khop_dataset)
        regions = doc_vung(nguon.vung_bo_qua(v))
        if so_khop_dataset:
            from mot_datasets.detrac import DetracDataset
            import baseline_track
            if DetracDataset(img_root=nguon.thu_muc_anh).ignored_regions(v, "test") != regions or \
                    [tuple(map(float, r)) for r in baseline_track.load_ignored_regions(v, "DETRAC-test")] != regions:
                raise LoiDauVao(f"{v}: vung bo qua khac ham doc hien hanh")
        t0 = time.time()
        arrays = det_cache.build_video_yolo(model, frames, regions, spec, IMG_HW)
        t_inf = time.time() - t0
        if arrays["unreadable"].any() or arrays["truncated"].any():
            raise LoiDauVao(f"{det}/{v}: unreadable {int(arrays['unreadable'].sum())} truncated "
                            f"{int(arrays['truncated'].sum())}")
        goc = arrays["data"]
        moi = doi_id_sang_coco(goc, bang_coco)
        if not (np.array_equal(moi[:, :5], goc[:, :5]) and moi.shape == goc.shape):
            raise LoiDauVao("doi ID lam doi box/conf/thu tu")
        arrays["data"] = moi
        arrays["cls_goc"] = goc[:, 5].astype(np.int16)
        extra = {"r54": True, "anh_xa_coco": out["anh_xa"], "names": {str(k): str(n) for k, n in model.names.items()},
                 "frames": [mau["frame_dau"], mau["frame_cuoi"]], "anh_sha256_ds": sha256_ds(hashes),
                 "vung_bo_qua": [list(r) for r in regions]}
        det_cache.save_video(cache_dir, v, arrays, extra=extra, spec=spec)
        npz = cache_dir / f"{v}.npz"
        out["video"][v] = {"npz_sha256": sha256_file(npz), "json_sha256": sha256_file(cache_dir / f"{v}.json"),
                           "n_frames": int(len(arrays["frame_idx"])), "n_boxes": int(len(moi)),
                           "cls_goc": sorted(set(int(c) for c in goc[:, 5])), "cls_coco": sorted(set(int(c) for c in moi[:, 5])),
                           "anh_sha256_ds": extra["anh_sha256_ds"], "giay_suy_luan": round(t_inf, 3)}
    out["meta_sha256"] = sha256_file(cache_dir / "meta.json")
    return out


def mo_cache(cache_dir: Path, video: str, conf_min: float, classes, max_det: int):
    """Nhu det_cache.open_cache nhung cache_dir tuong minh (cache smoke nam ngoai CACHE_ROOT)."""
    import det_cache
    meta = det_cache.load_meta(cache_dir)
    errs = det_cache.check_compatible(meta, conf_min, classes, max_det)
    if errs:
        raise LoiDauVao(f"cache {cache_dir.name} khong tai lap duoc: {errs}")
    st, why = det_cache.sequence_spec_status(cache_dir, video, meta)
    if st != "ok":
        raise LoiDauVao(f"cache {cache_dir.name}/{video}: {st} {why}")
    return det_cache.VideoDetections(cache_dir / f"{video}.npz", conf_min, classes, max_det, None,
                                     allow_truncated=False)


def chay_tracker(out_root: Path, video: str, vd, cfg, seed, registry, assoc_log: bool, pf_diag: bool) -> dict:
    """run_tracking.track_video + ghi dau ra Y HET run_tracking.run_video_to_dir; them seed thuc dung."""
    import pandas as pd
    import run_tracking as rt
    sink = {"assoc_log": assoc_log, "pf_diag": pf_diag}
    t0 = time.time()
    df, df_cls, df_t = rt.track_video(vd, cfg, 1.0 / FPS, "skip", seed, video, registry, log_sink=sink)
    sec = time.time() - t0
    (out_root / "data").mkdir(parents=True, exist_ok=True)
    (out_root / "data_cls").mkdir(exist_ok=True)
    (out_root / "timing").mkdir(exist_ok=True)
    df.to_csv(out_root / "data" / f"{video}.txt", header=False, index=False)
    df_cls.to_csv(out_root / "data_cls" / f"{video}.csv", index=False)
    df_t.to_csv(out_root / "timing" / f"{video}.csv", index=False)
    if assoc_log:
        from assoc_log import save_log
        save_log(sink["assoc_log"], out_root / "assoc_log", video)
    if pf_diag and isinstance(sink.get("pf_diag"), pd.DataFrame):
        (out_root / "pf_diag").mkdir(exist_ok=True)
        sink["pf_diag"].to_csv(out_root / "pf_diag" / f"{video}.csv", index=False)
    tr = sink["tracker"]
    kf = getattr(tr, "kalman_filter", None)
    info = {"video": video, "n_frames": len(vd), "n_buoc_timing": int(len(df_t)),
            "n_steps": sorted(set(int(x) for x in df_t["n_steps"])) if len(df_t) else [],
            "n_boxes": int(len(df)), "n_tracks": int(df["id"].nunique()) if len(df) else 0,
            "seed_truyen": seed, "seed_thuc_dung": getattr(kf, "seed", None) if seed is not None else None,
            "tracker_lop": type(tr).__name__, "track_buffer": int(cfg.track_buffer),
            "giay": round(sec, 3), "data_sha256": sha256_file(out_root / "data" / f"{video}.txt")}
    if hasattr(kf, "num_particles"):
        info.update({"num_particles": int(kf.num_particles), "k1_optimal_proposal": bool(kf.optimal_proposal),
                     "k2_roughening": float(kf.roughening)})
    return info


def cham(gt_folder: Path, trackers_folder: Path, split: str, run: str, seq_info: dict, out_folder: Path,
         vung: list[Path]) -> dict:
    """TrackEval MotChallenge2DBox + HOTA/CLEAR/Identity theo run_trackeval.build_configs; chi doi
    duong dan vao/ra va SEQ_INFO. Ghi per_video / combined / raw_fields nhu run_experiment.stage_eval."""
    import trackeval
    import run_trackeval as rte
    from boot_stats import save_raw_fields
    eval_cfg, ds_cfg, met_cfg = rte.build_configs(split, run, None)
    for p in (gt_folder, trackers_folder, out_folder):
        kiem_trong_vung(p, vung)
    out_folder.mkdir(parents=True, exist_ok=True)
    ds_cfg.update({"GT_FOLDER": str(gt_folder), "TRACKERS_FOLDER": str(trackers_folder),
                   "SEQMAP_FOLDER": str(gt_folder / "seqmaps"), "SEQ_INFO": dict(seq_info),
                   "OUTPUT_FOLDER": str(out_folder)})
    eval_cfg["LOG_ON_ERROR"] = str(kiem_trong_vung(out_folder / "trackeval_error.log", vung))
    cls_name = rte.dataset_class_for("detrac", split)
    ds_cls = rte.dataset_class(cls_name)
    if cls_name != "mot" or ds_cls is not trackeval.datasets.MotChallenge2DBox:
        raise LoiDauVao(f"evaluator {cls_name}/{ds_cls} khong phai MotChallenge2DBox")
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):      # bang in cua TrackEval -> file log cua run (khong doi cau hinh)
        evaluator = trackeval.Evaluator(eval_cfg)
        dataset = ds_cls(ds_cfg)
        metrics = [trackeval.metrics.HOTA(met_cfg), trackeval.metrics.CLEAR(met_cfg),
                   trackeval.metrics.Identity(met_cfg)]
        results, msg = evaluator.evaluate([dataset], metrics)
    (out_folder / run).mkdir(parents=True, exist_ok=True)
    (out_folder / run / "trackeval_stdout.txt").write_text(buf.getvalue(), encoding="utf-8")
    m = msg.get("MotChallenge2DBox", {}).get(run)
    if m != "Success":
        raise LoiDauVao(f"TrackEval {run}: {m}")
    df = rte.flatten_results(results, run)
    (out_folder / run).mkdir(parents=True, exist_ok=True)
    df[df.video != "COMBINED_SEQ"].sort_values("video").to_csv(out_folder / run / "per_video_metrics.csv", index=False)
    df[df.video == "COMBINED_SEQ"].to_csv(out_folder / run / "combined_metrics.csv", index=False)
    save_raw_fields(results, run, out_folder / run / "raw_fields.json")
    c = df[df.video == "COMBINED_SEQ"].iloc[0].to_dict()
    so = [float(x) for x in df.drop(columns=["video"]).to_numpy().ravel()]
    return {"combined": {k: (float(v) if k != "video" else v) for k, v in c.items()},
            "huu_han": all(math.isfinite(x) for x in so),
            "file": {f: sha256_file(out_folder / run / f) for f in
                     ("per_video_metrics.csv", "combined_metrics.csv", "raw_fields.json")},
            "cau_hinh": {"metric": met_cfg, "classes_to_eval": ds_cfg["CLASSES_TO_EVAL"],
                         "benchmark": ds_cfg["BENCHMARK"], "split_to_eval": ds_cfg["SPLIT_TO_EVAL"],
                         "dataset_class": "MotChallenge2DBox"}}


def tao_gt_smoke(nguon: Nguon, mau: dict, gt_root: Path, vung: list[Path]) -> dict:
    """GT smoke: chi dong frame trong mau (giu nguyen text cot); seqinfo chi doi seqLength."""
    d = kiem_trong_vung(gt_root / SPLIT_SMOKE, vung)
    n = mau["frame_cuoi"] - mau["frame_dau"] + 1
    out = {}
    for v in mau["videos"]:
        if mau["frame_dau"] != 1:
            raise LoiDauVao("mau smoke phai bat dau tu frame 1 (giu nguyen frame ID)")
        src_gt, src_si = nguon.gt_txt(v), nguon.seqinfo(v)
        if not src_gt.is_file() or not src_si.is_file():
            raise LoiDauVao(f"thieu GT/seqinfo {v}")
        lines = src_gt.read_bytes().decode("utf-8").splitlines(keepends=True)
        keep = [ln for ln in lines if ln.strip() and mau["frame_dau"] <= int(ln.split(",", 1)[0]) <= mau["frame_cuoi"]]
        (d / v / "gt").mkdir(parents=True, exist_ok=False)
        (d / v / "gt" / "gt.txt").write_bytes("".join(keep).encode("utf-8"))
        si_lines = src_si.read_bytes().decode("utf-8").splitlines(keepends=True)
        n_sl = sum(1 for ln in si_lines if ln.startswith("seqLength="))
        if n_sl != 1:
            raise LoiDauVao(f"seqinfo {v} co {n_sl} dong seqLength")
        si_moi = [(f"seqLength={n}" + ("\r\n" if ln.endswith("\r\n") else "\n")) if ln.startswith("seqLength=") else ln
                  for ln in si_lines]
        (d / v / "seqinfo.ini").write_bytes("".join(si_moi).encode("utf-8"))
        out[v] = {"so_dong_gt": len(keep), "so_frame_co_gt": len({int(ln.split(',', 1)[0]) for ln in keep}),
                  "gt_goc_sha256": sha256_file(src_gt), "gt_smoke_sha256": sha256_file(d / v / "gt" / "gt.txt"),
                  "seqinfo_smoke_sha256": sha256_file(d / v / "seqinfo.ini"), "seq_length": n}
    return out


def du_toan(tg: dict, gt: dict, mau: dict) -> dict:
    n_mau = len(mau["videos"]) * (mau["frame_cuoi"] - mau["frame_dau"] + 1)
    sp = gt["tap_du_lieu"]["so_anh"] / max(n_mau, 1)
    inf = sum(tg["suy_luan_s"].values())
    trk = sum(tg["tracking_s"].values())
    ev = tg["cham_s"]
    kich_ban = {"suy_luan_3_detector_s": round(inf * sp, 0), "tracking_36_run_s": round(trk * sp, 0),
                "cham_36_run_s_toi_thieu": round(ev, 0)}
    kich_ban["tong_s"] = round(sum(kich_ban.values()), 0)
    return {"loai": "kich ban theo gia dinh, KHONG bao dam", "he_so_quy_doi_anh": round(sp, 2),
            "kich_ban": kich_ban,
            "gia_dinh": ["thoi gian tuyen tinh theo so anh (56340 / so anh mau)",
                         "128 frame dau dai dien cho mat do xe ca video (chua kiem)",
                         "cham: TrackEval tang theo so frame/GT; so lieu o day chi la thoi gian cham tren mau",
                         "khong tinh nap model, hau kiem, kiem khoa, ghi assoc_log tren dia cham",
                         "may, GPU, nhiet do va tai nen giong luc smoke"]}


def chay_smoke(gt: dict, nguon: Nguon, phien: Path, mo_model, chuan_bi_spec, mau: dict | None = None,
               so_khop_dataset: bool = True, log=print) -> dict:
    """Toan bo smoke tren `mau`. `mo_model(det, ckpt)` -> model co .names/.predict; `chuan_bi_spec(spec)`
    dien moi truong (duong that: spec.fill_env()). Moi tracker cua mot detector dung NGUYEN cache do."""
    import pandas as pd
    import run_tracking as rt
    mau = mau or MAU_SMOKE
    vung = [phien]
    tt = {"bat_dau": bay_gio(), "tac_vu": {}}

    def ghi_tt():
        ghi_json(phien / "trang_thai.json", tt, vung, ghi_de=True)

    def xong(k, **kw):
        tt["tac_vu"][k] = {"trang_thai": "xong", **kw}
        ghi_tt()

    bang = doc_bang_hash(nguon.bang_hash)
    gt_root = phien / "gt"
    tg = {"suy_luan_s": {}, "tracking_s": {}, "cham_s": 0.0, "nap_model_s": {}}
    seq_info = {v: mau["frame_cuoi"] - mau["frame_dau"] + 1 for v in mau["videos"]}
    gt_smoke = tao_gt_smoke(nguon, mau, gt_root, vung)
    xong("gt_smoke", chi_tiet=gt_smoke)
    # --- cache ---
    caches = {}
    for det in gt["detector"]:
        ckpt = nguon.checkpoint[det]
        sha = sha256_file(ckpt)
        if sha != gt["detector"][det]["sha256"]:
            raise LoiDauVao(f"checkpoint {det} sha256 {sha} != khoa")
        t0 = time.time()
        model = mo_model(det, ckpt)
        tg["nap_model_s"][det] = round(time.time() - t0, 2)
        spec = tao_spec(gt, det, ckpt, sha)
        chuan_bi_spec(spec)
        cache_dir = kiem_trong_vung(phien / "cache" / det, vung)
        t0 = time.time()
        caches[det] = tao_cache(gt, nguon, det, model, spec, cache_dir, mau, bang, so_khop_dataset)
        tg["suy_luan_s"][det] = round(time.time() - t0, 2)
        caches[det]["spec_sha256"] = spec.spec_sha256()
        caches[det]["versions"] = spec.versions
        del model
        xong(f"cache:{det}", chi_tiet=caches[det])
        log(f"  [cache] {det}: " + ", ".join(f"{v} {c['n_boxes']} box" for v, c in caches[det]["video"].items()))
    # --- tracking + cham ---
    registry = rt.tracker_registry()
    trk_root = phien / "trackers"
    ev_root = phien / "trackeval"
    tr_cfg = gt["tracker"]
    yaml_of = {b["ten"]: b["yaml"] for b in tr_cfg["bo_loc"]}
    runs = {}
    for r in ma_tran(gt):
        cfg = rt.load_tracker_cfg(yaml_arg(r["yaml"]), {"track_buffer": TRACK_BUFFER_BUOC})
        out_root = kiem_trong_vung(trk_root / SPLIT_SMOKE / r["run"], vung)
        if out_root.exists():
            raise LoiGhiDe(f"{out_root} da ton tai")
        t0 = time.time()
        infos = []
        for v in mau["videos"]:
            vd = mo_cache(phien / "cache" / r["detector"], v, r["conf_min"], gt["tai"]["classes"], gt["tai"]["max_det"])
            infos.append(chay_tracker(out_root, v, vd, cfg, r["seed"], registry, tr_cfg["assoc_log"], tr_cfg["pf_diag"]))
        tg["tracking_s"][r["run"]] = round(time.time() - t0, 3)
        ghi_json(out_root / "run_meta_r54.json",
                 {**r, "tracker_cfg": rt.cfg_to_dict(cfg), "conf_so_sanh": ">", "classes": gt["tai"]["classes"],
                  "max_det": gt["tai"]["max_det"], "elapsed_mode": "skip", "fps": FPS, "cache": r["detector"],
                  "cache_npz_sha256": {v: caches[r["detector"]]["video"][v]["npz_sha256"] for v in mau["videos"]},
                  "video": infos, "hoan_tat": True, "luc": bay_gio()}, vung)
        t0 = time.time()
        ev = cham(gt_root, trk_root, SPLIT_SMOKE, r["run"], seq_info, ev_root, vung)
        tg["cham_s"] += time.time() - t0
        runs[r["run"]] = {**r, "video": infos, "cham": ev}
        xong(f"run:{r['run']}", hota=ev["combined"].get("HOTA"), huu_han=ev["huu_han"])
    # --- lap PF tren cache co san (khong suy luan lai) ---
    lap = {}
    v0 = mau["videos"][0]
    for det in gt["detector"]:
        cfg = rt.load_tracker_cfg(yaml_arg(yaml_of["pf500"]), {"track_buffer": TRACK_BUFFER_BUOC})
        for s in SEEDS_PF:
            run = ten_run(det, "c25", "pf500", s)
            out_root = kiem_trong_vung(phien / "lap_pf" / det / run, vung)
            vd = mo_cache(phien / "cache" / det, v0, DIEU_KIEN["c25"], gt["tai"]["classes"], gt["tai"]["max_det"])
            info = chay_tracker(out_root, v0, vd, cfg, s, registry, tr_cfg["assoc_log"], tr_cfg["pf_diag"])
            goc = (trk_root / SPLIT_SMOKE / run / "data" / f"{v0}.txt").read_bytes()
            moi = (out_root / "data" / f"{v0}.txt").read_bytes()
            lap[run] = {"detector": det, "video": v0, "trung_byte": goc == moi, "sha256_goc": sha256_bytes(goc),
                        "sha256_lap": sha256_bytes(moi), "seed_thuc_dung": info["seed_thuc_dung"]}
    xong("lap_pf", chi_tiet=lap)
    # --- cham lai tu MOT co san ---
    cham_lai = {}
    for det in gt["detector"]:
        for loc in ("cv", "ekf", "ukf", "pf500-s0"):
            run = f"r54-{det}-c25-{loc}"
            t0 = time.time()
            ev2 = cham(gt_root, trk_root, SPLIT_SMOKE, run, seq_info, phien / "cham_lai", vung)
            tg["cham_s"] += time.time() - t0
            ev1 = runs[run]["cham"]
            cham_lai[run] = {"file_trung_byte": ev1["file"] == ev2["file"], "combined_trung": ev1["combined"] == ev2["combined"],
                             "file": ev2["file"]}
    xong("cham_lai", chi_tiet=cham_lai)
    tg["cham_s"] = round(tg["cham_s"], 2)
    # --- tom tat ---
    loi = []
    for run, x in runs.items():
        for i in x["video"]:
            if i["n_frames"] != seq_info[i["video"]] or i["n_buoc_timing"] != seq_info[i["video"]] or i["n_steps"] != [1]:
                loi.append(f"{run}/{i['video']}: so frame/buoc {i['n_frames']}/{i['n_buoc_timing']}/{i['n_steps']}")
            if i["track_buffer"] != TRACK_BUFFER_BUOC:
                loi.append(f"{run}: track_buffer {i['track_buffer']}")
            if x["bo_loc"] == "pf500" and (i.get("seed_thuc_dung") != x["seed"] or i.get("num_particles") != 500
                                           or i.get("k1_optimal_proposal") or i.get("k2_roughening") != 0.0):
                loi.append(f"{run}: seed/N/K1/K2 thuc dung lech")
        if not x["cham"]["huu_han"]:
            loi.append(f"{run}: metric khong huu han")
    loi += [f"lap {k}: khong trung byte" for k, x in lap.items() if not x["trung_byte"]]
    loi += [f"cham lai {k}: lech" for k, x in cham_lai.items() if not (x["file_trung_byte"] and x["combined_trung"])]
    for det, c in caches.items():
        for v, cv in c["video"].items():
            if sha256_file(phien / "cache" / det / f"{v}.npz") != cv["npz_sha256"]:
                loi.append(f"cache {det}/{v} doi sau tracking")
    pd.DataFrame([{"run": k, "detector": x["detector"], "cond": x["cond"], "bo_loc": x["bo_loc"], "seed": x["seed"],
                   **{m: x["cham"]["combined"].get(m) for m in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW")}}
                  for k, x in runs.items()]).to_csv(phien / "tong_hop_smoke.csv", index=False)
    tt["ket_thuc"] = bay_gio()
    ghi_tt()
    return {"so_run": len(runs), "so_luot_video": sum(len(x["video"]) for x in runs.values()), "cache": caches,
            "runs": {k: {"video": x["video"], "cham": x["cham"]} for k, x in runs.items()}, "lap_pf": lap,
            "cham_lai": cham_lai, "thoi_gian": tg, "du_toan": du_toan(tg, gt, mau), "gt_smoke": gt_smoke,
            "loi": loi, "dat": not loi}


def mo_model_that(det: str, ckpt: Path):
    from ultralytics import YOLO
    return YOLO(str(ckpt))


def kiem_module_da_khoa(khoa: dict) -> list[str]:
    """Module dang nap (src + ultralytics + trackeval) khong co trong khoa."""
    da_khoa = {tuyet_doi(f["rel"]) for f in khoa["file"]}
    return sorted(rel_goc(p) for p in file_module_da_nap() if p not in da_khoa)


def kiem_truoc_suy_luan(gt: dict, khoa: dict, nguon: Nguon, thu_muc_predictor: Path) -> dict:
    """R54A-F2: sau kiem-khoa dat va TRUOC suy luan: backend (ke ca BYTETRACK_BACKEND ke thua), moi file ma
    thuc nap (src/Ultralytics/TrackEval, ke ca module DetectionPredictor.__init__ nap luoi) nam trong khoa va bam
    khop, cau hinh tracker resolve va phien ban thu vien khop khoa. Khong suy luan; chi nap module + lop nn cua
    checkpoint tren CPU + khoi tao predictor (khong setup_model/predict)."""
    lech = []
    try:
        nap, pred = phu_thuoc_thuc_dung(gt, nguon, thu_muc_predictor)
        bk = thong_tin_backend(gt)
    except LoiR54 as ex:
        return {"dat": False, "lech": [f"{type(ex).__name__}: {ex}"], "luc": bay_gio()}
    da_khoa = {tuyet_doi(f["rel"]): f["sha256"] for f in khoa["file"]}
    ngoai = [rel_goc(p) for p in nap if p not in da_khoa]
    if ngoai:
        lech.append(f"module nap ngoai khoa ({len(ngoai)}): {ngoai[:10]}")
    lech += [f"{rel_goc(p)}: sha256 khac khoa" for p in nap if p in da_khoa and sha256_file(p) != da_khoa[p]]
    if chuan_json(bk) != chuan_json(khoa.get("backend_tracker")):
        lech.append(f"backend tracker {bk} khac khoa {khoa.get('backend_tracker')}")
    cfg, loi = cau_hinh_hieu_luc(gt)
    lech += loi
    if chuan_json(cfg) != chuan_json(khoa.get("cau_hinh_hieu_luc")):
        lech.append("cau hinh tracker hieu luc (YAML resolve/override) khac khoa")
    tv = phien_ban_thu_vien()
    if tv != khoa.get("thu_vien"):
        lech.append(f"phien ban thu vien khac khoa: {tv}")
    return {"dat": not lech, "lech": lech, "so_module_nap": len(nap), "backend": bk["backend"], "predictor": pred,
            "luc": bay_gio()}


def buoc_smoke(gt: dict, vung: dict, args) -> int:
    k1 = kiem_khoa(vung, args.khoa)
    if not k1["dat"]:
        raise LoiDauVao(f"kiem-khoa truoc smoke lech: {k1['lech'][:5]}")
    khoa = k1["_khoa"]
    if tuple(khoa["mau_smoke"]["videos"]) != MAU_SMOKE["videos"] or \
            (khoa["mau_smoke"]["frame_dau"], khoa["mau_smoke"]["frame_cuoi"]) != (MAU_SMOKE["frame_dau"], MAU_SMOKE["frame_cuoi"]):
        raise LoiDauVao("mau smoke trong khoa khac MAU_SMOKE")
    phien = tao_phien(vung["smoke"], args.lan, "lan")
    dat_moi_truong(phien)
    t0 = time.time()
    nguon = Nguon(gt)
    tr = kiem_truoc_suy_luan(gt, khoa, nguon, phien)
    if not tr["dat"] or os.environ.get(ENV_TEST):
        # Lech -> dung, exit 2. Che do fixture (ENV_TEST) KHONG BAO GIO suy luan: dung o day, exit 5.
        ma = 2 if not tr["dat"] else 5
        ghi_json(phien / "ket_qua_smoke.json", {"buoc": "smoke", "phien": rel_goc(phien), "khoa": k1["khoa"],
                                                "khoa_sha256": k1["khoa_sha256"], "kiem_truoc_suy_luan": tr,
                                                "dung_truoc_suy_luan": True, "dat": False, "exit": ma}, [phien])
        print(f"[smoke] dung truoc suy luan ({'lech' if ma == 2 else 'che do fixture'}) -> exit {ma}",
              file=sys.stderr if ma == 2 else sys.stdout)
        for x in tr["lech"][:20]:
            print("   -", x, file=sys.stderr)
        return ma
    kq = chay_smoke(gt, nguon, phien, mo_model_that, lambda s: s.fill_env())
    lech_mod = kiem_module_da_khoa(khoa)
    k2 = kiem_khoa(vung, Path(k1["khoa"]).name)
    if lech_mod:
        kq["loi"].append(f"module nap ngoai khoa: {lech_mod}")
    if not k2["dat"]:
        kq["loi"].append(f"kiem-khoa sau smoke lech: {k2['lech'][:5]}")
    kq["dat"] = not kq["loi"]
    ma = 0 if kq["dat"] else 4
    kq.update({"buoc": "smoke", "phien": rel_goc(phien), "khoa": k1["khoa"], "khoa_sha256": k1["khoa_sha256"],
               "kiem_khoa_truoc": {"dat": k1["dat"], "luc": k1["luc"]}, "kiem_truoc_suy_luan": tr,
               "kiem_khoa_sau": {"dat": k2["dat"], "luc": k2["luc"]},
               "tong_giay": round(time.time() - t0, 1), "exit": ma})
    ghi_json(phien / "ket_qua_smoke.json", kq, [phien])
    print(f"[smoke] {kq['so_run']} run, {kq['so_luot_video']} luot video, {len(kq['loi'])} loi -> exit {ma}")
    return ma


def hau_kiem(phien_smoke: Path, gt: dict, mau: dict | None = None) -> list[str]:
    import pandas as pd
    mau = mau or MAU_SMOKE
    n_mau = mau["frame_cuoi"] - mau["frame_dau"] + 1
    loi = []
    kq = json.loads((phien_smoke / "ket_qua_smoke.json").read_text(encoding="utf-8"))
    if kq.get("exit") != 0:
        loi.append(f"smoke exit {kq.get('exit')}")
    if kq.get("dung_truoc_suy_luan") or "cache" not in kq:
        return loi + ["phien smoke dung truoc suy luan / khong co ket qua cache"]
    tt = json.loads((phien_smoke / "trang_thai.json").read_text(encoding="utf-8"))
    can = {"gt_smoke", "lap_pf", "cham_lai", *(f"cache:{d}" for d in gt["detector"]), *(f"run:{r['run']}" for r in ma_tran(gt))}
    thieu = sorted(can - {k for k, x in tt["tac_vu"].items() if x.get("trang_thai") == "xong"})
    if thieu:
        loi.append(f"tac vu chua xong: {thieu[:5]}")
    def khop(p: Path, want, nhan: str):
        """R54A-F5: file phai ton tai va bam lai khop hash da ghi trong ket qua smoke."""
        if not p.is_file():
            loi.append(f"{nhan}: thieu {rel_goc(p)}")
        elif not want or sha256_file(p) != want:
            loi.append(f"{nhan}: sha256 lech ({rel_goc(p)})")

    if sorted(kq.get("cache", {})) != sorted(gt["detector"]):
        loi.append(f"cache: detector {sorted(kq.get('cache', {}))}")
    for v, g in kq.get("gt_smoke", {}).items():
        khop(phien_smoke / "gt" / SPLIT_SMOKE / v / "gt" / "gt.txt", g.get("gt_smoke_sha256"), f"gt smoke {v}")
        khop(phien_smoke / "gt" / SPLIT_SMOKE / v / "seqinfo.ini", g.get("seqinfo_smoke_sha256"), f"seqinfo smoke {v}")
    if sorted(kq.get("gt_smoke", {})) != sorted(mau["videos"]):
        loi.append("gt smoke khong du video mau")
    for det, c in kq["cache"].items():
        khop(phien_smoke / "cache" / det / "meta.json", c.get("meta_sha256"), f"cache {det} meta.json")
        if sorted(c["video"]) != sorted(mau["videos"]):
            loi.append(f"cache {det}: video {sorted(c['video'])}")
        for v, cv in c["video"].items():
            p = phien_smoke / "cache" / det / f"{v}.npz"
            if not p.is_file() or sha256_file(p) != cv["npz_sha256"]:
                loi.append(f"cache {det}/{v} sha256 lech")
            khop(phien_smoke / "cache" / det / f"{v}.json", cv.get("json_sha256"), f"cache {det}/{v}.json")
            if cv["n_frames"] != n_mau:
                loi.append(f"cache {det}/{v}: {cv['n_frames']} frame")
            if gt["detector"][det]["loai_checkpoint"] == "lop5" and c["anh_xa"] != {"0": 2, "1": 7, "2": 5, "3": 3, "4": 1}:
                loi.append(f"anh xa {det} lech: {c['anh_xa']}")
            if gt["detector"][det]["loai_checkpoint"] == "coco80" and c["anh_xa"] is not None:
                loi.append(f"anh xa {det} phai None")
    for r in ma_tran(gt):
        d = phien_smoke / "trackers" / SPLIT_SMOKE / r["run"]
        mj = d / "run_meta_r54.json"
        if not mj.is_file():
            loi.append(f"{r['run']}: thieu run_meta_r54.json")
            continue
        m = json.loads(mj.read_text(encoding="utf-8"))
        for i in m["video"]:
            t = pd.read_csv(d / "timing" / f"{i['video']}.csv")
            if len(t) != n_mau or set(t["n_steps"]) != {1}:
                loi.append(f"{r['run']}/{i['video']}: timing {len(t)} dong, n_steps {sorted(set(t['n_steps']))}")
            if sha256_file(d / "data" / f"{i['video']}.txt") != i["data_sha256"]:
                loi.append(f"{r['run']}/{i['video']}: MOT doi sau smoke")
            if r["seed"] is not None and i.get("seed_thuc_dung") != r["seed"]:
                loi.append(f"{r['run']}: seed thuc dung {i.get('seed_thuc_dung')}")
        if len(m["video"]) != len(mau["videos"]):
            loi.append(f"{r['run']}: {len(m['video'])} video")
        rk = kq.get("runs", {}).get(r["run"])
        if rk is None:
            loi.append(f"{r['run']}: thieu trong ket_qua_smoke.json")
            continue
        for f, s in rk["cham"]["file"].items():
            khop(phien_smoke / "trackeval" / r["run"] / f, s, f"{r['run']} {f}")
        if sorted(rk["cham"]["file"]) != ["combined_metrics.csv", "per_video_metrics.csv", "raw_fields.json"]:
            loi.append(f"{r['run']}: danh sach file metric {sorted(rk['cham']['file'])}")
        e = phien_smoke / "trackeval" / r["run"] / "combined_metrics.csv"
        if not e.is_file():
            loi.append(f"{r['run']}: thieu combined_metrics.csv")
        else:
            vals = pd.read_csv(e).drop(columns=["video"]).iloc[0].tolist()
            if not all(math.isfinite(float(x)) for x in vals):
                loi.append(f"{r['run']}: metric khong huu han")
    if not all(x["trung_byte"] for x in kq["lap_pf"].values()) or len(kq["lap_pf"]) != 9:
        loi.append("lap PF khong du 9 / khong trung byte")
    for run, x in kq["lap_pf"].items():
        khop(phien_smoke / "lap_pf" / x["detector"] / run / "data" / f"{x['video']}.txt", x["sha256_lap"], f"lap {run}")
        khop(phien_smoke / "trackers" / SPLIT_SMOKE / run / "data" / f"{x['video']}.txt", x["sha256_goc"], f"goc lap {run}")
        if x["sha256_lap"] != x["sha256_goc"]:
            loi.append(f"lap {run}: hash lap != goc")
    if not all(x["file_trung_byte"] for x in kq["cham_lai"].values()) or len(kq["cham_lai"]) != 12:
        loi.append("cham lai khong du 12 / lech")
    for run, x in kq["cham_lai"].items():
        for f, s in x["file"].items():
            khop(phien_smoke / "cham_lai" / run / f, s, f"cham lai {run} {f}")
            if s != kq.get("runs", {}).get(run, {}).get("cham", {}).get("file", {}).get(f):
                loi.append(f"cham lai {run} {f}: hash khac lan cham dau")
    return loi


def buoc_hau_kiem(gt: dict, vung: dict, args) -> int:
    s = (vung["smoke"] / args.smoke) if args.smoke else phien_moi_nhat(vung["smoke"], "lan_", "ket_qua_smoke.json")
    if s is None or not (s / "ket_qua_smoke.json").is_file():
        raise LoiDauVao("chua co phien smoke hoan tat")
    kiem_trong_vung(s, [vung["smoke"]])
    k = kiem_khoa(vung, args.khoa)
    loi = hau_kiem(s, gt) + ([] if k["dat"] else [f"kiem-khoa lech: {k['lech'][:5]}"])
    kq_s = json.loads((s / "ket_qua_smoke.json").read_text(encoding="utf-8"))
    if kq_s.get("khoa_sha256") != k["khoa_sha256"]:
        loi.append("smoke chay voi khoa khac khoa dang kiem")
    if not (kq_s.get("kiem_truoc_suy_luan") or {}).get("dat"):
        loi.append("smoke khong co kiem truoc suy luan dat")
    phien = tao_phien(vung["smoke"], args.lan, "hau-kiem")
    ma = 0 if not loi else 4
    ghi_json(phien / "hau_kiem.json", {"buoc": "hau-kiem", "smoke": rel_goc(s), "khoa": k["khoa"],
                                       "khoa_sha256": k["khoa_sha256"], "loi": loi, "luc": bay_gio(), "exit": ma}, [phien])
    print(f"[hau-kiem] {rel_goc(s)}: {len(loi)} loi -> exit {ma}")
    for x in loi[:20]:
        print("   -", x)
    return ma


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R54: khoa giao thuc tracking test40 + smoke (khong co buoc full)")
    ap.add_argument("--buoc", required=True, choices=BUOC)
    ap.add_argument("--lan", default=None, help="ten phien moi (mac dinh <buoc>_<thoi diem>_pid<pid>)")
    ap.add_argument("--khoa", default=None, help="phien khoa (mac dinh moi nhat) cho kiem-khoa/smoke/hau-kiem")
    ap.add_argument("--tien-kiem", default=None, help="phien tien-kiem cho buoc khoa (mac dinh moi nhat)")
    ap.add_argument("--smoke", default=None, help="phien smoke cho hau-kiem (mac dinh moi nhat)")
    ap.add_argument("--luu", default=None, choices=("chuan_bi", "smoke"),
                    help="chi kiem-khoa: luu bang chung vao phien moi cua vung nay (mac dinh chi doc, khong ghi)")
    args = ap.parse_args(argv)
    if args.luu and args.buoc != "kiem-khoa":
        print("[LOI] --luu chi dung voi --buoc kiem-khoa", file=sys.stderr)
        return 2
    for x in (args.khoa, args.tien_kiem, args.smoke):
        if x is not None and (not TEN_HOP_LE.match(x) or ".." in x):
            print(f"[LOI] ten phien {x!r} khong hop le", file=sys.stderr)
            return 2
    try:
        gt = doc_giao_thuc()
        e = kiem_giao_thuc(gt)
        if e:
            raise LoiDauVao(f"giao thuc lech pham vi: {e}")
        vung = vung_ghi()
        return {"tien-kiem": buoc_tien_kiem, "khoa": buoc_khoa, "kiem-khoa": buoc_kiem_khoa,
                "smoke": buoc_smoke, "hau-kiem": buoc_hau_kiem}[args.buoc](gt, vung, args)
    except LoiR54 as ex:
        print(f"[LOI] {type(ex).__name__}: {ex}", file=sys.stderr)
        return ex.ma


if __name__ == "__main__":
    raise SystemExit(main())
