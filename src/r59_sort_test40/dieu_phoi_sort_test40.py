"""
dieu_phoi_sort_test40.py -- R59: doi chung association + vong doi kieu SORT (bo loc CV/EKF cua ByteTrack, max_age 30)
voi ByteTrack R56 tren UA-DETRAC test40, CHI c25. Giao thuc: configs/phase2/r59_sort_test40/giao_thuc.json.

Chi la lop DIEU PHOI: doc cache (W.mo_cache -> det_cache.VideoDetections), chay tracker (W.chay_tracker ->
run_tracking.track_video), cham (W.cham -> run_trackeval + TrackEval) deu goi helper R54 da review
(src/r54_tracking/chuan_bi_test40.py) va marker / lan thu cua R55 (src/r55_tracking/dieu_phoi_test40.py); KHONG sua
helper, KHONG goi detector, KHONG tao / xac nhan cache moi, KHONG ghi vao phien R55/R56.

CONG A (ghi results/phase2/r59_sort_test40/{chuan_bi,smoke,kiem_duong_cham}/):
  --buoc chuan-bi                                     nguon R56 (khoa, phien, nguon.json, 6 lan cham doi chung, MOT 6x40,
                                                      metric, cache 3 detector), baseline ma/backend + config.py (byte tho /
                                                      LF), cau hinh SORT hieu luc + bo loc tu instance, K2 parity 3x40,
                                                      timeline / D4. Lech -> exit 3, khong marker.
  --buoc khoa --chuan-bi <ID> --tests <ID>            khoa R59 (ten TUONG MINH)
  --buoc kiem-khoa --khoa <ID>                        bam lai MOI file khoa + module nap / backend / cau hinh
  --buoc smoke --khoa <ID>                            K7: 2 SORT x MVI_39031, MVI_39051 x frame 1-128 tu cache R56 yolov8n_coco
  --buoc cham-nguon --khoa <ID> --run r54-yolov8n_coco-c25-cv   cham lai MOT R56 da bam, 3 file metric trung byte
  --buoc dong-goi-a --khoa <ID> --smoke <ID> --cham-nguon <ID>  goi bang chung cong A
CONG B (chi sau review A approve; ghi results/phase2/r59_sort_test40/{thuc_thi,tong_hop}/):
  --buoc mo-phien --khoa <ID_A> --review-a <json> [--goi-a <ID>]
  --buoc full --phien <ID> [--gioi-han-giay 2400]     6 run x 40 video tuan tu; het gioi han giua don vi -> exit 10
  --buoc cham --phien <ID> --run <run>
  --buoc lap-tracker --phien <ID> --run <run> --video MVI_39031
  --buoc cham-lai --phien <ID> --run <run>
  --buoc hau-kiem --phien <ID>
  --buoc tong-hop --phien <ID>
Khong co mac dinh "chay tat ca". Moi lenh ghi cli_log.json (argv, stdout, stderr, exit) vao thu muc ban ghi cua no.
Marker HOAN_TAT.json ghi CUOI (rang buoc + sha256 moi file); loi -> LOI.json, khong marker; khong ghi de.

FIXTURE: R59_FIXTURE=<.../fixture.json> NAM TRONG results/phase2/r59_sort_test40/tests/ -> moi vung ghi va nguon du
lieu doi vao thu muc fixture; R59_NGAT=<diem>:<k> (chi fixture) gia lap ngat (os._exit 9).

    C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r59_sort_test40/dieu_phoi_sort_test40.py --buoc chuan-bi
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

GOC = Path(__file__).resolve().parents[2]
SRC = GOC / "src"
for _p in (SRC / "r54_tracking", SRC / "r55_tracking"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import chuan_bi_test40 as W  # noqa: E402  (R54 da review; chi goi, khong sua)
import dieu_phoi_test40 as R55  # noqa: E402  (R55/R56 da review; chi goi, khong sua)

VUNG = GOC / "results" / "phase2" / "r59_sort_test40"
GIAO_THUC = GOC / "configs" / "phase2" / "r59_sort_test40" / "giao_thuc.json"
FILE_MA = Path(__file__).resolve()
FILE_TEST = FILE_MA.parent / "test_dieu_phoi_sort_test40.py"
ENV_FIXTURE = "R59_FIXTURE"
ENV_NGAT = "R59_NGAT"
MARKER = "HOAN_TAT.json"
DETECTORS = ("yolov8n_coco", "yolov8n_mixed5", "yolo11n_mixed5")
BO_LOC = (("sort-cv30", "sort_cv", "cv", "configs/phase2/trackers/sort_cv_age30.yaml"),
          ("sort-ekf30", "sort_ekf_ctrv", "ekf", "configs/phase2/trackers/sort_ekf_ctrv_age30.yaml"))
RUNS = tuple(f"r59-{d}-c25-{b[0]}" for d in DETECTORS for b in BO_LOC)
DOI_CHUNG = {f"r59-{d}-c25-{b[0]}": f"r54-{d}-c25-{b[2]}" for d in DETECTORS for b in BO_LOC}
THAM_SO_SORT = {"det_thresh": 0.25, "sort_max_age": 30, "sort_iou_threshold": 0.3, "sort_min_hits": 3}
SMOKE = {"videos": ["MVI_39031", "MVI_39051"], "frame_dau": 1, "frame_cuoi": 128, "split": "DETRAC-smoke128"}
CHAM_NGUON_RUN = "r54-yolov8n_coco-c25-cv"
LAP_VIDEO = "MVI_39031"
LAP_RUNS = ("r59-yolov8n_coco-c25-sort-cv30", "r59-yolov8n_coco-c25-sort-ekf30")
METRIC_THU_TU = ("HOTA", "AssA", "DetA", "IDF1", "IDSW", "MOTA", "FP")
METRIC_TY_LE = ("HOTA", "AssA", "DetA", "IDF1", "MOTA")
METRIC_FILE = ("per_video_metrics.csv", "combined_metrics.csv", "raw_fields.json")
#: Tap module nap TRUOC moi khoa / cong viec: dung tap R55 (MODULE_THUC_THI, da chung minh du cho tracking + cham o R56)
#: + sort_tracker. Module detector chi duoc IMPORT (khong mo checkpoint, khong suy luan).
MODULE_NAP = R55.MODULE_THUC_THI + ("sort_tracker",)
MA_THOAT_CHUA_XONG = 10
R51_CONFIG_PY_LF = "0e3f8b7584b8f90d5c121cf978d8fd74553195e13f6135b3d9b79362f6ff4ad0"

LoiDauVao, LoiGhiDe = W.LoiDauVao, W.LoiGhiDe


class LoiChay(W.LoiR54):
    ma = 4


class LoiNguon(W.LoiR54):
    ma = 3


class ChuaXong(W.LoiR54):
    ma = MA_THOAT_CHUA_XONG


sha = W.sha256_file
rel = W.rel_goc
bay_gio = W.bay_gio


def sha_json(x) -> str:
    return W.sha256_bytes(json.dumps(W.chuan_json(x), sort_keys=True).encode())


# ---------------------------------------------------------------------------
# Ngu canh
# ---------------------------------------------------------------------------
def kiem_giao_thuc(gt: dict, fixture: bool) -> list[str]:
    e = []
    mt = gt.get("ma_tran", {})
    runs = [r.get("run") for r in mt.get("runs", [])]
    if runs != list(RUNS) or mt.get("so_run") != 6 or mt.get("so_video_moi_run") != 40:
        e.append(f"ma_tran.runs {runs} != {list(RUNS)} / so_run 6 / 40 video")
    for r in mt.get("runs", []):
        b = next((x for x in BO_LOC if x[0] == r.get("bo_loc")), None)
        if b is None or r.get("run") != f"r59-{r.get('detector')}-c25-{b[0]}" or r.get("doi_chung") != DOI_CHUNG.get(r.get("run")):
            e.append(f"run {r.get('run')}: detector/bo_loc/doi_chung lech")
    if tuple(gt.get("detector", ())) != DETECTORS:
        e.append("detector khac 3 detector R56")
    t = gt.get("tai", {})
    if (t.get("dieu_kien"), t.get("conf_min"), t.get("classes"), t.get("max_det")) != ("c25", 0.25, [2, 3, 5, 7], 300):
        e.append("tai phai c25 / conf_min 0.25 / classes [2,3,5,7] / max_det 300")
    tr = gt.get("tracker", {})
    if (tr.get("backend"), tr.get("fps"), tr.get("elapsed_mode"), tr.get("override"), tr.get("assoc_log"), tr.get("pf_diag")) != (
            "ultralytics", 25.0, "skip", {"track_buffer": 30}, False, True):
        e.append("tracker phai ultralytics / fps 25 / skip / override track_buffer 30 / assoc_log false / pf_diag true")
    if tr.get("tham_so_sort") != THAM_SO_SORT:
        e.append(f"tham_so_sort != {THAM_SO_SORT}")
    if [(b.get("ten"), b.get("tracker_type"), b.get("doi_chung"), b.get("yaml")) for b in tr.get("bo_loc", [])] != list(BO_LOC):
        e.append("bo_loc khac hai YAML SORT age30")
    sm = gt.get("smoke", {})
    if ([sm.get("videos"), sm.get("frame_dau"), sm.get("frame_cuoi"), sm.get("split")] != list(SMOKE.values())
            or sm.get("detector") != "yolov8n_coco" or sm.get("runs") != list(LAP_RUNS)):
        e.append("smoke khac 2 video x frame 1-128 / DETRAC-smoke128 / yolov8n_coco / 2 SORT")
    if gt.get("cham_nguon", {}).get("run") != CHAM_NGUON_RUN:
        e.append("cham_nguon.run khac r54-yolov8n_coco-c25-cv")
    tl = gt.get("tai_lap", {})
    if (tl.get("video"), tl.get("runs")) != (LAP_VIDEO, list(LAP_RUNS)):
        e.append("tai_lap khac MVI_39031 x 2 SORT yolov8n_coco")
    td = gt.get("tap_du_lieu", {})
    if not fixture and (td.get("so_video"), td.get("so_anh"), td.get("so_anh_D4")) != (40, 56340, 173):
        e.append("tap_du_lieu phai 40 video / 56340 anh / 173 D4")
    m = gt.get("metric", {}).get("doi_chieu_nguon_r56", {}).get("dung_sai", {})
    if any(m.get(k) != 0 for k in ("IDSW", "FP")) or set(m) != set(METRIC_THU_TU):
        e.append("dung_sai phai du 7 metric, IDSW/FP = 0")
    return e


class Ctx:
    def __init__(self):
        t = os.environ.get(ENV_FIXTURE)
        self.fixture = bool(t)
        if os.environ.get(ENV_NGAT) and not self.fixture:
            raise LoiDauVao(f"{ENV_NGAT} chi dung trong fixture")
        if self.fixture:
            fx = Path(os.path.abspath(t))
            tests = VUNG / "tests"
            if fx.name != "fixture.json" or not R55.trong_lexical(fx, tests) or fx.parent == tests or not fx.is_file():
                raise LoiDauVao(f"{ENV_FIXTURE}={t} phai la .../fixture.json ben trong {tests}")
            R55.kiem_doc_lap(fx, tests)
            cfg = R55.doc_json(fx, "fixture")
            self.goc = fx.parent
            self.gt_path = W.kiem_trong_vung(self.goc / cfg["giao_thuc"], [self.goc])
            self.goc_nguon = W.kiem_trong_vung(self.goc / cfg["goc_nguon"], [self.goc])
        else:
            self.goc, self.gt_path, self.goc_nguon = VUNG, GIAO_THUC, GOC
        self.gt = R55.doc_json(self.gt_path, "giao thuc R59")
        loi = kiem_giao_thuc(self.gt, self.fixture)
        if loi:
            raise LoiDauVao(f"giao thuc R59 lech: {loi[:6]}")
        self.runs = {r["run"]: r for r in self.gt["ma_tran"]["runs"]}
        self.bo_loc = {b["ten"]: b for b in self.gt["tracker"]["bo_loc"]}
        n, td = self.gt["nguon_r56"], self.gt["tap_du_lieu"]
        self.phien56, self.khoa56 = self.p(n["phien"]), self.p(n["khoa"])
        self.nguon_json, self.bang56 = self.p(n["nguon_json"]), self.p(n["bang_r56"])
        self.thu_muc_gt, self.manifest, self.loai_d4 = self.p(td["thu_muc_gt"]), self.p(td["manifest"]), self.p(td["loai_d4"])
        self.split = self.thu_muc_gt.name
        self._videos = None

    def p(self, r: str) -> Path:
        return self.goc_nguon / r

    def vung(self, k: str) -> Path:
        return R55.kiem_doc_lap(self.goc / k, self.goc)

    @property
    def videos(self) -> list[str]:
        if self._videos is None:
            v = R55.doc_json(self.manifest, "manifest")["videos"]
            if len(v) != self.gt["tap_du_lieu"]["so_video"] or len(set(v)) != len(v):
                raise LoiDauVao(f"manifest co {len(v)} video != {self.gt['tap_du_lieu']['so_video']}")
            self._videos = list(v)
        return self._videos

    def gt_txt(self, v): return self.thu_muc_gt / v / "gt" / "gt.txt"
    def seqinfo(self, v): return self.thu_muc_gt / v / "seqinfo.ini"


# ---------------------------------------------------------------------------
# Marker / ngat / ban ghi
# ---------------------------------------------------------------------------
_DEM_NGAT: dict[str, int] = {}


def ngat(ctx: Ctx, diem: str) -> bool:
    v = os.environ.get(ENV_NGAT)
    if not ctx.fixture or not v:
        return False
    _DEM_NGAT[diem] = _DEM_NGAT.get(diem, 0) + 1
    if v != f"{diem}:{_DEM_NGAT[diem]}":
        return False
    if diem.endswith("-marker-cat"):
        return True
    print(f"[ngat gia lap] {diem}:{_DEM_NGAT[diem]} -> exit 9", flush=True)
    os._exit(9)


def file_trong(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.is_file() and p.name not in (MARKER, "LOI.json")
                  and not p.name.startswith("cli_log") and "_cau_hinh_thu_vien" not in p.parts)


def ghi_marker(ctx: Ctx, diem: str, d: Path, rang_buoc: dict, them: dict) -> str:
    m = {"rang_buoc": rang_buoc, "file": {p.relative_to(d).as_posix(): sha(p) for p in file_trong(d)}, "luc": bay_gio(),
         **them}
    if ngat(ctx, f"{diem}-marker-cat"):
        b = (json.dumps(m, ensure_ascii=False, indent=1, default=str) + "\n").encode("utf-8")
        W.kiem_trong_vung(d / MARKER, [d]).write_bytes(b[: len(b) // 2])
        print(f"[ngat gia lap] {diem}-marker-cat -> exit 9", flush=True)
        os._exit(9)
    return W.ghi_json(d / MARKER, m, [d])


def lan_hop_le(d: Path, loai: str | None = None) -> dict:
    """Marker doc duoc + moi file khop (theo rang buoc cua chinh no). Loi -> LoiNguon."""
    m = R55.doc_marker(d)
    if m is None:
        raise LoiNguon(f"{rel(d)}: khong co marker hop le (thieu / hong)")
    tt, ly_do, _ = R55.danh_gia_lan(d, m["rang_buoc"])
    if tt != "hop_le":
        raise LoiNguon(f"{rel(d)}: {tt} {ly_do}")
    if loai and m["rang_buoc"].get("loai") != loai:
        raise LoiNguon(f"{rel(d)}: loai {m['rang_buoc'].get('loai')} != {loai}")
    return m


def doc_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or len(set(r.fieldnames)) != len(r.fieldnames):
            raise LoiNguon(f"{rel(p)}: header rong / trung")
        rows = list(r)
    for x in rows:
        if None in x or any(v is None for v in x.values()):
            raise LoiNguon(f"{rel(p)}: dong thua / thieu cot")
    return rows


def combined_csv(p: Path) -> dict:
    rows = doc_csv(p)
    if len(rows) != 1 or rows[0].get("video") != "COMBINED_SEQ":
        raise LoiNguon(f"{rel(p)}: khong phai mot dong COMBINED_SEQ")
    out = {k: (v if k == "video" else float(v)) for k, v in rows[0].items()}
    if not all(math.isfinite(v) for k, v in out.items() if k != "video"):
        raise LoiNguon(f"{rel(p)}: so khong huu han")
    return out


# ---------------------------------------------------------------------------
# Moi truong / module / cau hinh hieu luc
# ---------------------------------------------------------------------------
def nap_module(ctx: Ctx) -> None:
    import importlib
    env = os.environ.get("BYTETRACK_BACKEND")
    if env is not None and env != ctx.gt["tracker"]["backend"]:
        raise LoiDauVao(f"BYTETRACK_BACKEND={env!r} != {ctx.gt['tracker']['backend']!r}")
    for m in MODULE_NAP:
        try:
            importlib.import_module(m)
        except ImportError as e:
            raise LoiDauVao(f"khong nap duoc module {m}: {e}") from e
    import run_tracking as rt
    rt.tracker_registry()


def gt_backend(ctx: Ctx) -> dict:
    return {"tracker": {"backend": ctx.gt["tracker"]["backend"]}}


def cau_hinh_sort(ctx: Ctx, bl: dict):
    import run_tracking as rt
    y = W.yaml_arg(bl["yaml"])
    cfg = rt.load_tracker_cfg(y, dict(ctx.gt["tracker"]["override"]))
    return cfg, rt.cfg_to_dict(cfg), Path(rt.resolve_tracker_yaml(y))


def cau_hinh_hieu_luc_sort(ctx: Ctx) -> tuple[dict, list[str]]:
    """K1: cau hinh hieu luc DOC TU cfg + instance tracker that (make_tracker, khong suy tu YAML)."""
    import run_tracking as rt
    reg = rt.tracker_registry()
    out, lech = {}, []
    for bl in ctx.gt["tracker"]["bo_loc"]:
        cfg, d, res = cau_hinh_sort(ctx, bl)
        raw = res.read_bytes()
        tr = rt.make_tracker(cfg, reg, assoc_log=ctx.gt["tracker"]["assoc_log"])
        f = tr._new_filter()
        rec = {"yaml": bl["yaml"], "yaml_resolve": rel(res), "yaml_sha256": W.sha256_bytes(raw),
               "yaml_sha256_lf": W.sha256_bytes(raw.replace(b"\r\n", b"\n")), "cfg_hieu_luc": W.chuan_json(d),
               "lop_tracker": type(tr).__name__, "mro": [c.__name__ for c in type(tr).__mro__][:4],
               "instance": {"max_age": tr.max_age, "iou_threshold": tr.iou_threshold, "min_hits": tr.min_hits,
                            "det_thresh": tr.det_thresh, "filter": type(f).__name__,
                            "kf": f"{type(f.kf).__module__}.{type(f.kf).__name__}"},
               "co_assoc_log_mixin": any(c.__name__ == "AssociationLogMixin" for c in type(tr).__mro__),
               "pf_diag_hieu_luc": bool(hasattr(tr, "kalman_filter")),
               "khoa_ctrv_cv_trong_cfg": sorted(k for k in d if k.startswith(("ctrv_", "cv_")))}
        if d.get("tracker_type") != bl["tracker_type"]:
            lech.append(f"{bl['ten']}: tracker_type {d.get('tracker_type')}")
        for k, v in THAM_SO_SORT.items():
            if d.get(k) != v:
                lech.append(f"{bl['ten']}: cfg {k} = {d.get(k)} != {v}")
        i = rec["instance"]
        if (i["max_age"], i["iou_threshold"], i["min_hits"], i["det_thresh"]) != (30, 0.3, 3, 0.25):
            lech.append(f"{bl['ten']}: instance max_age/iou/min_hits/det_thresh {i}")
        if int(d.get("track_buffer", -1)) != 30:
            lech.append(f"{bl['ten']}: track_buffer {d.get('track_buffer')} != 30")
        if rec["lop_tracker"] != "ElapsedSORT" or rec["co_assoc_log_mixin"] or rec["pf_diag_hieu_luc"]:
            lech.append(f"{bl['ten']}: lop {rec['lop_tracker']} / assoc_log mixin / kalman_filter lech")
        if rec["khoa_ctrv_cv_trong_cfg"]:
            lech.append(f"{bl['ten']}: co khoa ctrv_/cv_ {rec['khoa_ctrv_cv_trong_cfg']}")
        if rec["yaml_sha256"][:12] != bl["sha256_12_r51"]:
            lech.append(f"{bl['ten']}: yaml sha256[:12] {rec['yaml_sha256'][:12]} != R51 {bl['sha256_12_r51']}")
        out[bl["ten"]] = rec
    return out, lech


def _tham_so(obj) -> dict:
    import numpy as np
    out = {}
    for k, v in sorted(vars(obj).items()):
        out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    for k in ("init_std_theta", "init_std_omega", "heading_std_after_init", "min_speed_for_heading"):
        if hasattr(type(obj), k):
            out[f"@{k}"] = getattr(obj, k)
    return out


def so_bo_loc_instance(ctx: Ctx) -> tuple[dict, list[str]]:
    """Q/R/P0, lop bo loc, tham so CTRV, dt: so tren INSTANCE bo loc cua SORT (tr._new_filter().kf) va ByteTrack
    (tr.kalman_filter) o CUNG trang thai tham chieu: initiate (P0) -> predict (Q) -> [try_initiate_from_motion] ->
    update (R); yeu cau trung bit (np.array_equal)."""
    import numpy as np
    import run_tracking as rt
    reg = rt.tracker_registry()
    ov = dict(ctx.gt["tracker"]["override"])
    out, lech = {}, []
    z0 = np.array([140.0, 140.0, 1.0, 80.0])
    z1 = z0 + np.array([6.0, 2.0, 0.0, 0.5])
    for bl in ctx.gt["tracker"]["bo_loc"]:
        tr_s = rt.make_tracker(cau_hinh_sort(ctx, bl)[0], reg)
        yb = ctx.gt["tracker"]["yaml_bytetrack"][bl["doi_chung"]]
        tr_b = rt.make_tracker(rt.load_tracker_cfg(W.yaml_arg(yb), ov), reg)
        kf_s, kf_b = tr_s._new_filter().kf, tr_b.kalman_filter
        rec = {"lop_sort": f"{type(kf_s).__module__}.{type(kf_s).__name__}",
               "lop_bytetrack": f"{type(kf_b).__module__}.{type(kf_b).__name__}",
               "lop_tracker_bytetrack": type(tr_b).__name__, "tham_so_sort": _tham_so(kf_s),
               "tham_so_bytetrack": _tham_so(kf_b), "dt_sort": getattr(kf_s, "dt", None), "dt_bytetrack": getattr(kf_b, "dt", None)}
        if type(kf_s) is not type(kf_b):
            lech.append(f"{bl['ten']}: lop bo loc {rec['lop_sort']} != {rec['lop_bytetrack']}")
        if W.chuan_json(rec["tham_so_sort"]) != W.chuan_json(rec["tham_so_bytetrack"]):
            lech.append(f"{bl['ten']}: tham so bo loc instance khac")
        if bl["doi_chung"] == "cv":
            from ultralytics.trackers.byte_tracker import STrack
            rec["tham_so_shared_kalman"] = _tham_so(STrack.shared_kalman)
            if W.chuan_json(rec["tham_so_shared_kalman"]) != W.chuan_json(rec["tham_so_sort"]):
                lech.append("sort-cv30: STrack.shared_kalman khac bo loc SORT")
        elif rec["dt_sort"] != 1.0 or rec["dt_bytetrack"] != 1.0:
            lech.append(f"{bl['ten']}: dt CTRV {rec['dt_sort']} / {rec['dt_bytetrack']} != 1.0 (buoc)")
        buoc = {}
        for ten, kf in (("sort", kf_s), ("bytetrack", kf_b)):
            m, P = kf.initiate(z0.copy())
            s = [("P0", m, P)]
            m, P = kf.predict(m, P)
            s.append(("predict", m, P))
            if bl["doi_chung"] == "ekf":
                m, P, ok = kf.try_initiate_from_motion(m, P, z1.copy(), n_frames=1, ref_pos=(float(z0[0]), float(z0[1])))
                s.append(("init_motion", m, P))
            m, P = kf.update(m, P, z1.copy())
            s.append(("update", m, P))
            buoc[ten] = s
        rec["trang_thai_tham_chieu"] = {}
        for (n1, m1, P1), (_, m2, P2) in zip(buoc["sort"], buoc["bytetrack"]):
            ok = bool(np.array_equal(m1, m2) and np.array_equal(P1, P2))
            rec["trang_thai_tham_chieu"][n1] = {"trung_bit": ok, "mean": np.asarray(m1).tolist()}
            if not ok:
                lech.append(f"{bl['ten']}: trang thai {n1} SORT != ByteTrack")
        out[bl["ten"]] = rec
    return out, lech


def _boxes(arr):
    import numpy as np
    import det_cache
    return det_cache.to_boxes(np.asarray(arr, dtype=np.float32).reshape(-1, 6), W.IMG_HW).cpu().numpy()


def dem_buoc_ctrv(ctx: Ctx) -> dict:
    """Bo dem buoc / quan sat duoi 'skip': cung chuoi detection (xe chay +6 px/frame, co mat o frame 1 va 4-8) qua
    SORT-EKF va ByteTrack-EKF THAT; ghi n_frames moi lan try_initiate_from_motion + so lan predict cua EKF."""
    import numpy as np
    import run_tracking as rt
    from ekf_ctrv import EKFTrackerCTRV
    reg = rt.tracker_registry()
    ov = dict(ctx.gt["tracker"]["override"])
    bl = ctx.bo_loc["sort-ekf30"]
    co_mat = {1, 4, 5, 6, 7, 8}
    out = {}
    goc_try, goc_pred, goc_mp = EKFTrackerCTRV.try_initiate_from_motion, EKFTrackerCTRV.predict, getattr(
        EKFTrackerCTRV, "multi_predict", None)
    for ten, cfg in (("sort", cau_hinh_sort(ctx, bl)[0]),
                     ("bytetrack", rt.load_tracker_cfg(W.yaml_arg(ctx.gt["tracker"]["yaml_bytetrack"]["ekf"]), ov))):
        ghi = {"n_frames": [], "predict": 0, "multi_predict_hang": 0}

        def _try(self, mean, cov, z, n_frames=1, ref_pos=None, _g=ghi):
            _g["n_frames"].append(int(n_frames))
            return goc_try(self, mean, cov, z, n_frames=n_frames, ref_pos=ref_pos)

        def _pred(self, *a, _g=ghi, **kw):
            _g["predict"] += 1
            return goc_pred(self, *a, **kw)

        EKFTrackerCTRV.try_initiate_from_motion, EKFTrackerCTRV.predict = _try, _pred
        if goc_mp is not None:
            def _mp(self, *a, _g=ghi, **kw):
                m0 = np.asarray(a[0] if a else kw.get("mean"))
                _g["multi_predict_hang"] += int(m0.shape[0]) if m0.ndim == 2 else 1
                return goc_mp(self, *a, **kw)
            EKFTrackerCTRV.multi_predict = _mp
        try:
            tr = rt.make_tracker(cfg, reg)
            for f in range(1, 9):
                x = 100.0 + 6.0 * (f - 1)
                arr = [[x, 100.0, x + 80.0, 180.0, 0.9, 2.0]] if f in co_mat else []
                tr.update(_boxes(arr), None, n_steps=1)
        finally:
            EKFTrackerCTRV.try_initiate_from_motion, EKFTrackerCTRV.predict = goc_try, goc_pred
            if goc_mp is not None:
                EKFTrackerCTRV.multi_predict = goc_mp
        out[ten] = ghi
    out["trung_n_frames"] = out["sort"]["n_frames"] == out["bytetrack"]["n_frames"]
    out["ghi_chu"] = ("so lan predict khac nhau la KHAC VONG DOI du kien (SORT predict moi track moi frame; ByteTrack chi "
                      "predict strack_pool, multi_predict vector hoa), khong phai khac tham so")
    return out


def thong_tin_ma(khoa56: dict | None, ctx: Ctx) -> tuple[list[dict], list[str]]:
    """Baseline sha256 MOI file module dang nap (src/, ultralytics, trackeval) + doi chieu khoa R56 (that)."""
    lk = {W.tuyet_doi(f["rel"]): f["sha256"] for f in (khoa56 or {}).get("file", [])}
    rows, lech = [], []
    rieng = FILE_MA.parent
    for p in W.file_module_da_nap():
        s = sha(p)
        k = lk.get(p)
        tt = ("khop_khoa_r56" if k == s else "lech_khoa_r56") if k else (
            "ma_r59" if W.trong(p, rieng) else "ngoai_khoa_r56")
        rows.append({"rel": rel(p), "sha256": s, "sha256_khoa_r56": k, "trang_thai": tt})
        if not ctx.fixture and tt in ("lech_khoa_r56", "ngoai_khoa_r56"):
            lech.append(f"{rel(p)}: {tt}")
    return rows, lech


def config_py_hash(khoa56: dict | None) -> dict:
    p = SRC / "config.py"
    b = p.read_bytes()
    lf = b.replace(b"\r\n", b"\n")
    k = next((f["sha256"] for f in (khoa56 or {}).get("file", []) if f["rel"] == "src/config.py"), None)
    return {"rel": "src/config.py", "size": len(b), "sha256_byte_tho": W.sha256_bytes(b), "sha256_lf": W.sha256_bytes(lf),
            "so_crlf": b.count(b"\r\n"), "so_lf_don": lf.count(b"\n") - b.count(b"\r\n"),
            "khoa_r56_sha256": k, "khoa_r56_quy_uoc": "sha256 byte tho (W.sha256_file)",
            "khop_khoa_r56_byte_tho": k == W.sha256_bytes(b),
            "r51_sha256_lf": R51_CONFIG_PY_LF, "r51_quy_uoc": "sha256(bytes.replace(CRLF, LF)) (audit_protocol.py dong 318)",
            "khop_r51_lf": W.sha256_bytes(lf) == R51_CONFIG_PY_LF,
            "ghi_chu": "hash LF chi giai thich khac quy uoc; toan ven theo khoa goc la hash byte tho"}


# ---------------------------------------------------------------------------
# Nguon R56
# ---------------------------------------------------------------------------
def manifest_cache_r56(ctx: Ctx, det: str, khoa_sha: str) -> tuple[Path, dict, str]:
    goc = ctx.phien56 / "xac_nhan_cache" / det
    ok = []
    for x in sorted(d for d in goc.iterdir() if d.is_dir()) if goc.is_dir() else []:
        m = R55.doc_marker(x)
        if m is None:
            continue
        tt, ly_do, _ = R55.danh_gia_lan(x, m["rang_buoc"])
        rb = m["rang_buoc"]
        if tt != "hop_le":
            raise LoiNguon(f"{det}: manifest {rel(x)} {tt} {ly_do}")
        if (rb.get("loai"), rb.get("detector"), rb.get("khoa_sha256")) == ("xac-nhan-cache", det, khoa_sha):
            ok.append(x)
    if len(ok) != 1:
        raise LoiNguon(f"{det}: {len(ok)} manifest cache da cong bo (can dung 1)")
    d = ok[0]
    s = sha(d / "manifest_cache.json")
    sums = (d / "SHA256SUMS").read_text(encoding="utf-8").split() if (d / "SHA256SUMS").is_file() else []
    if sums[:2] != [s, "manifest_cache.json"]:
        raise LoiNguon(f"{det}: manifest_cache.json khac SHA256SUMS")
    m = json.loads((d / "manifest_cache.json").read_text(encoding="utf-8"))
    if m.get("detector") != det or m.get("rang_buoc", {}).get("khoa_sha256") != khoa_sha or list(m.get("video", {})) != ctx.videos:
        raise LoiNguon(f"{det}: manifest cache khac detector / khoa / danh sach video manifest")
    return d, m, s


def seq_info_hien_tai(ctx: Ctx) -> dict:
    return {v: int(W.doc_seqinfo(ctx.seqinfo(v))["seqLength"]) for v in ctx.videos}


def kiem_npz_nhe(ctx: Ctx, lan: Path, v: str, n: int) -> list[str]:
    import numpy as np
    loi = []
    with np.load(lan / f"{v}.npz") as z:
        if set(z.files) != set(R55.KHOA_NPZ):
            return [f"{v}: khoa npz {sorted(z.files)}"]
        a = {k: z[k] for k in R55.KHOA_NPZ}
    loi += [f"{v}: {k} dtype {a[k].dtype}" for k, t in R55.KHOA_NPZ.items() if str(a[k].dtype) != t]
    if a["frame_idx"].shape != (n,) or not np.array_equal(a["frame_idx"], np.arange(1, n + 1)):
        loi.append(f"{v}: frame_idx khac 1..{n}")
    if a["unreadable"].any() or a["truncated"].any() or int(a["n_per_frame"].sum()) != len(a["data"]):
        loi.append(f"{v}: unreadable / truncated / n_per_frame lech")
    if a["orig_shape"].tolist() != list(W.IMG_HW):
        loi.append(f"{v}: orig_shape {a['orig_shape'].tolist()}")
    return loi


def kiem_nguon_r56(ctx: Ctx, log=print) -> dict:
    """Muc 4.3 dieu 1-2 (chi DOC): khoa R56, phien, nguon.json, 6 lan cham doi chung + 240 MOT + metric, cache 3
    detector theo manifest da cong bo, GT/seqinfo theo khoa R56. Tra ve {nguon, file, lech, khoa56}."""
    n = ctx.gt["nguon_r56"]
    td = ctx.gt["tap_du_lieu"]
    K = n["khoa_json_sha256"]
    lech, files = [], []

    def ghi(p: Path, vai: str, s: str | None = None) -> str:
        s = s or sha(p)
        files.append({"rel": rel(p), "vai_tro": vai, "sha256": s})
        return s

    k56 = W.kiem_khoa({"chuan_bi": ctx.khoa56.parent}, ctx.khoa56.name)
    lech += [f"khoa R56: {x}" for x in k56["lech"][:20]]
    if k56["khoa_sha256"] != K:
        lech.append(f"khoa R56 khoa.json {k56['khoa_sha256']} != {K}")
    khoa = k56["_khoa"]
    if khoa.get("buoc") != n["khoa_buoc"] or (not ctx.fixture and khoa.get("fixture") is not False):
        lech.append(f"khoa R56 buoc {khoa.get('buoc')} / fixture {khoa.get('fixture')}")
    lien_ket = None
    if not ctx.fixture:                          # khoa R56 -> khoa R55 cha (ham R55 that, chi doc)
        c55 = R55.Ctx()
        kq = R55.kiem_khoa_r55(c55, ctx.khoa56.name)
        lien_ket = {"dat": kq["dat"], "lech": kq["lech"][:10], "so_file": kq["so_file"]}
        lech += [f"khoa R56 (R55.kiem_khoa_r55): {x}" for x in kq["lech"][:10]]
    ghi(ctx.khoa56 / "khoa.json", "khoa_r56")
    ghi(ctx.khoa56 / "SHA256SUMS", "khoa_r56")
    pj = R55.doc_json(ctx.phien56 / "phien.json", "phien.json R56")
    if (pj.get("khoa"), pj.get("khoa_sha256")) != (ctx.khoa56.name, K):
        lech.append("phien.json R56 khong gan voi khoa R56 da dinh danh")
    ghi(ctx.phien56 / "phien.json", "phien_r56")
    for p, s_kv, vai in ((ctx.nguon_json, n["nguon_json_sha256"], "nguon_r56"), (ctx.bang56, n["bang_r56_sha256"], "bang_r56"),
                         (ctx.manifest, td["manifest_sha256"], "manifest_video"), (ctx.loai_d4, td["loai_d4_sha256"], "loai_d4")):
        if ghi(p, vai) != s_kv:
            lech.append(f"{rel(p)} sha256 != giao thuc")
    if not ctx.fixture:
        for k in ("giao_thuc_r54", "hop_dong_r55"):
            if ghi(ctx.p(n[k]), k) != n[f"{k}_sha256"]:
                lech.append(f"{n[k]} sha256 != giao thuc")
    nj = R55.doc_json(ctx.nguon_json, "nguon.json R56")
    # GT / seqinfo theo khoa R56
    lk = {W.tuyet_doi(f["rel"]): f["sha256"] for f in khoa.get("file", [])}
    gt_rows = []
    for v in ctx.videos:
        for p in (ctx.gt_txt(v), ctx.seqinfo(v)):
            s = ghi(p, "gt_r56")
            if lk.get(p.resolve()) != s:
                lech.append(f"{rel(p)}: khong co trong khoa R56 / sha256 lech")
            gt_rows.append((rel(p), s))
    si = seq_info_hien_tai(ctx)
    nguon = {"khoa_r56": {"thu_muc": rel(ctx.khoa56), "khoa_json_sha256": k56["khoa_sha256"], "lien_ket_cha": lien_ket},
             "phien_r56": rel(ctx.phien56), "gt_sha256_ds": W.sha256_ds(gt_rows), "seq_info": si, "cache": {},
             "doi_chung": {}}
    c55 = R55.Ctx() if not ctx.fixture else None
    for det in ctx.gt["detector"]:
        try:
            md, man, ms = manifest_cache_r56(ctx, det, K)
        except LoiNguon as e:
            lech.append(str(e))
            continue
        for f in ("manifest_cache.json", "SHA256SUMS", MARKER):
            ghi(md / f, "manifest_cache_r56")
        cv = {"manifest_dir": rel(md), "manifest_sha256": ms, "video": {}}
        for v in ctx.videos:
            e = man["video"][v]
            lan = W.tuyet_doi(e["lan"])
            if lan.parent != (ctx.phien56 / "cache" / det / v).resolve():
                lech.append(f"{det}/{v}: lan cache ngoai phien R56")
                continue
            for f, k in ((f"{v}.npz", "npz_sha256"), (f"{v}.json", "json_sha256"), ("meta.json", "meta_sha256"),
                         (MARKER, "marker_sha256")):
                if ghi(lan / f, "cache_r56") != e[k]:
                    lech.append(f"{det}/{v}/{f}: sha256 != manifest")
            try:
                mk = lan_hop_le(lan, "cache")
                loi = kiem_npz_nhe(ctx, lan, v, int(e["n_frames"]))
                if c55 is not None:
                    loi += R55.kiem_npz(c55, det, lan, v, int(e["n_frames"]), mk["rang_buoc"])
                if int(e["n_frames"]) != si[v]:
                    loi.append(f"{v}: n_frames {e['n_frames']} != seqLength {si[v]}")
                lech += [f"{det}/{x}" for x in loi]
            except LoiNguon as x:
                lech.append(str(x))
            cv["video"][v] = {"lan": rel(lan), "npz_sha256": e["npz_sha256"], "n_frames": int(e["n_frames"])}
        nguon["cache"][det] = cv
        log(f"  [nguon] cache {det}: {len(cv['video'])} video, manifest {ms[:12]}")
    for r in ctx.gt["ma_tran"]["runs"]:
        dc = r["doi_chung"]
        try:
            nguon["doi_chung"][dc] = kiem_doi_chung(ctx, r, dc, nj, K, nguon, ghi, lech)
        except (LoiNguon, KeyError) as e:
            lech.append(f"{dc}: {type(e).__name__} {e}")
        log(f"  [nguon] doi chung {dc}: {'dat' if dc in nguon['doi_chung'] else 'LECH'}")
    return {"nguon": nguon, "file": files, "lech": lech, "khoa56": khoa}


def kiem_doi_chung(ctx: Ctx, r: dict, dc: str, nj: dict, K: str, nguon: dict, ghi, lech: list) -> dict:
    s = nj["summary"][dc]
    f = W.tuyet_doi(s["file"])
    if ghi(f, "summary_r56") != s["sha256"]:
        raise LoiNguon("pedestrian_summary.txt sha256 != nguon.json")
    lan = f.parents[2]
    if lan.parent != (ctx.phien56 / "cham" / dc).resolve() or f != (lan / "trackeval" / dc / "pedestrian_summary.txt"):
        raise LoiNguon(f"lan cham {rel(lan)} khong nam o cham/{dc}/ cua phien R56")
    m = lan_hop_le(lan, "cham")
    rb = m["rang_buoc"]
    cv = nguon["cache"].get(r["detector"])
    if (rb.get("run"), rb.get("khoa_sha256"), rb.get("split")) != (dc, K, ctx.split) or cv is None or \
            rb.get("manifest_cache_sha256") != cv["manifest_sha256"]:
        raise LoiNguon("rang buoc lan cham (run / khoa / split / manifest cache) lech")
    if W.chuan_json(rb.get("seq_info")) != W.chuan_json(nguon["seq_info"]) or rb.get("gt_sha256_ds") is None:
        raise LoiNguon("seq_info lan cham != seqinfo hien tai")
    ghi(lan / MARKER, "marker_cham_r56")
    fm = {}
    for x in METRIC_FILE:
        p = lan / "trackeval" / dc / x
        fm[x] = ghi(p, "metric_r56")
        if m["file_metric"].get(x) != fm[x] or m["file"].get(f"trackeval/{dc}/{x}") != fm[x]:
            raise LoiNguon(f"{x}: sha256 != file_metric marker")
    comb = combined_csv(lan / "trackeval" / dc / "combined_metrics.csv")
    if comb != m["combined"]:
        raise LoiNguon("combined_metrics.csv != combined trong marker")
    mot = {}
    for v in ctx.videos:
        key = f"trackers/{ctx.split}/{dc}/data/{v}.txt"
        sm = m["file"][key]
        ghi(lan / key, "mot_r56_cham", sm)
        tl = W.tuyet_doi(m["mot_lan"][v])
        if tl.parent != (ctx.phien56 / "tracking" / dc / v).resolve():
            raise LoiNguon(f"{v}: lan tracking ngoai phien R56")
        mt = lan_hop_le(tl, "tracking")
        tb = mt["rang_buoc"]
        if mt["file"].get(f"data/{v}.txt") != sm:
            raise LoiNguon(f"{v}: MOT ban sao cham != MOT lan tracking")
        if (tb.get("run"), tb.get("detector"), tb.get("cond"), tb.get("conf_min"), tb.get("video"), tb.get("khoa_sha256"),
                tb.get("npz_sha256"), tb.get("manifest_cache_sha256")) != (
                dc, r["detector"], "c25", 0.25, v, K, cv["video"][v]["npz_sha256"], cv["manifest_sha256"]):
            raise LoiNguon(f"{v}: rang buoc lan tracking R56 lech")
        if int(mt["info"]["n_frames"]) != nguon["seq_info"][v] or int(mt["info"]["n_buoc_timing"]) != nguon["seq_info"][v]:
            raise LoiNguon(f"{v}: timeline tracking R56 lech")
        ghi(tl / "data" / f"{v}.txt", "mot_r56_tracking", sm)
        ghi(tl / MARKER, "marker_tracking_r56")
        mot[v] = {"sha256": sm, "lan_tracking": rel(tl)}
    if W.sha256_ds((v, x["sha256"]) for v, x in mot.items()) != rb.get("mot_sha256_ds"):
        raise LoiNguon("mot_sha256_ds lan cham != 40 MOT")
    khac = sorted(rel(x) for x in (ctx.phien56 / "cham" / dc).iterdir() if x.is_dir() and x != lan)
    return {"lan_cham": rel(lan), "marker_sha256": sha(lan / MARKER), "file_metric": fm, "combined": comb,
            "summary": {"file": rel(f), "sha256": s["sha256"]}, "mot": mot, "mot_sha256_ds": rb["mot_sha256_ds"],
            "gt_sha256_ds_r56": rb["gt_sha256_ds"], "seq_info": rb["seq_info"], "lan_cham_khac_khong_dung": khac}


def file_cache(nguon: dict, dets=None) -> list[Path]:
    out = []
    for det, cv in nguon["cache"].items():
        if dets is not None and det not in dets:
            continue
        md = W.tuyet_doi(cv["manifest_dir"])
        out += [md / "manifest_cache.json", md / "SHA256SUMS", md / MARKER]
        for v, e in cv["video"].items():
            lan = W.tuyet_doi(e["lan"])
            out += [lan / f"{v}.npz", lan / f"{v}.json", lan / "meta.json", lan / MARKER]
    return out


def bam_cache(nguon: dict, dets=None) -> dict:
    rows = sorted((rel(p), sha(p)) for p in file_cache(nguon, dets))
    return {"so_file": len(rows), "sha256_ds": W.sha256_ds(rows)}


# ---------------------------------------------------------------------------
# K2 parity detection / timeline / D4
# ---------------------------------------------------------------------------
def doc_d4(ctx: Ctx) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for x in doc_csv(ctx.loai_d4):
        out.setdefault(x["video"], set()).add(int(x["frame"]))
    return out


def k2_video(ctx: Ctx, det: str, v: str, lan: Path, n_kv: int, d4: set[int], tr_b) -> dict:
    """Mot (detector, video): runner (VideoDetections.frame) == tham chieu doc lap tu NPZ tho (conf > 0.25 ngat ->
    lop -> [:300]), float32 trung byte; tap SORT nhan (conf float64 > det_thresh) == tang cao ByteTrack THAT
    (tr._split_detections) ca noi dung lan thu tu; tang thap rong. Hash noi dung, khong chi dem."""
    import numpy as np
    t = ctx.gt["tai"]
    th = ctx.gt["tracker"]["tham_so_sort"]["det_thresh"]
    vd = W.mo_cache(lan, v, t["conf_min"], t["classes"], t["max_det"])
    cls_set = np.array(sorted(t["classes"]), dtype=np.int64)
    h_run, h_sort = hashlib.sha256(), hashlib.sha256()
    d = {"detector": det, "video": v, "n_frames": len(vd), "n_unreadable": int(vd.unreadable.sum()), "n_det_cache": int(len(vd._data)),
         "n_det_runner": 0, "n_det_sort": 0, "n_det_bt_cao": 0, "n_det_bt_thap": 0, "n_suy_bien": 0,
         "n_conf_bang_nguong_cache": 0, "n_frame_cat_300": 0, "n_frame_rong": 0, "n_d4": 0, "n_d4_co_det": 0}
    loi = None
    if len(vd) != n_kv or not np.array_equal(vd.frame_idx, np.arange(1, n_kv + 1)):
        loi = f"timeline {len(vd)} != {n_kv}"
    for i in range(len(vd)):
        fi = int(vd.frame_idx[i])
        raw = vd._data[vd.offsets[i]:vd.offsets[i + 1]]
        arr = vd.frame(i)
        if len(raw):
            keep = (raw[:, 4].astype(np.float64) > float(t["conf_min"])) & np.isin(raw[:, 5].astype(np.int64), cls_set)
            ref = raw[keep]
            d["n_conf_bang_nguong_cache"] += int((raw[:, 4] == np.float32(t["conf_min"])).sum())
        else:
            ref = raw.reshape(0, 6)
        if len(ref) > t["max_det"]:
            d["n_frame_cat_300"] += 1
        ref = np.ascontiguousarray(ref[: t["max_det"]], dtype=np.float32)
        if loi is None and (arr.dtype != np.float32 or arr.shape != ref.shape or arr.tobytes() != ref.tobytes()):
            loi = f"frame {fi}: runner != tham chieu doc lap"
        b = _boxes(arr)
        conf64 = np.asarray(b.conf, dtype=np.float64).reshape(-1)
        xyxy64 = np.asarray(b.xyxy, dtype=np.float64).reshape(-1, 4)
        idx_s = np.flatnonzero(conf64 > th)
        _, _, cao, thap = tr_b._split_detections(b)
        idx_b = np.flatnonzero(np.asarray(cao))
        if loi is None and (not np.array_equal(idx_s, idx_b) or np.asarray(thap).any()):
            loi = f"frame {fi}: tap SORT {idx_s.tolist()[:5]} != tang cao ByteTrack {idx_b.tolist()[:5]} / tang thap khac rong"
        if loi is None and not np.array_equal(xyxy64[idx_s], arr[idx_s, :4].astype(np.float64)):
            loi = f"frame {fi}: toa do SORT != runner"
        h_run.update(fi.to_bytes(4, "little") + arr.tobytes())
        h_sort.update(fi.to_bytes(4, "little") + xyxy64[idx_s].tobytes() + conf64[idx_s].tobytes())
        wh = np.asarray(b.xywh)[:, 2:4] if len(arr) else np.zeros((0, 2))
        d["n_suy_bien"] += int((~((wh[:, 0] > 0) & (wh[:, 1] > 0))).sum())
        d["n_det_runner"] += len(arr)
        d["n_det_sort"] += len(idx_s)
        d["n_det_bt_cao"] += len(idx_b)
        d["n_det_bt_thap"] += int(np.asarray(thap).sum())
        d["n_frame_rong"] += int(len(arr) == 0)
        if fi in d4:
            d["n_d4"] += 1
            d["n_d4_co_det"] += int(len(arr) > 0)
    if loi is None and d["n_d4"] != len(d4):
        loi = f"D4 {d['n_d4']} != {len(d4)} anh trong danh sach"
    d.update({"sha256_runner": h_run.hexdigest(), "sha256_sort": h_sort.hexdigest(), "dat": loi is None, "loi_dau": loi or ""})
    return d


def kiem_k2(ctx: Ctx, nguon: dict, log=print) -> tuple[list[dict], dict, list[str]]:
    import run_tracking as rt
    reg = rt.tracker_registry()
    ov = dict(ctx.gt["tracker"]["override"])
    cfg_b = {k: rt.load_tracker_cfg(W.yaml_arg(y), ov) for k, y in ctx.gt["tracker"]["yaml_bytetrack"].items()}
    lech = []
    nguong = {k: (float(c.track_high_thresh), float(c.track_low_thresh), float(c.new_track_thresh)) for k, c in cfg_b.items()}
    if len(set(nguong.values())) != 1 or nguong["cv"] != (0.25, 0.1, 0.25):
        lech.append(f"nguong ByteTrack cv/ekf {nguong} != (0.25, 0.1, 0.25)")
    tr_b = rt.make_tracker(cfg_b["cv"], reg)
    d4 = doc_d4(ctx)
    rows = []
    for det in ctx.gt["detector"]:
        cv = nguon["cache"].get(det)
        if cv is None:
            lech.append(f"K2 {det}: khong co cache da kiem")
            continue
        for v in ctx.videos:
            e = cv["video"][v]
            rows.append(k2_video(ctx, det, v, W.tuyet_doi(e["lan"]), e["n_frames"], d4.get(v, set()), tr_b))
            if not rows[-1]["dat"]:
                lech.append(f"K2 {det}/{v}: {rows[-1]['loi_dau']}")
        log(f"  [K2] {det}: {sum(r['dat'] for r in rows if r['detector'] == det)}/{len(ctx.videos)} dat")
    td = ctx.gt["tap_du_lieu"]
    tl = {"so_to_hop": len(rows), "so_dat": sum(r["dat"] for r in rows), "nguong_bytetrack": nguong,
          "anh_moi_detector": {det: sum(r["n_frames"] for r in rows if r["detector"] == det) for det in ctx.gt["detector"]},
          "d4_moi_detector": {det: sum(r["n_d4"] for r in rows if r["detector"] == det) for det in ctx.gt["detector"]},
          "so_anh_ky_vong": td["so_anh"], "so_d4_ky_vong": td["so_anh_D4"], "d4_danh_sach": sum(len(x) for x in d4.values()),
          "seq_info_tong": sum(nguon["seq_info"].values()), "fps": ctx.gt["tracker"]["fps"],
          "elapsed_mode": ctx.gt["tracker"]["elapsed_mode"],
          "tong_suy_bien": sum(r["n_suy_bien"] for r in rows), "tong_conf_bang_nguong_cache": sum(r["n_conf_bang_nguong_cache"] for r in rows),
          "tong_frame_cat_300": sum(r["n_frame_cat_300"] for r in rows)}
    so_to_hop = len(ctx.gt["detector"]) * len(ctx.videos)
    if tl["so_dat"] != so_to_hop or tl["so_to_hop"] != so_to_hop:
        lech.append(f"K2 {tl['so_dat']}/{so_to_hop} to hop dat")
    if any(x != td["so_anh"] for x in tl["anh_moi_detector"].values()) or tl["seq_info_tong"] != td["so_anh"]:
        lech.append(f"timeline {tl['anh_moi_detector']} / seqinfo {tl['seq_info_tong']} != {td['so_anh']}")
    if any(x != td["so_anh_D4"] for x in tl["d4_moi_detector"].values()) or tl["d4_danh_sach"] != td["so_anh_D4"]:
        lech.append(f"D4 {tl['d4_moi_detector']} != {td['so_anh_D4']}")
    if tl["tong_suy_bien"]:
        lech.append(f"{tl['tong_suy_bien']} box suy bien (w/h <= 0): ByteTrack loai, SORT giu - can giai thich")
    return rows, tl, lech


# ---------------------------------------------------------------------------
# Buoc A
# ---------------------------------------------------------------------------
def buoc_chuan_bi(ctx: Ctx, args, rec: list) -> int:
    d = W.tao_phien(ctx.vung("chuan_bi"), None, "chuan_bi")
    rec[0] = d
    W.dat_moi_truong(d)
    nap_module(ctx)
    t0 = time.time()
    ng = kiem_nguon_r56(ctx)
    lech = list(ng["lech"])
    khoa56 = ng["khoa56"]
    h0 = bam_cache(ng["nguon"])
    ma, l_ma = thong_tin_ma(khoa56, ctx)
    lech += l_ma
    cfgpy = config_py_hash(khoa56)
    if not ctx.fixture and not cfgpy["khop_khoa_r56_byte_tho"]:
        lech.append("config.py byte tho != khoa R56")
    bk = W.thong_tin_backend(gt_backend(ctx))
    tv = W.phien_ban_thu_vien()
    cf56 = None
    if not ctx.fixture:
        if W.chuan_json(bk) != W.chuan_json(khoa56.get("backend_tracker")):
            lech.append("backend tracker != khoa R56")
        if tv != khoa56.get("thu_vien"):
            lech.append("phien ban thu vien != khoa R56")
        cf56, l2 = W.cau_hinh_hieu_luc(W.doc_giao_thuc(W.GIAO_THUC))
        lech += l2
        if W.chuan_json(cf56) != W.chuan_json(khoa56.get("cau_hinh_hieu_luc")):
            lech.append("cau hinh ByteTrack hieu luc != khoa R56")
    ch, l3 = cau_hinh_hieu_luc_sort(ctx)
    bi, l4 = so_bo_loc_instance(ctx)
    db = dem_buoc_ctrv(ctx)
    lech += l3 + l4 + ([] if db["trung_n_frames"] else [f"n_frames SORT {db['sort']['n_frames']} != ByteTrack {db['bytetrack']['n_frames']}"])
    rows, tl, l5 = kiem_k2(ctx, ng["nguon"])
    lech += l5
    h1 = bam_cache(ng["nguon"])
    if h0 != h1:
        lech.append("cache R56 doi trong luc chuan-bi")
    cot = list(rows[0]) if rows else ["detector"]
    W.ghi_bang_csv(d / "k2_parity.csv", rows, [d]) if rows else None
    out = {"buoc": "chuan-bi", "fixture": ctx.fixture, "luc": bay_gio(), "giay": round(time.time() - t0, 1),
           "giao_thuc": rel(ctx.gt_path), "giao_thuc_sha256": sha(ctx.gt_path),
           "ma_sha256": {"dieu_phoi": sha(FILE_MA), "test": sha(FILE_TEST)},
           "ket_luan": "dat" if not lech else "lech", "lech": lech, "nguon": ng["nguon"], "file_nguon": ng["file"],
           "module_nap": ma, "config_py": cfgpy, "backend": bk, "thu_vien": tv, "cau_hinh_bytetrack_r56": cf56,
           "cau_hinh_sort": ch, "bo_loc_instance": bi, "dem_buoc_ctrv": db, "k2": {"cot": cot, "tom_tat": tl},
           "cache_truoc": h0, "cache_sau": h1}
    W.ghi_json(d / "chuan_bi.json", out, [d])
    print(f"[chuan-bi] {rel(d)}: {out['ket_luan']} ({len(lech)} lech); K2 {tl['so_dat']}/{tl['so_to_hop']}; "
          f"anh {tl['anh_moi_detector']}; D4 {tl['d4_moi_detector']}; config.py byte {cfgpy['khop_khoa_r56_byte_tho']} "
          f"LF-R51 {cfgpy['khop_r51_lf']}")
    for x in lech[:15]:
        print(f"  LECH: {x}")
    if lech:
        return 3
    ghi_marker(ctx, "chuan-bi", d, {"loai": "chuan-bi", "giao_thuc_sha256": out["giao_thuc_sha256"]}, {"ket_luan": "dat"})
    print(f"CHUAN_BI_ID={d.name}")
    return 0


def kiem_moi_truong_r59(ctx: Ctx, k: dict) -> list[str]:
    nap_module(ctx)
    lock = {W.tuyet_doi(f["rel"]): f["sha256"] for f in k["file"]}
    nap = W.file_module_da_nap()
    lech = [f"module nap ngoai khoa R59: {rel(p)}" for p in nap if p not in lock]
    lech += [f"{rel(p)}: sha256 khac khoa R59" for p in nap if p in lock and sha(p) != lock[p]]
    if W.chuan_json(W.thong_tin_backend(gt_backend(ctx))) != W.chuan_json(k.get("backend")):
        lech.append("backend tracker khac khoa R59")
    ch, l3 = cau_hinh_hieu_luc_sort(ctx)
    if W.chuan_json(ch) != W.chuan_json(k.get("cau_hinh_hieu_luc")):
        lech.append("cau hinh SORT hieu luc khac khoa R59")
    if W.phien_ban_thu_vien() != k.get("thu_vien"):
        lech.append("phien ban thu vien khac khoa R59")
    return lech + l3


def kiem_khoa_r59(ctx: Ctx, ten: str | None, moi_truong: bool = True) -> dict:
    if not ten or not W.TEN_HOP_LE.match(ten) or ".." in ten or not ten.startswith("khoa_"):
        raise LoiDauVao(f"can --khoa <khoa_...> hop le (khong tu chon moi nhat), nhan {ten!r}")
    kq = W.kiem_khoa({"chuan_bi": ctx.vung("chuan_bi")}, ten)
    k = kq["_khoa"]
    lech = kq["lech"]
    if k.get("buoc") != "R59_LOCK" or k.get("fixture") != ctx.fixture:
        lech.append(f"khoa {ten} ({k.get('buoc')}, fixture={k.get('fixture')}) khong phai R59_LOCK cua che do nay")
    if k.get("giao_thuc_sha256") != sha(ctx.gt_path):
        lech.append("giao thuc R59 khac khoa")
    if k.get("khoa_r56", {}).get("khoa_json_sha256") != ctx.gt["nguon_r56"]["khoa_json_sha256"]:
        lech.append("khoa R59 khong noi toi khoa R56 da dinh danh")
    if moi_truong:
        lech += kiem_moi_truong_r59(ctx, k)
    kq["dat"] = not lech
    return kq


def buoc_khoa(ctx: Ctx, args, rec: list) -> int:
    if not args.chuan_bi or not args.tests:
        raise LoiDauVao("can --chuan-bi <ID> --tests <ID> tuong minh")
    cb = R55.kiem_doc_lap(ctx.vung("chuan_bi") / args.chuan_bi, ctx.goc)
    mk = lan_hop_le(cb, "chuan-bi")
    cbj = R55.doc_json(cb / "chuan_bi.json", "chuan_bi.json")
    if cbj.get("ket_luan") != "dat" or mk.get("ket_luan") != "dat":
        raise LoiDauVao("chuan-bi chua dat")
    tdir = R55.kiem_doc_lap(ctx.vung("tests") / args.tests, ctx.goc)
    tj = R55.doc_json(tdir / "ket_qua_test.json", "ket_qua_test.json")
    hien = {"dieu_phoi": sha(FILE_MA), "test": sha(FILE_TEST), "giao_thuc": sha(GIAO_THUC)}
    if not tj.get("dat") or tj.get("ma_sha256") != hien:
        raise LoiDauVao(f"fixture chua dat hoac chay tren ma khac: {tj.get('ma_sha256')} != {hien}")
    if cbj["ma_sha256"] != {"dieu_phoi": hien["dieu_phoi"], "test": hien["test"]} or cbj["giao_thuc_sha256"] != sha(ctx.gt_path):
        raise LoiDauVao("chuan-bi chay tren ma / giao thuc khac hien tai")
    d = W.tao_phien(ctx.vung("chuan_bi"), None, "khoa")
    rec[0] = d
    W.dat_moi_truong(d)
    nap_module(ctx)
    lech = []
    files = [{"rel": rel(p), "vai_tro": "module_nap", "sha256": sha(p)} for p in W.file_module_da_nap()]
    da = {f["rel"] for f in files}
    them = [(FILE_MA, "ma_r59"), (FILE_TEST, "fixture_r59"), (ctx.gt_path, "giao_thuc_r59"), (cb / "chuan_bi.json", "chuan_bi"),
            (cb / "k2_parity.csv", "chuan_bi"), (cb / MARKER, "chuan_bi"), (tdir / "ket_qua_test.json", "ket_qua_test")]
    for bl in ctx.gt["tracker"]["bo_loc"]:
        them.append((cau_hinh_sort(ctx, bl)[2], "yaml_sort"))
    import run_tracking as rt
    for y in ctx.gt["tracker"]["yaml_bytetrack"].values():
        them.append((Path(rt.resolve_tracker_yaml(W.yaml_arg(y))), "yaml_bytetrack"))
    for p, vai in them:
        if rel(p) not in da:
            files.append({"rel": rel(p), "vai_tro": vai, "sha256": sha(p)})
            da.add(rel(p))
    for f in cbj["file_nguon"]:                  # nguon R56 da kiem o chuan-bi: bam lai, phai trung
        s = sha(W.tuyet_doi(f["rel"]))
        if s != f["sha256"]:
            lech.append(f"{f['rel']}: doi tu chuan-bi")
        if f["rel"] not in da:
            files.append(dict(f))
            da.add(f["rel"])
    ma_cb = {x["rel"]: x["sha256"] for x in cbj["module_nap"]}
    lech += [f"{f['rel']}: module khac luc chuan-bi" for f in files if f["vai_tro"] == "module_nap" and ma_cb.get(f["rel"], f["sha256"]) != f["sha256"]]
    ch, l3 = cau_hinh_hieu_luc_sort(ctx)
    lech += l3
    if W.chuan_json(ch) != W.chuan_json(cbj["cau_hinh_sort"]):
        lech.append("cau hinh SORT hieu luc khac chuan-bi")
    bk = W.thong_tin_backend(gt_backend(ctx))
    if W.chuan_json(bk) != W.chuan_json(cbj["backend"]):
        lech.append("backend khac chuan-bi")
    if lech:
        for x in lech[:10]:
            print(f"  LECH: {x}")
        W.ghi_json(d / "LOI.json", {"lech": lech, "luc": bay_gio()}, [d])
        return 2
    khoa = {"phien_ban": 1, "buoc": "R59_LOCK", "fixture": ctx.fixture, "tao_luc": bay_gio(), "phien": rel(d),
            "giao_thuc": rel(ctx.gt_path), "giao_thuc_sha256": sha(ctx.gt_path), "chuan_bi": rel(cb),
            "tests": rel(tdir), "so_test": tj.get("so_test"),
            "khoa_r56": {"thu_muc": cbj["nguon"]["khoa_r56"]["thu_muc"],
                         "khoa_json_sha256": cbj["nguon"]["khoa_r56"]["khoa_json_sha256"]},
            "nguon": cbj["nguon"], "cau_hinh_hieu_luc": ch, "bo_loc_instance": cbj["bo_loc_instance"],
            "dem_buoc_ctrv": cbj["dem_buoc_ctrv"], "config_py": cbj["config_py"], "backend": bk,
            "thu_vien": W.phien_ban_thu_vien(), "k2": cbj["k2"]["tom_tat"], "cache": bam_cache(cbj["nguon"]),
            "so_file": len(files), "file": files,
            "ghi_chu": "khong gom PLAN_DA_CHOT.md muc 0 / bao cao tien do; sua sau khoa -> khoa moi, giu khoa cu"}
    s = W.ghi_json(d / "khoa.json", khoa, [d])
    (d / "SHA256SUMS").write_text(f"{s}  khoa.json\n", encoding="utf-8")
    print(f"[khoa] {rel(d)}: {len(files)} file, khoa.json {s}")
    print(f"KHOA_ID={d.name}")
    return 0


def buoc_kiem_khoa(ctx: Ctx, args, rec: list) -> int:
    d = W.tao_phien(ctx.vung("chuan_bi"), None, "kiem_khoa")
    rec[0] = d
    W.dat_moi_truong(d)
    kq = kiem_khoa_r59(ctx, args.khoa)
    out = {k: v for k, v in kq.items() if not k.startswith("_")}
    W.ghi_json(d / "ket_qua.json", out, [d])
    print(f"[kiem-khoa] {kq['khoa']}: {kq['so_file']} file, khoa.json {kq['khoa_sha256']}, "
          f"{'DAT' if kq['dat'] else 'LECH ' + str(kq['lech'][:5])}")
    return 0 if kq["dat"] else 2


class KhungDau:
    """Chi n frame dau cua VideoDetections (smoke): cung iter_frames / orig_shape, khong doi detection."""

    def __init__(self, vd, n: int):
        self.vd, self.n, self.orig_shape = vd, n, vd.orig_shape

    def iter_frames(self):
        for i, x in enumerate(self.vd.iter_frames()):
            if i >= self.n:
                break
            yield x

    def __len__(self):
        return self.n


def kiem_info_sort(info: dict, n: int) -> list[str]:
    if (info["n_frames"], info["n_buoc_timing"], info["n_steps"], info["track_buffer"], info["tracker_lop"]) != (
            n, n, [1], 30, "ElapsedSORT"):
        return [f"{info['video']}: frame/buoc/n_steps/track_buffer/lop lech {info}"]
    return []


def mo_cache_kiem(ctx: Ctx, k: dict, det: str, v: str):
    e = k["nguon"]["cache"][det]["video"][v]
    lan = W.tuyet_doi(e["lan"])
    if sha(lan / f"{v}.npz") != e["npz_sha256"]:
        raise LoiDauVao(f"{det}/{v}.npz doi so voi khoa")
    t = ctx.gt["tai"]
    return W.mo_cache(lan, v, t["conf_min"], t["classes"], t["max_det"]), lan, e


def buoc_smoke(ctx: Ctx, args, rec: list) -> int:
    kk = kiem_khoa_r59(ctx, args.khoa, moi_truong=False)
    if not kk["dat"]:
        raise LoiDauVao(f"kiem-khoa lech: {kk['lech'][:5]}")
    k = kk["_khoa"]
    d = W.tao_phien(ctx.vung("smoke"), None, "smoke")
    rec[0] = d
    W.dat_moi_truong(d)                          # dua src/ vao sys.path + cau hinh thu vien TRUOC import run_tracking
    import run_tracking as rt
    lech = kiem_moi_truong_r59(ctx, k)
    if lech:
        raise LoiDauVao(f"moi truong lech khoa: {lech[:5]}")
    sm = ctx.gt["smoke"]
    det, n = sm["detector"], sm["frame_cuoi"] - sm["frame_dau"] + 1
    h0 = bam_cache(k["nguon"], [det])
    if h0 != bam_cache_khoa(k, [det]):
        raise LoiDauVao("cache khac khoa truoc smoke")

    class _NG:
        gt_txt = staticmethod(ctx.gt_txt)
        seqinfo = staticmethod(ctx.seqinfo)

    gs = W.tao_gt_smoke(_NG(), {"videos": sm["videos"], "frame_dau": sm["frame_dau"], "frame_cuoi": sm["frame_cuoi"]},
                        d / "gt", [d])
    reg = rt.tracker_registry()
    kq = {"run": {}, "gt_smoke": gs}
    for run in sm["runs"]:
        r = ctx.runs[run]
        cfg, cfg_d, _ = cau_hinh_sort(ctx, ctx.bo_loc[r["bo_loc"]])
        infos = []
        for v in sm["videos"]:
            vd, lan, _ = mo_cache_kiem(ctx, k, det, v)
            info = W.chay_tracker(d / "trackers" / sm["split"] / run, v, KhungDau(vd, n), cfg, None, reg,
                                  ctx.gt["tracker"]["assoc_log"], ctx.gt["tracker"]["pf_diag"])
            lech += kiem_info_sort(info, n)
            infos.append(info)
        ev = W.cham(d / "gt", d / "trackers", sm["split"], run, {v: n for v in sm["videos"]}, d / "trackeval", [d])
        if not ev["huu_han"]:
            lech.append(f"{run}: metric khong huu han")
        kq["run"][run] = {"info": infos, "combined": ev["combined"], "file_metric": ev["file"], "cfg_sha256": sha_json(cfg_d),
                          "assoc_log": ctx.gt["tracker"]["assoc_log"], "pf_diag_truyen": ctx.gt["tracker"]["pf_diag"],
                          "co_pf_diag": (d / "trackers" / sm["split"] / run / "pf_diag").exists(),
                          "co_assoc_log": (d / "trackers" / sm["split"] / run / "assoc_log").exists()}
        print(f"  [smoke] {run}: HOTA {ev['combined'].get('HOTA'):.4f}, {sum(i['n_boxes'] for i in infos)} box")
    h1 = bam_cache(k["nguon"], [det])
    if h1 != h0:
        lech.append("cache doi trong smoke")
    ms = R55.kiem_module_sau(k)
    lech += ms
    kq.update({"khoa": k["phien"], "khoa_sha256": kk["khoa_sha256"], "cache_truoc": h0, "cache_sau": h1,
               "khong_goi_detector": "chi W.mo_cache (doc NPZ R56); khong mo checkpoint, khong predict",
               "lech": lech, "dat": not lech, "luc": bay_gio()})
    W.ghi_json(d / "ket_qua_smoke.json", kq, [d])
    if lech:
        print(f"[smoke] LECH {lech[:5]}")
        return 4
    ghi_marker(ctx, "smoke", d, {"loai": "smoke", "khoa_sha256": kk["khoa_sha256"]}, {"dat": True})
    print(f"SMOKE_ID={d.name}")
    return 0


def bam_cache_khoa(k: dict, dets=None) -> dict:
    """Digest cache theo sha256 GHI TRONG khoa (khong bam lai) de so voi bam_cache hien tai."""
    lk = {f["rel"]: f["sha256"] for f in k["file"]}
    rows = sorted((rel(p), lk.get(rel(p), "THIEU")) for p in file_cache(k["nguon"], dets))
    return {"so_file": len(rows), "sha256_ds": W.sha256_ds(rows)}


def kiem_gt(ctx: Ctx, k: dict) -> str:
    lk = {f["rel"]: f["sha256"] for f in k["file"]}
    rows = []
    for v in ctx.videos:
        for p in (ctx.gt_txt(v), ctx.seqinfo(v)):
            s = sha(p)
            if lk.get(rel(p)) != s:
                raise LoiDauVao(f"GT/seqinfo {rel(p)} khac khoa")
            rows.append((rel(p), s))
    ds = W.sha256_ds(rows)
    if ds != k["nguon"]["gt_sha256_ds"]:
        raise LoiDauVao("GT digest khac khoa")
    return ds


def buoc_cham_nguon(ctx: Ctx, args, rec: list) -> int:
    run = (args.run or [None])[0]
    if run != ctx.gt["cham_nguon"]["run"] or len(args.run or []) != 1:
        raise LoiDauVao(f"cham-nguon chi cho --run {ctx.gt['cham_nguon']['run']} (dinh danh truoc)")
    kk = kiem_khoa_r59(ctx, args.khoa, moi_truong=False)
    if not kk["dat"]:
        raise LoiDauVao(f"kiem-khoa lech: {kk['lech'][:5]}")
    k = kk["_khoa"]
    d = W.tao_phien(ctx.vung("kiem_duong_cham"), None, "cham_nguon")
    rec[0] = d
    W.dat_moi_truong(d)
    lech = kiem_moi_truong_r59(ctx, k)
    if lech:
        raise LoiDauVao(f"moi truong lech khoa: {lech[:5]}")
    dc = k["nguon"]["doi_chung"][run]
    lan = W.tuyet_doi(dc["lan_cham"])
    m = lan_hop_le(lan, "cham")
    gt_ds = kiem_gt(ctx, k)
    trk = d / "trackers" / ctx.split / run / "data"
    trk.mkdir(parents=True)
    for v in ctx.videos:
        b = (lan / "trackers" / ctx.split / run / "data" / f"{v}.txt").read_bytes()
        if W.sha256_bytes(b) != dc["mot"][v]["sha256"]:
            raise LoiDauVao(f"{v}: MOT nguon khac khoa")
        (trk / f"{v}.txt").write_bytes(b)
    ev = W.cham(ctx.thu_muc_gt.parent, d / "trackers", ctx.split, run, dc["seq_info"], d / "trackeval",
                [d, ctx.thu_muc_gt.parent])
    so = {x: {"nguon": dc["file_metric"][x], "r59": ev["file"][x], "trung_byte": dc["file_metric"][x] == ev["file"][x]}
          for x in METRIC_FILE}
    kq = {"run": run, "lan_cham_nguon": dc["lan_cham"], "khoa": k["phien"], "khoa_sha256": kk["khoa_sha256"],
          "so_mot": len(ctx.videos), "mot_sha256_ds": W.sha256_ds((v, sha(trk / f"{v}.txt")) for v in ctx.videos),
          "file": so, "combined_trung": ev["combined"] == dc["combined"] == m["combined"],
          "gt_sha256_ds": gt_ds, "gt_trung_r56": gt_ds == dc["gt_sha256_ds_r56"],
          "khong_chay_tracker": "chi sao byte MOT R56 + W.cham", "luc": bay_gio()}
    lech += [f"{x}: khong trung byte" for x, s in so.items() if not s["trung_byte"]]
    if kq["mot_sha256_ds"] != dc["mot_sha256_ds"]:
        lech.append("ban sao MOT khac nguon")
    if not kq["combined_trung"]:
        lech.append("combined khac nguon")
    lech += R55.kiem_module_sau(k)
    kq.update({"lech": lech, "dat": not lech})
    W.ghi_json(d / "ket_qua_cham_nguon.json", kq, [d])
    if lech:
        print(f"[cham-nguon] LECH {lech[:5]}")
        return 4
    ghi_marker(ctx, "cham-nguon", d, {"loai": "cham-nguon", "khoa_sha256": kk["khoa_sha256"], "run": run}, {"dat": True})
    print("[cham-nguon] 3 file metric trung byte nguon")
    print(f"CHAM_NGUON_ID={d.name}")
    return 0


def buoc_dong_goi_a(ctx: Ctx, args, rec: list) -> int:
    if not (args.smoke and args.cham_nguon):
        raise LoiDauVao("can --khoa --smoke --cham-nguon tuong minh")
    d = W.tao_phien(ctx.vung("chuan_bi"), None, "goi_a")
    rec[0] = d
    W.dat_moi_truong(d)
    kk = kiem_khoa_r59(ctx, args.khoa)
    k = kk["_khoa"]
    W.ghi_json(d / "kiem_khoa_cuoi.json", {x: y for x, y in kk.items() if not x.startswith("_")}, [d])
    sd = R55.kiem_doc_lap(ctx.vung("smoke") / args.smoke, ctx.goc)
    cd = R55.kiem_doc_lap(ctx.vung("kiem_duong_cham") / args.cham_nguon, ctx.goc)
    ms, mc = lan_hop_le(sd, "smoke"), lan_hop_le(cd, "cham-nguon")
    sk, ck = R55.doc_json(sd / "ket_qua_smoke.json", "smoke"), R55.doc_json(cd / "ket_qua_cham_nguon.json", "cham-nguon")
    cb = W.tuyet_doi(k["chuan_bi"])
    cbj = R55.doc_json(cb / "chuan_bi.json", "chuan_bi")
    tj = R55.doc_json(W.tuyet_doi(k["tests"]) / "ket_qua_test.json", "test")
    lech = [] if kk["dat"] else [f"kiem-khoa: {x}" for x in kk["lech"][:10]]
    for ten, m in (("smoke", ms), ("cham-nguon", mc)):
        if m["rang_buoc"].get("khoa_sha256") != kk["khoa_sha256"]:
            lech.append(f"{ten} khong gan voi khoa nay")
    ng = k["nguon"]
    tl = k["k2"]
    smoke_ok = sk.get("dat") and all(len(x["info"]) == 2 and all(i["n_frames"] == 128 for i in x["info"]) for x in sk["run"].values()) \
        and len(sk["run"]) == 2
    ban_do = {
        "4.3-1 phien/khoa R56 nguyen ven, MOT + lan cham theo nguon.json khop hash": {
            "dat": cbj["ket_luan"] == "dat" and len(ng["doi_chung"]) == 6 and (ctx.fixture or bool(ng["khoa_r56"]["lien_ket_cha"]["dat"])),
            "bang_chung": [k["chuan_bi"] + "/chuan_bi.json#nguon", "khoa.json#nguon.doi_chung"]},
        "4.3-2 SORT doc dung manifest cache R56, bam lai khop, khong suy luan": {
            "dat": len(ng["cache"]) == 3 and cbj["cache_truoc"] == cbj["cache_sau"] and sk["cache_truoc"] == sk["cache_sau"],
            "bang_chung": ["chuan_bi.json#cache_truoc/cache_sau", "ket_qua_smoke.json#cache_truoc/cache_sau"]},
        "4.3-3 cung ma phu thuoc + backend nhu khoa R56": {
            "dat": all(x["trang_thai"] in ("khop_khoa_r56", "ma_r59") for x in cbj["module_nap"]) or ctx.fixture,
            "bang_chung": ["chuan_bi.json#module_nap", "chuan_bi.json#backend", "chuan_bi.json#config_py"]},
        "4.3-4 cham lai ByteTrack nguon trung byte 3 file metric": {
            "dat": bool(ck.get("dat")) and all(x["trung_byte"] for x in ck["file"].values()),
            "bang_chung": [rel(cd / "ket_qua_cham_nguon.json")]},
        "K1 parity cau hinh (instance) + config.py": {
            "dat": cbj["ket_luan"] == "dat", "bang_chung": ["chuan_bi.json#cau_hinh_sort/bo_loc_instance/dem_buoc_ctrv/config_py"]},
        "K2 parity dau vao 3 detector x 40 video": {
            "dat": tl["so_dat"] == tl["so_to_hop"] == 3 * len(ctx.videos), "bang_chung": [k["chuan_bi"] + "/k2_parity.csv"]},
        "K3 nguong float32 0.25 / tren / duoi": {"dat": bool(tj.get("dat")), "bang_chung": [k["tests"] + "/ket_qua_test.json"]},
        "K4 frame rong dau/giua/cuoi, giu tuoi 30, xoa > 30, reset ID, D4 trong timeline": {
            "dat": bool(tj.get("dat")) and tl["d4_danh_sach"] == ctx.gt["tap_du_lieu"]["so_anh_D4"],
            "bang_chung": [k["tests"] + "/ket_qua_test.json", "khoa.json#k2"]},
        "K5 cache bat bien (A)": {"dat": cbj["cache_truoc"] == cbj["cache_sau"] and sk["cache_truoc"] == sk["cache_sau"],
                                  "bang_chung": ["chuan_bi.json", "ket_qua_smoke.json"]},
        "K6 khoa moi lien ket khoa R56": {"dat": kk["dat"], "bang_chung": ["kiem_khoa_cuoi.json"]},
        "K7 smoke 2 SORT x 2 video x 128 frame, cham duoc": {"dat": bool(smoke_ok), "bang_chung": [rel(sd / "ket_qua_smoke.json")]},
    }
    lech += [f"{x}: chua dat" for x, y in ban_do.items() if not y["dat"]]
    bang = []
    for goc_d in (cb, W.tuyet_doi(k["tests"]), W.tuyet_doi(k["phien"]), sd, cd):
        for p in sorted(goc_d.rglob("*")):
            if p.is_file() and "_cau_hinh_thu_vien" not in p.parts:
                bang.append({"rel": rel(p), "sha256": sha(p)})
    goi = {"buoc": "dong-goi-a", "luc": bay_gio(), "khoa": k["phien"], "khoa_sha256": kk["khoa_sha256"],
           "chuan_bi": k["chuan_bi"], "tests": k["tests"], "smoke": rel(sd), "cham_nguon": rel(cd),
           "ban_do_4_3_K": ban_do, "lech": lech, "dat": not lech, "so_file": len(bang), "file": bang,
           "chua_chay_full": True}
    W.ghi_json(d / "goi_a.json", goi, [d])
    if lech:
        print(f"[dong-goi-a] LECH {lech[:8]}")
        return 4
    ghi_marker(ctx, "goi-a", d, {"loai": "goi-a", "khoa_sha256": kk["khoa_sha256"]}, {"dat": True})
    print(f"[dong-goi-a] {len(bang)} file; 4.3 + K1-K7 dat")
    print(f"GOI_A_ID={d.name}")
    return 0


# ---------------------------------------------------------------------------
# Buoc B
# ---------------------------------------------------------------------------
def doc_review(p: Path, ten_khoa: str, ten_goi: str) -> tuple[dict, bytes]:
    if not p.is_file():
        raise LoiDauVao(f"khong co review A {p}")
    b = p.read_bytes()
    try:
        j = json.loads(b.decode("utf-8"))
    except ValueError as e:
        raise LoiDauVao(f"review A khong doc duoc: {e}") from e
    v = str(j.get("verdict", j.get("ket_luan", ""))).strip().lower() if isinstance(j, dict) else ""
    if v != "approve":
        raise LoiDauVao(f"review A verdict {v!r} != approve - khong mo phien B")
    txt = b.decode("utf-8")
    if ten_khoa not in txt and ten_goi not in txt:
        raise LoiDauVao(f"review A khong nhac khoa {ten_khoa} hoac goi {ten_goi} - khong rang buoc duoc")
    mo = [x for x in (j.get("issues") or j.get("findings") or []) if isinstance(x, dict)
          and str(x.get("severity", "")).upper() in ("P0", "P1", "BLOCKER")
          and str(x.get("status", "open")).lower() not in ("closed", "resolved", "dong")]
    if mo:
        raise LoiDauVao(f"review A con {len(mo)} van de chan")
    return j, b


def kiem_goi_a(ctx: Ctx, g: Path, khoa_sha: str) -> list[str]:
    """Goi A nguyen ven: marker ngoai hop le (goi_a.json + kiem_khoa_cuoi.json) VA moi file goi_a.json#file (chuan-bi,
    tests, khoa, smoke, cham-nguon) con ton tai trong vung R59, khong qua junction/symlink, khop sha256. [] = dat."""
    try:
        m = lan_hop_le(g, "goi-a")
        j = R55.doc_json(g / "goi_a.json", "goi_a")
    except (LoiNguon, LoiDauVao) as e:
        return [f"{g.name}: {e}"]
    lech = []
    if m["rang_buoc"].get("khoa_sha256") != khoa_sha or j.get("khoa_sha256") != khoa_sha:
        lech.append(f"{g.name}: khong gan voi khoa {khoa_sha[:12]}")
    if not (m.get("dat") and j.get("dat")) or j.get("lech"):
        lech.append(f"{g.name}: goi A khong dat")
    ds = j.get("file") or []
    if not ds or j.get("so_file") != len(ds):
        lech.append(f"{g.name}: so_file {j.get('so_file')} != {len(ds)} file liet ke")
    for x in ds:
        q = Path(x["rel"])
        try:
            p = R55.kiem_doc_lap(q if q.is_absolute() else GOC / q, ctx.goc)
        except LoiDauVao as e:
            lech.append(f"{x['rel']}: {e}")
            continue
        if not p.is_file():
            lech.append(f"{x['rel']}: thieu")
        elif sha(p) != x["sha256"]:
            lech.append(f"{x['rel']}: sha256 lech")
    return lech


def tim_goi_a(ctx: Ctx, khoa_sha: str, ten: str | None) -> Path:
    goc = ctx.vung("chuan_bi")
    ds = [goc / ten] if ten else sorted(x for x in goc.iterdir() if x.is_dir() and x.name.startswith("goi_a_"))
    ok, ly_do = [], []
    for x in ds:
        e = kiem_goi_a(ctx, x, khoa_sha)
        if e:
            ly_do += e[:3]
        else:
            ok.append(x)
    if len(ok) != 1:
        raise LoiDauVao(f"{len(ok)} goi A nguyen ven gan voi khoa (can dung 1; dung --goi-a): {ly_do[:6]}")
    return ok[0]


def buoc_mo_phien(ctx: Ctx, args, rec: list) -> int:
    if not args.review_a:
        raise LoiDauVao("can --review-a <review A that cua luot nay>")
    kk = kiem_khoa_r59(ctx, args.khoa, moi_truong=False)
    if not kk["dat"]:
        raise LoiDauVao(f"kiem-khoa lech: {kk['lech'][:5]}")
    goi = tim_goi_a(ctx, kk["khoa_sha256"], args.goi_a)
    j, b = doc_review(Path(args.review_a), args.khoa, goi.name)
    d = W.tao_phien(ctx.vung("thuc_thi"), None, "phien")
    rec[0] = d / "nhat_ky" / f"mo_phien_{time.strftime('%Y%m%d_%H%M%S')}"
    rec[0].mkdir(parents=True)
    W.dat_moi_truong(rec[0])
    k = kk["_khoa"]
    lech = kiem_moi_truong_r59(ctx, k)
    if bam_cache(k["nguon"]) != bam_cache_khoa(k):
        lech.append("cache khac khoa")
    if lech:
        raise LoiDauVao(f"kiem truoc phien lech: {lech[:5]}")
    (d / "review_a.json").write_bytes(b)
    pj = {"khoa": args.khoa, "khoa_sha256": kk["khoa_sha256"], "goi_a": goi.name, "goi_a_sha256": sha(goi / "goi_a.json"),
          "review_a_goc": str(Path(args.review_a).resolve()), "review_a_sha256": W.sha256_bytes(b),
          "review_a_verdict": "approve", "giao_thuc_sha256": sha(ctx.gt_path), "ma_sha256": sha(FILE_MA),
          "fixture": ctx.fixture, "tao_luc": bay_gio(), "runs": list(RUNS), "so_video": len(ctx.videos)}
    W.ghi_json(d / "phien.json", pj, [d])
    print(f"PHIEN_ID={d.name}")
    return 0


class Phien:
    def __init__(self, ctx: Ctx, ten: str | None):
        if not ten or not W.TEN_HOP_LE.match(ten) or ".." in ten:
            raise LoiDauVao(f"--phien {ten!r} khong hop le / thieu")
        self.ctx = ctx
        self.dir = R55.kiem_doc_lap(ctx.vung("thuc_thi") / ten, ctx.goc)
        self.pj = R55.doc_json(self.dir / "phien.json", "phien.json")

    def xac_minh(self, moi_truong: bool = True) -> dict:
        kk = kiem_khoa_r59(self.ctx, self.pj["khoa"], moi_truong=moi_truong)
        lech = list(kk["lech"])
        if kk["khoa_sha256"] != self.pj["khoa_sha256"] or self.pj["giao_thuc_sha256"] != sha(self.ctx.gt_path) \
                or self.pj["ma_sha256"] != sha(FILE_MA):
            lech.append("khoa / giao thuc / ma khac luc mo phien")
        if sha(self.dir / "review_a.json") != self.pj["review_a_sha256"]:
            lech.append("ban sao review A bi doi")
        g = self.ctx.vung("chuan_bi") / self.pj["goi_a"]
        lg = kiem_goi_a(self.ctx, g, self.pj["khoa_sha256"])
        if lg or sha(g / "goi_a.json") != self.pj["goi_a_sha256"]:
            lech.append(f"goi A khong con nguyen ven: {lg[:3]}")
        if lech:
            raise LoiDauVao(f"phien {self.dir.name}: {lech[:5]}")
        self.khoa = kk["_khoa"]
        return kk

    @property
    def co_so(self) -> dict:
        return {"phien": self.dir.name, "khoa": self.pj["khoa"], "khoa_sha256": self.pj["khoa_sha256"],
                "giao_thuc_sha256": self.pj["giao_thuc_sha256"]}

    def nhat_ky(self, buoc: str) -> Path:
        d = R55.lan_moi(self.dir / "nhat_ky" / buoc, [self.dir])
        W.dat_moi_truong(d)
        return d


def rang_buoc_tracking(ph: Phien, r: dict, cfg_d: dict, v: str) -> dict:
    c = ph.khoa["nguon"]["cache"][r["detector"]]
    t, tr = ph.ctx.gt["tai"], ph.ctx.gt["tracker"]
    return {"loai": "tracking", **ph.co_so, "run": r["run"], "detector": r["detector"], "bo_loc": r["bo_loc"],
            "conf_min": t["conf_min"], "classes": t["classes"], "max_det": t["max_det"], "cfg_sha256": sha_json(cfg_d),
            "manifest_cache_sha256": c["manifest_sha256"], "npz_sha256": c["video"][v]["npz_sha256"], "video": v,
            "fps": tr["fps"], "elapsed_mode": tr["elapsed_mode"], "assoc_log": tr["assoc_log"], "pf_diag": tr["pf_diag"]}


def lay_run(ctx: Ctx, args, cho_phep=None) -> dict:
    if not args.run or len(args.run) != 1 or args.run[0] not in ctx.runs:
        raise LoiDauVao(f"can dung mot --run trong 6 run R59, nhan {args.run}")
    if cho_phep is not None and args.run[0] not in cho_phep:
        raise LoiDauVao(f"--run {args.run[0]} ngoai {list(cho_phep)}")
    return ctx.runs[args.run[0]]


def buoc_full(ctx: Ctx, args, rec: list) -> int:
    ph = Phien(ctx, args.phien)
    ph.xac_minh(moi_truong=False)
    nk = ph.nhat_ky("full")
    rec[0] = nk
    import run_tracking as rt                    # sau dat_moi_truong (nhat_ky): src/ trong sys.path
    k = ph.khoa
    lech = kiem_moi_truong_r59(ctx, k)
    if lech:
        raise LoiDauVao(f"moi truong lech khoa: {lech[:5]}")
    gioi_han = float(args.gioi_han_giay if args.gioi_han_giay is not None else ctx.gt["thuc_thi"]["gioi_han_giay_mac_dinh"])
    t0 = time.time()
    reg = rt.tracker_registry()
    tom = {"moi": [], "tai_su_dung": [], "lan_thu_khac": [], "cache": {}, "gioi_han_giay": gioi_han}
    dung = False
    try:
        for run in RUNS:
            r = ctx.runs[run]
            det = r["detector"]
            h0 = bam_cache(k["nguon"], [det])
            if h0 != bam_cache_khoa(k, [det]):
                raise LoiDauVao(f"cache {det} khac khoa truoc {run}")
            cfg, cfg_d, _ = cau_hinh_sort(ctx, ctx.bo_loc[r["bo_loc"]])
            for v in ctx.videos:
                rb = rang_buoc_tracking(ph, r, cfg_d, v)
                chon, mk, ds = R55.chon_lan(ph.dir / "tracking" / run / v, rb)
                tom["lan_thu_khac"] += [x for x in ds if chon is None or x["lan"] != rel(chon)]
                if chon is not None:
                    tom["tai_su_dung"].append(rel(chon))
                    continue
                if time.time() - t0 > gioi_han:
                    dung = True
                    break
                d = R55.lan_moi(ph.dir / "tracking" / run / v, [ph.dir])
                try:
                    vd, lan, e = mo_cache_kiem(ctx, k, det, v)
                    info = W.chay_tracker(d, v, vd, cfg, None, reg, ctx.gt["tracker"]["assoc_log"], ctx.gt["tracker"]["pf_diag"])
                    loi = kiem_info_sort(info, e["n_frames"]) + R55.kiem_module_sau(k)
                    if loi:
                        raise LoiChay(str(loi[:3]))
                except Exception as x:
                    W.ghi_json(d / "LOI.json", {"loi": f"{type(x).__name__}: {x}", "luc": bay_gio()}, [d])
                    raise
                ngat(ctx, "full-truoc-marker")
                ghi_marker(ctx, "full", d, rb, {"info": info, "lan_cache": rel(lan)})
                tom["moi"].append(rel(d))
                ngat(ctx, "full-sau-marker")
            h1 = bam_cache(k["nguon"], [det])
            tom["cache"][run] = {"truoc": h0, "sau": h1, "trung": h0 == h1}
            if h1 != h0:
                raise LoiChay(f"cache {det} doi trong {run}")
            if dung:
                break
    finally:
        tom["giay"] = round(time.time() - t0, 1)
        W.ghi_json(nk / "tom_tat.json", tom, [nk])
    print(f"[full] moi {len(tom['moi'])}, tai su dung {len(tom['tai_su_dung'])}, {tom['giay']} s")
    if dung:
        print(f"[full] CHUA XONG (het gioi han {gioi_han} s) - goi lai cung lenh / phien")
        return MA_THOAT_CHUA_XONG
    return 0


def mot_cua_run(ph: Phien, r: dict) -> tuple[dict, list[str]]:
    _, cfg_d, _ = cau_hinh_sort(ph.ctx, ph.ctx.bo_loc[r["bo_loc"]])
    out, thieu = {}, []
    for v in ph.ctx.videos:
        chon, mk, _ = R55.chon_lan(ph.dir / "tracking" / r["run"] / v, rang_buoc_tracking(ph, r, cfg_d, v))
        if chon is None:
            thieu.append(v)
        else:
            out[v] = (chon, mk["file"][f"data/{v}.txt"])
    return out, thieu


def rang_buoc_cham(ph: Phien, r: dict, mot: dict) -> dict:
    return {"loai": "cham", **ph.co_so, "run": r["run"], "split": ph.ctx.split, "gt_sha256_ds": ph.khoa["nguon"]["gt_sha256_ds"],
            "mot_lan_ds": W.sha256_ds((v, rel(lan)) for v, (lan, _) in mot.items()),
            "mot_sha256_ds": W.sha256_ds((v, s) for v, (_, s) in mot.items()), "seq_info": ph.khoa["nguon"]["seq_info"],
            "danh_gia": "W.cham (R54): run_trackeval.build_configs + MotChallenge2DBox HOTA/CLEAR/Identity"}


def lan_cham_chon(ph: Phien, r: dict) -> tuple[Path, dict]:
    mot, thieu = mot_cua_run(ph, r)
    if thieu:
        raise LoiDauVao(f"{r['run']}: chua du video ({len(thieu)} thieu)")
    chon, mk, _ = R55.chon_lan(ph.dir / "cham" / r["run"], rang_buoc_cham(ph, r, mot))
    if chon is None:
        raise LoiDauVao(f"{r['run']}: chua co lan cham hoan tat khop bo MOT")
    return chon, mk


def buoc_cham(ctx: Ctx, args, rec: list) -> int:
    r = lay_run(ctx, args)
    ph = Phien(ctx, args.phien)
    ph.xac_minh(moi_truong=False)
    rec[0] = ph.nhat_ky("cham")
    k = ph.khoa
    lech = kiem_moi_truong_r59(ctx, k)
    if lech:
        raise LoiDauVao(f"moi truong lech khoa: {lech[:5]}")
    mot, thieu = mot_cua_run(ph, r)
    if thieu:
        raise LoiDauVao(f"{r['run']}: chua du {len(ctx.videos)} video tracking hoan tat (thieu {thieu[:5]}) - khong cham")
    rb = rang_buoc_cham(ph, r, mot)
    chon, mk, ds = R55.chon_lan(ph.dir / "cham" / r["run"], rb)
    if chon is not None:
        print(f"[cham] {r['run']}: tai su dung {rel(chon)}")
        return 0
    gt_ds = kiem_gt(ctx, k)
    d = R55.lan_moi(ph.dir / "cham" / r["run"], [ph.dir])
    try:
        trk = d / "trackers" / ctx.split / r["run"] / "data"
        trk.mkdir(parents=True)
        for v, (lan, s) in mot.items():
            b = (lan / "data" / f"{v}.txt").read_bytes()
            if W.sha256_bytes(b) != s:
                raise LoiDauVao(f"{v}: MOT lech marker")
            (trk / f"{v}.txt").write_bytes(b)
        ev = W.cham(ctx.thu_muc_gt.parent, d / "trackers", ctx.split, r["run"], k["nguon"]["seq_info"], d / "trackeval",
                    [d, ctx.thu_muc_gt.parent])
        if not ev["huu_han"]:
            raise LoiChay("metric khong huu han")
        if kiem_gt(ctx, k) != gt_ds or any(sha(trk / f"{v}.txt") != s for v, (_, s) in mot.items()):
            raise LoiChay("GT / MOT doi trong luc cham")
        ms = R55.kiem_module_sau(k)
        if ms:
            raise LoiChay(f"module sau cham {ms[:3]}")
        if combined_csv(d / "trackeval" / r["run"] / "combined_metrics.csv") != ev["combined"]:
            raise LoiChay("combined_metrics.csv != combined evaluator")
    except Exception as x:
        W.ghi_json(d / "LOI.json", {"loi": f"{type(x).__name__}: {x}", "luc": bay_gio()}, [d])
        raise
    ghi_marker(ctx, "cham", d, rb, {"combined": ev["combined"], "file_metric": ev["file"], "cau_hinh": ev["cau_hinh"],
                                    "mot_lan": {v: rel(lan) for v, (lan, _) in mot.items()}})
    print(f"[cham] {r['run']}: HOTA {ev['combined']['HOTA']:.5f} -> {rel(d)}")
    return 0


def buoc_lap_tracker(ctx: Ctx, args, rec: list) -> int:
    r = lay_run(ctx, args, LAP_RUNS)
    if args.video != LAP_VIDEO:
        raise LoiDauVao(f"lap-tracker chi cho --video {LAP_VIDEO} (dinh truoc)")
    ph = Phien(ctx, args.phien)
    ph.xac_minh(moi_truong=False)
    rec[0] = ph.nhat_ky("lap_tracker")
    import run_tracking as rt                    # sau dat_moi_truong (nhat_ky): src/ trong sys.path
    k = ph.khoa
    lech = kiem_moi_truong_r59(ctx, k)
    if lech:
        raise LoiDauVao(f"moi truong lech khoa: {lech[:5]}")
    v = args.video
    cfg, cfg_d, _ = cau_hinh_sort(ctx, ctx.bo_loc[r["bo_loc"]])
    chon, mk, _ = R55.chon_lan(ph.dir / "tracking" / r["run"] / v, rang_buoc_tracking(ph, r, cfg_d, v))
    if chon is None:
        raise LoiDauVao(f"{r['run']}/{v}: chua co lan tracking hoan tat")
    h0 = bam_cache(k["nguon"], [r["detector"]])
    d = R55.lan_moi(ph.dir / "lap_tracker" / r["run"] / v, [ph.dir])
    vd, lan, e = mo_cache_kiem(ctx, k, r["detector"], v)
    info = W.chay_tracker(d, v, vd, cfg, None, rt.tracker_registry(), ctx.gt["tracker"]["assoc_log"], ctx.gt["tracker"]["pf_diag"])
    moi = sha(d / "data" / f"{v}.txt")
    goc = mk["file"][f"data/{v}.txt"]
    ms = R55.kiem_module_sau(k) + kiem_info_sort(info, e["n_frames"])
    h1 = bam_cache(k["nguon"], [r["detector"]])
    kq = {"run": r["run"], "video": v, "lan_goc": rel(chon), "sha256_goc": goc, "sha256_lap": moi, "trung_byte": moi == goc,
          "frame": [1, info["n_frames"]], "cache_truoc": h0, "cache_sau": h1, "kiem_sau": ms}
    W.ghi_json(d / "ket_qua_lap.json", kq, [d])
    if ms or not kq["trung_byte"] or h0 != h1:
        print(f"[lap-tracker] LECH {kq}")
        return 4
    ghi_marker(ctx, "lap", d, {"loai": "lap-tracker", **ph.co_so, "run": r["run"], "video": v}, {"trung_byte": True})
    print(f"[lap-tracker] {r['run']}/{v}: MOT trung byte")
    return 0


def buoc_cham_lai(ctx: Ctx, args, rec: list) -> int:
    r = lay_run(ctx, args, LAP_RUNS)
    ph = Phien(ctx, args.phien)
    ph.xac_minh(moi_truong=False)
    rec[0] = ph.nhat_ky("cham_lai")
    k = ph.khoa
    lech = kiem_moi_truong_r59(ctx, k)
    if lech:
        raise LoiDauVao(f"moi truong lech khoa: {lech[:5]}")
    goc_lan, mk = lan_cham_chon(ph, r)
    kiem_gt(ctx, k)
    d = R55.lan_moi(ph.dir / "cham_lai" / r["run"], [ph.dir])
    ev = W.cham(ctx.thu_muc_gt.parent, goc_lan / "trackers", ctx.split, r["run"], k["nguon"]["seq_info"], d / "trackeval",
                [d, goc_lan / "trackers", ctx.thu_muc_gt.parent])
    tt, _, _ = R55.danh_gia_lan(goc_lan, mk["rang_buoc"])
    ms = R55.kiem_module_sau(k)
    kq = {"run": r["run"], "lan_cham_goc": rel(goc_lan), "file_goc": mk["file_metric"], "file_lai": ev["file"],
          "file_trung_byte": ev["file"] == mk["file_metric"], "combined_trung": ev["combined"] == mk["combined"],
          "lan_goc_con_nguyen": tt == "hop_le", "kiem_module_sau": ms}
    W.ghi_json(d / "ket_qua_cham_lai.json", kq, [d])
    if ms or not (kq["file_trung_byte"] and kq["combined_trung"] and kq["lan_goc_con_nguyen"]):
        print(f"[cham-lai] LECH {kq}")
        return 4
    ghi_marker(ctx, "cham-lai", d, {"loai": "cham-lai", **ph.co_so, "run": r["run"]}, {"trung_byte": True})
    print(f"[cham-lai] {r['run']}: 3 file metric trung byte")
    return 0


def lan_kiem_hop_le(goc: Path, loai: str, run: str) -> Path | None:
    for x in sorted(goc.iterdir()) if goc.is_dir() else []:
        try:
            m = lan_hop_le(x, loai)
        except LoiNguon:
            continue
        if m["rang_buoc"].get("run") == run:
            return x
    return None


def buoc_hau_kiem(ctx: Ctx, args, rec: list) -> int:
    ph = Phien(ctx, args.phien)
    ph.xac_minh(moi_truong=False)
    nk = ph.nhat_ky("hau_kiem")
    rec[0] = nk
    k = ph.khoa
    lech = kiem_moi_truong_r59(ctx, k)
    dem = {"don_vi": 0, "cham": 0, "lap": 0, "cham_lai": 0}
    anh = {}
    for run in RUNS:
        r = ctx.runs[run]
        mot, thieu = mot_cua_run(ph, r)
        lech += [f"{run}/{v}: thieu don vi hop le" for v in thieu]
        dem["don_vi"] += len(mot)
        anh[run] = 0
        for v, (lan, s) in mot.items():
            mk = R55.doc_marker(lan)
            anh[run] += int(mk["info"]["n_frames"])
            if mk["info"]["n_buoc_timing"] != k["nguon"]["seq_info"][v]:
                lech.append(f"{run}/{v}: timeline lech")
        try:
            lan_cham_chon(ph, r)
            dem["cham"] += 1
        except LoiDauVao as e:
            lech.append(str(e))
    for run in LAP_RUNS:
        if lan_kiem_hop_le(ph.dir / "lap_tracker" / run / LAP_VIDEO, "lap-tracker", run):
            dem["lap"] += 1
        else:
            lech.append(f"{run}: chua co lap-tracker hop le")
        if lan_kiem_hop_le(ph.dir / "cham_lai" / run, "cham-lai", run):
            dem["cham_lai"] += 1
        else:
            lech.append(f"{run}: chua co cham-lai hop le")
    so_anh = ctx.gt["tap_du_lieu"]["so_anh"]
    lech += [f"{run}: {n} anh != {so_anh}" for run, n in anh.items() if n != so_anh]
    if bam_cache(k["nguon"]) != bam_cache_khoa(k):
        lech.append("cache khac khoa")
    try:
        kiem_gt(ctx, k)
    except LoiDauVao as e:
        lech.append(str(e))
    kq = {"phien": ph.dir.name, "dem": dem, "anh_moi_run": anh, "lech": lech, "dat": not lech, "luc": bay_gio(),
          "ky_vong": {"don_vi": 6 * len(ctx.videos), "cham": 6, "lap": 2, "cham_lai": 2}}
    if dem != kq["ky_vong"]:
        lech.append(f"dem {dem} != {kq['ky_vong']}")
        kq["dat"] = False
    W.ghi_json(nk / "hau_kiem.json", kq, [nk])
    if not kq["dat"]:
        print(f"[hau-kiem] LECH {lech[:8]}")
        return 4
    ghi_marker(ctx, "hau-kiem", nk, {"loai": "hau-kiem", **ph.co_so}, {"dat": True})
    print(f"[hau-kiem] dat: {dem}")
    return 0


def buoc_tong_hop(ctx: Ctx, args, rec: list) -> int:
    ph = Phien(ctx, args.phien)
    ph.xac_minh(moi_truong=False)
    k = ph.khoa
    hk = [x for x in sorted((ph.dir / "nhat_ky" / "hau_kiem").iterdir()) if (x / MARKER).is_file()] \
        if (ph.dir / "nhat_ky" / "hau_kiem").is_dir() else []
    hk_ok = None
    for x in hk:
        try:
            m = lan_hop_le(x, "hau-kiem")
        except LoiNguon:
            continue
        if m.get("dat") and m["rang_buoc"].get("khoa_sha256") == ph.pj["khoa_sha256"]:
            hk_ok = x
    if hk_ok is None:
        raise LoiDauVao("chua co hau-kiem dat trong phien")
    d = W.tao_phien(ctx.vung("tong_hop"), ph.dir.name, "tong_hop")
    rec[0] = d
    W.dat_moi_truong(d)
    mt = ctx.gt["metric"]
    ds = mt["doi_chieu_nguon_r56"]
    bang56 = {x["run"]: x for x in doc_csv(ctx.bang56)}
    if sha(ctx.bang56) != ctx.gt["nguon_r56"]["bang_r56_sha256"]:
        raise LoiDauVao("bang R56 doi")
    rows, kc, nguon, lech = [], [], {}, []
    for run in RUNS:
        r = ctx.runs[run]
        lan_s, mk_s = lan_cham_chon(ph, r)
        cs = combined_csv(lan_s / "trackeval" / run / "combined_metrics.csv")
        if cs != mk_s["combined"]:
            lech.append(f"{run}: combined_metrics.csv != marker")
        dc = k["nguon"]["doi_chung"][r["doi_chung"]]
        lan_b = W.tuyet_doi(dc["lan_cham"])
        pb = lan_b / "trackeval" / r["doi_chung"] / "combined_metrics.csv"
        if sha(pb) != dc["file_metric"]["combined_metrics.csv"]:
            lech.append(f"{r['doi_chung']}: combined_metrics.csv khac khoa")
        cb = combined_csv(pb)
        if cb != dc["combined"]:
            lech.append(f"{r['doi_chung']}: combined khac khoa")
        row = {"detector": r["detector"], "bo_loc": r["bo_loc"], "run_sort": run, "run_bytetrack": r["doi_chung"]}
        sai = {}
        for m in METRIC_THU_TU:
            b, s = cb[m], cs[m]
            if m in METRIC_TY_LE:
                row[f"{m}_bytetrack"], row[f"{m}_sort"], row[f"{m}_hieu_pp"] = round(b * 100, 3), round(s * 100, 3), round((b - s) * 100, 3)
                ref = float(bang56[r["doi_chung"]][ds["cot_bang_r56"][m]])
                sai[m] = abs(round(b * 100, 3) - ref)
            else:
                row[f"{m}_bytetrack"], row[f"{m}_sort"], row[f"{m}_hieu"] = int(b), int(s), int(b) - int(s)
                ref = float(bang56[r["doi_chung"]][ds["cot_bang_r56"][m]])
                sai[m] = abs(int(b) - ref)
            if sai[m] > ds["dung_sai"][m] + 1e-12:
                lech.append(f"{r['doi_chung']} {m}: sai so {sai[m]} > {ds['dung_sai'][m]}")
        rows.append(row)
        kc.append({"run_bytetrack": r["doi_chung"], "sai_so_bang_r56": sai})
        nguon[run] = {"cham_sort": rel(lan_s), "combined_sort_sha256": sha(lan_s / "trackeval" / run / "combined_metrics.csv"),
                      "cham_bytetrack": dc["lan_cham"], "combined_bytetrack_sha256": sha(pb)}
    W.ghi_bang_csv(d / "doi_chieu_6_cap.csv", rows, [d])
    W.ghi_json(d / "nguon.json", {"phien": ph.dir.name, "khoa": k["phien"], "hau_kiem": rel(hk_ok), "run": nguon,
                                  "bang_r56": rel(ctx.bang56)}, [d])
    W.ghi_json(d / "kiem_chung.json", {"doi_chieu_r56": kc, "dung_sai": ds["dung_sai"], "lech": lech, "dat": not lech,
                                       "don_vi": mt["don_vi"], "lam_tron": mt["lam_tron_bang"]}, [d])
    s = "".join(f"{sha(d / f)}  {f}\n" for f in ("doi_chieu_6_cap.csv", "nguon.json", "kiem_chung.json"))
    (d / "SHA256SUMS").write_text(s, encoding="utf-8")
    print(f"[tong-hop] {rel(d)}: {len(rows)} cap; {'dat' if not lech else 'LECH ' + str(lech[:5])}")
    return 0 if not lech else 4


# ---------------------------------------------------------------------------
# CLI + nhat ky
# ---------------------------------------------------------------------------
class Tee(io.TextIOBase):
    def __init__(self, goc):
        self.goc, self.buf = goc, io.StringIO()

    def write(self, s):
        self.goc.write(s)
        self.buf.write(s)
        return len(s)

    def flush(self):
        self.goc.flush()


HAM = {"chuan-bi": buoc_chuan_bi, "khoa": buoc_khoa, "kiem-khoa": buoc_kiem_khoa, "smoke": buoc_smoke,
       "cham-nguon": buoc_cham_nguon, "dong-goi-a": buoc_dong_goi_a, "mo-phien": buoc_mo_phien, "full": buoc_full,
       "cham": buoc_cham, "lap-tracker": buoc_lap_tracker, "cham-lai": buoc_cham_lai, "hau-kiem": buoc_hau_kiem,
       "tong-hop": buoc_tong_hop}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buoc", required=True, choices=list(HAM))
    for a in ("--chuan-bi", "--tests", "--khoa", "--smoke", "--cham-nguon", "--phien", "--review-a", "--goi-a", "--video"):
        ap.add_argument(a)
    ap.add_argument("--run", nargs="+")
    ap.add_argument("--gioi-han-giay", type=float)
    args = ap.parse_args(argv)
    out, err = Tee(sys.stdout), Tee(sys.stderr)
    goc_out, goc_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    rec: list = [None]
    t0, ma = bay_gio(), 0
    try:
        ctx = Ctx()
        ma = HAM[args.buoc](ctx, args, rec)
    except W.LoiR54 as e:
        print(f"[LOI {type(e).__name__}] {e}", file=sys.stderr)
        ma = e.ma
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        ma = 4
    finally:
        sys.stdout, sys.stderr = goc_out, goc_err
        d = rec[0]
        if d is not None and Path(d).is_dir():
            p = Path(d) / "cli_log.json"
            i = 1
            while p.exists():
                p, i = Path(d) / f"cli_log_{i}.json", i + 1
            p.write_text(json.dumps({"argv": sys.argv if argv is None else argv, "bat_dau": t0, "ket_thuc": bay_gio(),
                                     "exit": ma, "stdout": out.buf.getvalue(), "stderr": err.buf.getvalue(),
                                     "python": sys.executable, "fixture": bool(os.environ.get(ENV_FIXTURE))},
                                    ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[exit {ma}]")
    return ma


if __name__ == "__main__":
    sys.exit(main())
