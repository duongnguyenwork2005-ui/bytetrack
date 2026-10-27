"""
audit_protocol.py -- R51A: kiem ke run da co + kiem 572 anh dev12 khong co <frame> (CHI DOC).

Khong chay tracking, khong chay TrackEval tren du lieu that, khong sua file nao ngoai
results/phase2/r51_controlled/protocol/. Moi so lieu trong 01_giao_thuc.md mục 3-5 lay tu day.

Phan 1 - frame khong nhan (dev12):
    nguon XML (data/extracted/DETRAC-Train-Annotations-XML/<v>.xml, the <frame num>)
    anh tren dia (config.find_images_root()/<v>/img*.jpg)
    GT da chuan bi (data/processed/DETRAC-all/<v>/gt/gt.txt) + seqinfo.ini (seqLength)
    -> danh sach frame co anh nhung khong co <frame>, gom thanh doan, phan loai
       "co xe xuyen qua" (cung track id GT o frame co nhan ngay truoc VA ngay sau doan)
       -> frame KHONG NHAN, khong the la am tinh.
    -> so box tracker cua tung run nam tren cac frame do (MotChallenge2DBox tinh het la FP).
Phan 2 - kiem ke run: run_meta / eval_meta / hash dau ra (quy uoc run_fingerprint: .txt chuan hoa LF)
    / hash ma hien tai / so khop dau ra giua cac run cung cau hinh.

Chay: python -X utf8 src/r51_controlled/audit_protocol.py
"""
from __future__ import annotations

import configparser
import csv
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
import run_fingerprint as rf  # noqa: E402

ROOT = config.PROJECT_ROOT
OUT = ROOT / "results" / "phase2" / "r51_controlled" / "protocol"
SPLIT = "DETRAC-all"
MANIFEST = ROOT / "results" / "phase2" / "manifests" / "detrac_dev.json"
XML_DIR = config.EXTRACTED_DIR / "DETRAC-Train-Annotations-XML"
GT_ROOT = config.PROCESSED_DIR / SPLIT
OLD_TRK = config.PROCESSED_DIR / "trackers" / SPLIT
KC48_TRK = ROOT / "results" / "phase2" / "kc48_pf_dev" / "_processed" / "trackers" / SPLIT
EVAL_ROOT = config.RESULTS_DIR / "trackeval" / SPLIT

IMG_RE = re.compile(r"^img(\d+)\.jpg$")


# ---------------------------------------------------------------------------
# Ham thuan (co test trong test_protocol.py)
# ---------------------------------------------------------------------------
def segments(frames: list[int]) -> list[tuple[int, int]]:
    """Gom danh sach frame (khong can sap xep) thanh cac doan lien tiep [a, b]."""
    out: list[tuple[int, int]] = []
    for f in sorted(set(frames)):
        if out and f == out[-1][1] + 1:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


def classify_segment(a: int, b: int, labeled: set[int], ids_at: dict[int, set[int]]) -> dict:
    """Phan loai doan frame khong nhan [a, b] theo frame co nhan ngay truoc (a-1) va ngay sau (b+1).

    vi_tri: dau (khong co frame nhan truoc) | cuoi (khong co frame nhan sau) | giua.
    n_xuyen: so track id GT co mat o CA a-1 VA b+1 -> xe ton tai trong khung hinh suot doan
    (track GT cua UA-DETRAC lien tuc, khong co ID tai sinh). n_xuyen > 0 => doan la frame
    KHONG NHAN, khong phai am tinh. n_xuyen = 0 => khong ket luan duoc tu GT.
    """
    prev_f, next_f = a - 1, b + 1
    has_prev, has_next = prev_f in labeled, next_f in labeled
    vi_tri = "giua" if (has_prev and has_next) else ("dau" if not has_prev else "cuoi")
    ids_prev = ids_at.get(prev_f, set()) if has_prev else set()
    ids_next = ids_at.get(next_f, set()) if has_next else set()
    xuyen = ids_prev & ids_next
    return {"a": a, "b": b, "do_dai": b - a + 1, "vi_tri": vi_tri,
            "n_gt_truoc": len(ids_prev), "n_gt_sau": len(ids_next), "n_xuyen": len(xuyen),
            "loai": "khong_nhan_co_xe" if xuyen else "khong_ket_luan_tu_gt"}


def track_spans(ids_at: dict[int, set[int]]) -> dict[int, tuple[int, int]]:
    """id GT -> (frame dau, frame cuoi)."""
    span: dict[int, tuple[int, int]] = {}
    for f in sorted(ids_at):
        for i in ids_at[f]:
            span[i] = (span[i][0], f) if i in span else (f, f)
    return span


def boundary_check(a: int, b: int, ids_at: dict[int, set[int]], span: dict[int, tuple[int, int]]) -> dict:
    """Track GT ngay truoc doan co KET THUC tai a-1 va track ngay sau co BAT DAU tai b+1 khong."""
    prev_ids, next_ids = ids_at.get(a - 1, set()), ids_at.get(b + 1, set())
    return {"gt_truoc_ket_thuc_tai_a_1": sum(1 for i in prev_ids if span[i][1] == a - 1),
            "gt_sau_bat_dau_tai_b_1": sum(1 for i in next_ids if span[i][0] == b + 1)}


def ioa(box, region) -> float:
    """Ti le dien tich box (xywh) nam trong region (xywh)."""
    x1, y1 = max(box[0], region[0]), max(box[1], region[1])
    x2 = min(box[0] + box[2], region[0] + region[2]); y2 = min(box[1] + box[3], region[1] + region[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(1e-9, box[2] * box[3])


def mask_rows(rows, labeled: set[int]):
    """Quy tac de xuat R1: bo dong tracker (cot 0 = frame) nam o frame khong co nhan. GT giu nguyen."""
    return [r for r in rows if int(float(r[0])) in labeled]


def clear_after_mask(raw: dict, n_drop: int) -> dict:
    """CLEAR sau khi bo n_drop box tracker o frame khong co dong GT -- CHINH XAC, khong can chay lai.

    Frame khong co GT: TrackEval CLEAR cong toan bo box vao CLR_FP va bo qua frame (khong ghep, khong
    doi IDSW); ghep o frame co GT chi phu thuoc frame co GT truoc do. Nen bo cac box nay: FP giam dung
    n_drop, TP/FN/IDSW giu nguyen.
    HOTA/AssA/IDF1 KHONG suy duoc theo cach nay: box o frame khong GT con nam trong FPA cua ID tracker
    (AssA giam, xem test_evaluator_dem_fp_o_frame_khong_gt), trong diem ghep toan cuc cua HOTA va trong
    ghep ID cua IDF1. So chinh xac chi co khi cham lai (buoc B).
    """
    gt = raw["CLR_TP"] + raw["CLR_FN"]
    mota0 = (raw["CLR_TP"] - raw["CLR_FP"] - raw["IDSW"]) / max(1, gt)
    mota1 = (raw["CLR_TP"] - (raw["CLR_FP"] - n_drop) - raw["IDSW"]) / max(1, gt)
    return {"CLR_FP_cu": raw["CLR_FP"], "CLR_FP_moi_chinh_xac": raw["CLR_FP"] - n_drop,
            "MOTA_cu": mota0, "MOTA_moi_chinh_xac": mota1}


def sum_raw(per_video: dict) -> dict:
    """Cong truong tho tren cac video (AssA lay trung binh co trong so HOTA_TP, nhu TrackEval)."""
    import numpy as np
    keys_vec = ("HOTA_TP", "HOTA_FN", "HOTA_FP")
    out = {k: np.zeros(19) for k in keys_vec}
    ass_num = np.zeros(19)
    for d in per_video.values():
        for k in keys_vec:
            out[k] += np.asarray(d[k], float)
        ass_num += np.asarray(d["AssA"], float) * np.asarray(d["HOTA_TP"], float)
    out["AssA"] = ass_num / np.maximum(1e-10, out["HOTA_TP"])
    for k in ("CLR_TP", "CLR_FN", "CLR_FP", "IDSW"):
        out[k] = sum(int(d[k]) for d in per_video.values())
    return out


#: Truong bat buoc cua bang chung tai dung (01_giao_thuc.md muc 4.1). Thieu truong nao = KHONG DAT.
CACHE_KEYS = ("cache_id", "meta_sha256", "npz_sha256", "weights_sha256")
BACKEND_KEYS = ("backend", "ultralytics", "numpy", "scipy", "lap")


def hash_dau_ra_khop(per_video: dict, vh: dict, videos: list[str]) -> bool | None:
    """Hash dau ra hien tai (vh, sha256 day du) khop eval_meta.tracking.per_video (16 ky tu dau).

    None khi eval_meta khong co per_video. True CHI KHI moi video cua manifest deu co trong eval_meta,
    co file dau ra, va hash khop; thieu video / thua video / hash rong deu la False (R51A-F1).
    """
    if not per_video:
        return None
    if set(per_video) != set(videos):
        return False
    return all(vh.get(v, "MISSING") != "MISSING" and len(per_video[v] or "") == 16
               and vh[v].startswith(per_video[v]) for v in videos)


def kiem_provenance(rm: dict | None, em: dict | None, videos: list[str], vh: dict, rel_mods: tuple,
                    mods_now: dict, backend_now: dict, cache_now: dict, gt_now: dict, yaml_sha12_now: str | None,
                    config_khop: bool | None) -> dict:
    """Cong tai dung (01_giao_thuc.md muc 4.1, tieu chi 1-6). THIEU bang chung = KHONG DAT, khong bo qua.

    Moi dieu kien kiem ca SU CO MAT cua truong lan GIA TRI. Tra ve {"tai_dung_du_dieu_kien", "ly_do_khong_dat"}.
    """
    loi: list[str] = []

    def can(dk, ly_do):
        if not dk:
            loi.append(ly_do)

    rm = rm or {}
    e = em or {}
    can(bool(rm), "thieu run_meta")
    can(em is not None, "thieu eval_meta")
    can(rm.get("status") == "complete", f"run_meta.status={rm.get('status')!r}")
    # (1) du video, dung danh sach manifest
    can(sorted(rm.get("videos") or []) == sorted(videos), "run_meta.videos khac manifest")
    trk = e.get("tracking") or {}
    per = trk.get("per_video") or {}
    can(set(per) == set(videos),
        f"eval_meta.tracking.per_video thieu {sorted(set(videos) - set(per))} thua {sorted(set(per) - set(videos))}")
    can(trk.get("n_missing") == 0, f"eval_meta.tracking.n_missing={trk.get('n_missing')!r}")
    thieu_file = [v for v in videos if vh.get(v, "MISSING") == "MISSING"]
    can(not thieu_file, f"thieu file dau ra {thieu_file}")
    can(hash_dau_ra_khop(per, vh, videos) is True, "hash dau ra khong khop eval_meta tren du 12 video")
    # (2) cache: du truong, trung cache hien tai tren dia, cung cache_id voi run_meta
    c = e.get("cache") or {}
    thieu_c = [k for k in CACHE_KEYS if not c.get(k)]
    can(not thieu_c, f"eval_meta.cache thieu {thieu_c}")
    can(not thieu_c and all(c.get(k) == cache_now.get(k) for k in CACHE_KEYS), "cache eval_meta khac cache hien tai")
    can(rm.get("cache_id") == c.get("cache_id"), "run_meta.cache_id khac eval_meta")
    # (3) YAML luc chay = YAML hien tai
    y = rm.get("tracker_yaml_sha256_12")
    can(bool(y) and y == yaml_sha12_now, f"yaml luc chay {y!r} khac hien tai {yaml_sha12_now!r}")
    # (4) ma + backend: du module lien quan, trung HEAD, trung luc cham; backend du truong va trung hien tai
    code_run = rm.get("code") or {}
    mods_run = code_run.get("modules") or {}
    mods_eval = (e.get("code") or {}).get("modules") or {}
    thieu_m = [m for m in rel_mods if m not in mods_run]
    can(not thieu_m, f"run_meta.code thieu module {thieu_m}")
    lech_m = [m for m in rel_mods if m in mods_run and mods_run[m] != mods_now.get(m)]
    can(not lech_m, f"module lech HEAD {lech_m}")
    lech_e = [m for m in rel_mods if mods_eval.get(m) != mods_run.get(m)]
    can(not lech_e, f"module luc cham khac luc chay {lech_e}")
    be = code_run.get("backend") or {}
    thieu_b = [k for k in BACKEND_KEYS if not be.get(k)]
    can(not thieu_b, f"run_meta.code.backend thieu {thieu_b}")
    lech_b = [k for k in BACKEND_KEYS if be.get(k) and be.get(k) != backend_now.get(k)]
    can(not lech_b, f"backend lech hien tai {lech_b}")
    can(config_khop is True, "config.py luc chay khong xac minh duoc hoac khac HEAD/cay lam viec")
    # (6) GT: du dau van tay, khong thieu file, du video, trung GT hien tai
    g = e.get("gt") or {}
    can(bool(g.get("gt_sha256")) and bool(g.get("seqinfo_sha256")), "eval_meta thieu dau van tay GT")
    can(g.get("n_missing_gt") == 0 and g.get("n_missing_seqinfo") == 0, "eval_meta ghi thieu GT/seqinfo")
    can(set(g.get("gt_per_video") or {}) == set(videos), "eval_meta.gt.gt_per_video khac manifest")
    can(bool(g.get("gt_sha256")) and g.get("gt_sha256") == gt_now.get("gt_sha256")
        and g.get("seqinfo_sha256") == gt_now.get("seqinfo_sha256"), "GT eval_meta khac GT hien tai")
    return {"tai_dung_du_dieu_kien": not loi, "ly_do_khong_dat": loi}


#: Cot cua bang thong so bo loc hieu luc (gia tri None = khong ap dung cho loai tracker do)
BL_COLS = ("bl_lop", "bl_dt_buoc", "bl_std_weight_position", "bl_std_weight_velocity",
           "bl_ctrv_std_theta", "bl_ctrv_std_omega", "bl_ctrv_init_std_theta", "bl_ctrv_init_std_omega",
           "bl_ctrv_heading_std_after_init", "bl_ctrv_min_speed_for_heading", "bl_ctrv_omega_eps",
           "bl_ukf_alpha", "bl_ukf_beta", "bl_ukf_kappa",
           "bl_pf_N", "bl_pf_seed_yaml", "bl_pf_ess", "bl_pf_estimate", "bl_pf_roughening",
           "bl_pf_degenerate_maha2", "bl_pf_K1",
           "bl_sort_iou", "bl_sort_max_age", "bl_sort_min_hits", "bl_sort_det_thresh", "bl_khoa_yaml_bo_loc")


def bo_loc_hieu_luc(ttype: str, cfg: dict) -> dict:
    """Tham so bo loc HIEU LUC cua mot cau hinh tracker (R51A-F2).

    Dung lai DUNG duong tao bo loc cua tracker (get_kalmanfilter / SORT._new_filter) tren namespace cau
    hinh, roi doc thuoc tinh cua instance -- khong suy tu YAML. Khoa vang mat trong YAML lay mac dinh
    config.py (CTRV/UKF) hoac ultralytics (XYAH) dung nhu luc chay; provenance config.py kiem rieng.
    bl_dt_buoc: buoc thoi gian NOI BO cua bo loc (1 = mot frame), khac dt danh nghia 0,04 s giua hai anh.
    Khong tao tracker, khong doc du lieu, khong doi gi.
    """
    from types import SimpleNamespace
    ns = SimpleNamespace(**cfg)
    out = dict.fromkeys(BL_COLS)
    out["bl_khoa_yaml_bo_loc"] = ";".join(sorted(k for k in cfg if k.startswith(("ctrv_", "pf_", "cv_", "ukf_",
                                                                                 "sort_", "det_thresh"))))
    kf = None
    if ttype.startswith("sort_"):
        import sort_tracker
        s = sort_tracker.SORT(ns)
        out.update(bl_sort_iou=s.iou_threshold, bl_sort_max_age=s.max_age, bl_sort_min_hits=s.min_hits,
                   bl_sort_det_thresh=s.det_thresh)
        f = s._new_filter()
        kf = getattr(f, "kf", f)
    elif ttype == "bytetrack":
        from ul_backend import KalmanFilterXYAH
        kf = KalmanFilterXYAH()
    else:
        import tracker_ctrv as tc
        cls = tc.CTRV_TRACKERS[ttype]
        kf = cls.get_kalmanfilter(SimpleNamespace(args=ns, track_class=cls.track_class))
    out["bl_lop"] = type(kf).__name__
    if hasattr(kf, "_motion_mat") and not hasattr(kf, "omega_eps"):
        out["bl_dt_buoc"] = float(kf._motion_mat[0, 4])           # KalmanFilterXYAH: x' = x + dt*vx
    elif hasattr(kf, "dt"):
        out["bl_dt_buoc"] = float(kf.dt)
    # SortKF7 (sort_orig, 7-D upstream) khong co cac truong nay -> None (tham chieu, khong vao so sanh)
    out["bl_std_weight_position"] = getattr(kf, "_std_weight_position", None)
    out["bl_std_weight_velocity"] = getattr(kf, "_std_weight_velocity", None)
    if hasattr(kf, "omega_eps"):
        out.update(bl_ctrv_std_theta=kf._std_theta, bl_ctrv_std_omega=kf._std_omega,
                   bl_ctrv_init_std_theta=kf._init_std_theta, bl_ctrv_init_std_omega=kf._init_std_omega,
                   bl_ctrv_heading_std_after_init=kf._heading_std_after_init,
                   bl_ctrv_min_speed_for_heading=kf._min_speed_for_heading, bl_ctrv_omega_eps=kf.omega_eps)
    if hasattr(kf, "kappa") and hasattr(kf, "alpha"):
        out.update(bl_ukf_alpha=kf.alpha, bl_ukf_beta=kf.beta, bl_ukf_kappa=kf.kappa)
    if hasattr(kf, "num_particles"):
        out.update(bl_pf_N=kf.num_particles, bl_pf_seed_yaml=kf.seed, bl_pf_ess=kf.resample_ess_ratio,
                   bl_pf_estimate=kf.estimate, bl_pf_roughening=kf.roughening,
                   bl_pf_degenerate_maha2=kf.degenerate_maha2, bl_pf_K1=kf.optimal_proposal)
    return out


def _git_last(path: str) -> str | None:
    import subprocess
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h %cs %s", "--", path], cwd=ROOT, capture_output=True,
                              text=True, encoding="utf-8", check=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


CONFIG_PY = "src/config.py"
_CONFIG_CACHE: dict[str, str | None] = {}


def config_py_sha256(rev: str | None) -> str | None:
    """sha256 (chuan hoa LF) cua src/config.py tai mot commit git; rev=None -> file tren cay lam viec.

    config.py chua mac dinh CTRV/UKF nhung KHONG nam trong CODE_FILES cua run_fingerprint, nen provenance
    cua no kiem rieng: ban tai commit luc chay phai trung HEAD va cay lam viec. Loi git -> None.
    """
    import hashlib
    import subprocess
    if rev in _CONFIG_CACHE:
        return _CONFIG_CACHE[rev]
    try:
        if rev is None:
            b = (ROOT / CONFIG_PY).read_bytes()
        else:
            b = subprocess.run(["git", "show", f"{rev}:{CONFIG_PY}"], cwd=ROOT, capture_output=True,
                               check=True).stdout
        h = hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()
    except Exception:  # noqa: BLE001
        h = None
    _CONFIG_CACHE[rev] = h
    return h


# ---------------------------------------------------------------------------
# Doc nguon
# ---------------------------------------------------------------------------
def xml_frames(video: str) -> list[int]:
    root = ET.parse(XML_DIR / f"{video}.xml").getroot()
    return [int(f.get("num")) for f in root.findall("frame")]


def image_frames(img_root: Path, video: str) -> list[int]:
    d = img_root / video
    return sorted(int(m.group(1)) for p in d.iterdir() if (m := IMG_RE.match(p.name)))


def gt_ids_by_frame(video: str) -> dict[int, set[int]]:
    ids: dict[int, set[int]] = {}
    with open(GT_ROOT / video / "gt" / "gt.txt", encoding="utf-8") as fh:
        for line in fh:
            p = line.split(",")
            if len(p) >= 2:
                ids.setdefault(int(p[0]), set()).add(int(p[1]))
    return ids


def seq_length(video: str) -> int | None:
    cp = configparser.ConfigParser()
    cp.read(GT_ROOT / video / "seqinfo.ini")
    try:
        return int(cp["Sequence"]["seqLength"])
    except KeyError:
        return None


def tracker_frames(path: Path) -> list[int]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(int(float(line.split(",", 1)[0])))
    return out


#: Run neo de mo ta box tracker trong doan khong nhan (ByteTrack EKF, dieu kien A)
ANCHOR_DIR = "p2-kc48_pf_dev-A-ekf"


def read_regions(video: str) -> list[list[float]]:
    p = GT_ROOT / video / "gt" / "ignored_regions.txt"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as fh:
        return [[float(x) for x in line.split(",")[:4]] for line in fh if line.strip()]


def read_tracker_rows(path: Path) -> list[tuple]:
    """(frame, id, x, y, w, h) tu file MOT cua tracker."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split(",")
            if len(p) >= 6:
                out.append((int(float(p[0])), int(float(p[1])), *(float(x) for x in p[2:6])))
    return out


def chi_tiet_box_doan(rows: list[tuple], regions: list[list[float]]) -> dict:
    """Mo ta box tracker run neo trong mot doan: so box, so frame co box, so id, trung vi tam/kich thuoc,
    so box co IoA > 0,5 voi vung bo qua."""
    import numpy as np
    if not rows:
        return {"neo_n_box": 0, "neo_n_frame_co_box": 0, "neo_n_id": 0, "neo_tv_cx": None, "neo_tv_cy": None,
                "neo_tv_w": None, "neo_tv_h": None, "neo_n_box_trong_vung_bo_qua": 0}
    a = np.asarray([r[2:6] for r in rows], float)
    in_ign = sum(1 for r in rows if any(ioa(r[2:6], g) > 0.5 for g in regions))
    return {"neo_n_box": len(rows), "neo_n_frame_co_box": len({r[0] for r in rows}),
            "neo_n_id": len({r[1] for r in rows}),
            "neo_tv_cx": round(float(np.median(a[:, 0] + a[:, 2] / 2)), 1),
            "neo_tv_cy": round(float(np.median(a[:, 1] + a[:, 3] / 2)), 1),
            "neo_tv_w": round(float(np.median(a[:, 2])), 1), "neo_tv_h": round(float(np.median(a[:, 3])), 1),
            "neo_n_box_trong_vung_bo_qua": in_ign}


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------------------------------------------------------------------
# Danh sach run ung vien (vai tro dat TRUOC khi doc so lieu)
# ---------------------------------------------------------------------------
def candidates() -> list[dict]:
    c = []

    def add(cau_hoi, nhan, goc, d, seed=None, vai_tro="chinh"):
        c.append({"cau_hoi": cau_hoi, "nhan": nhan, "goc": goc, "dir": d, "seed": seed, "vai_tro": vai_tro})
    # Cau hoi 1: cung ByteTrack, chi doi bo loc
    add("Q1", "bt-cv", OLD_TRK, "p2-expA_detrac_dev-c25-cv")
    add("Q1", "bt-ukf", OLD_TRK, "p2-expA_detrac_dev-c25-ukf")
    add("Q1", "bt-ekf", KC48_TRK, "p2-kc48_pf_dev-A-ekf")
    add("Q1", "bt-ekf", OLD_TRK, "p2-expA_detrac_dev-c25-ekf", vai_tro="doi_chieu")
    for s in (0, 1, 2):
        add("Q1", "bt-pf100", KC48_TRK, f"p2-kc48_pf_dev-A-pf100-s{s}", s)
        add("Q1", "bt-pf100", OLD_TRK, f"p2-expA_detrac_dev-c25-pf100-s{s}", s, vai_tro="doi_chieu")
        add("Q1", "bt-pf100-k1", KC48_TRK, f"p2-kc48_pf_dev-A-pf100-k1-s{s}", s)
        for t in ("pf100-k2", "pf500", "pf500-k1"):
            add("Q1", f"bt-{t}", KC48_TRK, f"p2-kc48_pf_dev-A-{t}-s{s}", s, vai_tro="chi_tai_dung")
    # Cau hoi 2: cung bo loc + detection, SORT vs ByteTrack
    add("Q2", "sort-cv30", OLD_TRK, "p2-expB_detrac_dev_fix1-A-sort-cv30")
    add("Q2", "sort-ekf30", OLD_TRK, "p2-expB_detrac_dev_fix1-A-sort-ekf30")
    add("Q2", "bt-ekf", OLD_TRK, "p2-expB_detrac_dev_fix1-A-bt-ekf", vai_tro="doi_chieu")
    add("Q2", "sort-orig1", OLD_TRK, "p2-expB_detrac_dev_fix1-A-sort-orig1", vai_tro="tham_chieu_khac_bo_loc")
    return c


#: Module tracker lien quan theo loai (de so hash ma luc chay vs hien tai)
MOD_COMMON = ("run_tracking.py", "det_cache.py", "time_units.py", "ul_backend.py", "bytetrack_np.py")
MOD_BY_TYPE = {
    "bytetrack": (),
    "bytetrack_ekf_ctrv": ("tracker_ctrv.py", "ekf_ctrv.py", "lost_recovery.py", "lost_association.py"),
    "bytetrack_ukf_ctrv": ("tracker_ctrv.py", "ukf_ctrv.py", "ekf_ctrv.py", "lost_recovery.py",
                           "lost_association.py"),
    "bytetrack_pf_ctrv": ("tracker_ctrv.py", "pf_ctrv.py", "ekf_ctrv.py", "lost_recovery.py",
                          "lost_association.py"),
    "sort_cv": ("sort_tracker.py",),
    "sort_ekf_ctrv": ("sort_tracker.py", "tracker_ctrv.py", "ekf_ctrv.py"),
    "sort_orig": ("sort_tracker.py",),
}


def current_yaml_sha12(yaml_path: str) -> str:
    p = ROOT / yaml_path
    if not p.exists():
        try:
            import run_tracking
            p = Path(run_tracking.resolve_tracker_yaml(yaml_path))
        except Exception as e:  # noqa: BLE001
            return f"LOI:{type(e).__name__}"
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    videos = json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]
    img_root = config.find_images_root()

    # ---------------- Phan 1: frame khong nhan ----------------
    per_video, seg_rows, unl_by_video, labeled_by_video = [], [], {}, {}
    for v in videos:
        xf = xml_frames(v)
        imf = image_frames(img_root, v) if img_root else []
        gid = gt_ids_by_frame(v)
        sl = seq_length(v)
        xs, ims = set(xf), set(imf)
        missing = sorted(ims - xs)
        span = track_spans(gid)
        segs = [{**classify_segment(a, b, xs, gid), **boundary_check(a, b, gid, span)}
                for a, b in segments(missing)]
        regions = read_regions(v)
        anchor = KC48_TRK / ANCHOR_DIR / "data" / f"{v}.txt"
        rows_anchor = read_tracker_rows(anchor) if anchor.exists() else []
        for s in segs:
            inside = [r for r in rows_anchor if s["a"] <= r[0] <= s["b"]]
            s.update(chi_tiet_box_doan(inside, regions))
            seg_rows.append({"video": v, **s})
        unl_by_video[v] = set(missing)
        labeled_by_video[v] = xs
        per_video.append({
            "video": v, "n_anh": len(imf), "anh_min": min(imf) if imf else None,
            "anh_max": max(imf) if imf else None, "n_frame_xml": len(xf), "n_frame_xml_trung": len(xf) - len(xs),
            "xml_min": min(xf), "xml_max": max(xf), "n_frame_xml_khong_anh": len(xs - ims),
            "n_frame_gt": len(gid), "n_frame_xml_khong_gt": len(xs - set(gid)), "n_frame_gt_ngoai_xml": len(set(gid) - xs),
            "seqLength": sl, "seqLength_bang_n_anh": sl == len(imf),
            "n_khong_nhan": len(missing), "n_doan": len(segs),
            "n_doan_giua": sum(s["vi_tri"] == "giua" for s in segs),
            "n_doan_dau": sum(s["vi_tri"] == "dau" for s in segs),
            "n_doan_cuoi": sum(s["vi_tri"] == "cuoi" for s in segs),
            "n_frame_co_xe_xuyen": sum(s["do_dai"] for s in segs if s["loai"] == "khong_nhan_co_xe"),
            "doan_dai_nhat": max((s["do_dai"] for s in segs), default=0),
        })

    # ---------------- Phan 2: kiem ke run ----------------
    code_now = rf.code_fingerprint()
    mods_now = code_now["modules"]
    backend_now = code_now.get("backend") or {}
    gt_now = rf.gt_fingerprint(SPLIT, videos)
    cache_ids = sorted({(load_json(c["goc"] / c["dir"] / "run_meta.json") or {}).get("cache_id")
                        for c in candidates()} - {None})
    cache_now = {cid: rf.cache_fingerprint(cid, videos) for cid in cache_ids}
    cfg_py = {"cay_lam_viec": config_py_sha256(None), "HEAD": config_py_sha256("HEAD")}
    runs, unl_rows, hashes = [], [], {}
    for c in candidates():
        tdir = c["goc"] / c["dir"]
        rm = load_json(tdir / "run_meta.json") or {}
        em = load_json(EVAL_ROOT / c["dir"] / "eval_meta.json")
        raw = load_json(EVAL_ROOT / c["dir"] / "raw_fields.json")
        cm_path = EVAL_ROOT / c["dir"] / "combined_metrics.csv"
        cm = {}
        if cm_path.exists():
            with open(cm_path, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r.get("video") == "COMBINED_SEQ":
                        cm = r
        vh, n_rows, n_unl = {}, 0, 0
        for v in videos:
            p = tdir / "data" / f"{v}.txt"
            vh[v] = rf.sha256_file(p)
            if p.exists():
                fr = tracker_frames(p)
                k = sum(1 for f in fr if f in unl_by_video[v])
                n_rows += len(fr); n_unl += k
                unl_rows.append({"dir": c["dir"], "video": v, "n_box": len(fr), "n_box_frame_khong_nhan": k})
        hashes[c["dir"]] = vh
        em_track = (em or {}).get("tracking", {}).get("per_video", {})
        hash_khop_eval = hash_dau_ra_khop(em_track, vh, videos)
        ttype = (rm.get("tracker_cfg") or {}).get("tracker_type", "")
        rel_mods = MOD_COMMON + MOD_BY_TYPE.get(ttype, ())
        mods_run = ((em or {}).get("code") or rm.get("code") or {}).get("modules", {})
        mod_lech = [m for m in rel_mods if m in mods_run and mods_run[m] != mods_now.get(m)]
        mod_thieu = [m for m in rel_mods if m not in mods_run]
        cfg = rm.get("tracker_cfg") or {}
        est = clear_after_mask(sum_raw(raw), n_unl) if raw else {}
        cfg_run = config_py_sha256(rm["git_commit"]) if rm.get("git_commit") else None
        cfg_khop = bool(cfg_run) and cfg_run == cfg_py["HEAD"] == cfg_py["cay_lam_viec"]
        yaml_now = current_yaml_sha12(rm["tracker_yaml"]) if rm.get("tracker_yaml") else None
        prov = kiem_provenance(rm or None, em, videos, vh, rel_mods, mods_now, backend_now,
                               cache_now.get(rm.get("cache_id"), {}), gt_now, yaml_now, cfg_khop)
        bl = bo_loc_hieu_luc(ttype, cfg) if ttype else dict.fromkeys(BL_COLS)
        runs.append({
            "cau_hoi": c["cau_hoi"], "nhan": c["nhan"], "vai_tro": c["vai_tro"], "dir": c["dir"],
            "goc": str(c["goc"].relative_to(ROOT)).replace("\\", "/"),
            "dataset": "UA-DETRAC train-part, dev12 (manifest detrac_dev.json)",
            "experiment": rm.get("experiment"), "cond": rm.get("cond"), "seed": rm.get("seed"),
            "n_video_co_dau_ra": sum(1 for h in vh.values() if h != "MISSING"),
            "cache_id": rm.get("cache_id"),
            "cache_meta_sha256": ((em or {}).get("cache") or {}).get("meta_sha256"),
            "cache_npz_sha256": ((em or {}).get("cache") or {}).get("npz_sha256"),
            "weights_sha256": ((em or {}).get("cache") or {}).get("weights_sha256"),
            "conf_min": rm.get("conf_min"), "classes": json.dumps(rm.get("classes")), "max_det": rm.get("max_det"),
            "fps": rm.get("fps"), "elapsed_mode": rm.get("elapsed_mode"),
            "tracker_type": ttype, "tracker_yaml": rm.get("tracker_yaml"),
            "yaml_sha12_luc_chay": rm.get("tracker_yaml_sha256_12"),
            "yaml_sha12_hien_tai": yaml_now,
            "track_high": cfg.get("track_high_thresh"), "track_low": cfg.get("track_low_thresh"),
            "new_track": cfg.get("new_track_thresh"), "match_thresh": cfg.get("match_thresh"),
            "fuse_score": cfg.get("fuse_score"), "track_buffer_frames": rm.get("track_buffer_frames"),
            "overrides": json.dumps(rm.get("overrides")),
            "assoc_log": rm.get("assoc_log"), "pf_diag": rm.get("pf_diag"),
            "git_commit_chay": rm.get("git_commit"), "created": rm.get("created"),
            "eval_meta": em is not None, "git_commit_eval": (em or {}).get("code", {}).get("git_commit"),
            "gt_sha256_eval": ((em or {}).get("gt") or {}).get("gt_sha256"),
            "seqinfo_sha256_eval": ((em or {}).get("gt") or {}).get("seqinfo_sha256"),
            "evaluator": json.dumps((em or {}).get("evaluator")),
            "hash_dau_ra_khop_eval_meta": hash_khop_eval,
            "module_lech_hien_tai": ";".join(mod_lech), "module_khong_ghi": ";".join(mod_thieu),
            "backend_luc_chay": json.dumps((rm.get("code") or {}).get("backend"), sort_keys=True),
            "config_py_sha256_luc_chay": cfg_run, "config_py_khop_HEAD_va_cay": cfg_khop,
            "tai_dung_du_dieu_kien": prov["tai_dung_du_dieu_kien"],
            "ly_do_khong_dat": " | ".join(prov["ly_do_khong_dat"]),
            "dt_danh_nghia_s": rm.get("dt_nominal_s"), "seed_runner": rm.get("seed"),
            **bl,
            # PF: run_tracking.run_video goi tr.set_seed(seed, video) SAU khi tao bo loc -> seed hieu luc la seed
            # cua runner (run_meta.seed), khong phai pf_seed trong YAML; rng moi track = (seed, hash video, id).
            "bl_pf_seed_hieu_luc": rm.get("seed") if bl.get("bl_pf_N") is not None else None,
            "sha256_dau_ra_tong": rf.sha256_obj(vh),
            "n_box": n_rows, "n_box_frame_khong_nhan": n_unl,
            "ty_le_box_frame_khong_nhan": n_unl / n_rows if n_rows else None,
            "HOTA_cu": cm.get("HOTA"), "DetA_cu": cm.get("DetA"), "AssA_cu": cm.get("AssA"),
            "IDF1_cu": cm.get("IDF1"), "IDSW_cu": cm.get("IDSW"), "FP_cu": cm.get("FP"),
            **{f"r1_{k}": v for k, v in est.items()},
        })

    # So khop dau ra giua cac run cung cau hinh (khong dung lam bang chung duy nhat)
    pairs = [("p2-kc48_pf_dev-A-ekf", "p2-expA_detrac_dev-c25-ekf"),
             ("p2-kc48_pf_dev-A-ekf", "p2-expB_detrac_dev_fix1-A-bt-ekf")]
    pairs += [(f"p2-kc48_pf_dev-A-pf100-s{s}", f"p2-expA_detrac_dev-c25-pf100-s{s}") for s in (0, 1, 2)]
    khop = []
    for a, b in pairs:
        ha, hb = hashes.get(a, {}), hashes.get(b, {})
        n = sum(1 for v in videos if ha.get(v) not in (None, "MISSING") and ha.get(v) == hb.get(v))
        khop.append({"run_a": a, "run_b": b, "n_video_trung_sha256": n, "n_video": len(videos)})

    # ---------------- Ghi ----------------
    def wcsv(name, rows):
        with open(OUT / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
    wcsv("frame_khong_nhan_theo_video.csv", per_video)
    wcsv("frame_khong_nhan_theo_doan.csv", seg_rows)
    wcsv("kiem_ke_run.csv", runs)
    wcsv("box_tracker_tren_frame_khong_nhan.csv", unl_rows)
    wcsv("so_khop_dau_ra.csv", khop)
    (OUT / "kiem_ke_run.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "frame_khong_nhan_danh_sach.json").write_text(
        json.dumps({v: sorted(unl_by_video[v]) for v in videos}, indent=0), encoding="utf-8")
    yaml_now = {y: current_yaml_sha12(y) for y in sorted({r["tracker_yaml"] for r in runs if r["tracker_yaml"]})}
    # R51A-F1: GT kiem tren TUNG run, khong loai run thieu dau van tay khoi phep so. Run thieu dau van tay GT
    # duoc liet ke rieng va khong the tai dung (kiem_provenance dieu kien 6).
    gt_eval = {(r["gt_sha256_eval"], r["seqinfo_sha256_eval"]) for r in runs if r["gt_sha256_eval"]}
    run_thieu_gt = sorted(r["dir"] for r in runs if not (r["gt_sha256_eval"] and r["seqinfo_sha256_eval"]))
    gt_pair_now = (gt_now.get("gt_sha256"), gt_now.get("seqinfo_sha256"))
    run_tai_dung = [r for r in runs if r["tai_dung_du_dieu_kien"]]
    tong = {
        "tao_luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "thoi_gian_s": round(time.time() - t0, 1),
        "git_head": code_now.get("git_commit"), "ma_hien_tai": code_now,
        "yaml_sha12_hien_tai": yaml_now,
        "gt_hien_tai": {k: gt_now.get(k) for k in ("gt_sha256", "seqinfo_sha256", "seq_lengths", "n_missing_gt",
                                                   "n_missing_seqinfo")},
        "gt_eval_meta_cac_run": sorted(gt_eval),
        "run_thieu_dau_van_tay_gt": run_thieu_gt,
        "gt_hien_tai_trung_eval_meta": bool(run_tai_dung) and all(
            (r["gt_sha256_eval"], r["seqinfo_sha256_eval"]) == gt_pair_now for r in run_tai_dung),
        "pham_vi_gt_hien_tai_trung_eval_meta": "moi run tai_dung_du_dieu_kien (khong bo run nao)",
        "backend_hien_tai": backend_now,
        "cache_hien_tai": cache_now,
        "config_py": {**cfg_py, "lan_sua_cuoi": _git_last(CONFIG_PY),
                      "khong_nam_trong_run_fingerprint_CODE_FILES": True},
        "tai_dung": {"n_dat": len(run_tai_dung), "dat": sorted(r["dir"] for r in run_tai_dung),
                     "khong_dat": {r["dir"]: r["ly_do_khong_dat"] for r in runs if not r["tai_dung_du_dieu_kien"]}},
        "img_root": str(img_root), "xml_dir": str(XML_DIR), "gt_root": str(GT_ROOT),
        "n_video": len(videos), "tong_anh": sum(r["n_anh"] for r in per_video),
        "tong_frame_xml": sum(r["n_frame_xml"] for r in per_video),
        "tong_khong_nhan": sum(r["n_khong_nhan"] for r in per_video),
        "tong_doan": len(seg_rows),
        "tong_frame_co_xe_xuyen": sum(r["n_frame_co_xe_xuyen"] for r in per_video),
        "doan_theo_vi_tri": {k: sum(1 for s in seg_rows if s["vi_tri"] == k) for k in ("dau", "giua", "cuoi")},
        "doan_theo_loai": {k: sum(1 for s in seg_rows if s["loai"] == k)
                           for k in ("khong_nhan_co_xe", "khong_ket_luan_tu_gt")},
        "frame_theo_loai": {k: sum(s["do_dai"] for s in seg_rows if s["loai"] == k)
                            for k in ("khong_nhan_co_xe", "khong_ket_luan_tu_gt")},
        "seqLength_bang_n_anh_ca_12": all(r["seqLength_bang_n_anh"] for r in per_video),
        "frame_xml_ngoai_anh": sum(r["n_frame_xml_khong_anh"] for r in per_video),
        "frame_gt_ngoai_xml": sum(r["n_frame_gt_ngoai_xml"] for r in per_video),
        "frame_xml_khong_target": sum(r["n_frame_xml_khong_gt"] for r in per_video),
        "doan_gt_truoc_deu_ket_thuc_va_gt_sau_deu_bat_dau": sum(
            1 for s in seg_rows if s["gt_truoc_ket_thuc_tai_a_1"] == s["n_gt_truoc"]
            and s["gt_sau_bat_dau_tai_b_1"] == s["n_gt_sau"]),
        "neo_run": ANCHOR_DIR, "neo_tong_box_trong_doan": sum(s["neo_n_box"] for s in seg_rows),
        "so_khop_dau_ra": khop,
        "evaluator_detrac": "MotChallenge2DBox goc (run_trackeval.DATASET_EVALUATOR['detrac']='mot')",
    }
    (OUT / "audit_tong_hop.json").write_text(json.dumps(tong, ensure_ascii=False, indent=2, default=str),
                                             encoding="utf-8")
    print(json.dumps({k: tong[k] for k in ("tong_anh", "tong_frame_xml", "tong_khong_nhan", "tong_doan",
                                           "tong_frame_co_xe_xuyen", "doan_theo_vi_tri", "doan_theo_loai",
                                           "frame_theo_loai", "seqLength_bang_n_anh_ca_12", "so_khop_dau_ra",
                                           "thoi_gian_s")}, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
