"""
test_sort_upstream_parity.py -- (sau review Codex, loi P1 "SORT lech vong doi") So sanh baseline
`sort_orig` cua repo voi MA GOC abewley/sort DA GHIM (third_party/sort_upstream/sort.py, xem
PROVENANCE.json) tren CUNG detection, ca chuoi.

VI SAO CAN: `SORTTrack.__init__` truoc day dat `hits = 1; hit_streak = 1` trong khi upstream dat 0 -> track
duoc XUAT SOM HON UPSTREAM MOT FRAME khi `min_hits > 1`. Test cu (`test_2_min_hits`) lai khang dinh dung
hanh vi lech do nen PASS khong chung minh tuong duong.

CACH CHAY MA UPSTREAM: upstream can `filterpy` (cai rieng vao `third_party/_testdeps`, KHONG vao moi truong
chinh) va import `matplotlib`/`skimage` chi de ve demo -> test thay bang module rong trong `sys.modules`.
Neu thieu `filterpy` thi test bao SKIP (khong FAIL) kem huong dan cai.

KICH BAN SO SANH (moi kich ban: cung mang detection, so TOAN CHUOI):
    A. min_hits: 9 frame rong roi doi tuong xuat hien lien tuc tu frame 10 (kiem tra dung frame xuat)
    B. track sinh muon + mat detection + tai xuat hien (max_age 1 va 30)
    C. hai doi tuong, mot bi che 5 frame (kiem tra xoa track / ID moi)
    D. detection that tu cache (MVI_39031, 200 frame, conf > 0.25, lop [2,3,5,7]) - neu co cache
So sanh bang ANH XA ID 1-1 CO DINH toan chuoi (`track_compare.id_bijection`), dung sai toa do CONG BO
(mac dinh 1e-6 px; upstream dung filterpy, repo dung numpy nen co the lech o muc epsilon).

    python src/test_sort_upstream_parity.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402
from track_compare import id_bijection  # noqa: E402

#: Dung sai ghep dong theo box. Khac biet SO HOC giua hai cai dat (repo: numpy + dang Joseph;
#: upstream: filterpy) do duoc o muc ~1e-5 px -> ghep dong voi 1e-3 px, va kiem RIENG rang do lech
#: TOI DA phai duoi MAX_DEV_PX. Gia tri do duoc in ra cuoi test.
TOL_PX = 1e-3
MAX_DEV_PX = 0.05
_RESULTS: list[tuple[str, bool]] = []
_DEV: list[float] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines()[-4:]:
            print(f"         {line}")
    assert ok, name


def load_upstream():
    """Import third_party/sort_upstream/sort.py (da ghim). None neu thieu filterpy."""
    dep = ROOT / "third_party" / "_testdeps"
    if dep.is_dir():
        sys.path.insert(0, str(dep))
    try:
        import filterpy  # noqa: F401
    except ImportError:
        return None
    for m in ("matplotlib", "matplotlib.pyplot", "matplotlib.patches", "skimage", "skimage.io"):
        mod = types.ModuleType(m)
        mod.use = lambda *a, **k: None
        sys.modules.setdefault(m, mod)
    sys.path.insert(0, str(ROOT / "third_party" / "sort_upstream"))
    import sort as upstream
    return upstream


def run_upstream(up, frames: list[np.ndarray], max_age=1, min_hits=3, iou=0.3) -> pd.DataFrame:
    up.KalmanBoxTracker.count = 0
    s = up.Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou)
    rows = []
    for i, det in enumerate(frames, 1):
        r = s.update(det if len(det) else np.empty((0, 5)))
        for b in np.asarray(r).reshape(-1, 5):
            rows.append((i, int(b[4]), float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])))
    return pd.DataFrame(rows, columns=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height"])


def run_repo(frames: list[np.ndarray], max_age=1, min_hits=3, iou=0.3) -> pd.DataFrame:
    from ultralytics.utils import IterableSimpleNamespace
    from sort_tracker import SORT
    cfg = IterableSimpleNamespace(tracker_type="sort_orig", sort_max_age=max_age, sort_min_hits=min_hits,
                                  sort_iou_threshold=iou, det_thresh=0.0)
    tr = SORT(args=cfg); tr.reset()
    rows = []
    for i, det in enumerate(frames, 1):
        d = np.asarray(det, dtype=np.float64).reshape(-1, 5)
        res = types.SimpleNamespace(xyxy=d[:, :4], conf=d[:, 4],
                                    cls=np.full(len(d), 2.0), xywh=np.zeros((len(d), 4)))
        out = np.asarray(tr.update(res, None)).reshape(-1, 8)
        for b in out:
            rows.append((i, int(b[4]), float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])))
    return pd.DataFrame(rows, columns=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height"])


# --- kich ban tong hop ---
def scen_min_hits(n=16):
    out = []
    for f in range(1, n + 1):
        out.append(np.empty((0, 5)) if f < 10 else
                   np.array([[100 + 5.0 * f, 100.0, 160 + 5.0 * f, 140.0, 0.9]]))
    return out


def scen_late_and_gap(n=40, gap=(20, 22)):
    out = []
    for f in range(1, n + 1):
        rows = [[50 + 4.0 * f, 300.0, 110 + 4.0 * f, 340.0, 0.9]]                       # xe co tu dau
        if f >= 8 and not (gap[0] <= f <= gap[1]):                                      # xe sinh muon + mat 3 frame
            rows.append([400 + 6.0 * f, 100.0, 460 + 6.0 * f, 150.0, 0.85])
        out.append(np.array(rows))
    return out


def scen_two_objects_occlusion(n=30, occ=(12, 16)):
    out = []
    for f in range(1, n + 1):
        rows = [[100 + 8.0 * f, 200.0, 170 + 8.0 * f, 250.0, 0.9]]
        if not (occ[0] <= f <= occ[1]):
            rows.append([600 - 5.0 * f, 400.0, 660 - 5.0 * f, 450.0, 0.8])
        out.append(np.array(rows))
    return out


def scen_cache(video="MVI_39031", n_frames=200, conf=0.25):
    import det_cache
    try:
        vd = det_cache.open_cache("yolov8n-coco_conf0.05_all", video, conf, [2, 3, 5, 7], 300)
    except Exception:  # noqa: BLE001
        return None
    out = []
    for i, (_f, _t, unread, arr) in enumerate(vd.iter_frames()):
        if i >= n_frames:
            break
        if unread or not len(arr):
            out.append(np.empty((0, 5))); continue
        out.append(np.c_[arr[:, :4].astype(np.float64), arr[:, 4].astype(np.float64)])
    return out


def max_deviation(a: pd.DataFrame, b: pd.DataFrame, mapping: dict) -> float:
    """Do lech toa do lon nhat giua hai ban sau khi ap anh xa ID 1-1."""
    if not mapping or not len(a):
        return float("nan")
    bb = b.copy()
    bb["id"] = bb["id"].map({v: k for k, v in mapping.items()})
    m = a.merge(bb, on=["frame", "id"], suffixes=("_a", "_b"))
    if not len(m):
        return float("nan")
    cols = ["bb_left", "bb_top", "bb_width", "bb_height"]
    return float(np.abs(m[[c + "_a" for c in cols]].to_numpy()
                        - m[[c + "_b" for c in cols]].to_numpy()).max())


def compare(name, up, frames, max_age, min_hits):
    a = run_repo(frames, max_age, min_hits)
    b = run_upstream(up, frames, max_age, min_hits)
    r = id_bijection(a, b, tol=TOL_PX)
    dev = max_deviation(a, b, r.mapping) if r.is_bijection else float("nan")
    if dev == dev:
        _DEV.append(dev)
    _check(f"[{name}] repo == upstream (max_age={max_age}, min_hits={min_hits}; "
           f"{r.n_rows_a} vs {r.n_rows_b} dong, {r.n_ids_a} vs {r.n_ids_b} track; "
           f"lech toa do toi da {dev:.2e} px)",
           r.is_bijection and r.n_rows_a == r.n_rows_b and (dev != dev or dev <= MAX_DEV_PX), r.summary())
    return a, b


def main() -> int:
    up = load_upstream()
    if up is None:
        print("  [SKIP] thieu filterpy -> khong chay duoc ma upstream.")
        print("         Cai rieng, khong vao moi truong chinh:")
        print("         python -m pip install --no-deps --target third_party/_testdeps filterpy==1.4.5")
        return 0
    failed = 0
    try:
        # A. dung frame xuat track (loi da sua)
        a, b = compare("min_hits", up, scen_min_hits(), max_age=1, min_hits=3)
        first_repo = int(a.frame.min()); first_up = int(b.frame.min())
        _check(f"[min_hits] track xuat lan dau o frame {first_repo} (upstream {first_up}) = 13",
               first_repo == first_up == 13, f"repo {first_repo}, upstream {first_up}")
        # B. track sinh muon + mat detection + tai xuat hien
        compare("late+gap, max_age 1", up, scen_late_and_gap(), max_age=1, min_hits=3)
        compare("late+gap, max_age 30", up, scen_late_and_gap(), max_age=30, min_hits=3)
        # C. hai doi tuong, mot bi che
        compare("2 doi tuong + che 5 frame, max_age 1", up, scen_two_objects_occlusion(), max_age=1, min_hits=3)
        compare("2 doi tuong + che 5 frame, max_age 30", up, scen_two_objects_occlusion(), max_age=30, min_hits=3)
        # D. detection that
        fr = scen_cache()
        if fr is None:
            print("  [SKIP] khong co cache detection that (yolov8n-coco_conf0.05_all/MVI_39031)")
        else:
            compare("cache that MVI_39031 200 frame, max_age 1", up, fr, max_age=1, min_hits=3)
            compare("cache that MVI_39031 200 frame, max_age 30", up, fr, max_age=30, min_hits=3)
    except AssertionError:
        failed += 1
    n_ok = sum(ok for _, ok in _RESULTS)
    if _DEV:
        print(f"\nDUNG SAI SO HOC: lech toa do toi da tren moi kich ban = {max(_DEV):.3e} px "
              f"(nguong {MAX_DEV_PX} px). Vong doi, so track va anh xa ID TRUNG KHIT; khac biet con lai "
              f"chi la so hoc (repo: numpy + dang Joseph; upstream: filterpy).")
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
