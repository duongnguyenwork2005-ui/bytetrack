"""
test_sort_tracker.py -- Kiem tra SORT (sort_orig / sort_cv / sort_ekf_ctrv) trong src/sort_tracker.py.

    1. Contract output (N, 8), ID nguyen, box huu han
    2. min_hits: frame <= 3 xuat ngay; sau do phai du 3 lan khop lien tiep
    3. max_age = 1: mat 1 frame -> track bi xoa -> ID MOI (dung hanh vi nguyen ban)
    4. max_age = 30: mat 3 frame -> giu ID (bo loc du doan bac cau)
    5. Bo loc: SortKF7 predict/update dung cong thuc; XYAH == KalmanFilterXYAH; CTRV khoi tao 2 khung hinh
    6. det_thresh loc detection dau vao
    7. ElapsedStepsMixin: n_steps = k -> time_since_update += k
    8. Registry cua run_tracking co sort_*; chay tren cache tong hop qua run_tracking.track_video

    python src/test_sort_tracker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_tracking  # noqa: E402
from sort_tracker import SORT, CTRVFilter, SortKF7, XYAHFilter, associate, iou_batch  # noqa: E402
from time_units import with_elapsed_steps  # noqa: E402

from ultralytics.engine.results import Boxes  # noqa: E402
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH  # noqa: E402
from ultralytics.utils import IterableSimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def boxes(rows):
    return Boxes(np.array(rows, dtype=np.float32).reshape(-1, 6), (540, 960))


def args(tracker_type="sort_orig", **kw):
    d = {"tracker_type": tracker_type, "det_thresh": 0.25, "sort_iou_threshold": 0.3, "sort_max_age": 1,
         "sort_min_hits": 3, "track_buffer": 1}
    d.update(kw)
    return IterableSimpleNamespace(**d)


def straight(f, n=2, gap=()):
    if f in gap:
        return boxes(np.zeros((0, 6)))
    rows = [[100 + 5 * f + 150 * k, 100 + 2 * f, 150 + 5 * f + 150 * k, 140 + 2 * f, 0.9, 2.0] for k in range(n)]
    return boxes(rows)


def test_1_contract():
    for tt in ("sort_orig", "sort_cv", "sort_ekf_ctrv"):
        tr = SORT(args(tt)); tr.reset()
        outs = [tr.update(straight(f).cpu().numpy(), None) for f in range(1, 20)]
        o = outs[-1]
        _check(f"[{tt}] output (N,8) float32, 2 track, ID nguyen, box huu han w/h > 0",
               o.shape == (2, 8) and o.dtype == np.float32 and np.isfinite(o).all()
               and np.all(o[:, 4] == o[:, 4].astype(int)) and (o[:, 2] > o[:, 0]).all() and (o[:, 3] > o[:, 1]).all(),
               str(o))


def test_2_min_hits():
    tr = SORT(args("sort_orig", sort_min_hits=3)); tr.reset()
    outs = [tr.update(straight(f, n=1).cpu().numpy(), None) for f in range(1, 4)]
    _check("frame 1..3 (<= min_hits): xuat ngay", all(len(o) == 1 for o in outs))
    # track moi xuat hien o frame 10 -> can 3 lan khop lien tiep (frame 10, 11, 12) moi xuat
    tr = SORT(args("sort_orig", sort_min_hits=3)); tr.reset()
    for f in range(1, 10):
        tr.update(straight(f, n=1).cpu().numpy(), None)
    seen = []
    for f in range(10, 14):
        det = boxes([[400, 400, 450, 440, 0.9, 2.0], *straight(f, n=1).data.tolist()])
        out = tr.update(det.cpu().numpy(), None)
        seen.append(sorted(out[:, 4].astype(int).tolist()))
    _check("track moi (id 2) chi xuat tu lan khop thu 3", seen[0] == [1] and seen[1] == [1] and seen[2] == [1, 2], str(seen))


def test_3_max_age_1_new_id():
    tr = SORT(args("sort_orig", sort_max_age=1)); tr.reset()
    ids = []
    for f in range(1, 16):
        out = tr.update(straight(f, n=1, gap=(8, 9)).cpu().numpy(), None)
        ids += out[:, 4].astype(int).tolist()
    _check("max_age=1, mat 2 frame -> xoa track, ID moi (nguyen ban)", set(ids) == {1, 2}, str(sorted(set(ids))))


def test_4_max_age_30_keeps_id():
    for tt in ("sort_orig", "sort_cv", "sort_ekf_ctrv"):
        tr = SORT(args(tt, sort_max_age=30)); tr.reset()
        ids = []
        for f in range(1, 20):
            out = tr.update(straight(f, n=1, gap=(8, 9, 10)).cpu().numpy(), None)
            ids += out[:, 4].astype(int).tolist()
        _check(f"[{tt}] max_age=30, mat 3 frame -> giu ID (du doan bac cau)", set(ids) == {1}, str(sorted(set(ids))))


def test_5_filters():
    kf = SortKF7(); kf.initiate([100, 100, 150, 140])
    _check("SortKF7 initiate: x = [125,120,2000,1.25,0,0,0], P[4:,4:] = 1e4", np.allclose(kf.x[:4], [125, 120, 2000, 1.25])
           and np.allclose(kf.P[4, 4], 10000) and np.allclose(kf.P[0, 0], 10))
    kf.predict(); kf.update([105, 101, 155, 141]); kf.predict()
    _check("SortKF7 predict/update: dich chuyen theo huong quan sat, box huu han",
           kf.x[0] > 125 and np.isfinite(kf.xyxy).all())
    xf = XYAHFilter(); xf.initiate([100, 100, 150, 140]); ref = KalmanFilterXYAH()
    m, P = ref.initiate(np.array([125, 120, 50 / 40, 40.0]))
    _check("XYAHFilter == KalmanFilterXYAH.initiate", np.allclose(xf.mean, m) and np.allclose(xf.cov, P))
    cf = CTRVFilter(); cf.initiate([100, 100, 150, 140]); cf.predict(); cf.update([104, 103, 154, 143])
    _check("CTRVFilter: khoi tao 2 khung hinh -> v ~ 5, theta ~ 37 deg",
           cf._motion_initialized and 3 < cf.mean[2] < 8 and 20 < np.degrees(cf.mean[3]) < 55,
           f"v={cf.mean[2]:.2f} theta={np.degrees(cf.mean[3]):.1f}")
    cf2 = CTRVFilter(); cf2.initiate([100, 100, 150, 140])
    for _ in range(3):
        cf2.predict()
    seen = {}
    real = cf2.kf.try_initiate_from_motion

    def spy(mean, cov, z, n_frames=1.0, ref_pos=None):      # doc gia tri NGAY LUC SEED (nhu test_time_units)
        m, P, ok = real(mean, cov, z, n_frames=n_frames, ref_pos=ref_pos)
        seen.update(n_frames=n_frames, v=float(m[2]), ok=ok)
        return m, P, ok
    cf2.kf.try_initiate_from_motion = spy
    cf2.update([112, 109, 162, 149])       # 3 buoc, dich (12, 9) -> 5 px/buoc
    _check("CTRVFilter: n_frames = so buoc da troi qua (3), van toc seed = 15/3 = 5 px/buoc",
           seen.get("n_frames") == 3 and seen.get("ok") and np.isclose(seen.get("v", -1), 5.0), str(seen))


def test_6_det_thresh():
    tr = SORT(args("sort_orig", det_thresh=0.5)); tr.reset()
    det = boxes([[100, 100, 150, 140, 0.9, 2.0], [300, 300, 350, 340, 0.3, 2.0]])
    out = tr.update(det.cpu().numpy(), None)
    _check("det_thresh 0.5: chi 1 detection duoc dua vao", len(out) == 1 and int(out[0, 7]) == 0)


def test_7_elapsed_steps():
    T = with_elapsed_steps(SORT)
    tr = T(args("sort_orig", sort_max_age=30)); tr.reset()
    tr.update(straight(1, n=1).cpu().numpy(), None)
    tr.update(straight(2, n=1).cpu().numpy(), None)
    tr.update(boxes(np.zeros((0, 6))).cpu().numpy(), None, n_steps=4)
    t = tr.tracks[0]
    _check("n_steps=4 khong detection: time_since_update = 4, age = 5, frame_id = 6",
           t.time_since_update == 4 and t.age == 5 and tr.frame_id == 6, f"tsu={t.time_since_update} age={t.age} fid={tr.frame_id}")


def test_8_registry_and_runner():
    reg = run_tracking.tracker_registry()
    _check("registry co sort_orig / sort_cv / sort_ekf_ctrv", all(k in reg for k in ("sort_orig", "sort_cv", "sort_ekf_ctrv")))
    for tt in ("sort_orig", "sort_ekf_ctrv"):
        cfg = run_tracking.load_tracker_cfg(ROOT / f"configs/phase2/trackers/{tt}_age30.yaml")
        tr = run_tracking.make_tracker(cfg, reg)
        outs = [tr.update(straight(f).cpu().numpy(), None, n_steps=1) for f in range(1, 10)]
        _check(f"[{tt}] make_tracker + yaml + ElapsedSteps chay duoc", len(outs[-1]) == 2)
    _check("associate: khong track -> moi det chua khop", associate(np.zeros((2, 4)), np.zeros((0, 4)), 0.3)[1].tolist() == [0, 1])
    _check("iou_batch dung", np.isclose(iou_batch(np.array([[0, 0, 10, 10.]]), np.array([[5, 0, 15, 10.]]))[0, 0], 1 / 3))


def main() -> int:
    failed = 0
    for t in [test_1_contract, test_2_min_hits, test_3_max_age_1_new_id, test_4_max_age_30_keeps_id, test_5_filters,
              test_6_det_thresh, test_7_elapsed_steps, test_8_registry_and_runner]:
        print(f"\n--- {t.__name__} ---")
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  !! {e}")
    n_ok = sum(ok for _, ok in _RESULTS)
    print("\n" + "=" * 78)
    print(f"KET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
