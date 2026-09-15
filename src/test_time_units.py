"""
test_time_units.py -- Kiem tra module time_units (Giai doan 2, muc B/C/D).

    python src/test_time_units.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracker_ctrv  # noqa: E402
from tracker_ctrv import CTRVByteTracker, UKFByteTracker  # noqa: E402
from time_units import (convert_state_units, steps_between,  # noqa: E402
                        track_buffer_frames, with_elapsed_steps)

from ultralytics.engine.results import Boxes  # noqa: E402
from ultralytics.trackers.byte_tracker import BYTETracker, STrack  # noqa: E402
from ultralytics.utils import IterableSimpleNamespace, YAML  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def cfg(path="configs/bytetrack_ekf_ctrv.yaml", **kw):
    d = YAML.load(str(ROOT / path)); d.update(kw)
    return IterableSimpleNamespace(**d)


def boxes(rows):
    return Boxes(np.array(rows, dtype=np.float32).reshape(-1, 6), (540, 960))


def scene(f):
    """3 xe chuyen dong deu; frame chia het cho 5 khong co detection."""
    if f % 5 == 0:
        return boxes(np.zeros((0, 6)))
    rows = []
    for k in range(3):
        cx, cy = 100 + 80 * k + (4 + k) * f, 120 + 100 * k + 2 * f
        rows.append([cx - 20, cy - 15, cx + 20, cy + 15, 0.85 - 0.1 * k, 2.0])
    return boxes(rows)


def test_1_mixin_n_steps_1_is_bit_identical():
    """Chay TUAN TU (khong xen ke): bo dem track_id cua ultralytics la toan cuc
    (BaseTrack._count), hai tracker chay xen ke se cap ID chen nhau. Runner cung
    phai tuan thu: moi (tracker, seed, video) mot tracker moi, chay het roi moi sang."""
    for base, name in [(BYTETracker, "CV"), (CTRVByteTracker, "EKF"), (UKFByteTracker, "UKF")]:
        c = cfg()

        def run(cls, **kw):
            tr = cls(args=c); tr.reset()          # reset() -> reset_id() -> ID bat dau tu 1
            return [tr.update(scene(f).cpu().numpy(), None, **kw).copy() for f in range(1, 40)]

        oa = run(base)
        ob = run(with_elapsed_steps(base), n_steps=1)
        same = all(np.array_equal(x, y) for x, y in zip(oa, ob))
        _check(f"[{name}] ElapsedStepsMixin voi n_steps=1 bit-identical lop goc", same)


def test_2_n_steps_k_equals_k_predicts_cv():
    """CV (tuyen tinh, khong hook khoi tao): update(n_steps=3) == predict^3 roi update."""
    T = with_elapsed_steps(BYTETracker)
    tr = T(args=cfg("configs/bytetrack_ekf_ctrv.yaml", tracker_type="bytetrack")); tr.reset()
    z1 = [100, 100, 150, 140, 0.9, 2.0]
    tr.update(boxes([z1]).cpu().numpy(), None)
    tr.update(boxes([[104, 101, 154, 141, 0.9, 2.0]]).cpu().numpy(), None)
    t = tr.tracked_stracks[0]
    m, P = t.mean.copy(), t.covariance.copy()
    kf = tr.kalman_filter
    for _ in range(3):
        m, P = kf.predict(m, P)
    z4 = np.array([116, 104, 166, 144, 0.9, 2.0], dtype=np.float32)
    m_exp, P_exp = kf.update(m, P, STrack.tlwh_to_xyah(np.array([116, 104, 50, 40], np.float32)))
    tr.update(boxes([z4]).cpu().numpy(), None, n_steps=3)
    _check("frame_id tang dung k buoc (2 -> 5)", tr.frame_id == 5, f"frame_id={tr.frame_id}")
    _check("[CV] mean sau update(n_steps=3) == predict^3 + update",
           np.allclose(t.mean, m_exp, atol=1e-9) and np.allclose(t.covariance, P_exp, atol=1e-9),
           f"max|dm|={np.abs(t.mean - m_exp).max():.2e}")


def test_3_two_frame_init_uses_elapsed_steps():
    """EKF: khoi tao 2 khung hinh phai chia dich chuyen cho k buoc (van toc theo buoc)."""
    T = with_elapsed_steps(CTRVByteTracker)
    tr = T(args=cfg()); tr.reset()
    # SPY nhu test_tracker_lifecycle: boc try_initiate_from_motion cua bo loc de doc
    # n_frames va van toc NGAY LUC SEED (truoc khi buoc KF update lam nhieu no).
    seen = {}
    kf = tr.kalman_filter
    real = kf.try_initiate_from_motion

    def spy(mean, cov, z, n_frames=1.0, ref_pos=None):
        m, P, ok = real(mean, cov, z, n_frames=n_frames, ref_pos=ref_pos)
        seen.update(n_frames=n_frames, v=float(m[2]), ok=ok)
        return m, P, ok
    kf.try_initiate_from_motion = spy

    tr.update(boxes([[100, 100, 150, 140, 0.9, 2.0]]).cpu().numpy(), None)
    # sau 4 buoc, xe di 4 * (3, 4) = (12, 16) px
    tr.update(boxes([[112, 116, 162, 156, 0.9, 2.0]]).cpu().numpy(), None, n_steps=4)
    _check("[EKF] khoi tao 2 khung hinh nhan n_frames = 4 (so buoc da troi qua)",
           seen.get("n_frames") == 4 and seen.get("ok"), str(seen))
    _check("[EKF] van toc seed = 20 px / 4 buoc = 5 px/buoc (khong phai 20)",
           np.isclose(seen.get("v", -1), 5.0), str(seen))


def test_4_track_buffer_counts_steps():
    T = with_elapsed_steps(CTRVByteTracker)
    tr = T(args=cfg(track_buffer=30)); tr.reset()
    for f in (1, 2, 3):
        tr.update(boxes([[100 + 5 * f, 100, 150 + 5 * f, 140, 0.9, 2.0]]).cpu().numpy(), None)
    tr.update(boxes(np.zeros((0, 6))).cpu().numpy(), None)          # -> lost tai frame 4 (end_frame = 3)
    _check("track chuyen sang lost", len(tr.lost_stracks) == 1)
    tr.update(boxes(np.zeros((0, 6))).cpu().numpy(), None, n_steps=29)   # frame_id = 33, 33-3 = 30: chua xoa
    _check("chua xoa khi frame_id - end_frame == track_buffer", len(tr.lost_stracks) == 1 and tr.frame_id == 33)
    tr.update(boxes(np.zeros((0, 6))).cpu().numpy(), None, n_steps=1)    # 34 - 3 = 31 > 30: xoa
    # Luu y (hanh vi goc cua ultralytics 8.4.141, merge_track_pools): track vua bi xoa
    # con nam trong lost_stracks them dung 1 frame (sub_stracks chay TRUOC extend), nen
    # kiem tra qua removed_stracks + trang thai, khong qua do dai lost_stracks.
    from ultralytics.trackers.basetrack import TrackState
    _check("xoa khi vuot track_buffer tinh theo buoc",
           len(tr.removed_stracks) == 1 and tr.removed_stracks[0].state == TrackState.Removed)


def test_5_steps_between_and_buffer():
    _check("steps_between deu", steps_between(0.2, 0.2) == (1, True))
    _check("steps_between bo 1 frame", steps_between(0.41, 0.2) == (2, True))
    _check("steps_between khoang le -> bao khong deu", steps_between(0.3, 0.2)[1] is False)
    _check("steps_between dt <= 0 -> 1 buoc", steps_between(0.0, 0.2) == (1, False))
    _check("track_buffer 1.2 s: 25 fps -> 30, 5 fps -> 6",
           track_buffer_frames(1.2, 25) == 30 and track_buffer_frames(1.2, 5) == 6)


def test_6_unit_conversion_congruence():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(9, 9)); P = A @ A.T + 1e-3 * np.eye(9)
    m = rng.normal(size=9)
    m2, P2 = convert_state_units(m, P, fps=25.0, layout="ctrv", to_seconds=True)
    _check("v, omega, va, vh nhan fps; bien khac giu nguyen",
           np.allclose(m2[[2, 4, 7, 8]], m[[2, 4, 7, 8]] * 25) and np.allclose(m2[[0, 1, 3, 5, 6]], m[[0, 1, 3, 5, 6]]))
    _check("phan tu cheo Cov(v, omega) nhan fps^2, Cov(cx, v) nhan fps",
           np.isclose(P2[2, 4], P[2, 4] * 625) and np.isclose(P2[0, 2], P[0, 2] * 25) and np.isclose(P2[0, 1], P[0, 1]))
    _check("ban xac dinh duong duoc bao toan", np.linalg.eigvalsh(P2).min() > 0)
    m3, P3 = convert_state_units(m2, P2, fps=25.0, layout="ctrv", to_seconds=False)
    _check("round-trip", np.allclose(m3, m) and np.allclose(P3, P))
    mx, Px = convert_state_units(np.arange(8.0), np.eye(8), fps=5.0, layout="xyah")
    _check("layout xyah: vx, vy, va, vh nhan fps", np.allclose(mx[4:], np.arange(4, 8) * 5) and np.allclose(mx[:4], np.arange(4)))


def main() -> int:
    tracker_ctrv.register()
    failed = 0
    for t in [test_1_mixin_n_steps_1_is_bit_identical, test_2_n_steps_k_equals_k_predicts_cv,
              test_3_two_frame_init_uses_elapsed_steps, test_4_track_buffer_counts_steps,
              test_5_steps_between_and_buffer, test_6_unit_conversion_congruence]:
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
