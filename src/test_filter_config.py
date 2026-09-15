"""
test_filter_config.py -- Regression cho LOI `shared_kalman` va luong truyen tham so
bo loc tu file .yaml (Giai doan 2).

LOI DUOC BAO VE
    `CTRVSTrack.multi_predict` (ban cu) dung `cls.shared_kalman` - instance tao
    luc import voi tham so mac dinh. `BYTETracker.update()` chi goi
    `multi_predict`, nen tham so nhieu dat tren bo loc cua tracker (tu
    `get_kalmanfilter()`) KHONG BAO GIO toi buoc du doan. Da do: dat
    std_theta = 1.0 tren bo loc cua tracker, P[theta,theta] sau multi_predict van
    tang dung 0.010^2 (mac dinh). Test 3 di DUNG duong chay bi loi
    (BYTETracker.update -> multi_predict) chu khong goi thang bo loc.

DIEU PHAI GIU
    Khong truyen khoa nao (file .yaml cua Giai doan 1) -> bo loc moi TRUNG KHIT
    bo loc cu, bit-identical, ca predict lan multi_predict, ca EKF lan UKF.

KHONG can dataset, YOLO hay GPU.

    python src/test_filter_config.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402
from ukf_ctrv import UKFTrackerCTRV  # noqa: E402
import tracker_ctrv  # noqa: E402
from tracker_ctrv import (CTRVByteTracker, CTRVSTrack, UKFByteTracker,  # noqa: E402
                          UKFSTrack, ctrv_kwargs_from_args)

from ultralytics.engine.results import Boxes  # noqa: E402
from ultralytics.utils import IterableSimpleNamespace, YAML  # noqa: E402

_RESULTS: list[tuple[str, bool]] = []
ROOT = Path(__file__).resolve().parents[1]


def _check(name: str, ok: bool, detail: str = ""):
    _RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def load_cfg(path: str, **override) -> IterableSimpleNamespace:
    d = YAML.load(str(ROOT / path))
    d.update(override)
    return IterableSimpleNamespace(**d)


def synthetic_dets(frame: int) -> Boxes:
    """Hai xe: mot di thang (xe 1), mot dang re (xe 2). Frame 4 khong co detection."""
    if frame == 4:
        return Boxes(np.zeros((0, 6), dtype=np.float32), (540, 960))
    t = frame
    x1 = 100 + 6.0 * t
    ang = 0.15 * t
    cx2, cy2 = 400 + 60 * np.sin(ang), 300 - 60 * np.cos(ang) + 60
    rows = [[x1, 100.0, x1 + 50, 140.0, 0.9, 2.0],
            [cx2 - 25, cy2 - 20, cx2 + 25, cy2 + 20, 0.7, 7.0]]
    return Boxes(np.array(rows, dtype=np.float32), (540, 960))


def run_tracker(tracker, n_frames: int = 12):
    """Chay tracker qua chuoi detection tong hop, tra ve list output tung frame + track cuoi."""
    outs = []
    for f in range(1, n_frames + 1):
        outs.append(tracker.update(synthetic_dets(f).cpu().numpy(), None).copy())
    return outs


# ---------------------------------------------------------------------------
def test_1_default_filter_is_bit_identical_to_shared_kalman():
    """Bo loc mac dinh (khong truyen khoa) == shared_kalman cu, bit-identical."""
    rng = np.random.default_rng(0)
    for cls, name in [(CTRVSTrack, "EKF"), (UKFSTrack, "UKF")]:
        new = cls.filter_class()            # cach get_kalmanfilter() tao khi .yaml khong co khoa
        old = cls.shared_kalman              # instance tao luc import (duong cu)
        z = np.array([300.0, 200.0, 0.5, 40.0])
        m0, P0 = new.initiate(z)
        m0_old, P0_old = old.initiate(z)
        _check(f"[{name}] initiate trung khit", np.array_equal(m0, m0_old) and np.array_equal(P0, P0_old))
        m1, P1, ok = new.try_initiate_from_motion(m0, P0, z + [4, 3, 0, 0], n_frames=1, ref_pos=(300, 200))
        m1o, P1o, oko = old.try_initiate_from_motion(m0, P0, z + [4, 3, 0, 0], n_frames=1, ref_pos=(300, 200))
        _check(f"[{name}] try_initiate_from_motion trung khit",
               ok == oko and np.array_equal(m1, m1o) and np.array_equal(P1, P1o))
        means = np.stack([m1 + rng.normal(0, 0.1, 9) for _ in range(5)])
        means[:, 3] = np.clip(means[:, 3], -3, 3)
        covs = np.stack([P1] * 5)
        a = new.multi_predict(means, covs)
        b = old.multi_predict(means, covs)
        _check(f"[{name}] multi_predict trung khit", np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
        ma, Pa = new.update(a[0][0], a[1][0], z, 0.8)
        mb, Pb = old.update(b[0][0], b[1][0], z, 0.8)
        _check(f"[{name}] update trung khit", np.array_equal(ma, mb) and np.array_equal(Pa, Pb))


def test_2_tracker_end_to_end_default_identical():
    """Qua BYTETracker.update: bo loc tu get_kalmanfilter (moi) va shared_kalman (cu) cho cung output.

    Ep tracker dung dung duong cu bang cach gan track.kalman_filter = shared_kalman
    roi so voi duong moi. Ca hai phai bit-identical khi .yaml khong co khoa ctrv_*.
    """
    for T, name in [(CTRVByteTracker, "EKF"), (UKFByteTracker, "UKF")]:
        cfg = load_cfg("configs/bytetrack_ekf_ctrv.yaml")
        new = T(args=cfg); new.reset()
        outs_new = run_tracker(new)

        old = T(args=cfg); old.reset()
        old.kalman_filter = T.track_class.shared_kalman      # duong cu: instance luc import
        outs_old = run_tracker(old)
        same = all(np.array_equal(a, b) for a, b in zip(outs_new, outs_old))
        _check(f"[{name}] output tracker moi == cu (bit-identical, 12 frame, co frame rong)", same)


def test_3_yaml_override_reaches_multi_predict():
    """Di dung duong bi loi: BYTETracker.update -> multi_predict. Override phai co tac dung."""
    base = load_cfg("configs/bytetrack_ekf_ctrv.yaml")
    big = load_cfg("configs/bytetrack_ekf_ctrv.yaml", ctrv_std_theta=1.0)

    def theta_growth(cfg):
        tr = CTRVByteTracker(args=cfg); tr.reset()
        for f in (1, 2, 3):
            tr.update(synthetic_dets(f).cpu().numpy(), None)
        t = tr.tracked_stracks[0]
        before = float(t.covariance[3, 3])
        tr.update(synthetic_dets(4).cpu().numpy(), None)   # frame rong -> chi predict (track -> Lost)
        return before, float(t.covariance[3, 3]), tr, t

    b0, b1, tr_b, _ = theta_growth(base)
    g0, g1, tr_g, t_g = theta_growth(big)
    _check("bo loc cua tracker doc duoc ctrv_std_theta tu .yaml",
           tr_g.kalman_filter._std_theta == 1.0 and tr_b.kalman_filter._std_theta == config.CTRV_STD_THETA)
    _check("track dung dung instance bo loc cua tracker",
           t_g.kalman_filter is tr_g.kalman_filter and t_g in tr_g.lost_stracks)
    # Tang phuong sai theta sau 1 buoc predict: mac dinh ~0.010^2, override ~1.0^2
    # Tang cua mac dinh gom ca phan lan truyen F P F^T (omega -> theta) nen khong bang
    # dung 0.010^2; chi can no NHO hon nhieu so voi 1.0^2 cua override.
    _check("override toi multi_predict: P[theta,theta] tang ~1.0^2 (khong phai 0.010^2)",
           (g1 - g0) > 0.9 and (b1 - b0) < 0.1 and (g1 - g0) - (b1 - b0) > 0.9,
           f"mac dinh: {b0:.5f} -> {b1:.5f} (+{b1 - b0:.5f});  override: {g0:.5f} -> {g1:.5f} (+{g1 - g0:.5f})")
    _check("shared_kalman KHONG bi doi boi override (van 0.010)",
           CTRVSTrack.shared_kalman._std_theta == config.CTRV_STD_THETA)


def test_4_init_params_on_instance():
    """initiate / try_initiate_from_motion dung tham so cua instance, khong doc config luc goi."""
    kf = EKFTrackerCTRV(init_std_theta=0.3, init_std_omega=0.02,
                        heading_std_after_init=0.05, min_speed_for_heading=100.0)
    z = np.array([10.0, 10.0, 0.5, 40.0])
    m, P = kf.initiate(z)
    _check("init_std_theta / init_std_omega tren instance",
           np.isclose(P[3, 3], 0.3 ** 2) and np.isclose(P[4, 4], 0.02 ** 2))
    _, _, ok = kf.try_initiate_from_motion(m, P, z + [5, 0, 0, 0], n_frames=1, ref_pos=(10, 10))
    _check("min_speed_for_heading tren instance (100 px/buoc -> chua khoi tao)", ok is False)
    kf2 = EKFTrackerCTRV(heading_std_after_init=0.05)
    m2, P2, ok2 = kf2.try_initiate_from_motion(m, P, z + [5, 0, 0, 0], n_frames=1, ref_pos=(10, 10))
    _check("heading_std_after_init tren instance", ok2 and np.isclose(P2[3, 3], 0.05 ** 2))
    ukf = UKFTrackerCTRV(std_theta=0.123, std_omega=0.456)
    _check("UKF chuyen ctrv kwargs cho EKF", ukf._std_theta == 0.123 and ukf._std_omega == 0.456)


def test_5_ctrv_kwargs_from_args():
    cfg = load_cfg("configs/bytetrack_ekf_ctrv.yaml")
    _check("yaml Giai doan 1 -> khong co kwargs (giu mac dinh)", ctrv_kwargs_from_args(cfg) == {})
    cfg2 = load_cfg("configs/bytetrack_ekf_ctrv.yaml", ctrv_std_omega=0.02, ctrv_dt=1.0)
    _check("yaml co khoa -> kwargs dung ten",
           ctrv_kwargs_from_args(cfg2) == {"std_omega": 0.02, "dt": 1.0})


def test_6_config_override_before_tracker_creation_still_works():
    """Duong `baseline_track.py --ctrv-init-std-omega`: ghi de config.* TRUOC khi tao tracker."""
    saved = config.CTRV_INIT_STD_OMEGA
    try:
        config.CTRV_INIT_STD_OMEGA = 0.77
        tr = CTRVByteTracker(args=load_cfg("configs/bytetrack_ekf_ctrv.yaml"))
        _, P = tr.kalman_filter.initiate(np.array([1.0, 1.0, 0.5, 40.0]))
        _check("override config truoc khi tao tracker van co tac dung", np.isclose(P[4, 4], 0.77 ** 2))
    finally:
        config.CTRV_INIT_STD_OMEGA = saved


def test_7_late_binding_legacy_semantics():
    """Ngu nghia TRUOC Giai doan 2: initiate()/try_initiate_from_motion() doc config.* LUC GOI.

    Script cu (tune_omega_noise.py dong 64-67) tao EKFTrackerCTRV() roi MOI ghi de
    config.CTRV_INIT_STD_OMEGA. Bo loc khong truyen tham so tuong minh phai phan anh
    gia tri config tai thoi diem goi; bo loc truyen tuong minh thi co dinh.
    """
    saved = (config.CTRV_INIT_STD_OMEGA, config.CTRV_MIN_SPEED_FOR_HEADING)
    try:
        f_default = EKFTrackerCTRV()                    # tao TRUOC khi doi config
        f_pinned = EKFTrackerCTRV(init_std_omega=0.33)
        config.CTRV_INIT_STD_OMEGA = 0.55               # doi SAU khi tao (nhu tune_omega_noise.py)
        _, P0 = f_default.initiate(np.array([1.0, 1.0, 0.5, 40.0]))
        _, P1 = f_pinned.initiate(np.array([1.0, 1.0, 0.5, 40.0]))
        _check("late-binding: bo loc mac dinh doc config luc goi (0.55)", np.isclose(P0[4, 4], 0.55 ** 2))
        _check("tham so tuong minh co dinh (0.33), khong theo config", np.isclose(P1[4, 4], 0.33 ** 2))
        config.CTRV_MIN_SPEED_FOR_HEADING = 100.0       # ngat het: khong bo loc mac dinh nao khoi tao huong
        m, P = f_default.initiate(np.array([10.0, 10.0, 0.5, 40.0]))
        _, _, ok = f_default.try_initiate_from_motion(m, P, np.array([15.0, 10.0, 0.5, 40.0]), 1, (10, 10))
        f_fast = EKFTrackerCTRV(min_speed_for_heading=0.5)
        _, _, ok2 = f_fast.try_initiate_from_motion(m, P, np.array([15.0, 10.0, 0.5, 40.0]), 1, (10, 10))
        _check("late-binding min_speed_for_heading (mac dinh theo config, tuong minh co dinh)", ok is False and ok2)
        _check("explicit_params chi liet ke tham so tuong minh",
               f_pinned.explicit_params().get("init_std_omega") == 0.33 and "init_std_omega" not in f_default.explicit_params())
    finally:
        config.CTRV_INIT_STD_OMEGA, config.CTRV_MIN_SPEED_FOR_HEADING = saved


def test_8_single_predict_equals_batch_predict():
    """Du doan don (predict tung track) == du doan nhom (multi_predict) voi cung state/cau hinh/dt.

    EKF/UKF: multi_predict la vong lap predict -> bit-identical. CV (ultralytics
    KalmanFilterXYAH) vector hoa khac duong -> allclose 1e-12.
    """
    from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH
    rng = np.random.default_rng(3)
    cases = [("CV", KalmanFilterXYAH(), 8), ("EKF", EKFTrackerCTRV(std_theta=0.02), 9),
             ("UKF", UKFTrackerCTRV(std_omega=0.01), 9)]
    for name, kf, n in cases:
        if n == 8:
            base = np.array([300, 200, 0.5, 40, 2, -1, 0.001, 0.2])
        else:
            base = np.array([300, 200, 5, 0.4, 0.02, 0.5, 40, 0.001, 0.2])
        M = np.stack([base + rng.normal(0, 0.1, n) for _ in range(6)])
        C = np.stack([np.diag(rng.uniform(0.5, 5, n)) for _ in range(6)])
        bm, bc = kf.multi_predict(M.copy(), C.copy())
        ok = True
        for i in range(6):
            sm, sc = kf.predict(M[i].copy(), C[i].copy())
            if n == 8:
                ok &= np.allclose(sm, bm[i], atol=1e-12) and np.allclose(sc, bc[i], atol=1e-12)
            else:
                ok &= np.array_equal(sm, bm[i]) and np.array_equal(sc, bc[i])
        _check(f"[{name}] predict tung track == multi_predict (6 track)", ok)


def test_9_two_trackers_different_configs_same_process():
    """Hai tracker khac cau hinh chay XEN KE trong cung tien trinh khong anh huong nhau.

    Bo dem ID (BaseTrack._count) cua ultralytics la TOAN CUC -> ID cua tracker B khi
    chay xen ke khac khi chay mot minh (han che da biet, khong lien quan bo loc). Vi
    vay so sanh box theo frame (bo qua cot id) va hiep phuong sai theta cua track.
    """
    cfg_a = load_cfg("configs/bytetrack_ekf_ctrv.yaml", ctrv_std_theta=1.0)
    cfg_b = load_cfg("configs/bytetrack_ekf_ctrv.yaml")
    dets = [synthetic_dets(f).cpu().numpy() for f in range(1, 13)]

    solo_b = CTRVByteTracker(args=cfg_b); solo_b.reset()
    out_solo = [np.sort(solo_b.update(d, None).reshape(-1, 8)[:, :4].copy(), axis=0) for d in dets]

    A = CTRVByteTracker(args=cfg_a); A.reset()
    B = CTRVByteTracker(args=cfg_b); B.reset()
    out_a, out_b = [], []
    for d in dets:
        out_a.append(A.update(d, None).copy())              # xen ke A roi B
        out_b.append(np.sort(B.update(d, None).reshape(-1, 8)[:, :4].copy(), axis=0))
    _check("B chay xen ke voi A == B chay mot minh (box theo frame, bo qua ID)",
           all(np.array_equal(x, y) for x, y in zip(out_solo, out_b)))
    ta = (A.tracked_stracks + A.lost_stracks)[0]; tb = (B.tracked_stracks + B.lost_stracks)[0]
    _check("moi tracker giu bo loc rieng (instance khac nhau, tham so khac nhau)",
           ta.kalman_filter is not tb.kalman_filter and ta.kalman_filter._std_theta == 1.0
           and tb.kalman_filter._std_theta == config.CTRV_STD_THETA)
    # frame rong -> chi predict: P[theta] cua A tang ~1.0^2, cua B tang nho
    pa, pb = float(ta.covariance[3, 3]), float(tb.covariance[3, 3])
    A.update(synthetic_dets(4).cpu().numpy(), None); B.update(synthetic_dets(4).cpu().numpy(), None)
    da, db = float(ta.covariance[3, 3]) - pa, float(tb.covariance[3, 3]) - pb
    _check("sau 1 buoc predict: dP[theta] cua A ~1.0^2, cua B << (khong ro ri cau hinh)",
           da > 0.9 and db < 0.1, f"dA={da:.4f} dB={db:.4f}")
    _check("khong bien toan cuc nao bi ghi de: shared_kalman va config giu nguyen",
           CTRVSTrack.shared_kalman._std_theta == config.CTRV_STD_THETA == 0.010)


def main() -> int:
    tracker_ctrv.register()
    tests = [test_1_default_filter_is_bit_identical_to_shared_kalman,
             test_2_tracker_end_to_end_default_identical,
             test_3_yaml_override_reaches_multi_predict,
             test_4_init_params_on_instance,
             test_5_ctrv_kwargs_from_args,
             test_6_config_override_before_tracker_creation_still_works,
             test_7_late_binding_legacy_semantics,
             test_8_single_predict_equals_batch_predict,
             test_9_two_trackers_different_configs_same_process]
    failed = 0
    for t in tests:
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
