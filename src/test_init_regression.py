"""
test_init_regression.py -- Regression test cho buoc khoi tao chuyen dong va covariance.

Bao ve hai loi da tung xay ra (xem src/repro_init_covariance.py va FIX_REPORT.md):

  LOI 1  initiate_from_motion ghi de phan tu duong cheo P[V,V], P[THETA,THETA]
         nhung giu nguyen phan tu ngoai duong cheo -> P mat tinh ban xac dinh
         duong. Do duoc: sau 26 buoc predict, tri rieng nho nhat tu +2,7e-9
         xuong -2,2e-1, roi phan ky toi -703 trong luc ngoai suy mu.

  LOI 2  Tracker danh dau `_motion_initialized = True` ke ca khi bo loc BAO
         khong khoi tao duoc (xe dung yen). Hau qua: xe dung yen luc dau roi
         moi chay se khong bao gio duoc khoi tao huong. Do duoc: UKF ket han o
         v = 0,14 px/frame sau 6 frame chay that (dung phai ~8).

CAC KICH BAN KIEM TRA
   1. Nhieu buoc predict truoc khi co quan sat du de khoi tao (N = 1, 5, 10, 26)
   2. Xe dung yen roi chay theo phuong DOC va phuong NGANG
   3. Mat track roi tai kich hoat (re_activate)
   4. mean/covariance huu han, P doi xung, khong co tri rieng am vuot dung sai
   5. UKF tao duoc sigma point tu covariance sau khoi tao

    python src/test_init_regression.py       # in bao cao tung phep kiem tra
    pytest src/test_init_regression.py -q    # neu co pytest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV, set_marginal_variance  # noqa: E402
from ukf_ctrv import UKFTrackerCTRV  # noqa: E402

FILTERS = [EKFTrackerCTRV, UKFTrackerCTRV]
Z0 = np.array([100.0, 200.0, 0.5, 40.0])
# Dung sai cho tri rieng am: chi chap nhan muc sai so lam tron cua float64 tren
# ma tran co bien do ~1e3, KHONG phai mot nguong rong de bo qua loi that.
EIG_TOL = 1e-8

_RESULTS: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = ""):
    _RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def eig_min(P: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(0.5 * (P + P.T)).min())


def assert_valid_cov(P: np.ndarray, where: str) -> str:
    """Kiem tra P huu han, doi xung, khong co tri rieng am vuot dung sai."""
    problems = []
    if not np.all(np.isfinite(P)):
        problems.append("co phan tu khong huu han")
    a = float(np.abs(P - P.T).max())
    if a > 1e-9:
        problems.append(f"bat doi xung {a:.2e}")
    e = eig_min(P)
    if e < -EIG_TOL:
        problems.append(f"tri rieng am {e:.3e}")
    return f"{where}: " + ("; ".join(problems) if problems else
                           f"OK (tri rieng min {e:+.3e}, bat doi xung {a:.1e})")


# ---------------------------------------------------------------------------
# 1. Nhieu buoc predict truoc khi khoi tao chuyen dong
# ---------------------------------------------------------------------------
def test_predict_then_init_keeps_psd():
    """Predict N buoc roi moi khoi tao -- P phai van ban xac dinh duong.

    Day la LOI 1. Cang nhieu buoc predict thi hiep phuong sai cheo giua v/theta
    va vi tri cang lon, nen viec ghi de rieng phan tu duong cheo cang pha vo
    tinh ban xac dinh duong. N = 26 la truong hop that gap trong du lieu
    (MVI_40992 track 12 bi che 26 frame ngay sau khi sinh).
    """
    for FilterCls in FILTERS:
        lines = []
        worst = 0.0
        for N in (1, 5, 10, 26):
            f = FilterCls()
            m, P = f.initiate(Z0)
            for _ in range(N):
                m, P = f.predict(m, P)
            e_before = eig_min(P)
            z1 = np.array([100.0 + 3 * N, 200.0 + 4 * N, 0.5, 40.0])
            m2, P2, ok = f.try_initiate_from_motion(m, P, z1, n_frames=float(N))
            e_after = eig_min(P2)
            worst = min(worst, e_after)
            lines.append(f"N={N:>2}: tri rieng {e_before:+.3e} -> {e_after:+.3e}, "
                         f"ok={ok}, v={m2[f.V]:.3f}, theta={np.rad2deg(m2[f.THETA]):.2f}o")
            assert ok, f"{FilterCls.__name__} N={N}: dang le phai khoi tao duoc (speed=5)"
            assert abs(m2[f.V] - 5.0) < 1e-6, f"v sai: {m2[f.V]}"
            assert abs(np.rad2deg(m2[f.THETA]) - 53.13010235) < 1e-6
        _check(f"1. Predict N buoc roi khoi tao, P van PSD [{FilterCls.__name__}]",
               worst >= -EIG_TOL, "\n".join(lines))


def test_congruence_preserves_eigen_sign():
    """set_marginal_variance la phep dong dang -> KHONG doi dau tri rieng nao.

    Kiem tra truc tiep tinh chat toan hoc duoc dua ra lam co so cho cach sua,
    thay vi chi kiem tra ket qua cuoi.
    """
    rng = np.random.default_rng(0)
    bad = []
    for trial in range(20):
        A = rng.normal(size=(9, 9))
        P = A @ A.T + 1e-6 * np.eye(9)          # doi xung, xac dinh duong
        sign_before = np.sign(np.linalg.eigvalsh(P))
        Q = set_marginal_variance(P.copy(), idx=3, new_var=0.09)
        sign_after = np.sign(np.linalg.eigvalsh(Q))
        if not np.array_equal(sign_before, sign_after):
            bad.append(trial)
    _check("1b. set_marginal_variance giu nguyen dau moi tri rieng (Sylvester)",
           not bad, f"20 ma tran ngau nhien 9x9, so lan doi dau: {len(bad)}")


# ---------------------------------------------------------------------------
# 2. Xe dung yen roi chay -- phuong doc va phuong ngang
# ---------------------------------------------------------------------------
def _stationary_then_move(FilterCls, axis: str, n_static: int = 3):
    """Mo phong dung luong logic tracker DA SUA: thu lai den khi khoi tao duoc."""
    f = FilterCls()
    m, P = f.initiate(Z0)
    initialized = False
    last_obs = (float(Z0[0]), float(Z0[1]))
    cx, cy = 100.0, 200.0

    for _ in range(n_static):                    # dung yen: chi rung 0.1 px
        cx += 0.1
        z = np.array([cx, cy, 0.5, 40.0])
        if not initialized:
            m, P, ok = f.try_initiate_from_motion(m, P, z, n_frames=1.0, ref_pos=last_obs)
            initialized = ok
        m, P = f.predict(m, P)
        m, P = f.update(m, P, z)
        last_obs = (cx, cy)

    for _ in range(8):                           # chay that 8 px/frame
        if axis == "y":
            cy += 8.0
        else:
            cx += 8.0
        z = np.array([cx, cy, 0.5, 40.0])
        if not initialized:
            m, P, ok = f.try_initiate_from_motion(m, P, z, n_frames=1.0, ref_pos=last_obs)
            initialized = ok
        m, P = f.predict(m, P)
        m, P = f.update(m, P, z)
        last_obs = (cx, cy)
    return m, P, initialized


def test_stationary_then_move():
    """LOI 2: xe dung yen truoc roi moi chay phai hoc duoc ca v lan theta.

    Kiem tra CA HAI truc: voi truc x, theta khoi tao = 0 tinh co dung nen loi
    bi che giau; chi truc y moi lo ra. Do do phai test ca hai.
    """
    for FilterCls in FILTERS:
        for axis, want_theta in (("y", 90.0), ("x", 0.0)):
            m, P, ok = _stationary_then_move(FilterCls, axis)
            f = FilterCls()
            v = float(m[f.V])
            th = float(np.rad2deg(m[f.THETA]))
            dth = (th - want_theta + 180.0) % 360.0 - 180.0
            _check(f"2. Dung yen roi chay theo truc {axis} [{FilterCls.__name__}]",
                   ok and abs(v - 8.0) < 1.5 and abs(dth) < 15.0,
                   f"da khoi tao = {ok}, v = {v:.4f} (mong doi ~8.0), "
                   f"theta = {th:.2f}o (mong doi {want_theta:.0f}o)\n"
                   + assert_valid_cov(P, "P sau kich ban"))


def test_never_initialized_when_always_static():
    """Xe dung yen SUOT: khong duoc bia ra huong, va P phai van hop le."""
    for FilterCls in FILTERS:
        f = FilterCls()
        m, P = f.initiate(Z0)
        initialized = False
        last_obs = (float(Z0[0]), float(Z0[1]))
        for _ in range(10):
            z = np.array([100.0, 200.0, 0.5, 40.0])
            if not initialized:
                m, P, ok = f.try_initiate_from_motion(m, P, z, n_frames=1.0, ref_pos=last_obs)
                initialized = ok
            m, P = f.predict(m, P)
            m, P = f.update(m, P, z)
        _check(f"2b. Dung yen suot -> khong bia ra huong [{FilterCls.__name__}]",
               (not initialized) and abs(float(m[f.V])) < config.CTRV_MIN_SPEED_FOR_HEADING,
               f"da khoi tao = {initialized} (phai False), v = {float(m[f.V]):.4f}\n"
               + assert_valid_cov(P, "P sau 10 frame dung yen"))


# ---------------------------------------------------------------------------
# 3. Mat track roi tai kich hoat
# ---------------------------------------------------------------------------
def test_reactivate_after_gap():
    """Track bi mat n frame roi khop lai: van toc phai chia dung so frame.

    Neu truyen n_frames = 1 trong khi thuc te cach 27 frame thi van toc bi thoi
    phong dung 27 lan (loi da tung xay ra o cac script demo).
    """
    for FilterCls in FILTERS:
        f = FilterCls()
        m, P = f.initiate(Z0)
        gap = 27
        for _ in range(gap):                 # ngoai suy mu, chua khoi tao duoc
            m, P = f.predict(m, P)
        # Xe da di 3*gap, 4*gap px trong `gap` frame -> toc do that = 5 px/frame
        z = np.array([100.0 + 3 * gap, 200.0 + 4 * gap, 0.5, 40.0])
        m2, P2, ok = f.try_initiate_from_motion(m, P, z, n_frames=float(gap),
                                                ref_pos=(100.0, 200.0))
        v = float(m2[f.V])
        _check(f"3. Tai kich hoat sau {gap} frame mat dau [{FilterCls.__name__}]",
               ok and abs(v - 5.0) < 1e-6,
               f"v = {v:.4f} (dung phai 5.0; neu truyen nham n_frames=1 se ra {5*gap:.1f})\n"
               + assert_valid_cov(P2, "P sau tai kich hoat"))


def test_reactivate_uses_observed_not_predicted():
    """Moc phai la vi tri QUAN SAT, khong phai vi tri da predict.

    Khi v da khac 0, predict lam dich vi tri trong `mean`. Neu lay `mean` lam
    moc thi dx, dy thanh phan du sau predict chu khong phai do dich chuyen.
    """
    for FilterCls in FILTERS:
        f = FilterCls()
        m, P = f.initiate(Z0)
        m[f.V] = 6.0                         # gia lap da co van toc tu truoc
        m[f.THETA] = 0.0
        m, P = f.predict(m, P)               # predict lam cx dich +6
        z = np.array([110.0, 200.0, 0.5, 40.0])
        # Moc DUNG = vi tri quan sat truoc do (100, 200) -> dich 10 px
        _, _, ok_ref = f.try_initiate_from_motion(m, P, z, n_frames=1.0,
                                                  ref_pos=(100.0, 200.0))
        m_ref, _, _ = f.try_initiate_from_motion(m, P, z, n_frames=1.0,
                                                 ref_pos=(100.0, 200.0))
        # Moc SAI = vi tri da predict (106, 200) -> chi con 4 px
        m_bad, _, _ = f.try_initiate_from_motion(m, P, z, n_frames=1.0, ref_pos=None)
        v_ref, v_bad = float(m_ref[f.V]), float(m_bad[f.V])
        # Khong khang dinh v_bad bang dung 4.0: UKF lay trung binh cac diem sigma
        # da truyen qua ham phi tuyen nen vi tri predict khong dung bang cx + v*dt.
        # Dieu can bao ve la TINH CHAT: lay moc predict cho van toc thap hon han.
        _check(f"3b. Dung vi tri quan sat lam moc, khong dung vi tri predict "
               f"[{FilterCls.__name__}]",
               ok_ref and abs(v_ref - 10.0) < 1e-9 and v_bad < 6.0,
               f"moc = quan sat  -> v = {v_ref:.4f} (dung, dich that 10 px)\n"
               f"moc = predict   -> v = {v_bad:.4f} (sai, chi con phan du sau predict)")


# ---------------------------------------------------------------------------
# 4-5. P hop le va UKF tao duoc sigma point
# ---------------------------------------------------------------------------
def test_ukf_sigma_points_after_init():
    """UKF phai tao duoc sigma point tu P sau khoi tao.

    UKF phan ra Cholesky ma tran hiep phuong sai; P khong ban xac dinh duong se
    lam buoc nay hong hoac tra ve NaN. Day la ly do LOI 1 nguy hiem voi UKF hon
    la voi EKF.
    """
    f = UKFTrackerCTRV()
    lines = []
    for N in (1, 5, 10, 26):
        m, P = f.initiate(Z0)
        for _ in range(N):
            m, P = f.predict(m, P)
        z = np.array([100.0 + 3 * N, 200.0 + 4 * N, 0.5, 40.0])
        m2, P2, _ = f.try_initiate_from_motion(m, P, z, n_frames=float(N))
        # Buoc predict tiep theo cua UKF phai tao duoc sigma point tu P2
        m3, P3 = f.predict(m2, P2)
        good = np.all(np.isfinite(m3)) and np.all(np.isfinite(P3))
        lines.append(f"N={N:>2}: {assert_valid_cov(P3, 'P sau predict')} | "
                     f"mean huu han = {bool(np.all(np.isfinite(m3)))}")
        assert good, f"N={N}: UKF sinh NaN/inf sau khoi tao"
    _check("5. UKF tao duoc sigma point tu P sau khoi tao", True, "\n".join(lines))


def test_long_run_covariance_stays_valid():
    """Chay dai qua nhieu doan che: P phai luon huu han, doi xung, PSD."""
    for FilterCls in FILTERS:
        f = FilterCls()
        m, P = f.initiate(Z0)
        initialized = False
        last_obs = (float(Z0[0]), float(Z0[1]))
        cx, cy = 100.0, 200.0
        worst = 0.0
        for step in range(120):
            cx += 3.0
            cy += 4.0
            z = np.array([cx, cy, 0.5, 40.0])
            visible = not (30 <= step < 70)        # doan che 40 frame o giua
            if not initialized and visible:
                m, P, ok = f.try_initiate_from_motion(m, P, z, n_frames=1.0, ref_pos=last_obs)
                initialized = ok
            m, P = f.predict(m, P)
            if visible:
                m, P = f.update(m, P, z)
                last_obs = (cx, cy)
            worst = min(worst, eig_min(P))
            assert np.all(np.isfinite(m)) and np.all(np.isfinite(P)), f"NaN o buoc {step}"
        _check(f"4. Chay 120 frame qua doan che 40 frame, P luon hop le "
               f"[{FilterCls.__name__}]",
               worst >= -EIG_TOL,
               f"tri rieng nho nhat trong suot lan chay: {worst:+.3e} "
               f"(dung sai {-EIG_TOL:.0e})\n" + assert_valid_cov(P, "P cuoi"))


def main() -> int:
    print("=" * 78)
    print("REGRESSION TEST -- khoi tao chuyen dong va covariance")
    print("=" * 78)
    for fn in (test_predict_then_init_keeps_psd,
               test_congruence_preserves_eigen_sign,
               test_stationary_then_move,
               test_never_initialized_when_always_static,
               test_reactivate_after_gap,
               test_reactivate_uses_observed_not_predicted,
               test_ukf_sigma_points_after_init,
               test_long_run_covariance_stays_valid):
        print(f"\n--- {fn.__name__} ---")
        fn()
    n_ok = sum(1 for _, ok, _ in _RESULTS if ok)
    print("\n" + "=" * 78)
    print(f"KET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    print("=" * 78)
    return 0 if n_ok == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
