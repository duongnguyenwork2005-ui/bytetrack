"""
test_ukf_ctrv.py -- Unit test cho UKF + mo hinh CTRV.

    python src/test_ukf_ctrv.py          # in bao cao chi tiet
    pytest src/test_ukf_ctrv.py -q

CAC PHEP KIEM TRA
-----------------
 1. Sinh sigma point dung so luong (2n+1) va dung trong so
 2. Sigma point tai tao lai DUNG ky vong va hiep phuong sai ban dau
    (tinh chat co ban nhat cua Unscented Transform)
 3. Ham tuyen tinh -> UT phai cho ket qua CHINH XAC (khong sai so xap xi)
 4. Trung binh vong cho theta (goc vat qua bien +-pi)  <- loi kinh dien
 5. Predict tren quy dao cong: so voi tinh tay
 6. Lien tuc tai omega -> 0 (khong chia cho 0)
 7. Buoc update cua UKF trung khit voi EKF (vi mo hinh do tuyen tinh)
 8. P doi xung + xac dinh duong sau nhieu vong lap
 9. Tham so sigma point hong (alpha=1e-3 voi n=9) phai bao loi ro rang
10. Khoi tao 2 khung hinh (ke thua tu EKF) van hoat dong
11. UKF va EKF dung CHUNG mo hinh f(), Q, R -> chi khac cach truyen hiep phuong sai
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekf_ctrv import EKFTrackerCTRV, normalize_angle  # noqa: E402
from ukf_ctrv import UKFTrackerCTRV  # noqa: E402

CX, CY, V, TH, OM, A, H, VA, VH = range(9)
_RESULTS = []


def _check(name: str, ok: bool, detail: str = ""):
    _RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def make_state(cx=100.0, cy=200.0, v=10.0, theta=0.0, omega=0.0, a=0.5, h=40.0, va=0.0, vh=0.0):
    return np.array([cx, cy, v, theta, omega, a, h, va, vh], dtype=float)


def make_cov(scale=1.0, h=40.0):
    """Hiep phuong sai THUC TE (co tuong quan cheo), khong phai ma tran ngau nhien.

    LUU Y QUAN TRONG: khong duoc dung hiep phuong sai ngau nhien voi phuong sai
    theta lon (vai rad^2). Ly do: sigma point trai ra +-3*std quanh ky vong, neu
    std(theta) ~ 2 rad thi cac diem trai toi +-6 rad, vuot xa mot vong tron.
    Luc do phep goi goc ve [-pi, pi) lam mat thong tin va Unscented Transform
    KHONG con tai tao lai dung hiep phuong sai ban dau nua.
    Day khong phai loi cai dat - do la gioi han co ban cua bien goc: mot huong
    di chuyen "khong biet gi ca" khong the mo ta bang phan bo Gaussian.
    Trong thuc te theta luon duoc khoi tao tu 2 quan sat (std ~ 0.3 rad) nen
    khong bao gio roi vao vung nay.
    """
    std = np.array([
        2 * h / 20,      # cx
        2 * h / 20,      # cy
        10 * h / 160,    # v
        0.30,            # theta - sau khoi tao 2 khung hinh
        0.10,            # omega
        1e-2,            # a
        2 * h / 20,      # h
        1e-5,            # va
        10 * h / 160,    # vh
    ]) * np.sqrt(scale)

    # Them tuong quan cheo vua phai de test tong quat hon ma tran duong cheo
    n = len(std)
    C = np.eye(n)
    rng = np.random.default_rng(42)
    for i in range(n):
        for j in range(i + 1, n):
            C[i, j] = C[j, i] = rng.uniform(-0.25, 0.25)
    # Ep C ve ma tran tuong quan xac dinh duong
    w, V = np.linalg.eigh(C)
    C = V @ np.diag(np.clip(w, 0.05, None)) @ V.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)

    D = np.diag(std)
    return D @ C @ D


# ---------------------------------------------------------------------------
# 1-2. Tinh chat co ban cua Unscented Transform
# ---------------------------------------------------------------------------
def test_sigma_point_count_and_weights():
    ukf = UKFTrackerCTRV()
    sig = ukf.sigma_points(make_state(), make_cov())
    ok = (sig.shape == (19, 9) and np.isclose(ukf.Wm.sum(), 1.0)
          and np.all(ukf.Wc >= 0) and np.all(ukf.Wm >= 0))
    _check("1. Sinh 2n+1 = 19 sigma point, trong so hop le", ok,
           f"shape = {sig.shape} (phai la (19, 9))\n"
           f"sum(Wm) = {ukf.Wm.sum():.10f} (phai = 1)\n"
           f"Wm[0] = {ukf.Wm[0]:.5f}, Wc[0] = {ukf.Wc[0]:.5f}, Wi = {ukf.Wm[1]:.5f}\n"
           f"moi trong so khong am: {np.all(ukf.Wm >= 0) and np.all(ukf.Wc >= 0)} "
           f"-> P luon xac dinh duong")


def test_sigma_points_recover_moments():
    """Sigma point phai tai tao lai DUNG ky vong va hiep phuong sai ban dau.

    Day la tinh chat dinh nghia cua Unscented Transform. Neu sai o day thi
    moi thu phia sau deu sai.
    """
    ukf = UKFTrackerCTRV()
    mean, cov = make_state(theta=0.4), make_cov(0.5)
    sig = ukf.sigma_points(mean, cov)

    m_rec = ukf._state_mean(sig, ukf.Wm)
    d = ukf._state_residual(sig, m_rec)
    c_rec = (d * ukf.Wc[:, None]).T @ d

    err_m = np.max(np.abs(m_rec - mean))
    err_c = np.max(np.abs(c_rec - cov))
    ok = err_m < 1e-9 and err_c < 1e-8
    _check("2. Sigma point tai tao dung ky vong va hiep phuong sai", ok,
           f"sai lech ky vong      : {err_m:.3e}\n"
           f"sai lech hiep phuong sai: {err_c:.3e}")


def test_linear_function_exact():
    """Voi ham TUYEN TINH, Unscented Transform phai cho ket qua CHINH XAC.

    Dat omega = 0 va v = 0 thi f() suy bien thanh phep bien doi tuyen tinh
    (chi con khoi kich thuoc a' = a + va, h' = h + vh). Khi do UT phai trung
    khit voi phep truyen hiep phuong sai tuyen tinh F P F^T.
    """
    ukf = UKFTrackerCTRV()
    ekf = EKFTrackerCTRV()
    mean = make_state(v=0.0, omega=0.0, va=0.02, vh=0.5)
    cov = np.diag([4.0, 4.0, 1e-6, 1e-6, 1e-9, 0.01, 4.0, 1e-4, 0.25])

    m_u, c_u = ukf.predict(mean, cov)
    m_e, c_e = ekf.predict(mean, cov)

    err_m = np.max(np.abs(m_u - m_e))
    err_c = np.max(np.abs(c_u - c_e))
    ok = err_m < 1e-8 and err_c < 1e-6
    _check("3. Ham tuyen tinh: UKF trung khit EKF (khong sai so xap xi)", ok,
           f"sai lech ky vong      : {err_m:.3e}\n"
           f"sai lech hiep phuong sai: {err_c:.3e}")


# ---------------------------------------------------------------------------
# 4. Trung binh vong cho theta - loi kinh dien
# ---------------------------------------------------------------------------
def test_circular_mean():
    """Goc +179 do va -179 do cach nhau 2 do, trung binh phai la +-180 do.

    Trung binh cong thong thuong se cho 0 do - lech 180 do. Day la loi kinh
    dien cua UKF co bien goc.
    """
    ukf = UKFTrackerCTRV()
    sig = np.tile(make_state(), (19, 1))
    sig[:, TH] = 0.0
    sig[1:10, TH] = np.radians(179.0)
    sig[10:, TH] = np.radians(-179.0)
    w = np.full(19, 1.0 / 19)

    m = ukf._state_mean(sig, w)
    naive = float(w @ sig[:, TH])       # trung binh cong (SAI)
    got = float(m[TH])
    # Ket qua dung phai gan +-pi (180 do), khong phai 0
    ok = abs(abs(got) - np.pi) < np.radians(11.0)
    _check("4. Trung binh vong cho theta (goc vat bien +-pi)", ok,
           f"trung binh cong thong thuong: {np.degrees(naive):+8.2f} do  <- SAI\n"
           f"trung binh vong (dung dung) : {np.degrees(got):+8.2f} do  (mong doi ~ +-180)")


def test_residual_wraps():
    """Hieu goc phai duoc goi ve [-pi, pi) truoc khi lap hiep phuong sai."""
    ukf = UKFTrackerCTRV()
    sig = np.tile(make_state(), (3, 1))
    sig[0, TH] = np.radians(179.0)
    sig[1, TH] = np.radians(-179.0)
    sig[2, TH] = np.radians(180.0)
    mean = make_state(theta=np.radians(180.0))

    d = ukf._state_residual(sig, mean)
    ok = np.all(np.abs(d[:, TH]) <= np.pi + 1e-12) and abs(np.degrees(d[0, TH]) - (-1.0)) < 1e-6
    _check("4b. Hieu goc duoc goi ve [-pi, pi)", ok,
           "hieu (do): " + ", ".join(f"{np.degrees(x):+.2f}" for x in d[:, TH]) +
           "  (phai la -1, +1, 0 chu khong phai -359, +359, 0)")


# ---------------------------------------------------------------------------
# 5-6. Mo hinh chuyen dong
# ---------------------------------------------------------------------------
def test_predict_curved_mean():
    """Voi hiep phuong sai rat nho, ky vong sau predict phai bam sat f(x) tinh tay.

    v=10, theta=0, omega=pi/2 -> cx' = 100 + 20/pi, cy' = 200 + 20/pi (nhu EKF test 3).
    """
    ukf = UKFTrackerCTRV()
    x = make_state(cx=100, cy=200, v=10, theta=0.0, omega=np.pi / 2)
    P = np.eye(9) * 1e-10
    m, _ = ukf.predict(x, P)

    r = 20.0 / np.pi
    ok = (abs(m[CX] - (100 + r)) < 1e-3 and abs(m[CY] - (200 + r)) < 1e-3
          and abs(m[TH] - np.pi / 2) < 1e-6)
    _check("5. Predict quy dao cong khop tinh tay", ok,
           f"tinh tay : cx={100+r:.6f}, cy={200+r:.6f}, theta={np.pi/2:.6f}\n"
           f"UKF      : cx={m[CX]:.6f}, cy={m[CY]:.6f}, theta={m[TH]:.6f}")


def test_omega_zero_no_nan():
    """omega = 0 chinh xac: khong duoc sinh NaN/inf (chia cho 0).

    Sigma point trai rong quanh omega = 0 nen se co diem omega am, diem omega
    duong va diem omega ~ 0 - ham f() phai xu ly dung tung diem mot.
    """
    ukf = UKFTrackerCTRV()
    x = make_state(v=12.0, theta=0.7, omega=0.0)
    P = make_cov(0.2)
    m, c = ukf.predict(x, P)
    sig = ukf.sigma_points(x, P)
    omegas = sig[:, OM]
    ok = np.all(np.isfinite(m)) and np.all(np.isfinite(c))
    _check("6. omega = 0: khong sinh NaN/inf", ok,
           f"omega cua cac sigma point trai tu {omegas.min():+.4f} den {omegas.max():+.4f}\n"
           f"  (co ca diem am, duong va gan 0 -> f() phai xu ly tung diem)\n"
           f"ky vong finite = {np.all(np.isfinite(m))}, hiep phuong sai finite = {np.all(np.isfinite(c))}")


# ---------------------------------------------------------------------------
# 7. Buoc update
# ---------------------------------------------------------------------------
def test_update_matches_ekf():
    """Vi mo hinh do TUYEN TINH, buoc update cua UKF phai trung khit voi EKF."""
    ukf, ekf = UKFTrackerCTRV(), EKFTrackerCTRV()
    mean = make_state(v=8.0, theta=0.5, omega=0.05)
    cov = make_cov(0.3)
    z = np.array([105.0, 203.0, 0.52, 41.0])

    m_u, c_u = ukf.update(mean, cov, z)
    m_e, c_e = ekf.update(mean, cov, z)

    err_m = np.max(np.abs(m_u - m_e))
    err_c = np.max(np.abs(c_u - c_e))
    ok = err_m < 1e-8 and err_c < 1e-6
    _check("7. Update cua UKF trung khit EKF (mo hinh do tuyen tinh)", ok,
           f"sai lech ky vong      : {err_m:.3e}\n"
           f"sai lech hiep phuong sai: {err_c:.3e}\n"
           "=> moi khac biet ve ket qua giua EKF va UKF deu den tu buoc PREDICT")


# ---------------------------------------------------------------------------
# 8. On dinh so hoc
# ---------------------------------------------------------------------------
def test_covariance_stays_valid():
    """Sau nhieu vong predict/update, P phai luon doi xung va xac dinh duong."""
    rng = np.random.default_rng(0)
    ukf = UKFTrackerCTRV()
    mean, cov = ukf.initiate(np.array([100.0, 200.0, 0.5, 40.0]))
    mean, cov = ukf.initiate_from_motion(mean, cov, np.array([104.0, 200.0, 0.5, 40.0]))

    worst_asym, min_eig = 0.0, np.inf
    for t in range(200):
        mean, cov = ukf.predict(mean, cov)
        z = np.array([100 + 4 * t, 200 + 25 * np.sin(t / 25), 0.5, 40.0]) + rng.normal(0, 0.5, 4)
        mean, cov = ukf.update(mean, cov, z)
        worst_asym = max(worst_asym, float(np.max(np.abs(cov - cov.T))))
        min_eig = min(min_eig, float(np.min(np.linalg.eigvalsh(cov))))

    ok = worst_asym < 1e-8 and min_eig > 0 and np.all(np.isfinite(mean))
    _check("8. P doi xung + xac dinh duong sau 200 vong lap", ok,
           f"do bat doi xung lon nhat: {worst_asym:.3e}\n"
           f"tri rieng nho nhat: {min_eig:.3e} (phai > 0)")


# ---------------------------------------------------------------------------
# 9. Tham so sigma point hong
# ---------------------------------------------------------------------------
def test_bad_alpha_raises():
    """alpha = 1e-3 (mac dinh sach giao khoa) lam n + lambda = 0 khi n = 9.

    Phai bao loi RO RANG thay vi chay tiep roi sinh ra ket qua vo nghia.
    """
    try:
        UKFTrackerCTRV(alpha=1e-3, kappa=0.0)
        ok, msg = False, "KHONG bao loi - nguy hiem!"
    except ValueError as e:
        ok, msg = "chia cho 0" in str(e), str(e)[:120]
    _check("9. alpha=1e-3 voi n=9 bi bat loi ro rang", ok, msg)


# ---------------------------------------------------------------------------
# 10-11. Ke thua va tinh cong bang
# ---------------------------------------------------------------------------
def test_two_frame_init_inherited():
    """Khoi tao 2 khung hinh ke thua tu EKF phai hoat dong y het."""
    ukf, ekf = UKFTrackerCTRV(), EKFTrackerCTRV()
    z1 = np.array([100.0, 200.0, 0.5, 40.0])
    z2 = np.array([103.0, 204.0, 0.5, 40.0])
    mu, cu = ukf.initiate_from_motion(*ukf.initiate(z1), z2)
    me, ce = ekf.initiate_from_motion(*ekf.initiate(z1), z2)
    ok = np.allclose(mu, me) and np.allclose(cu, ce)
    _check("10. Khoi tao 2 khung hinh giong het EKF", ok,
           f"v = {mu[V]:.6f} (EKF: {me[V]:.6f}), theta = {np.degrees(mu[TH]):.2f} do")


def test_same_model_and_noise_as_ekf():
    """UKF va EKF phai dung CHUNG f(), Q, R -> dam bao so sanh cong bang."""
    ukf, ekf = UKFTrackerCTRV(), EKFTrackerCTRV()
    x = make_state(v=7.0, theta=0.3, omega=0.08, h=52.0)

    same_f = np.allclose(ukf.f(x), ekf.f(x))
    same_q = np.allclose(ukf._process_noise(x[H]), ekf._process_noise(x[H]))
    same_r = np.allclose(ukf._measurement_noise(x[H]), ekf._measurement_noise(x[H]))
    ok = same_f and same_q and same_r
    _check("11. UKF dung chung f(), Q, R voi EKF (so sanh cong bang)", ok,
           f"cung ham chuyen trang thai f(x) : {same_f}\n"
           f"cung nhieu qua trinh Q          : {same_q}\n"
           f"cung nhieu do R                 : {same_r}")


def main() -> int:
    print("=" * 78)
    print("UNIT TEST: UKF + mo hinh CTRV")
    print("=" * 78)
    tests = [
        test_sigma_point_count_and_weights, test_sigma_points_recover_moments,
        test_linear_function_exact,
        test_circular_mean, test_residual_wraps,
        test_predict_curved_mean, test_omega_zero_no_nan,
        test_update_matches_ekf,
        test_covariance_stays_valid,
        test_bad_alpha_raises,
        test_two_frame_init_inherited, test_same_model_and_noise_as_ekf,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError:
            failed += 1
        print()

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    print("=" * 78)
    print(f"KET QUA: {n_pass}/{len(_RESULTS)} phep kiem tra PASS")
    print("=" * 78)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
