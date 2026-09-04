"""
test_ekf_ctrv.py -- Unit test cho EKF + mo hinh CTRV.

Chay truoc khi tich hop vao pipeline, de chac chan phan TOAN hoc dung da.

    python src/test_ekf_ctrv.py          # in bao cao chi tiet tung phep kiem tra
    pytest src/test_ekf_ctrv.py -q       # neu co pytest

CAC PHEP KIEM TRA
-----------------
 1. Di thang (omega = 0), huong theta = 0     -> so sanh voi tinh tay
 2. Di thang (omega = 0), huong theta = pi/2  -> so sanh voi tinh tay
 3. Quy dao cong (omega = pi/2)               -> so sanh voi tinh tay
 4. Lien tuc tai omega -> 0                   -> cong thuc cung va gioi han phai khop
 5. Jacobian giai tich vs sai phan so         -> ca truong hop cong va truong hop thang
 6. Di het 1 vong tron thi ve dung cho cu     -> kiem tra hinh hoc tong the
 7. Update voi nhieu do rat nho               -> hau nghiem phai bam sat quan sat
 8. P luon doi xung va xac dinh duong         -> on dinh so hoc qua nhieu vong lap
 9. Khoi tao 2 khung hinh                     -> uoc luong dung v va theta
10. Khoi kich thuoc (a, h) hoat dong dung CV  -> giong baseline
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekf_ctrv import EKFTrackerCTRV, normalize_angle  # noqa: E402

CX, CY, V, TH, OM, A, H, VA, VH = range(9)

# Bo dem ket qua khi chay truc tiep (khong qua pytest)
_RESULTS = []


def _check(name: str, ok: bool, detail: str = ""):
    _RESULTS.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def make_state(cx=100.0, cy=200.0, v=10.0, theta=0.0, omega=0.0, a=0.5, h=40.0, va=0.0, vh=0.0):
    return np.array([cx, cy, v, theta, omega, a, h, va, vh], dtype=float)


# ---------------------------------------------------------------------------
# 1-2. Di thang: cong thuc gioi han
# ---------------------------------------------------------------------------
def test_straight_theta_zero():
    """omega = 0, theta = 0, v = 10 -> chi di theo truc x, moi frame 10 px."""
    ekf = EKFTrackerCTRV()
    x = make_state(cx=100, cy=200, v=10, theta=0.0, omega=0.0)
    out = ekf.f(x)

    # Tinh tay: cx' = 100 + 10*cos(0)*1 = 110 ; cy' = 200 + 10*sin(0)*1 = 200
    exp_cx, exp_cy = 110.0, 200.0
    ok = np.isclose(out[CX], exp_cx) and np.isclose(out[CY], exp_cy)
    _check("1. Di thang theta=0", ok,
           f"tinh tay : cx=110.0, cy=200.0\nEKF tra ve: cx={out[CX]:.6f}, cy={out[CY]:.6f}")


def test_straight_theta_90():
    """omega = 0, theta = pi/2 -> chi di theo truc y."""
    ekf = EKFTrackerCTRV()
    x = make_state(cx=100, cy=200, v=10, theta=np.pi / 2, omega=0.0)
    out = ekf.f(x)

    # Tinh tay: cx' = 100 + 10*cos(pi/2) = 100 ; cy' = 200 + 10*sin(pi/2) = 210
    ok = np.isclose(out[CX], 100.0) and np.isclose(out[CY], 210.0)
    _check("2. Di thang theta=90 do", ok,
           f"tinh tay : cx=100.0, cy=210.0\nEKF tra ve: cx={out[CX]:.6f}, cy={out[CY]:.6f}")


# ---------------------------------------------------------------------------
# 3. Quy dao cong - vi du tinh tay day du
# ---------------------------------------------------------------------------
def test_curved_quarter_turn():
    """v=10, theta=0, omega=pi/2 (quay 90 do moi frame).

    Tinh tay:
        v/omega = 10 / (pi/2) = 20/pi = 6.366197723675814
        s1 = sin(0 + pi/2) = 1 ,  s0 = sin(0) = 0
        c1 = cos(pi/2) = 0     ,  c0 = cos(0) = 1
        cx' = 100 + (20/pi)*(1 - 0) = 106.366197723675814
        cy' = 200 + (20/pi)*(1 - 0) = 206.366197723675814
        theta' = 0 + pi/2 = pi/2
    """
    ekf = EKFTrackerCTRV()
    x = make_state(cx=100, cy=200, v=10, theta=0.0, omega=np.pi / 2)
    out = ekf.f(x)

    r = 20.0 / np.pi
    exp_cx, exp_cy, exp_th = 100.0 + r, 200.0 + r, np.pi / 2
    ok = (np.isclose(out[CX], exp_cx) and np.isclose(out[CY], exp_cy)
          and np.isclose(out[TH], exp_th))
    _check("3. Quy dao cong omega=90 do/frame", ok,
           f"tinh tay : cx={exp_cx:.9f}, cy={exp_cy:.9f}, theta={exp_th:.9f}\n"
           f"EKF tra ve: cx={out[CX]:.9f}, cy={out[CY]:.9f}, theta={out[TH]:.9f}")


# ---------------------------------------------------------------------------
# 4. Lien tuc tai omega -> 0 (kiem tra xu ly chia cho 0)
# ---------------------------------------------------------------------------
def test_omega_continuity():
    """f(x) voi omega rat nho phai gan trung f(x) voi omega = 0 chinh xac.

    Day la phep kiem tra quan trong nhat cho viec xu ly diem ky di (v/omega).
    """
    ekf = EKFTrackerCTRV()
    theta = 0.7  # goc bat ky, khac 0 de test tong quat
    x0 = make_state(v=12.0, theta=theta, omega=0.0)
    ref = ekf.f(x0)

    rows, worst = [], 0.0
    for omega in [1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
        x = make_state(v=12.0, theta=theta, omega=omega)
        out = ekf.f(x)
        d = max(abs(out[CX] - ref[CX]), abs(out[CY] - ref[CY]))
        worst = max(worst, d if omega <= 1e-4 else 0.0)
        rows.append(f"omega={omega:.0e} -> lech toi da {d:.3e} px")

    # Voi omega <= nguong eps (1e-4) phai gan nhu trung khop tuyet doi
    ok = worst < 1e-6
    _check("4. Lien tuc tai omega -> 0 (khong chia cho 0)", ok,
           "\n".join(rows) + f"\nlech lon nhat khi omega <= eps: {worst:.3e} px")


def test_omega_no_nan():
    """Khong duoc sinh ra NaN/inf du omega = 0 chinh xac."""
    ekf = EKFTrackerCTRV()
    x = make_state(v=15.0, theta=1.2, omega=0.0)
    out = ekf.f(x)
    F = ekf.jacobian(x)
    ok = np.all(np.isfinite(out)) and np.all(np.isfinite(F))
    _check("4b. Khong sinh NaN/inf khi omega = 0", ok,
           f"f(x) finite = {np.all(np.isfinite(out))}, F finite = {np.all(np.isfinite(F))}")


# ---------------------------------------------------------------------------
# 5. Jacobian giai tich vs sai phan so
# ---------------------------------------------------------------------------
def _numeric_jacobian(ekf: EKFTrackerCTRV, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Jacobian bang sai phan trung tam: (f(x+e) - f(x-e)) / 2e."""
    n = len(x)
    J = np.zeros((n, n))
    for j in range(n):
        xp, xm = x.copy(), x.copy()
        xp[j] += eps
        xm[j] -= eps
        fp, fm = ekf.f(xp), ekf.f(xm)
        # theta co the bi goi vong -> dung hieu goc chuan hoa
        diff = fp - fm
        diff[TH] = normalize_angle(fp[TH] - fm[TH])
        J[:, j] = diff / (2 * eps)
    return J


def test_jacobian_curved():
    """Jacobian giai tich phai khop sai phan so (truong hop quy dao cong)."""
    ekf = EKFTrackerCTRV()
    x = make_state(cx=320, cy=180, v=8.5, theta=0.6, omega=0.25, a=0.8, h=55, va=0.01, vh=0.4)
    F_ana = ekf.jacobian(x)
    F_num = _numeric_jacobian(ekf, x)
    err = np.max(np.abs(F_ana - F_num))
    ok = err < 1e-5
    _check("5a. Jacobian dung (quy dao cong, omega=0.25)", ok,
           f"sai lech lon nhat giua giai tich va sai phan so: {err:.3e}")


def test_jacobian_straight():
    """Jacobian tai vung omega ~ 0 (dung cong thuc gioi han) cung phai khop.

    Day la cho de sai nhat: neu dat d(cx)/d(omega) = 0 thi test nay se FAIL.
    """
    ekf = EKFTrackerCTRV()
    x = make_state(cx=320, cy=180, v=9.0, theta=0.6, omega=0.0, a=0.8, h=55, va=0.01, vh=0.4)
    F_ana = ekf.jacobian(x)
    # Sai phan so quanh omega = 0 dung buoc lon hon eps de vuot ra khoi vung gioi han
    F_num = _numeric_jacobian(ekf, x, eps=1e-3)
    err = np.max(np.abs(F_ana - F_num))
    ok = err < 1e-4
    detail = (f"sai lech lon nhat: {err:.3e}\n"
              f"d(cx)/d(omega): giai tich={F_ana[CX, OM]:.6f}  so={F_num[CX, OM]:.6f}\n"
              f"d(cy)/d(omega): giai tich={F_ana[CY, OM]:.6f}  so={F_num[CY, OM]:.6f}")
    _check("5b. Jacobian dung tai omega ~ 0 (cong thuc gioi han)", ok, detail)


# ---------------------------------------------------------------------------
# 6. Kiem tra hinh hoc: di het 1 vong tron phai quay ve cho cu
# ---------------------------------------------------------------------------
def test_full_circle():
    """N buoc voi omega = 2*pi/N phai ve dung diem xuat phat.

    Neu cong thuc CTRV sai dau hoac sai he so thi quy dao se khong khep kin.
    """
    ekf = EKFTrackerCTRV()
    N = 36                      # 36 buoc, moi buoc 10 do
    omega = 2 * np.pi / N
    x = make_state(cx=500.0, cy=300.0, v=10.0, theta=0.0, omega=omega)
    start = x[[CX, CY]].copy()

    for _ in range(N):
        x = ekf.f(x)

    err = float(np.hypot(x[CX] - start[0], x[CY] - start[1]))
    ok = err < 1e-9
    _check("6. Di het 1 vong tron ve dung cho cu", ok,
           f"diem dau: ({start[0]:.6f}, {start[1]:.6f})\n"
           f"diem cuoi: ({x[CX]:.6f}, {x[CY]:.6f})\nsai lech: {err:.3e} px")


# ---------------------------------------------------------------------------
# 7. Buoc update
# ---------------------------------------------------------------------------
def test_update_pulls_to_measurement():
    """Voi nhieu do rat nho, hau nghiem phai bam rat sat quan sat."""
    ekf = EKFTrackerCTRV()
    mean, cov = ekf.initiate(np.array([100.0, 200.0, 0.5, 40.0]))
    # Phong to do bat dinh trang thai + thu nho nhieu do => tin quan sat gan tuyet doi
    cov = cov * 1e6
    ekf._std_weight_position = 1e-6

    z = np.array([150.0, 260.0, 0.55, 42.0])
    new_mean, new_cov = ekf.update(mean, cov, z)

    err = np.max(np.abs(new_mean[[CX, CY, A, H]] - z))
    ok = err < 1e-3
    _check("7. Update bam sat quan sat khi nhieu do -> 0", ok,
           f"quan sat  : {z}\n"
           f"hau nghiem: {new_mean[[CX, CY, A, H]]}\nsai lech lon nhat: {err:.3e}")


# ---------------------------------------------------------------------------
# 8. On dinh so hoc
# ---------------------------------------------------------------------------
def test_covariance_stays_valid():
    """Sau nhieu vong predict/update, P phai luon doi xung va xac dinh duong."""
    rng = np.random.default_rng(0)
    ekf = EKFTrackerCTRV()
    mean, cov = ekf.initiate(np.array([100.0, 200.0, 0.5, 40.0]))

    worst_asym = 0.0
    min_eig = np.inf
    for t in range(200):
        mean, cov = ekf.predict(mean, cov)
        # quan sat gia: xe chay theo duong cong + nhieu
        z = np.array([100 + 3 * t, 200 + 20 * np.sin(t / 30), 0.5, 40.0]) + rng.normal(0, 0.5, 4)
        mean, cov = ekf.update(mean, cov, z)

        worst_asym = max(worst_asym, float(np.max(np.abs(cov - cov.T))))
        min_eig = min(min_eig, float(np.min(np.linalg.eigvalsh(cov))))

    ok = worst_asym < 1e-8 and min_eig > 0 and np.all(np.isfinite(mean))
    _check("8. P doi xung + xac dinh duong sau 200 vong lap", ok,
           f"do bat doi xung lon nhat: {worst_asym:.3e}\n"
           f"tri rieng nho nhat: {min_eig:.3e} (phai > 0)")


# ---------------------------------------------------------------------------
# 9. Khoi tao 2 khung hinh
# ---------------------------------------------------------------------------
def test_two_frame_init():
    """v va theta phai duoc uoc luong dung tu 2 quan sat dau tien."""
    ekf = EKFTrackerCTRV()
    mean, cov = ekf.initiate(np.array([100.0, 200.0, 0.5, 40.0]))
    # Quan sat thu 2: dich chuyen (+3, +4) => speed = 5, theta = atan2(4,3) = 0.9273
    mean, cov = ekf.initiate_from_motion(mean, cov, np.array([103.0, 204.0, 0.5, 40.0]))

    exp_v, exp_th = 5.0, np.arctan2(4.0, 3.0)
    ok = np.isclose(mean[V], exp_v) and np.isclose(mean[TH], exp_th)
    _check("9a. Khoi tao v, theta tu 2 quan sat", ok,
           f"tinh tay : v=5.0, theta={exp_th:.6f} rad ({np.degrees(exp_th):.2f} do)\n"
           f"EKF      : v={mean[V]:.6f}, theta={mean[TH]:.6f} rad ({np.degrees(mean[TH]):.2f} do)")


def test_two_frame_init_static():
    """Xe dung yen -> KHONG duoc doan bua huong (giu nguyen trang thai)."""
    ekf = EKFTrackerCTRV()
    mean, cov = ekf.initiate(np.array([100.0, 200.0, 0.5, 40.0]))
    # dich chuyen 0.1 px < nguong 0.5 px/frame
    mean2, _ = ekf.initiate_from_motion(mean, cov, np.array([100.1, 200.0, 0.5, 40.0]))
    ok = np.isclose(mean2[V], 0.0) and np.isclose(mean2[TH], 0.0)
    _check("9b. Xe dung yen -> khong doan bua huong", ok,
           f"v={mean2[V]:.6f} (phai = 0), theta={mean2[TH]:.6f} (phai = 0)")


def test_degenerate_init_without_two_frame():
    """Chung minh VI SAO can khoi tao 2 khung hinh.

    Xe di THANG DUNG theo truc y, khoi tao naive (v=0, theta=0):
    bo loc khong the hoc duoc chuyen dong -> sai so bam duoi ngay cang lon.
    Voi khoi tao 2 khung hinh thi bam rat sat.
    """
    z_seq = [np.array([100.0, 200.0 + 8.0 * k, 0.5, 40.0]) for k in range(15)]

    def run(use_two_frame: bool) -> float:
        ekf = EKFTrackerCTRV()
        mean, cov = ekf.initiate(z_seq[0])
        if use_two_frame:
            mean, cov = ekf.initiate_from_motion(mean, cov, z_seq[1])
        for z in z_seq[1:]:
            mean, cov = ekf.predict(mean, cov)
            mean, cov = ekf.update(mean, cov, z)
        # Sai so du doan mot buoc sau chuoi quan sat
        pred, _ = ekf.predict(mean, cov)
        truth_y = z_seq[-1][1] + 8.0
        return abs(pred[CY] - truth_y)

    err_naive = run(False)
    err_two = run(True)
    ok = err_two < err_naive
    _check("9c. Khoi tao 2 khung hinh tot hon khoi tao naive", ok,
           f"sai so du doan 1 buoc (xe di thang theo truc y):\n"
           f"  khoi tao naive (v=0, theta=0): {err_naive:.3f} px\n"
           f"  khoi tao 2 khung hinh        : {err_two:.3f} px")


# ---------------------------------------------------------------------------
# 10. Khoi kich thuoc phai giong baseline (Constant Velocity tuyen tinh)
# ---------------------------------------------------------------------------
def test_size_block_is_constant_velocity():
    """a, h phai tien theo dung CV: a' = a + va*dt, h' = h + vh*dt."""
    ekf = EKFTrackerCTRV()
    x = make_state(a=0.5, h=40.0, va=0.02, vh=1.5)
    out = ekf.f(x)
    ok = (np.isclose(out[A], 0.52) and np.isclose(out[H], 41.5)
          and np.isclose(out[VA], 0.02) and np.isclose(out[VH], 1.5))
    _check("10. Khoi kich thuoc chay dung CV (giong baseline)", ok,
           f"tinh tay : a=0.52, h=41.5, va=0.02, vh=1.5\n"
           f"EKF      : a={out[A]:.6f}, h={out[H]:.6f}, va={out[VA]:.6f}, vh={out[VH]:.6f}")


def main() -> int:
    print("=" * 78)
    print("UNIT TEST: EKF + mo hinh CTRV")
    print("=" * 78)
    tests = [
        test_straight_theta_zero, test_straight_theta_90,
        test_curved_quarter_turn,
        test_omega_continuity, test_omega_no_nan,
        test_jacobian_curved, test_jacobian_straight,
        test_full_circle,
        test_update_pulls_to_measurement,
        test_covariance_stays_valid,
        test_two_frame_init, test_two_frame_init_static,
        test_degenerate_init_without_two_frame,
        test_size_block_is_constant_velocity,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError:
            failed += 1
        print()

    n = len(_RESULTS)
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    print("=" * 78)
    print(f"KET QUA: {n_pass}/{n} phep kiem tra PASS")
    print("=" * 78)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
