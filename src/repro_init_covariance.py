"""
repro_init_covariance.py -- Tai hien 2 loi nghi ngo o buoc khoi tao chuyen dong.

LOI 1: initiate_from_motion doi phuong sai cua v va theta nhung GIU NGUYEN cac
       phan tu hiep phuong sai cheo -> P mat tinh ban xac dinh duong.
LOI 2: tracker danh dau _motion_initialized = True ngay ca khi khoi tao chuyen
       dong KHONG thanh cong (dich chuyen qua nho) -> xe dung yen luc dau roi
       moi chuyen dong se khong bao gio duoc khoi tao huong.

KICH BAN TAI HIEN (theo dung yeu cau):
    - Khoi tao measurement [100, 200, 0.5, 40]
    - Predict lien tiep N buoc, N thuoc {1, 5, 10, 26}
    - Goi initiate_from_motion voi [100 + 3N, 200 + 4N, 0.5, 40], n_frames=N
    - Kiem tra tri rieng cua P truoc va sau khoi tao

CACH DUNG
    python src/repro_init_covariance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402
from ukf_ctrv import UKFTrackerCTRV  # noqa: E402

Z0 = np.array([100.0, 200.0, 0.5, 40.0])
N_LIST = [1, 5, 10, 26]
TOL = 1e-8      # dung sai cho tri rieng am do sai so so hoc


def eig_min(P: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(0.5 * (P + P.T)).min())


def asym(P: np.ndarray) -> float:
    return float(np.abs(P - P.T).max())


def repro_covariance(FilterCls) -> list[dict]:
    name = FilterCls.__name__
    print("=" * 84)
    print(f"LOI 1 -- P co mat tinh ban xac dinh duong khong?   [{name}]")
    print("=" * 84)
    print(f"  {'N':>4}{'tri rieng min TRUOC':>22}{'tri rieng min SAU':>21}"
          f"{'bat doi xung':>15}{'ket qua':>12}")
    print("  " + "-" * 72)
    rows = []
    for N in N_LIST:
        f = FilterCls()
        m, P = f.initiate(Z0)
        for _ in range(N):
            m, P = f.predict(m, P)
        before = eig_min(P)
        z1 = np.array([100.0 + 3 * N, 200.0 + 4 * N, 0.5, 40.0])
        m2, P2 = f.initiate_from_motion(m, P, z1, n_frames=float(N))
        after = eig_min(P2)
        ok = after >= -TOL
        rows.append(dict(filter=name, N=N, eig_before=before, eig_after=after,
                         asym=asym(P2), psd_ok=ok, v=float(m2[f.V]),
                         theta_deg=float(np.rad2deg(m2[f.THETA]))))
        print(f"  {N:>4}{before:>22.6e}{after:>21.6e}{asym(P2):>15.2e}"
              f"{('DAT' if ok else 'HONG'):>12}")
        print(f"      -> v = {m2[f.V]:.4f} px/frame (dung phai 5.0), "
              f"theta = {np.rad2deg(m2[f.THETA]):.2f} do (dung phai 53.13)")
    return rows


def _stationary_then_move(FilterCls, retry: bool, axis: str = "y"):
    """Xe dung yen 1 frame roi chay that. `retry=False` mo phong logic CU cua
    tracker (danh dau da xong bat ke ket qua), `retry=True` mo phong logic MOI.

    Tra ve (v_cuoi, theta_cuoi_do).
    """
    f = FilterCls()
    m, P = f.initiate(Z0)
    initialized = False
    last_obs = (float(Z0[0]), float(Z0[1]))

    # Quan sat 2: xe gan nhu dung yen (dich 0.1 px) -> khong du de suy huong
    z_static = np.array([100.1, 200.0, 0.5, 40.0])
    if not initialized:
        m, P, ok = f.try_initiate_from_motion(m, P, z_static, n_frames=1.0, ref_pos=last_obs)
        initialized = ok if retry else True
    m, P = f.predict(m, P)
    m, P = f.update(m, P, z_static)
    last_obs = (float(z_static[0]), float(z_static[1]))

    # Tu day xe CHAY THAT
    cx, cy = 100.1, 200.0
    for _ in range(6):
        if axis == "y":
            cy += 8.0
        else:
            cx += 8.0
        z = np.array([cx, cy, 0.5, 40.0])
        if not initialized:
            m, P, ok = f.try_initiate_from_motion(m, P, z, n_frames=1.0, ref_pos=last_obs)
            initialized = ok if retry else True
        m, P = f.predict(m, P)
        m, P = f.update(m, P, z)
        last_obs = (float(z[0]), float(z[1]))
    return float(m[f.V]), float(np.rad2deg(m[f.THETA]))


def repro_stationary_then_move(FilterCls) -> bool:
    """LOI 2: xe dung yen roi moi chuyen dong. So logic CU va logic MOI."""
    name = FilterCls.__name__
    print("\n" + "=" * 84)
    print(f"LOI 2 -- danh dau da khoi tao du KHONG thanh cong   [{name}]")
    print("=" * 84)
    print(f"  Nguong toc do de suy duoc huong: "
          f"CTRV_MIN_SPEED_FOR_HEADING = {config.CTRV_MIN_SPEED_FOR_HEADING} px/frame")
    print("  Kich ban: 1 frame dung yen (dich 0.1 px) roi chay that 8 px/frame, 6 frame.\n")
    print(f"  {'truc':>6}{'logic':>26}{'v cuoi':>12}{'theta cuoi':>14}{'ket qua':>12}")
    print("  " + "-" * 70)
    stuck_old = False
    for axis, want_theta in (("y", 90.0), ("x", 0.0)):
        for retry, lab in ((False, "CU: luon danh dau xong"), (True, "MOI: thu lai den khi duoc")):
            v, th = _stationary_then_move(FilterCls, retry=retry, axis=axis)
            good = abs(v - 8.0) < 1.5 and abs(normalize_deg(th - want_theta)) < 15.0
            if not retry and not good:
                stuck_old = True
            print(f"  {axis:>6}{lab:>26}{v:>12.4f}{th:>13.2f}o"
                  f"{('DAT' if good else 'HONG'):>12}")
        print(f"       (dung phai: v ~ 8.0, theta ~ {want_theta:.0f} do)")
    return stuck_old


def normalize_deg(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def main() -> int:
    all_rows = []
    for cls in (EKFTrackerCTRV, UKFTrackerCTRV):
        all_rows += repro_covariance(cls)

    n_bad = sum(1 for r in all_rows if not r["psd_ok"])
    print("\n" + "=" * 84)
    print(f"TONG KET LOI 1: {n_bad}/{len(all_rows)} truong hop lam P mat ban xac dinh duong")
    print("=" * 84)

    stuck_any = False
    for cls in (EKFTrackerCTRV, UKFTrackerCTRV):
        stuck_any |= repro_stationary_then_move(cls)

    print("\n" + "=" * 84)
    print("KET LUAN")
    print("=" * 84)
    print(f"  LOI 1 (P mat ban xac dinh duong)     : {'TAI HIEN DUOC' if n_bad else 'khong tai hien'}")
    print(f"  LOI 2 (khoi tao huong bi bo qua vinh vien): "
          f"{'TAI HIEN DUOC' if stuck_any else 'khong tai hien'}")
    return 0 if (n_bad or stuck_any) else 1


if __name__ == "__main__":
    raise SystemExit(main())
