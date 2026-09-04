"""
compare_extrapolation.py -- So sanh kha nang NGOAI SUY cua CV va CTRV khi bi
che khuat, tren du lieu MO PHONG (biet truoc su that tuyet doi).

VI SAO CAN THI NGHIEM NAY?
--------------------------
Tren video that, ket qua tracking bi tron lan boi rat nhieu yeu to: chat luong
detector, nguong ghep cap, mat do giao thong... Rat kho tach rieng dong gop cua
MOTION MODEL. Thi nghiem mo phong nay co lap DUNG mot thu: kha nang du doan vi
tri khi KHONG CON detection nao (dung nhu luc xe bi che khuat hoan toan).

Day chinh la co che ma khoa luan gia thiet rang CTRV vuot troi hon CV.

CACH LAM
--------
 1. Sinh quy dao THAT theo dung mo hinh CTRV (xe chay cung tron voi turn rate
    co dinh) hoac di thang, o nhieu muc turn rate khac nhau.
 2. Cho ca 2 bo loc "nhin" N_OBS frame dau (co detection, kem nhieu do).
 3. Cat detection: bat ca 2 bo loc CHI DU DOAN trong N_GAP frame tiep theo.
 4. Do sai so vi tri giua du doan va quy dao that o tung frame cua doan bi che.

Bo loc CV dung dung lop KalmanFilterXYAH cua ultralytics (baseline that su),
bo loc CTRV dung EKFTrackerCTRV cua ta -> so sanh tuyet doi cong bang.

CACH DUNG
---------
    python src/compare_extrapolation.py
    python src/compare_extrapolation.py --plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402

from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH  # noqa: E402


def make_trajectory(n: int, v: float, omega_deg: float, h: float = 50.0, a: float = 1.2):
    """Sinh quy dao that theo mo hinh CTRV.

    Args:
        n: so frame
        v: toc do [px/frame]
        omega_deg: turn rate [do/frame]. 0 = di thang.
    Returns:
        mang (n, 4): moi hang la quan sat that (cx, cy, a, h)
    """
    omega = np.radians(omega_deg)
    cx, cy, theta = 300.0, 300.0, 0.0
    out = []
    for _ in range(n):
        out.append([cx, cy, a, h])
        if abs(omega) > 1e-9:
            cx += (v / omega) * (np.sin(theta + omega) - np.sin(theta))
            cy += (v / omega) * (np.cos(theta) - np.cos(theta + omega))
        else:
            cx += v * np.cos(theta)
            cy += v * np.sin(theta)
        theta += omega
    return np.asarray(out)


def run_filter(kind: str, obs: np.ndarray, n_gap: int, rng) -> np.ndarray:
    """Cho bo loc quan sat `obs`, roi ngoai suy `n_gap` frame. Tra ve (n_gap, 2) vi tri du doan."""
    if kind == "cv":
        kf = KalmanFilterXYAH()
        # baseline dung xyah: [x, y, a, h, vx, vy, va, vh]
        mean, cov = kf.initiate(obs[0])
        for z in obs[1:]:
            mean, cov = kf.predict(mean, cov)
            mean, cov = kf.update(mean, cov, z)
        preds = []
        for _ in range(n_gap):
            mean, cov = kf.predict(mean, cov)
            preds.append([mean[0], mean[1]])
        return np.asarray(preds)

    kf = EKFTrackerCTRV()
    mean, cov = kf.initiate(obs[0])
    # Khoi tao 2 khung hinh (xem ekf_ctrv.py muc 5)
    mean, cov = kf.initiate_from_motion(mean, cov, obs[1])
    for z in obs[1:]:
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, z)
    preds = []
    for _ in range(n_gap):
        mean, cov = kf.predict(mean, cov)
        preds.append([mean[kf.CX], mean[kf.CY]])
    return np.asarray(preds)


def experiment(omega_deg: float, v: float, n_obs: int, n_gap: int,
               noise_std: float, n_trials: int, seed: int = 0):
    """Chay nhieu lan cho 1 kich ban, tra ve sai so trung binh cua tung bo loc."""
    rng = np.random.default_rng(seed)
    truth = make_trajectory(n_obs + n_gap, v, omega_deg)
    err = {"cv": [], "ctrv": []}
    for _ in range(n_trials):
        noisy = truth[:n_obs].copy()
        noisy[:, :2] += rng.normal(0, noise_std, (n_obs, 2))   # nhieu do vi tri
        for kind in ("cv", "ctrv"):
            pred = run_filter(kind, noisy, n_gap, rng)
            d = np.hypot(pred[:, 0] - truth[n_obs:, 0], pred[:, 1] - truth[n_obs:, 1])
            err[kind].append(d)
    return {k: np.mean(v_, axis=0) for k, v_ in err.items()}   # (n_gap,) sai so theo tung frame


def main() -> int:
    ap = argparse.ArgumentParser(description="So sanh ngoai suy CV vs CTRV khi bi che khuat")
    ap.add_argument("--n-obs", type=int, default=25, help="So frame co detection truoc khi bi che")
    ap.add_argument("--n-trials", type=int, default=200, help="So lan lap moi kich ban")
    ap.add_argument("--noise-std", type=float, default=1.5, help="Nhieu do vi tri [px]")
    ap.add_argument("--speed", type=float, default=8.0, help="Toc do xe [px/frame]")
    ap.add_argument("--plot", action="store_true", help="Ve bieu do vao results/")
    args = ap.parse_args()

    # Cac muc turn rate, doi chieu voi so lieu THUC DO o Giai doan 1:
    #   p90 |omega| ~ 0.89 rad/s (thang) va ~1.33 rad/s (cong)  [track_curvature.csv]
    #   -> /25 fps = 0.036 - 0.053 rad/frame = 2.0 - 3.0 do/frame
    omegas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    # Do dai doan che, theo phan nhom cua Giai doan 1 (25 fps)
    gaps = {"short (0.2s)": 5, "medium (0.8s)": 20, "long (2.0s)": 50}

    print("=" * 92)
    print("SO SANH KHA NANG NGOAI SUY KHI BI CHE KHUAT (du lieu mo phong)")
    print("=" * 92)
    print(f"  toc do xe = {args.speed} px/frame, nhieu do = {args.noise_std} px, "
          f"{args.n_obs} frame quan sat truoc khi che, {args.n_trials} lan lap")
    print()

    rows = []
    for gap_name, n_gap in gaps.items():
        print(f"--- Doan che {gap_name} = {n_gap} frame ---")
        print(f"  {'turn rate':>12} {'CV (px)':>10} {'CTRV (px)':>11} {'chenh lech':>12} {'ket luan':>14}")
        for om in omegas:
            e = experiment(om, args.speed, args.n_obs, n_gap, args.noise_std, args.n_trials)
            cv_err, ctrv_err = float(np.mean(e["cv"])), float(np.mean(e["ctrv"]))
            diff = cv_err - ctrv_err
            verdict = "CTRV tot hon" if diff > 0.5 else ("CV tot hon" if diff < -0.5 else "ngang nhau")
            print(f"  {om:>10.1f}d/f {cv_err:>10.2f} {ctrv_err:>11.2f} {diff:>+12.2f} {verdict:>14}")
            rows.append({"gap_name": gap_name, "n_gap": n_gap, "omega_deg_per_frame": om,
                         "err_cv_px": round(cv_err, 3), "err_ctrv_px": round(ctrv_err, 3),
                         "improvement_px": round(diff, 3)})
        print()

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "extrapolation_cv_vs_ctrv.csv"
    df.to_csv(out, index=False)
    print(f"Da luu bang -> {out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(gaps), figsize=(5.2 * len(gaps), 4.2), sharey=False)
        for ax, (gap_name, n_gap) in zip(np.atleast_1d(axes), gaps.items()):
            sub = df[df["gap_name"] == gap_name]
            ax.plot(sub["omega_deg_per_frame"], sub["err_cv_px"], "o-", label="CV (baseline KF)")
            ax.plot(sub["omega_deg_per_frame"], sub["err_ctrv_px"], "s-", label="CTRV (EKF)")
            ax.set_title(f"Doan che {gap_name}")
            ax.set_xlabel("turn rate that (do/frame)")
            ax.set_ylabel("sai so vi tri trung binh (px)")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle("Sai so ngoai suy khi bi che khuat: CV vs CTRV (mo phong)")
        fig.tight_layout()
        png = config.RESULTS_DIR / "extrapolation_cv_vs_ctrv.png"
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"Da luu hinh  -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
