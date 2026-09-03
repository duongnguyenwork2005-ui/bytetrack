"""
curvature.py -- Tinh do cong quy dao (curvature) cho tung track UA-DETRAC.

MUC DICH TRONG DE TAI
---------------------
Gia thuyet cua khoa luan: mo hinh CV (Constant Velocity) trong KF chuan gia dinh
xe di THANG DEU, nen se sai nhieu khi xe RE hoac di vao VONG XUYEN. Mo hinh CTRV
(Constant Turn Rate and Velocity) dung trong EKF/UKF mo ta duoc chuyen dong cong.
Muon CHUNG MINH dieu do, phai tach rieng nhom track "di thang" va nhom "di cong",
roi so sanh metric tren tung nhom.

Script nay tinh do cong cho tung track de lam co so phan nhom do.


CONG THUC TOAN
--------------
Quy dao la chuoi tam bbox theo thoi gian: (x(t), y(t)), t = 1, 2, ... (don vi: frame).

1) LAM TRON + LAY DAO HAM bang Savitzky-Golay
   Tam bbox trong annotation co nhieu (jitter vai pixel). Neu lay sai phan truc
   tiep (x[t+1] - x[t]) thi dao ham bac 2 se bi nhieu lan at hoan toan.
   Savitzky-Golay khop mot da thuc bac `p` len cua so `w` diem lan can, roi lay
   dao ham GIAI TICH cua da thuc do => vua lam tron vua lay dao ham trong 1 buoc.

       x'  = savgol(x, w, p, deriv=1)      [px / frame]
       x'' = savgol(x, w, p, deriv=2)      [px / frame^2]

2) DO CONG (curvature) cua duong cong tham so
                | x'*y'' - y'*x'' |
       kappa = ---------------------          [1 / px]
                ( x'^2 + y'^2 )^(3/2)

   Y nghia hinh hoc: kappa = 1/R voi R la ban kinh duong tron mat tiep
   (osculating circle). Duong thang: kappa = 0, R = vo cung.
   Luu y: mau so tien ve 0 khi xe DUNG YEN -> kappa khong xac dinh.
   Script danh dau NaN cho cac frame co toc do duoi nguong EPS_SPEED.

3) HUONG va TOC DO GOC (lien he truc tiep voi CTRV)
       theta = atan2(y', x')                 [rad]  - huong di chuyen
       omega = d(theta)/dt = kappa * v       [rad/s] - toc do quay (turn rate)

   `omega` chinh la BIEN TRANG THAI THU 5 cua mo hinh CTRV
   (state = [cx, cy, v, theta, omega]). Thong ke omega o day cho biet gia tri
   thuc te cua no trong UA-DETRAC, dung de dat nhieu qua trinh Q hop ly o
   Giai doan 3 va 4.

4) CAC CHI SO TONG HOP CHO CA TRACK
   - net_turn_deg   : |theta_cuoi - theta_dau| sau khi unwrap  -> xe re bao nhieu do
   - total_turn_deg : tong |delta theta| -> gom ca dao dong do nhieu
   - straightness   : (khoang cach 2 dau) / (tong chieu dai duong di)
                      = 1.0 neu di thang tuyet doi, cang nho cang ngoan ngoeo

PHAN NHOM TRACK
---------------
   static  : xe dung yen / gan nhu khong di chuyen -> khong xet do cong
             (median toc do < EPS_SPEED px/frame)
   unknown : track qua ngan (< CURVATURE_MIN_TRACK_LEN frame), khong du diem
             de uoc luong dao ham bac 2 mot cach tin cay
   straight: net_turn_deg < TURN_ANGLE_THRESHOLD_DEG VA
             straightness >= STRAIGHTNESS_THRESHOLD
   curved  : con lai (xe re, vao cua, doi lan gap...)

Ngoai ra co cot `turn_class` chia min hon theo goc re thuc te:
   straight (<10 deg) / gentle (10-45 deg) / sharp (>=45 deg)

OUTPUT
------
    data/interim/track_curvature.csv        <- 1 dong = 1 track (dung de phan nhom)
    data/interim/frame_kinematics.parquet   <- 1 dong = 1 frame cua 1 track
                                               (cx_s, cy_s, vx, vy, speed, heading,
                                                omega, kappa, radius)
    results/curvature_examples.png          <- (neu dung --plot) vai quy dao mau

CACH DUNG
---------
    python src/curvature.py
    python src/curvature.py --plot
    python src/curvature.py --turn-threshold 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# Duoi nguong toc do nay coi nhu xe dung yen -> huong va do cong vo nghia.
EPS_SPEED = 0.5   # px / frame  (0.5 px/frame = 12.5 px/s)


# ---------------------------------------------------------------------------
# Dao ham co lam tron
# ---------------------------------------------------------------------------
def smooth_derivatives(v: np.ndarray, window: int, poly: int):
    """Tra ve (v_smooth, v', v'') bang Savitzky-Golay, tu dong thu nho cua so.

    Yeu cau cua savgol_filter: window le, window <= len(v), poly < window.
    Neu track qua ngan de dung savgol, quay ve np.gradient (sai phan trung tam).
    """
    n = len(v)
    v = v.astype(float)

    # Track chi co 1-2 frame: khong the noi gi ve dao ham bac 2.
    # Tra ve 0 va de buoc phan nhom danh nhan 'unknown' cho nhung track nay.
    if n < 3:
        d1 = np.gradient(v) if n == 2 else np.zeros(n)
        return v, d1, np.zeros(n)

    w = min(window, n if n % 2 == 1 else n - 1)
    if w < poly + 2 or w < 5:
        # Track ngan: dung sai phan trung tam, khong lam tron.
        d1 = np.gradient(v)
        d2 = np.gradient(d1)
        return v, d1, d2

    if w % 2 == 0:
        w -= 1
    vs = savgol_filter(v, w, poly, deriv=0, delta=1.0)
    d1 = savgol_filter(v, w, poly, deriv=1, delta=1.0)
    d2 = savgol_filter(v, w, poly, deriv=2, delta=1.0)
    return vs, d1, d2


# ---------------------------------------------------------------------------
# Tinh dong hoc cho 1 track
# ---------------------------------------------------------------------------
def track_kinematics(frames: np.ndarray, cx: np.ndarray, cy: np.ndarray):
    """Tinh cac dai luong dong hoc theo tung frame cua 1 track.

    LUU Y VE FRAME BI THIEU: UA-DETRAC annotate lien tuc (da kiem tra: 0 track gap),
    nen o day coi khoang cach giua 2 mau lien tiep luon = 1 frame. Neu sau nay dung
    cho dataset khac co lo hong, can noi suy lai truoc khi goi ham nay.
    """
    w, p = config.SAVGOL_WINDOW, config.SAVGOL_POLY
    xs, vx, ax = smooth_derivatives(cx, w, p)
    ys, vy, ay = smooth_derivatives(cy, w, p)

    speed = np.hypot(vx, vy)                       # px / frame
    moving = speed > EPS_SPEED

    # --- Do cong kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2) ---
    num = np.abs(vx * ay - vy * ax)
    den = np.power(speed, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(moving, num / den, np.nan)
        radius = np.where(kappa > 0, 1.0 / kappa, np.inf)

    # --- Huong va toc do quay ---
    heading = np.where(moving, np.arctan2(vy, vx), np.nan)   # rad
    # omega = kappa * v ; doi don vi: (1/px) * (px/frame) * (frame/s) = rad/s
    omega = kappa * speed * config.FPS                       # rad / s

    return {
        "cx_smooth": xs, "cy_smooth": ys,
        "vx": vx, "vy": vy,
        "speed_px_per_frame": speed,
        "speed_px_per_s": speed * config.FPS,
        "heading_rad": heading,
        "omega_rad_per_s": omega,
        "kappa_per_px": kappa,
        "radius_px": radius,
        "is_moving": moving,
    }


def summarize_track(video: str, tid: int, frames: np.ndarray,
                    cx: np.ndarray, cy: np.ndarray, kin: dict,
                    turn_threshold_deg: float, straightness_threshold: float) -> dict:
    """Gop dong hoc theo frame thanh cac chi so mo ta ca track + gan nhan phan nhom."""
    n = len(frames)
    speed = kin["speed_px_per_frame"]
    moving = kin["is_moving"]
    heading = kin["heading_rad"]
    kappa = kin["kappa_per_px"]
    omega = kin["omega_rad_per_s"]

    # --- Chieu dai duong di va do lech thang ---
    dx = np.diff(kin["cx_smooth"])
    dy = np.diff(kin["cy_smooth"])
    path_len = float(np.sum(np.hypot(dx, dy)))
    net_disp = float(np.hypot(kin["cx_smooth"][-1] - kin["cx_smooth"][0],
                              kin["cy_smooth"][-1] - kin["cy_smooth"][0]))
    straightness = net_disp / path_len if path_len > 1e-6 else np.nan

    # --- Goc doi huong (chi tinh tren doan xe thuc su di chuyen) ---
    h = heading[moving]
    if h.size >= 2:
        # unwrap de tranh nhay 2*pi khi huong di qua +-180 do
        hu = np.unwrap(h)
        net_turn_deg = float(np.degrees(abs(hu[-1] - hu[0])))
        total_turn_deg = float(np.degrees(np.sum(np.abs(np.diff(hu)))))
    else:
        net_turn_deg = np.nan
        total_turn_deg = np.nan

    kap = kappa[np.isfinite(kappa)]
    om = omega[np.isfinite(omega)]
    med_speed = float(np.median(speed)) if n else np.nan

    # ------------------------------------------------------------------
    # Gan nhan phan nhom
    # ------------------------------------------------------------------
    if n < config.CURVATURE_MIN_TRACK_LEN:
        cls = "unknown"          # qua ngan, dao ham bac 2 khong dang tin
    elif med_speed < EPS_SPEED:
        cls = "static"           # xe dung / do ben duong
    elif (np.isfinite(net_turn_deg) and net_turn_deg < turn_threshold_deg
          and np.isfinite(straightness) and straightness >= straightness_threshold):
        cls = "straight"
    else:
        cls = "curved"

    # Phan nhom min hon theo goc re thuc te
    if not np.isfinite(net_turn_deg) or cls in ("unknown", "static"):
        turn_class = cls
    elif net_turn_deg < 10:
        turn_class = "straight"
    elif net_turn_deg < 45:
        turn_class = "gentle"
    else:
        turn_class = "sharp"

    return {
        "video": video, "track_id": tid,
        "n_frames": n,
        "first_frame": int(frames[0]), "last_frame": int(frames[-1]),
        # do dai / hinh dang quy dao
        "path_len_px": round(path_len, 2),
        "net_disp_px": round(net_disp, 2),
        "straightness": round(float(straightness), 4) if np.isfinite(straightness) else np.nan,
        "net_turn_deg": round(net_turn_deg, 2) if np.isfinite(net_turn_deg) else np.nan,
        "total_turn_deg": round(total_turn_deg, 2) if np.isfinite(total_turn_deg) else np.nan,
        # toc do
        "median_speed_px_per_frame": round(med_speed, 3),
        "median_speed_px_per_s": round(med_speed * config.FPS, 2),
        # do cong
        "mean_kappa": round(float(np.mean(kap)), 6) if kap.size else np.nan,
        "median_kappa": round(float(np.median(kap)), 6) if kap.size else np.nan,
        "p90_kappa": round(float(np.percentile(kap, 90)), 6) if kap.size else np.nan,
        "max_kappa": round(float(np.max(kap)), 6) if kap.size else np.nan,
        "min_radius_px": round(float(1.0 / np.max(kap)), 2) if kap.size and np.max(kap) > 0 else np.nan,
        # toc do quay - tham so omega cua CTRV
        "median_abs_omega_rad_s": round(float(np.median(np.abs(om))), 5) if om.size else np.nan,
        "p90_abs_omega_rad_s": round(float(np.percentile(np.abs(om), 90)), 5) if om.size else np.nan,
        "max_abs_omega_rad_s": round(float(np.max(np.abs(om))), 5) if om.size else np.nan,
        # nhan
        "curvature_class": cls,
        "turn_class": turn_class,
    }


# ---------------------------------------------------------------------------
# Ve minh hoa
# ---------------------------------------------------------------------------
def plot_examples(df_track: pd.DataFrame, df_frame: pd.DataFrame, out_png: Path) -> None:
    """Ve vai quy dao dai dien cho tung nhom de kiem tra bang mat."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ["straight", "curved", "static"]
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5))

    for ax, cls in zip(np.atleast_1d(axes), groups):
        sel = df_track[df_track["curvature_class"] == cls]
        # uu tien track dai de nhin ro hinh dang
        sel = sel.sort_values("n_frames", ascending=False).head(8)
        for r in sel.itertuples(index=False):
            t = df_frame[(df_frame["video"] == r.video) & (df_frame["track_id"] == r.track_id)]
            ax.plot(t["cx_smooth"], t["cy_smooth"], lw=1.5, alpha=0.85)
        ax.set_title(f"{cls}  (n = {(df_track['curvature_class'] == cls).sum()} track)")
        ax.set_xlim(0, config.IMG_W)
        ax.set_ylim(config.IMG_H, 0)     # dao truc y cho dung he toa do anh
        ax.set_xlabel("cx (px)")
        ax.set_ylabel("cy (px)")
        ax.grid(alpha=0.3)

    fig.suptitle("Quy dao tam bbox theo nhom do cong (he toa do anh)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  Da luu hinh -> {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Tinh do cong quy dao cho UA-DETRAC")
    ap.add_argument("--turn-threshold", type=float, default=config.TURN_ANGLE_THRESHOLD_DEG,
                    help="Goc re (do) de coi track la 'curved'")
    ap.add_argument("--straightness-threshold", type=float, default=config.STRAIGHTNESS_THRESHOLD,
                    help="Nguong chi so thang de coi track la 'straight'")
    ap.add_argument("--plot", action="store_true", help="Ve hinh minh hoa vao results/")
    ap.add_argument("--no-frame-output", action="store_true",
                    help="Khong luu file frame_kinematics.parquet (tiet kiem dung luong)")
    args = ap.parse_args()

    pq = config.INTERIM_DIR / "detrac_train_annotations.parquet"
    if not pq.exists():
        print(f"[ERROR] Chua co {pq}. Chay src/parse_detrac_xml.py truoc.")
        return 1

    df = pd.read_parquet(pq)
    print(f"[curvature] {len(df):,} bbox, {df.groupby(['video', 'track_id']).ngroups:,} track")
    print(f"[curvature] savgol window={config.SAVGOL_WINDOW} poly={config.SAVGOL_POLY}  "
          f"nguong re={args.turn_threshold} deg  nguong thang={args.straightness_threshold}")

    rows, frame_parts = [], []
    groups = df.groupby(["video", "track_id"], observed=True, sort=True)
    for (video, tid), g in tqdm(groups, total=groups.ngroups,
                                desc="  xu ly track", unit="track", ncols=90):
        g = g.sort_values("frame")
        frames = g["frame"].to_numpy()
        cx = g["cx"].to_numpy(dtype=float)
        cy = g["cy"].to_numpy(dtype=float)

        kin = track_kinematics(frames, cx, cy)
        rows.append(summarize_track(video, int(tid), frames, cx, cy, kin,
                                    args.turn_threshold, args.straightness_threshold))

        if not args.no_frame_output:
            part = pd.DataFrame({
                "video": video, "track_id": int(tid), "frame": frames,
                "cx": cx, "cy": cy,
                **{k: v for k, v in kin.items()},
            })
            frame_parts.append(part)

    df_track = pd.DataFrame(rows)
    out_track = config.INTERIM_DIR / "track_curvature.csv"
    df_track.to_csv(out_track, index=False)

    df_frame = pd.DataFrame()
    if frame_parts:
        df_frame = pd.concat(frame_parts, ignore_index=True)
        out_frame = config.INTERIM_DIR / "frame_kinematics.parquet"
        df_frame.to_parquet(out_frame, index=False)

    # ------------------------------------------------------------------
    # Bao cao
    # ------------------------------------------------------------------
    n = len(df_track)
    print()
    print("=" * 78)
    print("PHAN NHOM TRACK THEO DO CONG QUY DAO")
    print("=" * 78)
    for k, v in df_track["curvature_class"].value_counts().items():
        print(f"    {str(k):<10} {v:>6,}  ({v / n * 100:5.1f}%)")

    print()
    print("  Chi tiet theo goc re thuc te (turn_class):")
    for k, v in df_track["turn_class"].value_counts().items():
        print(f"    {str(k):<10} {v:>6,}  ({v / n * 100:5.1f}%)")

    usable = df_track[df_track["curvature_class"].isin(["straight", "curved"])]
    if len(usable):
        print()
        print(f"  Tren {len(usable):,} track dung duoc (bo static / unknown):")
        for col, unit in [("net_turn_deg", "do"), ("straightness", ""),
                          ("median_speed_px_per_s", "px/s"),
                          ("median_kappa", "1/px"), ("min_radius_px", "px"),
                          ("p90_abs_omega_rad_s", "rad/s")]:
            v = usable[col].dropna()
            if not len(v):
                continue
            print(f"    {col:<24} median={v.median():>10.4f}  p90={v.quantile(0.9):>10.4f}  "
                  f"max={v.max():>10.4f}  {unit}")

        print()
        print("  So sanh 2 nhom (gia tri median):")
        cmp = usable.groupby("curvature_class", observed=True).agg(
            n=("track_id", "size"),
            net_turn=("net_turn_deg", "median"),
            straightness=("straightness", "median"),
            median_kappa=("median_kappa", "median"),
            p90_omega=("p90_abs_omega_rad_s", "median"),
            speed=("median_speed_px_per_s", "median"),
        ).round(4)
        print(cmp.to_string())

    print()
    print("  Da luu:")
    print(f"    {out_track}")
    if len(df_frame):
        print(f"    {config.INTERIM_DIR / 'frame_kinematics.parquet'}  ({len(df_frame):,} dong)")

    if args.plot and len(df_frame):
        plot_examples(df_track, df_frame, config.RESULTS_DIR / "curvature_examples.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
