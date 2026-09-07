"""
render_videos.py -- DEMO Phan 3: xuat 4 video minh hoa, moi loai 1 doan.

CHON DOAN BANG TIEU CHI TU DONG, KHONG CHON TAY:
  Video 1 (CTRV thang)     : trong CASE_A, doan co FDE_CV - FDE_CTRV LON NHAT
  Video 2 (CV thang)       : trong STRAIGHT, doan co FDE_CTRV - FDE_CV LON NHAT
  Video 3 (tran thong tin) : trong CASE_B, doan co T_occ DAI NHAT
  Video 4 (tran detector)  : doan ca 3 model deu `lost`, T_occ trung binh

FDE = Final Displacement Error: khoang cach giua tam bbox du doan va tam bbox GT
o frame CUOI doan che. Bo loc duoc nap quan sat GT den truoc doan che, roi CHI
predict (khong update) - mo phong dung tinh huong mat detection hoan toan.

CACH DUNG
    python src/demo/render_videos.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH  # noqa: E402

THR_DEG = 0.25          # nguong phan loai dung de chon doan (xem case_ab.py)
WARM = 15               # so frame GT nap vao bo loc truoc doan che
PRE_SHOW = 12           # so frame hien thi TRUOC doan che
POST_SHOW = 5           # so frame hien thi SAU doan che
FPS = 10

COL_GT = (255, 255, 255)
COL_KF = (0, 220, 0)      # xanh la  (BGR)
COL_EKF = (0, 0, 235)     # do
TRAIL = 14


def xyah(r):
    w, h = r.bb_width, r.bb_height
    return np.array([r.cx, r.cy, w / max(h, 1e-6), h], float)


def run_filters(pre: pd.DataFrame, n_pred: int):
    """Nap GT truoc doan che roi CHI predict n_pred buoc. Tra ve quy dao (cx,cy,w,h)."""
    out = {}
    for name in ("KF", "EKF"):
        if name == "KF":
            f = KalmanFilterXYAH(); get = lambda m: (m[0], m[1], m[2], m[3]); ctrv = False
        else:
            f = EKFTrackerCTRV(); get = lambda m, f=f: (m[f.CX], m[f.CY], m[f.A], m[f.H]); ctrv = True
        rows = list(pre.itertuples())
        m, c = f.initiate(xyah(rows[0])); ini = False
        for r in rows[1:]:
            z = xyah(r)
            if ctrv and not ini:
                m, c = f.initiate_from_motion(m, c, z, n_frames=1.0); ini = True
            m, c = f.predict(m, c); m, c = f.update(m, c, z)
        traj = []
        for _ in range(n_pred):
            m, c = f.predict(m, c)
            cx, cy, a, h = get(m)
            traj.append((cx, cy, a * h, h))
        out[name] = np.array(traj)
    return out


def compute_fde(df: pd.DataFrame, g: dict) -> pd.DataFrame:
    """Tinh FDE cua KF va EKF cho tung doan (chi predict qua doan che)."""
    # Ten cot "case_0.25" co dau cham -> itertuples doi ten thanh _N.
    # Doi ten ve "case" truoc khi lap de truy cap duoc bang thuoc tinh.
    df = df.rename(columns={f"case_{THR_DEG}": "case"})
    rec = []
    for r in df.itertuples(index=False):
        t = g.get((r.video, r.track_id))
        if t is None:
            continue
        pre = t[(t.frame < r.frame_start) & (t.frame >= r.frame_start - WARM)].sort_values("frame")
        occ = t[(t.frame >= r.frame_start) & (t.frame <= r.frame_end)].sort_values("frame")
        if len(pre) < 8 or len(occ) < 3:
            continue
        try:
            tr = run_filters(pre, len(occ))
        except Exception:
            continue
        gt_last = np.array([occ.iloc[-1].cx, occ.iloc[-1].cy])
        rec.append(dict(video=r.video, track_id=r.track_id, seg_id=r.seg_id, lvl=r.lvl,
                        frame_start=r.frame_start, frame_end=r.frame_end, T_occ=r.T_occ,
                        ob_deg=r.ob_deg, od_deg=r.od_deg,
                        case=r.case,
                        fde_kf=float(np.hypot(*(tr["KF"][-1][:2] - gt_last))),
                        fde_ekf=float(np.hypot(*(tr["EKF"][-1][:2] - gt_last)))))
    d = pd.DataFrame(rec)
    if len(d):
        d["gain_ctrv"] = d.fde_kf - d.fde_ekf      # duong = CTRV tot hon
    return d


def render(sel: dict, g: dict, img_root: Path, out_path: Path) -> None:
    """Ve mot doan thanh video mp4."""
    v, tid = sel["video"], sel["track_id"]
    fs, fe = int(sel["frame_start"]), int(sel["frame_end"])
    t = g[(v, tid)]
    pre = t[(t.frame < fs) & (t.frame >= fs - WARM)].sort_values("frame")
    occ = t[(t.frame >= fs) & (t.frame <= fe)].sort_values("frame")
    post = t[(t.frame > fe) & (t.frame <= fe + POST_SHOW)].sort_values("frame")
    n_pred = len(occ) + len(post)
    tr = run_filters(pre, n_pred)

    show_pre = pre.tail(PRE_SHOW)
    frames = list(show_pre.frame) + list(occ.frame) + list(post.frame)
    gt_map = {int(r.frame): r for r in t.itertuples()}

    first = cv2.imread(str(img_root / v / f"img{frames[0]:05d}.jpg"))
    H, W = first.shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    trail_gt, trail_kf, trail_ekf = [], [], []
    for i, fr in enumerate(frames):
        im = cv2.imread(str(img_root / v / f"img{fr:05d}.jpg"))
        if im is None:
            continue
        in_occ = fs <= fr <= fe
        k = fr - fs                      # chi so trong mang du doan
        r = gt_map.get(fr)

        # --- GT (trang, net lien) ---
        if r is not None:
            x, y, w, h = r.bb_left, r.bb_top, r.bb_width, r.bb_height
            cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), COL_GT, 2)
            trail_gt.append((int(r.cx), int(r.cy)))

        # --- Du doan (chi ve khi da vao giai doan predict) ---
        if k >= 0 and k < n_pred:
            for nm, col, trail in (("KF", COL_KF, trail_kf), ("EKF", COL_EKF, trail_ekf)):
                cx, cy, w, h = tr[nm][k]
                cv2.rectangle(im, (int(cx - w / 2), int(cy - h / 2)),
                              (int(cx + w / 2), int(cy + h / 2)), col, 2)
                trail.append((int(cx), int(cy)))

        for trail, col in ((trail_gt, COL_GT), (trail_kf, COL_KF), (trail_ekf, COL_EKF)):
            pts = trail[-TRAIL:]
            for j in range(1, len(pts)):
                cv2.line(im, pts[j - 1], pts[j], col, 2)

        # --- Danh dau vung bi che ---
        if in_occ:
            cv2.rectangle(im, (0, 0), (W - 1, H - 1), (0, 215, 255), 6)
            cv2.putText(im, "OCCLUDED", (W - 210, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 215, 255), 2)

        # --- Overlay thong tin ---
        fde_kf = fde_ekf = None
        if r is not None and 0 <= k < n_pred:
            fde_kf = np.hypot(tr["KF"][k][0] - r.cx, tr["KF"][k][1] - r.cy)
            fde_ekf = np.hypot(tr["EKF"][k][0] - r.cx, tr["EKF"][k][1] - r.cy)
        lines = [
            f"{sel['label']}  [{sel['case']}]",
            f"{v} track {tid}   frame {max(0, k) + 1}/{sel['T_occ']}",
            f"|w_before|={abs(sel['ob_deg']):.3f}  |w_during|={abs(sel['od_deg']):.3f} deg/f",
        ]
        if fde_kf is not None:
            lines.append(f"sai so KF={fde_kf:6.1f}px   EKF={fde_ekf:6.1f}px")
        cv2.rectangle(im, (0, 0), (560, 20 + 24 * len(lines)), (0, 0, 0), -1)
        for j, s in enumerate(lines):
            cv2.putText(im, s, (8, 26 + 24 * j), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
        # chu thich mau
        cv2.putText(im, "GT", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_GT, 2)
        cv2.putText(im, "KF+CV", (48, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_KF, 2)
        cv2.putText(im, "EKF+CTRV", (140, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_EKF, 2)
        vw.write(im)
    vw.release()


def main() -> int:
    out = config.RESULTS_DIR / "demo"
    vids = out / "videos"; vids.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out / "segments_classified.csv")
    pq = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    pq = pq[pq.video.isin(df.video.unique())]
    g = {k: v.sort_values("frame") for k, v in pq.groupby(["video", "track_id"])}
    img_root = config.find_images_root()

    print("Tinh FDE cho tung doan (chi predict qua doan che)...")
    fde = compute_fde(df, g)
    fde.to_csv(out / "segments_fde.csv", index=False)
    print(f"  tinh duoc cho {len(fde)}/{len(df)} doan\n")

    # --- Doan `lost` voi ca 3 model (cho Video 4) ---
    lost_keys = set()
    rp = out / "case_ab_with_retention.csv"
    if rp.exists():
        rr = pd.read_csv(rp)
        cols = ["KF + CV", "EKF + CTRV", "UKF + CTRV"]
        if all(c in rr.columns for c in cols):
            m = rr.dropna(subset=cols)
            allost = m[(m[cols] == "lost").all(axis=1)]
            lost_keys = set(zip(allost.video, allost.track_id, allost.seg_id))

    picks = []
    # Video 1: CASE_A, CTRV thang dam nhat
    c = fde[fde.case == "CASE_A"]
    if len(c):
        picks.append(("1_ctrv_thang", "CTRV THANG", c.loc[c.gain_ctrv.idxmax()], len(c)))
    # Video 2: STRAIGHT, CV thang dam nhat
    c = fde[fde.case == "STRAIGHT"]
    if len(c):
        picks.append(("2_cv_thang", "CV THANG", c.loc[c.gain_ctrv.idxmin()], len(c)))
    # Video 3: CASE_B, T_occ dai nhat
    c = fde[fde.case == "CASE_B"]
    if len(c):
        picks.append(("3_tran_thong_tin", "TRAN THONG TIN", c.loc[c.T_occ.idxmax()], len(c)))
    # Video 4: ca 3 model deu lost, T_occ gan trung vi
    c = fde[[(v, t, s) in lost_keys for v, t, s in zip(fde.video, fde.track_id, fde.seg_id)]]
    if len(c):
        med = c.T_occ.median()
        picks.append(("4_tran_detector", "TRAN DETECTOR",
                      c.loc[(c.T_occ - med).abs().idxmin()], len(c)))

    print("=" * 78)
    print("DOAN DUOC CHON (bang TIEU CHI TU DONG, khong chon tay)")
    print("=" * 78)
    rows = []
    for fname, label, row, npool in picks:
        sel = dict(row)
        sel["label"] = label
        print(f"\n  {label}  -> chon tu {npool} doan ung vien")
        print(f"    {sel['video']} track {int(sel['track_id'])} "
              f"frame {int(sel['frame_start'])}-{int(sel['frame_end'])} (T_occ={int(sel['T_occ'])})")
        print(f"    |w_before|={abs(sel['ob_deg']):.3f}  |w_during|={abs(sel['od_deg']):.3f} deg/f"
              f"   FDE: KF={sel['fde_kf']:.1f}px  EKF={sel['fde_ekf']:.1f}px")
        p = vids / f"{fname}.mp4"
        render(sel, g, img_root, p)
        print(f"    -> {p.name}")
        rows.append(dict(video_file=f"{fname}.mp4", loai=label, n_ung_vien=npool,
                         **{k: sel[k] for k in ["video", "track_id", "seg_id", "case",
                                                "frame_start", "frame_end", "T_occ",
                                                "ob_deg", "od_deg", "fde_kf", "fde_ekf"]}))
    pd.DataFrame(rows).to_csv(out / "selected_segments.csv", index=False)
    print(f"\n  Da luu -> {vids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
