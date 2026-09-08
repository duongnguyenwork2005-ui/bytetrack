"""
diagnose_seg2.py -- GIAI DOAN 0: sai so 1208 px la LOI GHEP CAP hay SAI SO NGOAI SUY THAT?

BOI CANH. Video demo cho thay MVI_40992 track 12, doan che 2 (frame 524-563):
FDE cua KF+CV = 161,4 px nhung EKF+CTRV = 556,5 px (ban goc) / 1208,4 px (ban chan
omega). Doan 1 va doan 3 cua CUNG track thi hai model gan nhu bang nhau. Phai loai tru
kha nang day la loi ky thuat truoc khi di giai thich bang mo hinh chuyen dong.

BA KHA NANG PHAI LOAI TRU
  (a) Ghep cap GT<->tracker noi nham sang xe khac
  (b) Lech chi so / khoang trong frame trong track GT lam bo loc chay sai so buoc
  (c) Du lieu GT cua chinh doan do bat thuong

CACH DUNG
    python src/demo/diagnose_seg2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402
from ekf_ctrv_clamped import EKFTrackerCTRVClamped  # noqa: E402
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH  # noqa: E402

VIDEO, TRACK = "MVI_40992", 12
SPANS = [(492, 517), (524, 563), (575, 609)]
FRAME_W, FRAME_H = 960, 540


def xyah(r):
    return np.array([r.cx, r.cy, r.bb_width / max(r.bb_height, 1e-6), r.bb_height], float)


def run(t, occ_frames, kind):
    """Chay bo loc, tra ve dict frame -> (cx, cy, v, omega). Khoa theo SO FRAME THAT."""
    if kind == "KF":
        f = KalmanFilterXYAH()
        get = lambda m: (m[0], m[1], np.nan, np.nan)
        ctrv = False
    else:
        f = EKFTrackerCTRVClamped() if kind == "EKFC" else EKFTrackerCTRV()
        get = lambda m, f=f: (m[f.CX], m[f.CY], m[f.V], m[f.OMEGA])
        ctrv = True
    rows = list(t.itertuples())
    m, c = f.initiate(xyah(rows[0]))
    seeded = False
    last_obs = int(rows[0].frame)
    out = {int(rows[0].frame): get(m)}
    for r in rows[1:]:
        vis = int(r.frame) not in occ_frames
        if ctrv and not seeded and vis:
            # KHOANG CACH THAT giua hai quan sat, khong phai 1. Bo loc nay chay XUYEN QUA
            # cac doan che, nen quan sat dau tien sau mot doan che co the cach quan sat
            # truoc do hang chuc frame. `initiate_from_motion` chia dich chuyen cho
            # n_frames de ra van toc; truyen 1.0 se thoi phong van toc dung bang so frame
            # da bi che (do duoc: 294 px/frame thay vi 10,9 px/frame - gap 27 lan).
            m, c = f.initiate_from_motion(
                m, c, xyah(r), n_frames=max(1, int(r.frame) - last_obs))
            seeded = True
        m, c = f.predict(m, c)
        if vis:
            m, c = f.update(m, c, xyah(r))
            last_obs = int(r.frame)
        out[int(r.frame)] = get(m)
    return out


def main() -> int:
    gt = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    t = gt[(gt.video == VIDEO) & (gt.track_id == TRACK)].sort_values("frame").reset_index(drop=True)
    occ_frames = {f for s, e in SPANS for f in range(s, e + 1)}
    out = config.RESULTS_DIR / "demo" / "diagnosis"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"GIAI DOAN 0 -- {VIDEO} track {TRACK}")
    print("=" * 78)

    # ---- (a) Co buoc ghep cap khong? ----
    print("\n[A] LOI GHEP CAP GT<->TRACKER?")
    print("    long_compare.py nap GT cua DUNG (video, track_id) nay tu parquet, chay bo loc")
    print("    tren chinh chuoi do, roi tinh FDE so voi GT cua CHINH track do.")
    print("    Khong doc file tracker, khong Hungarian, khong IoU -> KHONG TON TAI buoc ghep cap.")
    print("    => Kha nang (a) bi loai tru VE MAT CAU TRUC.")

    # ---- (b) Khoang trong frame ----
    fr = t.frame.to_numpy()
    gaps = np.diff(fr)
    n_gap = int((gaps > 1).sum())
    print("\n[B] KHOANG TRONG FRAME TRONG TRACK GT?")
    print(f"    Track chay tu frame {fr[0]} den {fr[-1]}, co {len(fr)} dong GT.")
    print(f"    Neu lien tuc thi phai co {fr[-1] - fr[0] + 1} dong.")
    print(f"    So cho nhay frame (gap > 1): {n_gap}")
    if n_gap:
        for i in np.where(gaps > 1)[0]:
            print(f"      frame {fr[i]} -> {fr[i + 1]}  (nhay {gaps[i]} frame)")
        print("    => CO khoang trong: bo loc goi predict 1 lan nhung thoi gian troi qua nhieu hon")
        print("       -> DAY LA MOT LOI THAT trong code demo.")
    else:
        print("    => Khong co khoang trong. Moi frame ung dung mot buoc predict. Kha nang (b) bi loai.")

    # ---- (c) Du lieu GT cua tung doan ----
    print("\n[C] DU LIEU GT CUA TUNG DOAN CHE")
    traj = {k: run(t, occ_frames, k) for k in ("KF", "EKF", "EKFC")}
    recs = []
    for i, (s, e) in enumerate(SPANS, 1):
        b = t[(t.frame < s) & (t.frame >= s - 15)]
        d = t[(t.frame >= s) & (t.frame <= e)]
        if not len(b) or not len(d):
            print(f"\n  Doan {i} (frame {s}-{e}): du lieu mong -- "
                  f"{len(b)} frame truoc che, {len(d)} frame trong che")
            continue
        gt_end = np.array([d.cx.iloc[-1], d.cy.iloc[-1]])
        row = dict(doan=i, start=s, end=e, T=e - s + 1,
                   n_truoc_che=len(b), box_w=float(b.bb_width.iloc[-1]),
                   box_h=float(b.bb_height.iloc[-1]),
                   occ_max=float(d.occlusion_ratio.max()),
                   occ_anchor=float(b.occlusion_ratio.iloc[-1]),
                   gt_x=float(gt_end[0]), gt_y=float(gt_end[1]))
        for k in ("KF", "EKF", "EKFC"):
            cx, cy, _v, _w = traj[k][int(d.frame.iloc[-1])]
            row["fde_" + k] = float(np.hypot(cx - gt_end[0], cy - gt_end[1]))
            row["x_" + k], row["y_" + k] = float(cx), float(cy)
            if k != "KF":
                anchor = traj[k].get(s - 1, (0.0, 0.0, np.nan, np.nan))
                row["v_anchor_" + k] = float(anchor[2])
                row["omega_anchor_" + k] = float(anchor[3])
        recs.append(row)

        print(f"\n  --- Doan {i}: frame {s}-{e} ({row['T']} frame) ---")
        print(f"    Lich su truoc che : {row['n_truoc_che']} frame")
        print(f"    Kich thuoc xe     : {row['box_w']:.0f} x {row['box_h']:.0f} px")
        print(f"    occlusion_ratio   : anchor = {row['occ_anchor']:.2f}, "
              f"lon nhat trong che = {row['occ_max']:.2f}")
        print(f"    GT o frame cuoi   : ({row['gt_x']:.0f}, {row['gt_y']:.0f})")
        for k, lab in [("KF", "KF + CV   "), ("EKF", "EKF goc   "), ("EKFC", "EKF chan w")]:
            x, y = row["x_" + k], row["y_" + k]
            inside = 0 <= x <= FRAME_W and 0 <= y <= FRAME_H
            extra = ""
            if k != "KF":
                extra = ("  | anchor: v=%.2f px/f, w=%.3f do/f"
                         % (row["v_anchor_" + k], np.rad2deg(row["omega_anchor_" + k])))
            print(f"      {lab}: du doan ({x:>7.0f},{y:>7.0f}) sai {row['fde_' + k]:>7.1f} px  "
                  f"{'TRONG khung' if inside else 'NGOAI khung hinh'}{extra}")

    pd.DataFrame(recs).to_csv(out / "phase0_segments.csv", index=False)

    # ---- Chuoi tung frame cua doan 2 ----
    s, e = SPANS[1]
    print(f"\n[D] CHUOI TUNG FRAME CUA DOAN 2 (frame {s - 3} .. {e + 2})")
    hdr = ("    " + "frame".rjust(6) + "che?".rjust(6) + "GT x".rjust(8) + "GT y".rjust(8)
           + "KF x".rjust(8) + "KF y".rjust(8) + "EKF x".rjust(8) + "EKF y".rjust(8)
           + "EKFC x".rjust(9) + "EKFC y".rjust(8) + "v".rjust(8) + "w do/f".rjust(9))
    print(hdr)
    print("    " + "-" * 96)
    gmap = {int(r.frame): r for r in t.itertuples()}
    per = []
    for f in range(s - 3, min(e + 3, int(t.frame.max()) + 1)):
        if f not in traj["KF"]:
            continue
        g = gmap.get(f)
        kx, ky, _, _ = traj["KF"][f]
        ex, ey, ev, ew = traj["EKF"][f]
        c2x, c2y, _, _ = traj["EKFC"][f]
        gx = float(g.cx) if g is not None else np.nan
        gy = float(g.cy) if g is not None else np.nan
        mark = "CHE" if f in occ_frames else "."
        print(f"    {f:>6}{mark:>6}{gx:>8.0f}{gy:>8.0f}{kx:>8.0f}{ky:>8.0f}"
              f"{ex:>8.0f}{ey:>8.0f}{c2x:>9.0f}{c2y:>8.0f}{ev:>8.2f}{np.rad2deg(ew):>9.3f}")
        per.append(dict(frame=f, che=f in occ_frames, gt_x=gx, gt_y=gy,
                        kf_x=kx, kf_y=ky, ekf_x=ex, ekf_y=ey,
                        ekfc_x=c2x, ekfc_y=c2y, ekf_v=ev, ekf_omega=ew))
    pd.DataFrame(per).to_csv(out / "phase0_seg2_perframe.csv", index=False)
    print(f"\n  Da luu -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
