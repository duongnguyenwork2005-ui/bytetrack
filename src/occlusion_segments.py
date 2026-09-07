"""
occlusion_segments.py -- Phat hien cac DOAN che khuat lien tuc cua tung track
va tinh do dai moi doan (theo frame va theo giay).

MUC DICH TRONG DE TAI
---------------------
Cau hoi nghien cuu: "KF / EKF / UKF chiu duoc che khuat bao lau?".
De tra loi, ta phai biet moi track bi che trong nhung khoang nao va moi khoang
keo dai bao lau, roi moi phan nhom ket qua tracking theo do dai do:

    short  : < 0.5 s      (< 12.5 frame o 25 fps)
    medium : 0.5 - 1.5 s  (12.5 - 37.5 frame)
    long   : > 1.5 s      (> 37.5 frame)

BA MUC DO "CHE KHUAT" - PHAI TACH RIENG
----------------------------------------
1. CHE KHUAT MOT PHAN (partial occlusion) - bang A
   Xe van co trong annotation, occlusion_ratio >= OCCLUSION_MIN_RATIO (0.10).
   Detector thuong van thay xe (bbox nho hon / lech), tracker chu yeu sai o
   buoc data association.

2. CHE KHUAT GAN NHU HOAN TOAN - bang B
   occlusion_ratio >= OCCLUSION_FULL_RATIO (0.90). Xe VAN duoc annotate nhung
   gan nhu vo hinh trong anh => detector gan nhu chac chan khong ra detection.
   Tracker buoc phai ngoai suy vi tri hoan toan bang motion model.
   => DAY LA KICH BAN THEN CHOT de so sanh CV (KF) voi CTRV (EKF/UKF).

3. TRACK GAP (xe bien mat khoi annotation roi quay lai) - bang C
   KET QUA THUC TE TREN UA-DETRAC TRAIN: KHONG CO TRUONG HOP NAO (0 / 5952 track).
   UA-DETRAC annotate LIEN TUC moi frame tu luc xe xuat hien den luc roi khoi
   khung hinh, ke ca khi bi che 100%. Script van tinh bang C de:
     - lam bang chung kiem tra (ghi trong luan van),
     - dung lai duoc neu sau nay chay tren dataset khac (MOT17, KITTI...).

THUAT TOAN PHAT HIEN DOAN CHE KHUAT
------------------------------------
Voi moi track (video, track_id), sap xep theo frame:
  B1. Danh dau moi frame: occluded = (occlusion_ratio >= OCCLUSION_MIN_RATIO).
  B2. Gom cac frame occluded lien tiep (frame lien tiep nhau ve so hieu) thanh doan.
  B3. GOP hai doan gan nhau neu khoang trong giua chung <= OCCLUSION_MERGE_GAP frame.
      (Ly do: annotation co nhieu, mot frame lot ra ngoai nguong khong co nghia la
       xe da het bi che -> neu khong gop se bi cat vun thanh nhieu doan ngan gia.)
  B4. LOAI cac doan ngan hon OCCLUSION_MIN_LEN_FRAMES.
  B5. Voi moi doan, tinh: do dai (frame / giay), muc che trung binh / lon nhat,
      nguon che (xe khac / vat nen), quang duong tam bbox di duoc trong doan.

OUTPUT
------
    data/interim/occlusion_segments.csv       <- 1 dong = 1 doan che khuat mot phan
    data/interim/full_occlusion_segments.csv  <- 1 dong = 1 doan che khuat >= 90%
    data/interim/track_gaps.csv               <- 1 dong = 1 lan track bien mat roi quay lai
    data/interim/track_occlusion_summary.csv  <- 1 dong = 1 track, tong hop

CACH DUNG
---------
    python src/occlusion_segments.py
    python src/occlusion_segments.py --min-ratio 0.2 --merge-gap 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# Tim doan lien tuc
# ---------------------------------------------------------------------------
def find_runs(frames: np.ndarray, mask: np.ndarray, merge_gap: int):
    """Tim cac doan frame lien tuc thoa `mask`, cho phep gop qua khoang trong ngan.

    Parameters
    ----------
    frames : mang so hieu frame (da sap xep tang dan, co the khong lien tuc)
    mask   : mang bool cung do dai, True = frame nay thoa dieu kien
    merge_gap : gop 2 doan neu khoang cach giua chung <= merge_gap frame

    Returns
    -------
    list[(i_start, i_end)] : chi so (bao gom ca 2 dau) trong mang `frames`

    Vi du:
        frames = [1,2,3,4,5,6,7,8]
        mask   = [T,T,F,T,T,F,F,T]
        merge_gap = 1
        -> [(0,4), (7,7)]   # doan 0..1 va 3..4 duoc gop vi cach nhau 1 frame
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    runs = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        # Khoang cach tinh theo SO HIEU FRAME, khong phai chi so mang,
        # de xu ly dung khi track co frame bi thieu.
        gap = frames[i] - frames[prev] - 1
        if gap <= merge_gap:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    runs.append((start, prev))
    return runs


# ---------------------------------------------------------------------------
# Phan nhom do dai
# ---------------------------------------------------------------------------
def bucket_of(duration_s: float) -> str:
    """Xep 1 doan vao nhom short / medium / long theo config.OCCLUSION_BUCKETS."""
    for name, lo, hi in config.OCCLUSION_BUCKETS:
        if lo <= duration_s < hi:
            return name
    return config.OCCLUSION_BUCKETS[-1][0]


# ---------------------------------------------------------------------------
# Xu ly 1 track
# ---------------------------------------------------------------------------
def process_track(g: pd.DataFrame, min_ratio: float, merge_gap: int, min_len: int):
    """Tra ve (segments, full_segments, gaps, summary) cho 1 track."""
    g = g.sort_values("frame")
    frames = g["frame"].to_numpy()
    ratio = g["occlusion_ratio"].to_numpy()
    cx = g["cx"].to_numpy()
    cy = g["cy"].to_numpy()
    src = g["occlusion_source"].astype(str).to_numpy()
    video = g["video"].iat[0]
    tid = int(g["track_id"].iat[0])

    # ---------- 1. Doan che khuat mot phan ----------
    mask = ratio >= min_ratio
    segments = []
    for k, (i0, i1) in enumerate(find_runs(frames, mask, merge_gap)):
        n_frames = int(frames[i1] - frames[i0] + 1)
        if n_frames < min_len:
            continue
        seg_ratio = ratio[i0:i1 + 1]
        seg_src = src[i0:i1 + 1]
        duration_s = n_frames / config.FPS

        # Quang duong tam bbox di duoc trong doan (px) - dung de danh gia
        # motion model phai ngoai suy xa bao nhieu.
        dx = np.diff(cx[i0:i1 + 1])
        dy = np.diff(cy[i0:i1 + 1])
        path_len = float(np.sum(np.hypot(dx, dy)))
        net_disp = float(np.hypot(cx[i1] - cx[i0], cy[i1] - cy[i0]))

        # Nguon che chinh trong doan
        vals, counts = np.unique(seg_src[seg_src != "none"], return_counts=True)
        main_src = vals[np.argmax(counts)] if len(vals) else "none"

        segments.append({
            "video": video, "track_id": tid, "seg_id": k,
            "start_frame": int(frames[i0]), "end_frame": int(frames[i1]),
            "length_frames": n_frames,
            "duration_s": round(duration_s, 4),
            "bucket": bucket_of(duration_s),
            "mean_occlusion_ratio": round(float(seg_ratio.mean()), 4),
            "max_occlusion_ratio": round(float(seg_ratio.max()), 4),
            "is_heavy": bool(seg_ratio.max() >= 0.7),   # che nang
            "main_source": main_src,
            "path_len_px": round(path_len, 2),
            "net_disp_px": round(net_disp, 2),
        })

    # ---------- 1b. Doan CHE KHUAT GAN NHU HOAN TOAN ----------
    # UA-DETRAC van annotate xe ngay ca khi bi che 100%. Detector gan nhu chac
    # chan khong ra detection nao trong khoang nay => tracker phai ngoai suy
    # thuan tuy bang motion model. Day la kich ban then chot cua de tai.
    full_segments = []
    mask_full = ratio >= config.OCCLUSION_FULL_RATIO
    for k, (i0, i1) in enumerate(find_runs(frames, mask_full, merge_gap)):
        n_frames = int(frames[i1] - frames[i0] + 1)
        duration_s = n_frames / config.FPS
        dx = np.diff(cx[i0:i1 + 1])
        dy = np.diff(cy[i0:i1 + 1])
        full_segments.append({
            "video": video, "track_id": tid, "seg_id": k,
            "start_frame": int(frames[i0]), "end_frame": int(frames[i1]),
            "length_frames": n_frames,
            "duration_s": round(duration_s, 4),
            "bucket": bucket_of(duration_s),
            "mean_occlusion_ratio": round(float(ratio[i0:i1 + 1].mean()), 4),
            # quang duong xe di duoc trong luc "vo hinh" -> motion model phai
            # ngoai suy dung chung nay thi moi giu duoc id
            "path_len_px": round(float(np.sum(np.hypot(dx, dy))), 2),
            "net_disp_px": round(float(np.hypot(cx[i1] - cx[i0], cy[i1] - cy[i0])), 2),
        })

    # ---------- 2. Track gap: xe bien mat khoi annotation roi quay lai ----------
    gaps = []
    d = np.diff(frames)
    for k, j in enumerate(np.flatnonzero(d > 1)):
        gap_len = int(d[j] - 1)
        duration_s = gap_len / config.FPS
        gaps.append({
            "video": video, "track_id": tid, "gap_id": k,
            "last_seen_frame": int(frames[j]),
            "reappear_frame": int(frames[j + 1]),
            "gap_frames": gap_len,
            "duration_s": round(duration_s, 4),
            "bucket": bucket_of(duration_s),
            # Xe da di chuyen bao xa trong luc bien mat -> motion model phai
            # ngoai suy dung khoang nay thi moi ghep lai duoc dung id.
            "jump_px": round(float(np.hypot(cx[j + 1] - cx[j], cy[j + 1] - cy[j])), 2),
            # Muc che khuat ngay truoc khi bien mat (thuong cao -> bi che dan roi mat han)
            "occ_ratio_before": round(float(ratio[j]), 4),
            "occ_ratio_after": round(float(ratio[j + 1]), 4),
        })

    # ---------- 3. Tong hop theo track ----------
    n_occ_frames = int(mask.sum())
    summary = {
        "video": video, "track_id": tid,
        "first_frame": int(frames[0]), "last_frame": int(frames[-1]),
        "n_frames_observed": int(len(frames)),
        "track_span_frames": int(frames[-1] - frames[0] + 1),
        "n_occluded_frames": n_occ_frames,
        "occluded_frame_ratio": round(n_occ_frames / len(frames), 4),
        "mean_occlusion_ratio": round(float(ratio.mean()), 4),
        "max_occlusion_ratio": round(float(ratio.max()), 4),
        "n_occlusion_segments": len(segments),
        "longest_segment_frames": max((s["length_frames"] for s in segments), default=0),
        "longest_segment_s": round(max((s["duration_s"] for s in segments), default=0.0), 4),
        "n_full_occlusion_segments": len(full_segments),
        "longest_full_segment_frames": max((s_["length_frames"] for s_ in full_segments), default=0),
        "n_gaps": len(gaps),
        "longest_gap_frames": max((g_["gap_frames"] for g_ in gaps), default=0),
        "vehicle_type": str(g["vehicle_type"].iat[0]),
    }
    return segments, full_segments, gaps, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Phat hien doan che khuat trong UA-DETRAC")
    ap.add_argument("--min-ratio", type=float, default=config.OCCLUSION_MIN_RATIO,
                    help="Nguong occlusion_ratio de coi la bi che")
    ap.add_argument("--merge-gap", type=int, default=config.OCCLUSION_MERGE_GAP,
                    help="Gop 2 doan cach nhau <= so frame nay")
    ap.add_argument("--min-len", type=int, default=config.OCCLUSION_MIN_LEN_FRAMES,
                    help="Bo doan ngan hon so frame nay")
    ap.add_argument("--part", choices=["train", "test"], default="train",
                    help="Phan dataset can xu ly. Giai doan D dung --part test "
                         "(tap test 40 video). File ket qua cua train GIU NGUYEN ten cu.")
    args = ap.parse_args()

    pq = config.ann_parquet(args.part)
    if not pq.exists():
        print(f"[ERROR] Chua co {pq}. Chay src/parse_detrac_xml.py truoc.")
        return 1

    df = pd.read_parquet(pq)
    print(f"[occlusion] {len(df):,} bbox, {df.groupby(['video', 'track_id']).ngroups:,} track")
    print(f"[occlusion] nguong={args.min_ratio}  merge_gap={args.merge_gap}  min_len={args.min_len}")

    all_seg, all_full, all_gap, all_sum = [], [], [], []
    groups = df.groupby(["video", "track_id"], observed=True, sort=True)
    for _, g in tqdm(groups, total=groups.ngroups, desc="  xu ly track", unit="track", ncols=90):
        seg, full, gap, summ = process_track(g, args.min_ratio, args.merge_gap, args.min_len)
        all_seg.extend(seg)
        all_full.extend(full)
        all_gap.extend(gap)
        all_sum.append(summ)

    df_seg = pd.DataFrame(all_seg)
    df_full = pd.DataFrame(all_full)
    df_gap = pd.DataFrame(all_gap)
    df_sum = pd.DataFrame(all_sum)

    df_seg.to_csv(config.part_file("occlusion_segments.csv", args.part), index=False)
    df_full.to_csv(config.part_file("full_occlusion_segments.csv", args.part), index=False)
    df_gap.to_csv(config.part_file("track_gaps.csv", args.part), index=False)
    df_sum.to_csv(config.part_file("track_occlusion_summary.csv", args.part), index=False)

    # ------------------------------------------------------------------
    # Bao cao
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("A. DOAN CHE KHUAT MOT PHAN (xe van co trong annotation)")
    print("=" * 78)
    print(f"  Tong so doan            : {len(df_seg):,}")
    print(f"  So track co it nhat 1 doan: {df_sum['n_occlusion_segments'].gt(0).sum():,}"
          f" / {len(df_sum):,} ({df_sum['n_occlusion_segments'].gt(0).mean() * 100:.1f}%)")
    if len(df_seg):
        print(f"  Do dai doan (frame)     : min={df_seg['length_frames'].min()}  "
              f"median={df_seg['length_frames'].median():.0f}  "
              f"mean={df_seg['length_frames'].mean():.1f}  "
              f"max={df_seg['length_frames'].max()}")
        print(f"  Do dai doan (giay)      : median={df_seg['duration_s'].median():.2f}s  "
              f"max={df_seg['duration_s'].max():.2f}s")
        print("\n  Phan bo theo nhom do dai:")
        for name, lo, hi in config.OCCLUSION_BUCKETS:
            sel = df_seg[df_seg["bucket"] == name]
            hi_s = "inf" if hi == float("inf") else f"{hi}"
            print(f"    {name:<7} [{lo} - {hi_s} s) : {len(sel):>6,} doan "
                  f"({len(sel) / len(df_seg) * 100:5.1f}%)   "
                  f"che nang(>=0.7): {sel['is_heavy'].sum():>5,}")
        print("\n  Nguon che khuat chinh:")
        for k, v in df_seg["main_source"].value_counts().items():
            print(f"    {str(k):<12} {v:>6,}  ({v / len(df_seg) * 100:5.1f}%)")

    print("\n" + "=" * 78)
    print(f"B. DOAN CHE KHUAT GAN NHU HOAN TOAN (occlusion_ratio >= {config.OCCLUSION_FULL_RATIO})")
    print("=" * 78)
    print("  Xe VAN co trong GT nhung gan nhu vo hinh -> detector khong thay,")
    print("  tracker phai ngoai suy hoan toan bang motion model.")
    print(f"  Tong so doan            : {len(df_full):,}")
    n_tr_full = int(df_sum["n_full_occlusion_segments"].gt(0).sum())
    print(f"  So track dinh phai      : {n_tr_full:,} / {len(df_sum):,} "
          f"({n_tr_full / len(df_sum) * 100:.1f}%)")
    if len(df_full):
        print(f"  Do dai doan (frame)     : min={df_full['length_frames'].min()}  "
              f"median={df_full['length_frames'].median():.0f}  "
              f"mean={df_full['length_frames'].mean():.1f}  "
              f"max={df_full['length_frames'].max()}")
        print(f"  Quang duong di duoc(px) : median={df_full['path_len_px'].median():.1f}  "
              f"max={df_full['path_len_px'].max():.1f}")
        print()
        print("  Phan bo theo nhom do dai:")
        for name, lo, hi in config.OCCLUSION_BUCKETS:
            sel = df_full[df_full["bucket"] == name]
            hi_s = "inf" if hi == float("inf") else f"{hi}"
            print(f"    {name:<7} [{lo} - {hi_s} s) : {len(sel):>6,} doan "
                  f"({len(sel) / len(df_full) * 100:5.1f}%)")
    print()
    print("=" * 78)
    print("C. TRACK GAP (xe bien mat khoi annotation roi quay lai - che khuat hoan toan)")
    print("=" * 78)
    print(f"  Tong so gap             : {len(df_gap):,}")
    print(f"  So track co gap         : {df_sum['n_gaps'].gt(0).sum():,}"
          f" / {len(df_sum):,} ({df_sum['n_gaps'].gt(0).mean() * 100:.1f}%)")
    if len(df_gap):
        print(f"  Do dai gap (frame)      : min={df_gap['gap_frames'].min()}  "
              f"median={df_gap['gap_frames'].median():.0f}  "
              f"max={df_gap['gap_frames'].max()}")
        print(f"  Khoang nhay vi tri (px) : median={df_gap['jump_px'].median():.1f}  "
              f"max={df_gap['jump_px'].max():.1f}")
        print("\n  Phan bo theo nhom do dai:")
        for name, lo, hi in config.OCCLUSION_BUCKETS:
            sel = df_gap[df_gap["bucket"] == name]
            hi_s = "inf" if hi == float("inf") else f"{hi}"
            print(f"    {name:<7} [{lo} - {hi_s} s) : {len(sel):>6,} gap "
                  f"({len(sel) / len(df_gap) * 100:5.1f}%)")

    print("\n  Da luu:")
    for f in [config.part_file(x, args.part).name for x in
              ["occlusion_segments.csv", "full_occlusion_segments.csv",
               "track_gaps.csv", "track_occlusion_summary.csv"]]:
        print(f"    {config.INTERIM_DIR / f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
