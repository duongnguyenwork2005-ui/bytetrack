"""
stratified_analysis.py -- Giai doan 5: phan tich PHAN TANG theo do dai che khuat
va do cong quy dao.

===========================================================================
VI SAO CAN FILE NAY?
===========================================================================
Giai doan 3 va 4 cho thay mot nghich ly:

  - Mo phong (compare_extrapolation.py): CTRV tot hon CV toi 15x khi xe RE
    trong luc bi che khuat LAU.
  - Video that (chi so tong hop HOTA/MOTA/IDF1): EKF/UKF lai THAP hon baseline
    mot chut (-0.3%).

Ly do da phan tich: chi so tong hop tinh trung binh tren TOAN BO frame cua
TOAN BO track, trong khi uu the cua CTRV chi xuat hien o mot nhom nho:
63.4% doan che khuat >=0.90 trong UA-DETRAC la NGAN (<0.5s), la vung CV van
con tot. Hieu ung bi pha loang den muc khong con nhin thay.

File nay do dung cai can do: chia du lieu thanh cac TANG (stratum) theo
(do dai che khuat) x (do cong quy dao) roi so sanh 3 motion model TRONG TUNG TANG.

===========================================================================
DO CAI GI? -- "GIU DUOC ID XUYEN QUA DOAN CHE KHUAT"
===========================================================================
Day la phep do bam sat gia thuyet nhat. Voi moi doan che khuat cua mot xe:

    ... [xe ro] ... | [xe bi che tu start_frame den end_frame] | ... [xe ro lai] ...
                    ^                                          ^
                 id_before                                  id_after

  - `id_before` : tracker dang gan id nao cho xe nay NGAY TRUOC doan che
  - `id_after`  : tracker gan id nao cho xe nay NGAY SAU doan che

  => preserved : id_before == id_after  (tracker ngoai suy dung, giu duoc ID)
  => switched  : id_before != id_after  (tracker mat dau, cap ID moi)
  => lost      : khong tim lai duoc xe sau doan che

Trong luc bi che, detector gan nhu khong ra detection, nen viec giu duoc ID
phu thuoc TRUC TIEP vao chat luong ngoai suy cua motion model. Day chinh la
co che ma khoa luan gia thiet.

===========================================================================
GHEP GT <-> TRACKER BANG CACH NAO?
===========================================================================
File ket qua tracker dung id RIENG cua ByteTrack, khong lien quan gi den
track_id cua UA-DETRAC. Phai ghep lai theo tung frame:

  1. Voi moi frame: tinh ma tran IoU giua cac box GT va cac box tracker.
  2. Ghep cap toi uu bang thuat toan Hungarian (scipy linear_sum_assignment),
     bo cac cap co IoU < 0.5 (nguong chuan cua MOTChallenge).
  3. Ket qua: bang (video, frame, gt_id) -> tracker_id.

Dung Hungarian chu khong phai "chon IoU lon nhat" cho tung box: chon tham lam
co the gan 2 box GT vao cung 1 box tracker, lam sai lech thong ke.

===========================================================================
CACH DUNG
===========================================================================
    python src/stratified_analysis.py                      # ca 3 motion model
    python src/stratified_analysis.py --split-name DETRAC-all
    python src/stratified_analysis.py --plot               # kem hinh minh hoa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# Nguong IoU coi la ghep duoc GT <-> tracker (chuan MOTChallenge / TrackEval).
IOU_THRESHOLD = 0.5

# Cua so tim id_before / id_after quanh doan che khuat (frame).
# Khong gioi han vo han vi neu xe bien mat rat lau roi moi xuat hien lai thi
# do khong con phan anh chat luong ngoai suy nua.
CONTEXT_WINDOW = 30

GT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]
TRK_COLS = ["frame", "id", "x", "y", "w", "h", "conf"]


# ---------------------------------------------------------------------------
# 1. GHEP GT <-> TRACKER
# ---------------------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU giua 2 tap box dang (x, y, w, h). Tra ve ma tran (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1 = a[:, 0][:, None], a[:, 1][:, None]
    ax2, ay2 = (a[:, 0] + a[:, 2])[:, None], (a[:, 1] + a[:, 3])[:, None]
    bx1, by1 = b[:, 0][None, :], b[:, 1][None, :]
    bx2, by2 = (b[:, 0] + b[:, 2])[None, :], (b[:, 1] + b[:, 3])[None, :]

    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_video(gt: pd.DataFrame, trk: pd.DataFrame) -> pd.DataFrame:
    """Ghep GT voi tracker theo tung frame. Tra ve (frame, gt_id, tracker_id)."""
    rows = []
    # Gom theo frame mot lan thay vi loc lai DataFrame moi vong lap (nhanh hon nhieu).
    gt_by_frame = {f: g for f, g in gt.groupby("frame", sort=False)}
    trk_by_frame = {f: t for f, t in trk.groupby("frame", sort=False)}

    for frame, g in gt_by_frame.items():
        t = trk_by_frame.get(frame)
        if t is None or len(t) == 0:
            continue
        gb = g[["x", "y", "w", "h"]].to_numpy(dtype=float)
        tb = t[["x", "y", "w", "h"]].to_numpy(dtype=float)
        iou = iou_matrix(gb, tb)
        if iou.size == 0:
            continue
        # Hungarian toi da hoa IoU  ->  toi thieu hoa (-IoU)
        gi, ti = linear_sum_assignment(-iou)
        gid = g["id"].to_numpy()
        tid = t["id"].to_numpy()
        for r, c in zip(gi, ti):
            if iou[r, c] >= IOU_THRESHOLD:
                rows.append((frame, int(gid[r]), int(tid[c])))

    return pd.DataFrame(rows, columns=["frame", "gt_id", "tracker_id"])


# ---------------------------------------------------------------------------
# 2. DANH GIA TUNG DOAN CHE KHUAT
# ---------------------------------------------------------------------------
def eval_segments(seg: pd.DataFrame, match: pd.DataFrame,
                  gt_len: dict) -> pd.DataFrame:
    """Voi moi doan che khuat, xac dinh tracker co giu duoc ID hay khong.

    `match`  : bang (frame, gt_id, tracker_id) cua RIENG mot video
    `gt_len` : (gt_id) -> so frame GT co mat, de tinh recall
    """
    if len(match):
        by_track = {k: v for k, v in match.groupby("gt_id", sort=False)}
    else:
        by_track = {}

    out = []
    for r in seg.itertuples(index=False):
        m = by_track.get(r.track_id)
        status, id_before, id_after, recall_during = "no_before", None, None, 0.0

        if m is not None and len(m):
            fr = m["frame"].to_numpy()
            ti = m["tracker_id"].to_numpy()

            # --- trong luc bi che: co bao nhieu frame van bam duoc? ---
            during = (fr >= r.start_frame) & (fr <= r.end_frame)
            n_seg = r.end_frame - r.start_frame + 1
            recall_during = float(during.sum()) / max(1, n_seg)

            # --- ngay TRUOC doan che ---
            before = (fr < r.start_frame) & (fr >= r.start_frame - CONTEXT_WINDOW)
            # --- ngay SAU doan che ---
            after = (fr > r.end_frame) & (fr <= r.end_frame + CONTEXT_WINDOW)

            if before.any():
                id_before = int(ti[before][-1])   # lan bam cuoi cung truoc khi che
                if after.any():
                    id_after = int(ti[after][0])  # lan bam dau tien sau khi che
                    status = "preserved" if id_before == id_after else "switched"
                else:
                    status = "lost"

        out.append({
            "video": r.video, "track_id": r.track_id, "seg_id": r.seg_id,
            "bucket": r.bucket, "length_frames": r.length_frames,
            "duration_s": r.duration_s,
            "status": status, "id_before": id_before, "id_after": id_after,
            "recall_during": round(recall_during, 4),
            "gt_track_len": gt_len.get(r.track_id, 0),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 3. CHAY CHO 1 TRACKER
# ---------------------------------------------------------------------------
def analyse_tracker(split_name: str, tracker: str, seg_tables: dict,
                    videos: list[str]) -> dict:
    """Ghep GT<->tracker cho moi video roi danh gia tat ca cac doan che khuat."""
    trk_root = config.PROCESSED_DIR / "trackers" / split_name / tracker / "data"
    gt_root = config.PROCESSED_DIR / split_name

    per_seg = {k: [] for k in seg_tables}
    per_track_rows = []

    for video in tqdm(videos, desc=f"  {tracker}", unit="video"):
        gt_f = gt_root / video / "gt" / "gt.txt"
        tk_f = trk_root / f"{video}.txt"
        if not gt_f.exists() or not tk_f.exists() or tk_f.stat().st_size == 0:
            continue

        gt = pd.read_csv(gt_f, header=None, names=GT_COLS)
        trk = pd.read_csv(tk_f, header=None, names=TRK_COLS)
        match = match_video(gt, trk)

        gt_len = gt.groupby("id").size().to_dict()

        # recall tong the cua tung track GT (dung cho phan tich theo do cong)
        matched_per_track = match.groupby("gt_id").size().to_dict() if len(match) else {}
        for gid, n in gt_len.items():
            per_track_rows.append({
                "video": video, "track_id": int(gid), "gt_frames": int(n),
                "matched_frames": int(matched_per_track.get(gid, 0)),
                "recall": round(matched_per_track.get(gid, 0) / max(1, n), 4),
                "n_tracker_ids": int(match.loc[match["gt_id"] == gid, "tracker_id"].nunique())
                                  if len(match) else 0,
            })

        for key, tbl in seg_tables.items():
            s = tbl[tbl["video"] == video]
            if len(s):
                per_seg[key].append(eval_segments(s, match, gt_len))

    return {
        "segments": {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
                     for k, v in per_seg.items()},
        "tracks": pd.DataFrame(per_track_rows),
    }


# ---------------------------------------------------------------------------
# 4. TONG HOP THEO TANG
# ---------------------------------------------------------------------------
def summarise(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Ti le giu duoc ID theo tung tang."""
    # Chi tinh tren cac doan xac dinh duoc id_before (co can cu de danh gia).
    d = df[df["status"] != "no_before"]
    if not len(d):
        return pd.DataFrame()
    g = d.groupby(group_cols, observed=True)
    out = g.agg(
        n_segments=("status", "size"),
        n_preserved=("status", lambda s: (s == "preserved").sum()),
        n_switched=("status", lambda s: (s == "switched").sum()),
        n_lost=("status", lambda s: (s == "lost").sum()),
        recall_during=("recall_during", "mean"),
    ).reset_index()
    out["id_retention"] = (out["n_preserved"] / out["n_segments"]).round(4)
    out["recall_during"] = out["recall_during"].round(4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Giai doan 5: phan tich phan tang")
    ap.add_argument("--split-name", default="DETRAC-all")
    ap.add_argument("--trackers", nargs="*", default=None,
                    help="Mac dinh: cả 3 motion model trong config.MOTION_MODELS")
    ap.add_argument("--plot", action="store_true", help="Ve hinh minh hoa")
    args = ap.parse_args()

    # --- Danh sach tracker tuong ung 3 motion model ---
    stem = Path(config.YOLO_DEFAULT_MODEL).stem
    # Nhan ngan de bang ket qua doc duoc (desc trong config.py qua dai).
    SHORT = {"cv": "KF + CV", "ekf_ctrv": "EKF + CTRV", "ukf_ctrv": "UKF + CTRV"}
    label = {f"{stem}-{s['suffix']}": SHORT.get(k, k)
             for k, s in config.MOTION_MODELS.items()}
    trackers = args.trackers or [f"{stem}-{s['suffix']}" for s in config.MOTION_MODELS.values()]

    # --- Bang tang: do dai che khuat ---
    seg_tables = {
        "partial": pd.read_csv(config.INTERIM_DIR / "occlusion_segments.csv"),
        "full": pd.read_csv(config.INTERIM_DIR / "full_occlusion_segments.csv"),
    }
    # --- Bang tang: do cong quy dao ---
    cur = pd.read_csv(config.INTERIM_DIR / "track_curvature.csv")[
        ["video", "track_id", "curvature_class", "turn_class"]]

    videos = sorted(p.name for p in (config.PROCESSED_DIR / args.split_name).iterdir()
                    if p.is_dir())
    print(f"[stratified] split={args.split_name}  {len(videos)} video  "
          f"{len(trackers)} motion model")
    print(f"[stratified] nguong IoU={IOU_THRESHOLD}  cua so ngu canh={CONTEXT_WINDOW} frame")

    all_seg, all_trk = [], []
    for tracker in trackers:
        if not (config.PROCESSED_DIR / "trackers" / args.split_name / tracker / "data").is_dir():
            print(f"  [MISS] chua co ket qua cho {tracker}, bo qua.")
            continue
        res = analyse_tracker(args.split_name, tracker, seg_tables, videos)
        for key, d in res["segments"].items():
            if len(d):
                d = d.merge(cur, on=["video", "track_id"], how="left")
                d["occlusion_level"] = key
                d["tracker"] = tracker
                d["motion_model"] = label.get(tracker, tracker)
                all_seg.append(d)
        t = res["tracks"]
        if len(t):
            t = t.merge(cur, on=["video", "track_id"], how="left")
            t["tracker"] = tracker
            t["motion_model"] = label.get(tracker, tracker)
            all_trk.append(t)

    if not all_seg:
        print("[ERROR] Khong co ket qua tracker nao de phan tich.")
        return 1

    seg = pd.concat(all_seg, ignore_index=True)
    trk = pd.concat(all_trk, ignore_index=True)

    # --- CHI SO SANH TREN TAP DOAN CHUNG ---
    # Mot doan che khuat chi duoc dua vao bang so sanh neu CA 3 motion model
    # deu xac dinh duoc `id_before` cho no. Neu khong, moi model se co mau so
    # khac nhau (vi du 22 vs 24 doan) va ti le giu ID giua chung khong con so
    # sanh truc tiep duoc - day la loi tinh toan de mac phai khi phan tang.
    key = ["occlusion_level", "video", "track_id", "seg_id"]
    n_models = seg["motion_model"].nunique()
    cnt = (seg[seg["status"] != "no_before"]
           .groupby(key, observed=True)["motion_model"].nunique())
    common = set(cnt[cnt == n_models].index)
    seg["in_common"] = [tuple(r) in common for r in seg[key].to_numpy()]

    n_all = len(seg) // max(1, n_models)
    n_com = int(seg["in_common"].sum()) // max(1, n_models)
    print(f"[stratified] {n_com}/{n_all} doan duoc ca {n_models} motion model "
          f"xac dinh duoc id_before -> dung {n_com} doan de so sanh "
          f"(bo {n_all - n_com} doan khong du can cu).")
    seg_cmp = seg[seg["in_common"]]

    out_dir = config.RESULTS_DIR / "stratified"
    out_dir.mkdir(parents=True, exist_ok=True)
    seg.to_csv(out_dir / f"segment_outcomes_{args.split_name}.csv", index=False)
    trk.to_csv(out_dir / f"track_recall_{args.split_name}.csv", index=False)

    # --- Cac bang tong hop ---
    tables = {
        "by_bucket": summarise(seg_cmp, ["occlusion_level", "bucket", "motion_model"]),
        "by_curvature": summarise(seg_cmp, ["occlusion_level", "curvature_class", "motion_model"]),
        "by_bucket_curvature": summarise(
            seg_cmp, ["occlusion_level", "bucket", "curvature_class", "motion_model"]),
        "by_turn_class": summarise(seg_cmp, ["occlusion_level", "turn_class", "motion_model"]),
    }
    for name, t in tables.items():
        if len(t):
            t.to_csv(out_dir / f"{name}_{args.split_name}.csv", index=False)

    # --- In ra man hinh ---
    for lvl, lvl_name in [("full", "CHE KHUAT >= 0.90 (gan nhu vo hinh)"),
                          ("partial", "CHE KHUAT >= 0.10 (mot phan)")]:
        print("\n" + "=" * 78)
        print(f"{lvl_name} -- ti le GIU DUOC ID xuyen qua doan che")
        print("=" * 78)
        t = tables["by_bucket"]
        t = t[t["occlusion_level"] == lvl]
        if len(t):
            p = t.pivot_table(index="bucket", columns="motion_model",
                              values="id_retention", observed=True)
            n = t.pivot_table(index="bucket", columns="motion_model",
                              values="n_segments", observed=True)
            print("\n  [id_retention]"); print(p.to_string())
            print("\n  [n_segments]");   print(n.to_string())

        t2 = tables["by_bucket_curvature"]
        t2 = t2[(t2["occlusion_level"] == lvl) &
                (t2["curvature_class"].isin(["curved", "straight"]))]
        if len(t2):
            print("\n  [id_retention theo bucket x curvature]")
            p2 = t2.pivot_table(index=["bucket", "curvature_class"],
                                columns="motion_model", values="id_retention",
                                observed=True)
            n2 = t2.pivot_table(index=["bucket", "curvature_class"],
                                columns="motion_model", values="n_segments",
                                observed=True)
            print(p2.to_string())
            print("\n  [n_segments]"); print(n2.to_string())

    print(f"\n  Da luu -> {out_dir}")

    if args.plot:
        make_plots(tables, out_dir, args.split_name)
    return 0


def make_plots(tables: dict, out_dir: Path, split_name: str) -> None:
    """Hinh: ti le giu ID theo do dai che khuat, tach rieng nhom thang / cong."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tables["by_bucket_curvature"]
    t = t[(t["occlusion_level"] == "partial") &
          (t["curvature_class"].isin(["curved", "straight"]))]
    if not len(t):
        return
    order = ["short", "medium", "long"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, cls in zip(axes, ["straight", "curved"]):
        d = t[t["curvature_class"] == cls]
        for mm, g in d.groupby("motion_model", observed=True):
            g = g.set_index("bucket").reindex(order)
            ax.plot(order, g["id_retention"], marker="o", label=mm)
        ax.set_title(f"Quy dao {cls}")
        ax.set_xlabel("Do dai doan che khuat")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Ti le giu duoc ID")
    axes[0].legend(fontsize=8)
    fig.suptitle("Giai doan 5: giu ID xuyen qua che khuat, phan tang theo do cong")
    fig.tight_layout()
    p = out_dir / f"id_retention_{split_name}.png"
    fig.savefig(p, dpi=150)
    print(f"  Hinh -> {p}")


if __name__ == "__main__":
    raise SystemExit(main())
