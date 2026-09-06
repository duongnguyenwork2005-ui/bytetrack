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

# --- Phan tang theo DO CONG CUA TUNG DOAN CHE (Stage C) ---
# Duoi nguong toc do nay, huong di chuyen khong xac dinh duoc: xe dung yen /
# do xe, "goc quay" do duoc chi la nhieu annotation vai pixel moi frame.
# Da do: tren doan xe dung yen, MOI cach do do cong deu bi thoi phong 5.9-7.2 lan.
MIN_SEG_SPEED = 2.0          # px/frame

# Nguong chia nhom do cong, don vi do/frame (nhan 25 de ra do/giay).
# Chon tu phan vi thuc te cua 2.214 doan co xe di chuyen: 0.05 ~ phan vi 44%,
# 0.2 ~ phan vi 72% -> ba nhom co co mau tuong doi can (965 / 612 / 637).
SEG_CURV_BINS = [0.05, 0.20]
SEG_CURV_LABELS = ["thang", "cong nhe", "cong gat"]

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
        # sort_values("frame") la BAT BUOC: id_before/id_after lay phan tu dau/cuoi
        # cua mang theo gia dinh no tang dan theo frame. Dieu nay dung tren du lieu
        # hien tai (gt.txt va tracker .txt deu ghi tuan tu) nhung la gia dinh NGAM -
        # sap xep tuong minh o day de dung dan khong phu thuoc thu tu ghi file.
        by_track = {k: v for k, v in
                    match.sort_values("frame").groupby("gt_id", sort=False)}
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


def segment_curvature(seg_tables: dict) -> pd.DataFrame:
    """Do do cong CUA TUNG DOAN CHE (Stage C), thay vi do cong cua CA TRACK.

    VI SAO KHONG DUNG `curvature_class` SAN CO?
    `track_curvature.csv` gan nhan cho CA quy dao cua xe. Mot xe co the re gat o
    dau video roi di thang suot doan bi che - van bi gan nhan "curved". Phan tang
    kieu do khong tra loi duoc cau hoi cua de tai (CTRV co giup khi xe re TRONG
    LUC bi che khong). Da gap dung loi nay o Stage A2: mot track duoc gan
    net_turn = 139 do nhung trong doan che thi xe gan nhu dung yen.

    CACH DO: goc giua huong TRUNG BINH cua 1/3 dau va 1/3 cuoi doan, chia cho so
    frame -> don vi do/frame, so sanh truc tiep duoc voi omega cua CTRV.

    VI SAO KHONG LAY TONG |d(theta)| TICH LUY?
    Da do tren 2.911 doan: cach tich luy cho trung vi 5.92 do/frame o xe dung yen
    so voi 0.85 o xe di chuyen - bi nhieu annotation lan at (chi tuong quan 0.49
    voi phep khop duong tron hinh hoc). Cach lay huong trung binh dau/cuoi tuong
    quan 0.90 voi phep khop hinh hoc -> hai cach do cung mot dai luong.

    Luu y: MOI cach do deu vo nghia khi xe dung yen, nen phai loc theo
    MIN_SEG_SPEED chu khong phai tim cong thuc chong nhieu tot hon.
    """
    pq = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    g = {k: v.sort_values("frame") for k, v in pq.groupby(["video", "track_id"])}

    rows = []
    for lvl, tbl in seg_tables.items():
        for r in tbl.itertuples(index=False):
            t = g.get((r.video, r.track_id))
            if t is None:
                continue
            occ = t[(t.frame >= r.start_frame) & (t.frame <= r.end_frame)]
            if len(occ) < 6:
                continue
            p = np.c_[occ.cx.values, occ.cy.values]
            d = np.diff(p, axis=0)
            speed = float(np.hypot(*d.T).mean())
            k = max(2, len(d) // 3)
            v1, v2 = d[:k].mean(0), d[-k:].mean(0)
            if np.hypot(*v1) < 1e-9 or np.hypot(*v2) < 1e-9:
                turn_rate = 0.0
            else:
                a = np.arctan2(v1[1], v1[0])
                b = np.arctan2(v2[1], v2[0])
                turn_rate = abs(np.rad2deg((b - a + np.pi) % (2 * np.pi) - np.pi)) / len(d)
            rows.append(dict(occlusion_level=lvl, video=r.video, track_id=r.track_id,
                             seg_id=r.seg_id, seg_speed=round(speed, 3),
                             seg_turn_rate=round(turn_rate, 4)))
    df = pd.DataFrame(rows)
    # Xe dung yen -> do cong khong xac dinh, danh dau rieng thay vi gan bua vao 1 nhom
    df["seg_curv_class"] = np.where(
        df.seg_speed < MIN_SEG_SPEED, "dung yen",
        pd.cut(df.seg_turn_rate, [-np.inf] + SEG_CURV_BINS + [np.inf],
               labels=SEG_CURV_LABELS).astype(str))
    return df


def mcnemar(seg: pd.DataFrame, group_cols: list[str], baseline: str) -> pd.DataFrame:
    """Kiem dinh McNemar: chenh lech giu-ID giua CTRV va baseline co y nghia khong?

    VI SAO McNEMAR CHU KHONG PHAI CHI-SQUARE / t-TEST?
    Ba motion model chay tren CUNG MOT tap doan che khuat (cung video, cung xe,
    cung khoang frame) - day la du lieu GHEP CAP, khong phai 2 mau doc lap.
    Chi-square hay t-test 2 mau doc lap se danh gia sai vi bo qua tuong quan
    rat manh giua cac cap (doan nao kho thi kho voi ca 3 model).

    McNemar chi nhin vao cac cap BAT DONG:
        b = so doan baseline giu duoc ID nhung CTRV thi khong
        c = so doan CTRV giu duoc ID nhung baseline thi khong
    Neu 2 model tuong duong thi b va c phai xap xi nhau. Dung kiem dinh nhi
    thuc chinh xac (binomtest) thay vi xap xi chi-square, vi nhieu tang co
    b + c rat nho (chi vai doan).
    """
    from scipy.stats import binomtest

    out = []
    pv = seg.pivot_table(index=["occlusion_level", "video", "track_id", "seg_id"] ,
                         columns="motion_model", values="status",
                         aggfunc="first", observed=True)
    if baseline not in pv.columns:
        return pd.DataFrame()
    pv = pv.reset_index()
    # Lay MOI cot phan tang co trong seg (bucket, curvature_class, turn_class,
    # seg_curv_class...) thay vi liet ke cung, de goi mcnemar() voi bat ky
    # group_cols nao ma khong phai sua lai ham.
    key_cols = ["occlusion_level", "video", "track_id", "seg_id"]
    meta_cols = [c for c in seg.columns
                 if c in ("bucket", "curvature_class", "turn_class", "seg_curv_class")]
    meta = seg.drop_duplicates(subset=key_cols)[key_cols + meta_cols]
    pv = pv.merge(meta, on=["occlusion_level", "video", "track_id", "seg_id"], how="left")

    models = [c for c in seg["motion_model"].unique() if c != baseline]
    for keys, g in pv.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base_ok = (g[baseline] == "preserved")
        for m in models:
            if m not in g.columns:
                continue
            m_ok = (g[m] == "preserved")
            b = int((base_ok & ~m_ok).sum())    # baseline thang
            c = int((~base_ok & m_ok).sum())    # CTRV thang
            n = b + c
            # binomtest mac dinh alternative="two-sided" -> pvalue TRA VE DA LA
            # two-sided roi, KHONG duoc nhan doi lan nua (loi da mac phai truoc
            # do: lam moi p-value bao cao ra gap doi gia tri dung).
            p = binomtest(min(b, c), n, 0.5).pvalue if n else 1.0
            out.append(dict(zip(group_cols, keys)) | {
                "model": m, "n_segments": len(g),
                "baseline_only": b, "model_only": c,
                "delta_preserved": c - b,
                "p_value": round(min(1.0, p), 4),
                "significant_5pct": bool(n and min(1.0, p) < 0.05),
            })
    return pd.DataFrame(out)


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
    # --- Bang tang: do cong CA TRACK (Giai doan 1, giu de doi chung) ---
    cur = pd.read_csv(config.INTERIM_DIR / "track_curvature.csv")[
        ["video", "track_id", "curvature_class", "turn_class"]]
    # --- Bang tang: do cong TUNG DOAN CHE (Stage C - moi) ---
    segcur = segment_curvature(seg_tables)

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
                d = d.merge(segcur, on=["occlusion_level", "video", "track_id", "seg_id"],
                            how="left")
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
        # --- Stage C: do cong do theo TUNG DOAN CHE, khong phai ca track ---
        "by_seg_curv": summarise(seg_cmp, ["occlusion_level", "seg_curv_class", "motion_model"]),
        "by_bucket_seg_curv": summarise(
            seg_cmp, ["occlusion_level", "bucket", "seg_curv_class", "motion_model"]),
    }
    for name, t in tables.items():
        if len(t):
            t.to_csv(out_dir / f"{name}_{args.split_name}.csv", index=False)

    # --- Kiem dinh y nghia thong ke (McNemar ghep cap) ---
    base_label = "KF + CV"
    mc = {
        "by_bucket": mcnemar(seg_cmp, ["occlusion_level", "bucket"], base_label),
        "by_bucket_curvature": mcnemar(
            seg_cmp, ["occlusion_level", "bucket", "curvature_class"], base_label),
    }
    for name, t in mc.items():
        if len(t):
            t.to_csv(out_dir / f"mcnemar_{name}_{args.split_name}.csv", index=False)

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

    # --- Stage C: ma tran 2 chieu (do dai che x do cong TUNG DOAN) ---
    MIN_N = 20     # duoi nguong nay ti le % khong dang tin -> chi bao n, khong dien so
    t = tables["by_bucket_seg_curv"]
    for lvl, lvl_name in [("full", "CHE KHUAT >= 0.90"), ("partial", "CHE KHUAT >= 0.10")]:
        d = t[(t["occlusion_level"] == lvl) & (t["seg_curv_class"] != "dung yen")]
        if not len(d):
            continue
        print("\n" + "=" * 78)
        print(f"STAGE C - {lvl_name}: ti le giu ID theo (do dai che x DO CONG DOAN CHE)")
        print("=" * 78)
        p = d.pivot_table(index=["bucket", "seg_curv_class"], columns="motion_model",
                          values="id_retention", observed=True)
        nn = d.pivot_table(index=["bucket", "seg_curv_class"], columns="motion_model",
                           values="n_segments", observed=True)
        n_col = nn.iloc[:, 0]
        keep = n_col >= MIN_N
        print(f"\n  [O co n >= {MIN_N}]")
        print(p[keep].round(4).to_string() if keep.any() else "  (khong o nao du mau)")
        print(f"\n  [n moi o]")
        print(n_col.astype(int).to_string())
        if (~keep).any():
            print(f"\n  BO QUA {int((~keep).sum())} o co n < {MIN_N} (khong du mau de tin "
                  f"ti le %): {list(n_col[~keep].index)}")

    # --- In ket qua kiem dinh ---
    for name, t in mc.items():
        if not len(t):
            continue
        print("\n" + "=" * 78)
        print(f"KIEM DINH McNEMAR ({name}) -- CTRV so voi baseline {base_label}")
        print("=" * 78)
        print("  delta_preserved > 0 nghia la CTRV giu duoc ID tren NHIEU doan hon baseline.")
        print("  baseline_only / model_only = so doan CHI mot ben giu duoc ID (cap bat dong).")
        show = t[t["baseline_only"] + t["model_only"] > 0]
        if not len(show):
            print("  (khong co cap bat dong nao - 2 model cho ket qua y het nhau)")
            continue
        print(show.to_string(index=False))

    print(f"\n  Da luu -> {out_dir}")

    if args.plot:
        make_plots(tables, out_dir, args.split_name)
    return 0


def make_plots(tables: dict, out_dir: Path, split_name: str) -> None:
    """Hinh: ti le giu ID theo do dai che khuat.

    Ve CA HAI muc che khuat. Muc >=0.90 moi la kich ban then chot cua de tai
    (xe gan nhu vo hinh -> tracker buoc phai ngoai suy hoan toan); neu chi ve
    muc >=0.10 thi 3 duong gan nhu trung nhau va hinh khong noi len dieu gi.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["short", "medium", "long"]
    xlab = ["short\n(<0.5s)", "medium\n(0.5-1.5s)", "long\n(>1.5s)"]
    levels = [("full", "Che khuat >= 0.90 (xe gan nhu vo hinh)"),
              ("partial", "Che khuat >= 0.10 (mot phan)")]
    colors = {"KF + CV": "tab:orange", "EKF + CTRV": "tab:blue", "UKF + CTRV": "tab:green"}
    # Marker + do rong net khac nhau: nhieu tang co 2 model TRUNG y het gia tri,
    # neu chi khac mau thi duong ve sau che hoan toan duong ve truoc.
    styles = {"KF + CV": dict(marker="o", lw=2.6, ms=9, alpha=0.9, zorder=2),
              "EKF + CTRV": dict(marker="s", lw=1.8, ms=6, alpha=0.9, zorder=3),
              "UKF + CTRV": dict(marker="^", lw=1.2, ms=5, ls="--", alpha=0.9, zorder=4)}

    t_all = tables["by_bucket_curvature"]
    t_bkt = tables["by_bucket"]
    if not len(t_all):
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
    for i, (lvl, lvl_title) in enumerate(levels):
        # Cot 0: gop chung; cot 1-2: tach theo do cong
        panels = [("(gop chung)", t_bkt[t_bkt["occlusion_level"] == lvl], None)]
        for cls in ["straight", "curved"]:
            d = t_all[(t_all["occlusion_level"] == lvl) &
                      (t_all["curvature_class"] == cls)]
            panels.append((f"quy dao {cls}", d, cls))

        for j, (title, d, _) in enumerate(panels):
            ax = axes[i, j]
            if len(d):
                for mm, g in d.groupby("motion_model", observed=True):
                    g = g.set_index("bucket").reindex(order)
                    ax.plot(range(3), g["id_retention"], label=mm,
                            color=colors.get(mm), **styles.get(mm, {}))
                # Ghi co mau n ngay tren truc - nguoi doc phai thay n truoc khi tin
                g0 = d[d["motion_model"] == "KF + CV"].set_index("bucket").reindex(order)
                for k, n in enumerate(g0["n_segments"]):
                    if pd.notna(n):
                        ax.annotate(f"n={int(n)}", (k, 0), xytext=(0, 4),
                                    textcoords="offset points", ha="center",
                                    fontsize=7, color="gray")
            ax.set_xticks(range(3))
            ax.set_xticklabels(xlab, fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_title(f"{lvl_title}\n{title}" if j == 0 else title, fontsize=9)

        # Dat ylim theo GIA TRI LON NHAT CA HANG. Cac panel dung sharey nen neu
        # de moi panel tu autoscale thi diem cao nhat cua panel khac bi cat mat
        # (vd straight/short = 0.269 > max cua panel gop chung = 0.233).
        # max() python thuong khong an toan voi NaN (mot tang co the rong sau khi
        # loc theo in_common -> .max() tren Series rong tra ve NaN). Loc NaN truoc
        # khi so sanh, neu khong set_ylim co the nhan NaN va ve sai/vo hinh ca hang.
        candidates = [t_bkt.loc[t_bkt["occlusion_level"] == lvl, "id_retention"].max()]
        candidates += [t_all.loc[(t_all["occlusion_level"] == lvl) &
                                 (t_all["curvature_class"] == c), "id_retention"].max()
                      for c in ["straight", "curved"]]
        candidates = [v for v in candidates if pd.notna(v)]
        rmax = max(candidates) if candidates else 1.0
        axes[i, 0].set_ylim(0, float(rmax) * 1.18)
        axes[i, 0].set_ylabel("Ti le giu duoc ID")
    axes[0, 0].legend(fontsize=8)

    fig.suptitle("Giai doan 5: ti le giu duoc ID xuyen qua doan che khuat "
                 "(phan tang theo do dai che x do cong quy dao)", fontsize=12)
    fig.tight_layout()
    p = out_dir / f"id_retention_{split_name}.png"
    fig.savefig(p, dpi=150)
    print(f"  Hinh -> {p}")


if __name__ == "__main__":
    raise SystemExit(main())
