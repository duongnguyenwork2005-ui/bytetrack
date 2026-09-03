"""
to_motchallenge.py -- Chuyen bang annotation chuan hoa sang format MOTChallenge
de TrackEval doc duoc.

FORMAT gt.txt (moi dong 1 bbox, phan cach bang dau phay)
--------------------------------------------------------
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

    frame       : so thu tu frame, 1-based
    id          : track id (duy nhat trong 1 video)
    bb_left/top : goc trai-tren cua bbox (pixel)
    bb_width/height : kich thuoc bbox (pixel)
    conf        : 1 = doi tuong duoc tinh diem, 0 = bo qua
    class       : nhan lop. TrackEval preset MOT17/MOT20 chi cham diem class = 1.
                  UA-DETRAC co 4 loai xe -> gop het ve class 1 (bai toan theo vet
                  phuong tien khong phan biet loai). Loai xe van duoc giu rieng
                  trong parquet + file vehicle_types.csv de phan tich sau.
    visibility  : do nhin thay = 1 - occlusion_ratio  (0 = bi che hoan toan)

CAU TRUC THU MUC XUAT RA (dung chuan TrackEval MotChallenge2DBox)
-----------------------------------------------------------------
    data/processed/
      seqmaps/
        DETRAC-all.txt              <- danh sach video de eval
      DETRAC-all/
        MVI_20011/
          gt/gt.txt
          gt/ignored_regions.txt    <- vung bo qua (KHONG phai chuan MOT, dung rieng)
          seqinfo.ini
        MVI_20012/
          ...

    Khi chay TrackEval o Giai doan 2, ket qua tracker se dat o:
    data/processed/trackers/DETRAC-all/<ten_tracker>/data/<video>.txt

VE `ignored_region` CUA UA-DETRAC
---------------------------------
UA-DETRAC danh dau san mot so VUNG TINH trong anh (via he, bai do xe xa, duong
nguoc chieu...) - xe trong nhung vung do KHONG duoc annotate. Neu khong xu ly,
detector se phat hien xe o day va bi tinh la False Positive oan.

Format MOTChallenge khong co cho de mo ta "vung bo qua", nen script ho tro 2 cach:
  1. (MAC DINH) Ghi ra file rieng gt/ignored_regions.txt. Giai doan 2 se loc bo
     detection roi vao cac vung nay TRUOC khi dua vao TrackEval.
  2. (--ignore-as-distractor) Ghi cac vung do thanh dong GT voi class = 8
     ("distractor" trong bang class chuan MOT17 - da doi chieu voi source code
     trackeval/datasets/mot_challenge_2d_box.py: class_name_to_class_id =
     {'pedestrian': 1, ..., 'distractor': 8, ..., 'crowd': 13}). Buoc preproc
     cua TrackEval se tu dong go bo
     detection khop voi chung. Cach nay lam file gt.txt phinh to (moi vung x moi
     frame) nen chi dung khi can.

CACH DUNG
---------
    python src/to_motchallenge.py                       # toan bo 60 video
    python src/to_motchallenge.py --videos MVI_20011 MVI_20012 --split-name DETRAC-demo
    python src/to_motchallenge.py --min-visibility 0.0  # giu tat ca bbox (mac dinh)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DISTRACTOR_CLASS = 8  # class "distractor" theo bang chuan MOT17, dung cho vung bo qua


def write_seqinfo(seq_dir: Path, video: str, seq_length: int, from_images: bool) -> None:
    """Ghi file seqinfo.ini - TrackEval doc de biet do dai video va kich thuoc anh."""
    note = "so anh thuc te tren dia" if from_images else "suy tu frame lon nhat trong annotation"
    text = (
        "[Sequence]\n"
        f"name={video}\n"
        "imDir=img1\n"
        f"frameRate={config.FPS}\n"
        f"seqLength={seq_length}\n"
        f"imWidth={config.IMG_W}\n"
        f"imHeight={config.IMG_H}\n"
        "imExt=.jpg\n"
        f"; seqLength lay tu: {note}\n"
    )
    (seq_dir / "seqinfo.ini").write_text(text, encoding="utf-8")


def count_images(video: str) -> int:
    """So anh thuc te cua video (0 neu chua tai bo anh).

    Thu muc anh duoc tim tu dong (config.find_images_root) vi moi ban mirror
    long thu muc mot kieu khac nhau. Ket qua duoc cache trong _IMG_ROOT de khong
    phai quet lai cay thu muc cho tung video.
    """
    global _IMG_ROOT
    if _IMG_ROOT is _UNSET:
        _IMG_ROOT = config.find_images_root()
    if _IMG_ROOT is None:
        return 0
    d = _IMG_ROOT / video
    if not d.is_dir():
        return 0
    return sum(1 for _ in d.glob("img*.jpg"))


_UNSET = object()
_IMG_ROOT = _UNSET


def convert(df: pd.DataFrame, df_ign: pd.DataFrame, split_name: str,
            min_visibility: float = 0.0, ignore_as_distractor: bool = False) -> pd.DataFrame:
    """Ghi toan bo video ra format MOTChallenge. Tra ve bang tom tat."""
    split_dir = config.PROCESSED_DIR / split_name
    seqmap_dir = config.PROCESSED_DIR / "seqmaps"
    split_dir.mkdir(parents=True, exist_ok=True)
    seqmap_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(df["video"].unique())
    summary = []

    for video in tqdm(videos, desc="  ghi gt.txt", unit="video", ncols=90):
        sub = df[df["video"] == video].sort_values(["frame", "track_id"])

        # Loc theo nguong visibility neu nguoi dung yeu cau (mac dinh giu tat ca).
        n_before = len(sub)
        if min_visibility > 0:
            sub = sub[sub["visibility"] >= min_visibility]

        seq_dir = split_dir / video
        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)

        # ---- gt.txt ----
        gt = pd.DataFrame({
            "frame": sub["frame"].astype(int),
            "id": sub["track_id"].astype(int),
            "bb_left": sub["bb_left"].round(2),
            "bb_top": sub["bb_top"].round(2),
            "bb_width": sub["bb_width"].round(2),
            "bb_height": sub["bb_height"].round(2),
            "conf": 1,
            "class": config.MOT_CLASS_ID,
            "visibility": sub["visibility"].round(4),
        })

        ign = df_ign[df_ign["video"] == video] if len(df_ign) else df_ign

        # ---- (tuy chon) them vung bo qua duoi dang distractor ----
        n_distractor = 0
        if ignore_as_distractor and len(ign):
            frames = sorted(sub["frame"].unique())
            # id am de chac chan khong dung voi track id that
            rows = []
            for k, r in enumerate(ign.itertuples(index=False)):
                for f in frames:
                    rows.append((f, -(k + 1), r.left, r.top, r.width, r.height,
                                 1, DISTRACTOR_CLASS, 1.0))
            dist = pd.DataFrame(rows, columns=gt.columns)
            n_distractor = len(dist)
            gt = pd.concat([gt, dist], ignore_index=True).sort_values(["frame", "id"])

        gt.to_csv(seq_dir / "gt" / "gt.txt", header=False, index=False)

        # ---- vung bo qua ghi rieng (luon luon) ----
        if len(ign):
            ign[["left", "top", "width", "height"]].round(2).to_csv(
                seq_dir / "gt" / "ignored_regions.txt", header=False, index=False)

        # ---- seqinfo.ini ----
        n_img = count_images(video)
        from_images = n_img > 0
        seq_length = n_img if from_images else int(sub["frame"].max())
        write_seqinfo(seq_dir, video, seq_length, from_images)

        summary.append({
            "video": video,
            "n_gt_boxes": len(sub),
            "n_dropped_by_visibility": n_before - len(sub),
            "n_tracks": sub["track_id"].nunique(),
            "n_frames_with_gt": sub["frame"].nunique(),
            "max_frame": int(sub["frame"].max()),
            "seq_length": seq_length,
            "seq_length_from_images": from_images,
            "n_ignored_regions": len(ign),
            "n_distractor_rows": n_distractor,
        })

    # ---- seqmap: danh sach video cho TrackEval ----
    seqmap = seqmap_dir / f"{split_name}.txt"
    seqmap.write_text("name\n" + "\n".join(videos) + "\n", encoding="utf-8")

    # ---- bang tra loai xe (vi gt.txt da gop het ve class 1) ----
    vt = (df.groupby(["video", "track_id"], observed=True)["vehicle_type"]
            .agg(lambda s: s.mode().iat[0] if len(s.mode()) else None)
            .reset_index())
    vt.to_csv(split_dir / "vehicle_types.csv", index=False)

    return pd.DataFrame(summary)


def main() -> int:
    ap = argparse.ArgumentParser(description="Chuyen UA-DETRAC sang format MOTChallenge")
    ap.add_argument("--videos", nargs="*", default=None, help="Chi chuyen cac video nay")
    ap.add_argument("--split-name", default="DETRAC-all", help="Ten split (thu muc + seqmap)")
    ap.add_argument("--min-visibility", type=float, default=0.0,
                    help="Bo bbox co visibility thap hon nguong (mac dinh 0 = giu tat ca)")
    ap.add_argument("--ignore-as-distractor", action="store_true",
                    help="Ghi ignored_region thanh dong GT class 13 trong gt.txt")
    args = ap.parse_args()

    pq = config.INTERIM_DIR / "detrac_train_annotations.parquet"
    if not pq.exists():
        print(f"[ERROR] Chua co {pq}. Chay src/parse_detrac_xml.py truoc.")
        return 1

    df = pd.read_parquet(pq)
    ign_csv = config.INTERIM_DIR / "detrac_ignored_regions.csv"
    df_ign = pd.read_csv(ign_csv) if ign_csv.exists() else pd.DataFrame(
        columns=["video", "region_idx", "left", "top", "width", "height"])

    if args.videos:
        want = set(args.videos)
        df = df[df["video"].isin(want)]
        df_ign = df_ign[df_ign["video"].isin(want)]
        if df.empty:
            print(f"[ERROR] Khong tim thay video nao trong {sorted(want)}")
            return 1

    print(f"[convert] {df['video'].nunique()} video, {len(df):,} bbox -> split '{args.split_name}'")
    s = convert(df, df_ign, args.split_name, args.min_visibility, args.ignore_as_distractor)

    out = config.INTERIM_DIR / f"motchallenge_summary_{args.split_name}.csv"
    s.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("TONG KET CHUYEN DOI")
    print("=" * 78)
    print(f"  Thu muc GT       : {config.PROCESSED_DIR / args.split_name}")
    print(f"  Seqmap           : {config.PROCESSED_DIR / 'seqmaps' / (args.split_name + '.txt')}")
    print(f"  So video         : {len(s)}")
    print(f"  Tong bbox GT     : {s['n_gt_boxes'].sum():,}")
    print(f"  Tong track       : {s['n_tracks'].sum():,}")
    print(f"  Tong vung bo qua : {s['n_ignored_regions'].sum():,}")
    if s["n_dropped_by_visibility"].sum():
        print(f"  Bbox bi loai bo  : {s['n_dropped_by_visibility'].sum():,} "
              f"(visibility < {args.min_visibility})")
    if not s["seq_length_from_images"].any():
        print("\n  [LUU Y] Chua co anh nen seqLength duoc suy tu frame lon nhat trong")
        print("          annotation. Sau khi tai bo anh, chay lai script nay de cap nhat.")
    print(f"\n  Bang chi tiet -> {out}")

    # In vai dong dau cua 1 file gt.txt de kiem tra bang mat
    demo = config.PROCESSED_DIR / args.split_name / s.iloc[0]["video"] / "gt" / "gt.txt"
    print(f"\n  5 dong dau cua {demo.parent.parent.name}/gt/gt.txt:")
    with demo.open() as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            print("    " + line.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
