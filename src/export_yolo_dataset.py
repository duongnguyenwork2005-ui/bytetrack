"""
export_yolo_dataset.py -- Giai doan C: xuat UA-DETRAC sang dinh dang YOLO de fine-tune.

===========================================================================
BA QUYET DINH THIET KE (deu co ly do, can trinh bay duoc truoc hoi dong)
===========================================================================

--- 1. CHIA TRAIN/VAL THEO VIDEO, KHONG THEO FRAME ---
Cac frame lien ke trong mot video gan nhu trung nhau (25 fps, xe di vai pixel).
Neu chia ngau nhien theo frame thi mot frame trong val se co "anh em sinh doi"
nam trong train -> mo hinh da nhin thay gan nhu dung anh do luc hoc -> mAP tren
val cao gia, khong phan anh kha nang tong quat hoa. Bat buoc chia theo VIDEO.

--- 2. LAY MAU THUA FRAME (frame stride) ---
Cung ly do tren: 82.085 anh nhung so anh THUC SU KHAC NHAU it hon nhieu. Huan
luyen tren toan bo vua lang phi vua khong them thong tin. Mac dinh lay 1 frame
moi 5 frame (~16.400 anh), van du da dang ma nhanh hon 5 lan.

--- 3. TO XAM VUNG BO QUA (ignored_region) ---
UA-DETRAC danh dau mot so vung tinh (via he, bai do xa, duong nguoc chieu) noi
xe KHONG duoc annotate. Neu giu nguyen anh goc thi trong vung do co xe THAT
nhung KHONG co nhan -> YOLO hoc rang "xe o vung nay la nen", tao nhieu nhan
co hai.

Quan trong hon: pipeline suy dien (baseline_track.py) DA to xam cac vung nay
truoc khi dua vao detector. Neu huan luyen tren anh KHONG to xam thi train va
test lech nhau. Vi vay o day to xam Y HET cach lam luc suy dien, dung cung mau
(114,114,114 - trung mau padding cua YOLO nen khong tao canh gia).

--- Ve so lop ---
Xuat 1 LOP DUY NHAT ("vehicle"). Ly do: toan bo pipeline danh gia coi moi loai
xe la mot lop (config.MOT_CLASS_ID = 1, gt.txt ghi class=1). Phan loai car/van/
bus/others khong duoc dung o bat ky buoc danh gia nao.

===========================================================================
CACH DUNG
    python src/export_yolo_dataset.py --stride 5 --val-videos MVI_x MVI_y ...
    python src/export_yolo_dataset.py --stride 5 --n-val 12        # tu chon val
"""
from __future__ import annotations

import argparse

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def pick_val_videos(n_val: int, seed: int = 0) -> list[str]:
    """Chon video val phan tang theo thoi tiet + muc do che khuat.

    Khong chon ngau nhien thuan: tap val phai phu du dieu kien (ngay/dem/mua,
    che khuat nhieu/it), neu khong mAP tren val se khong dai dien.
    """
    df = pd.read_csv(config.INTERIM_DIR / "video_selection_stats.csv")
    df["occ_bin"] = pd.qcut(df["pct_tracks_occluded"].rank(method="first"), 3,
                            labels=["thap", "vua", "cao"])
    rng = np.random.default_rng(seed)
    out = []
    for _, grp in df.groupby(["sence_weather", "occ_bin"], observed=True):
        k = max(1, round(n_val * len(grp) / len(df)))
        out += list(rng.choice(grp.video.values, size=min(k, len(grp)), replace=False))
    out = sorted(set(out))[:n_val]
    return out


def make_folds(videos: list[str], n_folds: int, seed: int = 0) -> list[list[str]]:
    """Chia video thanh n_folds nhom, PHAN TANG theo thoi tiet + muc che khuat.

    Moi fold se lam tap VAL mot lan; detector cua fold do huan luyen tren cac
    fold con lai. Nho vay MOI video deu duoc suy dien bang mot detector CHUA
    TUNG NHIN THAY no -> khong ro ri sang buoc danh gia tracking.
    """
    df = pd.read_csv(config.INTERIM_DIR / "video_selection_stats.csv")
    df = df[df.video.isin(videos)].copy()
    df["occ_bin"] = pd.qcut(df["pct_tracks_occluded"].rank(method="first"), 3,
                            labels=["thap", "vua", "cao"])
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    # Rai deu tung to (thoi tiet x muc che) vao cac fold de fold nao cung du dang.
    # Bo dem `pos` chay LIEN TUC qua cac to: neu reset ve 0 o moi to thi cac to
    # nho (1-2 video) luon roi vao fold 0 -> fold lech han nhau (da gap: 25/20/15).
    pos = 0
    for _, grp in df.groupby(["sence_weather", "occ_bin"], observed=True):
        for v in rng.permutation(grp.video.values):
            folds[pos % n_folds].append(str(v))
            pos += 1
    return [sorted(f) for f in folds]


def _write_one(v: str, fr: int, img_root: Path, regions: np.ndarray, rows,
               img_dir: Path, lbl_dir: Path, mask: bool) -> int:
    """Ghi 1 anh (da to xam vung bo qua) + 1 file nhan YOLO. Tra ve so bbox."""
    src = img_root / v / f"img{fr:05d}.jpg"
    if not src.exists() or rows is None or not len(rows):
        return 0
    im = cv2.imread(str(src))
    if im is None:
        return 0
    H, W = im.shape[:2]
    if mask and len(regions):
        for x, y, w, h in regions:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(W, int(x + w)), min(H, int(y + h))
            if x2 > x1 and y2 > y1:
                im[y1:y2, x1:x2] = config.IGNORE_MASK_COLOR_BGR
    stem = f"{v}_{fr:05d}"
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lines = [f"0 {r.cx / W:.6f} {r.cy / H:.6f} {r.bb_width / W:.6f} {r.bb_height / H:.6f}"
             for r in rows.itertuples() if r.bb_width > 0 and r.bb_height > 0]
    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
    return len(lines)


def export_folds(a) -> int:
    """Xuat anh MOT LAN vao images/all, roi dinh nghia cac fold bang file .txt.

    Ultralytics chap nhan `train:`/`val:` tro toi file danh sach duong dan anh,
    nen khong can nhan ban anh cho tung fold - tiet kiem n lan dung luong dia.
    """
    out = Path(a.out) if a.out else config.DATA_DIR / "yolo_detrac"
    img_root = config.find_images_root()
    if img_root is None:
        print("[ERROR] Khong tim thay thu muc anh.")
        return 1

    pq = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    ign = pd.read_csv(config.INTERIM_DIR / "detrac_ignored_regions.csv")
    videos = sorted(pq.video.unique())
    folds = make_folds(videos, a.n_folds)

    print(f"[export] {len(videos)} video -> {a.n_folds} fold (chia theo VIDEO, phan tang)")
    for i, f in enumerate(folds):
        print(f"  fold {i}: {len(f)} video val | {' '.join(f)}")
    print(f"[export] stride={a.stride}  to xam vung bo qua: {'KHONG' if a.no_mask else 'CO'}")

    ig_by_video = {v: g[["left", "top", "width", "height"]].to_numpy(float)
                   for v, g in ign.groupby("video")} if len(ign) else {}
    ann = {k: g for k, g in pq.groupby(["video", "frame"])}
    img_dir = out / "images" / "all"; lbl_dir = out / "labels" / "all"
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)

    per_video: dict[str, list[str]] = {}
    n_img = n_box = 0
    for v in tqdm(videos, desc="  xuat anh", unit="video"):
        regions = ig_by_video.get(v, np.zeros((0, 4)))
        names = []
        for fr in sorted(pq.loc[pq.video == v, "frame"].unique())[::a.stride]:
            nb = _write_one(v, fr, img_root, regions, ann.get((v, fr)),
                            img_dir, lbl_dir, not a.no_mask)
            if nb or (out / "images" / "all" / f"{v}_{fr:05d}.jpg").exists():
                names.append((img_dir / f"{v}_{fr:05d}.jpg").resolve().as_posix())
                n_img += 1; n_box += nb
        per_video[v] = names

    for i, val_vids in enumerate(folds):
        tr = [p for v, ps in per_video.items() if v not in val_vids for p in ps]
        va = [p for v in val_vids for p in per_video[v]]
        (out / f"fold{i}_train.txt").write_text("\n".join(tr), encoding="utf-8")
        (out / f"fold{i}_val.txt").write_text("\n".join(va), encoding="utf-8")
        (out / f"fold{i}.yaml").write_text(
            f"# Giai doan C - fold {i}/{a.n_folds} (chia theo VIDEO, stride={a.stride})\n"
            f"# val = {len(val_vids)} video detector KHONG duoc nhin thay luc huan luyen\n"
            f"path: {out.resolve().as_posix()}\n"
            f"train: fold{i}_train.txt\nval: fold{i}_val.txt\n\nnc: 1\nnames:\n  0: vehicle\n",
            encoding="utf-8")
        print(f"  fold {i}: {len(tr):,} anh train / {len(va):,} anh val")

    (out / "folds.txt").write_text(
        "\n\n".join(f"fold {i} (val):\n" + "\n".join(f) for i, f in enumerate(folds)),
        encoding="utf-8")
    print(f"\n[export] xong: {n_img:,} anh, {n_box:,} bbox")
    print(f"[export] dung luong: {sum(f.stat().st_size for f in out.rglob('*')) / 1e9:.2f} GB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Xuat UA-DETRAC sang dinh dang YOLO")
    ap.add_argument("--out", default=None, help="Thu muc dich (mac dinh data/yolo_detrac)")
    ap.add_argument("--stride", type=int, default=5, help="Lay 1 frame moi N frame")
    ap.add_argument("--val-videos", nargs="*", default=None)
    ap.add_argument("--n-val", type=int, default=12)
    ap.add_argument("--n-folds", type=int, default=0,
                    help="Neu > 0: xuat anh MOT LAN vao images/all roi tao n fold bang "
                         "file danh sach (tranh nhan n lan dung luong dia).")
    ap.add_argument("--no-mask", action="store_true",
                    help="KHONG to xam vung bo qua (chi de doi chung, khong khuyen nghi)")
    a = ap.parse_args()

    if a.n_folds > 0:
        return export_folds(a)

    out = Path(a.out) if a.out else config.DATA_DIR / "yolo_detrac"
    img_root = config.find_images_root()
    if img_root is None:
        print("[ERROR] Khong tim thay thu muc anh.")
        return 1

    pq = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    ign = pd.read_csv(config.INTERIM_DIR / "detrac_ignored_regions.csv")
    videos = sorted(pq.video.unique())

    val = a.val_videos if a.val_videos else pick_val_videos(a.n_val)
    val = [v for v in val if v in videos]
    train = [v for v in videos if v not in val]
    print(f"[export] {len(train)} video train / {len(val)} video val (chia theo VIDEO)")
    print(f"[export] val: {' '.join(val)}")
    print(f"[export] stride={a.stride}  to xam vung bo qua: {'KHONG' if a.no_mask else 'CO'}")

    ig_by_video = {v: g[["left", "top", "width", "height"]].to_numpy(float)
                   for v, g in ign.groupby("video")} if len(ign) else {}
    ann = {k: g for k, g in pq.groupby(["video", "frame"])}

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_img = n_box = 0
    for split, vids in (("train", train), ("val", val)):
        for v in tqdm(vids, desc=f"  {split}", unit="video"):
            regions = ig_by_video.get(v, np.zeros((0, 4)))
            frames = sorted(pq.loc[pq.video == v, "frame"].unique())[::a.stride]
            for fr in frames:
                src = img_root / v / f"img{fr:05d}.jpg"
                if not src.exists():
                    continue
                rows = ann.get((v, fr))
                if rows is None or not len(rows):
                    continue
                im = cv2.imread(str(src))
                if im is None:
                    continue
                H, W = im.shape[:2]
                if not a.no_mask and len(regions):
                    for x, y, w, h in regions:
                        x1, y1 = max(0, int(x)), max(0, int(y))
                        x2, y2 = min(W, int(x + w)), min(H, int(y + h))
                        if x2 > x1 and y2 > y1:
                            im[y1:y2, x1:x2] = config.IGNORE_MASK_COLOR_BGR
                stem = f"{v}_{fr:05d}"
                cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), im,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                lines = []
                for r in rows.itertuples():
                    cx, cy = r.cx / W, r.cy / H
                    bw, bh = r.bb_width / W, r.bb_height / H
                    if bw <= 0 or bh <= 0:
                        continue
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
                n_img += 1
                n_box += len(lines)

    yaml_txt = (f"# UA-DETRAC cho fine-tune YOLO (Giai doan C)\n"
                f"# Chia theo VIDEO, stride={a.stride}, "
                f"vung bo qua {'KHONG to xam' if a.no_mask else 'da to xam'}\n"
                f"path: {out.resolve().as_posix()}\n"
                f"train: images/train\nval: images/val\n\nnc: 1\nnames:\n  0: vehicle\n")
    (out / "detrac.yaml").write_text(yaml_txt, encoding="utf-8")
    (out / "split_videos.txt").write_text(
        "train:\n" + "\n".join(train) + "\n\nval:\n" + "\n".join(val), encoding="utf-8")

    print(f"\n[export] xong: {n_img:,} anh, {n_box:,} bbox")
    print(f"[export] -> {out / 'detrac.yaml'}")
    print(f"[export] dung luong: {sum(f.stat().st_size for f in out.rglob('*')) / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
