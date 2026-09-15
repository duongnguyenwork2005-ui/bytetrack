"""
prepare_mixed_yolo_dataset.py -- Xuat dataset YOLO 5 lop {car, truck, bus, motorcycle, bicycle} tu
UA-DETRAC (train) + BDD100K box_track_20 (train, khi co) de fine-tune detector ung vien (Giai doan 2).

Nguyen tac:
    - Chia theo VIDEO (khong theo frame); UA-DETRAC test va BDD val KHONG BAO GIO duoc dua vao.
    - Lay thua frame (--stride) vi frame lien ke gan trung nhau.
    - UA-DETRAC: to xam ignored_region nhu luc suy dien (export_yolo_dataset.py); BDD: to xam box crowd.
    - Anh xa lop (configs/phase2/detector/mixed5.yaml): DETRAC car/van->car, others->truck, bus->bus
      (GIA DINH); --detrac-classes four-wheel-only bo 'others'; BDD 5 lop giu nguyen.
    - Ghi manifest (video train/val, so nhan theo lop, tham so) de tai lap.

    python src/prepare_mixed_yolo_dataset.py --dry-run                        # chi thong ke, khong ghi anh
    python src/prepare_mixed_yolo_dataset.py --detrac-stride 5 --detrac-n-val 12 --out data/yolo_mixed5
    python src/prepare_mixed_yolo_dataset.py --bdd-root data/raw/bdd100k --bdd-stride 2   # khi co BDD train
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from export_yolo_dataset import pick_val_videos  # noqa: E402

CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]
CID = {c: i for i, c in enumerate(CLASSES)}
DETRAC_MAP = {"car": "car", "van": "car", "others": "truck", "bus": "bus"}       # GIA DINH (xem yaml)
DETRAC_MAP_4W = {"car": "car", "van": "car", "bus": "bus"}                        # four-wheel-only: bo 'others'


def write_sample(img_path: Path, boxes, masks, out_img: Path, out_lbl: Path, dry: bool) -> int:
    """boxes: (cid, left, top, w, h) px; masks: (x, y, w, h) px to xam. Tra ve so nhan."""
    if dry:
        return len(boxes)
    im = cv2.imread(str(img_path))
    if im is None:
        return 0
    H, W = im.shape[:2]
    for x, y, w, h in masks:
        x1, y1, x2, y2 = max(0, int(x)), max(0, int(y)), min(W, int(x + w)), min(H, int(y + h))
        if x2 > x1 and y2 > y1:
            im[y1:y2, x1:x2] = config.IGNORE_MASK_COLOR_BGR
    lines = [f"{c} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
             for c, x, y, w, h in boxes if w > 0 and h > 0]
    cv2.imwrite(str(out_img), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    out_lbl.write_text("\n".join(lines))
    return len(lines)


def export_detrac(a, out: Path, stats: Counter, lists: dict, dry: bool) -> dict:
    from mot_datasets.detrac import DetracDataset
    ds = DetracDataset()
    df = ds.gt_dataframe("train")
    videos = sorted(df.video.unique())
    val = a.detrac_val_videos or pick_val_videos(a.detrac_n_val, a.seed)
    train = [v for v in videos if v not in val]
    cmap = DETRAC_MAP if a.detrac_classes == "mapped" else DETRAC_MAP_4W
    n_img = 0
    for split, vids in (("train", train), ("val", val)):
        for v in vids:
            g = df[df.video == v]
            regions = ds.ignored_regions(v, "train")
            frames = sorted(g.frame.unique())[::a.detrac_stride]
            for fr in frames:
                rows = g[g.frame == fr]
                boxes = []
                for r in rows.itertuples():
                    cat = cmap.get(str(r.vehicle_type))
                    if cat is None:
                        continue
                    boxes.append((CID[cat], float(r.bb_left), float(r.bb_top), float(r.bb_width), float(r.bb_height)))
                    stats[f"detrac_{split}_{cat}"] += 1
                if not boxes:
                    continue
                src = ds.img_root / v / f"img{fr:05d}.jpg"
                stem = f"detrac_{v}_{fr:05d}"
                n = write_sample(src, boxes, regions, out / "images" / split / f"{stem}.jpg",
                                 out / "labels" / split / f"{stem}.txt", dry)
                if n:
                    lists[split].append(f"images/{split}/{stem}.jpg")
                    n_img += 1
    return {"detrac_train_videos": train, "detrac_val_videos": val, "n_images": n_img, "class_map": cmap}


def export_bdd(a, out: Path, stats: Counter, lists: dict, dry: bool) -> dict:
    from mot_datasets.bdd100k_mot import BDD100KMOTDataset, parse_video
    ds = BDD100KMOTDataset(a.bdd_root)
    vids = json.loads(Path(a.bdd_manifest).read_text())["videos"] if a.bdd_manifest else ds.list_videos("train")
    if not vids:
        return {"note": "khong co video BDD train tren may"}
    rng = np.random.default_rng(a.seed)
    vids = sorted(vids)
    val = sorted(rng.choice(vids, size=max(1, int(0.1 * len(vids))), replace=False).tolist())
    train = [v for v in vids if v not in val]
    keep = {"motorcycle", "bicycle"} if a.bdd_only_two_wheel else set(CLASSES)
    n_img = 0
    for split, vv in (("train", train), ("val", val)):
        for v in vv:
            g = parse_video(ds.labels_dir("train") / f"{v}.json", "train")
            fr_map = {f.index: f for f in ds.frames(v, "train")}
            frames = sorted(g.frame.unique())[::a.bdd_stride]
            for fr in frames:
                rows = g[g.frame == fr]
                masks = [(r.bb_left, r.bb_top, r.bb_width, r.bb_height) for r in rows[rows.crowd].itertuples()]
                boxes = []
                for r in rows[~rows.crowd].itertuples():
                    if r.category in keep:
                        boxes.append((CID[r.category], float(r.bb_left), float(r.bb_top), float(r.bb_width), float(r.bb_height)))
                        stats[f"bdd_{split}_{r.category}"] += 1
                if not boxes or fr_map[fr].path is None:
                    continue
                stem = f"bdd_{v}_{fr:07d}"
                n = write_sample(fr_map[fr].path, boxes, masks, out / "images" / split / f"{stem}.jpg",
                                 out / "labels" / split / f"{stem}.txt", dry)
                if n:
                    lists[split].append(f"images/{split}/{stem}.jpg")
                    n_img += 1
    return {"bdd_train_videos": train, "bdd_val_videos": val, "n_images": n_img,
            "classes_kept": sorted(keep)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.DATA_DIR / "yolo_mixed5"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-detrac", action="store_true")
    ap.add_argument("--detrac-stride", type=int, default=5)
    ap.add_argument("--detrac-n-val", type=int, default=12)
    ap.add_argument("--detrac-val-videos", nargs="*", default=None)
    ap.add_argument("--detrac-classes", choices=["mapped", "four-wheel-only"], default="mapped")
    ap.add_argument("--bdd-root", default=None)
    ap.add_argument("--bdd-manifest", default=None)
    ap.add_argument("--bdd-stride", type=int, default=2)
    ap.add_argument("--bdd-only-two-wheel", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if not a.dry_run:
        for s in ("train", "val"):
            (out / "images" / s).mkdir(parents=True, exist_ok=True)
            (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    lists = {"train": [], "val": []}
    manifest = {"classes": CLASSES, "args": vars(a)}
    if not a.no_detrac:
        manifest["detrac"] = export_detrac(a, out, stats, lists, a.dry_run)
    bdd_train = (Path(a.bdd_root) if a.bdd_root else config.RAW_DIR / "bdd100k") / "labels" / "box_track_20" / "train"
    if bdd_train.is_dir():
        manifest["bdd100k"] = export_bdd(a, out, stats, lists, a.dry_run)
    else:
        manifest["bdd100k"] = {"note": f"chua co {bdd_train} tren may - bo qua (xem PHASE2_REPORT.md)"}
    manifest["label_counts"] = dict(sorted(stats.items()))
    manifest["n_train_images"], manifest["n_val_images"] = len(lists["train"]), len(lists["val"])
    if not a.dry_run:
        (out / "train.txt").write_text("\n".join(lists["train"]))
        (out / "val.txt").write_text("\n".join(lists["val"]))
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "args"}, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
