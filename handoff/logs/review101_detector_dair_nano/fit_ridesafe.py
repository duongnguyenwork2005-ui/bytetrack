"""Prepare and run matched detector fine-tuning pilots with and without RideSafe."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from ultralytics import YOLO
import yaml

LOGS = Path(__file__).resolve().parents[1]
ROOT = LOGS.parents[1]
sys.path.insert(0, str(LOGS / "review100_nano_pf_dataset"))
from pilot_ridesafe import CHECKPOINTS, read_annotations  # noqa: E402

MIXED = ROOT / "data/yolo_mixed5_v1"
COCO_TO_MIXED = {1: 4, 2: 0, 5: 2, 7: 1}
NAMES = ["car", "truck", "bus", "motorcycle", "bicycle"]
SEED = 101


def yolo_line(category, box, shape):
    height, width = shape
    x1, y1, x2, y2 = map(float, box)
    assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
    coords = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)
    return f"{category} " + " ".join(f"{value:.9f}" for value in coords)


def export_clip(folder, manifest, output, teacher):
    gt, stride, shape, _ = read_annotations(folder / manifest["labels"]["file"])
    available = [frame for frame, objects in gt.items() if objects]
    assert len(available) >= 100 and stride == 5
    selected = {available[i] for i in np.linspace(0, len(available) - 1, 100, dtype=int)}
    images = output / "images"
    labels = output / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    counts, paths, batch = Counter(), [], []
    capture = cv2.VideoCapture(str(folder / manifest["video"]["file"]), cv2.CAP_FFMPEG)
    assert capture.isOpened() and capture.get(cv2.CAP_PROP_FPS) > 0

    def write_batch():
        results = teacher.predict([image for _, image in batch], imgsz=640, conf=.35, iou=.7,
                                  classes=list(COCO_TO_MIXED), max_det=300, device=0, verbose=False)
        for (frame, image), result in zip(batch, results, strict=True):
            basename = f"{manifest['clip']}_{frame:04d}"
            path = images / f"{basename}.jpg"
            assert cv2.imwrite(str(path), image)
            lines = [yolo_line(3, box, shape) for box in gt[frame].values()]
            counts["gt_motorcycle"] += len(lines)
            for x1, y1, x2, y2, _, coco in result.boxes.data.cpu().numpy():
                category = COCO_TO_MIXED[int(coco)]
                if x2 > x1 and y2 > y1:
                    lines.append(yolo_line(category, (x1, y1, x2, y2), shape))
                    counts["pseudo_" + NAMES[category]] += 1
            (labels / f"{basename}.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
            paths.append(path)
        batch.clear()

    decoded = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if decoded in selected:
                assert image.shape[:2] == shape
                batch.append((decoded, image))
                if len(batch) == 8:
                    write_batch()
            decoded += 1
        if batch:
            write_batch()
        assert decoded == int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) and len(paths) == 100
    finally:
        capture.release()
    return paths, dict(counts)


def ordered(paths):
    return sorted(paths, key=lambda path: hashlib.sha256(f"{SEED}|{path.name}".encode()).hexdigest())


def make_split(sources, output):
    base_manifest = LOGS / "review91_pf_assoc/balanced_pilot/train.txt"
    pool = [Path(line) for line in base_manifest.read_text(encoding="utf-8").splitlines() if line]
    pool = ordered(pool)[:800]
    assert len(set(pool)) == 800 and all(p.parent == MIXED / "images/train" and p.is_file() for p in pool)
    with (MIXED / "manifest/anh.csv").open(encoding="utf-8-sig", newline="") as file:
        dev = ordered([MIXED / row["anh_ra"] for row in csv.DictReader(file) if row["split"] == "dev"])
    assert len(dev) == 7190 and all(p.is_file() for p in dev)
    source_records = json.loads((sources / "train_sources.json").read_text(encoding="utf-8"))
    assert {r["clip"][:8] for r in source_records}.isdisjoint({"20211109", "20211125"})
    teacher = YOLO(str(ROOT / CHECKPOINTS["yolov8n_coco"]))
    new, labels = [], {}
    for record in source_records:
        clip = record["clip"]
        paths, counts = export_clip(sources / clip, record, output / "ridesafe", teacher)
        new.extend(paths)
        labels[clip] = counts
    assert len(new) == 200 and len(set(new)) == 200
    variants = {"control": pool, "ridesafe": pool[:600] + new}
    configs = {}
    for name, paths in variants.items():
        train = output / f"{name}_train.txt"
        val = output / "monitor_dev300.txt"
        train.write_text("\n".join(p.as_posix() for p in paths) + "\n", encoding="utf-8")
        val.write_text("\n".join(p.as_posix() for p in dev[:300]) + "\n", encoding="utf-8")
        config = output / f"{name}.yaml"
        config.write_text(yaml.safe_dump({"path": output.as_posix(), "train": train.as_posix(),
                                          "val": val.as_posix(), "names": dict(enumerate(NAMES))}), encoding="utf-8")
        configs[name] = config
    metadata = {"seed": SEED, "shared_mixed5": 600, "train_images_per_arm": 800,
                "ridesafe_labels": labels, "monitor_dev": 300, "full_dev": len(dev)}
    (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.train:
        make_split(args.sources.resolve(), output)
        return
    random.seed(SEED)
    original = ROOT / CHECKPOINTS["yolo11n_mixed5"]
    for name in ("control", "ridesafe"):
        model = YOLO(str(original), task="detect")
        model.train(data=str(output / f"{name}.yaml"), epochs=1, imgsz=640, batch=8, workers=0,
                    optimizer="SGD", lr0=.001, warmup_epochs=0, mosaic=0, mixup=0, amp=False,
                    device=0, seed=SEED, deterministic=True, patience=0, plots=False,
                    project=str(output / "runs"), name=name, exist_ok=False, verbose=False)
        assert (output / "runs" / name / "weights/last.pt").is_file()


if __name__ == "__main__":
    main()
