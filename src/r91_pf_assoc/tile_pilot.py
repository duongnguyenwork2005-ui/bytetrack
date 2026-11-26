"""Fixed six-frame UTUAV detector pilot, native frame vs 960px overlapping tiles.

This is a recall diagnostic, not an AP/precision or tracker benchmark. Static
objects have separate source classes and are excluded from dynamic GT.
"""
from __future__ import annotations

import json
import argparse
import re
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from r67_utuav.prepare import IMAGES, LABEL_ARCHIVE, RANGES  # noqa: E402

FRAMES = {"B_validation": (1, 369, 738), "C_validation": (1, 376, 751)}
MODELS = {
    "yolov8n_coco": ROOT / "models/yolov8n.pt",
    "yolo11n_mixed5": ROOT / "models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt",
}
CLASSES = ("car", "bus", "motorcycle")
OUT = ROOT / "handoff/logs/review91_pf_assoc/tile_pilot.json"


def source_gt():
    wanted = {(scene, first + frame - 1): (name, frame, prefix)
              for name, (scene, first, _, prefix) in RANGES.items() for frame in FRAMES[name]}
    pattern = re.compile(r"([BC])_6class_PascalVOC/Annotations/(v[12])_(\d+)\.xml")
    found = {}
    with tarfile.open(LABEL_ARCHIVE, "r|gz") as tar:
        for member in tar:
            match = pattern.fullmatch(member.name)
            if match is None or not member.isfile():
                continue
            key = (match[1], int(match[3]))
            if key not in wanted:
                continue
            name, frame, prefix = wanted[key]
            if match[2] != prefix:
                raise ValueError(member.name)
            root = ET.fromstring(tar.extractfile(member).read())
            rows = defaultdict(list)
            for obj in root.findall("object"):
                cls = obj.findtext("name")
                if cls not in ("car", "bus", "motorbike"):
                    continue
                box = obj.find("bndbox")
                rows[{"motorbike": "motorcycle"}.get(cls, cls)].append(
                    [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
            found[(name, frame)] = rows
    if set(found) != {(name, frame) for name, frames in FRAMES.items() for frame in frames}:
        raise RuntimeError("incomplete fixed XML selection")
    return found


def origins(length, tile=960, stride=800):
    return sorted(set(min(i, length - tile) for i in range(0, length, stride)))


def iou(a, b):
    if not len(a) or not len(b):
        return np.empty((len(a), len(b)))
    aa, bb = np.asarray(a), np.asarray(b)
    lo = np.maximum(aa[:, None, :2], bb[None, :, :2])
    hi = np.minimum(aa[:, None, 2:], bb[None, :, 2:])
    wh = np.maximum(0, hi - lo)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.prod(aa[:, 2:] - aa[:, :2], axis=1)
    area_b = np.prod(bb[:, 2:] - bb[:, :2], axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


def nms(rows):
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r[4], reverse=True)
    keep = []
    for row in rows:
        if not keep or np.max(iou([row[:4]], [r[:4] for r in keep])) <= .7:
            keep.append(row)
    return keep


def matches(gt, det):
    ov = iou(gt, [row[:4] for row in det])
    if not ov.size:
        return 0
    ri, ci = linear_sum_assignment(1 - ov)
    return int(sum(ov[r, c] >= .5 for r, c in zip(ri, ci)))


def infer(model, image, tiled, tile_size=960):
    w, h = image.size
    stride = tile_size - 80 if tile_size == 640 else 800
    crops = [(0, 0, w, h)] if not tiled else [(x, y, x + tile_size, y + tile_size)
            for y in origins(h, tile_size, stride) for x in origins(w, tile_size, stride)]
    collected = defaultdict(list)
    start = time.perf_counter()
    for left, top, right, bottom in crops:
        crop = np.asarray(image.crop((left, top, right, bottom)))
        result = model.predict(crop, imgsz=640, conf=.05, iou=.7, max_det=3000,
                               device=0, half=False, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        ids = boxes.cls.cpu().numpy().astype(int)
        for box, score, cls in zip(xyxy, scores, ids):
            label = str(model.names[int(cls)]).lower()
            label = "motorcycle" if label == "motorbike" else label
            if label in CLASSES:
                collected[label].append([float(box[0] + left), float(box[1] + top),
                                         float(box[2] + left), float(box[3] + top), float(score)])
    duration = time.perf_counter() - start
    return {cls: nms(collected[cls]) for cls in CLASSES}, duration, len(crops)


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", type=int, choices=(640, 960), default=960)
    args = parser.parse_args()
    out_path = OUT if args.tile == 960 else OUT.with_name("tile_pilot_640.json")

    gt = source_gt()
    output = {"protocol": {"frames": FRAMES, "native_size": [3840, 2160],
              "tile": args.tile, "stride": args.tile - 80 if args.tile == 640 else 800, "imgsz": 640, "conf": .05,
              "nms_iou": .7, "match_iou": .5,
              "metric": "one-to-one class-aware dynamic GT recall; static GT excluded; precision unavailable"},
              "rows": []}
    for model_name, checkpoint in MODELS.items():
        model = YOLO(str(checkpoint))
        for name, frames in FRAMES.items():
            for frame in frames:
                image_path = IMAGES / name / f"{frame:04d}.jpg"
                with Image.open(image_path) as image:
                    for tiled in (False, True):
                        det, seconds, crops = infer(model, image, tiled, args.tile)
                        any_vehicle = [box for label in CLASSES for box in det[label]]
                        for cls in CLASSES:
                            target = gt[(name, frame)][cls]
                            output["rows"].append({"model": model_name, "video": name,
                                "frame": frame, "mode": f"tiles{args.tile}" if tiled else "native_resize",
                                "class": cls, "gt": len(target), "matched": matches(target, det[cls]),
                                "any_vehicle_matched": matches(target, any_vehicle),
                                "detections": len(det[cls]), "seconds_image": seconds,
                                "crops": crops})
                        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
                print(model_name, name, frame, "tiles" if tiled else "native", round(seconds, 2), flush=True)
    print("WROTE", out_path, flush=True)


if __name__ == "__main__":
    main()
