"""TrafficMOT train27 detection feasibility check; no tracking or test claim.

Sample frame0/frame30/frame58 from each fully annotated training clip.
Evaluate only classes with direct COCO equivalents: Bike, Bus, Motor_Bike,
Truck. LMV, Auto, LCV, E-rickshaw, Tractor, Pedestrian remain outside scope.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "r64_ngoai")]
import danh_gia_ngoai as r64  # noqa: E402
import det_cache  # noqa: E402

DATA = Path("F:/LABOTARY_data/trafficmot/TrafficMOT")
OUT = ROOT / "handoff/logs/external_eval/r66/trafficmot/smoke_det_train27.json"
SOURCE_TO_COCO = {1: 3, 2: 5, 5: 1, 10: 7}
FRAMES = (0, 30, 58)
THRESHOLDS = (0.05, 0.10, 0.25)


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum((a[:, 0] + a[:, 2])[:, None], (b[:, 0] + b[:, 2])[None, :])
    y1 = np.minimum((a[:, 1] + a[:, 3])[:, None], (b[:, 1] + b[:, 3])[None, :])
    inter = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
    return inter / (a[:, 2:4].prod(1)[:, None] + b[:, 2:4].prod(1)[None, :] - inter)


def main() -> None:
    from ultralytics import YOLO

    names = sorted(p.name for p in (DATA / "ground_truth/Fully_annotate").iterdir() if p.is_dir())
    if len(names) != 27:
        raise RuntimeError(f"Expected 27 fully annotated train clips, got {len(names)}")
    sample = []
    for clip in names:
        for frame in FRAMES:
            label = DATA / "ground_truth/Fully_annotate" / clip / f"frame{frame}.csv"
            image = DATA / "image/Fully_annotate" / clip / f"frame{frame}.jpg"
            if not label.is_file() or not image.is_file():
                raise FileNotFoundError(f"{label} / {image}")
            with label.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            sample.append((image, rows))

    result = {"scope": "TrafficMOT Fully_annotate train27; 3 fixed frames/clip; detection only",
              "frames": list(FRAMES), "source_to_coco": SOURCE_TO_COCO,
              "n_clips": len(names), "n_frames": len(sample), "detectors": {}}
    for detector in ("yolov8n_coco", "yolov8n_mixed5", "yolo11n_mixed5"):
        config = r64.detector_r54(detector)
        checkpoint = ROOT / config["checkpoint"]
        if r64.sha256_file(checkpoint) != config["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {detector}")
        spec = det_cache.DetectorSpec(
            backend="ultralytics", weights=str(checkpoint), weights_sha256=config["sha256"],
            imgsz=640, conf=0.05, iou=0.7, classes=None, max_det=3000, half=False,
            agnostic_nms=False, augment=False, batch=1, device="0",
            mask_ignored_regions=False, dataset="TrafficMOT", part="train27-smoke")
        model = YOLO(str(checkpoint))
        mapping = r64.bang_anh_xa_coco(model.names, config["loai_checkpoint"])
        gt_count = Counter()
        tp = {t: Counter() for t in THRESHOLDS}
        frame_cap = 0
        for path, rows in sample:
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Cannot decode {path}")
            det = r64.doi_id_sang_coco(det_cache.detect_frame(model, image, spec), mapping)
            frame_cap += int(len(det) >= 3000)
            for source_cls, coco_cls in SOURCE_TO_COCO.items():
                selected = [r for r in rows if int(r["classID"]) == source_cls]
                gt = np.array([[float(r[k]) for k in ("x", "y", "w", "h")] for r in selected],
                              dtype=float).reshape(-1, 4)
                gt_count[source_cls] += len(gt)
                for threshold in THRESHOLDS:
                    dd = det[(det[:, 4] > threshold) & (det[:, 5] == coco_cls)]
                    pred = np.c_[dd[:, 0], dd[:, 1], dd[:, 2] - dd[:, 0], dd[:, 3] - dd[:, 1]]
                    m = iou_xywh(gt, pred)
                    if m.size:
                        ri, ci = linear_sum_assignment(np.where(m >= 0.5, -m, 1e6))
                        tp[threshold][source_cls] += int(np.count_nonzero(m[ri, ci] >= 0.5))
        total = sum(gt_count.values())
        result["detectors"][detector] = {
            "gt_by_class": dict(gt_count), "gt_total": total, "frames_at_max_det": frame_cap,
            "tp_by_threshold_and_class": {str(t): dict(tp[t]) for t in THRESHOLDS},
            "recall_by_threshold": {str(t): sum(tp[t].values()) / total for t in THRESHOLDS},
            "recall_by_threshold_and_class": {
                str(t): {str(c): tp[t][c] / n for c, n in gt_count.items()} for t in THRESHOLDS}}
        print(detector, result["detectors"][detector]["recall_by_threshold"], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
