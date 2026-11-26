"""Vehicle-only detector feasibility on preselected UAVDT MOT training frames."""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import det_cache
from r64_ngoai import danh_gia_ngoai as r64
from r90_uavdt.audit import EVIDENCE, GT_PREFIX, SOURCE


def matched(gt: np.ndarray, det: np.ndarray) -> int:
    if not len(gt) or not len(det):
        return 0
    left = np.maximum(gt[:, None, :2], det[None, :, :2])
    right = np.minimum(gt[:, None, 2:4], det[None, :, 2:4])
    wh = np.maximum(0, right - left)
    inter = wh[:, :, 0] * wh[:, :, 1]
    ga = np.prod(gt[:, 2:4] - gt[:, :2], axis=1)
    da = np.prod(det[:, 2:4] - det[:, :2], axis=1)
    union = ga[:, None] + da[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    rows, cols = linear_sum_assignment(np.where(iou >= .5, -iou, 1e6))
    return int(np.sum(iou[rows, cols] >= .5))


def main() -> None:
    selected = json.loads((EVIDENCE / "selection.json").read_text(encoding="utf-8"))["all"]
    images: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    with zipfile.ZipFile(SOURCE / "UAV-benchmark-M.zip") as z, zipfile.ZipFile(
        SOURCE / "UAV-benchmark-MOTD_v1.0.zip"
    ) as labels:
        for seq in selected:
            gt = np.loadtxt(io.BytesIO(labels.read(f"{GT_PREFIX}{seq}_gt.txt")), delimiter=",", ndmin=2)
            n = int(gt[:, 0].max())
            samples = np.linspace(1, n, 10, dtype=int).tolist()
            images[seq] = []
            for frame in samples:
                member = f"UAV-benchmark-M/{seq}/img{frame:06d}.jpg"
                image = cv2.imdecode(np.frombuffer(z.read(member), np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != (540, 1024):
                    raise ValueError((seq, frame, "bad image"))
                rows = gt[(gt[:, 0] == frame) & (gt[:, 6] == 1)]
                truth = np.c_[rows[:, 2:4], rows[:, 2:4] + rows[:, 4:6]].astype(np.float32)
                images[seq].append((frame, image, truth))
    result = {"dataset": "UAVDT official Benchmark-M train; 8 sequence preselection",
              "selection": selected, "sample_rule": "10 evenly spaced frames per sequence",
              "imgsz": 640, "conf": .05, "iou_match": .5, "vehicle_coco_classes": [2, 5, 7],
              "detectors": {}}
    for name in r64.DETECTOR:
        cfg = r64.detector_r54(name)
        checkpoint = ROOT / cfg["checkpoint"]
        assert det_cache.sha256_file(checkpoint) == cfg["sha256"]
        model = YOLO(str(checkpoint))
        mapping = r64.bang_anh_xa_coco(model.names, cfg["loai_checkpoint"])
        spec = det_cache.DetectorSpec(weights=str(checkpoint), weights_sha256=cfg["sha256"],
            imgsz=640, conf=.05, iou=.7, max_det=3000,
            device="0" if torch.cuda.is_available() else "cpu", dataset="uavdt-r90", part="train-smoke")
        scenes = {}
        for seq in selected:
            gt_n = hit_n = cap_n = 0
            for frame, image, truth in images[seq]:
                det = r64.doi_id_sang_coco(det_cache.detect_frame(model, image, spec), mapping)
                cap_n += len(det) >= 3000
                vehicle = det[np.isin(det[:, 5], [2, 5, 7])]
                gt_n += len(truth)
                hit_n += matched(truth, vehicle[:, :4])
            scenes[seq] = {"gt": gt_n, "matched": hit_n, "recall": hit_n / gt_n if gt_n else None,
                           "frames_at_detection_cap": cap_n}
            print(name, seq, f"{hit_n}/{gt_n}", flush=True)
        total = sum(x["gt"] for x in scenes.values())
        hit = sum(x["matched"] for x in scenes.values())
        result["detectors"][name] = {"checkpoint": str(checkpoint), "sha256": cfg["sha256"],
            "gt": total, "matched": hit, "recall": hit / total,
            "scenes_at_least_15pct": sum(x["recall"] >= .15 for x in scenes.values()),
            "scenes": scenes}
        print("TOTAL", name, hit, total, hit / total, flush=True)
        (EVIDENCE / "detector_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
