"""Run isolated two-epoch class-balanced detector pilot and paired dev AP.

This never overwrites the locked mixed5 checkpoint or dataset definition.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "handoff/logs/review91_pf_assoc/balanced_pilot"
BASE = ROOT / "models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt"
DATA = OUT / "data.yaml"
RUNS = OUT / "runs"
NAMES = ("car", "truck", "bus", "motorcycle", "bicycle")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def report_metrics(model_path, name):
    model = YOLO(str(model_path))
    metrics = model.val(data=str(DATA), split="val", imgsz=640, batch=8, device=0,
                        workers=2, half=False, conf=.001, iou=.7, max_det=3000,
                        plots=False, save_json=False, project=str(RUNS), name="val_" + name,
                        exist_ok=True, verbose=False)
    per_class = {}
    for i, label in enumerate(NAMES):
        per_class[label] = {"AP50": float(metrics.box.ap50[i]),
                            "AP50_95": float(metrics.box.maps[i]),
                            "precision": float(metrics.box.p[i]),
                            "recall": float(metrics.box.r[i])}
    result = {"checkpoint": str(model_path), "sha256": digest(model_path),
              "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
              "per_class": per_class}
    (OUT / (name + "_dev_metrics.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("DEV", name, json.dumps(result), flush=True)


def main():
    report_metrics(BASE, "baseline")
    model = YOLO(str(BASE))
    model.train(data=str(DATA), epochs=2, imgsz=640, batch=8, device=0,
                workers=2, amp=False, cache=False, optimizer="SGD", lr0=.001,
                lrf=.1, warmup_epochs=.3, seed=9106, deterministic=True,
                patience=2, close_mosaic=0, plots=False, save=True,
                project=str(RUNS), name="train", exist_ok=True)
    candidate = RUNS / "train/weights/best.pt"
    report_metrics(candidate, "candidate")


if __name__ == "__main__":
    main()
