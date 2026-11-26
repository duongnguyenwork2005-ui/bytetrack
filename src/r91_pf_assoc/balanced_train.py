"""Prepare a small, isolated five-class fine-tuning pilot from train manifest.

No images or labels are changed. Sampling uses train rows only and prefers
motorcycle-positive images while retaining the other four classes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/yolo_mixed5_v1"
OUT = ROOT / "handoff/logs/review91_pf_assoc/balanced_pilot"
SEED = 9106


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (DATA / "manifest/anh.csv").open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == "train"]
    assert len(rows) == 57043
    rng = random.Random(SEED)
    selected = {}
    for cls, target in (("motorcycle", 1822), ("bicycle", 850), ("bus", 850), ("truck", 850)):
        candidates = [r for r in rows if int(r["n_" + cls]) > 0 and r["anh_ra"] not in selected]
        rng.shuffle(candidates)
        for row in candidates[:target]:
            selected[row["anh_ra"]] = row
    candidates = [r for r in rows if r["anh_ra"] not in selected]
    rng.shuffle(candidates)
    for row in candidates[:5000 - len(selected)]:
        selected[row["anh_ra"]] = row
    assert len(selected) == 5000
    names = sorted(selected)
    paths = [str((DATA / name).resolve()).replace("\\", "/") for name in names]
    assert all(Path(path).is_file() for path in paths)
    train = OUT / "train.txt"
    train.write_text("\n".join(paths) + "\n", encoding="utf-8")
    dataset = OUT / "data.yaml"
    dataset.write_text("\n".join([
        f"path: {DATA.as_posix()}", f"train: {train.as_posix()}",
        "val: images/dev", "nc: 5", "names: [car, truck, bus, motorcycle, bicycle]", ""
    ]), encoding="utf-8")
    report = {"seed": SEED, "source_manifest": "data/yolo_mixed5_v1/manifest/anh.csv",
              "train_images": len(names), "train_txt_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
              "class_images": {cls: sum(int(r["n_" + cls]) > 0 for r in selected.values())
                               for cls in ("car", "truck", "bus", "motorcycle", "bicycle")},
              "class_boxes": {cls: sum(int(r["n_" + cls]) for r in selected.values())
                              for cls in ("car", "truck", "bus", "motorcycle", "bicycle")},
              "checkpoint": "models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt",
              "purpose": "isolated low-cost class-balanced training pilot; no final detector selection"}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
