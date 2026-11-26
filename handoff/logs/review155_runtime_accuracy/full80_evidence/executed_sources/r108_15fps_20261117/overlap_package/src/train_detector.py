"""
train_detector.py -- Fine-tune detector ung vien tren dataset lop thong nhat (Giai doan 2). Boc YOLO.train
voi log day du. KHONG tu chay dai: --dry-run (1 epoch, fraction 0.02) chi de kiem tra pipeline.

    python src/train_detector.py --weights models/candidates/yolo11n.pt --data configs/phase2/detector/mixed5.yaml --dry-run
    python src/train_detector.py --weights models/candidates/yolo11n.pt --data configs/phase2/detector/mixed5.yaml \
        --epochs 50 --imgsz 640 --batch 8 --name mixed5_yolo11n          # may co GPU phu hop (GTX 1650 4 GB: batch 8)
Ket qua: models/finetune_phase2/<name>/ (weights/best.pt, results.csv, args.yaml) + results/phase2/detector/<name>.json
Ten thiet lap khi bao cao: 'coco-pretrained -> <tap dich>' (chuyen mien) vs 'finetune(<data>) -> <val giu lai>'.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(config.MODELS_DIR / "candidates" / "yolo11n.pt"))
    ap.add_argument("--data", default=str(config.CONFIGS_DIR / "phase2" / "detector" / "mixed5.yaml"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default=None)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true", help="1 epoch, fraction 0.02, kiem tra pipeline")
    a = ap.parse_args()
    from ultralytics import YOLO

    name = a.name or f"{Path(a.weights).stem}_{Path(a.data).stem}" + ("_dry" if a.dry_run else "")
    epochs, fraction = (1, 0.02) if a.dry_run else (a.epochs, a.fraction)
    project = config.MODELS_DIR / "finetune_phase2"
    t0 = time.time()
    model = YOLO(a.weights)
    res = model.train(data=a.data, epochs=epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                      project=str(project), name=name, exist_ok=True, fraction=fraction, seed=0,
                      deterministic=True, plots=False, verbose=False)
    out = {"weights": a.weights, "data": a.data, "epochs": epochs, "fraction": fraction, "imgsz": a.imgsz,
           "batch": a.batch, "seconds": round(time.time() - t0, 1), "save_dir": str(res.save_dir),
           "metrics": {k: float(v) for k, v in getattr(res, "results_dict", {}).items()}, "dry_run": a.dry_run}
    d = config.RESULTS_DIR / "phase2" / "detector"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
