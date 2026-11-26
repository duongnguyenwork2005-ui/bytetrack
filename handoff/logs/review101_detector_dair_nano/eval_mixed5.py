"""Measure all five classes on the existing mixed5 development split."""
import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("models", nargs="+", help="model_name=checkpoint_path")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, model_path in (spec.split("=", 1) for spec in args.models):
        path = Path(model_path)
        model = YOLO(str(path), task="detect")
        metrics = model.val(data="data/yolo_mixed5_v1/data.yaml", split="val", imgsz=640,
                            batch=16, workers=2, device=0, conf=.001, iou=.7, max_det=3000, plots=False,
                            verbose=False, project=str(args.out.parent / "val_runs"), name=name)
        box = metrics.box
        classes = {model.names[int(class_id)]: {"ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i]),
                                                "precision": float(box.p[i]), "recall": float(box.r[i])}
                   for i, class_id in enumerate(box.ap_class_index)}
        assert set(classes) == {"car", "truck", "bus", "motorcycle", "bicycle"}
        rows[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "mAP50": float(box.map50), "mAP50_95": float(box.map), "classes": classes}
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps({name: rows[name]}), flush=True)


if __name__ == "__main__":
    main()
