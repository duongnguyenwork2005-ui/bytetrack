"""Compare portable ONNX artifacts on the same mixed5 development images."""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = {}
    for model_path in map(Path, args.models):
        try:
            model = YOLO(str(model_path), task="detect")
            metrics = model.val(data=str(args.data.resolve()), split="val", imgsz=640, rect=False,
                                batch=1, workers=0, device="cpu", conf=.001, iou=.7, max_det=3000,
                                plots=False, verbose=False, project=str(args.out.parent / "onnx_val_runs"),
                                name=model_path.stem)
            box = metrics.box
            classes = {model.names[int(class_id)]: float(box.ap50[i]) for i, class_id in enumerate(box.ap_class_index)}
            assert set(classes) == {"car", "truck", "bus", "motorcycle", "bicycle"}
            result = {"mAP50": float(box.map50), "mAP50_95": float(box.map), "ap50": classes,
                      "speed_ms": metrics.speed, "bytes": model_path.stat().st_size}
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        results[model_path.stem] = result
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps({model_path.stem: result}), flush=True)


if __name__ == "__main__":
    main()
