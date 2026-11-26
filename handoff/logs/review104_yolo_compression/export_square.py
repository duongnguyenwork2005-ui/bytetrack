"""Export multiple checkpoints at the fixed shape used for ONNX development validation."""
import argparse
from pathlib import Path
from shutil import copyfile

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoints", nargs="+", help="name=checkpoint_path")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    for spec in args.checkpoints:
        name, original = spec.split("=", 1)
        checkpoint = args.output / f"{name}.pt"
        copyfile(original, checkpoint)
        YOLO(str(checkpoint), task="detect").export(format="onnx", imgsz=640, opset=13, simplify=False,
                                                   dynamic=False, device="cpu", batch=1, verbose=False)


if __name__ == "__main__":
    main()
