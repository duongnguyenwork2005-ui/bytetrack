"""Pilot mixed 4/8-bit weight-only quantization-aware training on labeled train images."""
import argparse
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO
import yaml

from fit_channels import PrunedTrainer, prepare_data
from prepare import ANCHOR


def mark_convolutions(model):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            layer.r104_qat_limit = 127 if name.startswith("model.23.") or layer.weight.numel() < 36_000 else 7


def fit(model, work, config):
    original = nn.Conv2d._conv_forward

    def fake_quant_forward(layer, x, weight, bias):
        limit = getattr(layer, "r104_qat_limit", None)
        if limit is not None:
            with torch.no_grad():
                scale = (weight.abs().amax(dim=(1, 2, 3), keepdim=True) / limit).clamp_min(1e-12)
                rounded = (weight / scale).round().clamp(-limit, limit) * scale
                delta = rounded - weight
            weight = weight + delta
        return original(layer, x, weight, bias)

    nn.Conv2d._conv_forward = fake_quant_forward
    try:
        model.train(trainer=PrunedTrainer, data=str(config), epochs=1, imgsz=384, batch=8, workers=0,
                    optimizer="SGD", lr0=.002, warmup_epochs=.25, mosaic=0, mixup=0, amp=False, device=0,
                    seed=104, deterministic=True, patience=0, plots=False, project=str(work / "runs"), name="qat",
                    exist_ok=False, verbose=False)
    finally:
        nn.Conv2d._conv_forward = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    folder = work / "qat_data"
    folder.mkdir(exist_ok=True)
    config, _ = prepare_data(folder)
    manifest = folder / "train.txt"
    paths = manifest.read_text().splitlines()
    chosen = paths[:320] + paths[-80:]
    manifest.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    assert len(set(chosen)) == 400 and yaml.safe_load(config.read_text())["train"] == manifest.as_posix()
    model = YOLO(str(ANCHOR), task="detect")
    mark_convolutions(model.model)
    fit(model, work, config)
    assert (work / "runs/qat/weights/best.pt").is_file()


if __name__ == "__main__":
    main()
