"""Test low-rank decomposition of the largest 3x3 convolutions."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv

from prepare import ANCHOR, check_model, metadata, write_checkpoint


def factorize(model):
    targets = [(path, layer) for path, layer in model.named_modules()
               if isinstance(layer, Conv) and layer.conv.kernel_size == (3, 3)
               and layer.conv.groups == 1 and layer.conv.weight.numel() >= 36_000]
    report = []
    for path, layer in targets:
        original = layer.conv
        channels_in, channels_out = original.in_channels, original.out_channels
        rank = (channels_out * 3 // 4) // 8 * 8
        u, s, vh = torch.linalg.svd(original.weight.detach().flatten(1), full_matrices=False)
        spatial = nn.Conv2d(channels_in, rank, 3, stride=original.stride,
                            padding=original.padding, dilation=original.dilation, bias=False)
        point = nn.Conv2d(rank, channels_out, 1, bias=False)
        spatial.weight = nn.Parameter(vh[:rank].reshape_as(spatial.weight).contiguous())
        point.weight = nn.Parameter((u[:, :rank] * s[:rank]).reshape_as(point.weight).contiguous())
        layer.conv = point
        block = nn.Sequential(spatial, layer)
        for key in ("f", "i", "type", "np"):
            if hasattr(layer, key):
                setattr(block, key, getattr(layer, key))
        parent, _, name = path.rpartition(".")
        setattr(model.get_submodule(parent), name, block)
        report.append({"layer": path, "rank": rank, "weights_before": original.weight.numel(),
                       "weights_after": spatial.weight.numel() + point.weight.numel()})
    assert report
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = YOLO(str(ANCHOR), task="detect").model.float().eval()
    layers = factorize(model)
    parameters = check_model(model)
    write_checkpoint(model, args.output, fp16=True)
    print(json.dumps(dict(metadata(args.output), parameters=parameters, layers=layers)), flush=True)


if __name__ == "__main__":
    main()
