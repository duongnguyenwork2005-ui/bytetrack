"""Create compact YOLO11n variants while preserving every output scale and class."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import lzma
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck, SPPF

ANCHOR = Path("models/r101_ridesafe_interp50/yolo11n_mixed5_interp50.pt")
SHA256 = "d6aad8667ea7d344592c5aa170a7dbcda968c4aa30930cecc08d3541566daa82"


def write_checkpoint(model, path, fp16=False):
    torch.save({"model": deepcopy(model).half() if fp16 else model,
                "train_args": {"task": "detect", "imgsz": 640}}, path)


def metadata(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "path": path.as_posix()}


def trim_pair(first, second, fraction):
    first_conv = first.conv
    second_conv = second.conv if hasattr(second, "conv") else second
    width = first_conv.out_channels
    kept = max(8, round(width * (1 - fraction) / 8) * 8)
    if kept >= width:
        return
    bn = first.bn
    score = first_conv.weight.detach().flatten(1).norm(dim=1) * bn.weight.detach().abs()
    indices = score.topk(kept).indices.sort().values
    # SPPF concatenates the original features and three pooled versions.
    assert second_conv.in_channels % width == 0
    repeats = second_conv.in_channels // width
    inputs = torch.cat([indices + repeat * width for repeat in range(repeats)])
    first_conv.out_channels = kept
    first_conv.weight = nn.Parameter(first_conv.weight[indices].detach().contiguous())
    assert first_conv.bias is None and first_conv.groups == second_conv.groups == 1
    bn.num_features = kept
    for key in ("weight", "bias"):
        setattr(bn, key, nn.Parameter(getattr(bn, key)[indices].detach().contiguous()))
    for key in ("running_mean", "running_var"):
        setattr(bn, key, getattr(bn, key)[indices].detach().contiguous())
    second_conv.in_channels = len(inputs)
    second_conv.weight = nn.Parameter(second_conv.weight[:, inputs].detach().contiguous())


def prune_channels(model, fraction):
    for module in list(model.modules()):
        if isinstance(module, (Bottleneck, SPPF)):
            trim_pair(module.cv1, module.cv2, fraction)
    for block in model.model[10].m:
        trim_pair(*block.ffn, fraction)
    for scale, branch in enumerate(model.model[-1].cv2):
        amount = min(fraction, .25) if scale == 0 else fraction
        trim_pair(branch[0], branch[1], amount)
        trim_pair(branch[1], branch[2], amount)


def prune_weights(model, fraction):
    projection = model.model[-1].dfl.conv
    weights = [m.weight for m in model.modules() if isinstance(m, nn.Conv2d) and m is not projection]
    flat = torch.cat([w.detach().reshape(-1).abs() for w in weights])
    cutoff = flat.kthvalue(int(len(flat) * fraction)).values
    with torch.no_grad():
        for weight in weights:
            weight.masked_fill_(weight.abs() <= cutoff, 0)
    return sum(int((w == 0).sum()) for w in weights) / len(flat)


def check_model(model):
    with torch.inference_mode():
        result = model.float().eval()(torch.zeros(1, 3, 384, 640))[0]
    assert result.shape == (1, 9, 5040) and torch.isfinite(result).all()
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    assert metadata(ANCHOR)["sha256"] == SHA256
    teacher = YOLO(str(ANCHOR), task="detect").model.float().eval()
    rows = {"anchor": dict(metadata(ANCHOR), parameters=check_model(teacher))}
    fp16 = args.models / "yolo11n_50_50_fp16.pt"
    write_checkpoint(teacher, fp16, fp16=True)
    rows["fp16"] = dict(metadata(fp16), parameters=check_model(YOLO(str(fp16), task="detect").model))
    for name, path in (("anchor", ANCHOR), ("fp16", fp16)):
        original = path.read_bytes()
        compressed = gzip.compress(original, compresslevel=6, mtime=0)
        assert gzip.decompress(compressed) == original
        rows[name]["gzip_bytes"] = len(compressed)
        compressed = lzma.compress(original, preset=3)
        assert lzma.decompress(compressed) == original
        rows[name]["xz_bytes"] = len(compressed)
        if name == "fp16":
            fp16.with_suffix(".pt.xz").write_bytes(compressed)
    sparse = deepcopy(teacher)
    zeros = prune_weights(sparse, .30)
    sparse_path = args.work / "weight_prune30.pt"
    write_checkpoint(sparse, sparse_path, fp16=True)
    packed = lzma.compress(sparse_path.read_bytes(), preset=3)
    rows["weight_prune30"] = dict(metadata(sparse_path), parameters=check_model(sparse),
                                   sparsity=zeros, xz_bytes=len(packed))
    for fraction in (.25, .50):
        candidate = deepcopy(teacher)
        prune_channels(candidate, fraction)
        name = f"channels{int(fraction * 100)}"
        path = args.work / f"{name}_init.pt"
        write_checkpoint(candidate, path)
        rows[name] = dict(metadata(path), parameters=check_model(candidate))
    (args.work / "sizes.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
