"""Export a YOLOv8n whose two repeated backbone C2f blocks each lose one bottleneck."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import onnx
import torch
from ultralytics import YOLO


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    source = YOLO(str(args.source), task="detect").model.float().eval()
    slim = deepcopy(source)
    changes = []
    for index in (4, 6):
        layer = slim.model[index]
        assert type(layer).__name__ == "C2f" and len(layer.m) == 2
        del layer.m[-1]
        conv = layer.cv2.conv
        before = conv.in_channels
        conv.in_channels = layer.c * (2 + len(layer.m))
        conv.weight = torch.nn.Parameter(conv.weight[:, :conv.in_channels, :, :].contiguous())
        slim.yaml["backbone"][index][1] = 3
        changes.append({"layer": index, "bottlenecks": [2, 1], "last_conv_in_channels": [before, conv.in_channels]})
    with torch.inference_mode():
        tensor = torch.rand(1, 3, 384, 640)
        assert slim(tensor)[0].shape == source(tensor)[0].shape == (1, 84, 5040)
        assert torch.isfinite(slim(tensor)[0]).all()
    pt = args.output / "yolov8n_c2f_slim.pt"
    torch.save({"model": slim, "train_args": {"task": "detect", "imgsz": 640}}, pt)
    onnx_path = Path(YOLO(str(pt), task="detect").export(
        format="onnx", imgsz=[384, 640], opset=12, simplify=False, dynamic=False,
        device="cpu", batch=1, verbose=False))
    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph)
    shape = lambda value: [d.dim_value for d in value.type.tensor_type.shape.dim]
    assert shape(graph.graph.input[0]) == [1, 3, 384, 640]
    assert shape(graph.graph.output[0]) == [1, 84, 5040]
    result = {"source_sha256": sha256(args.source), "pt_sha256": sha256(pt), "onnx_sha256": sha256(onnx_path),
              "source_params": sum(p.numel() for p in source.parameters()),
              "candidate_params": sum(p.numel() for p in slim.parameters()), "trained_after_pruning": False,
              "opset": graph.opset_import[0].version, "changes": changes,
              "onnx_input": shape(graph.graph.input[0]), "onnx_output": shape(graph.graph.output[0])}
    (args.output / "export.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
