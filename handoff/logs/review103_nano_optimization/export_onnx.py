"""Export a static five-class YOLO engine input, and verify its ONNX graph."""
import hashlib
import json
from pathlib import Path
from shutil import copyfile
import sys

import onnx
import torch
from ultralytics import YOLO


def main():
    checkpoint, target, height = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    staged = target.with_suffix(".pt")
    copyfile(checkpoint, staged)
    model = YOLO(str(staged), task="detect")
    assert list(model.names.values()) == ["car", "truck", "bus", "motorcycle", "bicycle"]
    output = Path(model.export(format="onnx", imgsz=[height, 640], opset=12, simplify=False,
                               dynamic=False, device="cpu", batch=1, verbose=False))
    assert output.resolve() == target.resolve()
    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    shape = lambda tensor: [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]
    anchors = sum(height // stride * (640 // stride) for stride in (8, 16, 32))
    assert shape(graph.graph.input[0]) == [1, 3, height, 640]
    assert shape(graph.graph.output[0]) == [1, 9, anchors]
    payload = {"pt_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "input": shape(graph.graph.input[0]), "output": shape(graph.graph.output[0]), "opset": 12}
    target.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
