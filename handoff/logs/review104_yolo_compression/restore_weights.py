"""Load the exact quantized ONNX weights as FP32 for a full GPU detection validation."""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from ultralytics import YOLO

from prepare import ANCHOR, check_model, write_checkpoint


def restore(onnx_path, destination):
    model = YOLO(str(ANCHOR), task="detect").model.float().eval().fuse(verbose=False)
    parameters = dict(model.named_parameters())
    graph = onnx.load(onnx_path).graph
    initializers = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    count = 0
    with torch.no_grad():
        for node in graph.node:
            if node.op_type != "DequantizeLinear":
                continue
            q, scale, zero = (initializers[key].astype(np.float32) for key in node.input)
            shape = (len(scale),) + (1,) * (q.ndim - 1)
            weight = (q - zero.reshape(shape)) * scale.reshape(shape)
            target = parameters[node.output[0]]
            assert target.shape == weight.shape
            target.copy_(torch.from_numpy(weight))
            count += 1
    assert count == sum(isinstance(layer, torch.nn.Conv2d) for layer in model.modules())
    check_model(model)
    write_checkpoint(model, destination)
    print(f"Loaded {count} convolution weights from {onnx_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    restore(args.source, args.destination)


if __name__ == "__main__":
    main()
