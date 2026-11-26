"""Export weight-only INT8/INT4 and calibrated INT8 convolution variants for ONNX Runtime."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from ultralytics.data.augment import LetterBox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review103_nano_optimization"))
from fit_student import prepare_data


class TrainingCalibration(CalibrationDataReader):
    def __init__(self, work, shape):
        config, _ = prepare_data(work.resolve())
        paths = (config.parent / "train.txt").read_text().splitlines()
        self.images = iter(paths[:56] + paths[-8:])
        self.letterbox = LetterBox(new_shape=shape)

    def get_next(self):
        path = next(self.images, None)
        if path is None:
            return None
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to read calibration image {path}")
        image = self.letterbox(image=image)[:, :, ::-1].transpose(2, 0, 1)
        return {"images": np.ascontiguousarray(image[None], dtype=np.float32) / 255}


def weights_only(source, destination, bits, mixed=False):
    model = onnx.load(source)
    if bits == 4:
        model = onnx.version_converter.convert_version(model, 21)
    weights = {tensor.name: tensor for tensor in model.graph.initializer}
    conv_weights = {node.input[1] for node in model.graph.node if node.op_type == "Conv"}
    replacements, dequantizers = [], []
    for name in sorted(conv_weights.intersection(weights)):
        tensor = weights[name]
        w = numpy_helper.to_array(tensor)
        keep_eight = mixed and (name.startswith("model.23.") or w.size < 36_000)
        precision = 8 if keep_eight else bits
        dtype = TensorProto.INT8 if precision == 8 else TensorProto.INT4
        limit = 2 ** (precision - 1) - 1
        axes = tuple(range(1, w.ndim))
        scales = np.maximum(np.max(np.abs(w), axis=axes) / limit, 1e-12).astype(np.float32)
        shape = (len(scales),) + (1,) * len(axes)
        values = np.clip(np.rint(w / scales.reshape(shape)), -limit, limit).astype(np.int8)
        raw = values.flatten().view(np.uint8)
        if precision == 4:
            raw = np.pad(raw & 15, (0, raw.size % 2))
            raw = raw[::2] | (raw[1::2] << 4)
        weight = helper.make_tensor(name + "_q", dtype, w.shape, raw.tobytes(), raw=True)
        zero_bytes = len(scales) if precision == 8 else (len(scales) + 1) // 2
        zero = helper.make_tensor(name + "_z", dtype, (len(scales),), bytes(zero_bytes), raw=True)
        scale = numpy_helper.from_array(scales, name + "_s")
        model.graph.initializer.remove(tensor)
        replacements.extend([weight, zero, scale])
        dequantizers.append(helper.make_node("DequantizeLinear", [weight.name, scale.name, zero.name], [name], axis=0))
    assert dequantizers
    model.graph.initializer.extend(replacements)
    nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(dequantizers + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, bits, mixed in (("weight_int8", 8, False), ("weight_int4", 4, False), ("weight_mixed_4_8", 4, True)):
        target = args.output / f"{name}.onnx"
        try:
            weights_only(args.source, target, bits, mixed=mixed)
            rows[target.stem] = {"bytes": target.stat().st_size}
        except (RuntimeError, ValueError, TypeError, onnx.checker.ValidationError) as error:
            rows[target.stem] = {"error": str(error)}
        print(json.dumps(rows), flush=True)
    shape = [dim.dim_value for dim in onnx.load(args.source).graph.input[0].type.tensor_type.shape.dim]
    target = args.output / "static_int8.onnx"
    quantize_static(args.source, target, TrainingCalibration(args.output, tuple(shape[-2:])),
                    quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, op_types_to_quantize=["Conv"])
    onnx.checker.check_model(onnx.load(target))
    rows[target.stem] = {"bytes": target.stat().st_size, "calibration_images": 64}
    (args.output / "quantized_sizes.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
