"""R85 CPU probe of byte-exact BGR uint8 -> RGB CHW float32 alternatives."""
import importlib.util
import json
import sys
import time

import cv2
import numpy as np


spec = importlib.util.spec_from_file_location("nano_detector_probe", sys.argv[1])
nd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nd)
src = "/home/jetson/nano_run/r63_20261030/v17c/nano_transfer/data/smoke/MVI_39031_f0001-0300"
images = []
for frame in (1, 100, 200, 300):
    img = cv2.imread(src + "/img{:05d}.jpg".format(frame))
    if img is None:
        raise RuntimeError("missing frame {}".format(frame))
    images.append(nd.letterbox(img, (384, 640))[0])
lut = np.arange(256, dtype=np.float32) / 255.0
out = np.empty((1, 3, 384, 640), dtype=np.float32)


def no_u8_copy(img):
    x = img.transpose(2, 0, 1)[::-1]
    return np.ascontiguousarray(lut[x]).reshape((1,) + x.shape)


def take_chw(img):
    x = img.transpose(2, 0, 1)[::-1]
    np.take(lut, x, out=out[0])
    return out


def take_planes(img):
    for dst, src_ch in ((0, 2), (1, 1), (2, 0)):
        np.take(lut, img[:, :, src_ch], out=out[0, dst])
    return out


def channel_assign(img):
    for dst, src_ch in ((0, 2), (1, 1), (2, 0)):
        out[0, dst] = lut[img[:, :, src_ch]]
    return out


fns = {"current": nd.preprocess, "no_u8_copy": no_u8_copy,
       "take_chw": take_chw, "take_planes": take_planes, "channel_assign": channel_assign}
for name, fn in fns.items():
    for img in images:
        if not np.array_equal(fn(img), nd.preprocess(img)):
            raise RuntimeError("parity mismatch " + name)
samples = {k: [] for k in fns}
order = list(fns)
for rep in range(30):
    order = order[1:] + order[:1]
    for name in order:
        fn = fns[name]
        for img in images:
            t = time.perf_counter()
            fn(img)
            samples[name].append((time.perf_counter() - t) * 1000)
print(json.dumps({k: {"p50_ms": float(np.median(v)), "p90_ms": float(np.percentile(v, 90))}
                  for k, v in samples.items()}, sort_keys=True, indent=2))
