"""Dense IoU NMS parity and CPU timing; synthetic probe only."""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np


def load_module(path):
    spec = importlib.util.spec_from_file_location("nano_detector_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def matrix_nms(boxes, scores, threshold):
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    b = boxes[order]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    xx1 = np.maximum(b[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(b[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(b[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    iou = inter / (area[:, None] + area[None, :] - inter)
    suppress = np.triu(iou > threshold, 1)
    active = np.ones(len(b), dtype=bool)
    keep = []
    for i in range(len(b)):
        if active[i]:
            keep.append(order[i])
            active[i + 1:] &= ~suppress[i, i + 1:]
    return np.asarray(keep, dtype=np.int64)


def main():
    module = load_module(Path(sys.argv[1]))
    rng = np.random.default_rng(85)
    cases = []
    for n in (20, 50, 100, 200, 400):
        for _ in range(20):
            centers = rng.uniform(0, 600, (n, 2))
            centers[::4] = centers[1::4][: len(centers[::4])] + rng.normal(0, 6, (len(centers[::4]), 2))
            sizes = rng.uniform(8, 120, (n, 2))
            boxes = np.column_stack((centers - sizes / 2, centers + sizes / 2)).astype(np.float32)
            classes = rng.integers(0, 5, n)
            boxes += (classes * module.MAX_WH).astype(np.float32)[:, None]
            scores = rng.uniform(0.05, 1, n).astype(np.float32)
            cases.append((boxes, scores))
    mismatches = []
    for i, (b, s) in enumerate(cases):
        a = module.nms_numpy(b, s, 0.7)
        c = matrix_nms(b, s, 0.7)
        if not np.array_equal(a, c):
            mismatches.append((i, len(a), len(c)))
    results = {}
    for name, fn in (("numpy", module.nms_numpy), ("matrix", matrix_nms)):
        samples = []
        for _ in range(5):
            t = time.perf_counter()
            for b, s in cases:
                fn(b, s, 0.7)
            samples.append((time.perf_counter() - t) * 1000 / len(cases))
        results[name] = sorted(samples)[2]
    print("numpy", np.__version__, "cases", len(cases), "mismatches", mismatches[:10])
    print("ms_per_case", results)
    for start, n in enumerate((20, 50, 100, 200, 400)):
        group = cases[start * 20:(start + 1) * 20]
        row = {}
        for name, fn in (("numpy", module.nms_numpy), ("matrix", matrix_nms)):
            t = time.perf_counter()
            for b, s in group:
                fn(b, s, 0.7)
            row[name] = (time.perf_counter() - t) * 1000 / len(group)
        print("n", n, "ms_per_case", row)
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
