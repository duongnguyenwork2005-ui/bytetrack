"""Isolated NMS parity/speed probe; synthetic data only, not a pipeline result."""
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def load_module(path):
    spec = importlib.util.spec_from_file_location("nano_detector_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cv_nms(boxes, scores, threshold):
    # cv2 accepts x,y,w,h and returns indices in descending score order.
    rect = boxes.copy()
    rect[:, 2:] -= rect[:, :2]
    found = cv2.dnn.NMSBoxes(rect.tolist(), scores.tolist(), 0.0, threshold)
    return np.asarray(found, dtype=np.int64).reshape(-1)


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
        c = cv_nms(b, s, 0.7)
        if not np.array_equal(a, c):
            mismatches.append((i, len(a), len(c)))
    results = {}
    for name, fn in (("numpy", module.nms_numpy), ("cv2", cv_nms)):
        samples = []
        for _ in range(5):
            t = time.perf_counter()
            for b, s in cases:
                fn(b, s, 0.7)
            samples.append((time.perf_counter() - t) * 1000 / len(cases))
        results[name] = sorted(samples)[2]
    print("opencv", cv2.__version__, "numpy", np.__version__, "cases", len(cases), "mismatches", mismatches[:10])
    print("ms_per_case", results)
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
