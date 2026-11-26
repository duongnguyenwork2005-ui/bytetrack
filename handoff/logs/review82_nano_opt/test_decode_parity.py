"""Check that the filtered class-score view keeps v17-C NMS output unchanged."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = load("nano_c1", ROOT / "handoff/logs/review63/v17_c/src/nano_detector.py")
candidate = load("nano_v18", Path(__file__).resolve().parent / "v18/src/nano_detector.py")
rng = np.random.default_rng(82)
for n in (1, 50, 5040):
    raw = rng.uniform(0, 0.04, (1, 84, n)).astype(np.float32)
    if n:
        raw[:, :4, :] *= 640
        raw[:, 4:, :min(n, 40)] = rng.uniform(0, 1, (1, 80, min(n, 40)))
    for conf in (0.05, 0.25, 0.95):
        for classes in (None, [1, 2, 3, 5, 7]):
            old = base.decode_nms(raw, conf, classes=classes)
            new = candidate.decode_nms(raw, conf, classes=classes)
            assert np.array_equal(old, new), (n, conf, classes)
print("PASS: decode output identical in 18 cases")
