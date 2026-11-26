"""Byte parity for the isolated R85 dense NMS candidate."""
import importlib.util
import sys
from pathlib import Path

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


root = Path(__file__).resolve().parents[3]
base_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "handoff/logs/review83_nano_opt/v21/src/nano_detector.py"
candidate_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "v22/src/nano_detector.py"
base = load("r85_nms_base", base_path)
candidate = load("r85_nms_candidate", candidate_path)
rng = np.random.default_rng(85)
count = 0
for n, n_valid in ((1, 1), (50, 40), (5040, 40), (5040, 300), (5040, 500)):
    raw = rng.uniform(0, 0.04, (1, 84, n)).astype(np.float32)
    raw[:, :4, :] *= 640
    raw[:, 4:, :n_valid] = rng.uniform(0, 1, (1, 80, n_valid))
    for conf in (0.05, 0.25, 0.95):
        for classes in (None, [1, 2, 3, 5, 7]):
            a = base.decode_nms(raw, conf, classes=classes)
            b = candidate.decode_nms(raw, conf, classes=classes)
            assert np.array_equal(a, b), (n, n_valid, conf, classes)
            count += 1
print("PASS: byte-identical decode in", count, "cases; numpy", np.__version__)
