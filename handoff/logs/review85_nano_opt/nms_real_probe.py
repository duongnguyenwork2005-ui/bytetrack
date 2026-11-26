"""Nano CPU parity and timing on 300 real FP32 NMS inputs exported from PC."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load("r85_real_base", Path(sys.argv[1]))
candidate = load("r85_real_candidate", Path(sys.argv[2]))
z = np.load(sys.argv[3])
data_boxes, data_scores, offsets = z["boxes"], z["scores"], z["offsets"]
assert len(offsets) == 301
samples = {"base": [], "matrix": []}
for pass_id in range(4):
    for frame in range(300):
        s, e = int(offsets[frame]), int(offsets[frame + 1])
        b, sc = data_boxes[s:e], data_scores[s:e]
        pair = (("base", base.nms_numpy), ("matrix", candidate.nms_matrix))
        if pass_id % 2:
            pair = pair[::-1]
        results = {}
        for name, fn in pair:
            t = time.perf_counter()
            results[name] = fn(b, sc, 0.7)
            samples[name].append((time.perf_counter() - t) * 1000)
        if not np.array_equal(results["base"], results["matrix"]):
            raise RuntimeError("NMS parity mismatch frame {} pass {}".format(frame + 1, pass_id))
out = {"frames": 300, "passes": 4, "parity": True, "numpy": np.__version__,
       "candidates_p50": float(np.median(np.diff(offsets))),
       "candidates_p90": float(np.percentile(np.diff(offsets), 90)),
       "base_p50_ms": float(np.median(samples["base"])),
       "matrix_p50_ms": float(np.median(samples["matrix"])),
       "base_p90_ms": float(np.percentile(samples["base"], 90)),
       "matrix_p90_ms": float(np.percentile(samples["matrix"], 90)),
       "base_total_ms": float(np.sum(samples["base"])),
       "matrix_total_ms": float(np.sum(samples["matrix"]))}
print(json.dumps(out, indent=2, sort_keys=True))
