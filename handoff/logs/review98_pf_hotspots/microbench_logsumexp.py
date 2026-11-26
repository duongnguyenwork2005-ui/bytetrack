"""Test equivalent logsumexp dispatch before touching PF code."""
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path.home() / "nano_run/r63_20261030/v21combo/nano_transfer"
sys.path.insert(0, str(BASE / "src"))
os.environ["BYTETRACK_BACKEND"] = "np"
from pf_ctrv import _logsumexp  # noqa: E402


def candidate(a):
    m = float(a.max())
    if not math.isfinite(m):
        return m
    return m + float(np.log(np.exp(a - m).sum()))


rng = np.random.default_rng(98)
data = [np.full(500, -np.log(500)), np.log(rng.dirichlet(np.full(500, .3))),
        np.full(500, -np.inf)]
out = []
for x in data:
    a, b = _logsumexp(x), candidate(x)
    assert a == b or (np.isneginf(a) and np.isneginf(b)), (a, b)
    row = {"length": len(x), "kind": "all_neg_inf" if np.isneginf(x[0]) else "finite"}
    for name, fn in (("current", _logsumexp), ("candidate", candidate)):
        for _ in range(100):
            fn(x)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(10000):
                fn(x)
            times.append((time.perf_counter() - t0) / 10000 * 1e6)
        row[name + "_us"] = float(np.median(times))
    out.append(row)
print(json.dumps(out))
