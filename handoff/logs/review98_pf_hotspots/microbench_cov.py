"""Compare equivalent weighted covariance kernels on Nano; no tracker edits."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path.home() / "nano_run/r63_20261030/v21combo/nano_transfer"
sys.path.insert(0, str(BASE / "src"))
os.environ["BYTETRACK_BACKEND"] = "np"
from ekf_ctrv import normalize_angle  # noqa: E402
from pf_ctrv import PFCTRV, _logsumexp  # noqa: E402

pf = PFCTRV(num_particles=500)
rng = np.random.default_rng(98)
X = rng.normal(size=(500, 9))
X[:, pf.THETA] = rng.uniform(-np.pi, np.pi, size=500)


def state(logw, variant):
    w = np.exp(logw - _logsumexp(logw))
    mean = w @ X
    th = X[:, pf.THETA]
    mean[pf.THETA] = np.arctan2(w @ np.sin(th), w @ np.cos(th))
    d = X - mean[None, :]
    d[:, pf.THETA] = normalize_angle(d[:, pf.THETA])
    if variant == "current":
        cov = (d * w[:, None]).T @ d
    elif variant == "right_weighted":
        cov = d.T @ (d * w[:, None])
    elif variant == "einsum":
        cov = np.einsum("ni,nj,n->ij", d, d, w, optimize=True)
    else:
        raise ValueError(variant)
    return mean, .5 * (cov + cov.T)


results = {}
for mode, logw in (("uniform", np.full(500, -np.log(500))),
                   ("peaked", np.log(rng.dirichlet(np.full(500, .3))))):
    baseline = state(logw, "current")
    results[mode] = {}
    for variant in ("current", "right_weighted", "einsum"):
        for _ in range(100):
            state(logw, variant)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(500):
                state(logw, variant)
            times.append((time.perf_counter() - t0) * 1000 / 500)
        got = state(logw, variant)
        results[mode][variant] = {"ms_per_call_median": float(np.median(times)),
                                  "max_abs_diff": float(max(np.max(np.abs(a-b))
                                                            for a, b in zip(got, baseline)))}
print(json.dumps(results, indent=2))
