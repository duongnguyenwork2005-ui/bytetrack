"""A/B/B/A replay of real Nano detections; require exact tracking parity."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path.home() / "nano_run/r63_20261030/v21combo/nano_transfer"
ROOT = Path.home() / "nano_run/r98_pf_hotspots_20261110"
sys.path.insert(0, str(BASE / "src"))
os.environ["BYTETRACK_BACKEND"] = "np"
import det_cache  # noqa: E402
import run_tracking  # noqa: E402
import bytetrack_np  # noqa: E402
import tracker_ctrv  # noqa: E402
import pf_ctrv  # noqa: E402

old_registry = bytetrack_np.tracker_registry_np
old_lse = pf_ctrv._logsumexp


def registry():
    reg = old_registry()
    reg["bytetrack_pf_ctrv"] = tracker_ctrv.PFByteTracker
    return reg


def candidate_lse(a):
    m = float(a.max())
    if not math.isfinite(m):
        return m
    return m + float(np.log(np.exp(a - m).sum()))


bytetrack_np.tracker_registry_np = registry
det_cache.CACHE_ROOT = ROOT / "cache"
cfg = run_tracking.load_tracker_cfg(BASE / "configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml",
                                    {"track_buffer": 30})
results = []
for video in ("MVI_39031", "MVI_40855"):
    canonical = None
    for phase, case in enumerate(("baseline", "candidate", "candidate", "baseline")):
        pf_ctrv._logsumexp = old_lse if case == "baseline" else candidate_lse
        vd = det_cache.open_cache("r98_" + video, video, .25, [2, 3, 5, 7])
        t0 = time.perf_counter()
        df, cls, timing = run_tracking.track_video(vd, cfg, dt_nominal=1/25,
                                                  elapsed_mode="skip", seed=None, video=video)
        elapsed = time.perf_counter() - t0
        assert len(timing) == 150
        if canonical is None:
            canonical = (df.copy(), cls.copy())
        assert df.equals(canonical[0]) and cls.equals(canonical[1]), (video, case, phase)
        payload = df.to_csv(index=False).encode()
        item = {"video": video, "phase": phase, "case": case, "n_mot_rows": len(df),
                "mot_sha256": hashlib.sha256(payload).hexdigest(),
                "elapsed_s": elapsed,
                "update_median_ms": float(timing["t_update_ms"].iloc[25:].median()),
                "update_mean_ms": float(timing["t_update_ms"].iloc[25:].mean()),
                "update_p90_ms": float(timing["t_update_ms"].iloc[25:].quantile(.9))}
        results.append(item)
        (ROOT / "lse_abba.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(item), flush=True)
pf_ctrv._logsumexp = old_lse
print("R98_LSE_ABBA_PASS", flush=True)
