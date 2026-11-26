"""Compare the R98 PF and uniform-weight reuse on the same Nano detection caches."""
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import sys

import numpy as np

NANO = Path.home() / "nano_run"
PACKAGE = NANO / "r63_20261030/v21combo/nano_transfer"
CACHE = NANO / "r98_pf_hotspots_20261110/cache"
OUT = NANO / "r100_profile_20261111"
sys.path.insert(0, str(PACKAGE / "src"))
os.environ["BYTETRACK_BACKEND"] = "np"

import bytetrack_np  # noqa: E402
import det_cache  # noqa: E402
import pf_ctrv  # noqa: E402
import run_tracking  # noqa: E402
import tracker_ctrv  # noqa: E402


def lse_r98(log_weights):
    maximum = float(log_weights.max())
    if not math.isfinite(maximum):
        return maximum
    return maximum + float(np.log(np.exp(log_weights - maximum).sum()))


base_estimate = pf_ctrv.PFCTRV.estimate_state


def estimate_uniform(self, particles, log_weights):
    first = float(log_weights[0])
    if not math.isfinite(first) or not np.equal(log_weights, first).all():
        return base_estimate(self, particles, log_weights)

    key = (len(log_weights), first, log_weights.dtype)
    cached = getattr(self, "_r100_uniform_weights", None)
    if cached is None or cached[0] != key:
        weights = np.exp(log_weights - lse_r98(log_weights))
        self._r100_uniform_weights = (key, weights)
    else:
        weights = cached[1]

    if self.estimate == "map":
        mean = particles[0].copy()
    else:
        mean = weights @ particles
        theta = particles[:, self.THETA]
        mean[self.THETA] = np.arctan2(weights @ np.sin(theta), weights @ np.cos(theta))
    residuals = particles - mean[None, :]
    residuals[:, self.THETA] = pf_ctrv.normalize_angle(residuals[:, self.THETA])
    covariance = (residuals * weights[:, None]).T @ residuals
    return mean, 0.5 * (covariance + covariance.T)


def register_pf():
    old_registry = bytetrack_np.tracker_registry_np

    def registry():
        return dict(old_registry(), bytetrack_pf_ctrv=tracker_ctrv.PFByteTracker)

    bytetrack_np.tracker_registry_np = registry


def machine_state():
    memory = {line.split(":")[0]: int(line.split()[1])
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("SwapTotal:", "SwapFree:"))}
    swap_mb = (memory["SwapTotal"] - memory["SwapFree"]) / 1024
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    errors = int(Path("/sys/fs/ext4/mmcblk1p1/errors_count").read_text())
    temperature = max(int(path.read_text()) / 1000
                      for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"))
    return boot, errors, swap_mb, temperature


def main():
    boot, errors, swap, temperature = machine_state()
    assert errors == 0 and swap < 100 and temperature < 75
    register_pf()
    pf_ctrv._logsumexp = lse_r98
    det_cache.CACHE_ROOT = CACHE
    cfg = run_tracking.load_tracker_cfg(
        PACKAGE / "configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml", {"track_buffer": 30})
    results = []
    for video in ("MVI_39031", "MVI_40855"):
        reference = None
        for label in ("r98", "uniform", "uniform", "r98"):
            pf_ctrv.PFCTRV.estimate_state = base_estimate if label == "r98" else estimate_uniform
            cache = det_cache.open_cache("r98_" + video, video, .25, [2, 3, 5, 7])
            mot, classes, timing = run_tracking.track_video(
                cache, cfg, dt_nominal=1 / 25, elapsed_mode="skip", seed=None, video=video)
            assert len(timing) == 150
            if reference is None:
                reference = mot.copy(), classes.copy()
            assert mot.equals(reference[0]) and classes.equals(reference[1]), (video, label)
            current_boot, current_errors, current_swap, current_temp = machine_state()
            assert current_boot == boot and current_errors == 0
            assert current_swap - swap <= 100 and current_temp < 75
            sample = timing["t_update_ms"].iloc[25:]
            row = {"video": video, "variant": label, "frames": len(timing),
                   "mot_sha256": hashlib.sha256(mot.to_csv(index=False).encode()).hexdigest(),
                   "classes_sha256": hashlib.sha256(classes.to_csv(index=False).encode()).hexdigest(),
                   "update_median_ms": float(sample.median()), "update_p90_ms": float(sample.quantile(.9)),
                   "temperature_c": current_temp, "swap_mb": current_swap, "boot": boot}
            results.append(row)
            (OUT / "pf_uniform_abba.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(row), flush=True)
        print(video, {label: median(r["update_median_ms"] for r in results
                                   if r["video"] == video and r["variant"] == label)
                      for label in ("r98", "uniform")}, flush=True)
    print("R100_REPLAY_PASS", flush=True)


if __name__ == "__main__":
    main()
