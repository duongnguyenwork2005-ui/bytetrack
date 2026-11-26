"""Ensure the normal registry bridge returns the experimentally measured MOT."""

from pathlib import Path
import run_pf as setup
import numpy as np
from backend import RBPFByteTracker
from rbpf import installed

runner = setup.runner
for seed in range(3):
    video = "MVI_20052"
    cache = runner.Prefix(
        runner.det_cache.open_cache(runner.CACHE, video, .25, [2, 3, 5, 7], 300), 128
    )
    registry = runner.rt.tracker_registry()
    registry["bytetrack_pf_ctrv"] = RBPFByteTracker
    mot, classes, timing = runner.rt.track_video(
        cache, runner.config(), .04, elapsed_mode="skip", seed=seed,
        video=video, registry=registry
    )
    saved = setup.OUT / "smoke/tracking/DETRAC-all" / f"candidate-s{seed}/data/{video}.txt"
    reference = np.loadtxt(saved, delimiter=",", ndmin=2)
    # Text snapshots retain ten fractional digits.
    np.testing.assert_allclose(mot.to_numpy(), reference, rtol=0, atol=5.1e-11)
    assert len(timing) == 128
    print("PASS application registry parity", seed, len(mot), flush=True)
