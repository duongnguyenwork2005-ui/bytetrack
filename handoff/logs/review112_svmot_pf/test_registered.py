"""Validate canonical RBPF numerical behavior, registry and unmodified old PF."""

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
import run_pf as setup
import numpy as np
from rbpf_ctrv import ConditionalPF, RBPFByteTracker
from tracker_ctrv import PFByteTracker
import test_rbpf as numerical

HERE, runner = setup.OUT, setup.runner
rt = runner.rt
numerical.ConditionalPF = ConditionalPF
numerical.test_conditional_kalman()
numerical.test_angles_lost_and_repeat()
registry = rt.tracker_registry()
assert registry["bytetrack_rbpf_ctrv"] is RBPFByteTracker
# The vendor NumPy registry exposes only its historical public subset. Add the
# old PF locally for regression, as R107 does, without changing that contract.
registry.setdefault("bytetrack_pf_ctrv", PFByteTracker)
assert registry["bytetrack_pf_ctrv"] is PFByteTracker
assert "bytetrack_rbpf_ctrv" in rt.STOCHASTIC_TRACKERS
yaml = setup.ROOT / "configs/phase2/trackers/r112/rbpf500.yaml"
cfg = rt.load_tracker_cfg(yaml)
video = "MVI_20052"
backend = rt.ul_backend.backend_name()
for seed in range(3):
    for case in ["base", "candidate"]:
        cache = runner.Prefix(runner.det_cache.open_cache(runner.CACHE, video, .25, [2, 3, 5, 7], 300), 128)
        mot, _, timing = rt.track_video(cache, cfg if case == "candidate" else runner.config(), .04,
                                       elapsed_mode="skip", seed=seed, video=video, registry=registry)
        saved = HERE / f"smoke/tracking/DETRAC-all/{case}-s{seed}/data/{video}.txt"
        np.testing.assert_allclose(mot.to_numpy(), np.loadtxt(saved, delimiter=",", ndmin=2), rtol=0, atol=5.1e-11)
        assert len(timing) == 128
        print("PASS normal registry prefix", backend, case, seed, flush=True)

for override in [{"pf_estimate": "map"}, {"pf_roughening": 1}, {"pf_optimal_proposal": True},
                 {"pf_measurement_height": "prediction"}, {"pf_lost_cv_fraction": .5}]:
    bad = rt.load_tracker_cfg(yaml, override)
    try:
        rt.make_tracker(bad, registry)
    except ValueError:
        pass
    else:
        raise AssertionError(("Unsupported option was silently ignored", override))
print("PASS incompatible configuration rejection", backend, flush=True)

# Exercise the real CLI and its seed/provenance/completion handling, writing only
# into this review folder instead of the project's normal trackers directory.
target = HERE / f"registered_cli_{backend}"
assert not target.exists()
runner.rt.config.PROCESSED_DIR = target
sys.argv = ["run_tracking.py", "--cache-id", runner.CACHE, "--split-name", "R112-dev",
            "--tracker-yaml", str(yaml), "--tracker-name", "rbpf", "--videos", video,
            "--fps", "25", "--elapsed-mode", "skip", "--seeds", "0", "1", "2"]
assert rt.main() == 0
for seed in range(3):
    folder = target / f"trackers/R112-dev/rbpf-s{seed}"
    meta = json.loads((folder / "run_meta.json").read_text())
    assert meta["seed"] == seed and meta["tracker_cfg"]["tracker_type"] == "bytetrack_rbpf_ctrv"
    assert "rbpf_ctrv.py" in json.dumps(meta["code"])
    mot = np.loadtxt(folder / f"data/{video}.txt", delimiter=",", ndmin=2)
    reference = np.loadtxt(HERE / f"dev/tracking/DETRAC-all/candidate-s{seed}/data/{video}.txt", delimiter=",", ndmin=2)
    np.testing.assert_allclose(mot, reference, rtol=0, atol=5.1e-11)
    assert len(__import__("pandas").read_csv(folder / f"timing/{video}.csv")) == 694
    print("PASS full CLI parity", backend, seed, 694, flush=True)
evidence = {"backend": backend, "prefix_units": 6, "cli_units": 3, "cli_frames": 2082,
            "canonical_source_sha256": hashlib.sha256((setup.ROOT / "src/rbpf_ctrv.py").read_bytes()).hexdigest()}
(HERE / f"registered_{backend}.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
