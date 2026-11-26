"""Opt in PF for one isolated Nano experiment; no installed package edits."""
import os
from pathlib import Path
import sys

os.environ["BYTETRACK_BACKEND"] = "np"
pkg = Path.home() / "nano_run/r63_20261030/v21combo/nano_transfer"
sys.path.insert(0, str(pkg / "src"))

import bytetrack_np  # noqa: E402
import tracker_ctrv  # noqa: E402

original_registry = bytetrack_np.tracker_registry_np


def experiment_registry():
    registry = original_registry()
    registry["bytetrack_pf_ctrv"] = tracker_ctrv.PFByteTracker
    return registry


bytetrack_np.tracker_registry_np = experiment_registry

import nano_live  # noqa: E402

if __name__ == "__main__":
    sys.exit(nano_live.main())
