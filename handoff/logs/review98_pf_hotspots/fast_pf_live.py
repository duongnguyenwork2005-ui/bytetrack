"""Opt-in faster logsumexp for PF in an isolated Nano live process."""
import math
import os
from pathlib import Path
import sys

import numpy as np

os.environ["BYTETRACK_BACKEND"] = "np"
pkg = Path.home() / "nano_run/r63_20261030/v21combo/nano_transfer"
sys.path.insert(0, str(pkg / "src"))
import bytetrack_np  # noqa: E402
import tracker_ctrv  # noqa: E402
import pf_ctrv  # noqa: E402

original_registry = bytetrack_np.tracker_registry_np


def experiment_registry():
    reg = original_registry()
    reg["bytetrack_pf_ctrv"] = tracker_ctrv.PFByteTracker
    return reg


def fast_logsumexp(a):
    m = float(a.max())
    if not math.isfinite(m):
        return m
    return m + float(np.log(np.exp(a - m).sum()))


bytetrack_np.tracker_registry_np = experiment_registry
pf_ctrv._logsumexp = fast_logsumexp
import nano_live  # noqa: E402

if __name__ == "__main__":
    sys.exit(nano_live.main())
