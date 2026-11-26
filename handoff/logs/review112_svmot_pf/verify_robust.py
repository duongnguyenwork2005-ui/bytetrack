"""Recombine robust raw metrics independently before any source integration."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from verify_all import combine, read

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / "robust"
    frozen = read(root / "inputs.json")["sha256"]
    for name, expected in frozen.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    summary = read(root / "dev_gate.json")
    checks, frames = 0, 0
    for seed in range(3):
        value, raw = combine(root / f"dev/evaluation/candidate-s{seed}/raw_fields.json")
        assert all(abs(value[k] - summary["candidate_seeds"][seed][k]) < 1e-12 for k in value)
        for video in raw:
            folder = root / f"dev/tracking/DETRAC-all/candidate-s{seed}"
            mot = np.loadtxt(folder / f"data/{video}.txt", delimiter=",", ndmin=2)
            assert np.isfinite(mot).all() and (mot[:, 4:6] > 0).all()
            assert len(np.unique(mot[:, :2], axis=0)) == len(mot)
            timing = pd.read_csv(folder / f"timing/{video}.csv")
            assert len(timing) == read(root / "inputs.json")["frames"][video]
            frames += len(timing)
            checks += 1
    transfer = root / "transfer_svmot/verified_summary.json"
    groups = 3
    if transfer.exists():
        results = read(transfer)
        for seed in range(3):
            value, raw = combine(root / f"transfer_svmot/evaluation/candidate-s{seed}/raw_fields.json")
            assert all(abs(value[k] - results["candidate_seeds"][seed][k]) < 1e-12 for k in value)
            checks += len(raw)
            groups += 1
    report = {"frozen_hashes_before_integration": len(frozen),
              "raw_combined_groups": groups, "clip_seed_checks": checks,
              "dev_replay_frames": frames, "dev_gate_pass": summary["gate_pass"],
              "transfer_checked": transfer.exists()}
    (root / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
