"""Descriptive paired tracker timing from cached BDD runs; not a controlled Nano benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/processed/trackers/BDDCOCOVID-val"
OUT = ROOT / "handoff/logs/review95_pf_reacquire/bdd_speed_audit.json"


def main():
    videos = sorted(p.stem for p in (BASE / "r95-bdd-full-rescue-s0/timing").glob("*.csv"))
    assert len(videos) == 200
    samples = {case: {field: [] for field in ("t_update_ms", "t_assoc_ms")}
               for case in ("baseline", "candidate")}
    for seed in (0, 1, 2):
        for video in videos:
            old = pd.read_csv(BASE / f"r74-bdd-full-motion200-s{seed}/timing/{video}.csv")
            new = pd.read_csv(BASE / f"r95-bdd-full-rescue-s{seed}/timing/{video}.csv")
            assert len(old) == len(new)
            for case, frame in (("baseline", old), ("candidate", new)):
                for field in samples[case]:
                    x = frame[field].to_numpy(float)
                    assert np.isfinite(x).all() and (x >= 0).all()
                    samples[case][field].extend(x)
    result = {"scope": "PC replay timing from separate runs; descriptive, not Nano or controlled A/B",
              "video_seed_pairs": 600, "frames": len(samples["baseline"]["t_update_ms"])}
    for field in ("t_update_ms", "t_assoc_ms"):
        old, new = (np.asarray(samples[case][field]) for case in ("baseline", "candidate"))
        result[field] = {"baseline_mean_ms": float(old.mean()), "candidate_mean_ms": float(new.mean()),
                         "ratio": float(new.mean() / old.mean()),
                         "baseline_median_ms": float(np.median(old)),
                         "candidate_median_ms": float(np.median(new))}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
