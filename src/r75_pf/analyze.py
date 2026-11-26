"""Compare R75 PF recentering with the paired R72 motion-2 baseline on UA dev12."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import boot_stats
from kc48 import boot_seed

METRICS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW"]
VIDEOS = json.loads((ROOT / "results/phase2/manifests/detrac_dev.json").read_text(encoding="utf-8"))["videos"]


def load(run: str) -> tuple[dict, dict]:
    folder = ROOT / "results/trackeval/DETRAC-all" / run
    raw = boot_stats.load_raw_fields(folder / "raw_fields.json")
    if set(raw) != set(VIDEOS):
        raise ValueError(f"raw coverage mismatch: {run}")
    score = boot_stats.combine(raw, VIDEOS)
    csv = pd.read_csv(folder / "combined_metrics.csv").iloc[0]
    if max(abs(score[m] - float(csv[m])) for m in METRICS) > 1e-9:
        raise ValueError(f"COMBINED mismatch: {run}")
    return raw, score


def main() -> None:
    old, new, rows = [], [], []
    for seed in (0, 1, 2):
        old_run = f"p2-kc48_r72_pf_dev-A-motion200-s{seed}"
        new_run = f"p2-kc48_r75_pf_dev-A-recenter-s{seed}"
        old_raw, old_score = load(old_run)
        new_raw, new_score = load(new_run)
        tracker = ROOT / "data/processed/trackers/DETRAC-all" / new_run
        meta = json.loads((tracker / "run_meta.json").read_text(encoding="utf-8"))
        if (meta["status"] != "complete" or meta["videos"] != VIDEOS or
                meta["tracker_cfg"]["pf_recenter_resample"] is not True or
                meta["seed"] != seed):
            raise ValueError(f"new run metadata mismatch: {new_run}")
        for video in VIDEOS:
            marker = json.loads((tracker / "video_meta" / f"{video}.json").read_text(encoding="utf-8"))
            if not marker["complete"] or marker["input_fingerprint"] != meta["input_fingerprint"]:
                raise ValueError(f"marker mismatch: {new_run}/{video}")
        old.append(old_raw)
        new.append(new_raw)
        for kind, score in (("motion200", old_score), ("recenter", new_score)):
            rows.append(dict(kind=kind, seed=seed, **{m: score[m] for m in METRICS}))
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("kind", sort=False)[METRICS].mean().reset_index()
    paired = boot_seed.paired_bootstrap_seed(new, old, VIDEOS, n_boot=5000, seed=0, metrics=METRICS)
    out = ROOT / "results/phase2/kc48_r75_pf_dev/analysis"
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame([dict(metric=m, **paired[m]) for m in METRICS]).to_csv(out / "paired_intervals.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame([dict(metric=m, **paired[m]) for m in METRICS]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
