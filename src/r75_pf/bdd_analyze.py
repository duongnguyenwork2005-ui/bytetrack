"""Audit and compare R75 BDD candidate with the paired R74 PF motion-2 run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import boot_stats
from kc48 import boot_seed

SPLIT = "BDDCOCOVID-val"
VIDEOS = json.loads((ROOT / "results/phase2/manifests/bdd100k_transfer_cocovid_val.json").read_text(encoding="utf-8"))["videos"]
METRICS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW"]


def load(name: str) -> tuple[dict, dict]:
    folder = ROOT / "results/trackeval" / SPLIT / name
    raw = boot_stats.load_raw_fields(folder / "raw_fields.json")
    if set(raw) != set(VIDEOS):
        raise ValueError(f"raw coverage mismatch: {name}")
    score = boot_stats.combine(raw, VIDEOS)
    csv = pd.read_csv(folder / "combined_metrics.csv").iloc[0]
    if max(abs(score[m] - float(csv[m])) for m in METRICS) > 1e-9:
        raise ValueError(f"COMBINED mismatch: {name}")
    return raw, score


def main() -> None:
    old, new, rows = [], [], []
    code = None
    for seed in (0, 1, 2):
        old_name = f"r74-bdd-full-motion200-s{seed}"
        new_name = f"r75-bdd-full-recenter-s{seed}"
        old_raw, old_score = load(old_name)
        new_raw, new_score = load(new_name)
        tracker = ROOT / "data/processed/trackers" / SPLIT / new_name
        meta = json.loads((tracker / "run_meta.json").read_text(encoding="utf-8"))
        if (meta["status"] != "complete" or meta["seed"] != seed or
                meta["tracker_cfg"]["pf_recenter_resample"] is not True or
                meta["cache_id"] != "yolov8n_coco" or meta["conf_min"] != .25 or
                meta["classes"] != [1, 2, 3, 5, 7] or meta["fps"] != 5 or
                meta["track_buffer_frames"] != 6 or meta["elapsed_mode"] != "steps"):
            raise ValueError(f"new metadata mismatch: {new_name}")
        if code is None:
            code = meta["code"]
        elif meta["code"] != code:
            raise ValueError(f"code fingerprint differs: {new_name}")
        for video in VIDEOS:
            marker = json.loads((tracker / "video_meta" / f"{video}.json").read_text(encoding="utf-8"))
            if not marker["complete"] or marker["input_fingerprint"] != meta["input_fingerprint"]:
                raise ValueError(f"tracking marker mismatch: {new_name}/{video}")
        old.append(old_raw)
        new.append(new_raw)
        for kind, score in (("motion200", old_score), ("recenter", new_score)):
            rows.append(dict(kind=kind, seed=seed, **{m: score[m] for m in METRICS}))
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("kind", sort=False)[METRICS].mean().reset_index()
    paired = boot_seed.paired_bootstrap_seed(new, old, VIDEOS, n_boot=5000, seed=0, metrics=METRICS)
    out = ROOT / "results/phase2/r75_pf_bdd/analysis"
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame([dict(metric=m, **paired[m]) for m in METRICS]).to_csv(out / "paired_intervals.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame([dict(metric=m, **paired[m]) for m in METRICS]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
