"""R81 audit of antithetic PF process noise against paired R72 UA runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kc48 import boot_seed
from r78_pf.analyze import METRICS, SPLIT, VIDEOS, load


def main() -> None:
    for seed in (0, 1, 2):
        old_smoke = ROOT / "results/trackeval" / SPLIT / f"p2-kc48_r72_pf_smoke-A-motion200-s{seed}" / "raw_fields.json"
        new_smoke = ROOT / "results/trackeval" / SPLIT / f"p2-kc48_r81_pf_smoke-A-baseline-s{seed}" / "raw_fields.json"
        if json.loads(old_smoke.read_text(encoding="utf-8")) != json.loads(new_smoke.read_text(encoding="utf-8")):
            raise ValueError(f"smoke raw fields differ: seed {seed}")
    old, new, rows = [], [], []
    fingerprint = None
    code = None
    for seed in (0, 1, 2):
        base = f"p2-kc48_r72_pf_dev-A-motion200-s{seed}"
        candidate = f"p2-kc48_r81_pf_dev-A-anti-s{seed}"
        old_raw, old_score, old_source = load(base)
        new_raw, new_score, new_source = load(candidate)
        if old_source != new_source or (fingerprint is not None and new_source != fingerprint):
            raise ValueError(f"source fingerprint mismatch: seed {seed}")
        fingerprint = new_source
        tracker = ROOT / "data/processed/trackers" / SPLIT / candidate
        meta = json.loads((tracker / "run_meta.json").read_text(encoding="utf-8"))
        cfg = meta["tracker_cfg"]
        if (meta["status"] != "complete" or meta["videos"] != VIDEOS or
                meta["seed"] != seed or cfg.get("pf_antithetic_process_noise") is not True or
                cfg.get("pf_roughening_motion_scale") != 2):
            raise ValueError(f"run metadata mismatch: {candidate}")
        if code is None:
            code = meta["code"]
        elif code != meta["code"]:
            raise ValueError(f"code fingerprint mismatch: {candidate}")
        for video in VIDEOS:
            marker = json.loads((tracker / "video_meta" / f"{video}.json").read_text(encoding="utf-8"))
            if not marker["complete"] or marker["input_fingerprint"] != meta["input_fingerprint"]:
                raise ValueError(f"video marker mismatch: {candidate}/{video}")
        old.append(old_raw)
        new.append(new_raw)
        rows.extend((dict(kind=kind, seed=seed, **{m: score[m] for m in METRICS})
                     for kind, score in (("baseline", old_score), ("antithetic", new_score))))
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("kind", sort=False)[list(METRICS)].mean().reset_index()
    pairs = boot_seed.paired_bootstrap_seed(new, old, VIDEOS, n_boot=5000, seed=0, metrics=METRICS)
    out = ROOT / "results/phase2/kc48_r81_pf_dev/analysis"
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame([dict(metric=m, **pairs[m]) for m in METRICS]).to_csv(out / "paired_intervals.csv", index=False)
    (out / "source_fingerprint.json").write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    (out / "smoke_validation.json").write_text(json.dumps({"baseline_raw_equals_R72": True,
                                                             "seeds": [0, 1, 2], "videos": ["MVI_20052", "MVI_20062"],
                                                             "old_MOT_available": False}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame([dict(metric=m, **pairs[m]) for m in METRICS]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
