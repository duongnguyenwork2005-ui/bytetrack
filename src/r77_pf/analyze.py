"""Audit R77 PF motion scales 6/8 vs R72 motion-2 on UA dev12."""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
    raw_by_kind = {k: [] for k in ("motion200", "motion600", "motion800")}
    rows = []
    code = None
    source_fingerprint = None
    for kind in raw_by_kind:
        for seed in (0, 1, 2):
            run = (f"p2-kc48_r72_pf_dev-A-motion200-s{seed}" if kind == "motion200"
                   else f"p2-kc48_r77_pf_dev-A-{kind}-s{seed}")
            raw, score = load(run)
            eval_meta = json.loads((ROOT / "results/trackeval/DETRAC-all" / run / "eval_meta.json").read_text(encoding="utf-8"))
            source = {key: eval_meta[key] for key in ("gt", "cache", "evaluator")}
            if source_fingerprint is None:
                source_fingerprint = source
            elif source != source_fingerprint:
                raise ValueError(f"GT/cache/evaluator fingerprint mismatch: {run}")
            if kind != "motion200":
                tracker = ROOT / "data/processed/trackers/DETRAC-all" / run
                meta = json.loads((tracker / "run_meta.json").read_text(encoding="utf-8"))
                if (meta["status"] != "complete" or meta["videos"] != VIDEOS or
                        meta["seed"] != seed or
                        meta["tracker_cfg"]["pf_roughening_motion_scale"] != int(kind[-3:]) / 100 or
                        meta["tracker_cfg"]["pf_allow_high_motion_roughening"] is not True or
                        meta["tracker_cfg"].get("pf_recenter_resample", False) is not False):
                    raise ValueError(f"run metadata mismatch: {run}")
                if code is None:
                    code = meta["code"]
                elif meta["code"] != code:
                    raise ValueError(f"code fingerprint differs: {run}")
                for video in VIDEOS:
                    marker = json.loads((tracker / "video_meta" / f"{video}.json").read_text(encoding="utf-8"))
                    if not marker["complete"] or marker["input_fingerprint"] != meta["input_fingerprint"]:
                        raise ValueError(f"marker mismatch: {run}/{video}")
            raw_by_kind[kind].append(raw)
            rows.append(dict(kind=kind, seed=seed, **{m: score[m] for m in METRICS}))
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("kind", sort=False)[METRICS].mean().reset_index()
    intervals = []
    for kind in ("motion600", "motion800"):
        pair = boot_seed.paired_bootstrap_seed(raw_by_kind[kind], raw_by_kind["motion200"],
                                               VIDEOS, n_boot=5000, seed=0, metrics=METRICS)
        intervals.extend(dict(kind=kind, metric=m, **pair[m]) for m in METRICS)
    out = ROOT / "results/phase2/kc48_r77_pf_dev/analysis"
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(intervals).to_csv(out / "paired_intervals.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(intervals).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
