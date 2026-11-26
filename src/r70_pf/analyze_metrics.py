"""Verify the R70 run matrix, reproduce TrackEval scores and compute paired intervals."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import boot_stats
import run_fingerprint as fp
from kc48 import boot_seed

SPLIT = "DETRAC-all"
METRICS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW"]
RECIPES = ["pf500", "pf500-k1", "pf500-k1k2"]
PF_NAMES = RECIPES + [f"{r}-rpred" for r in RECIPES]
REFERENCES = ["cv", "ekf", "ukf"]
TE = ROOT / "results/trackeval" / SPLIT
PHASE = ROOT / "results/phase2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(experiment, name, seed, videos):
    run = f"p2-{experiment}-A-{name}" + ("" if seed is None else f"-s{seed}")
    tracker = PHASE / experiment / "_processed/trackers" / SPLIT / run
    evaluation = TE / run
    meta = read_json(tracker / "run_meta.json")
    require(meta["status"] == "complete" and meta["videos"] == videos,
            f"Incomplete or unexpected sequence list: {run}")
    require(meta["tracker"] == name and meta["seed"] == seed and meta["experiment"] == experiment,
            f"Incorrect tracker identity: {run}")
    require({p.stem for p in (tracker / "video_meta").glob("*.json")} == set(videos),
            f"Coverage mismatch: {run}")
    for video in videos:
        marker = read_json(tracker / "video_meta" / f"{video}.json")
        require(marker["video"] == video and marker["complete"] and
                marker["input_fingerprint"] == meta["input_fingerprint"] and marker["n_frames"] > 0,
                f"Incomplete or inconsistent video marker: {run}/{video}")
    hashes = fp.tracking_fingerprint(tracker, videos)
    eval_meta = read_json(evaluation / "eval_meta.json")
    require(hashes["n_missing"] == 0 and eval_meta["tracking"] == hashes and
            eval_meta["evaluator"]["dataset_class"] == "mot" and eval_meta["dir"] == run,
            f"MOT file fingerprint or evaluator mismatch: {run}")
    raw = boot_stats.load_raw_fields(evaluation / "raw_fields.json")
    require(set(raw) == set(videos), f"Raw field sequence mismatch: {run}")
    combined_csv = pd.read_csv(evaluation / "combined_metrics.csv")
    require(combined_csv.video.tolist() == ["COMBINED_SEQ"], f"Combined score missing: {run}")
    combined = boot_stats.combine(raw, videos)
    reported = combined_csv.iloc[0]
    diffs = [abs(combined[m] - float(reported[m])) for m in METRICS]
    require(np.all(np.isfinite(diffs)) and max(diffs) < 1e-9, f"Recomposition failed: {run}, {diffs}")
    return raw, combined, max(diffs), eval_meta


def historical_parity(name, seed, videos, scores, new_meta):
    run = f"p2-kc48_pf_dev-A-{name}" + ("" if seed is None else f"-s{seed}")
    evaluation = TE / run
    old_meta = read_json(evaluation / "eval_meta.json")
    old_scores = boot_stats.combine(boot_stats.load_raw_fields(evaluation / "raw_fields.json"), videos)
    changed = [v for v in videos if old_meta["tracking"]["per_video"][v] != new_meta["tracking"]["per_video"][v]]
    return dict(tracker=name, seed=seed, videos=len(videos), mot_identical=len(videos)-len(changed),
                changed_videos=changed,
                gt_identical=all(old_meta["gt"]["gt_per_video"][v] == new_meta["gt"]["gt_per_video"][v]
                                 for v in videos),
                detector_meta_identical=old_meta["cache"]["meta_sha256"] == new_meta["cache"]["meta_sha256"],
                score_deltas={m: scores[m] - old_scores[m] for m in METRICS})


def analyze(smoke=False):
    experiment = "kc48_r70_pf_smoke" if smoke else "kc48_r70_pf_dev"
    config = yaml.safe_load((ROOT / "configs/phase2/experiments" / f"{experiment}.yaml").read_text(encoding="utf-8"))
    videos = config["videos"] if smoke else read_json(ROOT / config["manifest"])["videos"]
    require(len(set(videos)) == len(videos) == (2 if smoke else 12), "Unexpected manifest")
    all_raw = {}
    per_seed, history, summary = [], [], []
    for name in REFERENCES + PF_NAMES:
        raws = []
        for seed in ([0, 1, 2] if name in PF_NAMES else [None]):
            raw, scores, max_diff, eval_meta = load_run(experiment, name, seed, videos)
            if not smoke:
                # Historical helper has an epsilon in the AssA numerator. All
                # positive TP counts guarantee no bootstrap draw is all-zero.
                require(all(min(row["HOTA_TP"]) > 0 for row in raw.values()),
                        f"Review zero-TP AssA bootstrap handling before reporting {name}/{seed}")
            raws.append(raw)
            per_seed.append(dict(tracker=name, seed=seed, n_videos=len(videos),
                                 max_recomposition_error=max_diff, **{m: scores[m] for m in METRICS}))
            if name in ["ekf", "pf500", "pf500-k1"]:
                history.append(historical_parity(name, seed, videos, scores, eval_meta))
        all_raw[name] = raws
        values = pd.DataFrame([r for r in per_seed if r["tracker"] == name])[METRICS]
        by_metric = boot_seed.combine_tb_seed(raws, videos, METRICS)
        summary.append(dict(tracker=name, n_seeds=len(raws), **by_metric,
                            **{f"{m}_min": values[m].min() for m in METRICS},
                            **{f"{m}_max": values[m].max() for m in METRICS}))
    out = PHASE / experiment / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_seed).to_csv(out / "per_seed.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    if not smoke:
        comparisons = [("pf500-k1", "pf500"), ("pf500-k1k2", "pf500-k1")]
        comparisons += [(f"{name}-rpred", name) for name in RECIPES]
        comparisons += [(name, reference) for name in PF_NAMES for reference in ["cv", "ekf"]]
        intervals = []
        for a, b in comparisons:
            result = boot_seed.paired_bootstrap_seed(all_raw[a], all_raw[b], videos,
                                                     n_boot=5000, seed=0, metrics=METRICS)
            for metric in METRICS:
                intervals.append(dict(tracker=a, reference=b, metric=metric, **result[metric]))
        pd.DataFrame(intervals).to_csv(out / "paired_intervals.csv", index=False)
    validation = dict(experiment=experiment, complete_runs=len(per_seed), n_videos=len(videos),
                      verified_video_markers=len(per_seed)*len(videos),
                      max_recomposition_error=max(r["max_recomposition_error"] for r in per_seed),
                      historical_r48=history)
    (out / "metrics_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(f"{experiment}: validated {len(per_seed)} runs over {len(videos)} sequences; output {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    analyze(parser.parse_args().smoke)
