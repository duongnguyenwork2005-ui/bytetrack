"""R71 preassociation center-error diagnostic. Reads only completed tracker output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from occlusion_eval import GT_COLS, TRK_COLS, iou_matrix, match_frame

BOX = ["x", "y", "w", "h"]
TRACK_COLS = ["frame", "track_id", "state", "is_activated", *BOX]
BINS = ["all", "0", "1-2", "3-8", ">=9", "no_prior"]
REASONS = ["scored", "no_prior_output", "previous_output_unlinked", "current_gt_missing"]


def read_mot(path, columns):
    """MOT output may legitimately be empty."""
    frame = pd.read_csv(path, names=columns, header=None)
    frame = frame.astype({"frame": "int64", "id": "int64"})
    return frame.sort_values(["frame", "id"], ignore_index=True)


def read_gt(path):
    gt = read_mot(path, GT_COLS)
    gt = gt.loc[(gt["conf"] > 0) & (gt["cls"] == 1)].reset_index(drop=True)
    if gt.duplicated(["frame", "id"]).any() or (gt[["w", "h"]] <= 0).any().any():
        raise ValueError(f"Invalid scored GT boxes or duplicate frame/ID in {path}")
    return gt


def frame_groups(frames):
    """Return zero-copy row slices for an array sorted by source frame."""
    unique, starts = np.unique(frames, return_index=True)
    return dict(zip(unique, (slice(start, end) for start, end in
                             zip(starts, np.r_[starts[1:], len(frames)]))))


def link_outputs(mot, gt):
    """One-to-one assignment on the previous emitted boxes, as in R65 lien_ket."""
    if mot.duplicated(["frame", "id"]).any():
        raise ValueError("Duplicate emitted track ID within a frame")
    boxes = mot[BOX].to_numpy(dtype=float)
    gt_boxes = gt[BOX].to_numpy(dtype=float)
    gt_groups = frame_groups(gt["frame"])
    gt_ids = gt["id"].to_numpy()
    linked = np.full(len(mot), np.nan)
    for frame, rows in frame_groups(mot["frame"]).items():
        gt_rows = gt_groups.get(frame)
        if gt_rows is None:
            continue
        for i, j in match_frame(iou_matrix(boxes[rows], gt_boxes[gt_rows]), 0.5):
            linked[rows.start + i] = gt_ids[gt_rows.start + j]
    return pd.DataFrame({"track_id": mot["id"], "prior_frame": mot["frame"], "gt_id": linked})


def samples(predictions, mot, gt):
    """Keep unlinked candidates. Never use the output from the prediction's own frame."""
    targets = predictions.loc[predictions["state"].isin([1, 2]) & predictions["is_activated"]]
    targets = targets.sort_values(["frame", "track_id"], ignore_index=True)
    if targets.duplicated(["frame", "track_id"]).any():
        raise ValueError("Duplicate candidate frame and track ID")
    joined = pd.merge_asof(targets, link_outputs(mot, gt), by="track_id",
                           left_on="frame", right_on="prior_frame", allow_exact_matches=False)
    joined = joined.merge(gt[["frame", "id", *BOX]].rename(columns={"id": "gt_id"}),
                          on=["frame", "gt_id"], how="left", suffixes=("", "_gt"),
                          validate="many_to_one")
    gap = joined["frame"] - joined["prior_frame"] - 1
    bucket = np.select([gap.eq(0), gap.between(1, 2), gap.between(3, 8), gap.ge(9)],
                       BINS[1:5], default="no_prior")
    reason = np.select([joined["prior_frame"].isna(), joined["gt_id"].isna(), joined["h_gt"].isna()],
                       REASONS[1:], default="scored")
    dx = joined["x"] - joined["x_gt"] + (joined["w"] - joined["w_gt"]) / 2
    dy = joined["y"] - joined["y_gt"] + (joined["h"] - joined["h_gt"]) / 2
    error = np.hypot(dx, dy) / joined["h_gt"]
    if not np.array_equal(np.isfinite(error), reason == "scored"):
        raise ValueError("Coverage classification disagrees with finite center errors")
    return pd.DataFrame({"gap": pd.Categorical(bucket, categories=BINS[1:]),
                         "reason": pd.Categorical(reason, categories=REASONS), "error": error})


def summarize(sample):
    groups = dict(tuple(sample.groupby("gap", observed=False)))
    groups["all"] = sample
    summary = []
    for gap in BINS:
        group = groups[gap]
        counts = group["reason"].value_counts()
        n = len(group)
        ns = int(counts["scored"])
        err = group["error"].dropna().to_numpy()
        p50, p90 = np.quantile(err, [0.5, 0.9]) if ns else (np.nan, np.nan)
        summary.append({"gap": gap, "eligible": n, **{r: int(counts[r]) for r in REASONS},
                        "coverage": ns / n if n else np.nan, "p50_error_per_h": p50,
                        "p90_error_per_h": p90})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    scope = "smoke" if args.smoke else "dev"
    exp = ROOT / "results/phase2" / f"kc48_r71_pf_{scope}"
    split = exp / "_processed" / "DETRAC-all"
    videos = json.loads((exp / "videos.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(exp / "runs.csv")
    if len(runs) != 9 or len(videos) != (2 if args.smoke else 12):
        raise ValueError("Unexpected R71 run manifest")
    gts = {v: read_gt(split / v / "gt/gt.txt") for v in videos}
    per_seed, per_video = [], []
    for run in runs.itertuples():
        folder = exp / "_processed/trackers/DETRAC-all" / run.dir
        fingerprint = json.loads((folder / "run_meta.json").read_text(encoding="utf-8"))["input_fingerprint"]
        seed = "fixed" if pd.isna(run.seed) else str(int(run.seed))
        blocks = []
        for video in videos:
            marker = json.loads((folder / "video_meta" / f"{video}.json").read_text(encoding="utf-8"))
            if not marker["complete"] or marker["input_fingerprint"] != fingerprint:
                raise ValueError(f"Incomplete or mismatched tracking output {folder.name}/{video}")
            mot = read_mot(folder / "data" / f"{video}.txt", TRK_COLS)
            tracks = pd.read_parquet(folder / "assoc_log" / f"{video}_tracks.parquet",
                                     columns=TRACK_COLS, engine="fastparquet")
            block = samples(tracks, mot, gts[video])
            blocks.append(block)
            per_video.extend(dict(filter=run.tracker, seed=seed, video=video, **row)
                             for row in summarize(block))
        per_seed.extend(dict(filter=run.tracker, seed=seed, **row)
                        for row in summarize(pd.concat(blocks, ignore_index=True)))
        print(f"{folder.name}: {sum(map(len, blocks))} eligible", flush=True)
    output = exp / "analysis"
    output.mkdir(exist_ok=True)
    seed_rows = pd.DataFrame(per_seed)
    seed_rows.to_csv(output / "prediction_by_seed.csv", index=False)
    pd.DataFrame(per_video).to_csv(output / "prediction_by_video.csv", index=False)
    aggregate = seed_rows.groupby(["filter", "gap"], sort=False).agg(
        n_seeds=("seed", "size"), seeds_with_scored=("p50_error_per_h", "count"),
        eligible_mean=("eligible", "mean"), scored_mean=("scored", "mean"),
        coverage_mean=("coverage", "mean"), p50_mean=("p50_error_per_h", "mean"),
        p50_min=("p50_error_per_h", "min"), p50_max=("p50_error_per_h", "max"),
        p90_mean=("p90_error_per_h", "mean"), p90_min=("p90_error_per_h", "min"),
        p90_max=("p90_error_per_h", "max"))
    aggregate.to_csv(output / "prediction_summary.csv")
    info = {"scope": scope, "runs": len(runs), "videos": len(videos),
            "units": len(runs)*len(videos),
            "target": "All activated Tracked or Lost boxes logged after prediction and before association.",
            "link": "Most recent earlier MOT output for the same track ID, one-to-one assignment to positive-conf class-1 GT with IoU >=0.5. No fallback to an earlier linked output if the latest output has no link.",
            "gap": "Current source-frame index minus last emitted MOT source-frame index minus one. Can include skipped frames and detector misses; not a label for physical occlusion.",
            "metric": "Euclidean center error divided by CURRENT GT box height. Quantiles pool scored predictions across videos separately per run. Summary gives arithmetic mean and min/max across available seed quantiles.",
            "coverage": "Counts classified for every eligible candidate. All includes predictions lacking a prior emitted output, shown separately in no_prior.",
            "selection": "Predictions survive tracking and need a previous confidently linked output and current GT. Different variants may score different examples. A smaller conditional error can coexist with more missing tracks."}
    (output / "prediction_method.json").write_text(json.dumps(info, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
