"""Replay R61 BDD cache for the R75 recentered PF candidate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import det_cache
import run_tracking
import run_trackeval
import boot_stats
from r74_pf_bdd import CACHE, MANIFEST, SPLIT, SMOKE, preflight


def selected(scope: str) -> list[str]:
    return SMOKE if scope == "smoke" else json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]


def track(scope: str) -> None:
    videos = selected(scope)
    det_cache.CACHE_ROOT = CACHE.parent
    sys.argv = [str(__file__), "--cache-id", "yolov8n_coco", "--split-name", SPLIT,
                "--tracker-yaml", "configs/phase2/trackers/r75/recenter.yaml",
                "--tracker-name", f"r75-bdd-{scope}-recenter", "--videos", *videos,
                "--conf-min", "0.25", "--classes", "1", "2", "3", "5", "7",
                "--max-det", "300", "--fps", "5", "--track-buffer-sec", "1.2",
                "--elapsed-mode", "steps", "--seeds", "0", "1", "2", "--skip-existing"]
    if run_tracking.main():
        raise RuntimeError("BDD tracking failed")


def evaluate(scope: str) -> None:
    videos = selected(scope)
    for seed in (0, 1, 2):
        name = f"r75-bdd-{scope}-recenter-s{seed}"
        results, _ = run_trackeval.run(SPLIT, name, videos, "mot_crowd")
        df = run_trackeval.flatten_results(results, name)
        combined = df[df.video == "COMBINED_SEQ"]
        per_video = df[df.video != "COMBINED_SEQ"].sort_values("video")
        if len(combined) != 1 or len(per_video) != len(videos):
            raise ValueError(f"incomplete evaluation: {name}")
        folder = ROOT / "results/trackeval" / SPLIT / name
        folder.mkdir(parents=True, exist_ok=True)
        combined.to_csv(folder / "combined_metrics.csv", index=False)
        per_video.to_csv(folder / "per_video_metrics.csv", index=False)
        raw = boot_stats.save_raw_fields(results, name, folder / "raw_fields.json")
        score = boot_stats.combine(raw, videos)
        if abs(score["HOTA"] - float(combined.iloc[0].HOTA)) > 1e-9:
            raise ValueError(f"COMBINED mismatch: {name}")
        print(name, {k: score[k] for k in ("HOTA", "IDF1", "IDSW")}, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["smoke", "full"], required=True)
    ap.add_argument("--stage", choices=["track", "eval"], required=True)
    args = ap.parse_args()
    preflight(json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"])
    if args.stage == "track":
        track(args.scope)
    else:
        evaluate(args.scope)


if __name__ == "__main__":
    main()
