"""Replay the locked R74 BDD cache with one R78 Lost-recovery candidate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import boot_stats
import det_cache
import run_tracking
import run_trackeval
from r74_pf_bdd import CACHE, MANIFEST, SPLIT, SMOKE, preflight

OVERRIDES = {
    "iou3": ["lost_recover=iou", "lost_recover_iou=0.05", "lost_recover_max_lost=3"],
    "maha3": ["lost_recover=maha", "lost_recover_maha2=5.991", "lost_recover_max_lost=3"],
}


def videos(scope: str) -> list[str]:
    return SMOKE if scope == "smoke" else json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]


def track(scope: str, variant: str) -> None:
    det_cache.CACHE_ROOT = CACHE.parent
    sys.argv = [str(__file__), "--cache-id", "yolov8n_coco", "--split-name", SPLIT,
                "--tracker-yaml", "configs/phase2/trackers/r72/motion200.yaml",
                "--tracker-name", f"r78-bdd-{scope}-{variant}", "--videos", *videos(scope),
                "--conf-min", "0.25", "--classes", "1", "2", "3", "5", "7",
                "--max-det", "300", "--fps", "5", "--track-buffer-sec", "1.2",
                "--elapsed-mode", "steps", "--seeds", "0", "1", "2", "--skip-existing",
                "--set", *OVERRIDES[variant]]
    if run_tracking.main():
        raise RuntimeError("BDD tracking failed")


def evaluate(scope: str, variant: str) -> None:
    selected = videos(scope)
    for seed in (0, 1, 2):
        name = f"r78-bdd-{scope}-{variant}-s{seed}"
        results, _ = run_trackeval.run(SPLIT, name, selected, "mot_crowd")
        df = run_trackeval.flatten_results(results, name)
        combined = df[df.video == "COMBINED_SEQ"]
        per_video = df[df.video != "COMBINED_SEQ"].sort_values("video")
        if len(combined) != 1 or len(per_video) != len(selected):
            raise ValueError(f"incomplete evaluation: {name}")
        folder = ROOT / "results/trackeval" / SPLIT / name
        folder.mkdir(parents=True, exist_ok=True)
        combined.to_csv(folder / "combined_metrics.csv", index=False)
        per_video.to_csv(folder / "per_video_metrics.csv", index=False)
        raw = boot_stats.save_raw_fields(results, name, folder / "raw_fields.json")
        score = boot_stats.combine(raw, selected)
        if max(abs(score[m] - float(combined.iloc[0][m])) for m in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")) > 1e-9:
            raise ValueError(f"COMBINED mismatch: {name}")
        print(name, {k: score[k] for k in ("HOTA", "IDF1", "IDSW")}, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("smoke", "full"), required=True)
    ap.add_argument("--stage", choices=("track", "eval"), required=True)
    ap.add_argument("--variant", choices=tuple(OVERRIDES), required=True)
    args = ap.parse_args()
    preflight(json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"])
    if args.stage == "track":
        track(args.scope, args.variant)
    else:
        evaluate(args.scope, args.variant)


if __name__ == "__main__":
    main()
