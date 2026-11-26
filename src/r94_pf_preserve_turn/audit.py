"""Verify R94 full MOT output, markers, frame count and unique IDs per frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "handoff/logs/review94_pf_preserve_turn"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("ua", "bdd"), required=True)
    args = parser.parse_args()
    if args.domain == "ua":
        split, part, manifest = "DETRAC-all", "dev", ROOT / "results/phase2/manifests/detrac_dev.json"
        baseline = "r92-ua-dev-cv50"
    else:
        split, part = "BDDCOCOVID-val", "full"
        manifest = ROOT / "results/phase2/manifests/bdd100k_transfer_cocovid_val.json"
        baseline = "r93-bdd-full-cv50"
    videos = json.loads(manifest.read_text(encoding="utf-8"))["videos"]
    assert len(videos) == len(set(videos))
    reports = []
    for seed in (0, 1, 2):
        base = ROOT / "data/processed/trackers" / split / f"r94-{args.domain}-{part}-preserve-s{seed}"
        prior = ROOT / "data/processed/trackers" / split / f"{baseline}-s{seed}"
        meta = json.loads((base / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["status"] == "complete" and meta["seed"] == seed
        assert meta["n_videos_ran"] + meta["n_videos_reused"] == len(videos)
        assert set(p.stem for p in (base / "data").glob("*.txt")) == set(videos)
        assert set(p.stem for p in (base / "video_meta").glob("*.json")) == set(videos)
        frames = mot_rows = 0
        for video in videos:
            marker = json.loads((base / "video_meta" / f"{video}.json").read_text(encoding="utf-8"))
            prior_marker = json.loads((prior / "video_meta" / f"{video}.json").read_text(encoding="utf-8"))
            assert marker["complete"] is True and marker["seed"] == seed
            assert marker["video"] == video and marker["n_frames"] == prior_marker["n_frames"]
            frames += int(marker["n_frames"])
            path = base / "data" / f"{video}.txt"
            if path.stat().st_size == 0:
                continue
            rows = np.loadtxt(path, delimiter=",", ndmin=2)
            assert rows.shape[1] == 7 and np.isfinite(rows).all(), path
            assert (rows[:, 0] >= 1).all() and (rows[:, 0] <= marker["n_frames"]).all(), path
            assert (rows[:, 1] >= 1).all() and (rows[:, 4:6] > 0).all(), path
            assert np.array_equal(rows[:, :2], np.rint(rows[:, :2])), path
            assert len(np.unique(rows[:, :2].astype(np.int64), axis=0)) == len(rows), path
            assert len(rows) == marker["n_boxes"], path
            mot_rows += len(rows)
        reports.append({"seed": seed, "videos": len(videos), "frames": frames, "mot_rows": mot_rows})
    assert len({r["frames"] for r in reports}) == 1
    if args.domain == "bdd":
        assert reports[0]["frames"] == 39973
    (OUT / f"{args.domain}_full_audit.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
