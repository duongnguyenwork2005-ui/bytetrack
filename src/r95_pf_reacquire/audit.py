"""Verify R95 MOT coverage and numeric integrity before using TrackEval scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "handoff/logs/review95_pf_reacquire"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("ua", "bdd"), required=True)
    args = parser.parse_args()
    if args.domain == "ua":
        split, part, manifest = "DETRAC-all", "dev", ROOT / "results/phase2/manifests/detrac_dev.json"
        reference = "r92-ua-dev-cv50"
    else:
        split, part = "BDDCOCOVID-val", "full"
        manifest = ROOT / "results/phase2/manifests/bdd100k_transfer_cocovid_val.json"
        reference = "r93-bdd-full-cv50"
    videos = json.loads(manifest.read_text(encoding="utf-8"))["videos"]
    assert len(videos) == len(set(videos))
    reports = []
    for seed in (0, 1, 2):
        root = ROOT / "data/processed/trackers" / split / f"r95-{args.domain}-{part}-rescue-s{seed}"
        old = ROOT / "data/processed/trackers" / split / f"{reference}-s{seed}"
        run = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
        assert run["status"] == "complete" and run["seed"] == seed
        assert run["tracker_cfg"]["pf_assoc_particle_rescue"] is True
        assert run["n_videos_ran"] + run["n_videos_reused"] == len(videos)
        assert {p.stem for p in (root / "data").glob("*.txt")} == set(videos)
        assert {p.stem for p in (root / "video_meta").glob("*.json")} == set(videos)
        frame_count = row_count = 0
        for video in videos:
            marker = json.loads((root / "video_meta" / f"{video}.json").read_text(encoding="utf-8"))
            old_marker = json.loads((old / "video_meta" / f"{video}.json").read_text(encoding="utf-8"))
            assert marker["complete"] is True and marker["video"] == video and marker["seed"] == seed
            assert marker["n_frames"] == old_marker["n_frames"]
            frame_count += marker["n_frames"]
            path = root / "data" / f"{video}.txt"
            if not path.stat().st_size:
                continue
            rows = np.loadtxt(path, delimiter=",", ndmin=2)
            assert rows.shape[1] == 7 and np.isfinite(rows).all(), path
            assert (rows[:, 0] >= 1).all() and (rows[:, 0] <= marker["n_frames"]).all(), path
            assert (rows[:, 1] >= 1).all() and (rows[:, 4:6] > 0).all(), path
            assert np.array_equal(rows[:, :2], np.rint(rows[:, :2])), path
            assert len(np.unique(rows[:, :2].astype(np.int64), axis=0)) == len(rows), path
            assert len(rows) == marker["n_boxes"], path
            row_count += len(rows)
        reports.append({"seed": seed, "videos": len(videos), "frames": frame_count, "mot_rows": row_count})
    assert len({r["frames"] for r in reports}) == 1
    if args.domain == "bdd":
        assert reports[0]["frames"] == 39973
    (OUT / f"{args.domain}_{part}_audit.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
