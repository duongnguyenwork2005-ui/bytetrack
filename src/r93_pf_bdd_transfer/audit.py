"""Check completeness and MOT numeric integrity of the R93 BDD replay."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPLIT = "BDDCOCOVID-val"
MANIFEST = ROOT / "results/phase2/manifests/bdd100k_transfer_cocovid_val.json"
OUT = ROOT / "handoff/logs/review93_pf_bdd_transfer/full_audit.json"


def main():
    videos = json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]
    assert len(videos) == len(set(videos)) == 200
    reports = []
    for seed in (0, 1, 2):
        name = f"r93-bdd-full-cv50-s{seed}"
        base = ROOT / "data/processed/trackers" / SPLIT / name
        meta = json.loads((base / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["status"] == "complete" and meta["seed"] == seed
        assert meta["n_videos_ran"] + meta["n_videos_reused"] == 200
        assert len(meta["videos"]) == 200
        assert {p.stem for p in (base / "data").glob("*.txt")} == set(videos)
        assert {p.stem for p in (base / "video_meta").glob("*.json")} == set(videos)
        n_rows = n_frames = 0
        for video in videos:
            marker = json.loads((base / "video_meta" / f"{video}.json").read_text(encoding="utf-8"))
            assert marker["complete"] is True and marker["seed"] == seed
            assert marker["video"] == video and marker["n_frames"] > 0
            n_frames += int(marker["n_frames"])
            path = base / "data" / f"{video}.txt"
            if path.stat().st_size == 0:
                continue
            rows = np.loadtxt(path, delimiter=",", ndmin=2)
            assert rows.shape[1] == 7 and np.isfinite(rows).all(), path
            assert (rows[:, 0] >= 1).all() and (rows[:, 0] <= marker["n_frames"]).all(), path
            assert (rows[:, 1] >= 1).all() and (rows[:, 2:6][:, 2:4] > 0).all(), path
            assert np.allclose(rows[:, :2], np.rint(rows[:, :2])), path
            pairs = rows[:, :2].astype(np.int64)
            assert len(np.unique(pairs, axis=0)) == len(pairs), path
            assert len(rows) == marker["n_boxes"], path
            n_rows += len(rows)
        reports.append({"seed": seed, "videos": len(videos), "frames": n_frames, "mot_rows": n_rows})
    assert {r["frames"] for r in reports} == {39973}, reports
    OUT.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
