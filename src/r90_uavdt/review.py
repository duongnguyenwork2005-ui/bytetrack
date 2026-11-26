"""Independent artifact checks and compact R90 vehicle-only summary."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import boot_stats
import config
import det_cache
from r90_uavdt.audit import EVIDENCE, SOURCE
from r90_uavdt.cache import CACHE
from r90_uavdt.prepare import SPLIT

FILTERS = ("cv", "ekf", "ukf", "pf_default", "pf_motion100")
METRICS = ("HOTA", "IDF1", "MOTA", "IDSW")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    selection = json.loads((EVIDENCE / "selection.json").read_text(encoding="utf-8"))["all"]
    labels = json.loads((EVIDENCE / "label_audit.json").read_text(encoding="utf-8"))
    gt_manifest = json.loads((EVIDENCE / "gt_manifest.json").read_text(encoding="utf-8"))
    progress = json.loads((EVIDENCE / "matrix_progress.json").read_text(encoding="utf-8"))
    download = json.loads((SOURCE / "download_manifest.json").read_text(encoding="utf-8"))
    assert len(selection) == 8 and len(progress) == 10
    assert all(cell["status"] == "done" for cell in progress.values())
    assert all(rec["crc_error_member"] is None for rec in download.values())
    assert (sum(labels[s]["frames"] for s in selection),
            sum(labels[s]["eligible_vehicle_rows"] for s in selection)) == (4367, 59472)
    gt_rows = 0
    cache_boxes = 0
    for seq in selection:
        gt = config.PROCESSED_DIR / SPLIT / seq / "gt/gt.txt"
        info = config.PROCESSED_DIR / SPLIT / seq / "seqinfo.ini"
        assert sha(gt) == gt_manifest["sequences"][seq]["processed_gt_sha256"]
        assert sha(info) == gt_manifest["sequences"][seq]["seqinfo_sha256"]
        rows = np.loadtxt(gt, delimiter=",", ndmin=2)
        assert set(np.unique(rows[:, 7])).issubset({1, 8})
        assert len(np.unique(rows[:, :2].astype(int), axis=0)) == len(rows)
        gt_rows += int(np.sum(rows[:, 7] == 1))
        with np.load(det_cache.CACHE_ROOT / CACHE / f"{seq}.npz") as z:
            n = labels[seq]["frames"]
            assert np.array_equal(z["frame_idx"], np.arange(1, n + 1))
            assert np.allclose(z["time_s"], np.arange(n) / 30, atol=1e-12)
            assert np.array_equal(z["orig_shape"], [540, 1024])
            assert not z["unreadable"].any() and not z["truncated"].any()
            assert int(z["n_per_frame"].sum()) == len(z["data"])
            cache_boxes += len(z["data"])
    assert gt_rows == 59472
    cells = {}
    max_metric_error = 0.0
    for conf in ("c25", "c10"):
        for filt in FILTERS:
            seeds = (None,) if not filt.startswith("pf") else (0, 1, 2)
            seed_metrics = []
            per_video = []
            for seed in seeds:
                name = f"r90-{conf}-{filt}" + (f"-s{seed}" if seed is not None else "")
                base = config.RESULTS_DIR / "trackeval" / SPLIT / name
                combined = pd.read_csv(base / "combined_metrics.csv").iloc[0]
                individual = pd.read_csv(base / "per_video_metrics.csv").set_index("video")
                assert set(individual.index) == set(selection), name
                raw = boot_stats.load_raw_fields(base / "raw_fields.json")
                assert set(raw) == set(selection), name
                rebuilt = boot_stats.combine(raw, selection)
                for metric in METRICS:
                    err = abs(float(combined[metric]) - float(rebuilt[metric]))
                    max_metric_error = max(max_metric_error, err)
                    assert err < 1e-9, (name, metric, err)
                run = json.loads((config.PROCESSED_DIR / "trackers" / SPLIT / name / "run_meta.json").read_text())
                assert run["fps"] == 30 and run["track_buffer_frames"] == 36
                assert run["classes"] == [2, 5, 7] and run["cache_id"] == CACHE
                assert run["elapsed_mode"] == "steps"
                for seq in selection:
                    path = config.PROCESSED_DIR / "trackers" / SPLIT / name / "data" / f"{seq}.txt"
                    assert path.is_file(), (name, seq)
                    if path.stat().st_size:
                        tracks = np.loadtxt(path, delimiter=",", ndmin=2)
                        assert tracks.shape[1] == 7 and np.isfinite(tracks).all(), (name, seq)
                        assert len(np.unique(tracks[:, :2].astype(int), axis=0)) == len(tracks), (name, seq)
                        assert (tracks[:, 0] >= 1).all() and (tracks[:, 0] <= labels[seq]["frames"]).all()
                        assert (tracks[:, 4:6] > 0).all()
                seed_metrics.append({metric: float(combined[metric]) for metric in METRICS})
                per_video.append(individual)
            cells[f"{conf}/{filt}"] = {
                "seeds": list(seeds),
                "mean_combined_over_seeds": {metric: float(np.mean([r[metric] for r in seed_metrics]))
                                              for metric in METRICS},
                "seed_combined": seed_metrics,
                "per_video_mean": {seq: {metric: float(np.mean([float(df.loc[seq, metric]) for df in per_video]))
                                         for metric in METRICS} for seq in selection},
            }
    summary = {"dataset": "UAVDT official vehicle MOT train8 internal transfer test",
        "selected_sequences": selection, "frames": 4367, "vehicle_gt_rows": gt_rows,
        "vehicle_gt_ids_per_sequence_sum": sum(labels[s]["vehicle_ids"] for s in selection),
        "cache_boxes_all_coco_classes": cache_boxes,
        "detector_smoke": json.loads((EVIDENCE / "detector_smoke.json").read_text()),
        "detector_full_recall": json.loads((EVIDENCE / "full_detector_recall.json").read_text()),
        "cells": cells}
    (EVIDENCE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    audit = {"status": "PASS", "source_crc": "PASS", "source_files": len(download),
        "gt_rows": gt_rows, "frames": 4367, "cache_boxes": cache_boxes,
        "runs": sum(1 if not f.startswith("pf") else 3 for f in FILTERS) * 2,
        "evaluated_video_seed_units": sum(1 if not f.startswith("pf") else 3 for f in FILTERS) * 2 * len(selection),
        "max_combined_raw_metric_error": max_metric_error,
        "known_limits": ["Train GT and detector transfer test, not hidden test or official leaderboard",
            "M0501 c25 yields zero tracks; overall scores are detector-limited",
            "MOT class ID 1 is numerical alias for vehicle in the project evaluator",
            "Project TrackEval distractor handling for gt_ignore differs from publisher MATLAB full-containment rule",
            "Cache script calls detect_frame directly on unmasked images; default mask_ignored_regions field in cache meta is not applied by this route"]}
    (EVIDENCE / "final_review.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
