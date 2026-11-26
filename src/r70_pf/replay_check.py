"""Repeat the R70 smoke replay, or time the nine recipes on one fixed dev segment."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
REPORTS = ROOT / "handoff/logs/review70_pf"
REPEAT_RECIPE = "pf500-k1k2-rpred"
SPEED_VIDEO = "MVI_20052"
WARMUP = 32
MEASURE = 128
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def tracker_cfg(row, fps):
    from run_tracking import load_tracker_cfg
    from time_units import track_buffer_frames

    settings = dict(row["set"])
    if row["track_buffer_sec"] is not None:
        settings["track_buffer"] = track_buffer_frames(row["track_buffer_sec"], fps)
    return load_tracker_cfg(row["yaml"], settings)


def load_detections(exp, row, video):
    from det_cache import open_cache

    return open_cache(exp["cache_id"], video, row["conf_min"], exp["classes"], exp["max_det"])


def smoke_repeat(exp, rows):
    from run_tracking import track_video

    row, = [r for r in rows if r["tracker"] == REPEAT_RECIPE and r["seed"] == 0]
    folder = ROOT / "results/phase2" / exp["name"] / "_processed/trackers" / exp["split_name"] / row["dir"]
    cfg = tracker_cfg(row, exp["fps"])
    records = []
    for video in exp["videos"]:
        original = folder / "data" / f"{video}.txt"
        meta = json.loads((folder / "video_meta" / f"{video}.json").read_text(encoding="utf-8"))
        if not meta["complete"] or meta["seed"] != row["seed"] or meta["tracker"] != REPEAT_RECIPE:
            raise ValueError(f"Unexpected smoke marker for {video}")
        vd = load_detections(exp, row, video)
        output, _, _ = track_video(vd, cfg, 1. / exp["fps"], exp["elapsed_mode"], row["seed"], video,
                                   log_sink={"assoc_log": exp["assoc_log"], "pf_diag": exp["pf_diag"]})
        buffer = io.BytesIO()
        output.to_csv(buffer, header=False, index=False)
        expected = original.read_bytes()
        actual = buffer.getvalue()
        records.append(dict(video=video, expected_file=str(original.relative_to(ROOT)),
                            bytes=len(actual), mot_rows=len(output), passed=expected == actual,
                            original_sha256=hashlib.sha256(expected).hexdigest(),
                            repeat_sha256=hashlib.sha256(actual).hexdigest()))
    return dict(passed=all(r["passed"] for r in records), seed=0, recipe=REPEAT_RECIPE, comparisons=records)


def speed(exp, rows):
    from run_tracking import track_video
    import torch
    import cv2
    from ul_backend import backend_name

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    variants = [r for r in rows if r["seed"] in (None, 0)]
    if len(variants) != 9 or len({r["tracker"] for r in variants}) != 9:
        raise ValueError("Expected nine distinct deterministic/seed-0 variants")
    vd = load_detections(exp, variants[0], SPEED_VIDEO)
    frames = vd.frame_idx[~vd.unreadable.astype(bool)][:WARMUP + MEASURE]
    if len(frames) != WARMUP + MEASURE:
        raise ValueError("Insufficient readable frames for the fixed timing segment")
    cfgs = {r["tracker"]: tracker_cfg(r, exp["fps"]) for r in variants}
    shifted = variants[4:] + variants[:4]
    orders = [variants, variants[::-1], shifted, shifted[::-1]]
    results = []
    for repeat, order in enumerate(orders, 1):
        for pos, row in enumerate(order, 1):
            name = row["tracker"]
            _, _, times = track_video(vd, cfgs[name], 1. / exp["fps"], exp["elapsed_mode"],
                                      row["seed"], SPEED_VIDEO,
                                      on_frame=lambda fr, out: fr < int(frames[-1]))
            if times.frame.tolist() != frames.tolist():
                raise ValueError(f"Timing replay used different frames for {name}")
            elapsed = times.iloc[WARMUP:]
            results.append(dict(recipe=name, repeat=repeat, order=pos,
                                ms_per_frame=float(elapsed.t_update_ms.mean()),
                                detections=int(elapsed.n_dets.sum())))
    if len({r["detections"] for r in results}) != 1:
        raise ValueError("Detection workload changed between timing runs")
    summaries = {}
    for row in variants:
        name = row["tracker"]
        samples = [r["ms_per_frame"] for r in results if r["recipe"] == name]
        summaries[name] = dict(median_ms_per_frame=median(samples), min_ms_per_frame=min(samples),
                               max_ms_per_frame=max(samples))
    return dict(video=SPEED_VIDEO, cache=exp["cache_id"], seed_pf=0, backend=backend_name(),
                python=sys.version.split()[0], cpu=platform.processor(), opencv_threads=cv2.getNumThreads(),
                thread_env={key: os.environ[key] for key in THREAD_KEYS},
                torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                warmup_readable_frames=WARMUP, measured_readable_frames=MEASURE,
                first_measured_frame=int(frames[WARMUP]), last_measured_frame=int(frames[-1]),
                repeats=len(orders), statistic="Median of four replay means of tracker.update ms per frame",
                scope="One CPU process, one thread per numerical library, no PF/association diagnostic logs. "
                      "Time around tracker.update includes predict/association/update. "
                      "Detector, detection conversion, cache reads, MOT serialization and file I/O are excluded.",
                summary=summaries, runs=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--smoke-repeat", action="store_true")
    options.add_argument("--speed", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.speed:
        for key in THREAD_KEYS:
            os.environ[key] = "1"
    import yaml
    from run_experiment import plan
    from run_state import atomic_write_json

    split = "smoke" if args.smoke_repeat else "dev"
    cfg_path = ROOT / "configs/phase2/experiments" / f"kc48_r70_pf_{split}.yaml"
    exp = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    result = smoke_repeat(exp, plan(exp)) if args.smoke_repeat else speed(exp, plan(exp))
    result["experiment_sha256"] = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    filename = "smoke_repeat.json" if args.smoke_repeat else "serial_speed.json"
    atomic_write_json(REPORTS / filename, result)
    print(json.dumps(result.get("summary", result), indent=2), flush=True)
    return int(result.get("passed") is False)


if __name__ == "__main__":
    raise SystemExit(main())
