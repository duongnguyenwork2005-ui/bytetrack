"""Run the R72 development matrix through the existing tracking and evaluation code.

Master writes shared manifests once. Each tracking worker owns one tracker/seed
directory and always enters stage_track, which validates per-video fingerprints.
Parallel wall times are operational timings, not a speed benchmark.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kc48 import chay_thi_nghiem_kc48 as wrapper
import config


def preflight(exp: dict, videos: list[str], out_dir: Path) -> None:
    cache = ROOT / "data/interim/det_cache" / exp["cache_id"]
    required = [cache / "meta.json"]
    required.extend(cache / f"{video}{suffix}" for video in videos for suffix in (".json", ".npz"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing detection cache files: " + ", ".join(missing))
    wrapper.chuan_bi_gt(exp, videos, out_dir / "_processed", out_dir)


def worker(exp: dict, row: dict, videos: list[str], out_dir: Path) -> int:
    config.PROCESSED_DIR = out_dir / "_processed"
    import run_experiment

    start = time.perf_counter()
    run_experiment.stage_track(exp, [row], videos, skip_existing=True)
    record = {"run": row["dir"], "elapsed_seconds": time.perf_counter() - start,
              "timing_scope": "Tracking with concurrent workers; diagnostic only", "exit": 0}
    (out_dir / "_log" / f"{row['dir']}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "track", "eval"), required=True)
    parser.add_argument("--smoke", action="store_true", help="Use the two predefined development sequences")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--run-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("--workers must be between 1 and 3")
    os.chdir(ROOT)
    name = "kc48_r72_pf_smoke" if args.smoke else "kc48_r72_pf_dev"
    exp_path = f"configs/phase2/experiments/{name}.yaml"
    exp = wrapper.doc_exp(exp_path)
    if exp["name"] != name:
        raise ValueError("Experiment name must match its R72 configuration")
    videos = wrapper.videos_of(exp)
    out_dir = ROOT / "results/phase2" / name

    import run_experiment
    rows = run_experiment.plan(exp)
    if args.run_index is not None:
        if args.stage != "track" or not 0 <= args.run_index < len(rows):
            parser.error("A worker must select a valid tracking run index")
        return worker(exp, rows[args.run_index], videos, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "_log"
    log_dir.mkdir(exist_ok=True)
    preflight(exp, videos, out_dir)
    if args.stage == "eval":
        return wrapper.chay_runner(exp_path, exp, "eval", [], out_dir, out_dir / "_processed")

    # The existing runner writes the complete experiment.yaml, videos.json and
    # runs.csv. Workers do not call main, so those shared files cannot race.
    rc = wrapper.chay_runner(exp_path, exp, "track", ["--dry-run"], out_dir, out_dir / "_processed")
    if rc or args.stage == "prepare":
        return rc

    def launch(index: int, row: dict) -> dict:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--stage", "track",
                   "--run-index", str(index)]
        if args.smoke:
            command.append("--smoke")
        start = time.perf_counter()
        with (log_dir / f"{row['dir']}.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        return {"run": row["dir"], "exit": process.returncode,
                "elapsed_seconds": round(time.perf_counter() - start, 3)}

    completed = []
    print(f"R72 tracking {len(rows)} runs x {len(videos)} videos, workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(launch, index, row) for index, row in enumerate(rows)]
        for task in as_completed(tasks):
            record = task.result()
            completed.append(record)
            print(json.dumps(record), flush=True)
    with (out_dir / "dispatch.jsonl").open("a", encoding="utf-8") as log:
        for record in completed:
            log.write(json.dumps(record) + "\n")
    return int(any(record["exit"] != 0 for record in completed))


if __name__ == "__main__":
    raise SystemExit(main())
