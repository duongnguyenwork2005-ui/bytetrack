"""Run paired UAVDT vehicle tracking on one locked detector cache."""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import config
from r64_ngoai import danh_gia_ngoai as r64
from r90_uavdt.audit import EVIDENCE
from r90_uavdt.cache import CACHE
from r90_uavdt.prepare import SPLIT

VIDEOS = json.loads((EVIDENCE / "selection.json").read_text(encoding="utf-8"))["all"]
FILTERS = {
    "cv": ("bytetrack.yaml", None),
    "ekf": ("configs/bytetrack_ekf_ctrv.yaml", None),
    "ukf": ("configs/bytetrack_ukf_ctrv.yaml", None),
    "pf_default": ("configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml", [0, 1, 2]),
    "pf_motion100": ("configs/phase2/trackers/r72/motion100.yaml", [0, 1, 2]),
}
PROGRESS = EVIDENCE / "matrix_progress.json"


def cell(conf: str, filt: str) -> dict:
    yaml, seeds = FILTERS[filt]
    name = f"r90-{conf}-{filt}"
    cmd = [sys.executable, "-B", str(ROOT / "src/run_tracking.py"), "--cache-id", CACHE,
        "--split-name", SPLIT, "--tracker-yaml", yaml, "--tracker-name", name,
        "--videos", *VIDEOS, "--conf-min", {"c25": "0.25", "c10": "0.10"}[conf],
        "--classes", "2", "5", "7", "--max-det", "300", "--fps", "30",
        "--track-buffer-sec", "1.2", "--elapsed-mode", "steps", "--skip-existing"]
    if seeds is not None:
        cmd += ["--seeds", *map(str, seeds)]
    log = EVIDENCE / f"track_{name}.log"
    with log.open("w", encoding="utf-8") as f:
        f.write("argv: " + json.dumps(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=f, stderr=subprocess.STDOUT)
    metrics = {}
    import pandas as pd
    for seed in seeds if seeds is not None else [None]:
        tracker = name + (f"-s{seed}" if seed is not None else "")
        path = config.RESULTS_DIR / "trackeval" / SPLIT / tracker / "combined_metrics.csv"
        if path.exists():
            row = pd.read_csv(path).iloc[0]
            metrics[tracker] = {k: float(row[k]) for k in ("HOTA", "IDF1", "MOTA", "IDSW")}
        else:
            metrics[tracker] = r64.cham(SPLIT, tracker, VIDEOS)
        print("EVAL", tracker, metrics[tracker], flush=True)
    return metrics


def main() -> None:
    assert len(VIDEOS) == 8
    progress = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else {}
    for conf in ("c25", "c10"):
        for filt in FILTERS:
            key = f"{conf}/{filt}"
            if progress.get(key, {}).get("status") == "done":
                print("REUSE", key, flush=True)
                continue
            print("START", key, datetime.now().isoformat(), flush=True)
            try:
                metrics = cell(conf, filt)
                progress[key] = {"status": "done", "metrics": metrics,
                                 "finished": datetime.now().isoformat()}
            except Exception as exc:
                progress[key] = {"status": "failed", "error": repr(exc),
                                 "traceback": traceback.format_exc(),
                                 "finished": datetime.now().isoformat()}
                PROGRESS.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                raise
            PROGRESS.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print("DONE", key, flush=True)
    print("ALL DONE", len(progress), flush=True)


if __name__ == "__main__":
    main()
