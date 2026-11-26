"""Replay the frozen R92 PF candidate on the R74 BDD camera-vehicle cache."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import det_cache
import run_tracking
from r74_pf_bdd import CACHE, MANIFEST, SMOKE, SPLIT, preflight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    videos = json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]
    report = preflight(videos)
    print("PREFLIGHT", report, flush=True)
    det_cache.CACHE_ROOT = CACHE.parent
    selected = SMOKE if args.scope == "smoke" else videos
    sys.argv = [str(__file__), "--cache-id", "yolov8n_coco", "--split-name", SPLIT,
                "--tracker-yaml", "configs/phase2/trackers/r92/lost_cv50.yaml",
                "--tracker-name", f"r93-bdd-{args.scope}-cv50", "--videos", *selected,
                "--conf-min", "0.25", "--classes", "1", "2", "3", "5", "7",
                "--max-det", "300", "--fps", "5", "--track-buffer-sec", "1.2",
                "--elapsed-mode", "steps", "--seeds", "0", "1", "2", "--skip-existing"]
    return run_tracking.main()


if __name__ == "__main__":
    raise SystemExit(main())
