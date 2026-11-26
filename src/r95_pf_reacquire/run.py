"""Replay R95 opt-in PF association on fixed UA and BDD caches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import det_cache
import run_tracking
from r74_pf_bdd import CACHE, MANIFEST, SMOKE as BDD_SMOKE, SPLIT as BDD_SPLIT, preflight
from r92_pf_modes.eval import DEV, SMOKE as UA_SMOKE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("ua", "bdd"), required=True)
    parser.add_argument("--scope", choices=("smoke", "full"), required=True)
    parser.add_argument("--case", choices=("base", "rescue"), default="rescue")
    args = parser.parse_args()
    yaml = ("configs/phase2/trackers/r72/motion200.yaml" if args.case == "base"
            else "configs/phase2/trackers/r95/pf_reacquire.yaml")
    if args.domain == "ua":
        videos = list(UA_SMOKE if args.scope == "smoke" else DEV)
        split, cache_id, fps, elapsed = "DETRAC-all", "yolov8n-coco_conf0.05_all", "25", "skip"
        classes = ("2", "3", "5", "7")
        scope = "smoke" if args.scope == "smoke" else "dev"
    else:
        all_videos = json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]
        print("PREFLIGHT", preflight(all_videos), flush=True)
        det_cache.CACHE_ROOT = CACHE.parent
        videos = list(BDD_SMOKE if args.scope == "smoke" else all_videos)
        split, cache_id, fps, elapsed = BDD_SPLIT, "yolov8n_coco", "5", "steps"
        classes = ("1", "2", "3", "5", "7")
        scope = args.scope
    sys.argv = [str(__file__), "--cache-id", cache_id, "--split-name", split,
                "--tracker-yaml", yaml, "--tracker-name", f"r95-{args.domain}-{scope}-{args.case}",
                "--videos", *videos, "--conf-min", "0.25", "--classes", *classes,
                "--max-det", "300", "--fps", fps, "--track-buffer-sec", "1.2",
                "--elapsed-mode", elapsed, "--seeds", "0", "1", "2", "--skip-existing"]
    if args.scope == "smoke":
        sys.argv.append("--assoc-log")
    return run_tracking.main()


if __name__ == "__main__":
    raise SystemExit(main())
