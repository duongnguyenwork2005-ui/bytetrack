"""R87 exploratory conditional PF estimate on frozen R64 external caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from r64_ngoai import danh_gia_ngoai as r64  # noqa: E402
import config  # noqa: E402

SPECS = {"svmot": ("yolov8n_coco", 30.0), "roundabout": ("yolov8n_mixed5", 15.0)}
YAMLS = {"base": "configs/phase2/trackers/r72/motion100.yaml",
         "rb": "configs/phase2/trackers/r87/rb_observed.yaml"}
EVIDENCE = ROOT / "handoff/logs/review87_pf_rb"


def source_check(ds: str, smoke: bool) -> tuple[dict, str, str]:
    detector, _ = SPECS[ds]
    t = r64.cau_hinh(ds, smoke)
    cache = r64.ten_cache(ds, detector, smoke)
    checked = {v: r64.kiem_don_vi_cache(ds, detector, v, smoke) for v in t["videos"]}
    gt = r64.kiem_gt_cham(t)
    source = {"dataset": ds, "smoke": smoke, "split": t["split"], "cache": cache, "videos": t["videos"],
              "cache_checks": checked, "gt_sha256": gt,
              "yaml_sha256": {n: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for n, p in YAMLS.items()},
              "code_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                              for p in ("src/pf_ctrv.py", "src/tracker_ctrv.py", "src/run_tracking.py")}}
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    file = EVIDENCE / f"source_{ds}_{'smoke' if smoke else 'full'}.json"
    if file.exists() and json.loads(file.read_text()) != source:
        raise RuntimeError("source fingerprint changed")
    file.write_text(json.dumps(source, indent=2, default=str) + "\n")
    return t, cache, detector


def run(ds: str, smoke: bool, stage: str, which: str) -> None:
    t, cache, detector = source_check(ds, smoke)
    _, fps = SPECS[ds]
    prefix = f"r87-{ds}-{detector}-c25-{which}" + ("-smoke" if smoke else "")
    if stage == "track":
        cmd = [sys.executable, "-B", str(ROOT / "src/run_tracking.py"), "--cache-id", cache,
               "--split-name", t["split"], "--tracker-yaml", YAMLS[which], "--tracker-name", prefix,
               "--videos", *t["videos"], "--conf-min", "0.25", "--classes", "1", "2", "3", "5", "7",
               "--max-det", "300", "--fps", str(fps), "--track-buffer-sec", "1.2", "--elapsed-mode", "steps",
               "--seeds", "0", "1", "2", "--skip-existing"]
        subprocess.run(cmd, cwd=ROOT, check=True)
    else:
        for seed in (0, 1, 2):
            name = f"{prefix}-s{seed}"
            out = config.RESULTS_DIR / "trackeval" / t["split"] / name
            if out.exists() and (out / "combined_metrics.csv").exists() and (out / "raw_fields.json").exists():
                print("REUSE", name, flush=True)
                continue
            print("EVAL", name, r64.cham(t["split"], name, t["videos"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=SPECS, required=True)
    ap.add_argument("--stage", choices=("track", "eval"), required=True)
    ap.add_argument("--which", choices=YAMLS, required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.dataset, a.smoke, a.stage, a.which)
