"""R79 BDD audit: compare time-limited IoU recovery to R74 and R78."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kc48 import boot_seed
from r74_pf_bdd import preflight
from r78_pf.bdd_analyze import METRICS, SPLIT, VIDEOS, load


def main() -> None:
    source = preflight(VIDEOS)
    groups = {"baseline": [], "iou3": [], "iou012s": []}
    rows = []
    code = None
    for seed in (0, 1, 2):
        names = {"baseline": f"r74-bdd-full-motion200-s{seed}",
                 "iou3": f"r78-bdd-full-iou3-s{seed}",
                 "iou012s": f"r79-bdd-full-iou012s-s{seed}"}
        for kind, name in names.items():
            raw, score = load(name)
            groups[kind].append(raw)
            rows.append(dict(kind=kind, seed=seed, **{m: score[m] for m in METRICS}))
            if kind != "iou012s":
                continue
            tracker = ROOT / "data/processed/trackers" / SPLIT / name
            meta = json.loads((tracker / "run_meta.json").read_text(encoding="utf-8"))
            cfg = meta["tracker_cfg"]
            if (meta["status"] != "complete" or meta["seed"] != seed or
                    meta["videos"] != VIDEOS or meta["cache_id"] != "yolov8n_coco" or
                    meta["conf_min"] != .25 or meta["classes"] != [1, 2, 3, 5, 7] or
                    meta["fps"] != 5 or meta["track_buffer_frames"] != 6 or
                    meta["elapsed_mode"] != "steps" or
                    cfg.get("lost_recover") != "iou" or cfg.get("lost_recover_iou") != .05 or
                    cfg.get("lost_recover_max_lost") != 1 or
                    cfg.get("pf_roughening_motion_scale") != 2):
                raise ValueError(f"tracking metadata mismatch: {name}")
            if code is None:
                code = meta["code"]
            elif meta["code"] != code:
                raise ValueError(f"code fingerprint differs: {name}")
            for video in VIDEOS:
                marker = json.loads((tracker / "video_meta" / f"{video}.json").read_text(encoding="utf-8"))
                if not marker["complete"] or marker["input_fingerprint"] != meta["input_fingerprint"]:
                    raise ValueError(f"tracking marker mismatch: {name}/{video}")
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("kind", sort=False)[METRICS].mean().reset_index()
    intervals = []
    for kind in ("iou3", "iou012s"):
        pair = boot_seed.paired_bootstrap_seed(groups[kind], groups["baseline"], VIDEOS,
                                               n_boot=5000, seed=0, metrics=METRICS)
        intervals.extend(dict(kind=kind, metric=m, **pair[m]) for m in METRICS)
    out = ROOT / "results/phase2/r79_pf_bdd/analysis"
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(intervals).to_csv(out / "paired_intervals.csv", index=False)
    (out / "source_fingerprint.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(intervals).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
