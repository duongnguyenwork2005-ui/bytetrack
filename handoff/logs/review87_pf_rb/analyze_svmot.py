"""Summarize the fixed R87 SVMOT comparison against R73 motion100."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
split = ROOT / "results/trackeval/SVMOTSYS-test"
rows = []
for seed in (0, 1, 2):
    a = split / f"r73-svmot-yolov8n_coco-c25-motion100-s{seed}" / "combined_metrics.csv"
    b = split / f"r87-svmot-yolov8n_coco-c25-rb-s{seed}" / "combined_metrics.csv"
    with a.open(newline="") as f:
        base = next(r for r in csv.DictReader(f) if r["video"] == "COMBINED_SEQ")
    with b.open(newline="") as f:
        rb = next(r for r in csv.DictReader(f) if r["video"] == "COMBINED_SEQ")
    rows.append({"seed": seed, **{k: {"base": float(base[k]), "rb": float(rb[k]),
                                    "delta": float(rb[k]) - float(base[k])}
                                     for k in ("HOTA", "IDF1", "MOTA", "IDSW")}})

def avg(k: str, which: str) -> float:
    return sum(r[k][which] for r in rows) / len(rows)

summary = {k: {"base": avg(k, "base"), "rb": avg(k, "rb"), "delta": avg(k, "delta")}
           for k in ("HOTA", "IDF1", "MOTA", "IDSW")}
summary["IDSW"]["delta_pct"] = 100 * summary["IDSW"]["delta"] / summary["IDSW"]["base"]
summary["gate"] = {"HOTA_not_worse_than_minus_0_5pp": summary["HOTA"]["delta"] >= -0.005,
                   "IDSW_down_at_least_10pct": summary["IDSW"]["delta_pct"] <= -10}
source_a = json.loads((ROOT / "handoff/logs/review73_pf/source_svmot_full.json").read_text())
source_b = json.loads((OUT / "source_svmot_full.json").read_text())
for key in ("dataset", "split", "cache", "videos", "cache_checks", "gt_sha256"):
    if source_a[key] != source_b[key]:
        raise RuntimeError(f"source mismatch: {key}")
result = {"per_seed": rows, "summary": summary, "same_cache_gt_videos_as_r73": True}
(OUT / "result_svmot.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
