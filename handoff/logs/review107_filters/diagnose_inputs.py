"""Recalculate historical evidence and deterministic reference scores, without tracking."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "src"))
import boot_stats

trace_path = ROOT / "handoff/logs/review102_ridesafe_pf_nano/pf_association_replay.json"
trace = json.loads(trace_path.read_text())
rows = trace["rows"]
counts = {"events": len(rows), "old_lost": sum(r["old_pool_state"] == 2 for r in rows),
          "lost_good_det_cost_above_gate": sum(r["old_pool_state"] == 2 and (r["target_gt_iou"] or 0) >= .5
                                                and (r["fused_iou_cost"] or 0) > .8 for r in rows),
          "all_cost_above_gate": sum((r["fused_iou_cost"] or 0) > .8 for r in rows),
          "all_good_det": sum((r["target_gt_iou"] or 0) >= .5 for r in rows)}
assert counts == {"events": 13, "old_lost": 7, "lost_good_det_cost_above_gate": 4,
                  "all_cost_above_gate": 9, "all_good_det": 8}
paths = [trace_path, Path(__file__)]
dev = json.loads((ROOT / "results/phase2/manifests/detrac_dev.json").read_text())["videos"]
filters = {}
for name in ("cv", "ekf", "ukf"):
    path = ROOT / f"results/trackeval/DETRAC-all/p2-kc48_r70_pf_dev-A-{name}"
    raw = boot_stats.load_raw_fields(path / "raw_fields.json")
    assert set(raw) == set(dev)
    combined = boot_stats.combine(raw, dev)
    recorded = pd.read_csv(path / "combined_metrics.csv").iloc[0]
    for key in ("HOTA", "IDF1", "IDSW", "FN", "FP"):
        assert abs(combined[key] - recorded[key]) < 1e-9
    filters[name] = combined
    paths.extend([path / "raw_fields.json", path / "combined_metrics.csv"])
side_effect = ROOT / "Ultralytics/settings.json"
if side_effect.exists():
    paths.append(side_effect)
result = {"r102_counts_recalculated_from_saved_rows": counts,
          "limits": "No R102 tracking replay repeated; this verifies arithmetic from saved diagnostic rows, not causality.",
          "dev_reference_raw_recombined": filters,
          "unintended_write": {"path": str(side_effect), "exists": side_effect.exists(),
            "origin": "First unit process: Ultralytics fallback because requested settings parent did not exist.",
            "resolution": "No further modification/deletion outside write_paths. Created settings parent inside R107 before later imports."},
          "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(OUT / "diagnosis_verified.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
