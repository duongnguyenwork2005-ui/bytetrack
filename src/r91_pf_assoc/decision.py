"""Apply the predeclared five-class detector gate to paired dev AP50 files."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "handoff/logs/review91_pf_assoc/balanced_pilot"
CLASSES = ("car", "truck", "bus", "motorcycle", "bicycle")


def main():
    baseline = json.loads((OUT / "baseline_dev_metrics.json").read_text(encoding="utf-8"))
    candidate = json.loads((OUT / "candidate_dev_metrics.json").read_text(encoding="utf-8"))
    delta = {name: candidate["per_class"][name]["AP50"] - baseline["per_class"][name]["AP50"]
             for name in CLASSES}
    overall = candidate["mAP50"] - baseline["mAP50"]
    gate = (delta["motorcycle"] >= .03 and overall >= 0 and
            all(delta[name] >= -.02 for name in CLASSES if name != "motorcycle"))
    result = {"gate_pass": gate, "mAP50_delta": overall, "AP50_delta": delta,
              "required": {"motorcycle_AP50_min_delta": .03,
                           "other_class_max_drop": -.02, "overall_mAP50_min_delta": 0},
              "decision": "external_motorcycle_check" if gate else "reject_candidate_keep_locked_detector"}
    (OUT / "gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
