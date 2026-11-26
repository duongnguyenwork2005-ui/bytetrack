"""Inspect auxiliary PF on the previously image-verified FastTrack events."""

from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "review116_fasttrack_full")]
from score_occlusion import assignments
from run import DATA, score
import numpy as np


def run():
    baseline = json.loads((HERE.parent / "review116_fasttrack_full/scores/occlusion.json").read_text())
    result = {}
    for seed in range(3):
        counter = Counter()
        results = []
        score.WORK = DATA / "evaluation_r117" / f"seed{seed}"
        for name, events in baseline["clips"].items():
            if not events:
                continue
            record = json.loads((HERE / f"fasttrack/seed{seed}/{name}.json").read_text())
            ids, present, predicted = assignments(name, record)
            assigned = predicted[f"pf-s{seed}"]
            for event in events:
                index = np.searchsorted(ids, event["gt_car"])
                a, b = assigned[event["before"]-1, index], assigned[event["after"]-1, index]
                if a < 0:
                    status = "before_missing"
                elif b < 0:
                    status = "after_missing"
                else:
                    status = "same_id" if a == b else "different_id"
                counter[status] += 1
                results.append({"clip": name, "gt_car": event["gt_car"], "before": event["before"], "status": status})
        assert sum(counter.values()) == 82
        result[str(seed)] = {"counts": dict(counter), "events": results}
    score.write_json(HERE / "transfer_occlusion.json", result)
    print({k: v["counts"] for k, v in result.items()}, flush=True)


if __name__ == "__main__":
    run()
