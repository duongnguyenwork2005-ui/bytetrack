"""Compare PF fixture results by frame, ID/class and numeric boxes."""
import json
import sys

import numpy as np

a, b = (json.load(open(p, encoding="utf-8-sig")) for p in sys.argv[1:3])
assert len(a["events"]) == len(b["events"]) == 90
max_box = max_score = 0.0
rows = 0
failures = []
for x, y in zip(a["events"], b["events"]):
    assert x["frame"] == y["frame"]
    aa, bb = np.asarray(x["rows"]), np.asarray(y["rows"])
    if aa.shape != bb.shape:
        failures.append((x["frame"], "shape", aa.shape, bb.shape))
        continue
    if not aa.size:
        continue
    rows += len(aa)
    if not np.array_equal(aa[:, 4], bb[:, 4]) or not np.array_equal(aa[:, 6], bb[:, 6]):
        failures.append((x["frame"], "ID/class"))
    box = float(np.max(np.abs(aa[:, :4] - bb[:, :4])))
    score = float(np.max(np.abs(aa[:, 5] - bb[:, 5])))
    max_box, max_score = max(max_box, box), max(max_score, score)
    if box > .001 or score > 1e-6:
        failures.append((x["frame"], "numeric", box, score))
print(json.dumps({"frames": 90, "rows": rows, "max_box_diff_px": max_box,
                  "max_score_diff": max_score, "failures": failures[:10],
                  "n_failure_frames": len(set(f[0] for f in failures)),
                  "pass": not failures}))
sys.exit(0 if not failures else 1)
