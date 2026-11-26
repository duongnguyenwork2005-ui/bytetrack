"""Deterministic vehicle detections for PF backend parity; JSON stdout only."""
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

package = Path(os.environ["R97_PACKAGE"])
sys.path.insert(0, str(package / "src"))
os.chdir(package)
import det_cache  # noqa: E402
import run_tracking  # noqa: E402

if os.environ.get("BYTETRACK_BACKEND") == "np":
    import bytetrack_np  # noqa: E402
    import tracker_ctrv  # noqa: E402

    old_registry = bytetrack_np.tracker_registry_np

    def registry():
        reg = old_registry()
        reg["bytetrack_pf_ctrv"] = tracker_ctrv.PFByteTracker
        return reg

    bytetrack_np.tracker_registry_np = registry

yaml = "configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml"
cfg = run_tracking.load_tracker_cfg(yaml, {"track_buffer": 30, "pf_seed": 97})
tr = run_tracking.make_tracker(cfg)
tr.set_seed(97, "R97_synthetic")
events = []
for frame in range(1, 91):
    boxes = []
    # Three vehicles: straight, turning, then brief occlusion/reappearance.
    for obj in range(3):
        if obj == 1 and 28 <= frame <= 33:
            continue
        if obj == 2 and 51 <= frame <= 59:
            continue
        x = 70 + 145 * obj + 2.2 * frame + (13 * math.sin(frame / 8) if obj == 1 else 0)
        y = 100 + 90 * obj + (1.1 * max(0, frame - 24) if obj == 1 else 0)
        w, h = 52 + 5 * obj, 30 + 3 * obj
        score = .9 if frame % 11 else .27
        boxes.append([x - w/2, y - h/2, x + w/2, y + h/2, score, 2 if obj < 2 else 3])
    arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
    if os.environ.get("BYTETRACK_BACKEND") == "np":
        det = det_cache.to_boxes_np(arr, (720, 1280))
    else:
        det = det_cache.to_boxes(arr, (720, 1280)).cpu().numpy()
    out = tr.update(det, None, n_steps=1)
    events.append({"frame": frame, "rows": np.asarray(out).tolist()})
print(json.dumps({"backend": os.environ.get("BYTETRACK_BACKEND"), "events": events}, separators=(",", ":")))
