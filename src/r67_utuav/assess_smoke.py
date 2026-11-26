"""Class-agnostic IoU upper-bound recall on locked UTUAV smoke/full caches."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval as run


def upper_bound(gt, detections):
    if len(gt) == 0 or len(detections) == 0:
        return 0
    g1 = gt[:, 2:4]
    g2 = g1 + gt[:, 4:6]
    d1, d2 = detections[:, :2], detections[:, 2:4]
    lo = np.maximum(g1[:, None, :], d1[None, :, :])
    hi = np.minimum(g2[:, None, :], d2[None, :, :])
    inter = np.prod(np.maximum(hi-lo, 0), axis=2)
    ga = np.prod(g2-g1, axis=1)[:, None]
    da = np.prod(d2-d1, axis=1)[None, :]
    return int(((inter / np.maximum(ga+da-inter, 1e-9)).max(axis=1) >= 0.5).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "full"], default="smoke")
    phase = parser.parse_args().phase
    record = run.read(run.prep.LOG / phase / "phase_inputs.json")
    rows = []
    for detector in run.r64.DETECTOR:
        cache_id = f"r67_utuav_{phase}_{record['protocol_hash'][:12]}_{detector}_c005"
        for video, (_, first, last, _) in run.prep.RANGES.items():
            gt = np.loadtxt(run.config.PROCESSED_DIR /
                            (run.prep.SMOKE if phase == "smoke" else run.prep.SPLIT) / video / "gt/gt.txt",
                            delimiter=",", ndmin=2)
            gt = gt[gt[:, 7] == 1]
            order = np.argsort(gt[:, 0], kind="stable")
            gt = gt[order]
            n = 128 if phase == "smoke" else last-first+1
            gt_offsets = np.searchsorted(gt[:, 0], np.arange(1, n+2))
            with np.load(run.det_cache.CACHE_ROOT/cache_id/f"{video}.npz") as z:
                counts = z["n_per_frame"]
                detections = z["data"]
            offset = np.concatenate(([0], np.cumsum(counts)))
            for threshold in (0.05, 0.10, 0.25):
                matched, possible, boxes, active = 0, 0, 0, 0
                for frame in range(1, n+1):
                    g = gt[gt_offsets[frame-1]:gt_offsets[frame]]
                    d = detections[offset[frame-1]:offset[frame]]
                    d = d[(d[:, 4] > threshold) & np.isin(d[:, 5], run.CLASSES)]
                    matched += upper_bound(g, d)
                    possible += len(g)
                    boxes += len(d)
                    active += bool(len(d))
                rows.append({"detector": detector, "video": video, "conf_gt": threshold,
                    "gt_dynamic": possible, "det_boxes": boxes, "frames_with_det": active,
                    "gt_with_any_iou50": matched, "upper_bound_recall": matched/possible})
    out = run.prep.LOG/phase/"detector_upper_bound_recall.json"
    run.write(out, {"definition": "Per GT dynamic box, any tracked COCO-class detection "
              "at IoU>=0.5; class-agnostic and not one-to-one, hence an upper bound",
              "rows": rows})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
