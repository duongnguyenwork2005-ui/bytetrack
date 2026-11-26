"""Score manually confirmed FastTrack occlusion events from completed MOT outputs."""

from collections import Counter
import json

import numpy as np
from trackeval.datasets._base_dataset import _BaseDataset

from detect import DATA, FASTTRACK, HERE
from score import Cache, WORK, clear_assignments, trackeval, write_json


def candidate_coverage(gt, cache):
    """Upper bound at IoU 0.5 before one-to-one assignment."""
    groups = {"all": [0, 0], "height_lt25px": [0, 0], "height_ge25px": [0, 0]}
    for frame, _, _, det in cache.iter_frames():
        rows = gt[gt[:, 0] == frame]
        boxes = det[:, :4].copy()
        boxes[:, 2:4] -= boxes[:, :2]
        hit = np.zeros(len(rows), dtype=bool)
        if len(boxes) and len(rows):
            iou = _BaseDataset._calculate_box_ious(rows[:, 2:6], boxes, box_format="xywh")
            hit = iou.max(axis=1) >= .5
        for key, mask in [("all", rows[:, 5] > 0), ("height_lt25px", rows[:, 5] < 25),
                          ("height_ge25px", rows[:, 5] >= 25)]:
            groups[key][0] += int((hit & mask).sum())
            groups[key][1] += int(mask.sum())
    return groups


def run():
    folder = HERE / "scores"
    events = json.loads((HERE / "fasttrack_occlusion_events.json").read_text())
    clips, total, coverage = {}, {}, {}
    for name in FASTTRACK:
        baseline = json.loads((folder / f"{name}.json").read_text())
        gt = np.loadtxt(WORK / "gt" / name / "gt" / "gt.txt", delimiter=",")
        ids = np.unique(gt[:, 1]).astype(int)
        present = np.zeros((baseline["frames"], len(ids)), dtype=bool)
        present[gt[:, 0].astype(int) - 1, np.searchsorted(ids, gt[:, 1])] = True
        coverage[name] = candidate_coverage(gt, Cache(DATA / f"{name}_yolov8n.npz", baseline["fps"]))
        clips[name] = []
        if not events.get(name):
            continue
        data_by_tracker = {}
        for tag, variant in baseline["variants"].items():
            dataset = trackeval.datasets.MotChallenge2DBox(
                {"PRINT_CONFIG": False, "TRACKERS_TO_EVAL": [tag], "BENCHMARK": "FastTrack",
                 "SPLIT_TO_EVAL": "pilot", "SKIP_SPLIT_FOL": True, "DO_PREPROC": False,
                 "GT_FOLDER": str(WORK / "gt"), "TRACKERS_FOLDER": str(WORK / "trackers"),
                 "SEQ_INFO": {name: baseline["frames"]}})
            raw = dataset.get_raw_seq_data(tag, name)
            data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
            assigned, switched = clear_assignments(data, present.shape)
            assert int(switched.sum()) == variant["metrics"]["IDSW"]
            assert int((present & (assigned < 0)).sum()) == variant["metrics"]["FN"]
            data_by_tracker[tag] = assigned
        for ident, blocker, before, peak, after in events[name]:
            assert before < peak < after
            i = np.searchsorted(ids, ident)
            assert ids[i] == ident and present[[before - 1, peak - 1, after - 1], i].all()
            by_filter = {}
            for tag, assigned in data_by_tracker.items():
                a, b = assigned[before - 1, i], assigned[after - 1, i]
                if a < 0:
                    status = "before_missing"
                elif b < 0:
                    status = "after_missing"
                elif a == b:
                    status = "same_id"
                else:
                    status = "different_id"
                by_filter[tag] = status
                total.setdefault(tag, Counter())[status] += 1
            clips[name].append({"gt_car": ident, "gt_blocker": blocker, "before": before,
                                "peak": peak, "after": after, "filters": by_filter})
    write_json(folder / "fasttrack_verified_occlusion.json",
               {"totals": total, "clips": clips, "detector_candidate_coverage": coverage})
    print(json.dumps({"totals": total, "coverage": coverage}), flush=True)


if __name__ == "__main__":
    run()
