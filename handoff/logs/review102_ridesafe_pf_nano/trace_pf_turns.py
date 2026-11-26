"""Replay RoundaboutHD CLEAR matches and preserve evidence for each turning ID switch."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from trackeval.datasets._base_dataset import _BaseDataset

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "review99_roundabout_turns"))
from audit_turns import CAMERAS, ROOT, SPLIT, labels_for_camera, rte  # noqa: E402
import det_cache  # noqa: E402

CACHE = "r64_roundabout_yolov8n_mixed5_c0.05"
CLASSES = [1, 2, 3, 5, 7]
EPS = np.finfo(float).eps


def best_iou(box, other):
    return float(_BaseDataset._calculate_box_ious(box.reshape(1, 4), other, box_format="xywh").max(initial=0))


def replay(data, actual_gt_ids, turn_index, detections):
    last_id = np.full(len(actual_gt_ids), -1, dtype=int)
    prev_id = np.full_like(last_id, -1)
    last_match = np.full_like(last_id, -1)
    total_switches = matched = 0
    events = []
    for t, (gt, trackers) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
        if not len(gt) or not len(trackers):
            continue
        similarity = data["similarity_scores"][t]
        score = 1000 * (trackers == prev_id[gt, None]) + similarity
        score[similarity < .5 - EPS] = 0
        rows, cols = linear_sum_assignment(-score)
        valid = score[rows, cols] > EPS
        rows, cols = rows[valid], cols[valid]
        now_gt, now_track = gt[rows], trackers[cols]
        matched += len(rows)
        for row, column, g, new in zip(rows, cols, now_gt, now_track):
            old = last_id[g]
            if old < 0 or old == new:
                continue
            total_switches += 1
            episode = turn_index[t, g]
            if episode < 0:
                continue
            previous = int(last_match[g])
            gap = t - previous - 1
            old_columns = np.flatnonzero(trackers == old)
            assert len(old_columns) <= 1 and gap >= 0
            old_matches = now_gt[now_track == old]
            boxes = data["gt_dets"][t]
            box = boxes[row]
            det_boxes = detections.frame(t)[:, :4].astype(float, copy=True)
            det_boxes[:, 2:] -= det_boxes[:, :2]
            det_iou = best_iou(box, det_boxes)
            neighbor_iou = best_iou(box, np.delete(boxes, row, axis=0))
            events.append({
                "frame": t + 1, "gt_id": int(actual_gt_ids[g]), "turn_episode": int(episode),
                "previous_match_frame": previous + 1, "gap": gap,
                "previous_tracker_compact_id": int(old), "new_tracker_compact_id": int(new),
                "old_track_present": bool(len(old_columns)),
                "old_track_gt_iou": float(similarity[row, old_columns[0]]) if len(old_columns) else None,
                "old_track_other_gt": int(actual_gt_ids[old_matches[0]]) if len(old_matches) else None,
                "new_track_gt_iou": float(similarity[row, column]),
                "best_detection_iou": det_iou, "best_other_gt_iou": neighbor_iou,
            })
        last_id[now_gt] = now_track
        last_match[now_gt] = t
        prev_id.fill(-1)
        prev_id[now_gt] = now_track
    return total_switches, int(data["num_gt_dets"]) - matched, events


def summarize(events):
    counts = Counter()
    for e in events:
        counts["events"] += 1
        counts["after_gap" if e["gap"] else "consecutive_match"] += 1
        counts["previous_id_present" if e["old_track_present"] else "previous_id_absent"] += 1
        counts["previous_id_matched_other_gt"] += e["old_track_other_gt"] is not None
        counts["previous_id_still_overlaps_gt"] += (e["old_track_gt_iou"] or 0) >= .5
        counts["neighbor_gt_overlap_0_10"] += e["best_other_gt_iou"] >= .10
        counts["usable_detection_iou_0_50"] += e["best_detection_iou"] >= .50
        counts["after_gap_previous_id_absent"] += e["gap"] > 0 and not e["old_track_present"]
    return dict(counts)


def main():
    gt = {cam: labels_for_camera(ROOT / "data/processed" / SPLIT / cam / "gt/gt.txt", cam) for cam in CAMERAS}
    det = {cam: det_cache.open_cache(CACHE, cam, .25, CLASSES, max_det=300) for cam in CAMERAS}
    for source in det.values():
        assert np.array_equal(source.frame_idx, np.arange(1, 3001))
    references = json.loads((OUT.parent / "review101_detector_dair_nano/roundabout_pf_turns.json").read_text())
    turns = {(item["camera"], item["filter"]): item["idsw"] for item in references["rows"] if item["level"] == "turn25"}
    all_events, runs = {}, []
    for scale in (100, 200):
        for seed in (0, 1, 2):
            name = f"r73-roundabout-yolov8n_mixed5-c25-motion{scale}-s{seed}"
            key = f"pf_k1k2_motion{scale}-s{seed}"
            _, config, _ = rte.build_configs(SPLIT, name, CAMERAS)
            dataset = rte.dataset_class("mot")(config)
            with (ROOT / "results/trackeval" / SPLIT / name / "per_video_metrics.csv").open() as file:
                expected = {r["video"]: r for r in csv.DictReader(file)}
            all_events[key] = []
            for cam in CAMERAS:
                ids, _, levels, _ = gt[cam]
                raw = dataset.get_raw_seq_data(name, cam)
                data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
                for r, compact in zip(raw["gt_ids"], data["gt_ids"]):
                    assert np.array_equal(r, ids[compact])
                switches, fn, events = replay(data, ids, levels["turn25"], det[cam])
                assert switches == int(float(expected[cam]["IDSW"]))
                assert fn == int(float(expected[cam]["FN"]))
                assert len(events) == turns[(cam, key)]
                all_events[key].extend(dict(event, camera=cam) for event in events)
                runs.append({"camera": cam, "run": name, "whole_idsw": switches, "whole_fn": fn, "turn_idsw": len(events)})
    summaries = {name: summarize(events) for name, events in all_events.items()}
    pooled = {str(scale): summarize([e for key, events in all_events.items() if f"motion{scale}-" in key for e in events])
              for scale in (100, 200)}
    assert len(runs) == 24
    (OUT / "pf_turn_event_trace.json").write_text(json.dumps({"runs": runs, "summaries": summaries,
                                                               "pooled_by_scale": pooled, "events": all_events}, indent=2))
    print(json.dumps(pooled), flush=True)


if __name__ == "__main__":
    main()
