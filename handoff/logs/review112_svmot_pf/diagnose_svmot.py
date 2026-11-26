"""Reconstruct matched histories and detection coverage for all 30 SVMOT clips."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
settings = OUT / "ultralytics_settings"
settings.mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(settings)
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import det_cache
import run_trackeval as rte
from trackeval.datasets._base_dataset import _BaseDataset

ORIGINAL = ROOT / "handoff/logs/review108_followup/svmot30"
MANIFEST = json.loads((ORIGINAL / "manifest.json").read_text(encoding="utf-8"))["videos"]
HASHES = {}


def sha(path):
    HASHES[str(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def reconstruct(data, shape):
    last_id = np.full(shape[1], -1)
    previous = np.full_like(last_id, -1)
    last_frame = np.full_like(last_id, -1)
    matched = np.zeros(shape, bool)
    events = []
    eps = np.finfo(float).eps
    for frame, (gt_ids, track_ids) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
        if not len(gt_ids) or not len(track_ids):
            continue
        similarity = data["similarity_scores"][frame]
        scores = 1000 * (track_ids == previous[gt_ids, None]) + similarity
        scores[similarity < .5 - eps] = 0
        i, j = linear_sum_assignment(-scores)
        keep = scores[i, j] > eps
        g, t = gt_ids[i[keep]], track_ids[j[keep]]
        changed = (last_id[g] != -1) & (last_id[g] != t)
        for identifier in g[changed]:
            events.append((frame, int(identifier), int(last_frame[identifier])))
        matched[frame, g] = True
        last_id[g], last_frame[g] = t, frame
        previous.fill(-1)
        previous[g] = t
    return matched, events


def detector_frames(video, length, official):
    if official:
        cache = det_cache.open_cache("r64_svmot_yolov8n_coco_c0.05", video, .25, [1, 2, 3, 5, 7], 300)
        frames = [row[-1] for row in cache.iter_frames()]
    else:
        path = ORIGINAL / "cache" / f"{video}.npz"
        with np.load(path) as archive:
            frames = []
            for frame in range(length):
                boxes = archive[str(frame)]
                keep = (boxes[:, 4] > .25) & np.isin(boxes[:, 5], [1, 2, 3, 5, 7])
                frames.append(boxes[keep][:300])
    assert len(frames) == length
    return frames


def main():
    sha(Path(__file__))
    summaries, events_out, per_id = [], [], []
    coverage_rows = []
    for video, info in sorted(MANIFEST.items()):
        official = info["split"] == "official_test"
        split = "SVMOTSYS-test" if official else "SVMOT-release21"
        length = info["frames"]
        gt_root = ROOT / "data/processed" if official else ORIGINAL / "ground_truth"
        gt_path = gt_root / split / video / "gt/gt.txt"
        sha(gt_path)
        gt = np.loadtxt(gt_path, delimiter=",", ndmin=2)
        ids = np.unique(gt[:, 1]).astype(int)
        shape = (length, len(ids))
        visibility = np.full(shape, np.nan)
        present = np.zeros(shape, bool)
        indices = gt[:, 0].astype(int) - 1, np.searchsorted(ids, gt[:, 1])
        visibility[indices] = gt[:, 8]
        present[indices] = True
        low = present & (visibility <= .5)
        detections = detector_frames(video, length, official)
        available = np.zeros(shape, bool)
        for frame, boxes in enumerate(detections):
            target = gt[gt[:, 0] == frame + 1]
            if not len(target) or not len(boxes):
                continue
            xywh = boxes[:, :4].copy()
            xywh[:, 2:4] -= xywh[:, :2]
            overlap = _BaseDataset._calculate_box_ious(target[:, 2:6], xywh,
                                                      box_format="xywh")
            available[frame, np.searchsorted(ids, target[:, 1])] = overlap.max(axis=1) >= .5
        for mask_name, mask in [("low", low), ("other", present & ~low)]:
            coverage_rows.append({"split": info["split"], "video": video, "group": mask_name,
                                  "gt": int(mask.sum()), "det_available": int((mask & available).sum())})
        for method in ["cv", "ekf", "ukf"]:
            tracker = f"r64-yolov8n_coco-c25-{method}" if official else method
            ec, dc, mc = rte.build_configs(split, tracker, None)
            dc.update(GT_FOLDER=str(gt_root), PRINT_CONFIG=False, SEQ_INFO={video: length})
            if not official:
                dc["TRACKERS_FOLDER"] = str(ORIGINAL / "trackers")
            sha(Path(dc["TRACKERS_FOLDER"]) / split / tracker / "data" / f"{video}.txt")
            dataset = rte.dataset_class("mot")(dc)
            raw = dataset.get_raw_seq_data(tracker, video)
            data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
            assert all(np.array_equal(a, ids[b]) for a, b in zip(raw["gt_ids"], data["gt_ids"]))
            reference_path = (ROOT / "results/trackeval" / split / tracker / "per_video_metrics.csv"
                              if official else ORIGINAL / f"release21_{method}.csv")
            sha(reference_path)
            reference = pd.read_csv(reference_path).set_index("video").loc[video]
            matched, events = reconstruct(data, shape)
            missed = present & ~matched
            assert len(events) == int(reference.IDSW)
            assert missed.sum() == int(reference.FN)
            count = Counter()
            id_count = Counter()
            for frame, compact, previous_frame in events:
                current_low = bool(low[frame, compact])
                was_low = bool(low[previous_frame, compact])
                recent_low = bool(low[max(0, frame - 30):frame + 1, compact].any())
                gap = frame - previous_frame - 1
                count["current_low" if current_low else "current_other"] += 1
                count["after_gap" if gap > 0 else "continuous"] += 1
                count["current_or_recent_low"] += recent_low
                count["current_high_previous_low"] += not current_low and was_low
                count["low_after_gap"] += current_low and gap > 0
                count["low_continuous"] += current_low and gap == 0
                id_count[compact] += current_low
                events_out.append({"split": info["split"], "video": video, "filter": method,
                                   "frame": frame + 1, "gt_id": int(ids[compact]), "gap_frames": gap,
                                   "visibility": float(visibility[frame, compact]),
                                   "previous_visibility": float(visibility[previous_frame, compact]),
                                   "current_low": current_low, "recent_low_1s": recent_low,
                                   "previous_match_low": was_low})
            for compact in range(len(ids)):
                if low[:, compact].any():
                    per_id.append({"split": info["split"], "video": video, "filter": method,
                                   "gt_id": int(ids[compact]), "low_gt": int(low[:, compact].sum()),
                                   "low_IDSW": int(id_count[compact])})
            summaries.append({"split": info["split"], "video": video, "filter": method,
                              "IDSW": len(events), "FN": int(missed.sum()),
                              "low_gt": int(low.sum()), "low_FN": int((missed & low).sum()),
                              "low_FN_no_candidate": int((missed & low & ~available).sum()),
                              "low_FN_with_candidate": int((missed & low & available).sum()),
                              **{k: int(count[k]) for k in ["current_low", "current_other", "after_gap",
                                 "continuous", "current_or_recent_low", "current_high_previous_low",
                                 "low_after_gap", "low_continuous"]}})
        print("DIAGNOSED", video, length, flush=True)
    for name, rows in [("svmot_per_clip.csv", summaries), ("svmot_events.csv", events_out),
                       ("svmot_per_id.csv", per_id), ("svmot_detection_coverage.csv", coverage_rows)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False)
    aggregated = pd.DataFrame(summaries).groupby(["split", "filter"]).sum(numeric_only=True)
    write_json("svmot_diagnosis.json", {
        "clip_filter_checks": len(summaries), "summary": aggregated.reset_index().to_dict("records"),
        "input_sha256": HASHES,
        "coverage": pd.DataFrame(coverage_rows).groupby(["split", "group"]).sum(numeric_only=True)
                    .reset_index().to_dict("records"),
        "detection_coverage_definition": "candidate IoU >= 0.5, no one-to-one assignment; upper bound only",
        "causal_limit": "Historical matching labels do not directly reveal tracker state or causality",
    })
    print(aggregated.to_string(), flush=True)


if __name__ == "__main__":
    main()
