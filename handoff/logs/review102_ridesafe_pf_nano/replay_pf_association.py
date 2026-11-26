"""Inspect the actual association pool on a parity-checked RoundaboutHD replay."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from trackeval.datasets._base_dataset import _BaseDataset

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "src"))
import det_cache  # noqa: E402
import run_tracking  # noqa: E402

RUN = "r73-roundabout-yolov8n_mixed5-c25-motion200-s0"
SPLIT = "RoundaboutHD-prefix3000"
BOX = ["x", "y", "w", "h"]


def iou(first, second):
    return float(_BaseDataset._calculate_box_ious(np.asarray(first).reshape(1, 4),
                                                 np.asarray(second).reshape(1, 4))[0, 0])


def association_evidence(event, raw_ids, tracks, detections, logs, gt):
    frame, last = event["frame"], event["previous_match_frame"]
    old, new = (int(raw_ids[event[name]]) for name in ("previous_tracker_compact_id", "new_tracker_compact_id"))
    births = logs[(logs["track_id"] == new) & (logs["kind"] == "new")]
    assert len(births) == 1
    birth = int(births.iloc[0]["frame"])
    selected = birth if last < birth <= frame else frame
    if selected == birth:
        det_id = int(births.iloc[0]["det_idx"])
    else:
        new_row = tracks[(tracks["track_id"] == new) & (tracks["frame"] == selected)]
        det_id = int(new_row.iloc[0]["matched_det"]) if len(new_row) else -1
    target = detections[(detections["frame"] == selected) & (detections["det_idx"] == det_id)]
    pool = tracks[(tracks["frame"] == selected) & (tracks["track_id"] == old)]
    assert len(pool) <= 1 and len(target) <= 1
    states = logs[(logs["track_id"] == old) & (logs["frame"] <= selected) & (logs["frame"] >= last)]
    switches = states[["frame", "kind", "n_lost"]].to_dict("records")
    old_state = int(pool.iloc[0]["state"]) if len(pool) else None
    old_target_iou = iou(pool.iloc[0][BOX], target.iloc[0][BOX]) if len(pool) and len(target) else None
    confidence = float(target.iloc[0]["conf"]) if len(target) else None
    gt_row = gt[(gt[:, 0] == selected) & (gt[:, 1] == event["gt_id"]), 2:6]
    target_gt_iou = iou(gt_row[0], target.iloc[0][BOX]) if len(gt_row) == len(target) == 1 else None
    return dict(event, old_track_id=old, new_track_id=new, birth_frame=birth, association_frame=selected,
                old_pool_state=old_state, old_pool_stage=int(pool.iloc[0]["stage"]) if len(pool) else None,
                old_pool_lost_steps=int(pool.iloc[0]["n_lost"]) if len(pool) else None,
                old_events_since_last_match=switches, target_detection_id=det_id,
                target_confidence=confidence, target_gt_iou=target_gt_iou, old_target_iou=old_target_iou,
                fused_iou_cost=None if old_target_iou is None else 1 - old_target_iou * confidence)


def main():
    trace = json.loads((OUT / "pf_turn_event_trace.json").read_text())
    events = trace["events"]["pf_k1k2_motion200-s0"]
    counts = Counter(e["camera"] for e in events)
    camera = min(counts, key=lambda name: (-counts[name], name))
    expected_root = ROOT / "data/processed/trackers" / SPLIT / RUN
    meta = json.loads((expected_root / "run_meta.json").read_text())
    cache = det_cache.open_cache(meta["cache_id"], camera, meta["conf_min"], meta["classes"], meta["max_det"])
    cfg = run_tracking.load_tracker_cfg(ROOT / "configs/phase2/trackers/r72/motion200.yaml", meta["overrides"])
    assert run_tracking.cfg_to_dict(cfg) == meta["tracker_cfg"]
    sink = {"assoc_log": True}
    mot, _, timing = run_tracking.track_video(cache, cfg, dt_nominal=meta["dt_nominal_s"],
                                              elapsed_mode=meta["elapsed_mode"], seed=meta["seed"],
                                              video=camera, log_sink=sink)
    expected = np.loadtxt(expected_root / "data" / f"{camera}.txt", delimiter=",")
    assert len(timing) == 3000 and np.array_equal(mot.to_numpy(dtype=float), expected)
    raw_ids = np.unique(expected[:, 1]).astype(int)
    gt = np.loadtxt(ROOT / "data/processed" / SPLIT / camera / "gt/gt.txt", delimiter=",")
    tracks, detections, logs = sink["assoc_log"].frames()
    rows = [association_evidence(e, raw_ids, tracks, detections, logs, gt) for e in events if e["camera"] == camera]
    for row in rows:
        current = expected[(expected[:, 0] == row["frame"]) & (expected[:, 1] == row["new_track_id"]), 2:6]
        target = gt[(gt[:, 0] == row["frame"]) & (gt[:, 1] == row["gt_id"]), 2:6]
        assert len(current) == len(target) == 1 and abs(iou(current[0], target[0]) - row["new_track_gt_iou"]) < 1e-12
    summary = Counter("old_absent_from_pool" if r["old_pool_state"] is None else f"old_state_{r['old_pool_state']}" for r in rows)
    summary["pair_cost_above_match_thresh"] = sum((r["fused_iou_cost"] or 0) > cfg.match_thresh for r in rows)
    summary["matching_detection_at_association_frame"] = sum((r["target_gt_iou"] or 0) >= .5 for r in rows)
    (OUT / "pf_association_replay.json").write_text(json.dumps({"camera": camera, "run": RUN, "mot_rows": len(mot),
                                                               "exact_mot": True, "summary": dict(summary), "rows": rows}, indent=2))
    print(json.dumps(dict(summary)), flush=True)


if __name__ == "__main__":
    main()
