"""GT-only geometric scenario audit of the fixed HELMET test36 subset."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tile_pilot import iou

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "handoff/logs/external_eval/r66/helmet/manifest_test36.json"
SPLIT = ROOT / "data/processed/HELMET-test36-motorcycle-v1"
OUT = ROOT / "handoff/logs/review91_pf_assoc/helmet_scenarios.json"


def main():
    videos = [row["video_id"] for row in json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]]
    report = {"definition": {"turn": "angle(t-5,t,t+5)>=30 deg; each displacement>=0.5*box_height(t)",
                             "overlap": "two GT boxes same frame IoU>=0.25; not physical occlusion label"},
              "videos": []}
    for video in videos:
        arr = np.loadtxt(SPLIT / video / "gt/gt.txt", delimiter=",", ndmin=2)
        arr = arr[arr[:, 7] == 1]
        turn_windows = turn_events = 0
        turn_ids = set()
        overlap_frames = 0
        overlap_ids = set()
        multi_gt_frames = 0
        for tid in np.unique(arr[:, 1]).astype(int):
            track = {int(row[0]): row for row in arr[arr[:, 1] == tid]}
            for frame, now in track.items():
                if frame - 5 not in track or frame + 5 not in track:
                    continue
                def center(row):
                    return np.array([row[2] + row[4] / 2, row[3] + row[5] / 2])
                v1 = center(now) - center(track[frame - 5])
                v2 = center(track[frame + 5]) - center(now)
                turn_windows += 1
                if min(np.linalg.norm(v1), np.linalg.norm(v2)) < .5 * now[5]:
                    continue
                angle = abs(math.degrees(math.atan2(float(np.linalg.det(np.stack((v1, v2)))),
                                                     float(np.dot(v1, v2)))))
                if angle >= 30:
                    turn_events += 1
                    turn_ids.add(tid)
        for frame in range(1, 101):
            subset = arr[arr[:, 0] == frame]
            if len(subset) < 2:
                continue
            multi_gt_frames += 1
            boxes = np.c_[subset[:, 2:4], subset[:, 2:4] + subset[:, 4:6]]
            ov = iou(boxes, boxes)
            np.fill_diagonal(ov, 0)
            pairs = np.argwhere(np.triu(ov >= .25, 1))
            if len(pairs):
                overlap_frames += 1
                for a, b in pairs:
                    overlap_ids.add(int(subset[a, 1]))
                    overlap_ids.add(int(subset[b, 1]))
        report["videos"].append({"video": video, "turn_windows": turn_windows,
                                  "turn_events": turn_events, "turn_ids": len(turn_ids),
                                  "multi_gt_frames": multi_gt_frames,
                                  "overlap_frames": overlap_frames, "overlap_ids": len(overlap_ids)})
    report["summary"] = {"videos": len(videos),
                         "videos_with_turn_proxy": sum(x["turn_events"] > 0 for x in report["videos"]),
                         "videos_with_overlap_proxy": sum(x["overlap_frames"] > 0 for x in report["videos"]),
                         "turn_windows": sum(x["turn_windows"] for x in report["videos"]),
                         "turn_events": sum(x["turn_events"] for x in report["videos"]),
                         "multi_gt_frames": sum(x["multi_gt_frames"] for x in report["videos"]),
                         "overlap_frames": sum(x["overlap_frames"] for x in report["videos"])}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
