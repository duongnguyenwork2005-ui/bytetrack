"""Review motorcycle GT against cached detector output on the 12 preselected HELMET smoke clips.

Use --gt-only for the first visual inspection. For detector review, pass three
--cache DETECTOR=CACHE_ID arguments from the R66 HELMET smoke plan.
Both modes use frame nearest 50 with a GT box (ties choose the earlier frame).
Recall uses all 100 frames of each clip. This script does not run a detector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from helmet_prep import IMAGES, LOG, OUTPUT, SPLIT_SMOKE, config, manifest, r64, sha256
import det_cache

DETECTORS = ("yolov8n_coco", "yolov8n_mixed5", "yolo11n_mixed5")
THRESHOLDS = (0.05, 0.10, 0.25)
OUT = LOG / "smoke_review"


def matches(gt: np.ndarray, det: np.ndarray) -> int:
    if len(gt) == 0 or len(det) == 0:
        return 0
    gt_x2y2 = gt[:, :2] + gt[:, 2:4]
    area_gt = gt[:, 2:4].prod(axis=1)
    area_det = (det[:, 2:4] - det[:, :2]).prod(axis=1)
    sides = (np.minimum(gt_x2y2[:, None, :], det[None, :, 2:4])
             - np.maximum(gt[:, None, :2], det[None, :, :2]))
    overlap = np.maximum(sides, 0).prod(axis=2)
    iou = overlap / (area_gt[:, None] + area_det[None, :] - overlap)
    rid, cid = linear_sum_assignment(np.where(iou >= 0.5, -iou, 1e6))
    return int(np.sum(iou[rid, cid] >= 0.5))


def verified_cache(cache_id: str, detector: str, clip: str, gt_hash: str):
    folder = det_cache.CACHE_ROOT / cache_id
    marker_path = folder / f"{clip}.xong.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {"detector": detector, "video": clip, "split": SPLIT_SMOKE}
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError(f"wrong cache marker for {clip}: {cache_id}")
    for name in ("meta.json", f"{clip}.json", f"{clip}.npz"):
        if marker.get("sha256", {}).get(name) != sha256(folder / name):
            raise ValueError(f"{cache_id}/{name}: cache content differs from completion marker")
    info = json.loads((folder / f"{clip}.json").read_text(encoding="utf-8"))
    if (not marker.get("protocol_hash") or marker["protocol_hash"] != info.get("protocol_hash")
            or info.get("split") != SPLIT_SMOKE or info.get("gt_luc_sinh_sha256") != gt_hash):
        raise ValueError(f"{cache_id}/{clip}: cache protocol or GT hash mismatch")
    video = det_cache.open_cache(cache_id, clip, 0.05, [3], 3000)
    if (video.orig_shape != (1080, 1920) or not np.array_equal(video.frame_idx, np.arange(1, 101))
            or video.unreadable.any() or not np.allclose(video.time_s, np.arange(100) / 10, atol=1e-7, rtol=0)):
        raise ValueError(f"{cache_id}/{clip}: invalid smoke timeline or image geometry")
    return video, sha256(marker_path)


def overlay(clip: str, frame: int, gt: np.ndarray, det: np.ndarray | None, detector: str) -> None:
    path = IMAGES / clip / f"{frame}.jpg"
    with Image.open(path) as original:
        panel = original.convert("RGB")
    draw = ImageDraw.Draw(panel)
    for row in gt:
        x, y, w, h = row[2:6]
        draw.rectangle((x, y, x + w, y + h), outline="lime", width=6)
        draw.text((x + 2, y + 2), f"GT {int(row[1])}", fill="white", stroke_width=2, stroke_fill="black")
    if det is not None:
        for x, y, x2, y2, score, _cls in det:
            draw.rectangle((x, y, x2, y2), outline="magenta", width=3)
            draw.text((x + 2, y2 - 12), f"{score:.2f}", fill="white", stroke_width=2, stroke_fill="black")
    caption = f"{clip} | frame {frame:03d} | GT green | {detector}"
    if det is not None:
        caption += " | COCO motorcycle magenta, conf > 0.10"
    draw.text((12, 10), caption, fill="white", stroke_fill="black", stroke_width=2)
    dest = OUT / detector
    dest.mkdir(parents=True, exist_ok=True)
    panel.save(dest / f"{clip}.jpg", quality=90)


def stats(counts: dict, tp: dict, gt_count: int, n_frames: int) -> dict:
    return {"gt_boxes": gt_count, "frames": n_frames, "per_conf_strict_gt": {
        str(t): {"det_boxes": counts[t], "tp": tp[t], "recall": tp[t] / gt_count if gt_count else None}
        for t in THRESHOLDS}}


def review(cache_ids: dict[str, str]) -> dict:
    choices = [row for row in manifest()["videos"] if row["rank_in_group"] == 1]
    if len(choices) != 12:
        raise ValueError("expected one selected clip in each of 12 groups")
    root = config.PROCESSED_DIR / SPLIT_SMOKE
    gt_log = json.loads((root / "gt_log.json").read_text(encoding="utf-8"))
    if gt_log["manifest_sha256"] != sha256(OUTPUT) or set(gt_log["videos"]) != {c["video_id"] for c in choices}:
        raise ValueError("wrong smoke GT manifest")
    out = {"split": SPLIT_SMOKE, "manifest_sha256": sha256(OUTPUT), "cache_ids": cache_ids,
           "scope": "motorcycle COCO 3, strict confidence cut, max 300 detections, one-to-one IoU >= 0.5",
           "overlay_rule": "nearest annotated frame to 50, earlier on ties; all frames used for recall",
           "clips": {}, "detectors": {}}
    for row in choices:
        clip = row["video_id"]
        path = root / clip / "gt/gt.txt"
        digest = sha256(path)
        if digest != gt_log["videos"][clip]["gt_sha256"]:
            raise ValueError(f"{clip}: GT hash differs from source prep")
        data = np.loadtxt(path, delimiter=",", ndmin=2)
        frame = min(set(data[:, 0].astype(int)), key=lambda f: (abs(f - 50), f)) if len(data) else 50
        shown_gt = data[data[:, 0] == frame]
        overlay(clip, frame, shown_gt, None, "gt")
        by_det = {}
        for det, cache_id in cache_ids.items():
            video, marker_sha = verified_cache(cache_id, det, clip, digest)
            counts = dict.fromkeys(THRESHOLDS, 0)
            tp = dict.fromkeys(THRESHOLDS, 0)
            for k, _t, _unreadable, detections in video.iter_frames():
                gt_xywh = data[data[:, 0] == k, 2:6]
                for threshold in THRESHOLDS:
                    dd = detections[detections[:, 4] > threshold][:300]
                    counts[threshold] += len(dd)
                    tp[threshold] += matches(gt_xywh, dd)
                    if k == frame and threshold == 0.10:
                        overlay(clip, frame, shown_gt, dd, det)
            by_det[det] = {**stats(counts, tp, len(data), 100), "cache_marker_sha256": marker_sha}
        out["clips"][clip] = {"group": row["group"], "gt_sha256": digest, "gt_boxes": len(data),
                              "overlay_frame": int(frame), "detectors": by_det}
    for det in cache_ids:
        clips = [c["detectors"][det] for c in out["clips"].values()]
        def total(t: float, metric: str) -> int:
            return sum(c["per_conf_strict_gt"][str(t)][metric] for c in clips)
        out["detectors"][det] = stats({t: total(t, "det_boxes") for t in THRESHOLDS},
                                      {t: total(t, "tp") for t in THRESHOLDS},
                                      sum(c["gt_boxes"] for c in clips), sum(c["frames"] for c in clips))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gt-only", action="store_true")
    group.add_argument("--cache", action="append", metavar="DETECTOR=CACHE_ID")
    args = parser.parse_args()
    cache_ids = dict(keyval.split("=", 1) for keyval in args.cache or [])
    if args.cache and (set(cache_ids) != set(DETECTORS) or len(args.cache) != 3 or
                       any(Path(v).name != v or not v for v in cache_ids.values())):
        parser.error("pass each of the three detectors once, with a smoke cache ID")
    out = review(cache_ids)
    out_path = OUT / ("gt.json" if args.gt_only else "recall.json")
    r64.ghi_json_nguyen_tu(out_path, out)
    print(out_path, json.dumps(out["detectors"], ensure_ascii=False))


if __name__ == "__main__":
    main()
