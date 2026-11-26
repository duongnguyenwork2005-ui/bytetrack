"""Build a separate UTUAV GT-synthetic detection cache for tracker diagnostics.

Uses the repository's existing jitter/dropout generator. This is an oracle-input
ablation, not a YOLO detector result or a final dataset benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import det_cache  # noqa: E402
from mot_datasets import FrameRef  # noqa: E402


CLASS_TO_MODEL = {"car": 0, "bus": 2, "motorbike": 3}


def replay_source_ids(gt: pd.DataFrame, frames: list[FrameRef], spec: det_cache.DetectorSpec) -> tuple[np.ndarray, np.ndarray]:
    """Replay repository GT RNG exactly, carrying the originating track ID."""
    video = str(gt["video"].iloc[0]) if len(gt) else ""
    seed = np.random.SeedSequence([spec.gt_seed, int(hashlib.sha1(video.encode()).hexdigest(), 16) % (2**31)])
    rng = np.random.default_rng(seed)
    by_frame = {key: value for key, value in gt.groupby("frame")}
    chunks, id_chunks = [], []
    for frame in frames:
        group = by_frame.get(frame.index)
        if group is None:
            continue
        rows = []
        for _, box in group.iterrows():
            coco = CLASS_TO_MODEL.get(str(box["category"]))
            if coco is None or rng.random() < spec.gt_dropout:
                continue
            width, height = float(box["bb_width"]), float(box["bb_height"])
            jx, jy = rng.normal(0, spec.gt_jitter * width), rng.normal(0, spec.gt_jitter * height)
            jw, jh = rng.normal(0, spec.gt_jitter * width), rng.normal(0, spec.gt_jitter * height)
            x1 = float(box["bb_left"]) + jx
            y1 = float(box["bb_top"]) + jy
            x2 = x1 + max(2.0, width + jw)
            y2 = y1 + max(2.0, height + jh)
            x1, x2 = np.clip([x1, x2], 0, 3840)
            y1, y2 = np.clip([y1, y2], 0, 2160)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            conf = rng.uniform(*spec.gt_conf_range)
            rows.append([x1, y1, x2, y2, conf, float(coco), float(box["track_id"])])
        if rows:
            arr = np.array(rows, dtype=np.float32)
            arr = arr[np.argsort(-arr[:, 4], kind="stable")]
            arr = arr[arr[:, 4] > spec.conf][:spec.max_det]
            chunks.append(arr[:, :6])
            id_chunks.append(arr[:, 6].astype(np.int32))
    data = np.concatenate(chunks) if chunks else np.zeros((0, 6), dtype=np.float32)
    ids = np.concatenate(id_chunks) if id_chunks else np.zeros(0, dtype=np.int32)
    return data, ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--extracted", type=Path, required=True)
    ap.add_argument("--cache-id", required=True)
    ap.add_argument("--jitter", type=float, default=0.03)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-gap-frames", type=int, default=0)
    ap.add_argument("--block-gap-modulus", type=int, default=2)
    args = ap.parse_args()
    plan = json.loads(args.selection.read_text(encoding="utf-8"))
    if args.block_gap_frames and args.block_gap_modulus < 1:
        raise ValueError("block-gap-modulus must be positive")
    block_policy = (f"sha256(sequence:track_id) mod {args.block_gap_modulus}=0; "
                    f"{args.block_gap_frames} frames at track median; min track length 20"
                    if args.block_gap_frames else "none")
    spec = det_cache.DetectorSpec(
        backend="gt-synthetic", weights="GT-derived; no detector weights",
        dataset="utuav_pilot", part="validation", conf=0.05, max_det=3000,
        gt_jitter=args.jitter, gt_dropout=args.dropout, gt_seed=args.seed,
        gt_conf_range=(0.3, 0.95), device="cpu",
        versions={"block_gap_policy": block_policy},
    ).fill_env()
    cache_dir = det_cache.CACHE_ROOT / args.cache_id
    det_cache.write_meta(cache_dir, spec, mode="create")
    summary = {}
    for seq in plan["sequences"]:
        name = seq["name"]
        first, last = seq["first_source_frame"], seq["last_source_frame"]
        rows = []
        for i, frame_number in enumerate(range(first, last + 1), start=1):
            prefix = "v1" if seq["scene"] == "B" else "v2"
            label = args.extracted / name / "annotations" / f"{prefix}_{frame_number:06d}.xml"
            root = ET.parse(label).getroot()
            for obj in root.findall("object"):
                category = obj.findtext("name", default="")
                if category not in CLASS_TO_MODEL:
                    continue
                track_id = int(obj.findtext("./attributes/attribute[name='track_id']/value"))
                box = obj.find("bndbox")
                x1, y1 = float(box.findtext("xmin")), float(box.findtext("ymin"))
                x2, y2 = float(box.findtext("xmax")), float(box.findtext("ymax"))
                rows.append({"video": name, "frame": i, "bb_left": x1, "bb_top": y1,
                             "bb_width": x2 - x1, "bb_height": y2 - y1, "category": category,
                             "track_id": track_id})
        gt = pd.DataFrame(rows)
        removed_gt = 0
        removed_detections = 0
        selected_tracks = 0
        gaps = {}
        if args.block_gap_frames:
            for track_id, group in gt.groupby("track_id"):
                unique_frames = sorted(group["frame"].unique())
                if len(unique_frames) < 20:
                    continue
                digest = hashlib.sha256(f"{name}:{track_id}".encode()).digest()
                if int.from_bytes(digest[:8], "big") % args.block_gap_modulus:
                    continue
                selected_tracks += 1
                center = unique_frames[len(unique_frames) // 2]
                start = center - args.block_gap_frames // 2
                gaps[track_id] = (start, start + args.block_gap_frames - 1)
            removed_gt = int(sum((track_id in gaps and gaps[track_id][0] <= frame <= gaps[track_id][1])
                                 for frame, track_id in zip(gt["frame"], gt["track_id"])))
        frames = [FrameRef(index=i, path=None, time_s=(i - 1) / 30)
                  for i in range(1, last - first + 2)]
        arrays = det_cache.build_video_gt(gt, frames, spec, (2160, 3840), CLASS_TO_MODEL)
        replayed, source_ids = replay_source_ids(gt, frames, spec)
        if not np.array_equal(arrays["data"], replayed):
            raise AssertionError(f"GT generator replay mismatch for {name}")
        if gaps:
            chunks, counts = [], []
            offset = 0
            for frame, count in zip(arrays["frame_idx"], arrays["n_per_frame"]):
                chunk = arrays["data"][offset:offset + count]
                ids = source_ids[offset:offset + count]
                keep = np.array([not (int(track_id) in gaps and
                                      gaps[int(track_id)][0] <= frame <= gaps[int(track_id)][1])
                                 for track_id in ids], dtype=bool)
                chunks.append(chunk[keep])
                counts.append(int(keep.sum()))
                offset += count
            removed_detections = int(len(arrays["data"]) - sum(counts))
            arrays["data"] = np.concatenate(chunks)
            arrays["n_per_frame"] = np.array(counts, dtype=np.int32)
        extra = {"input_type": "GT-synthetic diagnostic; NOT YOLO", "source_frames": [first, last],
                 "fps_assumption": 30, "selection_file": str(args.selection),
                 "block_gap_policy": block_policy, "block_gap_tracks": selected_tracks,
                 "block_gap_removed_gt_detections": removed_gt,
                 "block_gap_removed_cached_detections": removed_detections}
        det_cache.save_video(cache_dir, name, arrays, extra=extra, spec=spec)
        summary[name] = {"frames": len(frames), "moving_gt": len(gt),
                         "synthetic_detections": len(arrays["data"]),
                         "block_gap_tracks": selected_tracks, "block_gap_removed_gt": removed_gt,
                         "block_gap_removed_cached": removed_detections}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
