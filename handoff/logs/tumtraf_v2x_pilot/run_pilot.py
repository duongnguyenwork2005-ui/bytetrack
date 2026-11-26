"""Camera-only pilot with audited image timestamps, 2D boxes and 3D ground coordinates."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "src"), str(HERE.parent / "review99_roundabout_turns")]
import config
import det_cache
from audit_turns import runs, trace
from mot_datasets.base import FrameRef
from r64_ngoai.danh_gia_ngoai import BO_LOC, LOP_TRACKER, bang_anh_xa_coco, detector_r54, sha256_file
import run_tracking
import run_trackeval as rte

SPLIT = "TUMTrafV2X-pilot"
SEQ = "camera"


def load_inputs(images_csv, gt_csv, fps):
    with images_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    timestamps = np.asarray([int(row["timestamp_ns"]) for row in rows], dtype=np.int64)
    if len(rows) < 2 or fps <= 0 or not np.allclose(np.diff(timestamps) * 1e-9, 1 / fps, rtol=0.25, atol=0):
        raise ValueError("Image timestamps must cover consecutive frames at the verified labeled FPS")
    frames = []
    shape = None
    for i, row in enumerate(rows):
        path = (images_csv.parent / row["image"]).resolve()
        image = cv2.imread(str(path))
        if image is None or (shape is not None and image.shape[:2] != shape):
            raise ValueError(f"Unreadable image or inconsistent size: {path}")
        shape = image.shape[:2]
        frames.append(FrameRef(i + 1, path, int(timestamps[i] - timestamps[0]) * 1e-9))
    if len({frame.path for frame in frames}) != len(frames):
        raise ValueError("Image paths must be unique")
    with gt_csv.open(encoding="utf-8-sig", newline="") as stream:
        gt = np.asarray([[float(row[key]) for key in ("frame", "id", "x", "y", "w", "h", "world_x", "world_y")]
                         for row in csv.DictReader(stream)], dtype=np.float64)
    if not len(gt) or not np.isfinite(gt).all():
        raise ValueError("GT must contain finite 2D and world-plane coordinates")
    frame_id = gt[:, :2]
    x, y, w, h = gt[:, 2:6].T
    if (frame_id != frame_id.astype(int)).any() or (frame_id < 1).any() or gt[:, 0].max() > len(frames):
        raise ValueError("Invalid GT frame or ID")
    if (len(np.unique(frame_id, axis=0)) != len(gt) or (np.minimum(x, y) < 0).any()
            or (np.minimum(w, h) <= 0).any() or (x + w > shape[1]).any() or (y + h > shape[0]).any()):
        raise ValueError("Duplicate or out-of-image GT box")
    return frames, shape, gt


def ground_turns(gt, fps, nframes):
    ids = np.unique(gt[:, 1]).astype(int)
    mask = np.zeros((nframes, len(ids)), dtype=bool)
    radius = round(fps)
    sustain = max(1, round(fps * 0.5))
    episodes = []
    for column, track_id in enumerate(ids):
        track = gt[gt[:, 1] == track_id]
        track = track[np.argsort(track[:, 0])]
        for segment in np.split(track, np.flatnonzero(np.diff(track[:, 0]) != 1) + 1):
            if len(segment) < 2 * radius + sustain:
                continue
            xy = segment[:, -2:]
            before = xy[radius:-radius] - xy[:-2 * radius]
            after = xy[2 * radius:] - xy[radius:-radius]
            angle = np.degrees(np.arctan2(before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0],
                                          np.sum(before * after, axis=1)))
            moving = np.minimum(np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)) >= 1.0
            for direction in (1, -1):
                for start, stop in runs(moving & (direction * angle >= 25)):
                    if stop - start < sustain:
                        continue
                    first, last = segment[[start + radius, stop + radius - 1], 0].astype(int)
                    mask[first - 1:last, column] = True
                    episodes.append({"id": int(track_id), "first": int(first), "last": int(last)})
    return ids, mask, episodes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    frames, shape, gt = load_inputs(args.frames, args.gt, args.fps)
    ids, turn_mask, episodes = ground_turns(gt, args.fps, len(frames))
    if not episodes:
        raise ValueError("No qualifying turn visible in this camera. Check scene, camera and GT first.")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config.PROCESSED_DIR = out / "processed"
    config.RESULTS_DIR = out
    directory = config.PROCESSED_DIR / SPLIT / SEQ
    (directory / "gt").mkdir(parents=True)
    mot = np.column_stack((gt[:, :6], np.ones((len(gt), 3))))
    np.savetxt(directory / "gt" / "gt.txt", mot, fmt=["%d", "%d", "%.6f", "%.6f", "%.6f", "%.6f", "%d", "%d", "%d"], delimiter=",")
    (directory / "seqinfo.ini").write_text(
        f"[Sequence]\nname={SEQ}\nframeRate={args.fps}\nseqLength={len(frames)}\nimWidth={shape[1]}\nimHeight={shape[0]}\n",
        encoding="utf-8")
    detector = detector_r54("yolov8n_coco")
    ckpt = ROOT / detector["checkpoint"]
    spec = det_cache.DetectorSpec(backend="ultralytics", weights=str(ckpt), weights_sha256=detector["sha256"],
                                  imgsz=640, conf=0.05, iou=0.7, max_det=3000, batch=1, half=False,
                                  device=args.device, mask_ignored_regions=False, dataset=SPLIT, part="pilot")
    spec.fill_env()
    model = YOLO(str(ckpt))
    assert bang_anh_xa_coco(model.names, detector["loai_checkpoint"]) is None
    detections = det_cache.build_video_yolo(model, frames, [], spec, shape)
    if detections["unreadable"].any() or detections["truncated"].any():
        raise ValueError("Invalid detector cache: unreadable or truncated frames")
    cache_dir = out / "cache"
    det_cache.write_meta(cache_dir, spec, mode="create")
    det_cache.save_video(cache_dir, SEQ, detections, spec=spec)
    cache = cache_dir / f"{SEQ}.npz"
    present = np.zeros_like(turn_mask)
    present[gt[:, 0].astype(int) - 1, np.searchsorted(ids, gt[:, 1])] = True
    data = {"frames": len(frames), "fps": args.fps, "turn_episodes": episodes,
            "turn_gt": int(turn_mask.sum()), "gt_sha256": sha256_file(args.gt),
            "frames_csv_sha256": sha256_file(args.frames), "cache_sha256": sha256_file(cache), "filters": {}}
    for method in ("cv", "ekf", "ukf"):
        yaml = BO_LOC[method][0]
        cfg = run_tracking.load_tracker_cfg(yaml, {"track_buffer": round(1.2 * args.fps)})
        vd = det_cache.VideoDetections(cache, 0.25, LOP_TRACKER, max_det=300)
        track_dir = config.PROCESSED_DIR / "trackers" / SPLIT / method
        run_tracking.run_video_to_dir(track_dir, SEQ, vd, cfg, 1 / args.fps, "steps")
        results, dataset = rte.run(SPLIT, method, [SEQ])
        scores = rte.flatten_results(results, method).set_index("video").loc[SEQ].to_dict()
        raw = dataset.get_raw_seq_data(method, SEQ)
        prepared = dataset.get_preprocessed_seq_data(raw, "pedestrian")
        assert len(ids) == prepared["num_gt_ids"]
        for original, compact in zip(raw["gt_ids"], prepared["gt_ids"]):
            assert np.array_equal(original, ids[compact])
        matched, changed = trace(prepared, present.shape)
        assert changed.sum() == scores["IDSW"] and (present & ~matched).sum() == scores["FN"]
        data["filters"][method] = {
            "all": scores, "turn_idsw": int((changed & turn_mask).sum()),
            "turn_fn": int((~matched & turn_mask).sum()),
            "yaml_sha256": sha256_file(Path(run_tracking.resolve_tracker_yaml(yaml))),
        }
    (out / "result.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(out / "result.json")


if __name__ == "__main__":
    main()
