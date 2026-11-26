"""Verify a public RideSafe MP4/XML pair and run a motorcycle-only pilot."""
import hashlib
import json
from pathlib import Path
import sys

import cv2
from lxml import etree
import numpy as np
from scipy.optimize import linear_sum_assignment
from trackeval.datasets._base_dataset import _BaseDataset
from trackeval.metrics import CLEAR, HOTA, Identity
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from run_tracking import load_tracker_cfg, track_video  # noqa: E402

CHECKPOINTS = {
    "yolov8n_coco": "models/yolov8n.pt",
    "yolov8n_mixed5": "models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/best.pt",
    "yolo11n_mixed5": "models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt",
}
TRACKERS = {
    "kf": "configs/phase2/trackers/bytetrack_cv.yaml",
    "ekf": "configs/bytetrack_ekf_ctrv.yaml",
    "ukf": "configs/bytetrack_ukf_ctrv.yaml",
    "pf_motion2": "configs/phase2/trackers/r72/motion200.yaml",
    "pf_r95": "configs/phase2/trackers/r95/pf_reacquire.yaml",
}


def read_annotations(path):
    root = etree.parse(str(path))
    info = root.find(".//frame_filter").getparent()
    stride = int(info.findtext("frame_filter").removeprefix("step="))
    start, stop = int(info.findtext("start_frame")), int(info.findtext("stop_frame"))
    size = root.find(".//original_size")
    shape = (int(size.findtext("height")), int(size.findtext("width")))
    expected = list(range(start, stop + 1, stride))
    assert len(expected) == int(info.findtext("size"))
    frames = {frame: {} for frame in expected}
    occluded = 0
    for track in root.xpath('/annotations/track[@label="motorcycle"]'):
        track_id = int(track.get("id"))
        for box in track.xpath('box[@outside="0"]'):
            frame = int(box.get("frame"))
            x1, y1, x2, y2 = (float(box.get(key)) for key in ("xtl", "ytl", "xbr", "ybr"))
            assert 0 <= x1 < x2 <= shape[1] and 0 <= y1 < y2 <= shape[0]
            assert track_id not in frames[frame]
            frames[frame][track_id] = [x1, y1, x2, y2]
            occluded += box.get("occluded") == "1"
    return frames, stride, shape, occluded


def predict_clip(video, frames, shape, preview_dir):
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert fps > 0 and max(frames) < frame_count
    checkpoints = {name: ROOT / path for name, path in CHECKPOINTS.items()}
    models = {name: YOLO(str(path)) for name, path in checkpoints.items()}
    metadata = {name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "motorcycle_class": next(i for i, label in models[name].names.items()
                                                if label.lower() in {"motorcycle", "motorbike"})}
                for name, path in checkpoints.items()}
    detections = {name: {} for name in models}
    timestamps = []
    images, indices = [], []

    def infer_batch():
        for name, model in models.items():
            motorcycle = metadata[name]["motorcycle_class"]
            batch = model.predict(images, classes=[motorcycle], imgsz=640, conf=.05, iou=.7,
                                  max_det=300, device=0, verbose=False)
            assert len(batch) == len(indices)
            for index, result in zip(indices, batch):
                arr = result.boxes.data.cpu().numpy().astype(np.float32)
                assert len(arr) < 300
                detections[name][index] = arr
        images.clear()
        indices.clear()

    decoded = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            index = decoded
            decoded += 1
            assert image.shape[:2] == shape
            if index not in frames:
                continue
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            images.append(image)
            indices.append(index)
            if index in (100, 750, 1450):
                scale = 960 / shape[1]
                preview = cv2.resize(image, (960, round(shape[0] * scale)))
                for x1, y1, x2, y2 in frames[index].values():
                    cv2.rectangle(preview, (round(x1 * scale), round(y1 * scale)),
                                  (round(x2 * scale), round(y2 * scale)), (0, 255, 0), 2)
                assert cv2.imwrite(str(preview_dir / f"gt_{index:04}.jpg"), preview)
            if len(images) == 8:
                infer_batch()
        if images:
            infer_batch()
    finally:
        cap.release()
    assert decoded == frame_count and all(rows.keys() == frames.keys() for rows in detections.values())
    stamps = np.asarray(timestamps)
    reference = 1000 * np.asarray(list(frames)) / fps
    drift = float(np.abs(stamps - reference).max())
    assert drift < 1000 / fps, f"timestamp drift {drift} ms"
    return detections, fps, decoded, drift, metadata


def recall(gt, detections, threshold):
    hits = num_gt = num_det = 0
    for index, objects in gt.items():
        boxes = np.asarray(list(objects.values()), dtype=float).reshape(-1, 4)
        prediction = detections[index][detections[index][:, 4] > threshold, :4]
        iou = _BaseDataset._calculate_box_ious(boxes, prediction, box_format="x0y0x1y1")
        valid = iou >= .5
        r, c = linear_sum_assignment(valid * (1 + iou / (1 + min(iou.shape))), maximize=True)
        hits += int(valid[r, c].sum())
        num_gt += len(boxes)
        num_det += len(prediction)
    return {"matches": hits, "gt": num_gt, "detections": num_det, "recall": hits / num_gt}


class CachedMotorcycles:
    def __init__(self, detections, shape, fps):
        self.detections = detections
        self.orig_shape = shape
        self.fps = fps

    def iter_frames(self):
        for step, boxes in enumerate(self.detections.values()):
            yield step + 1, step / self.fps, False, boxes[boxes[:, 4] > .25]


def mot_scores(gt, mot):
    pred = {frame: group for frame, group in mot.groupby("frame")}
    gt_names = sorted({track for frame in gt.values() for track in frame})
    tr_names = sorted(mot["id"].unique())
    gt_map, tr_map = ({track: i for i, track in enumerate(names)} for names in (gt_names, tr_names))
    gt_ids, tr_ids, similarities = [], [], []
    for step, objects in enumerate(gt.values(), 1):
        gt_ids.append(np.array([gt_map[i] for i in objects], dtype=int))
        gt_boxes = np.asarray(list(objects.values()), dtype=float).reshape(-1, 4)
        rows = pred.get(step)
        if rows is None:
            tr_ids.append(np.empty(0, dtype=int))
            tr_boxes = np.empty((0, 4))
        else:
            assert not rows["id"].duplicated().any()
            tr_ids.append(np.array([tr_map[i] for i in rows["id"]], dtype=int))
            tr_boxes = rows[["bb_left", "bb_top", "bb_width", "bb_height"]].to_numpy()
            tr_boxes[:, 2:] += tr_boxes[:, :2]
        similarities.append(_BaseDataset._calculate_box_ious(gt_boxes, tr_boxes, box_format="x0y0x1y1"))
    data = dict(num_timesteps=len(gt), gt_ids=gt_ids, tracker_ids=tr_ids, num_gt_ids=len(gt_names),
                num_tracker_ids=len(tr_names), num_gt_dets=sum(map(len, gt_ids)),
                num_tracker_dets=len(mot), similarity_scores=similarities)
    hota = HOTA().eval_sequence(data)
    clear = CLEAR({"PRINT_CONFIG": False}).eval_sequence(data)
    identity = Identity({"PRINT_CONFIG": False}).eval_sequence(data)
    return {"hota": float(hota["HOTA"].mean()), "idf1": float(identity["IDF1"]),
            "idsw": int(clear["IDSW"]), "fn": int(clear["CLR_FN"]), "fp": int(clear["CLR_FP"])}


def main():
    root = Path(sys.argv[1])
    manifest = json.loads((root / "source.json").read_text())
    gt, stride, shape, occluded = read_annotations(root / manifest["labels"]["file"])
    detections, fps, decoded, drift, checkpoints = predict_clip(root / manifest["video"]["file"], gt, shape, root)
    effective_fps = fps / stride
    scores = {name: {str(th): recall(gt, rows, th) for th in (.05, .25)} for name, rows in detections.items()}
    result = {"clip": manifest["clip"], "decoded_frames": decoded, "source_fps": fps, "stride": stride,
              "effective_fps": effective_fps, "timestamp_max_abs_error_ms": drift, "shape_hw": shape,
              "labeled_frames": len(gt), "motorcycle_boxes": sum(map(len, gt.values())),
              "motorcycle_ids": len({track for frame in gt.values() for track in frame}),
              "occluded_boxes": occluded, "checkpoints": checkpoints, "recall": scores}
    out = root / "pilot.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if scores["yolov8n_coco"]["0.05"]["recall"] < .30:
        return

    cache = CachedMotorcycles(detections["yolov8n_coco"], shape, effective_fps)
    tracked = []
    for name, path in TRACKERS.items():
        cfg = load_tracker_cfg(ROOT / path, {"track_buffer": round(1.2 * effective_fps)})
        for seed in ((0, 1, 2) if name.startswith("pf_") else (None,)):
            mot, classes, timing = track_video(cache, cfg, dt_nominal=1 / effective_fps,
                                             elapsed_mode="steps", seed=seed, video=manifest["clip"])
            assert len(timing) == len(gt) and set(classes["cls"]) <= {checkpoints["yolov8n_coco"]["motorcycle_class"]}
            row = {"tracker": name, "seed": seed, **mot_scores(gt, mot)}
            mot.to_csv(root / f"mot_{name}_{seed}.csv", index=False)
            tracked.append(row)
            print(json.dumps(row), flush=True)
    result["tracking"] = tracked
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
