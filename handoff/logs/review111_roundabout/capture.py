"""Infer shared detections for RoundaboutHD frames 3001–9000, one cache per chunk."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import monotonic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SETTINGS = HERE / "ultralytics"
SETTINGS.mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(SETTINGS)
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from ultralytics import YOLO

SOURCE = Path("C:/LABOTARY_data/RoundaboutHD")
PREFIX = ROOT / "data/interim/det_cache/r64_roundabout_yolov8n_coco_c0.05"
CAMERAS = tuple(f"imagesc{index:03d}" for index in range(1, 5))
SHAPE = (2160, 3840)
FIRST = 3001
LAST = 9000
CHUNK = 600


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def predict(model, images):
    results = model.predict(images, conf=.05, iou=.7, imgsz=640, classes=None, max_det=3000,
                            agnostic_nms=False, augment=False, rect=True, device=0, verbose=False)
    assert len(results) == len(images)
    detections = [r.boxes.data.cpu().numpy().astype(np.float32) for r in results]
    assert all(d.ndim == 2 and d.shape[1] == 6 and len(d) < 3000 and np.isfinite(d).all() for d in detections)
    return detections


def frames(cap, count):
    batch = []
    for _ in range(count):
        ok, image = cap.read()
        assert ok and image is not None and image.shape[:2] == SHAPE
        batch.append(image)
    return batch


def match_prefix(model, cap, camera):
    prefix_frames = frames(cap, 16)
    actual = [d for i in range(0, 16, 4) for d in predict(model, prefix_frames[i:i + 4])]
    with np.load(PREFIX / (camera + ".npz")) as cache:
        offsets = np.r_[0, np.cumsum(cache["n_per_frame"][:16])]
        expected = [cache["data"][a:b] for a, b in zip(offsets, offsets[1:])]
    def equivalent(a, b):
        return (a.shape == b.shape and np.array_equal(a[:, 5], b[:, 5])
                and np.allclose(a[:, :4], b[:, :4], rtol=0, atol=.02)
                and np.allclose(a[:, 4], b[:, 4], rtol=0, atol=1e-4))
    batch_ok = sum(equivalent(a, b) for a, b in zip(actual, expected))
    if batch_ok == 16:
        return 4, batch_ok
    single = [predict(model, [image])[0] for image in prefix_frames]
    assert all(equivalent(a, b) for a, b in zip(single, expected)), camera
    return 1, batch_ok


def main():
    output = HERE / "cache"
    output.mkdir(exist_ok=True)
    weights = ROOT / "models/yolov8n.pt"
    meta = read(PREFIX / "meta.json")
    assert sha256(weights.read_bytes()).hexdigest() == meta["weights_sha256"]
    assert (meta["imgsz"], meta["conf"], meta["iou"], meta["rect"], meta["half"]) == (640, .05, .7, True, False)
    cv2.setNumThreads(1)
    model = YOLO(str(weights))
    sources = {}
    for camera in CAMERAS:
        start = monotonic()
        video = SOURCE / camera / "video.mp4"
        cap = cv2.VideoCapture(str(video))
        try:
            assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) == LAST
            assert abs(cap.get(cv2.CAP_PROP_FPS) - 15.) < 1e-5
            batch, checked = match_prefix(model, cap, camera)
            sources[camera] = {"video_bytes": video.stat().st_size, "batch": batch, "prefix_match_of_16_batch4": checked}
            print("PREFLIGHT", camera, sources[camera], flush=True)
            for index in range(16, FIRST - 1):
                assert cap.grab(), (camera, index + 1)
            assert cap.get(cv2.CAP_PROP_POS_FRAMES) == FIRST - 1
            for first in range(FIRST, LAST + 1, CHUNK):
                path = output / f"{camera}_{first:04d}.npz"
                count = min(CHUNK, LAST - first + 1)
                if path.exists():
                    for _ in range(count):
                        assert cap.grab()
                    continue
                detections = []
                for offset in range(0, count, batch):
                    detections.extend(predict(model, frames(cap, min(batch, count - offset))))
                assert len(detections) == count and cap.get(cv2.CAP_PROP_POS_FRAMES) == first + count - 1
                pending = path.with_suffix(".tmp.npz")
                np.savez_compressed(pending, frame_idx=np.arange(first, first + count, dtype=np.int32),
                                    n_per_frame=np.array([len(d) for d in detections], dtype=np.int32),
                                    data=np.concatenate(detections))
                pending.replace(path)
                print("CACHE", camera, first, first + count - 1, f"{monotonic() - start:.1f}s", flush=True)
            assert cap.get(cv2.CAP_PROP_POS_FRAMES) == LAST
        finally:
            cap.release()
    (HERE / "capture.json").write_text(json.dumps({"weights_sha256": meta["weights_sha256"], "sources": sources,
        "sparse_cache_chunks": 40, "frames": 4 * (LAST - FIRST + 1)}, indent=2), encoding="utf-8")
    print("INFERENCE_COMPLETE", 4 * (LAST - FIRST + 1), flush=True)


if __name__ == "__main__":
    main()
