"""Cache common YOLO detections on the selected contiguous sequences."""

import configparser
import hashlib
import os
from pathlib import Path
import sys
from zipfile import ZipFile

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = Path("F:/LABOTARY_data/goc_tai_ve/FastTrack-R115")
KITTI = Path("D:/LABOTARY/ByteTrack + EKF + Kitty/datasets/KITTI")
FASTTRACK = ("task_day_occlusion", "task_day_right_turn_occlusion", "task_tunnel")
KITTI_SEQUENCES = ("0001", "0011", "0020")
COCO_VEHICLES = (2, 3, 5, 7)
WEIGHTS = ROOT / "models/yolov8n.pt"
WEIGHTS_SHA = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
FASTTRACK_SHA = {
    "task_day_occlusion": "33ab4cd7986c360968836b8a99a9603c8acee433abc186c155064eb7b8530dad",
    "task_day_right_turn_occlusion": "63d16f864e72e62e52773cb2f303c55a88fed831099055739f0f5d07b982ae6f",
    "task_tunnel": "5daa3c2ad1cff3a7681b7ce28d6a4c176aaaa4e9282bd6fbb4de5401738e6a5a",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fasttrack_info(archive, name):
    metadata = configparser.ConfigParser()
    metadata.read_string(archive.read(f"{name}/seqinfo.ini").decode())
    seq = metadata["Sequence"]
    frames, fps = seq.getint("seqLength"), seq.getint("frameRate")
    shape = (seq.getint("imHeight"), seq.getint("imWidth"))
    secondary = dict(line.split(": ") for line in
                     archive.read(f"{name}/video/video_info.txt").decode().splitlines())
    assert (int(secondary["num_frames"]), int(secondary["fps"])) == (frames, fps)
    assert seq["name"] == name and seq["imDir"] == "img1"
    files = [f"{name}/img1/{i:06d}.jpg" for i in range(1, frames + 1)]
    assert sorted(p for p in archive.namelist() if p.startswith(f"{name}/img1/")
                  and p.endswith(".jpg")) == files
    return files, fps, shape


def source_frames(name, fasttrack_metadata=fasttrack_info):
    if name.startswith("kitti-"):
        sequence = name.removeprefix("kitti-")
        folder = KITTI / "data_tracking_image_2/training/image_02" / sequence
        files = sorted(folder.glob("*.png"))
        assert [x.stem for x in files] == [f"{i:06d}" for i in range(len(files))]
        for path in files:
            yield cv2.imread(str(path))
        return
    path = DATA / f"{name}.zip"
    assert sha256(path) == FASTTRACK_SHA[name]
    with ZipFile(path) as archive:
        files, _, shape = fasttrack_metadata(archive, name)
        for filename in files:
            image = cv2.imdecode(np.frombuffer(archive.read(filename), np.uint8), cv2.IMREAD_COLOR)
            assert image is not None and image.shape[:2] == shape, filename
            yield image


def run(names, fasttrack_metadata=fasttrack_info):
    from itertools import islice
    from ultralytics import YOLO

    assert sha256(WEIGHTS) == WEIGHTS_SHA
    model = YOLO(str(WEIGHTS))
    for name in names:
        destination = DATA / f"{name}_yolov8n.npz"
        if destination.is_file():
            print("EXISTS", destination, flush=True)
            continue
        frames, detections, shape = source_frames(name, fasttrack_metadata), [], None
        for batch in iter(lambda: list(islice(frames, 4)), []):
            assert all(frame is not None for frame in batch)
            if shape is None:
                shape = batch[0].shape[:2]
            assert all(frame.shape[:2] == shape for frame in batch)
            results = model.predict(batch, imgsz=640, conf=.05, iou=.7,
                                    max_det=3000, half=False, rect=True, device=0, verbose=False)
            assert len(results) == len(batch)
            for result in results:
                array = result.boxes.data.cpu().numpy().astype(np.float32)
                assert len(array) < 3000 and np.isfinite(array).all()
                detections.append(array)
            if len(detections) % 400 == 0:
                print(name, len(detections), flush=True)
        assert detections
        partial = destination.with_suffix(".partial.npz")
        np.savez_compressed(partial, shape=shape, **{str(i): a for i, a in enumerate(detections)})
        partial.replace(destination)
        print("CACHE", name, len(detections), sha256(destination), flush=True)


if __name__ == "__main__":
    os.environ["YOLO_CONFIG_DIR"] = str(DATA / "ultralytics_settings")
    run(sys.argv[1:] or list(FASTTRACK) + [f"kitti-{s}" for s in KITTI_SEQUENCES])
