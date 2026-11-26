"""Preselected METEOR clip: native track labels, shared detector and MOT scoring."""

from collections import Counter, defaultdict
import configparser
import hashlib
import json
import os
from pathlib import Path
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SETTINGS = OUT / "ultralytics_settings"
SETTINGS.mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(SETTINGS)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/r64_ngoai"),
                str(OUT.parent / "review99_roundabout_turns")]
import cv2
import numpy as np
import trackeval
import boot_stats
import run_tracking as rt
import run_trackeval as rte
import danh_gia_ngoai as r64
from audit_turns import labels_for_camera, trace
from trackeval.datasets._base_dataset import _BaseDataset
from backend import RBPFByteTracker

DEST = OUT / "meteor_pilot"
SPLIT = "METEOR-R1122wheel"
VIDEO = "REC_2020_07_12_02_55_27_F"


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class Cache:
    orig_shape = (1080, 1920)

    def __init__(self, arrays, fps):
        self.arrays, self.fps = arrays, fps

    def iter_frames(self):
        for i, arr in enumerate(self.arrays):
            keep = (arr[:, 4] > .25) & (arr[:, 5] == 3)
            yield i + 1, i / self.fps, False, arr[keep][:300]


def prepare():
    source = json.loads((OUT / "meteor_video_source.json").read_text())
    audit = json.loads((OUT / "meteor_label_audit.json").read_text())
    capture = cv2.VideoCapture(source["video"])
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    shape = (int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    capture.release()
    assert count == 1800 and abs(fps - 30) < .001 and shape == Cache.orig_shape
    rows, classes = [], defaultdict(set)
    with ZipFile(audit["downloaded_label_archive"]) as archive:
        xmls = sorted(name for name in archive.namelist() if name.endswith(".xml"))
        assert len(xmls) == count
        for frame, name in enumerate(xmls, 1):
            assert Path(name).stem == f"frame_{frame - 1:06}"
            document = ET.fromstring(archive.read(name))
            for obj in document.findall("object"):
                attributes = {a.findtext("name"): a.findtext("value")
                              for a in obj.findall("attributes/attribute")}
                identity, category = int(attributes["track_id"]), obj.findtext("name")
                classes[identity].add(category)
                if category not in ["MotorBike", "Scooter"]:
                    continue
                x1, y1, x2, y2 = [float(obj.findtext("bndbox/" + k))
                                  for k in ["xmin", "ymin", "xmax", "ymax"]]
                assert 0 <= x1 < x2 <= shape[1] and 0 <= y1 < y2 <= shape[0]
                rows.append([frame, identity, x1, y1, x2 - x1, y2 - y1,
                             1, 1, 1 - int(obj.findtext("occluded"))])
    for categories in classes.values():
        assert not (categories & {"MotorBike", "Scooter"} and
                    categories - {"MotorBike", "Scooter"}), categories
    gt = np.asarray(rows)
    assert len(np.unique(gt[:, :2], axis=0)) == len(gt) == 2023
    base = DEST / "ground_truth" / SPLIT / VIDEO
    (base / "gt").mkdir(parents=True, exist_ok=True)
    gt_path = base / "gt/gt.txt"
    np.savetxt(gt_path, gt, delimiter=",", fmt="%.10g")
    config = configparser.ConfigParser()
    config["Sequence"] = {"name": VIDEO, "imDir": "img1", "frameRate": str(round(fps)),
                          "seqLength": str(count), "imWidth": str(shape[1]),
                          "imHeight": str(shape[0]), "imExt": ".jpg"}
    with (base / "seqinfo.ini").open("w") as stream:
        config.write(stream)
    ids, present, groups, episodes = labels_for_camera(gt_path, VIDEO, count, round(fps))
    result = {"video": VIDEO, "frames": count, "fps": fps, "shape": shape,
              "gt_boxes": len(gt), "ids": len(ids), "occluded_boxes": int((gt[:, 8] == 0).sum()),
              "angle_groups": {k: {"boxes": int((v >= 0).sum()),
                                   "episodes": sum(e["level"] == k for e in episodes)}
                               for k, v in groups.items()},
              "gt_sha256": hashlib.sha256(gt_path.read_bytes()).hexdigest(),
              "source": source, "scope": "single preselected dashcam clip, motorcycle and scooter only"}
    write(DEST / "manifest.json", result)
    return result, gt


def detection(manifest):
    from ultralytics import YOLO
    path = DEST / "detections.npz"
    checkpoint = r64.detector_r54("yolov8n_coco")
    weights = ROOT / checkpoint["checkpoint"]
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == checkpoint["sha256"]
    if path.exists():
        with np.load(path) as cache:
            return [cache[str(i)] for i in range(manifest["frames"])]
    capture = cv2.VideoCapture(manifest["source"]["video"])
    model = YOLO(str(weights))
    arrays = []
    for start in range(0, manifest["frames"], 4):
        batch = []
        for _ in range(min(4, manifest["frames"] - start)):
            ok, image = capture.read()
            assert ok and image.shape[:2] == Cache.orig_shape
            batch.append(image)
        results = model.predict(batch, imgsz=640, conf=.05, iou=.7, max_det=3000,
                                half=False, rect=True, device=0, verbose=False)
        for result in results:
            arr = result.boxes.data.cpu().numpy().astype(np.float32)
            assert len(arr) < 3000 and np.isfinite(arr).all()
            arrays.append(arr)
    assert not capture.read()[0] and len(arrays) == manifest["frames"]
    capture.release()
    np.savez_compressed(path, **{str(i): arr for i, arr in enumerate(arrays)})
    return arrays


def run():
    DEST.mkdir(exist_ok=True)
    manifest, gt = prepare()
    arrays = detection(manifest)
    hits = 0
    for frame, arr in enumerate(arrays, 1):
        objects = gt[gt[:, 0] == frame]
        boxes = arr[(arr[:, 4] > .25) & (arr[:, 5] == 3), :4].copy()
        if len(objects) and len(boxes):
            boxes[:, 2:4] -= boxes[:, :2]
            iou = _BaseDataset._calculate_box_ious(objects[:, 2:6], boxes, box_format="xywh")
            hits += int((iou.max(axis=1) >= .5).sum())
    write(DEST / "detection_coverage.json", {"gt": len(gt), "candidate_hits": hits,
          "candidate_coverage": hits / len(gt), "one_to_one": False, "confidence": .25})
    assert hits / len(gt) >= .10, "Detector coverage below registered screen"
    registry = rt.tracker_registry()
    registry["bytetrack_rbpf_ctrv"] = RBPFByteTracker
    configs = {
        "kf": rt.load_tracker_cfg("bytetrack.yaml", {"track_buffer": 36}),
        "ekf": rt.load_tracker_cfg(ROOT / "configs/bytetrack_ekf_ctrv.yaml", {"track_buffer": 36}),
        "ukf": rt.load_tracker_cfg(ROOT / "configs/bytetrack_ukf_ctrv.yaml", {"track_buffer": 36}),
    }
    for seed in range(3):
        configs[f"pf-s{seed}"] = rt.load_tracker_cfg(
            ROOT / "configs/phase2/trackers/r72/motion200.yaml", {"track_buffer": 36})
        configs[f"rbpf-s{seed}"] = rt.load_tracker_cfg(
            ROOT / "configs/phase2/trackers/r72/motion200.yaml",
            {"track_buffer": 36, "tracker_type": "bytetrack_rbpf_ctrv"})
    scores = {}
    for name, cfg in configs.items():
        path = DEST / "tracking" / SPLIT / name / "data"
        path.mkdir(parents=True, exist_ok=True)
        assert not (path / f"{VIDEO}.txt").exists()
        seed = int(name[-1]) if "-s" in name else None
        mot, classes, timing = rt.track_video(
            Cache(arrays, manifest["fps"]), cfg, 1 / manifest["fps"],
            elapsed_mode="steps", seed=seed, video=VIDEO, registry=registry)
        assert len(timing) == manifest["frames"] and not mot.duplicated(["frame", "id"]).any()
        mot.to_csv(path / f"{VIDEO}.txt", header=False, index=False, float_format="%.10f")
        ec, dc, mc = rte.build_configs(SPLIT, name, None)
        ec.update(PRINT_RESULTS=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
                  LOG_ON_ERROR=str(DEST / "error.log"))
        dc.update(GT_FOLDER=str(DEST / "ground_truth"), TRACKERS_FOLDER=str(DEST / "tracking"),
                  SEQ_INFO={VIDEO: manifest["frames"]})
        dataset = trackeval.datasets.MotChallenge2DBox(dc)
        metrics = [trackeval.metrics.HOTA(mc), trackeval.metrics.CLEAR(mc), trackeval.metrics.Identity(mc)]
        result, message = trackeval.Evaluator(ec).evaluate([dataset], metrics)
        assert all(value == "Success" for value in next(iter(message.values())).values())
        dest = DEST / f"{name}_raw.json"
        raw = boot_stats.save_raw_fields(result, name, dest)
        scores[name] = boot_stats.combine(raw, [VIDEO])
        print("METEOR", name, scores[name], flush=True)
    for name in ["pf", "rbpf"]:
        scores[name + "_mean"] = {
            key: float(np.mean([scores[f"{name}-s{seed}"][key] for seed in range(3)]))
            for key in scores[f"{name}-s0"]
        }
    write(DEST / "verified_summary.json", {"scope": manifest, "quality": scores})


if __name__ == "__main__":
    run()
