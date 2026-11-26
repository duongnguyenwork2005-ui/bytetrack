"""Evaluate all official HELMET test clips from the verified release archives."""
from collections import defaultdict
from contextlib import ExitStack
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = Path("F:/LABOTARY_data/goc_tai_ve/HELMET")
OUT = HERE / "helmet182"
OUT.mkdir(exist_ok=True)
SETTINGS = OUT / "ultralytics"
SETTINGS.mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(SETTINGS)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/r64_ngoai"))

import cv2
import numpy as np
import trackeval
from ultralytics import YOLO
import boot_stats
import danh_gia_ngoai as r64
from r66_ngoai.helmet_manifest import parse_boxes
import run_tracking as rt
import run_trackeval as rte


class Detections:
    orig_shape = (1080, 1920)

    def __init__(self, boxes):
        self.boxes = boxes

    def iter_frames(self):
        for i, box in enumerate(self.boxes):
            keep = (box[:, 5] == 3) & (box[:, 4] > .10)
            yield i + 1, i / 10, False, box[keep][:300]


def source_catalog(stack):
    verified = json.loads((ROOT / "handoff/logs/external_eval/r66/helmet/nguon_kiem_sha_crc.json").read_text())
    for name, entry in verified.items():
        stat = (SOURCE / name).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (entry["size"], entry["mtime_ns"]), name
    with (SOURCE / "data_split.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({r["video_id"] for r in rows}) == 910
    videos = [r["video_id"] for r in rows if r["Set"] == "test"]
    assert len(videos) == 182 and all(Path(v).name == v for v in videos)
    members = defaultdict(dict)
    for i in range(1, 8):
        archive = stack.enter_context(ZipFile(SOURCE / f"part_{i}.zip"))
        for entry in archive.infolist():
            parts = entry.filename.split("/")
            if len(parts) == 3 and parts[1] in videos and parts[2].endswith(".jpg"):
                frame = int(Path(parts[2]).stem)
                assert frame not in members[parts[1]]
                members[parts[1]][frame] = archive, entry
    assert all(set(members[v]) == set(range(1, 101)) for v in videos)
    return videos, members


def create_gt(video, boxes):
    ids = {key: i for i, key in enumerate(sorted({k[0] for k in boxes}), 1)}
    folder = OUT / "ground_truth/HELMET-test182" / video / "gt"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [[f, ids[k], *box, 1, 1, 1] for (k, f), box in boxes.items()]
    rows.sort(key=lambda row: (row[0], row[1]))
    np.savetxt(folder / "gt.txt", np.asarray(rows, dtype=float).reshape(-1, 9), delimiter=",",
               fmt=["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%d", "%d", "%.3f"])


def infer(model, mapping, members):
    detected = []
    for first in range(1, 101, 4):
        images = []
        for frame in range(first, first + 4):
            archive, entry = members[frame]
            image = cv2.imdecode(np.frombuffer(archive.read(entry), dtype=np.uint8), cv2.IMREAD_COLOR)
            assert image is not None and image.shape[:2] == Detections.orig_shape
            images.append(image)
        results = model.predict(images, imgsz=640, conf=.05, iou=.7, max_det=3000,
                                device=0, half=False, rect=True, verbose=False)
        assert len(results) == 4
        for result in results:
            boxes = result.boxes.data.cpu().numpy().astype(np.float32)
            assert len(boxes) < 3000 and np.isfinite(boxes).all()
            detected.append(r64.doi_id_sang_coco(boxes, mapping))
    return detected


def score(videos, names, groups=None):
    ec, dc, mc = rte.build_configs("HELMET-test182", names[0], None)
    ec.update(PRINT_RESULTS=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
              LOG_ON_ERROR=str(OUT / "error.log"))
    dc.update(GT_FOLDER=str(OUT / "ground_truth"), TRACKERS_FOLDER=str(OUT / "trackers"),
              TRACKERS_TO_EVAL=names, SEQ_INFO=dict.fromkeys(videos, 100))
    metric = [trackeval.metrics.HOTA(mc), trackeval.metrics.CLEAR(mc), trackeval.metrics.Identity(mc)]
    if groups is None:
        groups = (("official_test182", videos), ("expansion146", [v for v in videos if v not in OLD]),
                  ("reference36", [v for v in videos if v in OLD]))
    for label, group in groups:
        dc["SEQ_INFO"] = dict.fromkeys(group, 100)
        dataset = trackeval.datasets.MotChallenge2DBox(dc)
        result, messages = trackeval.Evaluator(ec).evaluate([dataset], metric)
        assert all(m == "Success" for m in next(iter(messages.values())).values())
        for name in names:
            rte.flatten_results(result, name).to_csv(OUT / f"{label}_{name}.csv", index=False)
            boot_stats.save_raw_fields(result, name, OUT / f"{label}_{name}_raw.json")


OLD = {r["video_id"] for r in json.loads((ROOT / "handoff/logs/external_eval/r66/helmet/manifest_test36.json").read_text())["videos"]}


def main():
    with ExitStack() as stack:
        videos, catalog = source_catalog(stack)
        labels = stack.enter_context(ZipFile(SOURCE / "annotation.zip"))
        config = r64.detector_r54("yolo11n_mixed5")
        checkpoint = ROOT / config["checkpoint"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == config["sha256"]
        model = YOLO(str(checkpoint))
        mapping = r64.bang_anh_xa_coco(model.names, config["loai_checkpoint"])
        modes = {kind if seed is None else f"{kind}-s{seed}": (rt.load_tracker_cfg(path, {"track_buffer": 12}), seed)
                 for kind, (path, seeds) in r64.BO_LOC.items() for seed in (seeds or [None])}
        assert len(modes) == 6 and OLD <= set(videos)
        catalog_summary = {}
        for video in videos:
            raw = labels.read(f"annotation/{video}.csv")
            boxes, audit = parse_boxes(raw, video)
            create_gt(video, boxes)
            catalog_summary[video] = audit
        (OUT / "manifest.json").write_text(json.dumps({"detector": config, "fps": 10, "cutoff": .10,
            "batch": 4, "previous_test36": sorted(OLD), "clips": catalog_summary}, indent=2), encoding="utf-8")
        for index, video in enumerate(videos, 1):
            cache = OUT / "cache" / f"{video}.npz"
            cache.parent.mkdir(exist_ok=True)
            if cache.exists():
                with np.load(cache) as data:
                    boxes = [data[str(i)] for i in range(100)]
            else:
                boxes = infer(model, mapping, catalog[video])
                pending = cache.with_suffix(".tmp.npz")
                np.savez_compressed(pending, **{str(i): box for i, box in enumerate(boxes)})
                pending.replace(cache)
            for name, (cfg, seed) in modes.items():
                dest = OUT / "trackers/HELMET-test182" / name / "data" / f"{video}.txt"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    continue
                mot, classes, timing = rt.track_video(Detections(boxes), cfg, dt_nominal=.1,
                                                      elapsed_mode="steps", seed=seed, video=video)
                assert len(timing) == 100 and set(classes["cls"]) <= {3}
                assert not mot.duplicated(["frame", "id"]).any()
                pending = dest.with_suffix(".tmp")
                mot.to_csv(pending, index=False, header=False, float_format="%.10f")
                pending.replace(dest)
            print(index, len(videos), video, flush=True)
        score(videos, list(modes))
        print("HELMET182_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
