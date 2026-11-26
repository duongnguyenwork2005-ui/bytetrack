"""Evaluate fixed mixed-class METEOR clips with one detector and three filters."""
from collections import Counter, defaultdict
import configparser
import hashlib
import json
from pathlib import Path
import shutil

from meteor_audit import HERE, CLASSES, labels_for_camera, np, write
import meteor_pilot as pilot
from audit_turns import trace
from tracker_ctrv import UKFByteTracker

OUT = HERE / "meteor_scores"
COCO_IDS = {"car": 2, "motorcycle": 3}
RT = pilot.rt
ROOT = pilot.ROOT
METHODS = {"kf": "bytetrack.yaml", "ekf": ROOT / "configs/bytetrack_ekf_ctrv.yaml",
           "ukf": ROOT / "configs/bytetrack_ukf_ctrv.yaml"}


class ClassCache(pilot.Cache):
    def __init__(self, arrays, fps, category):
        super().__init__(arrays, fps)
        self.category = category

    def iter_frames(self):
        for i, arr in enumerate(self.arrays):
            keep = (arr[:, 4] > .25) & (arr[:, 5] == COCO_IDS[self.category])
            yield i + 1, i / self.fps, False, arr[keep][:300]


def run():
    selected = json.loads((HERE / "meteor_selection.json").read_text(encoding="utf-8"))
    downloads = json.loads((HERE / "meteor_downloads.json").read_text(encoding="utf-8"))
    assert set(downloads) == set(selected["selected"])
    OUT.mkdir(exist_ok=True)
    registry = RT.tracker_registry()
    registry["bytetrack_ukf_ctrv"] = UKFByteTracker
    metrics = [pilot.trackeval.metrics.HOTA(), pilot.trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               pilot.trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    totals = defaultdict(dict)
    rows, gates = [], []
    for item in selected["sample"]:
        name = item["name"]
        if name not in downloads:
            continue
        source = downloads[name]
        with Path(source["video"]).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == source["sha256"]
        video = pilot.cv2.VideoCapture(source["video"])
        frames = int(video.get(pilot.cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(pilot.cv2.CAP_PROP_FPS)
        shape = (int(video.get(pilot.cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(pilot.cv2.CAP_PROP_FRAME_WIDTH)))
        video.release()
        assert (frames, fps, shape) == (1800, 30.0, (1080, 1920)), (name, frames, fps, shape)
        pilot.DEST = OUT / name
        pilot.DEST.mkdir(exist_ok=True)
        arrays = pilot.detection({"frames": frames, "source": source})
        for category in CLASSES:
            target = item["classes"][category]
            gt_path = Path(target["gt"])
            assert hashlib.sha256(gt_path.read_bytes()).hexdigest() == target["gt_sha256"]
            gt = np.loadtxt(gt_path, delimiter=",", ndmin=2)
            split = f"METEOR-{category}"
            base = OUT / "gt" / split / name
            (base / "gt").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(gt_path, base / "gt/gt.txt")
            seq = configparser.ConfigParser()
            seq["Sequence"] = {"name": name, "imDir": "img1", "frameRate": "30", "seqLength": str(frames),
                               "imWidth": "1920", "imHeight": "1080", "imExt": ".jpg"}
            with (base / "seqinfo.ini").open("w", encoding="utf-8") as stream:
                seq.write(stream)
            ids, present, levels, _ = labels_for_camera(gt_path, name, frames, 30)
            occ = np.zeros_like(present)
            occ[gt[:, 0].astype(int) - 1, np.searchsorted(ids, gt[:, 1])] = gt[:, 8] == 0
            recent = occ.copy()
            for lag in range(1, 31):
                recent[lag:] |= occ[:-lag]
            turn = levels["turn25"] >= 0
            masks = {"all": present, "turn25": turn, "occluded": occ, "occluded_recent": present & recent,
                     "turn25_occluded": turn & occ, "straight8": levels["straight8"] >= 0}
            detections = ClassCache(arrays, fps, category)
            available = np.zeros_like(present)
            for frame, _, _, det in detections.iter_frames():
                target_gt = gt[gt[:, 0] == frame]
                boxes = det[:, :4].copy()
                if len(boxes) and len(target_gt):
                    boxes[:, 2:4] -= boxes[:, :2]
                    similarity = pilot._BaseDataset._calculate_box_ious(target_gt[:, 2:6], boxes, box_format="xywh")
                    available[frame-1, np.searchsorted(ids, target_gt[:, 1])] = similarity.max(axis=1) >= .5
            gates.append({"clip": name, "class": category,
                          "coverage": {group: [int((available & mask).sum()), int(mask.sum())] for group, mask in masks.items()}})
            for method, yaml in METHODS.items():
                cfg = RT.load_tracker_cfg(yaml, {"track_buffer": 36})
                mot, _, timing = RT.track_video(detections, cfg, 1/fps, elapsed_mode="steps", video=name, registry=registry)
                assert len(timing) == frames and not mot.duplicated(["frame", "id"]).any() and np.isfinite(mot.to_numpy(float)).all()
                destination = OUT / "tracking" / split / method / "data"
                destination.mkdir(parents=True, exist_ok=True)
                mot.to_csv(destination / f"{name}.txt", index=False, header=False)
                _, dc, _ = pilot.rte.build_configs(split, method, None)
                dc.update(PRINT_CONFIG=False, GT_FOLDER=str(OUT / "gt"), TRACKERS_FOLDER=str(OUT / "tracking"), SEQ_INFO={name: frames})
                dataset = pilot.trackeval.datasets.MotChallenge2DBox(dc)
                raw = dataset.get_raw_seq_data(method, name)
                data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
                assert all(np.array_equal(a, ids[b]) for a, b in zip(raw["gt_ids"], data["gt_ids"]))
                scores = {m.get_name(): m.eval_sequence(data) for m in metrics}
                matched, switches = trace(data, present.shape)
                misses = present & ~matched
                assert switches.sum() == scores["CLEAR"]["IDSW"] and misses.sum() == scores["CLEAR"]["CLR_FN"]
                totals[(category, method)][name] = scores
                for group, mask in masks.items():
                    rows.append({"clip": name, "class": category, "method": method, "group": group,
                                 "gt": int(mask.sum()), "IDSW": int((mask & switches).sum()), "FN": int((mask & misses).sum())})
                print(name, category, method, "IDSW", scores["CLEAR"]["IDSW"], flush=True)
    aggregate = defaultdict(Counter)
    for row in rows:
        aggregate[(row["class"], row["method"], row["group"])].update({k: row[k] for k in ["gt", "IDSW", "FN"]})
    combined = {}
    for (category, method), raw in totals.items():
        score = {m.get_name(): m.combine_sequences({video: result[m.get_name()] for video, result in raw.items()}) for m in metrics}
        combined[f"{category}/{method}"] = {"HOTA": float(np.mean(score["HOTA"]["HOTA"])),
             "IDF1": float(score["Identity"]["IDF1"]), "IDSW": int(score["CLEAR"]["IDSW"]),
             "FN": int(score["CLEAR"]["CLR_FN"]), "FP": int(score["CLEAR"]["CLR_FP"])}
    result = {"clips": selected["selected"], "combined": combined, "by_clip": rows, "coverage": gates,
              "groups": {"/".join(key): dict(value) for key, value in aggregate.items()}}
    write(OUT / "scores.json", result)
    print(json.dumps(combined), flush=True)


if __name__ == "__main__":
    run()
