"""Replay one detection cache across the existing ByteTrack motion filters."""

from collections import Counter
from zipfile import ZipFile
import json

from detect import DATA, FASTTRACK, HERE, KITTI, KITTI_SEQUENCES, ROOT, fasttrack_info, sha256

import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

os.environ["YOLO_CONFIG_DIR"] = str(DATA / "ultralytics_settings")
sys.path[:0] = [str(ROOT / "src"), str(HERE.parent / "review99_roundabout_turns")]
import run_tracking as rt
import trackeval
from audit_turns import labels_for_camera, runs


WORK = DATA / "evaluation"
TYPES = {"kf": "bytetrack.yaml", "ekf": ROOT / "configs/bytetrack_ekf_ctrv.yaml",
         "ukf": ROOT / "configs/bytetrack_ukf_ctrv.yaml",
         "pf": ROOT / "configs/phase2/trackers/r72/motion200.yaml"}
METHODS = [(m, None) for m in ("kf", "ekf", "ukf")] + [("pf", s) for s in range(3)]
METRICS = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
           trackeval.metrics.Identity({"PRINT_CONFIG": False})]


class Cache:
    def __init__(self, path, fps):
        with np.load(path) as npz:
            self.orig_shape = tuple(npz["shape"])
            self.arrays = [npz[str(i)] for i in range(len(npz.files) - 1)]
        self.fps = fps

    def iter_frames(self):
        for i, array in enumerate(self.arrays):
            selected = array[(array[:, 4] > .25) & (array[:, 5] == 2)]
            yield i + 1, i / self.fps, False, selected[:300]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.tolist()),
                    encoding="utf-8")


def write_gt(name, frames, fasttrack_metadata=fasttrack_info):
    is_kitti = name.startswith("kitti-")
    sequence = name.removeprefix("kitti-")
    if is_kitti:
        source = KITTI / "data_tracking_label_2/training/label_02" / f"{sequence}.txt"
        text = source.read_text(encoding="ascii")
        target = WORK / "kitti-gt" / "label_02" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="ascii")
        assert max(int(line.split()[0]) for line in text.splitlines()) + 1 == frames
        rows = np.asarray([[int(c[0]) + 1, int(c[1]), float(c[6]), float(c[7]),
                            float(c[8]) - float(c[6]), float(c[9]) - float(c[7]), 1, 1, int(c[4])]
                           for line in text.splitlines() if (c := line.split())[2] == "Car"
                           and int(c[4]) <= 2 and int(float(c[3])) <= 0])
        (WORK / "kitti-gt" / "evaluate_tracking.seqmap.training").write_text(
            f"{sequence} empty 0 {frames}\n", encoding="ascii")
    else:
        with ZipFile(DATA / f"{name}.zip") as archive:
            files, _, _ = fasttrack_metadata(archive, name)
            assert len(files) == frames
            raw = np.loadtxt(archive.open(f"{name}/gt/gt.txt"), delimiter=",")
        assert np.isfinite(raw).all() and np.all(raw[:, 8] == 1)
        rows = raw[(raw[:, 6] == 1) & (raw[:, 7] == 6)].copy()
        rows[:, 7] = 1
    assert np.all((rows[:, 0] >= 1) & (rows[:, 0] <= frames))
    assert len(rows) == len(np.unique(rows[:, :2], axis=0))
    assert (rows[:, 4:6] > 0).all()
    target = WORK / "gt" / name / "gt" / "gt.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(target, rows, delimiter=",", fmt="%.10g")
    return target, rows


def run_tracker(cache, name, method, seed):
    cfg = rt.load_tracker_cfg(TYPES[method], {"track_buffer": cache.fps})
    expected = {"track_high_thresh": .25, "track_low_thresh": .1, "new_track_thresh": .25,
                "match_thresh": .8, "fuse_score": True}
    assert all(getattr(cfg, field) == value for field, value in expected.items()), rt.cfg_to_dict(cfg)
    frames = len(cache.arrays)
    mot, _, timing = rt.track_video(cache, cfg, 1 / cache.fps, elapsed_mode="steps",
                                   seed=seed, video=name)
    assert len(timing) == frames and timing["n_steps"].eq(1).all()
    assert not mot.duplicated(["frame", "id"]).any() and np.isfinite(mot.to_numpy(float)).all()
    tag = method if seed is None else f"{method}-s{seed}"
    destination = WORK / "trackers" / tag / "data"
    destination.mkdir(parents=True, exist_ok=True)
    if name.startswith("kitti-"):
        sequence = name.removeprefix("kitti-")
        path = destination / f"{sequence}.txt"
        with path.open("w", encoding="ascii") as stream:
            for frame, track, x, y, w, h, confidence in mot.itertuples(index=False, name=None):
                stream.write(f"{frame-1} {track} Car -1 -1 -10 {x:.10f} {y:.10f} "
                             f"{x+w:.10f} {y+h:.10f} -1 -1 -1 -1000 -1000 -1000 -10 {confidence:.10f}\n")
        dataset = trackeval.datasets.Kitti2DBox(
            {"PRINT_CONFIG": False, "TRACKERS_TO_EVAL": [tag], "CLASSES_TO_EVAL": ["car"],
             "GT_FOLDER": str(WORK / "kitti-gt"), "TRACKERS_FOLDER": str(WORK / "trackers")})
        cls = "car"
    else:
        sequence = name
        path = destination / f"{name}.txt"
        mot.to_csv(path, index=False, header=False, float_format="%.10f")
        dataset = trackeval.datasets.MotChallenge2DBox(
            {"PRINT_CONFIG": False, "TRACKERS_TO_EVAL": [tag], "BENCHMARK": "FastTrack",
             "SPLIT_TO_EVAL": "pilot", "SKIP_SPLIT_FOL": True, "DO_PREPROC": False,
             "GT_FOLDER": str(WORK / "gt"), "TRACKERS_FOLDER": str(WORK / "trackers"),
             "SEQ_INFO": {name: frames}})
        cls = "pedestrian"  # adapter encodes annotated cars as the single MOTChallenge class.
    assert path.is_file()
    raw = dataset.get_raw_seq_data(tag, sequence)
    data = dataset.get_preprocessed_seq_data(raw, cls)
    return tag, data, raw, sha256(path)


def clear_assignments(data, shape):
    """Record TrackEval CLEAR identities at IoU 0.5, including after GT gaps."""
    previous = np.full(shape[1], -1, dtype=int)
    last = previous.copy()
    assigned = np.full(shape, -1, dtype=int)
    switched = np.zeros(shape, dtype=bool)
    eps = np.finfo(float).eps
    for t, (targets, tracks, overlap) in enumerate(zip(data["gt_ids"], data["tracker_ids"],
                                                      data["similarity_scores"])):
        if not len(targets) or not len(tracks):
            continue
        affinity = overlap + 1000 * (tracks == previous[targets, None])
        affinity[overlap < .5 - eps] = 0
        a, b = linear_sum_assignment(-affinity)
        keep = affinity[a, b] > eps
        gt, pred = targets[a[keep]], tracks[b[keep]]
        assigned[t, gt] = pred
        switched[t, gt] = (last[gt] >= 0) & (last[gt] != pred)
        last[gt] = pred
        previous.fill(-1)
        previous[gt] = pred
    return assigned, switched


def concise(raw):
    return {"HOTA": float(np.mean(raw["HOTA"]["HOTA"])), "IDF1": float(raw["Identity"]["IDF1"]),
            "IDSW": int(raw["CLEAR"]["IDSW"]), "FN": int(raw["CLEAR"]["CLR_FN"]),
            "FP": int(raw["CLEAR"]["CLR_FP"])}


def score_sequence(name, out, fps=None, fasttrack_metadata=fasttrack_info):
    is_kitti = name.startswith("kitti-")
    cache_path = DATA / f"{name}_yolov8n.npz"
    cache = Cache(cache_path, fps if fps is not None else (10 if is_kitti else 30))
    frames = len(cache.arrays)
    gt_path, gt = write_gt(name, frames, fasttrack_metadata)
    ids, present, levels, turns = labels_for_camera(gt_path, name, frames, cache.fps)
    masks = {"all": present, "turn25": levels["turn25"] >= 0}
    events = []
    if is_kitti:
        severe = np.zeros_like(present)
        severe[gt[:, 0].astype(int) - 1, np.searchsorted(ids, gt[:, 1])] = gt[:, 8] == 2
        masks["occlusion2"] = severe
        masks["turn25_occlusion2"] = severe & masks["turn25"]
        for index, identity in enumerate(ids):
            for start, stop in runs(severe[:, index]):
                if 0 < start and stop < frames and stop - start >= 3:
                    if present[start - 1, index] and present[stop, index]:
                        events.append((int(identity), index, int(start - 1), int(stop)))
    record = {"frames": frames, "fps": cache.fps, "gt_boxes": len(gt), "gt_ids": len(ids),
              "candidate_turns": sum(t["level"] == "turn25" for t in turns),
              "cache_sha256": sha256(cache_path), "gt_sha256": sha256(gt_path), "variants": {}}
    for method, seed in METHODS:
        tag, data, raw, mot_sha = run_tracker(cache, name, method, seed)
        for t, (compact, source) in enumerate(zip(data["gt_ids"], raw["gt_ids"])):
            if is_kitti:
                extra = raw["gt_extras"][t]
                source = source[(extra["truncation"] <= 0) & (extra["occlusion"] <= 2)
                                & (raw["gt_classes"][t] == 1)]
            assert np.array_equal(ids[compact], source)
            assert set(compact) == set(np.flatnonzero(present[t]))
        scores = {m.get_name(): m.eval_sequence(data) for m in METRICS}
        assigned, switches = clear_assignments(data, present.shape)
        missed = present & (assigned == -1)
        assert switches.sum() == scores["CLEAR"]["IDSW"] and missed.sum() == scores["CLEAR"]["CLR_FN"]
        groups = {key: {"boxes": int(mask.sum()), "IDSW": int((mask & switches).sum()),
                        "FN": int((mask & missed).sum())} for key, mask in masks.items()}
        recovery = Counter()
        for _, ind, before, after in events:
            x, y = assigned[before, ind], assigned[after, ind]
            if x < 0:
                recovery["before_missing"] += 1
            elif y < 0:
                recovery["after_missing"] += 1
            elif x == y:
                recovery["same_id"] += 1
            else:
                recovery["different_id"] += 1
        assert sum(recovery.values()) == len(events)
        record["variants"][tag] = {"metrics": concise(scores), "groups": groups, "raw": scores,
                                   "recovery": dict(recovery), "mot_sha256": mot_sha}
        write_json(out / f"{name}.json", record)
        print(name, tag, concise(scores), groups, dict(recovery), flush=True)


def aggregate(out):
    combined = {}
    for dataset, names in [("FastTrack", FASTTRACK), ("KITTI", [f"kitti-{s}" for s in KITTI_SEQUENCES])]:
        records = {name: json.loads((out / f"{name}.json").read_text(encoding="utf-8")) for name in names}
        rows = {}
        for method, seed in METHODS:
            tag = method if seed is None else f"{method}-s{seed}"
            variants = {name: record["variants"][tag] for name, record in records.items()}
            scores = {}
            for metric in METRICS:
                key = metric.get_name()
                fields = {name: {k: np.asarray(v) if isinstance(v, list) else v
                                 for k, v in row["raw"][key].items()} for name, row in variants.items()}
                scores[key] = metric.combine_sequences(fields)
            sums = {}
            for row in variants.values():
                for group, counts in row["groups"].items():
                    sums.setdefault(group, Counter()).update(counts)
            recovery = sum((Counter(row["recovery"]) for row in variants.values()), Counter())
            metrics = concise(scores)
            assert metrics["IDSW"] == sums["all"]["IDSW"] and metrics["FN"] == sums["all"]["FN"]
            rows[tag] = dict(metrics=metrics, groups=sums, recovery=recovery)
        combined[dataset] = {"frames": sum(v["frames"] for v in records.values()),
                             "clips": list(records), "results": rows}
    write_json(out / "scores.json", combined)


if __name__ == "__main__":
    output = HERE / "scores"
    output.mkdir(exist_ok=True)
    if sys.argv[1:] == ["aggregate"]:
        aggregate(output)
    else:
        for clip in sys.argv[1:]:
            score_sequence(clip, output)
