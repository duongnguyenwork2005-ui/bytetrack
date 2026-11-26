"""Replay a fixed Jetson detection cache with four ByteTrack motion models."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean
import sys

os.environ["BYTETRACK_BACKEND"] = "np"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "src"), str(HERE.parent / "review102_ridesafe_pf_nano")]

import det_cache
import eval_nano as evaluation
import numpy as np
import run_tracking as tracking
import trackeval
import tracker_ctrv

CONFIGS = {"kf": "bytetrack.yaml", "ekf": ROOT / "configs/bytetrack_ekf_ctrv.yaml",
           "ukf": ROOT / "configs/bytetrack_ukf_ctrv.yaml", "pf": ROOT / "configs/phase2/trackers/r72/motion200.yaml"}
METHODS = (("kf", "kf", None), ("ekf", "ekf", None), ("ukf", "ukf", None),
           ("pf_s0", "pf", 0), ("pf_s1", "pf", 1), ("pf_s2", "pf", 2))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def array_from(path):
    if path.stat().st_size == 0:
        return np.empty((0, 7))
    return np.loadtxt(path, delimiter=",", ndmin=2)


def replay(task):
    capture, session, previous, output = task
    name, ds = capture["name"], capture["dataset"]
    split = evaluation.SPLITS[ds][0]
    path = session / "cache" / name / (name + ".npz")
    assert sha256(path.read_bytes()).hexdigest() == capture["cache_sha256"]
    baseline = previous / "overlap/runs/full/pipeline" / ds / name / "B"
    condition = read(baseline / "run_meta.json")["cau_hinh"]
    assert condition["conf_min"] == .25 and condition["max_det"] == 300
    class_map = {int(source): int(dest) for source, dest in condition["class_map"].items()}
    vd = det_cache.VideoDetections(path, condition["conf_min"], condition["classes"],
                                   condition["max_det"], class_map)
    fps = 5 if ds == "bdd" else 25
    assert np.array_equal(vd.frame_idx, np.arange(1, 151)) and not vd.unreadable.any()
    assert np.allclose(vd.time_s, np.arange(150) / fps, rtol=0, atol=1e-9)
    for frame in vd.iter_frames():
        assert np.isfinite(frame[3]).all() and len(frame[3]) <= 300
    registry = tracking.tracker_registry()
    registry.update(bytetrack_ukf_ctrv=tracker_ctrv.UKFByteTracker, bytetrack_pf_ctrv=tracker_ctrv.PFByteTracker)
    for method, config, seed in METHODS:
        cfg = tracking.load_tracker_cfg(CONFIGS[config], {"track_buffer": condition["track_buffer_frames"]})
        mot, cls, timing = tracking.track_video(vd, cfg, 1 / fps, condition["elapsed_mode"],
                                                seed=seed, video=name, registry=registry)
        assert len(timing) == 150 and not mot.duplicated(["frame", "id"]).any()
        assert np.isfinite(mot.to_numpy(dtype=float)).all()
        destination = output / "trackers" / split / method / "data" / (name + ".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        mot.to_csv(destination, header=False, index=False)
        if method == "kf":
            actual = mot.to_numpy(dtype=float)
            expected = array_from(baseline / "data" / (name + ".txt"))
            old_cls = np.loadtxt(baseline / "data_cls" / (name + ".csv"), delimiter=",", skiprows=1, ndmin=2)
            parity = actual.shape == expected.shape and np.allclose(actual, expected, rtol=0, atol=1e-3)
            parity = parity and np.array_equal(actual[:, [0, 1, 6]], expected[:, [0, 1, 6]])
            parity = parity and np.array_equal(cls.to_numpy(float).reshape(-1, 3), old_cls.reshape(-1, 3))
    return {"dataset": ds, "name": name, "kf_matches_r109": bool(parity)}


def score(output, captures):
    result = {}
    frameworks = (trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
                  trackeval.metrics.Identity({"PRINT_CONFIG": False}))
    for ds, (split, eval_name) in evaluation.SPLITS.items():
        clips = [c for c in captures if c["dataset"] == ds]
        names = [c["name"] for c in clips]
        for clip in clips:
            name = clip["name"]
            dest = output / "gt" / split / name / "gt/gt.txt"
            dest.parent.mkdir(parents=True)
            original = ROOT / "data/processed" / split / name / "gt/gt.txt"
            with dest.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows(evaluation.clipped_gt(original, clip["gt_first"], clip["gt_last"]))
        result[ds] = {}
        for method, _, _ in METHODS:
            _, cfg, _ = evaluation.rte.build_configs(split, method, None)
            cfg.update(GT_FOLDER=str(output / "gt"), TRACKERS_FOLDER=str(output / "trackers"),
                       SEQ_INFO=dict.fromkeys(names, 150))
            dataset = evaluation.rte.dataset_class(eval_name)(cfg)
            raw = {}
            for name in names:
                data = dataset.get_preprocessed_seq_data(dataset.get_raw_seq_data(method, name), "pedestrian")
                raw[name] = {m.get_name(): m.eval_sequence(data) for m in frameworks}
            combined = {m.get_name(): m.combine_sequences({n: u[m.get_name()] for n, u in raw.items()})
                        for m in frameworks}
            result[ds][method] = {"combined": evaluation.scalars(combined),
                                  "videos": {n: evaluation.scalars(u) for n, u in raw.items()}}
        pf = [result[ds][key]["combined"] for key in ("pf_s0", "pf_s1", "pf_s2")]
        result[ds]["pf_mean"] = {key: mean(row[key] for row in pf) for key in pf[0]}
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("previous", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clips", type=int, default=80)
    args = parser.parse_args()
    capture = read(args.session / "captures.json")[:args.clips]
    assert len(capture) == args.clips and {r["dataset"] for r in capture} == {"bdd", "ua"}
    assert len({(r["dataset"], r["name"]) for r in capture}) == len(capture)
    original = read(args.previous / "inputs.json")["videos"]
    expected = {(r["dataset"], r["name"]): (r["first"], r["last"]) for r in original}
    for row in capture:
        assert expected[row["dataset"], row["name"]] == (row["gt_first"], row["gt_last"])
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(row, args.session, args.previous, args.output) for row in capture]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        verified = []
        for entry in pool.map(replay, tasks):
            verified.append(entry)
            print(len(verified), entry, flush=True)
    quality = score(args.output, capture)
    result = {"clips": len(capture), "track_runs": len(capture) * len(METHODS), "capture": capture,
              "replay_checks": verified, "quality": quality}
    (args.output / "scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FILTER_REPLAY_COMPLETE", result["track_runs"], flush=True)


if __name__ == "__main__":
    main()
