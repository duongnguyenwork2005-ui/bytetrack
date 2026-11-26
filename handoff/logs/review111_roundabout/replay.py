"""Track full RoundaboutHD timelines and score the previously untested suffix."""
from argparse import ArgumentParser
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import os
from pathlib import Path
from statistics import mean
import sys
from time import sleep

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ.update(BYTETRACK_BACKEND="np", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                  YOLO_CONFIG_DIR=str(HERE / "ultralytics"))
sys.path[:0] = [str(ROOT / "src"), str(HERE.parent / "review99_roundabout_turns"),
                str(HERE.parent / "review102_ridesafe_pf_nano")]

from audit_turns import CAMERAS, labels_for_camera, trace
import det_cache
from eval_nano import scalars
import numpy as np
import psutil
import run_trackeval as rte
import run_tracking as rt
import trackeval
import tracker_ctrv

FIRST = 3001
FRAMES = 9000
FPS = 15
PREFIX = ROOT / "data/interim/det_cache/r64_roundabout_yolov8n_coco_c0.05"
LABELS = HERE.parent / "review108_followup/roundabout_gt"
FULL = "RoundaboutHD-r111full"
SUFFIX = "RoundaboutHD-r111suffix"
SETTINGS = {"kf": "bytetrack.yaml", "ekf": "configs/bytetrack_ekf_ctrv.yaml",
            "ukf": "configs/bytetrack_ukf_ctrv.yaml", "pf": "configs/phase2/trackers/r72/motion200.yaml"}
METHODS = (("kf", "kf", None), ("ekf", "ekf", None), ("ukf", "ukf", None),
           ("pf_s0", "pf", 0), ("pf_s1", "pf", 1), ("pf_s2", "pf", 2))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class Timeline:
    orig_shape = (2160, 3840)
    classes = np.array([1, 2, 3, 5, 7])

    def __init__(self, camera):
        self.camera = camera

    def iter_frames(self):
        end = 0
        paths = [PREFIX / (self.camera + ".npz")]
        paths.extend(HERE / "cache" / f"{self.camera}_{first:04d}.npz" for first in range(FIRST, FRAMES + 1, 600))
        for path in paths:
            with np.load(path) as chunk:
                timeline, data = chunk["frame_idx"], chunk["data"]
                offsets = np.r_[0, np.cumsum(chunk["n_per_frame"])]
                assert np.array_equal(timeline, np.arange(end + 1, end + len(timeline) + 1))
                assert offsets[-1] == len(data) and np.isfinite(data).all()
                if path.parent == PREFIX:
                    assert not (chunk["unreadable"].any() or chunk["truncated"].any())
                    assert tuple(chunk["orig_shape"]) == self.orig_shape
                for frame, lo, hi in zip(timeline, offsets, offsets[1:]):
                    boxes = det_cache.filter_dets(data[lo:hi], .25, self.classes, 300, None)
                    yield int(frame), (int(frame) - 1) / FPS, False, boxes
                end += len(timeline)
        assert end == FRAMES


def replay(task):
    camera, (name, method, seed), output = task
    cfg = rt.load_tracker_cfg(SETTINGS[method], {"track_buffer": 18})
    registry = rt.tracker_registry()
    registry.update(bytetrack_ukf_ctrv=tracker_ctrv.UKFByteTracker, bytetrack_pf_ctrv=tracker_ctrv.PFByteTracker)
    mot, _, latency = rt.track_video(Timeline(camera), cfg, 1 / FPS, "steps", seed, camera, registry)
    assert len(latency) == FRAMES and not mot.duplicated(["frame", "id"]).any()
    assert np.isfinite(mot.to_numpy(dtype=float)).all()
    remaining = mot[mot["frame"] >= FIRST].copy()
    remaining["frame"] -= FIRST - 1
    for split, rows in ((FULL, mot), (SUFFIX, remaining)):
        target = output / "trackers" / split / name / "data" / (camera + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(target, index=False, header=False, float_format="%.10f")
    entry = {"camera": camera, "method": name, "frames": len(latency), "mot_rows": len(mot)}
    if method != "pf":
        prior_name = "cv" if method == "kf" else method
        prefix = ROOT / "data/processed/trackers/RoundaboutHD-prefix3000" / f"r64-yolov8n_coco-c25-{prior_name}/data" / (camera + ".txt")
        old = np.loadtxt(prefix, delimiter=",", ndmin=2)
        new = mot[mot["frame"] < FIRST].to_numpy(dtype=float)
        entry["prefix_mot_parity"] = bool(old.shape == new.shape and np.array_equal(old[:, :2], new[:, :2])
                                           and np.allclose(old, new, rtol=0, atol=1e-3))
    return entry


def gt_and_groups(output):
    stored = read(LABELS / "coverage.json")
    groups = {}
    for camera in CAMERAS:
        raw = LABELS / (camera + ".txt")
        data = np.loadtxt(raw, delimiter=",", ndmin=2)
        assert np.isfinite(data).all() and data.shape[1] == 9 and len(data) == stored[camera]["gt_boxes"]
        kept = data[data[:, 0] >= FIRST].copy()
        kept[:, 0] -= FIRST - 1
        for split, rows in ((FULL, data), (SUFFIX, kept)):
            target = output / "gt" / split / camera / "gt/gt.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(target, rows, delimiter=",", fmt=["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%d", "%d", "%.3f"])
        ids, present, episodes, _ = labels_for_camera(raw, camera, frames=FRAMES, fps=FPS)
        mask = {name: array >= 0 for name, array in episodes.items()}
        for name, label in mask.items():
            assert int(label[FIRST - 1:].sum()) == stored[camera]["new_window_box_counts"][name]
        groups[camera] = (ids, present, mask)
    return groups


def score(output, groups):
    metrics = (trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False}))
    result = {FULL: {}, SUFFIX: {}}
    for split, count in ((FULL, FRAMES), (SUFFIX, FRAMES - FIRST + 1)):
        for method, _, _ in METHODS:
            _, cfg, _ = rte.build_configs(split, method, None)
            cfg.update(GT_FOLDER=str(output / "gt"), TRACKERS_FOLDER=str(output / "trackers"),
                       SEQ_INFO=dict.fromkeys(CAMERAS, count))
            dataset = rte.dataset_class("mot")(cfg)
            raw, scenario = {}, {}
            for camera in CAMERAS:
                original = dataset.get_raw_seq_data(method, camera)
                data = dataset.get_preprocessed_seq_data(original, "pedestrian")
                raw[camera] = {m.get_name(): m.eval_sequence(data) for m in metrics}
                if split == FULL:
                    ids, present, masks = groups[camera]
                    assert all(np.array_equal(actual, ids[compact]) for actual, compact in zip(original["gt_ids"], data["gt_ids"]))
                    matched, changed = trace(data, present.shape)
                    missing = present & ~matched
                    clear = raw[camera]["CLEAR"]
                    assert changed.sum() == clear["IDSW"] and missing.sum() == clear["CLR_FN"]
                    window = slice(FIRST - 1, None)
                    scenario[camera] = {group: {"gt": int(mask[window].sum()),
                        "idsw": int((changed[window] & mask[window]).sum()),
                        "fn": int((missing[window] & mask[window]).sum())} for group, mask in {"all_suffix": present, **masks}.items()}
            combined = {m.get_name(): m.combine_sequences({key: value[m.get_name()] for key, value in raw.items()})
                        for m in metrics}
            entry = {"combined": scalars(combined), "per_camera": {key: scalars(value) for key, value in raw.items()}}
            if scenario:
                entry["suffix_scenarios_per_camera"] = scenario
                entry["suffix_scenarios"] = {group: {key: sum(v[group][key] for v in scenario.values())
                    for key in ("gt", "idsw", "fn")} for group in scenario[CAMERAS[0]]}
            result[split][method] = entry
            print("SCORED", split, method, flush=True)
    for group in result.values():
        seeds = [group[f"pf_s{i}"] for i in range(3)]
        group["pf_mean"] = {"combined": {k: mean(s["combined"][k] for s in seeds) for k in seeds[0]["combined"]}}
        if "suffix_scenarios" in seeds[0]:
            group["pf_mean"]["suffix_scenarios"] = {label: {k: mean(s["suffix_scenarios"][label][k] for s in seeds)
                for k in obj} for label, obj in seeds[0]["suffix_scenarios"].items()}
    return result


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--capture-pid", type=int, help="Replay each camera as its cache becomes available")
    args = cli.parse_args()
    output = HERE / "evaluation"
    output.mkdir(exist_ok=False)
    groups = gt_and_groups(output)
    entries = []
    remaining = set(CAMERAS)
    jobs = set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while remaining or jobs:
            ready = [cam for cam in sorted(remaining) if all((HERE / "cache" / f"{cam}_{i:04d}.npz").is_file()
                                                              for i in range(FIRST, FRAMES + 1, 600))]
            for cam in ready:
                jobs.update(pool.submit(replay, (cam, method, output)) for method in METHODS)
                remaining.remove(cam)
            if remaining and (not args.capture_pid or not psutil.pid_exists(args.capture_pid)):
                raise RuntimeError(f"Capture exited before completing {sorted(remaining)}")
            if not jobs:
                sleep(5)
                continue
            done, jobs = wait(jobs, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                entry = future.result()
                entries.append(entry)
                print("REPLAY", len(entries), entry, flush=True)
    assert read(HERE / "capture.json")["frames"] == 24000 and len(entries) == 24
    scores = score(output, groups)
    (output / "scores.json").write_text(json.dumps({"runs": entries, "quality": scores}, indent=2), encoding="utf-8")
    print("ROUNDABOUT_COMPLETE", len(entries), flush=True)


if __name__ == "__main__":
    main()
