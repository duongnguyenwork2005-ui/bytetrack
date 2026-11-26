"""Select a detection-cache cutoff on dev, then score fixed settings on new clips."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

LOGS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOGS / "review100_nano_pf_dataset"))
import pilot_ridesafe as pilot  # noqa: E402

BASE = .25
OPTIONS = (BASE, .15, .10)
MODELS = {
    "coco": pilot.CHECKPOINTS["yolov8n_coco"],
    "interp50": "models/r101_ridesafe_interp50/yolo11n_mixed5_interp50.pt",
}


class CachedMotorcycles(pilot.CachedMotorcycles):
    def __init__(self, detections, shape, fps, cutoff):
        super().__init__(detections, shape, fps)
        self.cutoff = cutoff

    def iter_frames(self):
        for step, boxes in enumerate(self.detections.values()):
            yield step + 1, step / self.fps, False, boxes[boxes[:, 4] > self.cutoff]


def track(gt, arrays, fps, shape, clip, cutoff, method, seed):
    cache = CachedMotorcycles(arrays, shape, fps, cutoff)
    cfg = pilot.load_tracker_cfg(pilot.ROOT / pilot.TRACKERS[method], {"track_buffer": round(1.2 * fps)})
    assert (cfg.track_low_thresh, cfg.track_high_thresh, cfg.new_track_thresh) == (.10, .25, .25)
    mot, classes, timing = pilot.track_video(cache, cfg, dt_nominal=1 / fps, elapsed_mode="steps",
                                            seed=seed, video=clip)
    assert len(timing) == len(gt) and set(classes["cls"]) <= {3}
    return {"cutoff": cutoff, "method": method, "seed": seed, **pilot.mot_scores(gt, mot)}


def evaluate_clip(root, record, settings, methods):
    clip = record["clip"]
    root = root / clip
    gt, stride, shape, occluded = pilot.read_annotations(root / record["labels"]["file"])
    predictions, fps, frames, drift, checkpoints = pilot.predict_clip(
        root / record["video"]["file"], gt, shape, root)
    cadence = fps / stride
    assert stride == 5 and len(gt) > 0 and all(m["motorcycle_class"] == 3 for m in checkpoints.values())
    row = {"clip": clip, "frames": frames, "labeled_frames": len(gt), "source_fps": fps, "stride": stride,
           "effective_fps": cadence, "timestamp_max_abs_error_ms": drift, "checkpoints": checkpoints,
           "gt_boxes": sum(map(len, gt.values())), "gt_ids": len({x for a in gt.values() for x in a}),
           "occluded_boxes": occluded, "models": {}}
    for model, detections in predictions.items():
        output = {"recall": {str(cut): pilot.recall(gt, detections, cut) for cut in (.05, *OPTIONS)},
                  "tracking": []}
        for cut in settings[model]:
            for method, seeds in methods.items():
                for seed in seeds:
                    output["tracking"].append(track(gt, detections, cadence, shape, clip, cut, method, seed))
        row["models"][model] = output
    return row


def choose(rows):
    locks = {}
    for model in MODELS:
        summaries = {}
        for cut in OPTIONS:
            cells = [item for row in rows for item in row["models"][model]["tracking"] if item["cutoff"] == cut]
            assert len(cells) == len(rows)
            summaries[cut] = {"hota_mean": mean(item["hota"] for item in cells),
                              "idsw": sum(item["idsw"] for item in cells), "fn": sum(item["fn"] for item in cells)}
        base = summaries[BASE]
        accepted = [cut for cut in OPTIONS[1:] if summaries[cut]["idsw"] < base["idsw"]
                    and summaries[cut]["hota_mean"] >= base["hota_mean"] - .005
                    and summaries[cut]["fn"] <= base["fn"]]
        locked = min(accepted, key=lambda cut: (summaries[cut]["idsw"], -summaries[cut]["hota_mean"], -cut),
                     default=BASE)
        locks[model] = {"cutoff": locked, "dev": summaries}
    return locks


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    samples = json.loads((args.sources / "sources.json").read_text(encoding="utf-8"))
    assert len(samples["dev"]) == len(samples["test"]) == 2
    assert {r["clip"][:8] for r in samples["dev"]} == {"20221010"}
    assert {r["clip"][:8] for r in samples["test"]} == {"20221017"}
    pilot.CHECKPOINTS = MODELS
    dev = []
    for record in samples["dev"]:
        dev.append(evaluate_clip(args.sources / "dev", record, dict.fromkeys(MODELS, OPTIONS), {"kf": (None,)}))
        write(args.out / "ridesafe_dev.json", dev)
        print("dev", record["clip"], flush=True)
    locks = choose(dev)
    write(args.out / "ridesafe_cutoff_lock.json", locks)
    print(json.dumps(locks), flush=True)
    settings = {model: tuple(dict.fromkeys((BASE, lock["cutoff"]))) for model, lock in locks.items()}
    test = []
    for record in samples["test"]:
        test.append(evaluate_clip(args.sources / "test", record, settings,
                                  {"kf": (None,), "pf_motion2": (0, 1, 2)}))
        write(args.out / "ridesafe_test.json", test)
        print("test", record["clip"], flush=True)


if __name__ == "__main__":
    main()
