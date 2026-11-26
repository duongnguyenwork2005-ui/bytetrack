"""Verify hardware timing and score the corresponding Nano MOT output."""
from argparse import ArgumentParser
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import harmonic_mean, mean, median
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "review105_nano_fps"))
from eval_benchmark import quality, verify_timing, evaluation, SPLITS, frame_checks, paired_speed


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def throughput_summary(items, modes):
    summary = {}
    for dataset in ("bdd", "ua", "all"):
        selected = [row for row in items if dataset == "all" or row["dataset"] == dataset]
        if not selected:
            continue
        group = {}
        for mode in modes:
            runs = [row for row in selected if row["mode"] == mode]
            fps = [row["fps_steady"] for row in runs]
            group[mode] = {"clips": len(runs), "mean_fps": mean(fps), "median_fps": median(fps),
                           "aggregate_fps": harmonic_mean(fps), "min_fps": min(fps), "max_fps": max(fps),
                           "clips_at_15_fps": sum(value >= 15 for value in fps),
                           "median_clip_p90_latency_ms": median(row["p90_latency_ms"] for row in runs)}
        summary[dataset] = group
    return summary


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    args = parser.parse_args()
    logs = args.logs
    inputs = read_json(logs / "inputs.json")
    after = read_json(logs / "after.json")
    assert inputs["boot"] == after["boot"] and after["errors"] == "0"
    videos = inputs["videos"]
    items = read_json(logs / "pairs.json")
    keys = {(row["dataset"], row["name"], row["mode"]): row for row in items}
    modes = inputs.get("modes", ["v21combo", "full", "early"])
    assert len(keys) == len(items) == len(videos) * len(modes)
    with TemporaryDirectory(prefix="nano_score_", dir=logs) as tmp:
        stage = Path(tmp)
        for row in videos:
            ds, name = row["dataset"], row["name"]
            split = SPLITS[ds][0]
            gt = stage / "gt" / split / name / "gt/gt.txt"
            gt.parent.mkdir(parents=True)
            source = evaluation.ROOT / "data/processed" / split / name / "gt/gt.txt"
            with gt.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows(evaluation.clipped_gt(source, row["first"], row["last"]))
            image_manifest = (logs / f"{ds}_{name}_images.json").read_bytes()
            assert len(json.loads(image_manifest)) == 150
            for mode in modes:
                item = keys[ds, name, mode]
                assert sha256(image_manifest).hexdigest() == item["source_sha256"]
                base = mode if mode == "v21combo" else "full"
                version = "A" if mode == "v21combo" else "B"
                folder = logs / mode / "runs" / base / "pipeline" / ds / name / version
                meta = read_json(folder / "run_meta.json")
                assert meta["hello_con"]["engine_sha256"] == inputs["engine_sha256"][version]
                assert meta["hello_con"]["letterbox_hw"] == [384, 640]
                assert not meta["hien_thi"]["bat"]
                assert meta["cau_hinh"]["conf_min"] == .25
                assert meta["cau_hinh"]["det_conf"] == (.25 if mode in ("early", "matrix", "overlap") else .05)
                with (folder / "pipeline" / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
                    frames = list(csv.DictReader(stream))
                assert not frame_checks(frames, base, ds)
                measure = verify_timing(frames, meta["thoi_gian"]["tom_tat_sau_warmup"])
                assert abs(measure["fps_recalculated"] - item["fps_steady"]) < 1e-9
                assert abs(measure["p90_recalculated_ms"] - item["p90_latency_ms"]) < 1e-9
                mot = (folder / "data" / f"{name}.txt").read_bytes()
                label = (folder / "data_cls" / f"{name}.csv").read_bytes()
                assert sha256(mot).hexdigest() == item["mot_sha256"]
                assert sha256(label).hexdigest() == item["cls_sha256"]
                tracker = {modes[0]: "A", modes[1]: "B"}.get(mode, mode)
                destination = stage / "trackers" / split / tracker / "data" / f"{name}.txt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(mot)
        metrics = quality(stage, videos)
        summary = {dataset: {modes[0]: group["A"], modes[1]: group["B"]} for dataset, group in metrics.items()}
        previous_mode = modes[1]
        for mode in modes[2:]:
            # Reuse the same evaluator with the next candidate in slot B.
            for dataset in summary:
                trackers = stage / "trackers" / SPLITS[dataset][0]
                (trackers / "B").rename(trackers / previous_mode)
                (trackers / mode).rename(trackers / "B")
            for dataset, group in quality(stage, videos).items():
                summary[dataset][mode] = group["B"]
            previous_mode = mode
    speed = {f"{mode}_vs_{modes[0]}": paired_speed(keys, mode, modes[0]) for mode in modes[1:]}
    speed.update({f"{mode}_vs_{modes[1]}": paired_speed(keys, mode, modes[1]) for mode in modes[2:]})
    output = {"clips": len(videos), "verified_frames": 150 * len(items), "quality": summary, "speed": speed,
              "throughput": throughput_summary(items, modes)}
    (logs / "score_verified.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({ds: {mode: entry["combined"] for mode, entry in group.items()}
                      for ds, group in summary.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
