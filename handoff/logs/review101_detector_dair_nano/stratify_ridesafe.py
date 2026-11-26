"""Stratify RideSafe motorcycle detector recall by 640-pixel input object area."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review100_nano_pf_dataset"))
import pilot_ridesafe as pilot  # noqa: E402


def stratify(gt, detections, threshold, shape):
    counts = np.zeros((3, 2), dtype=int)
    scale = 640 / max(shape)
    false_positives = 0
    for index, objects in gt.items():
        boxes = np.asarray(list(objects.values()), dtype=float).reshape(-1, 4)
        prediction = detections[index][detections[index][:, 4] > threshold, :4]
        iou = pilot._BaseDataset._calculate_box_ious(boxes, prediction, box_format="x0y0x1y1")
        valid = iou >= .5
        r, c = linear_sum_assignment(valid * (1 + iou / (1 + min(iou.shape))), maximize=True)
        matched = r[valid[r, c]]
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) * scale ** 2
        group = np.digitize(areas, [32 ** 2, 96 ** 2])
        counts[:, 0] += np.bincount(group, minlength=3)
        counts[:, 1] += np.bincount(group[matched], minlength=3)
        false_positives += len(prediction) - len(matched)
    return counts.tolist(), false_positives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--weights", action="append", default=[], help="model_name=checkpoint_path")
    parser.add_argument("--track", action="store_true", help="also score fixed c25 KF and PF500 on these detections")
    args = parser.parse_args()
    if args.weights:
        pilot.CHECKPOINTS = dict(item.split("=", 1) for item in args.weights)
    totals = {}
    results = []
    for root in args.clips:
        manifest = json.loads((root / "source.json").read_text())
        gt, stride, shape, occluded = pilot.read_annotations(root / manifest["labels"]["file"])
        detections, fps, _, drift, checkpoints = pilot.predict_clip(root / manifest["video"]["file"], gt, shape, root)
        row = {"clip": manifest["clip"], "fps": fps / stride, "max_timestamp_drift_ms": drift,
               "occluded": occluded, "checkpoints": checkpoints, "metrics": {}}
        for model, arrays in detections.items():
            row["metrics"][model] = {}
            for threshold in (.05, .25):
                counts, fp = stratify(gt, arrays, threshold, shape)
                row["metrics"][model][str(threshold)] = {"size_gt_matched": counts, "fp": fp}
                key = f"{model}@{threshold}"
                totals[key] = totals.get(key, np.zeros((3, 2), dtype=int)) + counts
            if args.track:
                frequency = fps / stride
                cache = pilot.CachedMotorcycles(arrays, shape, frequency)
                row.setdefault("tracking", {})[model] = []
                for method in ("kf", "pf_motion2"):
                    cfg = pilot.load_tracker_cfg(pilot.ROOT / pilot.TRACKERS[method],
                                                 {"track_buffer": round(1.2 * frequency)})
                    for seed in ((None,) if method == "kf" else (0, 1, 2)):
                        mot, classes, timing = pilot.track_video(cache, cfg, dt_nominal=1 / frequency,
                                                                 elapsed_mode="steps", seed=seed,
                                                                 video=manifest["clip"])
                        assert len(timing) == len(gt)
                        assert set(classes["cls"]) <= {checkpoints[model]["motorcycle_class"]}
                        row["tracking"][model].append({"method": method, "seed": seed, **pilot.mot_scores(gt, mot)})
        results.append(row)
        print(json.dumps(row), flush=True)
        args.out.write_text(json.dumps({"clips": results, "size_groups": ["small", "medium", "large"],
                                        "totals_size_gt_matched": {k: v.tolist() for k, v in totals.items()}},
                                       indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
