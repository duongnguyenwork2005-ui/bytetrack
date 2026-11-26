"""Confirm motorcycle recall by size and ByteTrack KF on the two R102 holdout clips."""
import argparse
import json
from pathlib import Path
import sys

LOGS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(LOGS / folder) for folder in ("review101_detector_dair_nano", "review102_ridesafe_pf_nano")]
from eval_ridesafe import pilot, track
from stratify_ridesafe import stratify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("weights", nargs="+")
    args = parser.parse_args()
    pilot.CHECKPOINTS = dict(spec.split("=", 1) for spec in args.weights)
    records = json.loads((args.sources / "sources.json").read_text(encoding="utf-8"))["test"]
    assert len(records) == 2 and {record["clip"][:8] for record in records} == {"20221017"}
    report = []
    for record in records:
        clip = record["clip"]
        root = args.sources / "test" / clip
        preview = args.output.parent / "ridesafe_previews" / clip
        preview.mkdir(parents=True, exist_ok=True)
        gt, stride, shape, _ = pilot.read_annotations(root / record["labels"]["file"])
        predictions, fps, frames, drift, checkpoints = pilot.predict_clip(root / record["video"]["file"], gt, shape, preview)
        models = {}
        for name, detections in predictions.items():
            counts, fp = stratify(gt, detections, .25, shape)
            models[name] = {"size_gt_matched": counts, "fp": fp,
                            "tracking": track(gt, detections, fps / stride, shape, clip, .25, "kf", None)}
        report.append({"clip": clip, "frames": frames, "labeled_frames": len(gt), "stride": stride,
                       "source_fps": fps, "timestamp_drift_ms": drift, "checkpoints": checkpoints, "models": models})
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(clip, len(gt), flush=True)


if __name__ == "__main__":
    main()
