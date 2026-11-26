"""Inventory V2X-Seq-SPD roadside labels and frame cadence before scoring."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from decimal import Decimal
import json
import math
from pathlib import Path
from statistics import median


CLASSES = ("Car", "Motorcyclist")
UNITS = {"s": 1, "ms": 1000, "us": 1000000, "ns": 1000000000}


def source_file(side, relative):
    path = (side / relative).resolve()
    if not path.is_relative_to(side):
        raise ValueError(f"Data path escapes infrastructure-side: {relative!r}")
    return path


def box_is_valid(value):
    try:
        left, top, right, bottom = (float(value[field]) for field in ("xmin", "ymin", "xmax", "ymax"))
    except (KeyError, TypeError, ValueError):
        return False
    return all(map(math.isfinite, (left, top, right, bottom))) and left < right and top < bottom


def load_audit(dataset):
    side = (dataset / "infrastructure-side").resolve(strict=True)
    manifest = json.loads((side / "data_info.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Expected a nonempty list in infrastructure-side/data_info.json")
    by_scene = defaultdict(list)
    for item in manifest:
        key = (str(item["intersection_loc"]), str(item["camera_id"]), str(item["sequence_id"]))
        by_scene[key].append(item)
    scenes = []
    raw_steps = []
    for key, frames in sorted(by_scene.items()):
        ordered = sorted(frames, key=lambda item: Decimal(str(item["image_timestamp"])))
        times = [Decimal(str(item["image_timestamp"])) for item in ordered]
        steps = [b - a for a, b in zip(times, times[1:])]
        raw_steps.extend(step for step in steps if step > 0)
        totals = Counter()
        classes = {category: Counter() for category in CLASSES}
        state = {category: Counter() for category in CLASSES}
        tracks = {category: set() for category in CLASSES}
        frame_ids = [str(item["frame_id"]) for item in ordered]
        for item in ordered:
            picture = source_file(side, item["image_path"])
            label = source_file(side, item["label_camera_std_path"])
            totals["images"] += picture.is_file()
            if not label.is_file():
                continue
            totals["label_files"] += 1
            objects = json.loads(label.read_text(encoding="utf-8"))
            seen = set()
            for obj in objects:
                category = obj.get("type")
                if category not in classes:
                    continue
                counts = classes[category]
                counts["objects"] += 1
                state[category][str(obj.get("occluded_state", "MISSING"))] += 1
                identity = str(obj.get("track_id", ""))
                if not identity or identity in {"-1", "None"}:
                    counts["invalid_id"] += 1
                    continue
                if identity in seen:
                    counts["duplicate_id"] += 1
                seen.add(identity)
                if box_is_valid(obj.get("2d_box")):
                    counts["valid_2d_boxes"] += 1
                    tracks[category].add(identity)
                else:
                    counts["invalid_2d_boxes"] += 1
        scenes.append({"intersection": key[0], "camera": key[1], "sequence": key[2], "frames": len(ordered),
                       "images_present": totals["images"], "labels_present": totals["label_files"],
                       "duplicate_frame_ids": len(frame_ids) - len(set(frame_ids)),
                       "nonpositive_time_steps": sum(s <= 0 for s in steps),
                       "median_positive_timestamp_delta_raw": str(median(s for s in steps if s > 0)) if any(s > 0 for s in steps) else None,
                       "classes": {category: {**classes[category], "track_ids_with_valid_box": len(tracks[category]),
                                               "occluded_state_raw": dict(state[category])} for category in CLASSES}})
    candidate_units = []
    if raw_steps:
        mid = median(raw_steps)
        candidate_units = [unit for unit, scale in UNITS.items() if Decimal("0.08") <= mid / scale <= Decimal("0.12")]
    return {"source": str(side), "frame_rows": len(manifest), "sequence_camera_groups": len(scenes),
            "timestamp_units_consistent_with_nominal_10hz": candidate_units,
            "occlusion_codes_retained_raw": True, "has_scoring_results": False, "scenes": scenes}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="extracted root containing infrastructure-side")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = load_audit(args.dataset)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
