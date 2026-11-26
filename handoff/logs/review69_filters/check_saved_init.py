"""Count nonuniform priors at motion initialization from saved R65 diagnostic rows."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data/processed/trackers/RoundaboutHD-prefix3000"


def count_initializations(path):
    previous = {}
    initialized = set()
    counts = Counter()
    with path.open(encoding="utf-8", newline="") as src:
        for row in csv.DictReader(src):
            track_id = int(row["track_id"])
            frame = int(row["frame"])
            old = previous.get(track_id)
            if old is not None:
                assert old[0] < frame, (path, frame, track_id)
            if row["kind"] == "update":
                # Prediction preserves logw until the next observation.
                previous[track_id] = (frame, row["resampled"] == "True",
                                      row["degenerate"] == "True", float(row["ess_ratio"]))
                continue
            assert row["kind"] == "init_motion" and track_id not in initialized
            initialized.add(track_id)
            if old is None:
                counts["uniform_from_activation"] += 1
            elif old[1] or old[2]:
                counts["uniform_from_previous_reset"] += 1
            elif old[3] < 1 - 1e-12:
                counts["nonuniform"] += 1
            else:
                counts["near_uniform_ess"] += 1
    return counts


if __name__ == "__main__":
    result = {}
    for seed in range(3):
        folder = BASE / f"r65bon-yolov8n_mixed5-c25-pf500-s{seed}" / "pf_diag"
        scenes = {f"imagesc{i:03d}": count_initializations(folder / f"imagesc{i:03d}.csv")
                  for i in range(1, 5)}
        total = sum(scenes.values(), Counter())
        result[seed] = dict(scenes=scenes, by_cause=total, init_events=sum(total.values()))
    print(json.dumps(result, indent=2))
