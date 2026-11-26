"""Replay R73 PF CLEAR events inside the established R99 RoundaboutHD GT strata."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / "review99_roundabout_turns"
sys.path.insert(0, str(PREVIOUS))
from audit_turns import CAMERAS, ROOT, SPLIT, labels_for_camera, rte, trace  # noqa: E402


def main():
    sources = {camera: labels_for_camera(ROOT / "data/processed" / SPLIT / camera / "gt/gt.txt", camera)
               for camera in CAMERAS}
    prior = json.loads((PREVIOUS / "audit.json").read_text(encoding="utf-8"))
    rows = [row for row in prior["summary"] if row["config"] == "yolov8n_mixed5/c25"
            and row["filter"] in {"cv", "ekf", "ukf"}]
    checks = 0
    for scale in (100, 200):
        for seed in (0, 1, 2):
            run = f"r73-roundabout-yolov8n_mixed5-c25-motion{scale}-s{seed}"
            _, config, _ = rte.build_configs(SPLIT, run, CAMERAS)
            dataset = rte.dataset_class("mot")(config)
            with (ROOT / "results/trackeval" / SPLIT / run / "per_video_metrics.csv").open() as file:
                reference = {row["video"]: row for row in csv.DictReader(file)}
            for camera, (ids, present, levels, episodes) in sources.items():
                raw = dataset.get_raw_seq_data(run, camera)
                data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
                assert len(ids) == data["num_gt_ids"]
                for raw_ids, compact in zip(raw["gt_ids"], data["gt_ids"]):
                    assert np.array_equal(raw_ids, ids[compact])
                matched, changed = trace(data, present.shape)
                assert not (matched & ~present).any()
                missed = present & ~matched
                assert changed.sum() == int(float(reference[camera]["IDSW"]))
                assert missed.sum() == int(float(reference[camera]["FN"]))
                checks += 1
                for level, indices in levels.items():
                    mask = indices >= 0
                    rows.append({"camera": camera, "config": "yolov8n_mixed5/c25",
                                 "filter": f"pf_k1k2_motion{scale}-s{seed}", "level": level,
                                 "episodes": sum(item["level"] == level for item in episodes),
                                 "gt": int(mask.sum()), "idsw": int((changed & mask).sum()),
                                 "fn": int((missed & mask).sum())})
    assert checks == 24
    (OUT / "roundabout_pf_turns.json").write_text(json.dumps({"parity_checks": checks, "rows": rows}, indent=2))
    print(f"RoundaboutHD replay exact on {checks} camera/run pairs", flush=True)


if __name__ == "__main__":
    main()
