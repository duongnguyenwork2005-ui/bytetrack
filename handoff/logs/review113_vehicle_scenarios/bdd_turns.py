"""Stratify the R110 BDD40 vehicle CLEAR histories by native GT class and heading."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "review99_roundabout_turns"))
from audit_turns import labels_for_camera, np, rte, trace

PRIOR = HERE.parent / "review110_nano_filters" / "replay_full"
SPLIT = "BDDCOCOVID-val"


def run():
    stored = json.loads((PRIOR / "scores.json").read_text(encoding="utf-8"))
    captures = [row for row in stored["capture"] if row["dataset"] == "bdd"]
    assert len(captures) == 40
    types = pd.read_csv(ROOT / f"data/processed/{SPLIT}/vehicle_types.csv")
    methods = ["kf", "ekf", "ukf"]
    counters, rows, episodes_count = defaultdict(Counter), [], Counter()
    for item in captures:
        clip = item["name"]
        gt = np.loadtxt(PRIOR / "gt" / SPLIT / clip / "gt/gt.txt", delimiter=",", ndmin=2)
        eligible = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
        clean = HERE / "bdd_clean_gt" / f"{clip}.txt"
        clean.parent.mkdir(exist_ok=True)
        np.savetxt(clean, eligible, delimiter=",", fmt="%.10g")
        ids, present, levels, episodes = labels_for_camera(clean, clip, 150, 5)
        class_map = types[types.video == clip].set_index("track_id").category.to_dict()
        classes = np.array([class_map[k] for k in ids])
        masks = {"all": present, "turn25": levels["turn25"] >= 0,
                 "straight8": levels["straight8"] >= 0}
        for e in episodes:
            episodes_count[(class_map[e["gt_id"]], e["level"])] += 1
        for method in methods:
            _, cfg, _ = rte.build_configs(SPLIT, method, None)
            cfg.update(PRINT_CONFIG=False, GT_FOLDER=str(PRIOR / "gt"), TRACKERS_FOLDER=str(PRIOR / "trackers"), SEQ_INFO={clip: 150})
            dataset = rte.dataset_class("mot_crowd")(cfg)
            raw = dataset.get_raw_seq_data(method, clip)
            data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
            for t, idx in enumerate(data["gt_ids"]):
                assert np.array_equal(ids[idx], eligible[eligible[:, 0] == t + 1, 1])
            matched, changed = trace(data, present.shape)
            misses = present & ~matched
            expected = stored["quality"]["bdd"][method]["videos"][clip]
            assert changed.sum() == expected["idsw"] and misses.sum() == expected["fn"], (clip, method)
            for category in ["car", "motorcycle", "vehicle"]:
                class_mask = classes == category if category != "vehicle" else np.ones(len(ids), bool)
                for group, mask in masks.items():
                    chosen = mask & class_mask[None, :]
                    count = {"gt": int(chosen.sum()), "FN": int((chosen & misses).sum()), "IDSW": int((chosen & changed).sum())}
                    row = {"clip": clip, "class": category, "group": group, "method": method, **count}
                    rows.append(row)
                    counters[category, group, method].update(count)
        print(clip, flush=True)
    result = {"clips": 40, "parity_checks": 120,
              "aggregate": {"/".join(k): dict(v) for k, v in counters.items()},
              "episodes": {"/".join(k): v for k, v in episodes_count.items()}, "per_clip": rows}
    (HERE / "bdd_turns.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"]), flush=True)


if __name__ == "__main__":
    run()
