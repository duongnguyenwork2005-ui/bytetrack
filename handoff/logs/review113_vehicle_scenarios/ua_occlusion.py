"""Audit real UA-DETRAC occlusion strata on the frozen R110 40 clips."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "review110_nano_filters/replay_full"
sys.path.insert(0, str(HERE.parent / "review99_roundabout_turns"))
from audit_turns import labels_for_camera, np, rte, trace


def run():
    stored = json.loads((PRIOR / "scores.json").read_text(encoding="utf-8"))
    captures = [row for row in stored["capture"] if row["dataset"] == "ua"]
    assert len(captures) == 40
    counters, rows = defaultdict(Counter), []
    split = "DETRAC-test"
    for item in captures:
        clip = item["name"]
        path = PRIOR / "gt" / split / clip / "gt/gt.txt"
        gt = np.loadtxt(path, delimiter=",", ndmin=2)
        assert np.isfinite(gt[:, 8]).all() and (gt[:, 8] >= 0).all() and (gt[:, 8] <= 1).all()
        ids, present, levels, _ = labels_for_camera(path, clip, 150, 25)
        low = np.zeros_like(present)
        low[gt[:, 0].astype(int) - 1, np.searchsorted(ids, gt[:, 1])] = gt[:, 8] <= .5
        recent = low.copy()
        for lag in range(1, 26):
            recent[lag:] |= low[:-lag]
        turn = levels["turn25"] >= 0
        masks = {"all": present, "low": low, "recent": recent & present,
                 "recovery": recent & present & ~low, "turn25": turn, "turn25_low": turn & low}
        for method in ["kf", "ekf", "ukf"]:
            _, cfg, _ = rte.build_configs(split, method, None)
            cfg.update(PRINT_CONFIG=False, GT_FOLDER=str(PRIOR / "gt"), TRACKERS_FOLDER=str(PRIOR / "trackers"), SEQ_INFO={clip: 150})
            dataset = rte.dataset_class("mot")(cfg)
            raw = dataset.get_raw_seq_data(method, clip)
            data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
            assert all(np.array_equal(a, ids[b]) for a, b in zip(raw["gt_ids"], data["gt_ids"]))
            matched, changed = trace(data, present.shape)
            miss = present & ~matched
            old = stored["quality"]["ua"][method]["videos"][clip]
            assert changed.sum() == old["idsw"] and miss.sum() == old["fn"]
            for group, mask in masks.items():
                row = {"gt": int(mask.sum()), "IDSW": int((mask & changed).sum()), "FN": int((mask & miss).sum())}
                counters[method, group].update(row)
                rows.append({"clip": clip, "method": method, "group": group, **row})
    result = {"clips": 40, "parity_checks": 120, "groups": {"/".join(k): dict(v) for k, v in counters.items()}, "per_clip": rows}
    (HERE / "ua_occlusion.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["groups"]), flush=True)


if __name__ == "__main__":
    run()
