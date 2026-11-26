"""Index every qualifying GT turn in the tested RoundaboutHD window."""
import json
from pathlib import Path
from statistics import mean

import replay
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    output = root / "evaluation"
    source = replay.read(output / "scores.json")["quality"][replay.FULL]
    episodes = []
    for camera in replay.CAMERAS:
        gt_file = output / "gt" / replay.FULL / camera / "gt/gt.txt"
        gt = np.loadtxt(gt_file, delimiter=",", ndmin=2)
        ids, present, masks, turns = replay.labels_for_camera(gt_file, camera, frames=replay.FRAMES, fps=replay.FPS)
        runs = {}
        for label, _, _ in replay.METHODS:
            _, cfg, _ = replay.rte.build_configs(replay.FULL, label, None)
            cfg.update(GT_FOLDER=str(output / "gt"), TRACKERS_FOLDER=str(output / "trackers"),
                       SEQ_INFO=dict.fromkeys(replay.CAMERAS, replay.FRAMES))
            evaluator = replay.rte.dataset_class("mot")(cfg)
            raw = evaluator.get_raw_seq_data(label, camera)
            data = evaluator.get_preprocessed_seq_data(raw, "pedestrian")
            assert all(np.array_equal(original, ids[compact]) for original, compact in zip(raw["gt_ids"], data["gt_ids"]))
            matched, switched = replay.trace(data, present.shape)
            expected = source[label]["per_camera"][camera]
            assert switched.sum() == expected["idsw"] and (present & ~matched).sum() == expected["fn"]
            runs[label] = matched, switched
        by_camera = []
        for turn in turns:
            if turn["level"] != "turn25" or turn["last"] < replay.FIRST:
                continue
            first, last = max(turn["first"], replay.FIRST), turn["last"]
            gt_id = turn["gt_id"]
            index = np.searchsorted(ids, gt_id)
            interval = slice(first - 1, last)
            rows = gt[(gt[:, 1] == gt_id) & (first <= gt[:, 0]) & (gt[:, 0] <= last)]
            rows = rows[np.argsort(rows[:, 0])]
            assert np.array_equal(rows[:, 0], np.arange(first, last + 1))
            points = rows[:, 2:4] + rows[:, 4:6] * [.5, 1.]
            sampled = sorted({*range(0, len(points), 4), len(points) - 1})
            metrics = {}
            for label, (matched, switched) in runs.items():
                switches = (np.flatnonzero(switched[interval, index]) + first).tolist()
                metrics[label] = {"idsw": len(switches), "fn": int((~matched[interval, index]).sum()),
                                  "switch_frames": switches}
            seeds = [metrics[f"pf_s{i}"] for i in range(3)]
            by_camera.append({"camera": camera, "gt_id": gt_id, "first": first, "last": last,
                "original_first": turn["first"], "frames": last - first + 1, "peak_deg": turn["peak_deg"],
                "first_s": round((first - 1) / replay.FPS, 3), "end_s": round(last / replay.FPS, 3),
                "frames_turn35": int((masks["turn35"][interval, index] >= 0).sum()),
                "path": [[first + i, *[round(v, 1) for v in points[i]]] for i in sampled],
                "metrics": metrics, "pf_mean": {key: mean(s[key] for s in seeds) for key in ("idsw", "fn")}})
        for label, _, _ in replay.METHODS:
            reference = source[label]["suffix_scenarios_per_camera"][camera]["turn25"]
            assert sum(r["frames"] for r in by_camera) == reference["gt"]
            for key in ("idsw", "fn"):
                assert sum(r["metrics"][label][key] for r in by_camera) == reference[key], (camera, label, key)
        episodes.extend(sorted(by_camera, key=lambda row: (row["first"], row["gt_id"], row["last"])))
    assert sum(row["frames"] for row in episodes) == 13509
    result = {"dataset": "RoundaboutHD", "evaluation": "R111", "fps": replay.FPS,
              "first_frame": replay.FIRST, "last_frame": replay.FRAMES, "episodes": episodes}
    (root / "turn_episodes.json").write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("TURN_EPISODES", len(episodes), "FRAMES", sum(r["frames"] for r in episodes), flush=True)


if __name__ == "__main__":
    main()
