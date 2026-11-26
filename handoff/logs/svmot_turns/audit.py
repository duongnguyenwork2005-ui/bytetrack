"""Replay R64 ByteTrack outputs on all SVMOT test sequences by GT motion."""
from concurrent.futures import ProcessPoolExecutor
from configparser import ConfigParser
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "review99_roundabout_turns"))
from audit_turns import DETECTORS, ROOT, labels_for_camera, rte, trace

SPLIT = "SVMOTSYS-test"
SEQUENCES = [f"seq_{i}_{'morning' if i < 26 else 'evening'}" for i in range(22, 31)]
CONFIGS = list(itertools.product(DETECTORS, ("c25", "c10")))


def audit_config(job):
    detector, condition = job
    gt_root = ROOT / "data/processed" / SPLIT
    source = {}
    for seq in SEQUENCES:
        cfg = ConfigParser()
        assert cfg.read(gt_root / seq / "seqinfo.ini")
        assert cfg.getint("Sequence", "frameRate") == 30
        source[seq] = labels_for_camera(gt_root / seq / "gt/gt.txt", seq,
                                        frames=cfg.getint("Sequence", "seqLength"), fps=30)
    result = []
    for method in ("cv", "ekf", "ukf"):
        tracker = f"r64-{detector}-{condition}-{method}"
        _, cfg, _ = rte.build_configs(SPLIT, tracker, SEQUENCES)
        dataset = rte.dataset_class("mot")(cfg)
        with (ROOT / "results/trackeval" / SPLIT / tracker / "per_video_metrics.csv").open() as stream:
            expected = {row["video"]: row for row in csv.DictReader(stream)}
        for seq, (ids, present, groups, episodes) in source.items():
            raw = dataset.get_raw_seq_data(tracker, seq)
            data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
            assert len(ids) == data["num_gt_ids"]
            for t, (original, mapped) in enumerate(zip(raw["gt_ids"], data["gt_ids"])):
                assert np.array_equal(original, ids[mapped]) and present[t].sum() == len(original)
            matched, changed = trace(data, present.shape)
            missing = present & ~matched
            assert (matched & ~present).sum() == 0
            assert changed.sum() == int(expected[seq]["IDSW"]), (tracker, seq, "IDSW")
            assert missing.sum() == int(expected[seq]["FN"]), (tracker, seq, "FN")
            for group, index in groups.items():
                selected = index >= 0
                result.append({"sequence": seq, "detector": detector, "condition": condition, "filter": method,
                               "group": group, "episodes": sum(ep["level"] == group for ep in episodes),
                               "gt": int(selected.sum()), "idsw": int((selected & changed).sum()),
                               "fn": int((selected & missing).sum())})
                for metric, matrix in (("idsw", changed), ("fn", missing)):
                    counts = np.bincount(index[selected & matrix], minlength=len(episodes))
                    for i in np.unique(index[selected]):
                        episodes[i].setdefault("scores", {}).setdefault(method, {})[metric] = int(counts[i])
    return result, [ep for seq in SEQUENCES for ep in source[seq][-1]]


def main():
    rows = []
    examples = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        for (detector, condition), (group, episodes) in zip(CONFIGS, pool.map(audit_config, CONFIGS)):
            rows.extend(group)
            examples[f"{detector}/{condition}"] = [ep for ep in episodes if ep["level"] == "turn25"]
            print(detector, condition, "27/27 IDSW and FN", flush=True)
    parity = len(CONFIGS) * len(SEQUENCES) * 3
    (HERE / "audit.json").write_text(json.dumps({"parity_checks": parity, "rows": rows, "turns": examples},
                                               separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
