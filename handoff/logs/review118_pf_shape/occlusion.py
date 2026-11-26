"""Attribute matched CLEAR events to SVMOT's existing low-visibility group."""

from collections import Counter
import json
import sys

import run as job
from transfer import VIDEOS, TOP
sys.path.insert(0, str(TOP.parent / "review99_roundabout_turns"))
from audit_turns import trace


def run():
    result, grouped = [], {}
    for mode in ("parent", "candidate"):
        for seed in range(3):
            tracker = f"{mode}-s{seed}"
            _, cfg, _ = job.base.rte.build_configs("SVMOTSYS-test", tracker, None)
            cfg.update(PRINT_CONFIG=False, TRACKERS_FOLDER=str(TOP / "transfer/tracking"),
                       SEQ_INFO={video: None for video in VIDEOS})
            dataset = job.trackeval.datasets.MotChallenge2DBox(cfg)
            metrics = job.base.boot_stats.load_raw_fields(TOP / f"transfer/evaluation/{tracker}/raw_fields.json")
            total = Counter()
            for video in VIDEOS:
                gt = job.np.loadtxt(job.ROOT / "data/processed/SVMOTSYS-test" / video / "gt/gt.txt", delimiter=",")
                assert (gt[:, 6:8] == 1).all()
                ids = job.np.unique(gt[:, 1]).astype(int)
                raw = dataset.get_raw_seq_data(tracker, video)
                data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
                shape = data["num_timesteps"], len(ids)
                present = job.np.zeros(shape, dtype=bool)
                low = job.np.zeros_like(present)
                indexes = gt[:, 0].astype(int)-1, job.np.searchsorted(ids, gt[:, 1])
                present[indexes] = True
                low[indexes] = gt[:, 8] <= .5
                assert present.sum() == data["num_gt_dets"]
                assert all(job.np.array_equal(ids[compact], original) for compact, original in zip(data["gt_ids"], raw["gt_ids"]))
                matched, changed = trace(data, shape)
                missed = present & ~matched
                assert changed.sum() == metrics[video]["IDSW"]
                assert missed.sum() == metrics[video]["CLR_FN"]
                row = {"boxes": int(low.sum()), "IDSW": int((low & changed).sum()),
                       "FN": int((low & missed).sum())}
                total.update(row)
                result.append({"mode": mode, "seed": seed, "video": video, **row})
            grouped.setdefault(mode, {})[str(seed)] = dict(total)
    means = {mode: {key: float(job.np.mean([row[key] for row in seeds.values()]))
                    for key in ("boxes", "IDSW", "FN")} for mode, seeds in grouped.items()}
    job.write(TOP / "occlusion_summary.json", {"group": "native visibility <= 0.5",
                                              "means": means, "seeds": grouped, "by_video": result})
    print(means, flush=True)


if __name__ == "__main__":
    run()
