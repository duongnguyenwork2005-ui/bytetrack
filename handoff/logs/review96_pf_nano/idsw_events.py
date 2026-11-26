"""Replay TrackEval CLEAR's matching to locate *exact* IDSW frames.

The gap labels are observational descriptions, not internal tracker state or causal proof.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import run_trackeval as rte
from r92_pf_modes.eval import DEV
from r74_pf_bdd import MANIFEST, SPLIT

OUT = ROOT / "handoff/logs/review96_pf_nano"


def trace(data):
    """Same score/match/IDSW rule as installed trackeval.metrics.clear.CLEAR."""
    prev_id = np.full(data["num_gt_ids"], np.nan)
    prev_step_id = np.full(data["num_gt_ids"], np.nan)
    last_matched = np.full(data["num_gt_ids"], -1)
    last_reappearance = np.full(data["num_gt_ids"], -1)
    events = []
    count = 0
    eps = np.finfo(float).eps
    for t, (gt, tr) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
        if len(gt) == 0 or len(tr) == 0:
            continue
        sim = data["similarity_scores"][t]
        score = 1000 * (tr[np.newaxis, :] == prev_step_id[gt[:, np.newaxis]]) + sim
        score[sim < .5 - eps] = 0
        rows, cols = linear_sum_assignment(-score)
        keep = score[rows, cols] > eps
        rows, cols = rows[keep], cols[keep]
        matched_gt, matched_tr = gt[rows], tr[cols]
        old = prev_id[matched_gt]
        switched = ~np.isnan(old) & (matched_tr != old)
        count += int(switched.sum())
        for g, old_id, new_id in zip(matched_gt[switched], old[switched], matched_tr[switched]):
            gap = t - int(last_matched[g]) - 1
            events.append({"frame": t + 1, "gt_id_compact": int(g),
                           "old_track_compact": int(old_id), "new_track_compact": int(new_id),
                           "unmatched_frames_since_last_match": gap,
                           "at_reappearance": bool(gap > 0),
                           "within_3_after_reappearance": bool(gap == 0 and last_reappearance[g] >= 0
                                                                 and t - last_reappearance[g] <= 3)})
        gaps = t - last_matched[matched_gt] - 1
        reappeared = (last_matched[matched_gt] >= 0) & (gaps > 0)
        last_reappearance[matched_gt[reappeared]] = t
        last_matched[matched_gt] = t
        prev_id[matched_gt] = matched_tr
        prev_step_id[:] = np.nan
        prev_step_id[matched_gt] = matched_tr
    return count, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ua", "bdd"), required=True)
    ap.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--limit", type=int, default=0, help="0 = all videos")
    a = ap.parse_args()
    if a.dataset == "ua":
        split, kind, videos = "DETRAC-all", "mot", list(DEV)
        # Old R72 MOT files were archived. R95 smoke retained byte-identical baseline.
        names = ((f"r95-ua-smoke-base-s{a.seed}", f"r95-ua-smoke-rescue-s{a.seed}")
                 if a.limit == 2 else (f"r95-ua-dev-rescue-s{a.seed}",))
    else:
        split, kind = SPLIT, "mot_crowd"
        videos = json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]
        names = (f"r74-bdd-full-motion200-s{a.seed}",
                 f"r95-bdd-full-rescue-s{a.seed}")
    if a.limit:
        videos = videos[:a.limit]
    all_out = {}
    labels = ("baseline", "candidate") if len(names) == 2 else ("candidate",)
    for label, tracker in zip(labels, names):
        _, cfg, _ = rte.build_configs(split, tracker, videos)
        dataset = rte.dataset_class(kind)(cfg)
        metrics = pd.read_csv(ROOT / "results/trackeval" / split / tracker / "per_video_metrics.csv")
        expected = metrics.set_index("video")["IDSW"]
        rows = []
        for v in videos:
            raw = dataset.get_raw_seq_data(tracker, v)
            data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
            count, events = trace(data)
            assert count == int(expected[v]), (tracker, v, count, expected[v])
            c = Counter("reappearance" if e["at_reappearance"] else "continuous" for e in events)
            soon = sum(e["within_3_after_reappearance"] for e in events)
            rows.append({"video": v, "IDSW": count, "reappearance": c["reappearance"],
                         "continuous": c["continuous"], "within_3_after_reappearance": soon,
                         "events": events})
        all_out[label] = {"tracker": tracker, "videos": rows,
                          "summary": {k: sum(r[k] for r in rows)
                                      for k in ("IDSW", "reappearance", "continuous",
                                                "within_3_after_reappearance")}}
        print(label, all_out[label]["summary"], flush=True)
    out = OUT / f"idsw_{a.dataset}_s{a.seed}_{len(videos)}.json"
    out.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(out, flush=True)


if __name__ == "__main__":
    main()
