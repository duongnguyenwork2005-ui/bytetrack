"""Evaluate the frozen R95 PF association on BDD and apply its preregistered gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from r64_ngoai import danh_gia_ngoai as r64
from r74_pf_bdd import MANIFEST, SMOKE, SPLIT
import boot_stats

OUT = ROOT / "handoff/logs/review95_pf_reacquire"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    videos = SMOKE if args.scope == "smoke" else json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]
    rows = []
    for seed in (0, 1, 2):
        name = f"r95-bdd-{args.scope}-rescue-s{seed}"
        path = ROOT / "results/trackeval" / SPLIT / name
        if not (path / "combined_metrics.csv").exists():
            r64.cham(SPLIT, name, videos)
        c = pd.read_csv(path / "combined_metrics.csv").iloc[0]
        rows.append({"seed": seed, **{key: float(c[key]) for key in ("HOTA", "IDF1", "MOTA", "IDSW", "FP")}})
        print(rows[-1], flush=True)
    (OUT / f"bdd_{args.scope}_scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.scope == "smoke":
        return

    totals, per_video = {}, {}
    for case, prefix in (("baseline", "r74-bdd-full-motion200"),
                         ("candidate", "r95-bdd-full-rescue")):
        combined, video = [], []
        for seed in (0, 1, 2):
            path = ROOT / "results/trackeval" / SPLIT / f"{prefix}-s{seed}"
            assert (path / "raw_fields.json").exists()
            row = pd.read_csv(path / "combined_metrics.csv").iloc[0]
            recombined = boot_stats.combine(boot_stats.load_raw_fields(path / "raw_fields.json"), videos)
            for key in ("HOTA", "IDF1", "MOTA", "IDSW", "FP"):
                assert abs(recombined[key] - float(row[key])) < 1e-9, (case, seed, key)
            combined.append({key: float(row[key]) for key in ("HOTA", "IDF1", "MOTA", "IDSW", "FP")})
            d = pd.read_csv(path / "per_video_metrics.csv")
            assert len(d) == 200 and set(d.video) == set(videos)
            video.append(d.set_index("video")["IDSW"].reindex(videos).to_numpy(float))
        totals[case] = {key: float(np.mean([r[key] for r in combined])) for key in combined[0]}
        per_video[case] = np.mean(np.stack(video), axis=0)
    base, cand = totals["baseline"], totals["candidate"]
    limits = {"IDSW_max": base["IDSW"] * .97, "HOTA_min": base["HOTA"] - .005,
              "IDF1_min": base["IDF1"] - .005, "FP_max": base["FP"] * 1.05}
    diff = per_video["candidate"] - per_video["baseline"]
    rng = np.random.default_rng(9507)
    bs = diff[rng.integers(0, len(videos), (5000, len(videos)))].sum(axis=1)
    passed = (cand["IDSW"] <= limits["IDSW_max"] and cand["HOTA"] >= limits["HOTA_min"]
              and cand["IDF1"] >= limits["IDF1_min"] and cand["FP"] <= limits["FP_max"]
              and np.sum(diff < 0) > np.sum(diff > 0))
    result = {"scope": "BDD COCO-video val200 already viewed; exploratory", "baseline": base,
              "candidate": cand, "limits": limits, "gate_pass": bool(passed),
              "lower_IDSW_videos": int(np.sum(diff < 0)),
              "higher_IDSW_videos": int(np.sum(diff > 0)),
              "equal_IDSW_videos": int(np.sum(diff == 0)),
              "paired_video_bootstrap95_IDSW_delta": [float(v) for v in np.quantile(bs, [.025, .975])],
              "per_video_IDSW_delta": {v: float(d) for v, d in zip(videos, diff)}}
    (OUT / "bdd_full_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_video_IDSW_delta"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
