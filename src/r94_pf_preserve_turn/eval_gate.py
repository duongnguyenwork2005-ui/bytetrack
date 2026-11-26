"""TrackEval and preregistered two-domain gate for R94."""
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
from r92_pf_modes.eval import DEV, SMOKE as UA_SMOKE
from r74_pf_bdd import MANIFEST, SMOKE as BDD_SMOKE

OUT = ROOT / "handoff/logs/review94_pf_preserve_turn"


def setup(domain, scope):
    if domain == "ua":
        split = "DETRAC-all"
        videos = list(UA_SMOKE if scope == "smoke" else DEV)
        name = f"r94-ua-{'smoke' if scope == 'smoke' else 'dev'}-preserve"
        baseline = "p2-kc48_r72_pf_dev-A-motion200"
        delta_idsw = .9
    else:
        split = "BDDCOCOVID-val"
        videos = list(BDD_SMOKE if scope == "smoke" else json.loads(MANIFEST.read_text())["videos"])
        name = f"r94-bdd-{scope}-preserve"
        baseline = "r74-bdd-full-motion200"
        delta_idsw = .97
    return split, videos, name, baseline, delta_idsw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("ua", "bdd"), required=True)
    parser.add_argument("--scope", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    split, videos, name, baseline, idsw_ratio = setup(args.domain, args.scope)
    rows = []
    for seed in (0, 1, 2):
        run = f"{name}-s{seed}"
        path = ROOT / "results/trackeval" / split / run
        if not (path / "combined_metrics.csv").exists():
            r64.cham(split, run, videos)
        c = pd.read_csv(path / "combined_metrics.csv").iloc[0]
        rows.append({"seed": seed, **{key: float(c[key]) for key in ("HOTA", "IDF1", "MOTA", "IDSW", "FP")}})
        print(rows[-1], flush=True)
    (OUT / f"{args.domain}_{args.scope}_scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.scope == "smoke":
        return

    totals, by_video = {}, {}
    for case, prefix in (("baseline", baseline), ("candidate", name)):
        combined, video = [], []
        for seed in (0, 1, 2):
            root = ROOT / "results/trackeval" / split / f"{prefix}-s{seed}"
            assert (root / "raw_fields.json").exists()
            row = pd.read_csv(root / "combined_metrics.csv").iloc[0]
            combined.append({key: float(row[key]) for key in ("HOTA", "IDF1", "MOTA", "IDSW", "FP")})
            per = pd.read_csv(root / "per_video_metrics.csv")
            assert len(per) == len(videos) and set(per.video) == set(videos)
            video.append(per.set_index("video")["IDSW"].reindex(videos).to_numpy(float))
        totals[case] = {key: float(np.mean([r[key] for r in combined])) for key in combined[0]}
        by_video[case] = np.mean(np.stack(video), axis=0)
    base, candidate = totals["baseline"], totals["candidate"]
    limits = {"IDSW_max": base["IDSW"] * idsw_ratio,
              "HOTA_min": base["HOTA"] - .005,
              "IDF1_min": base["IDF1"] - .005,
              "FP_max": base["FP"] * 1.05}
    diff = by_video["candidate"] - by_video["baseline"]
    rng = np.random.default_rng(9406)
    bootstrap = diff[rng.integers(0, len(videos), (5000, len(videos)))].sum(axis=1)
    pass_metrics = (candidate["IDSW"] <= limits["IDSW_max"] and
                    candidate["HOTA"] >= limits["HOTA_min"] and
                    candidate["IDF1"] >= limits["IDF1_min"] and
                    candidate["FP"] <= limits["FP_max"])
    if args.domain == "bdd":
        pass_metrics = pass_metrics and np.sum(diff < 0) > np.sum(diff > 0)
    result = {"domain": args.domain, "scope": "already viewed; exploratory",
              "baseline": base, "candidate": candidate, "limits": limits,
              "gate_pass": bool(pass_metrics),
              "lower_IDSW_videos": int(np.sum(diff < 0)),
              "higher_IDSW_videos": int(np.sum(diff > 0)),
              "equal_IDSW_videos": int(np.sum(diff == 0)),
              "IDSW_delta": float(diff.sum()),
              "paired_video_bootstrap95_IDSW_delta": [float(v) for v in np.quantile(bootstrap, [.025, .975])],
              "per_video_IDSW_delta": {v: float(d) for v, d in zip(videos, diff)}}
    (OUT / f"{args.domain}_full_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_video_IDSW_delta"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
