"""Apply R92 UA gate and paired video bootstrap to the fixed dev12 runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "results/trackeval/DETRAC-all"
OUT = ROOT / "handoff/logs/review92_pf_modes"
VIDEOS = tuple(json.loads((ROOT / "results/phase2/manifests/detrac_dev.json").read_text())["videos"])


def path(case, seed):
    if case == "baseline":
        name = f"p2-kc48_r72_pf_dev-A-motion200-s{seed}"
    else:
        name = f"r92-ua-dev-cv50-s{seed}"
    return SCORES / name


def main():
    totals = {}
    by_video = {}
    for case in ("baseline", "candidate"):
        combined, video = [], []
        for seed in (0, 1, 2):
            base = path(case, seed)
            assert (base / "raw_fields.json").exists()
            c = pd.read_csv(base / "combined_metrics.csv").iloc[0]
            combined.append({name: float(c[name]) for name in ("HOTA", "IDF1", "MOTA", "IDSW", "FP")})
            v = pd.read_csv(base / "per_video_metrics.csv")
            assert set(v.video) == set(VIDEOS) and len(v) == 12
            video.append(v.set_index("video")["IDSW"].reindex(VIDEOS).to_numpy(float))
        totals[case] = {key: float(np.mean([row[key] for row in combined])) for key in combined[0]}
        by_video[case] = np.mean(np.stack(video), axis=0)
    baseline, candidate = totals["baseline"], totals["candidate"]
    requirements = {"IDSW_max": .9 * baseline["IDSW"],
                    "HOTA_min": baseline["HOTA"] - .005,
                    "IDF1_min": baseline["IDF1"] - .005,
                    "FP_max": 1.05 * baseline["FP"]}
    passed = (candidate["IDSW"] <= requirements["IDSW_max"] and
              candidate["HOTA"] >= requirements["HOTA_min"] and
              candidate["IDF1"] >= requirements["IDF1_min"] and
              candidate["FP"] <= requirements["FP_max"])
    diff = by_video["candidate"] - by_video["baseline"]
    rng = np.random.default_rng(9206)
    ix = rng.integers(0, len(VIDEOS), size=(5000, len(VIDEOS)))
    bootstrap = diff[ix].sum(axis=1)
    result = {"scope": "UA-DETRAC train dev12; exploratory", "baseline": baseline,
              "candidate": candidate, "requirements": requirements, "gate_pass": bool(passed),
              "video_count_candidate_lower_IDSW": int(np.sum(diff < 0)),
              "video_count_candidate_higher_IDSW": int(np.sum(diff > 0)),
              "paired_video_IDSW_delta_sum": float(diff.sum()),
              "paired_video_bootstrap95_IDSW_delta": [float(v) for v in np.quantile(bootstrap, [.025, .975])],
              "per_video_IDSW_delta": {name: float(value) for name, value in zip(VIDEOS, diff)}}
    (OUT / "dev_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
