"""R93 fixed BDD domain-transfer gate and paired video IDSW bootstrap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "results/trackeval/BDDCOCOVID-val"
OUT = ROOT / "handoff/logs/review93_pf_bdd_transfer"
VIDEOS = tuple(json.loads((ROOT / "results/phase2/manifests/bdd100k_transfer_cocovid_val.json").read_text())["videos"])


def path(case, seed):
    name = f"r74-bdd-full-motion200-s{seed}" if case == "baseline" else f"r93-bdd-full-cv50-s{seed}"
    return SCORES / name


def main():
    totals, by_video = {}, {}
    for case in ("baseline", "candidate"):
        combined, video = [], []
        for seed in (0, 1, 2):
            root = path(case, seed)
            assert (root / "raw_fields.json").exists()
            row = pd.read_csv(root / "combined_metrics.csv").iloc[0]
            combined.append({name: float(row[name]) for name in ("HOTA", "IDF1", "MOTA", "IDSW", "FP")})
            d = pd.read_csv(root / "per_video_metrics.csv")
            assert len(d) == 200 and set(d.video) == set(VIDEOS)
            video.append(d.set_index("video").reindex(VIDEOS)["IDSW"].to_numpy(float))
        totals[case] = {key: float(np.mean([r[key] for r in combined])) for key in combined[0]}
        by_video[case] = np.mean(np.stack(video), axis=0)
    baseline, candidate = totals["baseline"], totals["candidate"]
    limits = {"IDSW_max": baseline["IDSW"] * .95,
              "HOTA_min": baseline["HOTA"] - .005,
              "IDF1_min": baseline["IDF1"] - .005,
              "FP_max": baseline["FP"] * 1.05}
    diff = by_video["candidate"] - by_video["baseline"]
    rng = np.random.default_rng(9306)
    bs = diff[rng.integers(0, len(diff), (5000, len(diff)))].sum(axis=1)
    passed = (candidate["IDSW"] <= limits["IDSW_max"] and
              candidate["HOTA"] >= limits["HOTA_min"] and
              candidate["IDF1"] >= limits["IDF1_min"] and
              candidate["FP"] <= limits["FP_max"] and
              np.sum(diff < 0) > np.sum(diff > 0))
    result = {"scope": "BDD COCO-video val200, already viewed; exploratory",
              "baseline": baseline, "candidate": candidate, "limits": limits,
              "transfer_gate_pass": bool(passed),
              "lower_IDSW_videos": int(np.sum(diff < 0)),
              "higher_IDSW_videos": int(np.sum(diff > 0)),
              "equal_IDSW_videos": int(np.sum(diff == 0)),
              "IDSW_delta": float(diff.sum()),
              "paired_video_bootstrap95_IDSW_delta": [float(v) for v in np.quantile(bs, [.025, .975])],
              "per_video_IDSW_delta": {video: float(value) for video, value in zip(VIDEOS, diff)}}
    (OUT / "full_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_video_IDSW_delta"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
