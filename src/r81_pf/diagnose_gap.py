"""Describe where PF motion-2 ID switches exceed EKF on the two reused sets."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRACK = ROOT / "results/trackeval"
OUT = ROOT / "results/phase2/r81_pf_diagnosis"


def compare(split: str, pf_names: list[str], ekf_csv: Path, label: str) -> dict:
    pf = pd.concat([pd.read_csv(TRACK / split / n / "per_video_metrics.csv") for n in pf_names])
    pf = pf.groupby("video", as_index=False)[["IDSW", "HOTA", "IDF1", "GT_dets"]].mean()
    ekf = pd.read_csv(ekf_csv)[["video", "IDSW", "HOTA", "IDF1"]]
    pair = pf.merge(ekf, on="video", suffixes=("_pf", "_ekf"), validate="one_to_one")
    pair["IDSW_gap"] = pair["IDSW_pf"] - pair["IDSW_ekf"]
    pair["IDSW_gap_per_1000_GT"] = 1000 * pair["IDSW_gap"] / pair["GT_dets"]
    pair.sort_values("IDSW_gap", ascending=False).to_csv(OUT / f"{label}_per_video.csv", index=False)
    n = len(pair)
    summary = {"videos": n, "PF_worse": int((pair.IDSW_gap > 0).sum()),
               "equal": int((pair.IDSW_gap == 0).sum()), "PF_better": int((pair.IDSW_gap < 0).sum()),
               "mean_PF_IDSW_sum": float(pair.IDSW_pf.sum()), "EKF_IDSW_sum": float(pair.IDSW_ekf.sum()),
               "total_gap": float(pair.IDSW_gap.sum()),
               "top_4_positive_gap_share_of_net": float(pair.nlargest(4, "IDSW_gap").IDSW_gap.sum() / pair.IDSW_gap.sum()),
               "top_20_positive_gap_share_of_net": float(pair.nlargest(min(20, n), "IDSW_gap").IDSW_gap.sum() / pair.IDSW_gap.sum())}
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ua = compare("DETRAC-all", [f"p2-kc48_r72_pf_dev-A-motion200-s{s}" for s in (0, 1, 2)],
                 TRACK / "DETRAC-all/p2-kc48_r70_pf_dev-A-ekf/per_video_metrics.csv", "ua")
    bdd = compare("BDDCOCOVID-val", [f"r74-bdd-full-motion200-s{s}" for s in (0, 1, 2)],
                  ROOT / "results/phase2/r61_bdd/thuc_thi/cong_b_20261027_01/cham/r61-yolov8n_coco-c25-ekf/per_video_metrics.csv",
                  "bdd")
    (OUT / "summary.json").write_text(json.dumps({"UA": ua, "BDD": bdd}, indent=2), encoding="utf-8")
    print(json.dumps({"UA": ua, "BDD": bdd}, indent=2), flush=True)


if __name__ == "__main__":
    main()
