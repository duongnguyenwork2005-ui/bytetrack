"""TrackEval R93 fixed candidate on BDD smoke/full with R74 evaluator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from r64_ngoai import danh_gia_ngoai as r64
from r74_pf_bdd import MANIFEST, SMOKE, SPLIT

OUT = ROOT / "handoff/logs/review93_pf_bdd_transfer"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    videos = SMOKE if args.scope == "smoke" else json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]
    rows = []
    for seed in (0, 1, 2):
        name = f"r93-bdd-{args.scope}-cv50-s{seed}"
        path = ROOT / "results/trackeval" / SPLIT / name
        if not (path / "combined_metrics.csv").exists():
            r64.cham(SPLIT, name, videos)
        row = pd.read_csv(path / "combined_metrics.csv").iloc[0]
        rows.append({"seed": seed, **{key: float(row[key]) for key in ("HOTA", "IDF1", "MOTA", "FP", "IDSW")}})
        print(rows[-1], flush=True)
    target = OUT / f"{args.scope}_scores.json"
    target.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("WROTE", target, flush=True)


if __name__ == "__main__":
    main()
