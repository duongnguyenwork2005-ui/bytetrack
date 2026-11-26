"""R92 fixed-video TrackEval helper and per-seed summary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from r64_ngoai import danh_gia_ngoai as r64

SPLIT = "DETRAC-all"
SMOKE = ("MVI_20052", "MVI_20062")
DEV = tuple(json.loads((ROOT / "results/phase2/manifests/detrac_dev.json").read_text())["videos"])
OUT = ROOT / "handoff/logs/review92_pf_modes"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "dev"), required=True)
    args = parser.parse_args()
    videos = SMOKE if args.scope == "smoke" else DEV
    cases = ("base", "cv50") if args.scope == "smoke" else ("cv50",)
    rows = []
    for case in cases:
        for seed in (0, 1, 2):
            name = f"r92-ua-{args.scope}-{case}-s{seed}"
            path = ROOT / "results/trackeval" / SPLIT / name
            if not (path / "combined_metrics.csv").exists():
                r64.cham(SPLIT, name, list(videos))
            comb = pd.read_csv(path / "combined_metrics.csv").iloc[0]
            rows.append({"scope": args.scope, "case": case, "seed": seed,
                         **{k: float(comb[k]) for k in ("HOTA", "IDF1", "MOTA", "FP", "IDSW")}})
            print(rows[-1], flush=True)
    out = OUT / f"{args.scope}_scores.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
