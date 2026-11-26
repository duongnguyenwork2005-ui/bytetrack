"""Evaluate R74 BDD PF runs with the R61 mot_crowd rule."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import boot_stats
import run_trackeval


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["smoke", "full"], required=True)
    args = ap.parse_args()
    manifest = json.loads((ROOT / "results/phase2/manifests/bdd100k_transfer_cocovid_val.json").read_text(encoding="utf-8"))
    videos = (["b1e1a7b8-a7426a97", "b22f385b-5d7e5202", "b23b2649-8349d2a1"]
              if args.scope == "smoke" else manifest["videos"])
    split = "BDDCOCOVID-val"
    for scale in (100, 200):
        for seed in (0, 1, 2):
            name = f"r74-bdd-{args.scope}-motion{scale}-s{seed}"
            out = ROOT / "results/trackeval" / split / name
            if (out / "raw_fields.json").exists():
                raw = boot_stats.load_raw_fields(out / "raw_fields.json")
                if set(raw) == set(videos):
                    print(f"reuse {name}", flush=True)
                    continue
            print(f"evaluate {name}", flush=True)
            results, _ = run_trackeval.run(split, name, videos, "mot_crowd")
            df = run_trackeval.flatten_results(results, name)
            combined = df[df.video == "COMBINED_SEQ"]
            per_video = df[df.video != "COMBINED_SEQ"].sort_values("video")
            if len(combined) != 1 or len(per_video) != len(videos):
                raise ValueError(f"incomplete evaluation: {name}")
            out.mkdir(parents=True, exist_ok=True)
            combined.to_csv(out / "combined_metrics.csv", index=False)
            per_video.to_csv(out / "per_video_metrics.csv", index=False)
            raw = boot_stats.save_raw_fields(results, name, out / "raw_fields.json")
            score = boot_stats.combine(raw, videos)
            if abs(score["HOTA"] - float(combined.iloc[0].HOTA)) > 1e-9:
                raise ValueError(f"COMBINED mismatch: {name}")
            print(f"{name}: HOTA {score['HOTA']:.6f}, IDSW {score['IDSW']:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
