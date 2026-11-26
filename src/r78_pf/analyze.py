"""R78: audit UA Lost-recovery candidates against immutable R72 and R70 runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import boot_stats
from kc48 import boot_seed

SPLIT = "DETRAC-all"
VIDEOS = json.loads((ROOT / "results/phase2/manifests/detrac_dev.json").read_text(encoding="utf-8"))["videos"]
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")
NAMES = ("baseline", "nofuse", "iou3", "maha3")


def run_name(kind: str, seed: int) -> str:
    if kind == "baseline":
        return f"p2-kc48_r72_pf_dev-A-motion200-s{seed}"
    return f"p2-kc48_r78_pf_dev-A-{kind}-s{seed}"


def load(run: str):
    folder = ROOT / "results/trackeval" / SPLIT / run
    raw = boot_stats.load_raw_fields(folder / "raw_fields.json")
    if set(raw) != set(VIDEOS):
        raise ValueError(f"Incomplete raw fields: {run}")
    score = boot_stats.combine(raw, VIDEOS)
    published = pd.read_csv(folder / "combined_metrics.csv").iloc[0]
    if max(abs(score[m] - float(published[m])) for m in METRICS) > 1e-9:
        raise ValueError(f"COMBINED mismatch: {run}")
    meta = json.loads((folder / "eval_meta.json").read_text(encoding="utf-8"))
    return raw, score, {key: meta[key] for key in ("gt", "cache", "evaluator")}


def main() -> None:
    out = ROOT / "results/phase2/kc48_r78_pf_dev/analysis"
    out.mkdir(parents=True, exist_ok=True)
    raws = {kind: [] for kind in NAMES}
    rows = []
    fingerprint = None
    for kind in NAMES:
        for seed in (0, 1, 2):
            name = run_name(kind, seed)
            raw, score, source = load(name)
            if fingerprint is None:
                fingerprint = source
            elif source != fingerprint:
                raise ValueError(f"source fingerprint differs: {name}")
            if kind != "baseline":
                tracker = ROOT / "data/processed/trackers" / SPLIT / name
                meta = json.loads((tracker / "run_meta.json").read_text(encoding="utf-8"))
                if meta["status"] != "complete" or meta["videos"] != VIDEOS or meta["seed"] != seed:
                    raise ValueError(f"Incomplete run: {name}")
                for video in VIDEOS:
                    marker = json.loads((tracker / "video_meta" / f"{video}.json").read_text(encoding="utf-8"))
                    if not marker["complete"] or marker["input_fingerprint"] != meta["input_fingerprint"]:
                        raise ValueError(f"Marker differs: {name}/{video}")
                cfg = meta["tracker_cfg"]
                expected = {"nofuse": {"lost_no_fuse": True},
                            "iou3": {"lost_recover": "iou", "lost_recover_iou": 0.05,
                                     "lost_recover_max_lost": 3},
                            "maha3": {"lost_recover": "maha", "lost_recover_maha2": 5.991,
                                      "lost_recover_max_lost": 3}}[kind]
                if any(cfg.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Config differs: {name}")
            raws[kind].append(raw)
            rows.append(dict(kind=kind, seed=seed, **{m: score[m] for m in METRICS}))
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("kind", sort=False)[list(METRICS)].mean().reset_index()
    pairs = []
    for kind in NAMES[1:]:
        pair = boot_seed.paired_bootstrap_seed(raws[kind], raws["baseline"], VIDEOS,
                                               n_boot=5000, seed=0, metrics=METRICS)
        pairs.extend(dict(kind=kind, metric=m, **pair[m]) for m in METRICS)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "paired_intervals.csv", index=False)
    (out / "source_fingerprint.json").write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
