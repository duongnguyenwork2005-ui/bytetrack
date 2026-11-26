"""Validate R71 metrics, R70 baseline parity and paired video intervals."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import boot_stats
from kc48 import boot_seed
from r70_pf.analyze_metrics import load_run, METRICS

NAMES = ["k2box100", "k2box050", "k2box000"]

def analyze(smoke: bool):
    exp = "kc48_r71_pf_smoke" if smoke else "kc48_r71_pf_dev"
    old = "kc48_r70_pf_smoke" if smoke else "kc48_r70_pf_dev"
    cfg = yaml.safe_load((ROOT / "configs/phase2/experiments" / f"{exp}.yaml").read_text(encoding="utf-8"))
    videos = cfg["videos"] if smoke else json.loads((ROOT / cfg["manifest"]).read_text(encoding="utf-8"))["videos"]
    assert len(videos) == (2 if smoke else 12)
    rows, raw_by_name, parity = [], {}, []
    for name in NAMES:
        raw_by_name[name] = []
        for seed in (0, 1, 2):
            raw, combined, err, _ = load_run(exp, name, seed, videos)
            raw_by_name[name].append(raw)
            rows.append(dict(tracker=name, seed=seed, max_recomposition_error=err, **{m: combined[m] for m in METRICS}))
            if name == "k2box100":
                a = ROOT / "results/phase2" / exp / "_processed/trackers/DETRAC-all" / f"p2-{exp}-A-{name}-s{seed}" / "data"
                b = ROOT / "results/phase2" / old / "_processed/trackers/DETRAC-all" / f"p2-{old}-A-pf500-k1k2-s{seed}" / "data"
                for video in videos:
                    parity.append(dict(seed=seed, video=video, identical=(a / f"{video}.txt").read_bytes() == (b / f"{video}.txt").read_bytes()))
    if not all(p["identical"] for p in parity):
        raise ValueError("R71 baseline differs from R70 K1K2 MOT")
    out = ROOT / "results/phase2" / exp / "analysis"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_seed.csv", index=False)
    summary=[]
    for name in NAMES:
        rr=pd.DataFrame([r for r in rows if r["tracker"]==name])
        pooled=boot_seed.combine_tb_seed(raw_by_name[name],videos,METRICS)
        summary.append(dict(tracker=name,n_seeds=3,**pooled,**{f"{m}_min":rr[m].min() for m in METRICS},**{f"{m}_max":rr[m].max() for m in METRICS}))
    pd.DataFrame(summary).to_csv(out / "summary.csv",index=False)
    if not smoke:
        intervals=[]
        for name in NAMES[1:]:
            result=boot_seed.paired_bootstrap_seed(raw_by_name[name],raw_by_name[NAMES[0]],videos,n_boot=5000,seed=0,metrics=METRICS)
            for m in METRICS:
                intervals.append(dict(tracker=name,reference=NAMES[0],metric=m,**result[m]))
        pd.DataFrame(intervals).to_csv(out / "paired_intervals.csv",index=False)
    validation=dict(experiment=exp,complete_runs=len(rows),verified_video_markers=len(rows)*len(videos),max_recomposition_error=max(r["max_recomposition_error"] for r in rows),baseline_parity=parity)
    (out / "validation.json").write_text(json.dumps(validation,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in validation.items() if k!="baseline_parity"}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--smoke",action="store_true");analyze(p.parse_args().smoke)
