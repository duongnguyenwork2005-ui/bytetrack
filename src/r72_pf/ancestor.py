"""Summarize one-step resampling genealogy; no tracker mutation."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
exp=ROOT/"results/phase2/kc48_r72_pf_dev"
runs=pd.read_csv(exp/"runs.csv")
videos=json.loads((exp/"videos.json").read_text(encoding="utf-8"))
rows=[]
for run in runs.itertuples():
    folder=exp/"_processed/trackers/DETRAC-all"/run.dir/"pf_diag"
    ds=[]
    for video in videos:
        d=pd.read_csv(folder/f"{video}.csv")
        r=d[(d.kind=="update") & (d.resampled==True) & d.n_ancestors.notna()]
        if len(r): ds.append(r[["n_ancestors","n_unique","ess_ratio"]])
    all_rows=pd.concat(ds,ignore_index=True)
    rows.append(dict(tracker=run.tracker,seed=int(run.seed),n_resamples=len(all_rows),ancestor_median=all_rows.n_ancestors.median(),ancestor_p10=all_rows.n_ancestors.quantile(.1),ancestor_lt100=int((all_rows.n_ancestors<100).sum()),jitter_unique_median=all_rows.n_unique.median(),ess_median=all_rows.ess_ratio.median()))
out=exp/"analysis"
out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out/"ancestors_by_seed.csv",index=False)
pd.DataFrame(rows).groupby("tracker").agg(n_resamples_mean=("n_resamples","mean"),ancestor_median_mean=("ancestor_median","mean"),ancestor_p10_mean=("ancestor_p10","mean"),ancestor_lt100_mean=("ancestor_lt100","mean"),jitter_unique_median_mean=("jitter_unique_median","mean"),ess_median_mean=("ess_median","mean")).to_csv(out/"ancestor_summary.csv")
print(pd.read_csv(out/"ancestor_summary.csv").to_string(index=False))
