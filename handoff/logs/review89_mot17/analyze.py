"""Audit R89 and recombine raw TrackEval fields across the three FPS groups."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
import boot_stats
import config
from r89_mot17.prepare import SEQS,SPLIT
from r89_mot17.run_matrix import GROUPS,FPS,FILTERS,CACHE

HERE=Path(__file__).resolve().parent
PROGRESS=json.loads((HERE/"matrix_progress.json").read_text(encoding="utf-8"))
LABELS=json.loads((HERE/"label_audit.json").read_text(encoding="utf-8"))
MANIFEST=json.loads((HERE/"gt_manifest.json").read_text(encoding="utf-8"))
summary={"status":"pass","split":SPLIT,"source_zip_sha256_local":json.loads((Path(r"F:\LABOTARY_data\goc_tai_ve\MOT17-R89\images_manifest.json")).read_text())["sha256_local"],
         "n_unique_sequences":len(SEQS),"n_eval_runs":0,"n_video_runs":0,"confidence":{},"per_video":{}}
assert len(PROGRESS)==30 and all(x["status"]=="done" for x in PROGRESS.values())
for seq in SEQS:
    import hashlib
    gt=(config.PROCESSED_DIR/SPLIT/seq/"gt/gt.txt").read_bytes()
    assert hashlib.sha256(gt).hexdigest()==MANIFEST["sequences"][seq]["gt_sha256"]==LABELS[seq]["sha256_gt"]
    cdir=config.DATA_DIR/"interim"/"det_cache"/CACHE
    # Use det_cache root for portability across workspace config versions.
    import det_cache
    zp=det_cache.CACHE_ROOT/CACHE/f"{seq}.npz"
    with np.load(zp) as z:
        n=LABELS[seq]["frames"]
        assert len(z["frame_idx"])==n and np.array_equal(z["frame_idx"],np.arange(1,n+1))
        assert not z["unreadable"].any() and not z["truncated"].any()
        assert int(z["n_per_frame"].sum())==len(z["data"])
for conf in ("c25","c10"):
    by_filter={};raw_by_filter={};per_video={}
    for filt,(_,seeds) in FILTERS.items():
        runs=[];raws=[]
        for seed in seeds if seeds is not None else [None]:
            raw={}
            for group,videos in GROUPS.items():
                name=f"r89-{group}-{conf}-{filt}"+(f"-s{seed}" if seed is not None else "")
                assert name in PROGRESS[f"{group}/{conf}/{filt}"]["metrics"]
                base=config.PROCESSED_DIR/"trackers"/SPLIT/name
                rm=json.loads((base/"run_meta.json").read_text(encoding="utf-8"))
                assert rm["cache_id"]==CACHE and rm["classes"]==[0]
                assert rm["conf_min"]==({"c25":.25,"c10":.1}[conf])
                assert rm["fps"]==FPS[group] and rm["track_buffer_frames"]==round(1.2*FPS[group])
                assert rm["elapsed_mode"]=="steps"
                for seq in videos:
                    a=np.loadtxt(base/"data"/f"{seq}.txt",delimiter=",",ndmin=2)
                    assert a.ndim==2 and np.isfinite(a).all()
                    assert len(a)==0 or (a[:,0].min()>=1 and a[:,0].max()<=LABELS[seq]["frames"] and (a[:,4:6]>0).all())
                    assert len(a)==len({(int(f),int(i)) for f,i in a[:,:2]})
                    tm=pd.read_csv(base/"timing"/f"{seq}.csv")
                    assert np.array_equal(tm["frame"].to_numpy(),np.arange(1,LABELS[seq]["frames"]+1))
                    summary["n_video_runs"]+=1
                out=config.RESULTS_DIR/"trackeval"/SPLIT/name
                rs=boot_stats.load_raw_fields(out/"raw_fields.json")
                assert sorted(rs)==sorted(videos)
                group_metrics=boot_stats.combine(rs,videos)
                csv=pd.read_csv(out/"combined_metrics.csv").iloc[0]
                for k in ("HOTA","IDF1","MOTA","IDSW"):
                    assert abs(group_metrics[k]-float(csv[k]))<1e-9
                assert not set(raw).intersection(rs)
                raw.update(rs)
                summary["n_eval_runs"]+=1
            assert sorted(raw)==sorted(SEQS)
            m=boot_stats.combine(raw,SEQS)
            runs.append({"seed":seed,"metrics":m});raws.append(raw)
        by_filter[filt]={"per_seed":runs,"mean_metrics":{k:statistics.mean(r["metrics"][k] for r in runs)
                                                         for k in boot_stats.METRICS}}
        raw_by_filter[filt]=boot_stats.average_seeds(raws)
        per_video[filt]={seq:{k:statistics.mean(boot_stats.combine(r,[seq])[k] for r in raws)
                              for k in ("HOTA","IDF1","MOTA","IDSW")} for seq in SEQS}
    summary["confidence"][conf]=by_filter
    summary["per_video"][conf]=per_video
    summary["confidence"][conf]["paired_bootstrap_pf_motion100_minus_pf_default"] = \
        boot_stats.paired_bootstrap(raw_by_filter["pf_motion100"],raw_by_filter["pf_default"],SEQS,
                                    n_boot=5000,seed=89,metrics=["HOTA","IDF1","IDSW"])
    summary["confidence"][conf]["paired_bootstrap_pf_motion100_minus_cv"] = \
        boot_stats.paired_bootstrap(raw_by_filter["pf_motion100"],raw_by_filter["cv"],SEQS,
                                    n_boot=5000,seed=90,metrics=["HOTA","IDF1","IDSW"])
assert summary["n_eval_runs"]==54 and summary["n_video_runs"]==126
(HERE/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps({"status":summary["status"],"n_eval_runs":summary["n_eval_runs"],
                  "n_video_runs":summary["n_video_runs"],
                  "c25":{k:v["mean_metrics"] for k,v in summary["confidence"]["c25"].items() if "mean_metrics" in v},
                  "c10":{k:v["mean_metrics"] for k,v in summary["confidence"]["c10"].items() if "mean_metrics" in v}},indent=2))
