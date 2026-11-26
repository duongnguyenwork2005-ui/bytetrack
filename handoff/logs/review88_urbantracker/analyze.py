"""Audit and summarize R88 comparisons without pooling two scenes."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
import config

HERE=Path(__file__).resolve().parent
SPLIT="UrbanTracker-R88-car"
PROGRESS=json.loads((HERE/"matrix_progress.json").read_text(encoding="utf-8"))
GT=json.loads((HERE/"gt_audit.json").read_text(encoding="utf-8"))
summary={"status":"pass","scenes":{},"run_count":0,"notes":[]}
for scene in ("rene","stmarc"):
    group={}
    for conf in ("c25","c10"):
        rows={}
        for filt in ("cv","ekf","ukf","pf_default","pf_motion100"):
            cell=PROGRESS[f"{scene}/{conf}/{filt}"]
            assert cell["status"]=="done"
            metrics=[]
            for name,m in cell["metrics"].items():
                base=config.PROCESSED_DIR/"trackers"/SPLIT/name
                run=json.loads((base/"run_meta.json").read_text(encoding="utf-8"))
                assert run["cache_id"]==("r88_urbantracker_rene_roi_yolov8n_coco_c005" if scene=="rene" else "r88_urbantracker_yolov8n_coco_c005")
                assert run["classes"]==[2] and run["conf_min"]==({"c25":.25,"c10":.1}[conf])
                assert run["track_buffer_frames"]==36 and run["elapsed_mode"]=="steps"
                assert abs(run["fps"]-GT[scene]["fps"])<1e-9
                assert run["status"]=="complete" if "status" in run else True
                mot=base/"data"/f"{scene}.txt"
                a=np.loadtxt(mot,delimiter=",",ndmin=2) if mot.stat().st_size else np.empty((0,10))
                assert a.ndim==2 and len(a)>0 and np.isfinite(a).all()
                assert a[:,0].min()>=1 and a[:,0].max()<=1000 and (a[:,4:6]>0).all()
                assert len(a)==len({(int(f),int(i)) for f,i in a[:,:2]})
                tm=pd.read_csv(base/"timing"/f"{scene}.csv")
                assert np.array_equal(tm["frame"].to_numpy(),np.arange(1,1001))
                out=config.RESULTS_DIR/"trackeval"/SPLIT/name
                assert (out/"combined_metrics.csv").exists() and (out/"raw_fields.json").exists()
                metrics.append(m);summary["run_count"]+=1
            rows[filt]={k:statistics.mean(r[k] for r in metrics) for k in ("HOTA","IDF1","MOTA","IDSW")}
            rows[filt]["seeds"]=len(metrics)
        group[conf]=rows
    summary["scenes"][scene]=group
assert summary["run_count"]==36
summary["notes"]=["Each scene is one sequence, so no independence CI or pooled headline metric.",
                  "Rene uses published ROI mask on detection centers; St-Marc has no ROI mask.",
                  "Car-only GT; other source classes marked distractor. Results exploratory after detector preflight and one CV inspection."]
(HERE/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps({"status":summary["status"],"run_count":summary["run_count"],
                  "rene_c10":summary["scenes"]["rene"]["c10"],
                  "stmarc_c25":summary["scenes"]["stmarc"]["c25"]},indent=2))
