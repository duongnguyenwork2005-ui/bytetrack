"""Run shared-cache MOT17 comparison, grouped by source FPS."""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import config
from r64_ngoai import danh_gia_ngoai as r64
from r89_mot17.prepare import SEQS, SPLIT
from r89_mot17.cache import CACHE

EVIDENCE=ROOT/"handoff/logs/review89_mot17"
LABELS=json.loads((EVIDENCE/"label_audit.json").read_text(encoding="utf-8"))
GROUPS={"f30":[s for s in SEQS if LABELS[s]["fps"]==30],
        "f14":[s for s in SEQS if LABELS[s]["fps"]==14],
        "f25":[s for s in SEQS if LABELS[s]["fps"]==25]}
FPS={"f30":30,"f14":14,"f25":25}
FILTERS={"cv":("bytetrack.yaml",None),"ekf":("configs/bytetrack_ekf_ctrv.yaml",None),
         "ukf":("configs/bytetrack_ukf_ctrv.yaml",None),
         "pf_default":("configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml",[0,1,2]),
         "pf_motion100":("configs/phase2/trackers/r72/motion100.yaml",[0,1,2])}
PROGRESS=EVIDENCE/"matrix_progress.json"


def cell(group,conf,filt):
    videos=GROUPS[group]
    assert videos
    yaml,seeds=FILTERS[filt]
    name=f"r89-{group}-{conf}-{filt}"
    cmd=[sys.executable,"-B",str(ROOT/"src/run_tracking.py"),"--cache-id",CACHE,"--split-name",SPLIT,
         "--tracker-yaml",yaml,"--tracker-name",name,"--videos",*videos,"--conf-min",
         {"c25":"0.25","c10":"0.10"}[conf],"--classes","0","--max-det","300",
         "--fps",str(FPS[group]),"--track-buffer-sec","1.2","--elapsed-mode","steps","--skip-existing"]
    if seeds is not None:cmd += ["--seeds",*map(str,seeds)]
    subprocess.run(cmd,cwd=ROOT,check=True)
    metrics={}
    import pandas as pd
    for seed in seeds if seeds is not None else [None]:
        n=name+(f"-s{seed}" if seed is not None else "")
        p=config.RESULTS_DIR/"trackeval"/SPLIT/n/"combined_metrics.csv"
        if p.exists():
            row=pd.read_csv(p).iloc[0]
            metrics[n]={k:float(row[k]) for k in ("HOTA","IDF1","MOTA","IDSW")}
        else:metrics[n]=r64.cham(SPLIT,n,videos)
        print("EVAL",n,metrics[n],flush=True)
    return metrics


def main():
    assert sum(map(len,GROUPS.values()))==7
    state=json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else {}
    for group in GROUPS:
        for conf in ("c25","c10"):
            for filt in FILTERS:
                key=f"{group}/{conf}/{filt}"
                if state.get(key,{}).get("status")=="done":
                    print("REUSE",key,flush=True);continue
                print("START",key,datetime.now().isoformat(),flush=True)
                try:
                    metrics=cell(group,conf,filt)
                    state[key]={"status":"done","metrics":metrics,"finished":datetime.now().isoformat()}
                except Exception as e:
                    state[key]={"status":"failed","error":repr(e),"traceback":traceback.format_exc(),
                                "finished":datetime.now().isoformat()}
                    PROGRESS.write_text(json.dumps(state,indent=2),encoding="utf-8")
                    raise
                PROGRESS.write_text(json.dumps(state,indent=2),encoding="utf-8")
                print("DONE",key,flush=True)
    print("ALL DONE",len(state),flush=True)


if __name__=="__main__":main()
