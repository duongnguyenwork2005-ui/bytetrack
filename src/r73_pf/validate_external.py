"""R73 external validation with frozen R64 GT and detection caches."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import config
from r64_ngoai import danh_gia_ngoai as r64

CHOICES={"svmot":("yolov8n_coco",30.0),"roundabout":("yolov8n_mixed5",15.0)}
NAMES={"motion100":"configs/phase2/trackers/r72/motion100.yaml","motion200":"configs/phase2/trackers/r72/motion200.yaml"}

def run(ds:str,smoke:bool,stage:str):
    det,fps=CHOICES[ds]
    t=r64.cau_hinh(ds,smoke)
    split=t["split"]
    cache=r64.ten_cache(ds,det,smoke)
    log=ROOT/"handoff/logs/review73_pf"
    log.mkdir(parents=True,exist_ok=True)
    # The R64 contract verifies marker, content and provenance, including each NPZ.
    checked={v:r64.kiem_don_vi_cache(ds,det,v,smoke) for v in t["videos"]}
    gt=r64.kiem_gt_cham(t)
    source=dict(dataset=ds,smoke=smoke,split=split,cache=cache,videos=t["videos"],cache_checks=checked,gt_sha256=gt,
                yaml_sha256={n:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for n,p in NAMES.items()},
                code_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ("src/pf_ctrv.py","src/tracker_ctrv.py","src/run_tracking.py")})
    source_path=log/f"source_{ds}_{'smoke' if smoke else 'full'}.json"
    if source_path.exists() and json.loads(source_path.read_text(encoding="utf-8")) != source:
        raise ValueError("Source fingerprint changed")
    source_path.write_text(json.dumps(source,indent=2,default=str)+"\n",encoding="utf-8")
    for name,yaml in NAMES.items():
        prefix=f"r73-{ds}-{det}-c25-{name}"+("-smoke" if smoke else "")
        if stage=="track":
            cmd=[sys.executable,"-B",str(ROOT/"src/run_tracking.py"),"--cache-id",cache,"--split-name",split,
                 "--tracker-yaml",yaml,"--tracker-name",prefix,"--videos",*t["videos"],"--conf-min","0.25",
                 "--classes","1","2","3","5","7","--max-det","300","--fps",str(fps),
                 "--track-buffer-sec","1.2","--elapsed-mode","steps","--seeds","0","1","2","--skip-existing"]
            print("RUN",prefix,flush=True)
            subprocess.run(cmd,cwd=ROOT,check=True)
        else:
            for seed in (0,1,2):
                name_seed=f"{prefix}-s{seed}"
                out=config.RESULTS_DIR/"trackeval"/split/name_seed
                if out.exists() and (out/"combined_metrics.csv").exists() and (out/"raw_fields.json").exists():
                    print("REUSE",name_seed,flush=True)
                    continue
                score=r64.cham(split,name_seed,t["videos"])
                print("EVAL",name_seed,score,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=CHOICES,required=True)
    p.add_argument("--stage",choices=("track","eval"),required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    run(a.dataset,a.smoke,a.stage)
