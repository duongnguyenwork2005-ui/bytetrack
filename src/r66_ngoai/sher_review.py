"""Validate the smoke review package and issue the Sherbrooke full-run decision."""
from pathlib import Path
import csv
import json
from sher_matrix import LOCK, ROOT, METRICS, read, write, sha, require, matrix, name_for, paths, base
import pandas as pd

def review():
    folder=ROOT/"smoke"
    coverage=read(folder/"coverage.json")
    require(coverage["lock"]==sha(LOCK) and coverage["verified_runs"]==36 and
            coverage["scores"]==sha(folder/"scores.csv"),"coverage mismatch")
    checks=read(folder/"fault_checks.json")
    require(checks["n"]>=19 and "seqinfo changed" in checks["passed"] and
            "interrupted sibling seed" in checks["passed"],"missing fixture")
    for f in ("ngan_sach.jsonl","budget_v2.jsonl"):
        lines=[json.loads(line) for line in (folder/f).read_text(encoding="utf-8").splitlines()]
        require(lines and all(line.get("dat",line.get("ok")) and line.get("het_han",line.get("deadline")) for line in lines),
                "missing budget or failed preflight")
    historic=read(ROOT.parent/"smoke/04_cham_lai_b3_b4_b6.json")
    require(historic["smoke_runs"]==36 and historic["repeat_runs"]==2 and historic["max_rescore_error"]<1e-9
            and historic["max_recompose_error"]<1e-9,"historical rescore discrepancy")
    df=pd.read_csv(folder/"scores.csv")
    t=base.cau_hinh("sherbrooke",True)
    seals={}
    for d,c,l,s in matrix():
        name=name_for(True,d,c,l,s)
        path=paths(t,name)[1]/"reviewed.json"
        seals[name]=sha(path)
        row=df[(df.detector==d)&(df.condition==c)&(df["filter"]==l)&(df.seed.isna() if s is None else (df.seed==s))]
        require(len(row)==1 and all(abs(float(row[k].iloc[0])-read(path)["metrics"][k])<1e-9 for k in METRICS),
                f"{name}: CSV differs from sealed score")
    reps={}
    for seed in (0,1):
        args=("yolov8n_mixed5","c25","pf500",seed)
        artifacts=[]
        for rep in (False,True):
            name=name_for(True,*args,rep)
            path=paths(t,name)[1]/"reviewed.json"
            artifacts.append(read(path)["artifacts"])
            if rep: reps[name]=sha(path)
        require(artifacts[0]["score"]==artifacts[1]["score"] and
                all(artifacts[0]["track"][k]==artifacts[1]["track"][k] for k in
                    ("data/sherbrooke.txt","data_tho/sherbrooke.txt")),"repeat mismatch")
    geometry=ROOT/"review/02_geometry.md"
    require(geometry.exists(),"geometry review missing")
    write(ROOT/"review_smoke.json",{"decision":"PASS","lock":sha(LOCK),"coverage_sha256":sha(folder/"coverage.json"),
        "fault_checks_sha256":sha(folder/"fault_checks.json"),"smoke_seals":seals,"repeat_seals":reps,
        "geometry_review_sha256":sha(geometry),"historical_rescore_sha256":sha(ROOT.parent/"smoke/04_cham_lai_b3_b4_b6.json"),
        "scope":"Sherbrooke cars ROI v2 only. Old full was interrupted and excluded."})
    print("smoke PASS: 36 sealed, 2 repeat checks and 19 fault fixtures")

if __name__=="__main__": review()

