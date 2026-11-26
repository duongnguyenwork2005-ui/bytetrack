"""Format verified R66 scores for the experiment record (no tracking)."""
from pathlib import Path
import json
import pandas as pd
from sher_metric import combine

ROOT=Path(__file__).resolve().parents[2]
LOG=ROOT/"handoff/logs/external_eval/r66"
def summarize(df):
    parts=[]
    for (d,c,l),group in df.groupby(["detector","condition","filter"],sort=False):
        assert len(group)==(3 if l=="pf500" else 1)
        row={"detector":d,"condition":c,"filter":l,"runs":len(group)}
        for k in ("HOTA","DetA","AssA","IDF1","MOTA","IDSW","FP","FN"):
            vals=group[k]*(100 if k not in ("IDSW","FP","FN") else 1)
            row[k+"_mean"]=vals.mean()
            row[k+"_min"],row[k+"_max"]=vals.min(),vals.max()
        parts.append(row)
    return pd.DataFrame(parts)

def results(dataset):
    is_sher=dataset=="sherbrooke"
    folder=LOG/dataset/("v2/full" if is_sher else "full")
    split="Sherbrooke-cars-f2754-3754-roi-v1" if is_sher else "HELMET-test36-motorcycle-v1"
    df=pd.read_csv(folder/("scores.csv" if is_sher else "combined_tung_run.csv"))
    df=df.rename(columns={"cond":"condition","loc":"filter"})
    assert len(df)==36 and not df.duplicated(["detector","condition","filter","seed"]).any()
    sm=summarize(df)
    groups=[]
    clip_tables=[]
    membership = ({'sherbrooke':['sherbrooke']} if is_sher else {})
    if not is_sher:
        for entry in json.loads((LOG/dataset/"manifest_test36.json").read_text(encoding="utf-8"))["videos"]:
            membership.setdefault(entry["group"],[]).append(entry["video_id"])
    for _,row in df.iterrows():
        run=row.get("run")
        if is_sher:
            run=f"r66v2-{row.detector}-{row.condition}-{row['filter']}"
            if pd.notna(row.seed): run+=f"-s{int(row.seed)}"
        path=ROOT/"results/trackeval"/split/run
        raw=json.loads((path/"raw_fields.json").read_text(encoding="utf-8"))
        assert set(raw)==set().union(*membership.values())
        per=pd.read_csv(path/"per_video_metrics.csv")
        assert set(per.video)==set(raw) and len(per)==len(raw)
        clip_tables.append(per.assign(run=run,detector=row.detector,condition=row.condition,
                                filter=row["filter"],seed=row.seed))
        for group,clips in membership.items():
            groups.append(dict(run=run,group=group,**combine(raw,clips)))
    sm.to_csv(folder/"filter_summary.csv",index=False)
    pd.DataFrame(groups).to_csv(folder/"per_scene.csv",index=False)
    pd.concat(clip_tables,ignore_index=True).to_csv(folder/"per_clip.csv",index=False)
    pivot=sm.pivot(index=["detector","condition"],columns="filter",values="HOTA_mean")
    for loc in ("ekf","ukf","pf500"):
        pivot["delta_"+loc]=pivot[loc]-pivot.cv
    print(pivot.round(3).to_string())
    print("ran",len(df),"; scene groups",len(membership))
if __name__=="__main__":
    import sys
    results(sys.argv[1])

