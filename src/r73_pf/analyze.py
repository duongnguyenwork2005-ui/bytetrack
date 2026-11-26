"""R73 descriptive external comparison; verify raw-field recomposition and R64 references."""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import boot_stats
from r64_ngoai import danh_gia_ngoai as r64
METRICS=["HOTA","DetA","AssA","IDF1","MOTA","IDSW"]
CASES=[("svmot","yolov8n_coco"),("roundabout","yolov8n_mixed5")]
rows=[]; per_video=[]; validations=[]
for ds,det in CASES:
    t=r64.cau_hinh(ds,False)
    split=t["split"]; videos=t["videos"]
    src=json.loads((ROOT/f"handoff/logs/review73_pf/source_{ds}_full.json").read_text(encoding="utf-8"))
    if src["videos"]!=videos or src["gt_sha256"]!=r64.kiem_gt_cham(t): raise ValueError(f"source GT changed: {ds}")
    if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest for p,digest in src["code_sha256"].items()):
        raise ValueError(f"tracker code changed since preflight: {ds}")
    if any(hashlib.sha256((ROOT/f"configs/phase2/trackers/r72/{kind}.yaml").read_bytes()).hexdigest()!=digest for kind,digest in src["yaml_sha256"].items()):
        raise ValueError(f"tracker YAML changed since preflight: {ds}")
    run_specs=[(f"r64-{det}-c25-cv","cv",None),(f"r64-{det}-c25-ekf","ekf",None)]
    run_specs += [(f"r73-{ds}-{det}-c25-{kind}-s{seed}",kind,seed)
                  for kind in ("motion100","motion200") for seed in (0,1,2)]
    for name,kind,seed in run_specs:
        folder=ROOT/"results/trackeval"/split/name
        raw=boot_stats.load_raw_fields(folder/"raw_fields.json")
        if set(raw)!=set(videos): raise ValueError(f"missing raw video: {name}")
        combined=boot_stats.combine(raw,videos)
        reported=pd.read_csv(folder/"combined_metrics.csv").iloc[0]
        error=max(abs(combined[m]-float(reported[m])) for m in METRICS)
        if error>1e-9: raise ValueError(f"metric mismatch: {name} {error}")
        if kind in ("cv","ekf"):
            old=json.loads((folder/"cong_r64.json").read_text(encoding="utf-8"))
            if old["gt_cham_sha256"]!=src["gt_sha256"]: raise ValueError(f"R64 GT mismatch: {name}")
            ref_meta=json.loads((ROOT/"data/processed/trackers"/split/name/"run_meta.json").read_text(encoding="utf-8"))
            if ref_meta["cache_id"]!=src["cache"] or ref_meta["conf_min"]!=.25 or ref_meta["classes"]!=[1,2,3,5,7] or ref_meta["max_det"]!=300 or ref_meta["fps"]!=t["fps"] or ref_meta["track_buffer_frames"]!=round(1.2*t["fps"]) or ref_meta["elapsed_mode"]!="steps":
                raise ValueError(f"R64 reference configuration mismatch: {name}")
        else:
            tracker=ROOT/"data/processed/trackers"/split/name
            meta=json.loads((tracker/"run_meta.json").read_text(encoding="utf-8"))
            if meta["cache_id"]!=src["cache"] or meta["seed"]!=seed or meta["elapsed_mode"]!="steps" or meta["fps"]!=t["fps"] or meta["conf_min"]!=.25 or meta["classes"]!=[1,2,3,5,7] or meta["max_det"]!=300 or meta["track_buffer_frames"]!=round(1.2*t["fps"]):
                raise ValueError(f"tracker metadata mismatch: {name}")
            if meta["tracker_yaml_sha256_12"]!=src["yaml_sha256"][kind][:12]:
                raise ValueError(f"YAML fingerprint mismatch: {name}")
            for v in videos:
                marker_path=tracker/"video_meta"/f"{v}.json"
                if not marker_path.exists() or not (tracker/"data"/f"{v}.txt").exists():
                    raise ValueError(f"tracking output missing: {name}/{v}")
                marker=json.loads(marker_path.read_text(encoding="utf-8"))
                if not marker.get("complete") or marker["video"]!=v or marker["seed"]!=seed or marker["n_frames"]!=r64.so_frame(t,v):
                    raise ValueError(f"tracking marker incomplete: {name}/{v}")
        rows.append(dict(dataset=ds,tracker=kind,seed=seed if seed is not None else "fixed",run=name,max_recomposition_error=error,**{m:combined[m] for m in METRICS}))
        detail=pd.read_csv(folder/"per_video_metrics.csv")
        if set(detail.video)!=set(videos): raise ValueError(f"per-video mismatch: {name}")
        per_video.extend(dict(dataset=ds,tracker=kind,seed=seed if seed is not None else "fixed",video=d.video,
                              HOTA=d.HOTA,IDF1=d.IDF1,IDSW=d.IDSW) for d in detail.itertuples())
    validations.append(dict(dataset=ds,split=split,GT_verified=True,videos=len(videos),new_runs=6,reference_runs=2))
out=ROOT/"results/phase2/r73_external_pf"
out.mkdir(parents=True,exist_ok=True)
seeds=pd.DataFrame(rows); seeds.to_csv(out/"per_seed.csv",index=False)
pd.DataFrame(per_video).to_csv(out/"per_video.csv",index=False)
summary=seeds.groupby(["dataset","tracker"],sort=False)[METRICS].mean().reset_index()
summary.to_csv(out/"summary.csv",index=False)
paired=[]
for ds,_ in CASES:
    a=seeds[(seeds.dataset==ds)&(seeds.tracker=="motion200")].sort_values("seed")
    b=seeds[(seeds.dataset==ds)&(seeds.tracker=="motion100")].sort_values("seed")
    for m in METRICS:
        paired.append(dict(dataset=ds,metric=m,difference_mean=float((a[m].to_numpy()-b[m].to_numpy()).mean()),
                           seed_differences=[float(x) for x in a[m].to_numpy()-b[m].to_numpy()]))
(out/"paired_descriptive.json").write_text(json.dumps(paired,indent=2)+"\n",encoding="utf-8")
(out/"validation.json").write_text(json.dumps(dict(cases=validations,runs=len(rows),new_run_video_units=6*(9+4),max_recomposition_error=float(seeds.max_recomposition_error.max())),indent=2)+"\n",encoding="utf-8")
print(summary.to_string(index=False),flush=True)
