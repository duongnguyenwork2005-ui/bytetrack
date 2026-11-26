"""Reviewed Sherbrooke runner. Examples: lock, smoke --minutes 90, full --minutes 90.
Each call has a 120 minute cap with 30 minutes reserved for postchecks.
A stopped matrix keeps sealed units for the next invocation.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib
import io
import itertools
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import danh_gia_r66 as base
from sher_metric import combine, roi_contains

import run_fingerprint as fp
import run_tracking as rt

DS, V = "sherbrooke", "sherbrooke"
DETS = ("yolov8n_coco", "yolov8n_mixed5", "yolo11n_mixed5")
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "FP", "FN")
LOCK = base.SHER_LOCK
ROOT = LOCK.parent
sha, write = base.sha256_file, base.ghi_json

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def roilines(raw, mask):
    lines = raw.splitlines(keepends=True)
    if not lines:
        return b""
    boxes = np.array([[float(x) for x in row.split(b",")[2:6]] for row in lines])
    keep = roi_contains(mask, *boxes.T)
    return b"".join(row for row, inside in zip(lines, keep) if inside)

def source_lock():
    from ultralytics import YOLO
    from configparser import ConfigParser
    ds = {}
    for smoke in (True, False):
        t = base.cau_hinh(DS, smoke)
        folder = base.config.PROCESSED_DIR / t["split"] / V
        a, _ = base.doc_gt_sherbrooke(t)
        keep = roi_contains(base.mask_roi(t), *a[:, 2:6].T) & (a[:, 0] <= t["n_frame"])
        a = a[keep]
        a = a[np.lexsort((a[:, 1], a[:, 0]))]
        require(np.array_equal(a, np.loadtxt(folder / "gt/gt.txt", delimiter=",")), "source GT disagreement")
        ini = ConfigParser()
        ini.read(folder / "seqinfo.ini", encoding="utf-8")
        seqmap = base.config.PROCESSED_DIR / "seqmaps" / (t["split"] + ".txt")
        require(ini["Sequence"].getint("seqLength") == t["n_frame"] and ini["Sequence"].getint("frameRate") == 30
                and seqmap.read_text(encoding="utf-8").split() == ["name", V], "source timeline")
        ds[t["split"]] = {"gt":sha(folder/"gt/gt.txt"), "ini":sha(folder/"seqinfo.ini"), "map":sha(seqmap),
                          "images":base.nguon_don_vi(t,V)}
    dets = {}
    for det in DETS:
        spec = base.r64.detector_r54(det)
        weights = base.GOC / spec["checkpoint"]
        require(sha(weights) == spec["sha256"], "checkpoint changed")
        names = dict(YOLO(str(weights)).names)
        mapping = base.r64.bang_anh_xa_coco(names, spec["loai_checkpoint"])
        dets[det] = {"weights":sha(weights), "names":{str(k):v for k,v in names.items()}, "mapping":mapping}
    evaluators = ["trackeval.eval","trackeval.datasets.mot_challenge_2d_box","trackeval.datasets._base_dataset",
                  "trackeval.metrics.hota","trackeval.metrics.clear","trackeval.metrics.identity","trackeval.metrics.count"]
    lock = {"protocol":"Sherbrooke R66 v2", "geometry":"continuous xyxy difference, floor in-frame center mask=1",
            "inputs":base.giao_thuc(DS), "splits":ds, "detectors":dets, "tracker":fp.code_fingerprint(),
            "cache_code":{p:sha(base.GOC/p) for p in base.MA_THUC_DUNG},
            "driver":{p:sha(Path(__file__).with_name(p)) for p in ("sher_matrix.py","sher_metric.py")},
            "evaluator":{name:sha(importlib.import_module(name).__file__) for name in evaluators},
            "configs":{loc:sha(rt.resolve_tracker_yaml(yaml)) for loc,(yaml,_) in base.r64.BO_LOC.items()}}
    return json.loads(json.dumps(lock))

def budget(t, deadline, label, estimate, size):
    remaining = deadline - time.time()
    free = {d:shutil.disk_usage(d).free for d in ("D:/","F:/")}
    ok = remaining >= estimate + 1800 and free["D:/"] >= size + 5 * 1024**3 and free["F:/"] >= 5 * 1024**3
    path = base.log_dir(DS,t["n_frame"]==128)/"budget_v2.jsonl"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(dict(unit=label,at=base.r64.bay_gio(),seconds=estimate,free=free,
                                expected_bytes=size,deadline=deadline,ok=ok))+"\n")
    if not ok:
        raise base.HetNganSach(label)

def paths(t, name):
    track = base.config.PROCESSED_DIR/"trackers"/t["split"]/name
    result = base.config.RESULTS_DIR/"trackeval"/t["split"]/name
    return track, result

def filehashes(track,result):
    track_files = ["run_meta.json","summary.csv","video_meta/sherbrooke.json","data/sherbrooke.txt",
                   "data_tho/sherbrooke.txt","data_cls/sherbrooke.csv","timing/sherbrooke.csv"]
    score_files = ["combined_metrics.csv","per_video_metrics.csv","raw_fields.json"]
    return {"track":{p:sha(track/p) for p in track_files}, "score":{p:sha(result/p) for p in score_files}}

def cfg_inputs(t,det,cond,loc,seed,cache_id):
    yaml = base.r64.BO_LOC[loc][0]
    cfg = rt.cfg_to_dict(rt.load_tracker_cfg(yaml,{"track_buffer":36}))
    code = fp.code_fingerprint()
    yhash = sha(rt.resolve_tracker_yaml(yaml))[:12]
    spec = {"tracker_cfg":cfg,"tracker_yaml_sha256_12":yhash,"overrides":{"track_buffer":36},
            "conf_min":base.r64.DIEU_KIEN[cond],"classes":[2],"class_map":None,"max_det":300,
            "elapsed_mode":"steps","fps":30.0,"seed":seed,"assoc_log":False,"pf_diag":False,
            "cache":fp.cache_fingerprint(cache_id,[V]),"sequences":[V],"code":code}
    expected = {k:v for k,v in spec.items() if k not in ("cache","sequences","assoc_log","pf_diag")}
    expected.update({"cache_id":cache_id,"tracker_yaml":yaml,"track_buffer_frames":36,
                     "dt_nominal_s":1/30,"input_fingerprint":fp.sha256_obj(spec),
                     "tracker_backend":code["backend"]["backend"],"status":"complete","videos":[V]})
    return expected

def inspect(t,det,cond,loc,seed,name,cache_id):
    track,result = paths(t,name)
    expected = cfg_inputs(t,det,cond,loc,seed,cache_id)
    rm,vm = read(track/"run_meta.json"),read(track/"video_meta/sherbrooke.json")
    require(all(rm.get(k)==v for k,v in expected.items()), f"{name}: run config/backend/fingerprint")
    require(vm["complete"] and vm["input_fingerprint"]==expected["input_fingerprint"] and vm["seed"]==seed
            and vm["video"]==V and vm["n_frames"]==t["n_frame"], "video completion")
    base.r64.hau_kiem_tracking(t,name,36,base.r64.max_time_lost_thuc(base.r64.BO_LOC[loc][0],36))
    raw=(track/"data_tho/sherbrooke.txt").read_bytes()
    require(roilines(raw,base.mask_roi(t))==(track/"data/sherbrooke.txt").read_bytes(), "ROI mismatch")
    require(vm["n_boxes"]==len(raw.splitlines()), "raw track count")
    require({p.stem for p in (track/"data").glob("*.txt")} == {V}, "unexpected sequence in tracking")
    fields=read(result/"raw_fields.json")
    require(set(fields)=={V}, "raw sequence set")
    per=pd.read_csv(result/"per_video_metrics.csv")
    scores=pd.read_csv(result/"combined_metrics.csv")
    require(list(per.video)==[V] and list(scores.video)==["COMBINED_SEQ"], "metric sequence set")
    recomputed=combine(fields,[V])
    values={k:float(scores[k].iloc[0]) for k in METRICS}
    require(all(math.isfinite(values[k]) and abs(values[k]-recomputed[k])<1e-9 and
                abs(values[k]-per[k].iloc[0])<1e-9 for k in METRICS), "score recomposition")
    return values

def matrix():
    return [(d,c,l,s) for d,c,l in itertools.product(DETS,("c25","c10"),base.r64.BO_LOC)
            for s in (base.r64.BO_LOC[l][1] or [None])]

def name_for(smoke,det,cond,loc,seed,repeat=False):
    return "r66v2"+("smoke" if smoke else "")+("-rep" if repeat else "")+f"-{det}-{cond}-{loc}"+(
        f"-s{seed}" if seed is not None else "")

def unit(t,smoke,deadline,d,c,l,s,repeat=False):
    name=name_for(smoke,d,c,l,s,repeat)
    cid=base.ten_cache(DS,d,smoke)
    cache=base.kiem_don_vi_cache(DS,d,V,smoke)
    track,result=paths(t,name)
    marker=result/"reviewed.json"
    if marker.exists():
        m=read(marker)
        require(m["lock"]==sha(LOCK) and m["cache"]==cache and m["artifacts"]==filehashes(track,result), "seal changed")
        values=inspect(t,d,c,l,s,name,cid)
        require(m["metrics"]==values, "sealed scores changed")
        return values
    require(not track.exists() and not result.exists(), f"{name} incomplete/orphan; preserve before recovery")
    budget(t,deadline,name,60+t["n_frame"]*(0.12 if l=="pf500" else 0.035), t["n_frame"]*50000)
    cmd=["run_tracking","--cache-id",cid,"--split-name",t["split"],"--tracker-yaml",base.r64.BO_LOC[l][0],
         "--tracker-name",name.rsplit("-s",1)[0] if s is not None else name,"--videos",V,
         "--conf-min",str(base.r64.DIEU_KIEN[c]),"--classes","2","--max-det","300","--fps","30.0",
         "--track-buffer-sec","1.2","--elapsed-mode","steps"]
    if s is not None:
        cmd+=["--seeds",str(s)]
    previous=sys.argv
    try:
        sys.argv=cmd
        require(rt.main()==0,"tracker failed")
    finally:
        sys.argv=previous
    base.r64.hau_kiem_tracking(t,name,36,base.r64.max_time_lost_thuc(base.r64.BO_LOC[l][0],36))
    base.ap_roi_output(t,track)
    base.r64.cham(t["split"],name,[V])
    values=inspect(t,d,c,l,s,name,cid)
    write(marker,{"lock":sha(LOCK),"cache":cache,"artifacts":filehashes(track,result),"metrics":values,
                  "finished":base.r64.bay_gio()})
    return values

def parity(smoke,det):
    """Older snapshots with uncertain generation code are only compared as historical observations."""
    cid=base.ten_cache(DS,det,smoke)
    old=f"r66{'smoke' if smoke else ''}_sherbrooke_00c7363e_{det}_c0.05"
    p=base.det_cache.CACHE_ROOT
    if not (p/old/f"{V}.npz").is_file():
        return {"status":"historical cache unavailable"}
    with np.load(p/cid/f"{V}.npz") as new,np.load(p/old/f"{V}.npz") as oldz:
        return {k:np.array_equal(new[k],oldz[k]) for k in new.files}

def run(phase,deadline):
    smoke=phase=="smoke"
    require(source_lock()==read(LOCK),"source/code differs from locked manifest")
    if not smoke:
        review=read(ROOT/"review_smoke.json")
        smoke_log = ROOT/"smoke"
        require(review["decision"]=="PASS" and review["lock"]==sha(LOCK) and
                review["coverage_sha256"]==sha(smoke_log/"coverage.json") and
                read(smoke_log/"coverage.json")["scores"]==sha(smoke_log/"scores.csv") and
                review["fault_checks_sha256"]==sha(smoke_log/"fault_checks.json"),"full requires reviewed smoke")
        st=base.cau_hinh(DS,True)
        for args in matrix():
            run_name=name_for(True,*args)
            seal=paths(st,run_name)[1]/"reviewed.json"
            require(review["smoke_seals"][run_name]==sha(seal),"smoke seal changed")
            unit(st,True,deadline,*args)
    t=base.cau_hinh(DS,smoke)
    log=base.log_dir(DS,smoke)
    log.mkdir(parents=True,exist_ok=True)
    start=time.time()
    due=dt.datetime.fromtimestamp(deadline).astimezone().isoformat()
    for d in DETS:
        budget(t,deadline,"cache/"+d,110+t["n_frame"]/8,t["n_frame"]*50000)
        base.buoc_cache(DS,d,smoke,due)
    write(log/"cache_parity_historical.json",{d:parity(smoke,d) for d in DETS})
    rows=[]
    for d,c,l,s in matrix():
        values=unit(t,smoke,deadline,d,c,l,s)
        rows.append(dict(detector=d,condition=c,filter=l,seed=s,**values))
        print(f"{len(rows)}/36 {d} {c} {l} {s} HOTA {100*values['HOTA']:.2f}",flush=True)
    if smoke:
        sig=[]
        for seed in (0,1):
            d,c,l="yolov8n_mixed5","c25","pf500"
            unit(t,True,deadline,d,c,l,seed,True)
            markers=[read(paths(t,name_for(True,d,c,l,seed,rep))[1]/"reviewed.json") for rep in (False,True)]
            keys=["data/sherbrooke.txt","data_tho/sherbrooke.txt"]
            require(all(markers[0]["artifacts"]["track"][k]==markers[1]["artifacts"]["track"][k] for k in keys)
                    and markers[0]["artifacts"]["score"]==markers[1]["artifacts"]["score"], "repeat mismatch")
            sig.append(markers[0]["artifacts"]["track"][keys[1]])
        require(sig[0]!=sig[1],"different seeds identical")
    require(source_lock()==read(LOCK),"post-run source drift")
    pd.DataFrame(rows).to_csv(log/"scores.csv",index=False)
    write(log/"coverage.json",{"phase":phase,"verified_runs":len(rows),"lock":sha(LOCK),
                              "scores":sha(log/"scores.csv"),"runtime_seconds":time.time()-start})
    print(f"verified {len(rows)}/36",flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=["lock","smoke","full"])
    ap.add_argument("--minutes",type=int,default=90)
    ap.add_argument("--worker-deadline",type=float,help=argparse.SUPPRESS)
    a=ap.parse_args()
    if a.stage=="lock":
        lock=source_lock()
        if LOCK.exists():
            require(read(LOCK)==lock,"existing source lock differs")
        else:
            write(LOCK,lock)
        print("source lock",sha(LOCK))
        return 0
    if a.worker_deadline is not None:
        run(a.stage,a.worker_deadline)
        return 0
    require(30<a.minutes<=120,"deadline must leave 30 minutes for audit")
    deadline=time.time()+a.minutes*60
    log=ROOT/a.stage
    log.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,"-B",__file__,a.stage,"--worker-deadline",str(deadline)]
    out=log/("job_"+dt.datetime.now().strftime("%H%M%S_%f")+".log")
    print(out,flush=True)
    try:
        with out.open("w",encoding="utf-8") as stream:
            result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=(a.minutes-30)*60)
        rc=result.returncode
    except subprocess.TimeoutExpired:
        rc=10
    if rc:
        t=base.cau_hinh(DS,a.stage=="smoke")
        missing=[name_for(a.stage=="smoke",*args) for args in matrix()
                 if not (paths(t,name_for(a.stage=="smoke",*args))[1]/"reviewed.json").exists()]
        write(log/"partial.json",{"rc":rc,"job":str(out),"units_without_seal":missing})
    return rc

if __name__=="__main__":
    raise SystemExit(main())

