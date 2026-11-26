"""Exploratory car-only Urban Tracker benchmark on shared COCO detections."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import config
import det_cache

SOURCE = Path(r"F:\LABOTARY_data\goc_tai_ve\UrbanTracker-R69")
EVIDENCE = ROOT / "handoff/logs/review88_urbantracker"
SPLIT = "UrbanTracker-R88-car"
CACHE = "r88_urbantracker_yolov8n_coco_c005"
ROI_CACHE = "r88_urbantracker_rene_roi_yolov8n_coco_c005"
WEIGHTS = ROOT / "models/yolov8n.pt"
SCENES = {"rene": (7200, 30000/1001), "stmarc": (1000, 30.0)}
FILTERS = {"cv": ("bytetrack.yaml", None), "ekf": ("configs/bytetrack_ekf_ctrv.yaml", None),
           "ukf": ("configs/bytetrack_ukf_ctrv.yaml", None),
           "pf_default": ("configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml", [0,1,2]),
           "pf_motion100": ("configs/phase2/trackers/r72/motion100.yaml", [0,1,2])}


def read_gt(z, scene):
    with tempfile.TemporaryDirectory() as tmp:
        name=next(n for n in z.namelist() if n.endswith(f"{scene}_gt.sqlite"))
        p=Path(tmp)/"gt.sqlite";p.write_bytes(z.read(name))
        db=sqlite3.connect(str(p))
        try:
            rows=list(db.execute("SELECT b.frame_number,b.object_id,b.x_top_left,b.y_top_left,b.x_bottom_right,b.y_bottom_right,o.road_user_type FROM bounding_boxes b JOIN objects o ON b.object_id=o.object_id ORDER BY b.frame_number,b.object_id"))
        finally: db.close()
    return rows


def prepare():
    dest=config.PROCESSED_DIR/SPLIT
    seqmap=config.PROCESSED_DIR/"seqmaps"/f"{SPLIT}.txt"
    if dest.exists() or seqmap.exists(): raise RuntimeError("GT namespace already exists; inspect before overwrite")
    audit={}
    for scene,(start,fps) in SCENES.items():
        with zipfile.ZipFile(SOURCE/f"{scene}_annotations.zip") as z: rows=read_gt(z,scene)
        assert len(rows)>0
        lines=[];counts={"car":0,"distractor":0,"unknown":0}
        keys=set()
        for frame,oid,x1,y1,x2,y2,typ in rows:
            assert start<=frame<start+1000 and x2>x1 and y2>y1
            key=(frame,oid)
            assert key not in keys;keys.add(key)
            cl=1 if typ==1 else 8
            counts["car" if cl==1 else "distractor"]+=1
            counts["unknown"]+=int(typ==0)
            lines.append(f"{frame-start+1},{oid+1},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},1,{cl},1.000\n")
        d=dest/scene; (d/"gt").mkdir(parents=True)
        (d/"gt"/"gt.txt").write_text("".join(lines),encoding="utf-8")
        (d/"seqinfo.ini").write_text(f"[Sequence]\nname={scene}\nimDir=img\nframeRate={fps:.8f}\nseqLength=1000\nimWidth=1280\nimHeight=720\nimExt=.jpg\n",encoding="utf-8")
        audit[scene]={"original_gt_frames":[start,start+999],"jpg_frames":[start+1,start+1000],"fps":fps,
                      "counts":counts,"gt_sha256":hashlib.sha256((d/"gt"/"gt.txt").read_bytes()).hexdigest()}
    seqmap.write_text("name\nrene\nstmarc\n",encoding="utf-8")
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/"gt_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit,indent=2))


def build_cache():
    from ultralytics import YOLO
    import torch
    cdir=det_cache.CACHE_ROOT/CACHE
    spec=det_cache.DetectorSpec(weights=str(WEIGHTS),imgsz=640,conf=.05,iou=.7,max_det=3000,
                                device="0" if torch.cuda.is_available() else "cpu",dataset="urbantracker-r88",part="test")
    spec.fill_env()
    det_cache.write_meta(cdir,spec,mode="append")
    model=YOLO(str(WEIGHTS))
    for scene,(start,fps) in SCENES.items():
        if (cdir/f"{scene}.npz").exists():
            print("EXISTS",scene,"verify before reuse",flush=True)
            det_cache.open_cache(CACHE,scene,.05,[2],300)
            continue
        idx=np.arange(1,1001,dtype=np.int32)
        nper=np.zeros(1000,dtype=np.int32)
        unreadable=np.zeros(1000,dtype=bool);truncated=np.zeros(1000,dtype=bool)
        chunks=[]
        with zipfile.ZipFile(SOURCE/f"{scene}_frames.zip") as z:
            for k in range(1000):
                n=start+k+1
                image=cv2.imdecode(np.frombuffer(z.read(f"{scene}_frames/{n:08d}.jpg"),np.uint8),cv2.IMREAD_COLOR)
                if image is None or image.shape[:2]!=(720,1280): raise RuntimeError((scene,n,"decode/shape"))
                det=det_cache.detect_frame(model,image,spec)
                nper[k]=len(det);truncated[k]=len(det)>=spec.max_det
                chunks.append(det)
                if (k+1)%100==0: print("CACHE",scene,k+1,flush=True)
        assert not truncated.any()
        arrays=dict(frame_idx=idx,n_per_frame=nper,unreadable=unreadable,truncated=truncated,
                    time_s=(idx-1).astype(np.float64)/fps,data=np.concatenate(chunks).astype(np.float32),
                    orig_shape=np.array([720,1280],dtype=np.int32))
        info=det_cache.save_video(cdir,scene,arrays,
                                  extra={"source_zip":str(SOURCE/f"{scene}_frames.zip"),
                                         "source_frame_range":[start+1,start+1000]},spec=spec)
        print("SAVED",scene,info,flush=True)


def roi_cache():
    """Filter René detections by center in the publisher's ROI mask; no inference rerun."""
    src=det_cache.CACHE_ROOT/CACHE
    dst=det_cache.CACHE_ROOT/ROI_CACHE
    meta=json.loads((src/"meta.json").read_text(encoding="utf-8"))
    mask_path=SOURCE/"rene_mask.png"
    mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
    assert mask is not None and mask.shape==(720,1280)
    sha=hashlib.sha256(mask_path.read_bytes()).hexdigest()
    if dst.exists(): raise RuntimeError("ROI cache exists; inspect before reuse")
    dst.mkdir(parents=True)
    meta["dataset"]="urbantracker-r88-rene-roi"
    meta["derived_from_cache"]=CACHE
    meta["roi_rule"]="keep detection iff integer center pixel of xyxy is >127 in publisher rene_mask.png"
    meta["roi_mask_sha256"]=sha
    import copy
    spec=det_cache.DetectorSpec(**{k:v for k,v in meta.items() if k in det_cache.DetectorSpec.__dataclass_fields__ and k not in ("GEN_FIELDS","STRICT_FIELDS")})
    meta["spec_sha256"]=spec.spec_sha256()
    (dst/"meta.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    z=np.load(src/"rene.npz")
    nper=z["n_per_frame"]; data=z["data"]
    offset=0;chunks=[];new_n=[]
    for n in nper:
        frame=data[offset:offset+int(n)];offset+=int(n)
        center=((frame[:,:2]+frame[:,2:4])/2).astype(int)
        center[:,0]=np.clip(center[:,0],0,1279);center[:,1]=np.clip(center[:,1],0,719)
        frame=frame[mask[center[:,1],center[:,0]]>127]
        chunks.append(frame);new_n.append(len(frame))
    assert offset==len(data)
    arrays={k:z[k] for k in ("frame_idx","unreadable","truncated","time_s","orig_shape")}
    arrays["n_per_frame"]=np.array(new_n,dtype=np.int32)
    arrays["data"]=np.concatenate(chunks).astype(np.float32)
    info=det_cache.save_video(dst,"rene",arrays,extra={"derived_from":CACHE,"roi_mask_sha256":sha,
                 "roi_rule":meta["roi_rule"]},spec=spec)
    print("ROI",info,flush=True)


def track(scene,filter_name,conf):
    assert scene in SCENES and filter_name in FILTERS
    yaml,seeds=FILTERS[filter_name]
    cache=ROI_CACHE if scene=="rene" else CACHE
    name=f"r88{'roi' if scene=='rene' else ''}-{scene}-{conf}-{filter_name}"
    cmd=[sys.executable,"-B",str(ROOT/"src/run_tracking.py"),"--cache-id",cache,"--split-name",SPLIT,
         "--tracker-yaml",yaml,"--tracker-name",name,"--videos",scene,"--conf-min","0.25",
         "--classes","2","--max-det","300","--fps",str(SCENES[scene][1]),"--track-buffer-sec","1.2",
         "--elapsed-mode","steps","--skip-existing"]
    cmd[cmd.index("--conf-min")+1]={"c25":"0.25","c10":"0.10"}[conf]
    if seeds is not None: cmd += ["--seeds",*map(str,seeds)]
    subprocess.run(cmd,cwd=ROOT,check=True)


def evaluate(scene,filter_name,conf):
    from r64_ngoai import danh_gia_ngoai as r64
    _,seeds=FILTERS[filter_name]
    out={}
    for seed in (seeds if seeds is not None else [None]):
        name=f"r88{'roi' if scene=='rene' else ''}-{scene}-{conf}-{filter_name}"+(f"-s{seed}" if seed is not None else "")
        p=config.RESULTS_DIR/"trackeval"/SPLIT/name/"combined_metrics.csv"
        if p.exists():
            import pandas as pd
            row=pd.read_csv(p).iloc[0]
            out[name]={k:float(row[k]) for k in ("HOTA","IDF1","MOTA","IDSW")}
        else: out[name]=r64.cham(SPLIT,name,[scene])
        print("EVAL",name,out[name],flush=True)
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["prepare","cache","roi-cache","track","eval"])
    ap.add_argument("--scene",choices=SCENES);ap.add_argument("--filter",choices=FILTERS)
    ap.add_argument("--conf",choices=["c25","c10"],default="c25")
    a=ap.parse_args()
    if a.stage=="prepare":prepare()
    elif a.stage=="cache":build_cache()
    elif a.stage=="roi-cache":roi_cache()
    elif a.stage=="track":track(a.scene,a.filter,a.conf)
    else:evaluate(a.scene,a.filter,a.conf)
