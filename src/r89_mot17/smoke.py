"""Detector recall on fixed 10-frame-per-sequence MOT17 train sample."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import det_cache
from r89_mot17.prepare import SEQS

SOURCE=Path(r"F:\LABOTARY_data\goc_tai_ve\MOT17-R89")
OUT=ROOT/"handoff/logs/review89_mot17/detector_smoke.json"
WEIGHTS=ROOT/"models/yolov8n.pt"


def iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb-inter>0 else 0


def match(truth,dets):
    pairs=[]
    for i,g in enumerate(truth):
        for j,d in enumerate(dets):
            v=iou(g,d)
            if v>=.5:pairs.append((v,i,j))
    used_gt=set();used_det=set()
    for _,i,j in sorted(pairs,reverse=True):
        if i not in used_gt and j not in used_det:
            used_gt.add(i);used_det.add(j)
    return used_gt


def main():
    model=YOLO(str(WEIGHTS))
    spec=det_cache.DetectorSpec(weights=str(WEIGHTS),imgsz=640,conf=.05,iou=.7,max_det=3000,
                                device="0" if torch.cuda.is_available() else "cpu",dataset="mot17-r89",part="train")
    result={"checkpoint":str(WEIGHTS),"checkpoint_sha256":hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
            "sample_rule":"10 evenly spaced frames per unique FRCNN-named training video", "scenes":{}}
    with zipfile.ZipFile(SOURCE/"MOT17Labels.zip") as labels, zipfile.ZipFile(SOURCE/"MOT17.zip") as imgs:
        names=imgs.namelist()
        for seq in SEQS:
            gt=np.loadtxt(io.BytesIO(labels.read(f"train/{seq}/gt/gt.txt")),delimiter=",",ndmin=2)
            n=int(gt[:,0].max())
            samples=np.linspace(1,n,10,dtype=int).tolist()
            first=next(x for x in names if x.endswith(f"/{seq}/img1/000001.jpg") or x==f"{seq}/img1/000001.jpg")
            prefix=first[:-10]
            counts={"all":{"gt":0,"hit":0},"vis_lt_0_5":{"gt":0,"hit":0},"vis_ge_0_5":{"gt":0,"hit":0}}
            for fr in samples:
                raw=imgs.read(prefix+f"{fr:06d}.jpg")
                image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                if image is None:raise RuntimeError((seq,fr))
                det=det_cache.detect_frame(model,image,spec)
                det=det[det[:,5]==0]
                rows=gt[(gt[:,0]==fr)&(gt[:,6]==1)&(gt[:,7]==1)]
                truth=[(r[2],r[3],r[2]+r[4],r[3]+r[5],r[8]) for r in rows]
                hit=match(truth,det)
                for i,g in enumerate(truth):
                    key="vis_lt_0_5" if g[4]<.5 else "vis_ge_0_5"
                    for k in ("all",key):
                        counts[k]["gt"]+=1;counts[k]["hit"]+=int(i in hit)
            result["scenes"][seq]={"samples":samples,"counts":counts,"image_member_prefix":prefix}
            print("SMOKE",seq,counts,flush=True)
    OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__=="__main__":main()
