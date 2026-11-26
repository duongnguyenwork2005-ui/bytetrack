"""Stratified source detector recall for Urban Tracker, no frame extraction."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import det_cache

SOURCE = Path(r"F:\LABOTARY_data\goc_tai_ve\UrbanTracker-R69")
OUT = ROOT / "handoff/logs/review88_urbantracker"
CHECKPOINTS = {
    "yolov8n_coco": ROOT / "models/yolov8n.pt",
    "yolov8n_mixed5": ROOT / "models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/best.pt",
}
COCO = {1:2, 2:0, 4:1, 6:7}


def source(scene):
    with zipfile.ZipFile(SOURCE / f"{scene}_annotations.zip") as z, tempfile.TemporaryDirectory() as t:
        n = next(n for n in z.namelist() if n.endswith(f"{scene}_gt.sqlite"))
        p = Path(t) / "gt.sqlite"
        p.write_bytes(z.read(n))
        db = sqlite3.connect(str(p))
        try:
            rows = list(db.execute("SELECT b.frame_number,b.x_top_left,b.y_top_left,b.x_bottom_right,b.y_bottom_right,o.road_user_type FROM bounding_boxes b JOIN objects o ON b.object_id=o.object_id"))
        finally:
            db.close()
    by_frame = defaultdict(list)
    for frame,x1,y1,x2,y2,typ in rows:
        if typ in COCO:
            by_frame[frame].append((x1,y1,x2,y2,COCO[typ]))
    return by_frame


def iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    area_a=(a[2]-a[0])*(a[3]-a[1]);area_b=(b[2]-b[0])*(b[3]-b[1])
    return inter/(area_a+area_b-inter) if area_a+area_b-inter>0 else 0.0


def match(gt, det, cls):
    pairs=[]
    for i,g in enumerate(gt):
        for j,d in enumerate(det):
            if (not cls or int(g[4])==int(d[5])) and iou(g,d)>=.5:
                pairs.append((iou(g,d),i,j))
    used_g=set();used_d=set()
    for _,i,j in sorted(pairs,reverse=True):
        if i not in used_g and j not in used_d:
            used_g.add(i);used_d.add(j)
    return used_g


def main():
    import detector_classes as dc
    import hashlib
    result={"device":"0" if torch.cuda.is_available() else "cpu", "sample_rule":"50 evenly spaced annotated GT frames per scene; JPG=GT+1", "detectors":{}}
    for label,weights in CHECKPOINTS.items():
        assert weights.is_file()
        model=YOLO(str(weights)); names=model.names
        cls_map=None if label.endswith("coco") else {int(k):dc.sang_id_coco(v) for k,v in dc.anh_xa_model_sang_lop5(names).items()}
        spec=det_cache.DetectorSpec(weights=str(weights),imgsz=640,conf=.05,iou=.7,max_det=3000,
                                    device=result["device"],dataset="urbantracker-r88",part="test")
        summary={"checkpoint":str(weights),"checkpoint_sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),"scenes":{}}
        for scene,start in (("rene",7200),("stmarc",1000)):
            gt=source(scene)
            samples=np.linspace(start,start+999,50,dtype=int).tolist()
            count={str(c):{"gt":0,"matched_class":0,"matched_any":0} for c in sorted(set(COCO.values()))}
            count["all"]={"gt":0,"matched_class":0,"matched_any":0}
            with zipfile.ZipFile(SOURCE/f"{scene}_frames.zip") as z:
                for k,fr in enumerate(samples):
                    raw=z.read(f"{scene}_frames/{fr+1:08d}.jpg")
                    im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                    if im is None: raise RuntimeError((scene,fr))
                    det=det_cache.detect_frame(model,im,spec)
                    if cls_map is not None and len(det):
                        det[:,5]=[cls_map[int(x)] for x in det[:,5]]
                    truth=gt.get(fr,[])
                    hit_class=match(truth,det,True);hit_any=match(truth,det,False)
                    for i,g in enumerate(truth):
                        for key in (str(g[4]),"all"):
                            count[key]["gt"]+=1
                            count[key]["matched_class"]+=int(i in hit_class)
                            count[key]["matched_any"]+=int(i in hit_any)
                    if (k+1)%10==0: print(label,scene,k+1,"/",len(samples),flush=True)
            summary["scenes"][scene]={"samples":samples,"counts":count}
        result["detectors"][label]=summary
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/"detector_smoke.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["detectors"],indent=2))


if __name__=="__main__": main()
