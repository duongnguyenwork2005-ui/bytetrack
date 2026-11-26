"""Generate one shared YOLOv8n COCO cache for seven unique MOT17 train scenes."""
from __future__ import annotations

import json
import sys
import zipfile
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
CACHE="r89_mot17_yolov8n_coco_c005"
WEIGHTS=ROOT/"models/yolov8n.pt"
LABELS=json.loads((ROOT/"handoff/logs/review89_mot17/label_audit.json").read_text(encoding="utf-8"))


def main():
    cdir=det_cache.CACHE_ROOT/CACHE
    spec=det_cache.DetectorSpec(weights=str(WEIGHTS),imgsz=640,conf=.05,iou=.7,max_det=3000,
                                device="0" if torch.cuda.is_available() else "cpu",dataset="mot17-r89",part="train")
    spec.fill_env()
    det_cache.write_meta(cdir,spec,mode="append")
    model=YOLO(str(WEIGHTS))
    with zipfile.ZipFile(SOURCE/"MOT17.zip") as z:
        names=z.namelist()
        for seq in SEQS:
            if (cdir/f"{seq}.npz").exists():
                det_cache.open_cache(CACHE,seq,.05,[0],300)
                print("REUSE",seq,flush=True)
                continue
            first=next(x for x in names if x.endswith(f"/{seq}/img1/000001.jpg") or x==f"{seq}/img1/000001.jpg")
            prefix=first[:-10]
            info=LABELS[seq]
            n=int(info["frames"]);fps=float(info["fps"]);w,h=info["resolution"]
            counts=np.zeros(n,dtype=np.int32);bad=np.zeros(n,dtype=bool);truncated=np.zeros(n,dtype=bool)
            chunks=[]
            for k in range(1,n+1):
                member=prefix+f"{k:06d}.jpg"
                img=cv2.imdecode(np.frombuffer(z.read(member),np.uint8),cv2.IMREAD_COLOR)
                if img is None or img.shape[:2]!=(h,w):raise RuntimeError((member,"decode/shape"))
                det=det_cache.detect_frame(model,img,spec)
                counts[k-1]=len(det);truncated[k-1]=len(det)>=spec.max_det
                chunks.append(det)
                if k%100==0 or k==n:print("CACHE",seq,k,"/",n,flush=True)
            assert not bad.any() and not truncated.any()
            idx=np.arange(1,n+1,dtype=np.int32)
            arrays=dict(frame_idx=idx,n_per_frame=counts,unreadable=bad,truncated=truncated,
                        time_s=(idx-1).astype(np.float64)/fps,data=np.concatenate(chunks).astype(np.float32),
                        orig_shape=np.array([h,w],dtype=np.int32))
            result=det_cache.save_video(cdir,seq,arrays,extra={"source_archive":str(SOURCE/"MOT17.zip"),
                    "source_member_prefix":prefix,"gt_sha256":info["sha256_gt"]},spec=spec)
            print("SAVED",seq,result,flush=True)


if __name__=="__main__":main()
