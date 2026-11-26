"""Read the official MOT17 training labels without extracting images."""
from __future__ import annotations

import configparser
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=Path(r"F:\LABOTARY_data\goc_tai_ve\MOT17-R89\MOT17Labels.zip")
OUT=ROOT/"handoff/logs/review89_mot17/label_audit.json"

def main():
    result={}
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in sorted(n for n in z.namelist() if n.startswith("train/") and n.endswith("-FRCNN/gt/gt.txt")):
            seq=name.split("/")[1]
            lines=z.read(name)
            gt=np.loadtxt(__import__("io").BytesIO(lines),delimiter=",",ndmin=2)
            cfg=configparser.ConfigParser()
            cfg.read_string(z.read(f"train/{seq}/seqinfo.ini").decode("utf-8-sig"))
            sec=cfg["Sequence"]
            pair=np.stack((gt[:,0],gt[:,1]),axis=1).astype(np.int64)
            assert len(np.unique(pair,axis=0))==len(gt)
            assert gt.shape[1]==9 and np.isfinite(gt).all() and (gt[:,4:6]>0).all()
            assert gt[:,0].min()>=1 and gt[:,0].max()<=int(sec["seqLength"])
            classes=Counter(map(int,gt[:,7]))
            valid=(gt[:,6]==1)&(gt[:,7]==1)
            result[seq]={"frames":int(sec["seqLength"]),"fps":int(sec["frameRate"]),
                         "resolution":[int(sec["imWidth"]),int(sec["imHeight"])],
                         "all_rows":len(gt),"pedestrian_marked":int(valid.sum()),
                         "pedestrian_ids":len(np.unique(gt[valid,1])),
                         "visibility_lt_0_5":int(((gt[:,8]<.5)&valid).sum()),
                         "visibility_lt_0_25":int(((gt[:,8]<.25)&valid).sum()),
                         "class_rows":dict(classes),"sha256_gt":hashlib.sha256(lines).hexdigest()}
    OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
