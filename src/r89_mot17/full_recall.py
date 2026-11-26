"""Full-cache one-to-one GT recall by source visibility and confidence."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import det_cache
from r89_mot17.prepare import SEQS,SPLIT
from r89_mot17.cache import CACHE

OUT=ROOT/"handoff/logs/review89_mot17/full_detector_recall.json"


def greedy(gt,det):
    if not len(gt) or not len(det):return set()
    g1=gt[:,:2];g2=g1+gt[:,2:4];d1=det[:,:2];d2=det[:,2:4]
    lo=np.maximum(g1[:,None,:],d1[None,:,:]);hi=np.minimum(g2[:,None,:],d2[None,:,:])
    wh=np.maximum(hi-lo,0)
    inter=wh[:,:,0]*wh[:,:,1]
    ga=gt[:,2]*gt[:,3];da=(d2[:,0]-d1[:,0])*(d2[:,1]-d1[:,1])
    union=ga[:,None]+da[None,:]-inter
    iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
    ii,jj=np.where(iou>=.5)
    order=np.argsort(-iou[ii,jj],kind="stable")
    used_g=set();used_d=set()
    for ix in order:
        i=int(ii[ix]);j=int(jj[ix])
        if i not in used_g and j not in used_d:used_g.add(i);used_d.add(j)
    return used_g


def empty():return {"all":{"gt":0,"hit":0},"vis_lt_0_5":{"gt":0,"hit":0},"vis_ge_0_5":{"gt":0,"hit":0}}


def main():
    result={"cache":CACHE,"iou":.5,"matching":"greedy one-to-one by descending IoU", "thresholds":{}}
    for th in (.05,.1,.25):
        scenes={}
        for seq in SEQS:
            gt=np.loadtxt(ROOT/"data/processed"/SPLIT/seq/"gt/gt.txt",delimiter=",",ndmin=2)
            gt=gt[(gt[:,6]==1)&(gt[:,7]==1)]
            grouped=defaultdict(list)
            for row in gt:grouped[int(row[0])].append(row)
            by_frame={k:np.asarray(v) for k,v in grouped.items()}
            counts=empty()
            with np.load(det_cache.CACHE_ROOT/CACHE/f"{seq}.npz") as z:
                nper=z["n_per_frame"];data=z["data"];off=0
                for k,n in enumerate(nper,1):
                    det=data[off:off+n];off+=int(n)
                    det=det[(det[:,4]>th)&(det[:,5]==0)]
                    rows=by_frame.get(k,np.empty((0,9)))
                    hit=greedy(rows[:,2:6],det)
                    for i,g in enumerate(rows):
                        key="vis_lt_0_5" if g[8]<.5 else "vis_ge_0_5"
                        for c in ("all",key):
                            counts[c]["gt"]+=1;counts[c]["hit"]+=int(i in hit)
                assert off==len(data)
            scenes[seq]=counts
            print(th,seq,counts["all"],flush=True)
        total=empty()
        for counts in scenes.values():
            for c in total:
                for k in ("gt","hit"):total[c][k]+=counts[c][k]
        result["thresholds"][str(th)]={"scenes":scenes,"total":total}
        OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("DONE",OUT)


if __name__=="__main__":main()
