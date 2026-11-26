from pathlib import Path
from zipfile import ZipFile
from itertools import islice
import json
import sys
import os
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
os.environ.setdefault('YOLO_CONFIG_DIR',str(HERE/'settings'))
sys.path[:0]=[str(HERE.parent/'review115_new_datasets')]
import detect
import score
DATA=Path('F:/LABOTARY_data/goc_tai_ve/GMOT40-R119')


def write(path,value):score.write_json(path,value)


def images():
    with ZipFile(DATA/'car-0.zip') as z:
        for p in sorted(z.namelist()):
            yield cv2.imdecode(np.frombuffer(z.read(p),np.uint8),cv2.IMREAD_COLOR)


def positions(length):
    return sorted(set(list(range(0,length-640+1,512))+[length-640]))


def merge(arrays):
    a=np.concatenate(arrays)
    indices=[]
    for cls in np.unique(a[:,5]):
        rows=np.flatnonzero(a[:,5]==cls);boxes=a[rows,:4].copy();boxes[:,2:]-=boxes[:,:2]
        selected=cv2.dnn.NMSBoxes(boxes.tolist(),a[rows,4].tolist(),.05,.7)
        indices.extend(rows[np.asarray(selected).reshape(-1)].tolist())
    order=np.asarray(indices,dtype=int)
    order=order[np.argsort(-a[order,4],kind='stable')][:3000]
    return a[order]


def preflight():
    assert not (HERE/'inputs.json').exists()
    paths=[Path(__file__),HERE/'00_protocol.md',DATA/'car-0.zip',DATA/'track_label.zip',detect.WEIGHTS,
           HERE.parent/'review121_gmot_vehicle/inputs.json',HERE.parent/'review99_roundabout_turns/audit_turns.py']
    write(HERE/'inputs.json',{'sha256':{str(p):detect.sha256(p) for p in paths},'thresholds':[.25,.05],'methods':['full1280','sliced640']})


def infer():
    from ultralytics import YOLO
    assert detect.sha256(detect.WEIGHTS)==detect.WEIGHTS_SHA
    model=YOLO(str(detect.WEIGHTS))
    for mode in ('full1280','sliced640'):
        target=DATA/f'car-0_{mode}.npz';assert not target.exists()
        arrays=[]
        for i,image in enumerate(images()):
            assert image.shape[:2]==(1080,1920)
            if mode=='full1280':
                results=model.predict(image,imgsz=1280,conf=.05,iou=.7,max_det=3000,half=False,rect=True,device=0,verbose=False)
                a=results[0].boxes.data.cpu().numpy().astype(np.float32)
            else:
                offsets=[(x,y) for y in positions(image.shape[0]) for x in positions(image.shape[1])]
                tiles=[image[y:y+640,x:x+640] for x,y in offsets]
                chunks=[]
                for j in range(0,len(tiles),4):
                    batch=tiles[j:j+4]
                    results=model.predict(batch,imgsz=640,conf=.05,iou=.7,max_det=3000,half=False,rect=True,device=0,verbose=False)
                    for result,(x,y) in zip(results,offsets[j:j+4]):
                        b=result.boxes.data.cpu().numpy().astype(np.float32)
                        b[:,:4]+=np.array([x,y,x,y]);chunks.append(b)
                a=merge(chunks)
            assert len(a)<3000 and np.isfinite(a).all()
            arrays.append(a)
            if i%50==0:print(mode,i,flush=True)
        np.savez_compressed(target,shape=(1080,1920),**{str(i):a for i,a in enumerate(arrays)})
        print('CACHE',mode,len(arrays),detect.sha256(target),flush=True)


def coverage():
    gtpath=HERE.parent/'review121_gmot_vehicle/gt/car-0/gt/gt.txt'
    gt=np.loadtxt(gtpath,delimiter=',');results={}
    for mode,path in [('full640',DATA/'car-0_yolov8n.npz'),('full1280',DATA/'car-0_full1280.npz'),('sliced640',DATA/'car-0_sliced640.npz')]:
        cache=score.Cache(path,30);seen={}
        for conf in (.25,.05):
            hits=[]
            for t,a in enumerate(cache.arrays):
                g=gt[gt[:,0]==t+1];boxes=g[:,2:6].copy();boxes[:,2:]+=boxes[:,:2]
                a=a[(a[:,4]>conf)&np.isin(a[:,5],[2,5,7])][:300]
                if len(a):
                    intersection=np.maximum(0,np.minimum(boxes[:,None,2:],a[None,:,2:4])-np.maximum(boxes[:,None,:2],a[None,:,:2])).prod(axis=2)
                    union=(boxes[:,2:]-boxes[:,:2]).prod(axis=1)[:,None]+(a[:,2:4]-a[:,:2]).prod(axis=1)[None,:]-intersection
                    matched=(intersection/union).max(axis=1)>=.5
                else:matched=np.zeros(len(g),bool)
                hits.extend((int(r[0]),int(r[1]),bool(ok)) for r,ok in zip(g,matched))
            groups={}
            for fps in (24,30):
                ids,present,levels,_=score.labels_for_camera(gtpath,'car-0',228,fps)
                hit=np.zeros_like(present)
                for t,identity,ok in hits:hit[t-1,np.searchsorted(ids,identity)]=ok
                groups[str(fps)]={k:{'boxes':int(mask.sum()),'candidate':int((mask&hit).sum()),'fraction':float((mask&hit).sum()/mask.sum())} for k,mask in {'all':present,'turn25':levels['turn25']>=0}.items()}
            seen[str(conf)]=groups
        results[mode]={'sha256':detect.sha256(path),'coverage':seen}
    qualified={k:min(v['coverage']['0.25'][f]['turn25']['fraction'] for f in ('24','30')) for k,v in results.items() if k!='full640'}
    qualified={k:v for k,v in qualified.items() if v>=.25}
    selected=max(qualified,key=qualified.get) if qualified else None
    write(HERE/'coverage.json',{'results':results,'selected':selected})
    print('COVERAGE',results,'SELECTED',selected,flush=True)


if __name__=='__main__':{'preflight':preflight,'infer':infer,'coverage':coverage}[sys.argv[1]]()
