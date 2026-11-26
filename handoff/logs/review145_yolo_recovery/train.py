"""Matched-budget KD recovery for dense and physically pruned YOLO11n."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('YOLO_CONFIG_DIR',str(HERE/'settings'))
sys.path[:0]=[str(ROOT/'src'),str(HERE.parent/'review104_yolo_compression')]
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from fit_channels import PrunedTrainer,train_with_distillation
from prepare import ANCHOR,SHA256

torch.set_num_threads(2)
NAMES={0:'car',1:'truck',2:'bus',3:'motorcycle',4:'bicycle'}
PRUNED=ROOT/'tmp/r104_yolo_compression/channels25_init.pt'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def label(image):
    text=str(image).replace('\\','/')
    assert '/images/' in text,text
    return Path(text.replace('/images/','/labels/')).with_suffix('.txt')


def prepare():
    assert not (HERE/'inputs.json').exists()
    assert sha(ANCHOR)==SHA256
    mixed=ROOT/'data/yolo_mixed5_v1'
    primary=ROOT/'handoff/logs/review91_pf_assoc/balanced_pilot/train.txt'
    supplement=ROOT/'tmp/r103_student/train.txt'
    monitor=ROOT/'tmp/r103_student/monitor_dev300.txt'
    train=[Path(p) for p in primary.read_text().splitlines() if p]
    assert len(train)==len(set(train))==5000
    assert all(p.parent==mixed/'images/train' for p in train)
    extra=[Path(p) for p in supplement.read_text().splitlines() if '/ridesafe/images/' in p.replace('\\','/')]
    assert len(extra)==200 and {p.name[:8] for p in extra}=={'20220824','20221119'}
    train+=extra
    train_set=set(train)
    dev=[Path(p) for p in monitor.read_text().splitlines() if p]
    assert len(dev)==300 and not set(train)&set(dev)
    data=HERE/'data';data.mkdir(parents=True,exist_ok=True)
    (data/'train.txt').write_text('\n'.join(p.as_posix() for p in train)+'\n',encoding='utf-8')
    (data/'dev300.txt').write_text('\n'.join(p.as_posix() for p in dev)+'\n',encoding='utf-8')
    (data/'data.yaml').write_text(yaml.safe_dump({'path':data.as_posix(),'train':str(data/'train.txt'),
                    'val':str(data/'dev300.txt'),'names':NAMES}),encoding='utf-8')
    paths=[Path(__file__),HERE/'00_protocol.md',Path(ANCHOR),PRUNED,primary,supplement,monitor,
           HERE.parent/'review104_yolo_compression/fit_channels.py',HERE.parent/'review104_yolo_compression/prepare.py',
           data/'train.txt',data/'dev300.txt',data/'data.yaml',mixed/'data.yaml',mixed/'manifest/anh.csv']
    hashes={str(p):sha(p) for p in paths};class_counts=np.zeros(5,dtype=int);image_hashes={}
    for image in train+dev:
        p=label(image);assert image.is_file() and p.is_file(),(image,p)
        digest=sha(image);hashes[str(image)]=digest;hashes[str(p)]=sha(p)
        image_hashes[str(image)]=digest
        a=np.loadtxt(p,ndmin=2) if p.stat().st_size else np.empty((0,5))
        assert a.shape[1]==5 and np.isfinite(a).all()
        if len(a):
            assert ((a[:,0]>=0)&(a[:,0]<5)&(a[:,0]==a[:,0].astype(int))).all()
            assert ((a[:,1:]>=0)&(a[:,1:]<=1)).all() and (a[:,3:]>0).all()
            if image in train_set:class_counts+=np.bincount(a[:,0].astype(int),minlength=5)
    assert not {image_hashes[str(p)] for p in train}&{image_hashes[str(p)] for p in dev}
    models={}
    for mode,path in (('control',Path(ANCHOR)),('channel25_KD',PRUNED)):
        m=YOLO(str(path),task='detect').model.float().eval()
        assert m.names==NAMES
        with torch.inference_mode():out=m(torch.zeros(1,3,384,640))[0]
        assert out.shape==(1,9,5040) and torch.isfinite(out).all()
        models[mode]={'path':str(path),'parameters':sum(p.numel() for p in m.parameters())}
    assert models['channel25_KD']['parameters']<models['control']['parameters']
    write(HERE/'inputs.json',{'sha256':hashes,'models':models,'train_images':len(train),'dev_monitor_images':len(dev),
              'train_class_boxes':dict(zip(NAMES.values(),class_counts.tolist())),
              'training':{'epochs':5,'imgsz':640,'batch':4,'seed':145,'distillation':True,'amp':False}})
    print('PREPARE PASS',models,class_counts,flush=True)


def check():
    frozen=read(HERE/'inputs.json')
    for p,h in frozen['sha256'].items():assert sha(p)==h,p
    return frozen


def train(mode):
    frozen=check();assert mode in frozen['models'];assert not (HERE/'runs'/mode).exists()
    model=YOLO(frozen['models'][mode]['path'],task='detect')
    args=dict(trainer=PrunedTrainer,data=str(HERE/'data/data.yaml'),epochs=5,imgsz=640,batch=4,workers=0,
         optimizer='SGD',lr0=.002,warmup_epochs=.5,mosaic=0,mixup=0,amp=False,device=0,seed=145,
         deterministic=True,patience=0,plots=False,project=str(HERE/'runs'),name=mode,exist_ok=False,verbose=False)
    train_with_distillation(model,args)
    best=HERE/'runs'/mode/'weights/best.pt';last=HERE/'runs'/mode/'weights/last.pt'
    trained=YOLO(str(best),task='detect').model
    assert sum(p.numel() for p in trained.parameters())==frozen['models'][mode]['parameters']
    check();write(HERE/f'{mode}_training.json',{'status':'PASS','best':str(best),'best_sha256':sha(best),
              'last_sha256':sha(last),'history_sha256':sha(HERE/'runs'/mode/'results.csv'),
              'parameters':frozen['models'][mode]['parameters'],'epochs_budget':5,'Nano_FPS':None})
    print('TRAINING PASS',mode,flush=True)


def evaluate():
    check();result={}
    models={'anchor':Path(ANCHOR),**{m:Path(read(HERE/f'{m}_training.json')['best']) for m in ('control','channel25_KD')}}
    for mode,path in models.items():
        model=YOLO(str(path),task='detect')
        score=model.val(data=str(ROOT/'data/yolo_mixed5_v1/data.yaml'),split='val',imgsz=640,batch=4,device=0,
               workers=0,conf=.001,iou=.7,max_det=300,half=False,plots=False,verbose=False,
               project=str(HERE/'evaluation'),name=mode,exist_ok=False)
        per={NAMES[int(k)]:{'AP50':float(score.box.ap50[i]),'AP50_95':float(score.box.ap[i]),
              'precision':float(score.box.p[i]),'recall':float(score.box.r[i])} for i,k in enumerate(score.box.ap_class_index)}
        result[mode]={'checkpoint':str(path),'sha256':sha(path),'parameters':sum(p.numel() for p in model.model.parameters()),
               'mAP50':float(score.box.map50),'mAP50_95':float(score.box.map),'classes':per,'split':'mixed5 dev7190'}
        write(HERE/'quality_partial.json',result)
    current=result['channel25_KD'];gates={}
    for ref in ('anchor','control'):
        gates[ref+'_mAP50']=current['mAP50']>=result[ref]['mAP50']-.005
        gates[ref+'_mAP50_95']=current['mAP50_95']>=result[ref]['mAP50_95']-.005
        gates[ref+'_per_class']=all(current['classes'][c]['AP50']>=result[ref]['classes'][c]['AP50']-.02 for c in NAMES.values())
    write(HERE/'quality.json',{'scores':result,'gates':gates,'recommended_quality':all(gates.values()),
                 'Nano_FPS_measured':False,'PC_timing_not_Nano':True})
    check();print('QUALITY COMPLETE',gates,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','train','evaluate'));parser.add_argument('mode',nargs='?')
    a=parser.parse_args()
    if a.action=='train':train(a.mode)
    else:{'prepare':prepare,'evaluate':evaluate}[a.action]()
