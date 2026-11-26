"""Confirm a development-selected R132 filter on the complete UA test40 split."""
from pathlib import Path
from contextlib import nullcontext
import configparser
import json
import os
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'test40'
os.environ['BYTETRACK_BACKEND']='ultralytics'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ.setdefault('YOLO_CONFIG_DIR',str(OUT/'settings'))
sys.path[:0]=[str(ROOT/'src')]
import numpy as np
import pandas as pd
import trackeval
import boot_stats
import det_cache
import run_tracking as rt
import run_trackeval as rte
from predicted_noise import installed

MANIFEST=ROOT/'results/phase2/manifests/detrac_test.json'
SPLIT='DETRAC-test'
CACHE='yolov8n-coco_conf0.05_all'
CONFIGS={'kf':'bytetrack.yaml','ekf':ROOT/'configs/bytetrack_ekf_ctrv.yaml',
         'ukf':ROOT/'configs/bytetrack_ukf_ctrv.yaml',
         'native100':HERE.parent/'review123_pf100/rbpf100.yaml',
         'candidate100':HERE.parent/'review123_pf100/rbpf100.yaml'}


def videos():return json.loads(MANIFEST.read_text())['videos']
def read(path):return json.loads(path.read_text())
def sha(path):
    import hashlib
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def tag(mode,seed):return mode if mode in ('kf','ekf','ukf') else f'{mode}-s{seed}'
def config(mode):
    cfg=rt.load_tracker_cfg(CONFIGS[mode],{'track_buffer':30})
    assert (cfg.track_high_thresh,cfg.track_low_thresh,cfg.new_track_thresh,cfg.match_thresh,cfg.fuse_score)==(.25,.1,.25,.8,True)
    return cfg


class Prefix:
    def __init__(self,parent,length):self.parent,self.length,self.orig_shape=parent,length,parent.orig_shape
    def __len__(self):return self.length
    def iter_frames(self):
        from itertools import islice
        return islice(self.parent.iter_frames(),self.length)


def replay(video,mode,seed,length=None):
    vd=det_cache.open_cache(CACHE,video,.25,[2,3,5,7],300)
    if length is not None:vd=Prefix(vd,length)
    context=installed(mode=='candidate100') if mode in ('native100','candidate100') else nullcontext()
    with context:mot,cls,timing=rt.track_video(vd,config(mode),1/25,elapsed_mode='skip',seed=seed,video=video)
    assert len(timing)==len(vd) and np.isfinite(mot.to_numpy(float)).all()
    assert not mot.duplicated(['frame','id']).any()
    assert (mot[['bb_width','bb_height']]>0).all().all()
    return mot,cls,timing


def preflight():
    assert read(HERE/'bdd/summary.json')['pass']
    assert not (OUT/'inputs.json').exists()
    assert len(videos())==40
    old=read(HERE/'bdd/inputs.json')
    for p,h in old['sha256'].items():assert sha(p)==h,p
    paths=[Path(p) for p in old['sha256'] if '/data/' not in p.replace('\\','/')]
    paths += [Path(__file__),HERE/'01_test40_confirmation.md',HERE/'test40_bootstrap.py',MANIFEST,HERE/'bdd/summary.json']
    paths += [Path(rt.resolve_tracker_yaml(p)) for p in CONFIGS.values()]
    cache_root=det_cache.CACHE_ROOT/CACHE
    paths += [cache_root/'meta.json']
    frames={};configs={m:rt.cfg_to_dict(config(m)) for m in CONFIGS}
    for video in videos():
        cp=configparser.ConfigParser();cp.read(ROOT/f'data/processed/{SPLIT}/{video}/seqinfo.ini')
        assert cp.getint('Sequence','frameRate')==25
        vd=det_cache.open_cache(CACHE,video,.25,[2,3,5,7],300)
        assert len(vd)==cp.getint('Sequence','seqLength')
        frames[video]=len(vd)
        paths += [cache_root/f'{video}{suffix}' for suffix in ('.json','.npz')]
        paths += [ROOT/f'data/processed/{SPLIT}/{video}/gt/gt.txt',ROOT/f'data/processed/{SPLIT}/{video}/seqinfo.ini']
    write(OUT/'inputs.json',{'frames':frames,'configs':configs,'effective_measurement_noise_height':'prediction','sha256':{str(p):sha(p) for p in paths}})
    print('TEST40 PREFLIGHT',len(frames),sum(frames.values()),flush=True)


def smoke():
    assert (OUT/'inputs.json').exists()
    for video in (videos()[0],videos()[-1]):
        for mode in CONFIGS:
            for seed in (range(3) if mode.endswith('100') else (0,)):
                x,_,_=replay(video,mode,seed,64);y,_,_=replay(video,mode,seed,64)
                assert np.array_equal(x.to_numpy(float),y.to_numpy(float))
    write(OUT/'smoke.json',{'status':'PASS','repeat_pairs':18})
    print('TEST40 SMOKE PASS',flush=True)


def score(mode,seed):
    tracker=tag(mode,seed)
    ec,dc,mc=rte.build_configs(SPLIT,tracker,videos())
    ec.update(PRINT_RESULTS=False,PRINT_CONFIG=False,OUTPUT_SUMMARY=False,OUTPUT_DETAILED=False,
              LOG_ON_ERROR=str(OUT/f'error_{tracker}.log'))
    dc.update(TRACKERS_FOLDER=str(OUT/'tracking'),OUTPUT_FOLDER=str(OUT/'evaluation'))
    dataset=trackeval.datasets.MotChallenge2DBox(dc)
    result,message=trackeval.Evaluator(ec).evaluate([dataset],[trackeval.metrics.HOTA(mc),trackeval.metrics.CLEAR(mc),trackeval.metrics.Identity(mc)])
    assert all(v=='Success' for v in next(iter(message.values())).values())
    target=OUT/f'evaluation/{tracker}';target.mkdir(parents=True,exist_ok=True)
    raw=boot_stats.save_raw_fields(result,tracker,target/'raw_fields.json')
    assert set(raw)==set(videos())
    combined=boot_stats.combine(raw,videos());write(target/'combined.json',combined)
    print('TEST40 SCORE',tracker,combined,flush=True)


def full(mode,seed):
    assert read(OUT/'smoke.json')['status']=='PASS'
    frozen=read(OUT/'inputs.json')
    for video in videos():
        folder=OUT/f'tracking/{SPLIT}/{tag(mode,seed)}';marker=folder/f'meta/{video}.json'
        if marker.exists():
            assert read(marker)['mot_sha256']==sha(folder/f'data/{video}.txt')
            continue
        mot,cls,timing=replay(video,mode,seed)
        assert len(timing)==frozen['frames'][video]
        data=folder/f'data/{video}.txt';data.parent.mkdir(parents=True,exist_ok=True)
        mot.to_csv(data,index=False,header=False,float_format='%.10f')
        times=folder/f'timing/{video}.csv';times.parent.mkdir(exist_ok=True);timing.to_csv(times,index=False)
        classes=folder/f'data_cls/{video}.csv';classes.parent.mkdir(exist_ok=True);cls.to_csv(classes,index=False)
        write(marker,{'frames':len(timing),'mot_sha256':sha(data),'timing_sha256':sha(times),'classes_sha256':sha(classes)})
        print('DONE',tag(mode,seed),video,len(timing),flush=True)
    score(mode,seed)


def summary():
    frozen=read(OUT/'inputs.json')
    for p,h in frozen['sha256'].items():assert sha(p)==h,p
    scores={};units=0
    for mode in CONFIGS:
        rows=[]
        for seed in (range(3) if mode.endswith('100') else (0,)):
            tracker=tag(mode,seed);raw=boot_stats.load_raw_fields(OUT/f'evaluation/{tracker}/raw_fields.json')
            assert set(raw)==set(videos())
            row=boot_stats.combine(raw,videos());saved=read(OUT/f'evaluation/{tracker}/combined.json')
            assert all(abs(row[k]-saved[k])<1e-9 for k in boot_stats.METRICS);rows.append(row)
            for video in videos():
                folder=OUT/f'tracking/{SPLIT}/{tracker}';marker=read(folder/f'meta/{video}.json')
                assert marker['frames']==frozen['frames'][video]
                for kind,suffix,key in (('data','.txt','mot_sha256'),('timing','.csv','timing_sha256'),('data_cls','.csv','classes_sha256')):
                    assert sha(folder/kind/f'{video}{suffix}')==marker[key]
                units+=1
        scores[mode]={'mean':{k:float(np.mean([r[k] for r in rows])) for k in boot_stats.METRICS},'seeds':rows}
    current,old=scores['candidate100']['mean'],scores['native100']['mean']
    lowest=min(scores[m]['mean']['IDSW'] for m in ('kf','ekf','ukf'))
    gates={'IDSW_vs_all':current['IDSW']<lowest,'two_seeds':sum(r['IDSW']<lowest for r in scores['candidate100']['seeds'])>=2,
           'HOTA':current['HOTA']>=old['HOTA']-.001,'FN':current['FN']<=old['FN']*1.002}
    write(OUT/'summary.json',{'scores':scores,'gates':gates,'pass':all(gates.values())})
    write(OUT/'verification.json',{'status':'PASS','units':units,'hashes':len(frozen['sha256'])})
    print('TEST40 SUMMARY',scores,gates,flush=True)


if __name__=='__main__':
    if sys.argv[1]=='full':full(sys.argv[2],int(sys.argv[3]))
    else:{'preflight':preflight,'smoke':smoke,'summary':summary}[sys.argv[1]]()
