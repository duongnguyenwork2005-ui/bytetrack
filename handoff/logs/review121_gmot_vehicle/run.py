"""Vehicle-only GMOT pilot at both ends of the published unknown FPS range."""
from pathlib import Path
from zipfile import ZipFile
from itertools import islice
from collections import Counter
import json
import os
import sys
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = Path('F:/LABOTARY_data/goc_tai_ve/GMOT40-R119')
os.environ.setdefault('YOLO_CONFIG_DIR', str(HERE / 'settings'))
sys.path.insert(0, str(HERE.parent / 'review115_new_datasets'))
import detect
import score
import trackeval

NAMES = [f'car-{i}' for i in range(3)]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    score.write_json(path, value)


def gt(name):
    with ZipFile(DATA / 'track_label.zip') as z:
        a = np.loadtxt(z.open(f'track_label/{name}.txt'), delimiter=',')
    a[:, 0] += 1
    a[:, 7:9] = 1
    return a


def audit():
    assert not (HERE / 'inputs.json').exists()
    manifest = json.loads((HERE.parent / 'review119_kitti_full/source_probes/gmot_downloads.json').read_text())
    paths = [Path(__file__), HERE / '00_protocol.md', DATA / 'track_label.zip', detect.WEIGHTS,
             Path(detect.__file__), Path(score.__file__)]
    paths += [ROOT / 'src' / p for p in ('run_tracking.py','tracker_ctrv.py','ekf_ctrv.py','ukf_ctrv.py','pf_ctrv.py','time_units.py','ul_backend.py')]
    paths += [Path(score.TYPES[p]) for p in ('ekf', 'ukf')]
    records = {}
    for name in NAMES:
        path = DATA / f'{name}.zip'; paths.append(path)
        assert detect.sha256(path) == manifest['sequences'][name]['sha256']
        with ZipFile(path) as z:
            assert z.testzip() is None
            images = sorted(z.namelist())
            assert [Path(x).stem for x in images] == [f'{i:06d}' for i in range(len(images))]
            for p in images:
                import hashlib
                assert hashlib.sha256(z.read(p)).hexdigest() == manifest['sequences'][name]['members'][p]['sha256']
        a = gt(name)
        assert a[:,0].max() == len(images) and a[:,0].min() == 1
        assert len(a) == len(np.unique(a[:,:2],axis=0)) and (a[:,4:6]>0).all()
        records[name] = {'frames': len(images), 'boxes':len(a),'ids':len(np.unique(a[:,1]))}
        output = HERE / f'gt/{name}/gt/gt.txt'
        output.parent.mkdir(parents=True,exist_ok=True)
        np.savetxt(output,a,delimiter=',',fmt='%.10g')
        paths.append(output)
    write(HERE / 'inputs.json', {'sequences':records,'assumed_fps':[24,30],
                                'sha256':{str(p):detect.sha256(p) for p in paths}})


def cache():
    from ultralytics import YOLO
    assert detect.sha256(detect.WEIGHTS) == detect.WEIGHTS_SHA
    model = YOLO(str(detect.WEIGHTS))
    hashes = {}
    for name in NAMES:
        path = DATA / f'{name}_yolov8n.npz'
        assert not path.exists()
        with ZipFile(DATA / f'{name}.zip') as z:
            images = sorted(z.namelist())
            frames = (cv2.imdecode(np.frombuffer(z.read(p),np.uint8),cv2.IMREAD_COLOR) for p in images)
            arrays, shape = [], None
            for batch in iter(lambda:list(islice(frames,4)),[]):
                assert all(x is not None for x in batch)
                shape = shape or batch[0].shape[:2]
                assert all(x.shape[:2] == shape for x in batch)
                for result in model.predict(batch,imgsz=640,conf=.05,iou=.7,max_det=3000,half=False,rect=True,device=0,verbose=False):
                    a=result.boxes.data.cpu().numpy().astype(np.float32)
                    assert len(a)<3000 and np.isfinite(a).all()
                    arrays.append(a)
            assert len(arrays)==len(images)
        np.savez_compressed(path,shape=shape,**{str(i):a for i,a in enumerate(arrays)})
        hashes[str(path)] = detect.sha256(path)
        print('CACHE',name,len(arrays),flush=True)
    write(HERE / 'cache_hashes.json',hashes)


class VehicleCache(score.Cache):
    def iter_frames(self):
        for i,a in enumerate(self.arrays):
            a=a[(a[:,4]>.25)&np.isin(a[:,5],[2,5,7])].copy()
            a[:,5]=2
            yield i+1,i/self.fps,False,a[:300]


def track():
    results = {}
    for fps in (24,30):
        records = {}
        for name in NAMES:
            cache=VehicleCache(DATA/f'{name}_yolov8n.npz',fps)
            gtp=HERE/f'gt/{name}/gt/gt.txt'
            ids,present,levels,turns=score.labels_for_camera(gtp,name,len(cache.arrays),fps)
            record={'frames':len(cache.arrays),'boxes':int(present.sum()),'turn_episodes':sum(t['level']=='turn25' for t in turns),'methods':{}}
            for method in ('kf','ekf','ukf'):
                cfg=score.rt.load_tracker_cfg(score.TYPES[method],{'track_buffer':fps})
                assert (cfg.track_high_thresh,cfg.track_low_thresh,cfg.new_track_thresh,cfg.match_thresh,cfg.fuse_score)==(.25,.1,.25,.8,True)
                mot,_,timing=score.rt.track_video(cache,cfg,1/fps,elapsed_mode='steps',seed=None,video=name)
                assert len(timing)==len(cache.arrays) and timing['n_steps'].eq(1).all()
                assert not mot.duplicated(['frame','id']).any() and np.isfinite(mot.to_numpy(float)).all()
                folder=HERE/f'fps{fps}/trackers/{method}/data';folder.mkdir(parents=True,exist_ok=True)
                path=folder/f'{name}.txt';mot.to_csv(path,index=False,header=False,float_format='%.10f')
                dataset=trackeval.datasets.MotChallenge2DBox({'PRINT_CONFIG':False,'TRACKERS_TO_EVAL':[method],'BENCHMARK':'GMOT40-car','SPLIT_TO_EVAL':'pilot','SKIP_SPLIT_FOL':True,'DO_PREPROC':False,'GT_FOLDER':str(HERE/'gt'),'TRACKERS_FOLDER':str(folder.parent.parent),'SEQ_INFO':{name:len(cache.arrays)}})
                raw=dataset.get_raw_seq_data(method,name);data=dataset.get_preprocessed_seq_data(raw,'pedestrian')
                assert data['num_gt_ids']==len(ids)
                for t,(compact,source) in enumerate(zip(data['gt_ids'],raw['gt_ids'])):
                    assert np.array_equal(ids[compact],source) and set(compact)==set(np.flatnonzero(present[t]))
                metrics={m.get_name():m.eval_sequence(data) for m in score.METRICS}
                assigned,changed=score.clear_assignments(data,present.shape);missed=present&(assigned<0)
                assert changed.sum()==metrics['CLEAR']['IDSW'] and missed.sum()==metrics['CLEAR']['CLR_FN']
                groups={k:{'boxes':int(mask.sum()),'IDSW':int((mask&changed).sum()),'FN':int((mask&missed).sum())} for k,mask in {'all':present,'turn25':levels['turn25']>=0}.items()}
                record['methods'][method]={'metrics':score.concise(metrics),'groups':groups,'raw':metrics,'mot_sha256':detect.sha256(path)}
                print('SCORE',fps,name,method,score.concise(metrics),groups,flush=True)
            records[name]=record
            write(HERE/f'fps{fps}/{name}.json',record)
        combined={}
        for method in ('kf','ekf','ukf'):
            metrics={}
            for m in score.METRICS:
                key=m.get_name();metrics[key]=m.combine_sequences({n:r['methods'][method]['raw'][key] for n,r in records.items()})
            groups={g:sum((Counter(r['methods'][method]['groups'][g]) for r in records.values()),Counter()) for g in ('all','turn25')}
            combined[method]={'metrics':score.concise(metrics),'groups':groups}
        results[str(fps)]={'sequences':records,'combined':combined}
        write(HERE/'scores.json',results)


def verify():
    frozen=json.loads((HERE/'inputs.json').read_text());caches=json.loads((HERE/'cache_hashes.json').read_text())
    for p,h in {**frozen['sha256'],**caches}.items():assert detect.sha256(Path(p))==h,p
    records=json.loads((HERE/'scores.json').read_text());units=0
    for fps in ('24','30'):
        for name,r in records[fps]['sequences'].items():
            for method,x in r['methods'].items():
                assert detect.sha256(HERE/f'fps{fps}/trackers/{method}/data/{name}.txt')==x['mot_sha256']
                units+=1
    write(HERE/'verification.json',{'status':'PASS','units':units,'hashes':len(frozen['sha256'])+len(caches),'cadence_confirmed':False})


if __name__=='__main__':{'audit':audit,'cache':cache,'track':track,'verify':verify}[sys.argv[1]]()
