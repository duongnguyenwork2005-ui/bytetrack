"""Frozen R142 transfer; outputs confined to this experiment directory."""
from contextlib import nullcontext
import configparser
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('YOLO_CONFIG_DIR',str(HERE/'ultralytics_settings'))
sys.path[:0]=[str(ROOT/'src'),str(HERE.parent/'review142_pf_confidence'),
              str(ROOT/'src/r66_ngoai'),str(HERE.parent/'review99_roundabout_turns')]
import numpy as np
import pandas as pd
import trackeval
import det_cache
import run_tracking as rt
import run_trackeval as rte
import boot_stats
import ul_backend
from sher_metric import combine
from confidence import installed
from audit_turns import labels_for_camera,trace

R142=HERE.parent/'review142_pf_confidence'
HELMET=HERE.parent/'external_eval/r66/helmet'
PF_CFG=HERE.parent/'review123_pf100/rbpf100.yaml'
MODES=('kf','ekf','ukf','parent','candidate')
FIELDS=tuple(boot_stats.METRICS)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,p)


def cases():
    helmet=read(HELMET/'full/phase_inputs.json')
    sv=[f"seq_{i}_{'morning' if i<26 else 'evening'}" for i in range(22,31)]
    result={
        'roundabout':dict(split='RoundaboutHD-prefix3000',videos=[f'imagesc{i:03}' for i in range(1,5)],
             fps=15,buffer=18,cache='r64_roundabout_yolov8n_mixed5_c0.05',classes=[1,2,3,5,7],
             detector='yolov8n_mixed5',condition='c25',conf=.25,expected_frames=12000),
        'svmot':dict(split='SVMOTSYS-test',videos=sv,fps=30,buffer=36,
             cache='r64_svmot_yolov8n_coco_c0.05',classes=[1,2,3,5,7],
             detector='yolov8n_coco',condition='c25',conf=.25,expected_frames=8250)}
    for label,conf in (('c25',.25),('c10',.10)):
        result['helmet_'+label]=dict(split=helmet['split'],videos=helmet['clips'],fps=10,buffer=12,
              cache=helmet['cache_ids']['yolo11n_mixed5'],classes=[3],detector='yolo11n_mixed5',
              condition=label,conf=conf,expected_frames=3600)
    return result


def config(c,mode):
    path={'kf':'bytetrack.yaml','ekf':ROOT/'configs/bytetrack_ekf_ctrv.yaml',
          'ukf':ROOT/'configs/bytetrack_ukf_ctrv.yaml'}.get(mode,PF_CFG)
    return rt.load_tracker_cfg(path,{'track_buffer':c['buffer']})


def tag(mode,seed):return mode if mode in MODES[:3] else f'{mode}-s{seed}'
def historical(c,mode):
    old='cv' if mode=='kf' else mode
    prefix='r66helmet' if c['split'].startswith('HELMET') else 'r64'
    return f"{prefix}-{c['detector']}-{c['condition']}-{old}"
def historical_folder(c,mode):
    return ROOT/'data/processed/trackers'/c['split']/historical(c,mode)
def folder(case,mode,seed):
    c=cases()[case]
    return HERE/case/'tracking'/c['split']/tag(mode,seed)
def load_mot(p):return np.loadtxt(p,delimiter=',',ndmin=2) if Path(p).stat().st_size else np.empty((0,7))


class Prefix:
    def __init__(self,cache,n):self.cache,self.n,self.orig_shape=cache,n,cache.orig_shape
    def iter_frames(self):
        for i,row in enumerate(self.cache.iter_frames()):
            if i>=self.n:break
            yield row


def replay(c,video,mode,seed,length=None,route_off=False):
    cache=det_cache.open_cache(c['cache'],video,c['conf'],c['classes'],300)
    expected=length or len(cache)
    if length:cache=Prefix(cache,length)
    context=installed(enabled=mode=='candidate',forward=not route_off) if mode in MODES[3:] else nullcontext()
    with context:
        mot,cls,timing=rt.track_video(cache,config(c,mode),1/c['fps'],elapsed_mode='steps',
                                     seed=seed,video=video)
    assert len(timing)==expected
    assert np.isfinite(mot.to_numpy(float)).all() and not mot.duplicated(['frame','id']).any()
    assert (mot[['bb_width','bb_height']]>0).all().all()
    assert ((mot['frame']>=1)&(mot['frame']<=expected)).all()
    return mot,cls,timing


def preflight():
    assert not (HERE/'inputs.json').exists()
    proof=read(R142/'final_verification.json');assert proof['goal_observed_met']
    # Bind the exact approved PF implementation and its actual backend dependencies.
    source={p:h for p,h in proof['sha256'].items()
            if '/src/' in p.replace('\\','/') or Path(p).suffix.lower() in ('.py','.c','.dll','.yaml')}
    for p,h in source.items():assert sha(p)==h,p
    paths=[Path(__file__),HERE/'00_protocol.md',R142/'confidence.py',R142/'final_summary.json',
           R142/'final_verification.json',PF_CFG,Path(inspect.getfile(ul_backend.STrack)),
           ROOT/'src/r66_ngoai/sher_metric.py',HERE.parent/'review99_roundabout_turns/audit_turns.py',
           HELMET/'full/phase_inputs.json',HELMET/'manifest_test36.json']
    paths += [Path(inspect.getfile(x)) for x in (trackeval.Evaluator,trackeval.metrics.HOTA,
             trackeval.metrics.CLEAR,trackeval.metrics.Identity,trackeval.datasets.MotChallenge2DBox)]
    paths += [ROOT/'src'/name for name in ('run_tracking.py','ul_backend.py','tracker_ctrv.py','pf_ctrv.py',
              'ekf_ctrv.py','ukf_ctrv.py','rbpf_ctrv.py','det_cache.py','run_trackeval.py','boot_stats.py')]
    details={}
    for name,c in cases().items():
        gtroot=ROOT/'data/processed'/c['split'];gtlog=read(gtroot/'gt_log.json')
        paths += [gtroot/'gt_log.json']
        frames={};boxes=0;tracks=0
        cr=ROOT/'data/interim/det_cache'/c['cache'];paths += [cr/'meta.json']
        meta=read(cr/'meta.json')
        assert meta['conf']==.05 and meta['imgsz']==640 and meta['max_det']==3000
        assert meta['half'] is False and meta['mask_ignored_regions'] is False
        for video in c['videos']:
            gtpath=gtroot/video/'gt/gt.txt';ini=gtroot/video/'seqinfo.ini'
            assert sha(gtpath)==gtlog['videos'][video]['gt_sha256']
            cp=configparser.ConfigParser();cp.read(ini,encoding='utf-8')
            n=cp['Sequence'].getint('seqLength');assert cp['Sequence'].getfloat('frameRate')==c['fps']
            a=np.loadtxt(gtpath,delimiter=',',ndmin=2)
            assert (a[:,6:8]==1).all() and np.isfinite(a).all() and (a[:,4:6]>0).all()
            assert len(np.unique(a[:,:2],axis=0))==len(a)
            with np.load(cr/f'{video}.npz') as z:
                assert np.array_equal(z['frame_idx'],np.arange(1,n+1))
                assert np.allclose(z['time_s'],np.arange(n)/c['fps'],atol=1e-9,rtol=0)
                assert not z['unreadable'].any() and not z['truncated'].any()
                assert z['n_per_frame'].sum()==len(z['data'])
                d=z['data'];assert np.isfinite(d).all() and (d[:,2:4]>d[:,:2]).all()
                assert ((d[:,4]>=.05-1e-6)&(d[:,4]<=1)).all()
                assert tuple(z['orig_shape'])==(cp['Sequence'].getint('imHeight'),cp['Sequence'].getint('imWidth'))
            info=read(cr/f'{video}.json')
            # Early R64 SVMOT cache used the field name gt_sha256.
            assert info.get('gt_luc_sinh_sha256',info.get('gt_sha256'))==sha(gtpath)
            assert info['spec_sha256']==meta['spec_sha256'] and info['detector']==c['detector']
            assert info['checkpoint_sha256']==meta['weights_sha256']
            marker=read(cr/f'{video}.xong.json')
            if 'sha256' in marker:
                for rel,h in marker['sha256'].items():assert sha(cr/rel)==h
            frames[video]=n;boxes+=len(a);tracks+=len(np.unique(a[:,1]))
            paths += [gtpath,ini]+[cr/f'{video}{s}' for s in ('.npz','.json','.xong.json')]
            for mode in MODES[:3]:paths += [historical_folder(c,mode)/f'data/{video}.txt']
        assert sum(frames.values())==c['expected_frames']
        for mode in MODES[:3]:
            old=ROOT/'results/trackeval'/c['split']/historical(c,mode)
            certificate=old/('cong_r66.json' if name.startswith('helmet') else 'cong_r64.json')
            paths += [old/'raw_fields.json',old/'combined_metrics.csv',certificate]
            if name.startswith('helmet'):
                for rel,h in read(certificate)['sha256'].items():assert sha(old/rel)==h
            else:
                assert read(certificate)['gt_cham_sha256']=={v:sha(gtroot/v/'gt/gt.txt') for v in c['videos']}
        configs={m:rt.cfg_to_dict(config(c,m)) for m in MODES}
        for m in MODES[:3]:
            cfgpath={'kf':'bytetrack.yaml','ekf':ROOT/'configs/bytetrack_ekf_ctrv.yaml',
                     'ukf':ROOT/'configs/bytetrack_ukf_ctrv.yaml'}[m]
            paths.append(Path(rt.resolve_tracker_yaml(cfgpath)))
        assert configs['parent']['pf_num_particles']==configs['candidate']['pf_num_particles']==100
        details[name]={**c,'frames':frames,'boxes':boxes,'IDs_by_clip':tracks,'configs':configs}
    assert details['helmet_c25']['boxes']==9371 and details['helmet_c25']['IDs_by_clip']==287
    hashes={**source,**{str(p):sha(p) for p in paths}}
    write(HERE/'inputs.json',{'cases':details,'seeds':[0,1,2],'quality_units':765,'sha256':hashes})
    print('PREFLIGHT PASS',len(hashes),{n:(len(c['videos']),sum(c['frames'].values()),c['boxes']) for n,c in details.items()},flush=True)


def frozen_check():
    inputs=read(HERE/'inputs.json')
    for p,h in inputs['sha256'].items():assert sha(p)==h,p
    return inputs


def smoke():
    frozen_check();rows=[]
    for name,c in cases().items():
        for video in (c['videos'][0],c['videos'][-1]):
            for mode in MODES[:3]:
                mot,_,_=replay(c,video,mode,None,64)
                old=load_mot(historical_folder(c,mode)/f'data/{video}.txt');old=old[old[:,0]<=64]
                values=mot.to_numpy(float)
                assert values.shape==old.shape and np.array_equal(values[:,:2],old[:,:2]),(name,mode,video)
                assert np.allclose(values[:,2:6],old[:,2:6],atol=1e-3,rtol=0)
                assert np.allclose(values[:,6],old[:,6],atol=1e-9,rtol=0)
            for seed in range(3):
                a,ac,_=replay(c,video,'parent',seed,64)
                off,oc,_=replay(c,video,'candidate',seed,64,True)
                assert np.array_equal(a.to_numpy(float),off.to_numpy(float)) and ac.equals(oc)
                x,xc,_=replay(c,video,'candidate',seed,64)
                y,yc,_=replay(c,video,'candidate',seed,64)
                assert x.equals(y) and xc.equals(yc)
            rows.append({'case':name,'video':video,'historical_controls':3,'route_off_controls':3,'PF_repeats':3})
            print('SMOKE',name,video,flush=True)
    frozen_check();write(HERE/'smoke.json',{'status':'PASS','rows':rows})
    print('SMOKE PASS',flush=True)


def score(name,c,mode,seed):
    tracker=tag(mode,seed);target=HERE/name/'evaluation'/tracker
    if (target/'score_marker.json').exists():
        for rel,h in read(target/'score_marker.json')['sha256'].items():assert sha(target/rel)==h
        return
    ec,dc,mc=rte.build_configs(c['split'],tracker,c['videos'])
    ec.update(PRINT_RESULTS=False,PRINT_CONFIG=False,OUTPUT_SUMMARY=False,OUTPUT_DETAILED=False,
              LOG_ON_ERROR=str(HERE/name/f'error_{tracker}.log'))
    dc.update(TRACKERS_FOLDER=str(HERE/name/'tracking'),OUTPUT_FOLDER=str(HERE/name/'evaluation'))
    dataset=trackeval.datasets.MotChallenge2DBox(dc)
    result,message=trackeval.Evaluator(ec).evaluate([dataset],[trackeval.metrics.HOTA(mc),
                         trackeval.metrics.CLEAR(mc),trackeval.metrics.Identity(mc)])
    assert all(v=='Success' for v in next(iter(message.values())).values())
    target.mkdir(parents=True,exist_ok=True)
    raw=boot_stats.save_raw_fields(result,tracker,target/'raw_fields.json');assert set(raw)==set(c['videos'])
    combined=combine(raw,c['videos'])
    direct=next(iter(result.values()))[tracker]['COMBINED_SEQ']['pedestrian']
    expected={k:float(np.mean(direct['HOTA'][k])) for k in ('HOTA','DetA','AssA')}
    expected.update(IDF1=float(direct['Identity']['IDF1']))
    expected.update({k:float(direct['CLEAR'][k]) for k in ('MOTA','IDSW')})
    expected.update(FP=float(direct['CLEAR']['CLR_FP']),FN=float(direct['CLEAR']['CLR_FN']))
    assert all(abs(combined[k]-expected[k])<1e-9 for k in FIELDS),(name,tracker,combined,expected)
    if mode in MODES[:3]:
        reference=ROOT/'results/trackeval'/c['split']/historical(c,mode)
        old=boot_stats.load_raw_fields(reference/'raw_fields.json')
        assert raw==old,(name,mode,'historical per-video raw mismatch')
        old_csv=pd.read_csv(reference/'combined_metrics.csv').iloc[0]
        assert all(abs(combined[k]-old_csv[k])<1e-9 for k in FIELDS)
    write(target/'combined.json',combined)
    write(target/'score_marker.json',{'sha256':{p:sha(target/p) for p in ('raw_fields.json','combined.json')}})
    print('SCORE',name,tracker,combined,flush=True)


def tasks():
    return [(n,m,s) for n in cases() for m in MODES for s in (range(3) if m in MODES[3:] else (None,))]


def worker(index):
    frozen=frozen_check();assert read(HERE/'smoke.json')['status']=='PASS'
    for ix,(name,mode,seed) in enumerate(tasks()):
        if ix%3!=index:continue
        c=cases()[name];root=folder(name,mode,seed)
        for video in c['videos']:
            marker=root/f'meta/{video}.json'
            if marker.exists():
                for rel,h in read(marker)['sha256'].items():assert sha(root/rel)==h
                assert read(marker)['frames']==frozen['cases'][name]['frames'][video]
                continue
            mot,cls,timing=replay(c,video,mode,seed)
            assert len(timing)==frozen['cases'][name]['frames'][video]
            paths=[f'data/{video}.txt',f'data_cls/{video}.csv',f'timing/{video}.csv']
            for frame,rel,header in zip((mot,cls,timing),paths,(False,False,True)):
                path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
                frame.to_csv(path,index=False,header=header,float_format='%.10f')
            write(marker,{'frames':len(timing),'seed':seed,'mode':mode,
                          'sha256':{rel:sha(root/rel) for rel in paths}})
            print('DONE',name,tag(mode,seed),video,len(timing),flush=True)
        score(name,c,mode,seed)
    frozen_check();print('WORKER COMPLETE',index,flush=True)


def summary():
    frozen=frozen_check();out={};unit_count=0;artifact_hashes={}
    for name,c in cases().items():
        data={}
        for mode in MODES:
            seeds=[]
            for seed in (range(3) if mode in MODES[3:] else (None,)):
                root=folder(name,mode,seed);evaluation=HERE/name/'evaluation'/tag(mode,seed)
                raw=boot_stats.load_raw_fields(evaluation/'raw_fields.json');assert set(raw)==set(c['videos'])
                current=combine(raw,c['videos']);saved=read(evaluation/'combined.json')
                assert all(abs(current[k]-saved[k])<1e-9 for k in FIELDS)
                seeds.append(current)
                assert {p.stem for p in (root/'meta').glob('*.json')}==set(c['videos'])
                for video in c['videos']:
                    marker=root/f'meta/{video}.json';entry=read(marker)
                    assert entry['frames']==frozen['cases'][name]['frames'][video]
                    for rel,h in entry['sha256'].items():assert sha(root/rel)==h
                    artifact_hashes[str(marker)]=sha(marker);unit_count+=1
                for p in ('raw_fields.json','combined.json','score_marker.json'):
                    artifact_hashes[str(evaluation/p)]=sha(evaluation/p)
            data[mode]={'mean':{k:float(np.mean([r[k] for r in seeds])) for k in FIELDS},'seeds':seeds}
        candidate,parent=data['candidate']['mean'],data['parent']['mean']
        checks={'IDSW_below_all':candidate['IDSW']<min(data[m]['mean']['IDSW'] for m in MODES[:3]),
                'HOTA_vs_parent':candidate['HOTA']>=parent['HOTA']-.001,
                'FN_vs_parent':candidate['FN']<=parent['FN']*1.002}
        out[name]={'scores':data,'checks_descriptive':checks,'frames':c['expected_frames'],'videos':len(c['videos'])}
    assert unit_count==765
    write(HERE/'summary.json',out)
    write(HERE/'verification.json',{'status':'PASS','units':unit_count,'groups':len(tasks()),
          'historical_deterministic_groups':12,'sha256':artifact_hashes,'frozen_inputs':len(frozen['sha256'])})
    print('SUMMARY PASS',unit_count,{n:{m:r['mean']['IDSW'] for m,r in d['scores'].items()} for n,d in out.items()},flush=True)


if __name__=='__main__':
    if sys.argv[1].isdigit():worker(int(sys.argv[1]))
    else:{'preflight':preflight,'smoke':smoke,'summary':summary}[sys.argv[1]]()
