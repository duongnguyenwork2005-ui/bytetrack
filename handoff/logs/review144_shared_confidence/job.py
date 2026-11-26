"""Same-input common-NSA experiment; full scenario transfer remains gated."""
from pathlib import Path
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
os.environ['BYTETRACK_BACKEND']='ultralytics'
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('YOLO_CONFIG_DIR',str(HERE/'settings'))
sys.path[:0]=[str(ROOT/'src'),str(HERE.parent/'review142_pf_confidence')]
import numpy as np
import trackeval
import det_cache
import run_tracking as rt
import run_trackeval as rte
import boot_stats
import ul_backend
import tracker_ctrv as tc
from shared import installed,score_scope
from confidence import installed as pf_installed

R142=HERE.parent/'review142_pf_confidence'
METHODS=('kf','ekf','ukf')
CONFIGS={'kf':'bytetrack.yaml','ekf':ROOT/'configs/bytetrack_ekf_ctrv.yaml',
         'ukf':ROOT/'configs/bytetrack_ukf_ctrv.yaml','pf':HERE.parent/'review123_pf100/rbpf100.yaml'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def case(stage):
    if stage=='bdd':
        frozen=read(R142/'bdd/inputs.json')
        return dict(split='BDDCOCOVID-val',videos=list(frozen['frames']),frames=frozen['frames'],fps=5,buffer=6,
           cache=ROOT/'results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01/cache/yolov8n_coco',classes=[1,2,3,5,7],
           evaluator='mot_crowd',fixed=HERE.parent/'review125_pf_bdd',pf=R142/'bdd',pf_tag='pf100',
           fixed_tag=lambda m:m,fixed_stage='',fixed_split='BDDCOCOVID-val',pf_stage='')
    if stage=='dev':
        frozen=read(R142/'inputs.json');split='DETRAC-all';fixed=HERE.parent/'review123_pf100/reference'
        pf=R142/'dev';tag='cartesian'
    else:
        assert stage=='test40';frozen=read(R142/'test40/inputs.json');split='DETRAC-test'
        fixed=HERE.parent/'review132_pf_predicted_noise/test40';pf=R142/'test40';tag='candidate100'
    return dict(split=split,videos=list(frozen['frames']),frames=frozen['frames'],fps=25,buffer=30,
          cache=ROOT/'data/interim/det_cache/yolov8n-coco_conf0.05_all',classes=[2,3,5,7],evaluator='mot',
          fixed=fixed,pf=pf,pf_tag=tag,fixed_tag=(lambda m:m+'-s0') if stage=='dev' else (lambda m:m))


def ref_raw(c,method):return c['fixed']/'evaluation'/c['fixed_tag'](method)/'raw_fields.json'
def ref_mot(c,method,video):return c['fixed']/'tracking'/c['split']/c['fixed_tag'](method)/'data'/f'{video}.txt'
def pf_raw(c,seed):return c['pf']/'evaluation'/f"{c['pf_tag']}-s{seed}"/'raw_fields.json'
def pf_mot(c,seed,video):return c['pf']/'tracking'/c['split']/f"{c['pf_tag']}-s{seed}"/'data'/f'{video}.txt'
def config(c,mode):return rt.load_tracker_cfg(CONFIGS[mode],{'track_buffer':c['buffer']})


class Prefix:
    def __init__(self,cache,length):self.cache,self.length,self.orig_shape=cache,length,cache.orig_shape
    def iter_frames(self):
        from itertools import islice
        return islice(self.cache.iter_frames(),self.length)


def replay(c,video,mode,seed=None,length=None,forward=True,audit=None):
    original=det_cache.CACHE_ROOT;det_cache.CACHE_ROOT=c['cache'].parent
    try:cache=det_cache.open_cache(c['cache'].name,video,.25,c['classes'],300)
    finally:det_cache.CACHE_ROOT=original
    expected=length or len(cache)
    if length:cache=Prefix(cache,length)
    context=pf_installed() if mode=='pf' else installed(forward,audit)
    with context:mot,cls,timing=rt.track_video(cache,config(c,mode),1/c['fps'],elapsed_mode='steps',seed=seed,video=video)
    assert len(timing)==expected and timing.n_steps.eq(1).all()
    assert np.isfinite(mot.to_numpy(float)).all() and not mot.duplicated(['frame','id']).any()
    assert (mot[['bb_width','bb_height']]>0).all().all()
    return mot,cls,timing


def numerical():
    from ekf_ctrv import EKFTrackerCTRV
    from ukf_ctrv import UKFTrackerCTRV
    rows=[]
    for maker in (ul_backend.KalmanFilterXYAH,EKFTrackerCTRV,UKFTrackerCTRV):
        a,b=maker(),maker();z=np.array([100.,200.,1.4,60.]);ma,pa=a.initiate(z);mb,pb=b.initiate(z)
        for t in range(120):
            ma,pa=a.predict(ma,pa);mb,pb=b.predict(mb,pb)
            z=np.array([100+t*.6,200+np.sin(t*.08)*8,1.4,60.]);score=(t%11)/10
            ma,pa=a.update(ma,pa,z,confidence=score)
            with score_scope(b,score):mb,pb=b.update(mb,pb,z)
            assert np.array_equal(ma,mb) and np.array_equal(pa,pb)
            assert np.linalg.eigvalsh(pb).min()>-1e-7
        assert 'update' not in b.__dict__
        try:
            with score_scope(b,.8):
                with score_scope(b,.3):raise RuntimeError('fixture')
        except RuntimeError:pass
        assert 'update' not in b.__dict__
        x,p=b.project(mb,pb,confidence=0);y,q=b.project(mb,pb)
        assert np.array_equal(x,y) and np.array_equal(p,q)
        rows.append({'filter':maker.__name__,'direct_score_steps':120,'PSD':True,'restore':True})
    originals=[cls.track_class for cls in (ul_backend.BYTETracker,tc.CTRVByteTracker,tc.UKFByteTracker)]
    with installed():pass
    assert originals==[cls.track_class for cls in (ul_backend.BYTETracker,tc.CTRVByteTracker,tc.UKFByteTracker)]
    write(HERE/'numerical.json',{'status':'PASS','rows':rows,'hook_restore':True})
    print('NUMERICAL PASS',flush=True)


def check(stage):
    frozen=read(HERE/stage/'inputs.json')
    for p,h in frozen['sha256'].items():assert sha(p)==h,p
    return frozen


def preflight(stage):
    assert read(HERE/'numerical.json')['status']=='PASS'
    if stage!='dev':assert read(HERE/('dev' if stage=='bdd' else 'bdd')/'summary.json')['pass']
    target=HERE/stage/'inputs.json';assert not target.exists()
    c=case(stage);legacy=read(R142/'final_verification.json');assert legacy['goal_observed_met']
    stage_input=R142/({'dev':'inputs.json','bdd':'bdd/inputs.json','test40':'test40/inputs.json'}[stage])
    hashes=read(stage_input)['sha256'].copy()
    for p,h in hashes.items():assert sha(p)==h,p
    paths=[Path(__file__),HERE/'shared.py',HERE/'00_protocol.md',HERE/'numerical.json',
           stage_input,R142/'final_verification.json',Path(inspect.getfile(ul_backend.STrack)),
           Path(inspect.getfile(ul_backend.KalmanFilterXYAH))]
    for path in CONFIGS.values():paths.append(Path(rt.resolve_tracker_yaml(path)))
    for mode in METHODS:
        paths += [ref_raw(c,mode)]+[ref_mot(c,mode,v) for v in c['videos']]
    for seed in range(3):
        raw=pf_raw(c,seed);assert str(raw) in legacy['sha256'] and sha(raw)==legacy['sha256'][str(raw)]
        paths += [raw]+[pf_mot(c,seed,v) for v in c['videos']]
    hashes.update({str(p):sha(p) for p in paths})
    write(target,{'sha256':hashes,'frames':c['frames'],'backend':ul_backend.backend_name(),
                  'STrack':inspect.getfile(ul_backend.STrack),'reuse_PF_R142':True})
    print('PREFLIGHT',stage,len(c['videos']),sum(c['frames'].values()),flush=True)


def equal_prefix(values,path,n):
    old=np.loadtxt(path,delimiter=',',ndmin=2) if path.stat().st_size else np.empty((0,7))
    old=old[old[:,0]<=n]
    assert values.shape==old.shape and np.array_equal(values[:,:2],old[:,:2])
    assert np.allclose(values[:,2:6],old[:,2:6],atol=1e-3,rtol=0)
    assert np.allclose(values[:,6],old[:,6],atol=1e-9,rtol=0)


def smoke(stage):
    check(stage);c=case(stage);seen=[];rows=[]
    for video in (c['videos'][0],c['videos'][-1]):
        for mode in METHODS:
            off,_,_=replay(c,video,mode,length=128,forward=False)
            equal_prefix(off.to_numpy(float),ref_mot(c,mode,video),128)
            a,ac,_=replay(c,video,mode,length=128,audit=lambda current,delivered:seen.append((current,delivered)))
            b,bc,_=replay(c,video,mode,length=128)
            assert a.equals(b) and ac.equals(bc)
            rows.append({'mode':mode,'video':video,'off_historical':True,'on_repeat':True})
        for seed in range(3):
            mot,_,_=replay(c,video,'pf',seed,length=128)
            equal_prefix(mot.to_numpy(float),pf_mot(c,seed,video),128)
    assert seen and all(p==q and 0<=p<=1 for p,q in seen)
    check(stage);write(HERE/stage/'smoke.json',{'status':'PASS','rows':rows,'current_score_updates':len(seen),'PF_parity':6})
    print('SMOKE PASS',stage,len(seen),flush=True)


def score(stage,mode):
    c=case(stage);out=HERE/stage;target=out/'evaluation'/mode
    ec,dc,mc=rte.build_configs(c['split'],mode,c['videos'])
    ec.update(PRINT_RESULTS=False,PRINT_CONFIG=False,OUTPUT_SUMMARY=False,OUTPUT_DETAILED=False,LOG_ON_ERROR=str(out/f'error_{mode}.log'))
    dc.update(TRACKERS_FOLDER=str(out/'tracking'),OUTPUT_FOLDER=str(out/'evaluation'))
    dataset=rte.dataset_class(c['evaluator'])(dc)
    result,message=trackeval.Evaluator(ec).evaluate([dataset],[trackeval.metrics.HOTA(mc),trackeval.metrics.CLEAR(mc),trackeval.metrics.Identity(mc)])
    assert all(v=='Success' for v in next(iter(message.values())).values())
    target.mkdir(parents=True,exist_ok=True);raw=boot_stats.save_raw_fields(result,mode,target/'raw_fields.json')
    combined=boot_stats.combine(raw,c['videos']);write(target/'combined.json',combined)
    print('SCORE',stage,mode,combined,flush=True)


def full(stage,mode):
    frozen=check(stage);assert read(HERE/stage/'smoke.json')['status']=='PASS';c=case(stage)
    root=HERE/stage/'tracking'/c['split']/mode
    for video in c['videos']:
        marker=root/f'meta/{video}.json'
        if marker.exists():
            for rel,h in read(marker)['sha256'].items():assert sha(root/rel)==h
            continue
        mot,cls,timing=replay(c,video,mode)
        assert len(timing)==frozen['frames'][video]
        paths=[f'data/{video}.txt',f'data_cls/{video}.csv',f'timing/{video}.csv']
        for frame,rel,header in zip((mot,cls,timing),paths,(False,False,True)):
            path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
            frame.to_csv(path,index=False,header=header,float_format='%.10f')
        write(marker,{'frames':len(timing),'sha256':{p:sha(root/p) for p in paths}})
        print('DONE',stage,mode,video,len(timing),flush=True)
    score(stage,mode);check(stage)


def summary(stage):
    check(stage);c=case(stage);scores={};controls={};paths=[];count=0
    for mode in METHODS:
        raw=boot_stats.load_raw_fields(HERE/stage/'evaluation'/mode/'raw_fields.json')
        assert set(raw)==set(c['videos']);scores[mode]=boot_stats.combine(raw,c['videos'])
        controls[mode]=boot_stats.combine(boot_stats.load_raw_fields(ref_raw(c,mode)),c['videos'])
        for video in c['videos']:
            root=HERE/stage/'tracking'/c['split']/mode;marker=root/f'meta/{video}.json';entry=read(marker)
            assert entry['frames']==c['frames'][video]
            for rel,h in entry['sha256'].items():assert sha(root/rel)==h
            paths.append(marker);count+=1
        paths += [HERE/stage/'evaluation'/mode/p for p in ('raw_fields.json','combined.json')]
    pf=[boot_stats.combine(boot_stats.load_raw_fields(pf_raw(c,s)),c['videos']) for s in range(3)]
    scores['pf']={k:float(np.mean([r[k] for r in pf])) for k in boot_stats.METRICS}
    target=min(scores[m]['IDSW'] for m in ('ekf','ukf'))
    gates={'EKF_below_KF':scores['ekf']['IDSW']<scores['kf']['IDSW'],
           'UKF_below_KF':scores['ukf']['IDSW']<scores['kf']['IDSW'],
           'PF_not_above_EKF_UKF':scores['pf']['IDSW']<=target,'PF_two_seeds':sum(r['IDSW']<=target for r in pf)>=2}
    for m in METHODS:
        gates[m+'_HOTA']=scores[m]['HOTA']>=controls[m]['HOTA']-.001
        gates[m+'_FN']=scores[m]['FN']<=1.002*controls[m]['FN']
    result={'scores':scores,'fixed_controls':controls,'PF_seeds':pf,'gates':gates,'pass':all(gates.values())}
    write(HERE/stage/'summary.json',result)
    write(HERE/stage/'verification.json',{'status':'PASS','new_units':count,'reused_PF_units':len(c['videos'])*3,
         'sha256':{str(p):sha(p) for p in paths}})
    print('SUMMARY',stage,result,flush=True)


if __name__=='__main__':
    action=sys.argv[1]
    if action=='numerical':numerical()
    elif action=='full':full(sys.argv[2],sys.argv[3])
    else:{'preflight':preflight,'smoke':smoke,'summary':summary}[action](sys.argv[2])
