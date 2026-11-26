"""Gated candidate replay, keeping the R144 common-noise comparators frozen."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;R144=HERE.parent/'review144_shared_confidence'
sys.path.insert(0,str(R144))
spec=importlib.util.spec_from_file_location('r144_harness',R144/'job.py')
j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
from filter import installed
from confidence import installed as parent_installed
from ordered import OrderedPF,PredictedNoisePF,ordered_selection
from pf_ctrv import PFCTRV

j.HERE=HERE;j.pf_installed=installed


def controls(stage):return (R144 if stage=='dev' else HERE)/stage
def candidate_root(stage,seed):
    return HERE/stage/'tracking'/j.case(stage)['split']/f'pf-s{seed}'
def candidate_raw(stage,seed):return HERE/stage/'evaluation'/f'pf-s{seed}'/'raw_fields.json'


def numerical():
    proof=j.read(HERE.parent/'review140_pf_ordered_resample/final_verification.json')
    path=HERE.parent/'review140_pf_ordered_resample/ordered.py'
    frozen_path=path.parent/'inputs.json'
    assert proof['sha256'][str(frozen_path)]==j.sha(frozen_path)
    assert j.read(frozen_path)['sha256'][str(path)]==j.sha(path)
    rng=np.random.default_rng(146);steps=0;resamples=0
    for seed in range(3):
        objects=[]
        for cls in (PredictedNoisePF,OrderedPF):
            pf=PFCTRV(num_particles=100,resample_ess_ratio=.9)
            m,p=pf.initiate(np.array([50.,80.,1.5,70.]))
            particles=pf.sample_initial(m,p,np.random.default_rng(seed))
            objects.append(cls(pf,m,p,particles,np.random.default_rng(seed+10)))
        a,b=objects;b.ordered_stratified=False
        for t in range(100):
            for method,args in [('predict',(t%11<3,)),('update',(np.array([50+6*t,80+10*np.sin(t/8),1.5,70.]),(t%10)/10))]:
                x,p=getattr(a,method)(*args);y,q=getattr(b,method)(*args)
                assert np.array_equal(x,y) and np.array_equal(p,q)
                for f in ('means','covariances','omega','logw','shape_mean','shape_cov'):
                    assert np.array_equal(getattr(a,f),getattr(b,f))
                assert a.rng.bit_generator.state==b.rng.bit_generator.state;steps+=1
            assert np.linalg.eigvalsh(q).min()>-1e-6
        resamples+=a.resamples
    assert resamples>0
    for _ in range(100):
        n=int(rng.integers(2,101));w=rng.random(n);omega=rng.normal(size=n);u=rng.random(n)
        order=sorted(range(n),key=lambda i:omega[i]);cdf=np.cumsum(w[order]/w.sum());cdf[-1]=1
        expected=[order[next(k for k,c in enumerate(cdf) if c>(i+u[i])/n)] for i in range(n)]
        assert np.array_equal(ordered_selection(w,omega,u),expected)
    j.write(HERE/'numerical.json',{'status':'PASS','scored_control_steps':steps,'control_resamples':resamples,
            'selection_cases':100,'RNG_PSD':True,'source_R140_verified':True})
    print('NUMERICAL PASS',steps,resamples,flush=True)


def preflight(stage):
    assert j.read(HERE/'numerical.json')['status']=='PASS'
    if stage!='dev':assert j.read(HERE/('dev' if stage=='bdd' else 'bdd')/'summary.json')['pass']
    target=HERE/stage/'inputs.json';assert not target.exists()
    c=j.case(stage);old=j.R142/({'dev':'inputs.json','bdd':'bdd/inputs.json','test40':'test40/inputs.json'}[stage])
    frozen=j.read(old);hashes=frozen['sha256'].copy()
    for p,h in hashes.items():assert j.sha(p)==h,p
    paths=[Path(__file__),HERE/'filter.py',HERE/'00_protocol.md',HERE/'numerical.json',R144/'job.py',R144/'shared.py',
           HERE.parent/'review140_pf_ordered_resample/ordered.py',j.R142/'final_verification.json',old]
    for s in range(3):
        paths += [j.pf_raw(c,s)]+[j.pf_mot(c,s,v) for v in c['videos']]
    for method in j.METHODS:
        paths += [j.ref_raw(c,method)]+[j.ref_mot(c,method,v) for v in c['videos']]
    if stage=='dev':
        for p,h in j.read(R144/'dev/inputs.json')['sha256'].items():assert j.sha(p)==h,p
        for p,h in j.read(R144/'dev/verification.json')['sha256'].items():assert j.sha(p)==h,p
        paths += [R144/'dev/summary.json',R144/'dev/verification.json']
        for method in j.METHODS:
            paths += [R144/f'dev/evaluation/{method}/raw_fields.json']
    hashes.update({str(p):j.sha(p) for p in paths})
    j.write(target,{'sha256':hashes,'frames':c['frames'],'reuse_R144_dev_controls':stage=='dev'})
    print('PREFLIGHT',stage,len(c['videos']),sum(c['frames'].values()),flush=True)


def smoke(stage):
    j.check(stage);c=j.case(stage)
    saved=j.pf_installed;j.pf_installed=parent_installed
    try:j.smoke(stage)
    finally:j.pf_installed=saved
    for video in (c['videos'][0],c['videos'][-1]):
        for seed in range(3):
            a,ac,_=j.replay(c,video,'pf',seed,length=128)
            b,bc,_=j.replay(c,video,'pf',seed,length=128)
            assert a.equals(b) and ac.equals(bc)
    record=j.read(HERE/stage/'smoke.json');record['ordered_candidate_repeat']=6
    j.write(HERE/stage/'smoke.json',record);print('ORDERED SMOKE PASS',stage,flush=True)


def full_pf(stage,seed):
    frozen=j.check(stage);assert j.read(HERE/stage/'smoke.json')['status']=='PASS';c=j.case(stage)
    root=candidate_root(stage,seed)
    for video in c['videos']:
        marker=root/f'meta/{video}.json'
        if marker.exists():
            for rel,h in j.read(marker)['sha256'].items():assert j.sha(root/rel)==h
            continue
        mot,cls,timing=j.replay(c,video,'pf',seed)
        assert len(timing)==frozen['frames'][video]
        paths=[f'data/{video}.txt',f'data_cls/{video}.csv',f'timing/{video}.csv']
        for df,rel,header in zip((mot,cls,timing),paths,(False,False,True)):
            p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);df.to_csv(p,index=False,header=header,float_format='%.10f')
        j.write(marker,{'frames':len(timing),'sha256':{p:j.sha(root/p) for p in paths}})
        print('DONE',stage,'pf',seed,video,len(timing),flush=True)
    j.score(stage,f'pf-s{seed}');j.check(stage)


def worker(stage,index):
    tasks=[('pf',s) for s in range(3)] if stage=='dev' else [(m,None) for m in j.METHODS]+[('pf',s) for s in range(3)]
    for ix,(mode,seed) in enumerate(tasks):
        if ix%2!=index:continue
        if mode=='pf':full_pf(stage,seed)
        else:j.full(stage,mode)
    print('WORKER COMPLETE',stage,index,flush=True)


def summary(stage):
    j.check(stage);c=j.case(stage);scores={};fixed={};paths=[];new_units=0
    for method in j.METHODS:
        path=controls(stage)/'evaluation'/method/'raw_fields.json'
        raw=j.boot_stats.load_raw_fields(path);assert set(raw)==set(c['videos'])
        scores[method]=j.boot_stats.combine(raw,c['videos']);paths.append(path)
        fixed[method]=j.boot_stats.combine(j.boot_stats.load_raw_fields(j.ref_raw(c,method)),c['videos'])
        if stage!='dev':
            for video in c['videos']:
                root=HERE/stage/'tracking'/c['split']/method;marker=root/f'meta/{video}.json';entry=j.read(marker)
                for rel,h in entry['sha256'].items():assert j.sha(root/rel)==h
                assert entry['frames']==c['frames'][video];paths.append(marker);new_units+=1
    rows=[];parent=[]
    for seed in range(3):
        raw=j.boot_stats.load_raw_fields(candidate_raw(stage,seed));assert set(raw)==set(c['videos'])
        row=j.boot_stats.combine(raw,c['videos']);rows.append(row);paths.append(candidate_raw(stage,seed))
        parent.append(j.boot_stats.combine(j.boot_stats.load_raw_fields(j.pf_raw(c,seed)),c['videos']))
        saved=j.read(HERE/stage/'evaluation'/f'pf-s{seed}'/'combined.json')
        assert all(abs(row[k]-saved[k])<1e-9 for k in j.boot_stats.METRICS)
        for video in c['videos']:
            root=candidate_root(stage,seed);marker=root/f'meta/{video}.json';entry=j.read(marker)
            assert entry['frames']==c['frames'][video]
            for rel,h in entry['sha256'].items():assert j.sha(root/rel)==h
            paths.append(marker);new_units+=1
    mean=lambda rows:{k:float(np.mean([r[k] for r in rows])) for k in j.boot_stats.METRICS}
    scores['pf']=mean(rows);parent=mean(parent);target=min(scores[m]['IDSW'] for m in ('ekf','ukf'))
    gates={'EKF_below_KF':scores['ekf']['IDSW']<scores['kf']['IDSW'],
           'UKF_below_KF':scores['ukf']['IDSW']<scores['kf']['IDSW'],
           'PF_not_above_EKF_UKF':scores['pf']['IDSW']<=target,'PF_two_seeds':sum(r['IDSW']<=target for r in rows)>=2,
           'PF_HOTA':scores['pf']['HOTA']>=parent['HOTA']-.001,'PF_FN':scores['pf']['FN']<=parent['FN']*1.002}
    for m in j.METHODS:
        gates[m+'_HOTA']=scores[m]['HOTA']>=fixed[m]['HOTA']-.001
        gates[m+'_FN']=scores[m]['FN']<=fixed[m]['FN']*1.002
    data={'scores':scores,'fixed_controls':fixed,'PF_parent':parent,'PF_seeds':rows,'gates':gates,'pass':all(gates.values())}
    j.write(HERE/stage/'summary.json',data)
    j.write(HERE/stage/'verification.json',{'status':'PASS','new_units':new_units,
            'reused_controls':36 if stage=='dev' else 0,'sha256':{str(p):j.sha(p) for p in paths}})
    print('SUMMARY',stage,data,flush=True)


if __name__=='__main__':
    action=sys.argv[1]
    if action=='numerical':numerical()
    elif action=='worker':worker(sys.argv[2],int(sys.argv[3]))
    else:{'preflight':preflight,'smoke':smoke,'summary':summary}[action](sys.argv[2])
