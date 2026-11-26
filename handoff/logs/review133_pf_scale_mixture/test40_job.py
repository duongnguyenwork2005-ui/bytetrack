"""Full test40 transfer of R133, with the completed fresh R132 controls frozen."""
from pathlib import Path
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'review132_pf_predicted_noise'
sys.path.insert(0,str(PARENT))
spec=importlib.util.spec_from_file_location('r132_test40_harness',PARENT/'test40_job.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import numpy as np
from mixture import installed
from predicted_noise import installed as parent_installed

base.HERE=HERE;base.OUT=HERE/'test40';base.installed=installed
OUT=base.OUT


def preflight():
    assert base.read(HERE/'bdd/summary.json')['pass']
    assert not (OUT/'inputs.json').exists()
    ua=base.read(PARENT/'test40/inputs.json');prior=base.read(HERE/'bdd/inputs.json')
    proof=base.read(PARENT/'final_verification.json')
    assert proof['status']=='PASS' and base.read(PARENT/'test40/verification.json')['units']==360
    hashes={**ua['sha256'],**prior['sha256']}
    for p,h in {**hashes,**proof['sha256']}.items():assert base.sha(p)==h,p
    paths=[Path(__file__),HERE/'01_test40_confirmation.md',PARENT/'test40/summary.json',PARENT/'final_verification.json']
    for mode in base.CONFIGS:
        for seed in (range(3) if mode.endswith('100') else (0,)):
            paths += [PARENT/f'test40/evaluation/{base.tag(mode,seed)}/raw_fields.json',
                      PARENT/f'test40/evaluation/{base.tag(mode,seed)}/combined.json']
    for seed in range(3):
        paths += [PARENT/f'test40/tracking/{base.SPLIT}/candidate100-s{seed}/data/{v}.txt'
                  for v in (base.videos()[0],base.videos()[-1])]
    hashes.update({str(p):base.sha(p) for p in paths})
    base.write(OUT/'inputs.json',{'frames':ua['frames'],'configs':ua['configs'],
              'effective_measurement_noise_height':'prediction','position_noise':'three-scale-Gaussian-mixture',
              'sha256':hashes})
    print('TEST40 PREFLIGHT',len(ua['frames']),sum(ua['frames'].values()),flush=True)


def smoke():
    assert (OUT/'inputs.json').exists()
    for seed in range(3):
        for video in (base.videos()[0],base.videos()[-1]):
            saved=base.installed;base.installed=parent_installed
            try:control,_,_=base.replay(video,'candidate100',seed,64)
            finally:base.installed=saved
            expected=np.loadtxt(PARENT/f'test40/tracking/{base.SPLIT}/candidate100-s{seed}/data/{video}.txt',delimiter=',',ndmin=2)
            assert np.allclose(control.to_numpy(float),expected[expected[:,0]<=64],atol=1e-9,rtol=0)
            x,_,_=base.replay(video,'candidate100',seed,64);y,_,_=base.replay(video,'candidate100',seed,64)
            assert np.array_equal(x.to_numpy(float),y.to_numpy(float))
    base.write(OUT/'smoke.json',{'status':'PASS','parent_controls':6,'repeat':6})
    print('TEST40 SMOKE PASS',flush=True)


def summary():
    frozen=base.read(OUT/'inputs.json')
    for p,h in frozen['sha256'].items():assert base.sha(p)==h,p
    parent=base.read(PARENT/'test40/summary.json')
    scores={m:parent['scores'][m] for m in ('kf','ekf','ukf','native100')}
    scores['parent100']=parent['scores']['candidate100']
    raws={};names=base.videos();units=0;rows=[]
    for seed in range(3):
        tracker=base.tag('candidate100',seed)
        raw=base.boot_stats.load_raw_fields(OUT/f'evaluation/{tracker}/raw_fields.json')
        assert set(raw)==set(names)
        score=base.boot_stats.combine(raw,names);recorded=base.read(OUT/f'evaluation/{tracker}/combined.json')
        assert all(abs(score[k]-recorded[k])<1e-9 for k in base.boot_stats.METRICS);rows.append(score)
        for video in names:
            folder=OUT/f'tracking/{base.SPLIT}/{tracker}';marker=base.read(folder/f'meta/{video}.json')
            assert marker['frames']==frozen['frames'][video]
            for kind,suffix,key in (('data','.txt','mot_sha256'),('timing','.csv','timing_sha256'),('data_cls','.csv','classes_sha256')):
                assert base.sha(folder/kind/f'{video}{suffix}')==marker[key]
            units+=1
    scores['candidate100']={'mean':{k:float(np.mean([r[k] for r in rows])) for k in base.boot_stats.METRICS},'seeds':rows}
    new,old=scores['candidate100']['mean'],scores['native100']['mean']
    target=min(scores[m]['mean']['IDSW'] for m in ('kf','ekf','ukf'));assert target==2498
    gates={'IDSW_vs_all':new['IDSW']<target,'two_seeds':sum(r['IDSW']<target for r in rows)>=2,
           'HOTA':new['HOTA']>=old['HOTA']-.001,'FN':new['FN']<=old['FN']*1.002}
    base.write(OUT/'summary.json',{'scores':scores,'gates':gates,'pass':all(gates.values())})
    base.write(OUT/'verification.json',{'status':'PASS','units':units,'reference_units':240,'hashes':len(frozen['sha256'])})
    print('TEST40 SUMMARY',new,gates,flush=True)


def bootstrap():
    names=base.videos();scores=base.read(OUT/'summary.json')['scores'];all_raw={}
    for mode in scores:
        folder=OUT if mode=='candidate100' else PARENT/'test40'
        actual='candidate100' if mode=='parent100' else mode
        all_raw[mode]=[base.boot_stats.load_raw_fields(folder/f'evaluation/{base.tag(actual,s)}/raw_fields.json')
                       for s in (range(3) if actual.endswith('100') else (0,))]
    stacks={m:[base.boot_stats._stack(r,names) for r in rows] for m,rows in all_raw.items()}
    fields=('HOTA','IDF1','IDSW','FN');refs=('kf','ekf','ukf','native100','parent100')
    samples={m:{f:[] for f in fields} for m in refs};rng=np.random.default_rng(133)
    for _ in range(5000):
        idx=rng.integers(0,len(names),size=len(names))
        estimates={m:[base.boot_stats.combine_stacked({f:st[f][idx] for f in st}) for st in rows] for m,rows in stacks.items()}
        means={m:{f:float(np.mean([r[f] for r in rows])) for f in fields} for m,rows in estimates.items()}
        for m in refs:
            for f in fields:samples[m][f].append(means['candidate100'][f]-means[m][f])
    base.write(OUT/'bootstrap.json',{'unit':'video','samples':5000,'candidate_minus_reference':
        {m:{f:{'difference':scores['candidate100']['mean'][f]-scores[m]['mean'][f],
                'ci95':np.quantile(v,[.025,.975]).tolist()} for f,v in rows.items()} for m,rows in samples.items()}})
    print('TEST40 BOOTSTRAP COMPLETE',flush=True)


if __name__=='__main__':
    if sys.argv[1].isdigit():base.full('candidate100',int(sys.argv[1]))
    else:{'preflight':preflight,'smoke':smoke,'summary':summary,'bootstrap':bootstrap}[sys.argv[1]]()
