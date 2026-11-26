"""Unchanged BDD replay/evaluator from R125, using the frozen R128 candidate."""
from pathlib import Path
import json
import sys

TOP=Path(__file__).resolve().parent
sys.path.insert(0,str(TOP.parent/'review125_pf_bdd'))
import run as job
import numpy as np
import pandas as pd
from heading import installed

OLD=job.HERE
HERE=TOP/'bdd'
job.HERE=HERE
job.installed=installed


def preflight():
    assert json.loads((TOP/'dev_summary.json').read_text())['pass']
    assert not (HERE/'inputs.json').exists()
    original=json.loads((OLD/'inputs.json').read_text())
    ua=json.loads((TOP/'inputs.json').read_text())
    hashes={**original['sha256'],**ua['sha256']}
    for p,h in hashes.items():assert job.sha(p)==h,p
    paths=[Path(__file__),TOP/'dev_summary.json',TOP/'verification.json',TOP/'smoke.json',
           OLD/'summary.json',OLD/'final_verification.json']
    paths += [OLD/f'evaluation/pf100-s{s}/raw_fields.json' for s in range(3)]
    paths += [OLD/f'evaluation/{mode}/raw_fields.json' for mode in ('kf','ekf','ukf')]
    hashes.update({str(p):job.sha(p) for p in paths})
    job.write(HERE/'inputs.json',{'sha256':hashes,'frames':original['frames'],
              'configs':original['configs'],'timing_videos':original['timing_videos']})


def smoke():
    assert (HERE/'inputs.json').exists()
    for s in range(3):
        for video in job.prior.SMOKE[:2]:
            # Temporarily install R126 for the control path, without changing job source.
            import native_pf
            saved=job.installed;job.installed=native_pf.installed
            try:observed,_,_=job.replay(video,'pf100',s,64)
            finally:job.installed=saved
            path=OLD/f'tracking/{job.SPLIT}/pf100-s{s}/data/{video}.txt'
            ref=np.loadtxt(path,delimiter=',',ndmin=2);ref=ref[ref[:,0]<=64]
            assert np.allclose(observed.to_numpy(float),ref,atol=1e-9,rtol=0)
            x,_,_=job.replay(video,'pf100',s,64);y,_,_=job.replay(video,'pf100',s,64)
            assert np.array_equal(x.to_numpy(float),y.to_numpy(float))
    job.write(HERE/'smoke.json',{'status':'PASS','control':6,'repeat':6})
    print('BDD SMOKE PASS',flush=True)


def summary():
    frozen=json.loads((HERE/'inputs.json').read_text())
    for p,h in frozen['sha256'].items():assert job.sha(p)==h,p
    names=job.videos();raws=[];scores=[];units=0
    for seed in range(3):
        raw=job.boot_stats.load_raw_fields(HERE/f'evaluation/pf100-s{seed}/raw_fields.json')
        assert set(raw)==set(names)
        raw_score=job.boot_stats.combine(raw,names);raws.append(raw);scores.append(raw_score)
        recorded=json.loads((HERE/f'evaluation/pf100-s{seed}/combined.json').read_text())
        assert all(abs(raw_score[k]-recorded[k])<1e-9 for k in job.boot_stats.METRICS)
        folder=HERE/f'tracking/{job.SPLIT}/pf100-s{seed}'
        for video in names:
            marker=json.loads((folder/f'meta/{video}.json').read_text())
            assert marker['frames']==frozen['frames'][video]
            for kind,suffix,key in (('data','.txt','mot_sha256'),('data_cls','.csv','classes_sha256'),('timing','.csv','timing_sha256')):
                assert job.sha(folder/kind/f'{video}{suffix}')==marker[key]
            timing=pd.read_csv(folder/f'timing/{video}.csv')
            assert len(timing)==marker['frames'] and timing.n_steps.eq(1).all();units+=1
    mean=lambda rows:{k:float(np.mean([r[k] for r in rows])) for k in job.boot_stats.METRICS}
    new=mean(scores);refs=json.loads((OLD/'summary.json').read_text())['summary'];old=refs['pf100']
    gates={'IDSW_vs_all':new['IDSW']<min(refs[m]['IDSW'] for m in ('kf','ekf','ukf')),
           'two_seeds':sum(r['IDSW']<5548 for r in scores)>=2,
           'HOTA':new['HOTA']>=old['HOTA']-.001,'FN':new['FN']<=old['FN']*1.002}
    job.write(HERE/'summary.json',{'candidate':new,'seeds':scores,'references':refs,'gates':gates,'pass':all(gates.values())})
    job.write(HERE/'verification.json',{'status':'PASS','units':units,'hashes':len(frozen['sha256']),'evaluator':'mot_crowd'})
    fields=('HOTA','IDF1','IDSW','FN');all_raw={'candidate':raws}
    for mode in ('pf100','kf','ekf','ukf'):
        all_raw[mode]=[job.boot_stats.load_raw_fields(OLD/f'evaluation/{job.tag(mode,s)}/raw_fields.json')
                       for s in (range(3) if mode=='pf100' else (None,))]
    stacks={m:[job.boot_stats._stack(r,names) for r in rows] for m,rows in all_raw.items()}
    samples={m:{f:[] for f in fields} for m in ('pf100','kf','ekf','ukf')}
    rng=np.random.default_rng(128)
    for _ in range(5000):
        idx=rng.integers(0,len(names),size=len(names))
        estimates={m:[job.boot_stats.combine_stacked({f:st[f][idx] for f in st}) for st in rows]
                   for m,rows in stacks.items()}
        means={m:{f:float(np.mean([r[f] for r in rows])) for f in fields} for m,rows in estimates.items()}
        for m in samples:
            for f in fields:samples[m][f].append(means['candidate'][f]-means[m][f])
    job.write(HERE/'bootstrap.json',{'unit':'video','samples':5000,'candidate_minus_reference':
        {m:{f:{'difference':new[f]-refs[m][f],'ci95':np.quantile(values,[.025,.975]).tolist()}
             for f,values in byfield.items()} for m,byfield in samples.items()}})
    print('BDD SUMMARY',new,gates,flush=True)


if __name__=='__main__':
    if sys.argv[1].isdigit():job.full('pf100',int(sys.argv[1]))
    else:{'preflight':preflight,'smoke':smoke,'summary':summary}[sys.argv[1]]()
