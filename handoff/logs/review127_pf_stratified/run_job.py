from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review118_pf_shape'))
import run as job
from stratified import installed, gaussian_strata
OLD=HERE.parent/'review123_pf100'
job.HERE=HERE
job.CFG=OLD/'rbpf100.yaml'
job.installed=installed


def preflight():
    assert not (HERE/'inputs.json').exists()
    prior=json.loads((OLD/'inputs.json').read_text())
    for p,h in prior['sha256'].items():assert job.base.sha(Path(p))==h,p
    paths=[Path(p) for p in prior['sha256']]+[Path(__file__),HERE/'stratified.py',HERE/'00_protocol.md',
           HERE.parent/'review126_pf_efficiency/fast.py',HERE.parent/'review126_pf_efficiency/native_pf.py',
           HERE.parent/'review126_pf_efficiency/native/kernel.c',HERE.parent/'review126_pf_efficiency/native/kernel.dll',
           HERE.parent/'review126_pf_efficiency/native/build.json']
    paths += [OLD/f'dev/evaluation/cartesian-s{s}/raw_fields.json' for s in range(3)]
    baseline=[job.base.boot_stats.combine(job.base.boot_stats.load_raw_fields(OLD/f'dev/evaluation/cartesian-s{s}/raw_fields.json'),job.base.DEV) for s in range(3)]
    job.write(HERE/'inputs.json',{'frames':prior['frames'],'baseline_seeds':baseline,'sha256':{str(p):job.base.sha(p) for p in paths}})


def checks():
    rng=np.random.default_rng(127)
    a=np.array([gaussian_strata(rng,100) for _ in range(2000)])
    assert np.isfinite(a).all() and abs(a.mean())<.005 and abs(a.var()-1)<.02
    assert np.array_equal(gaussian_strata(np.random.default_rng(2),100),gaussian_strata(np.random.default_rng(2),100))
    for s in range(3):
        for video in job.base.SMOKE:
            observed,_=job.replay(video,s,False,128)
            raw=np.loadtxt(OLD/f'dev/tracking/DETRAC-all/cartesian-s{s}/data/{video}.txt',delimiter=',',ndmin=2)
            assert np.allclose(observed.to_numpy(float),raw[raw[:,0]<=128],atol=1e-9,rtol=0)
            x,_=job.replay(video,s,True,128);y,_=job.replay(video,s,True,128)
            assert np.array_equal(x.to_numpy(float),y.to_numpy(float))
    job.write(HERE/'smoke.json',{'status':'PASS','parent_parity':6,'repeat':6,'noise_mean':float(a.mean()),'noise_variance':float(a.var())})
    print('CHECKS PASS',float(a.mean()),float(a.var()),flush=True)


def summary():
    frozen=json.loads((HERE/'inputs.json').read_text())
    for p,h in frozen['sha256'].items():assert job.base.sha(Path(p))==h,p
    scores=[];units=0
    for s in range(3):
        raw=job.base.boot_stats.load_raw_fields(HERE/f'dev/evaluation/cartesian-s{s}/raw_fields.json')
        assert set(raw)==set(job.base.DEV)
        score=job.base.boot_stats.combine(raw,job.base.DEV);scores.append(score)
        saved=json.loads((HERE/f'dev/evaluation/cartesian-s{s}/combined.json').read_text())
        assert all(abs(score[k]-saved[k])<1e-9 for k in job.base.METRICS)
        for v in job.base.DEV:
            folder=HERE/f'dev/tracking/DETRAC-all/cartesian-s{s}'
            marker=json.loads((folder/f'meta/{v}.json').read_text())
            assert marker['frames']==frozen['frames'][v] and marker['mot_sha256']==job.base.sha(folder/f'data/{v}.txt');units+=1
    mean=lambda rows:{k:float(np.mean([r[k] for r in rows])) for k in job.base.METRICS}
    old,new=mean(frozen['baseline_seeds']),mean(scores)
    gate={'HOTA':new['HOTA']>=old['HOTA'],'IDSW':new['IDSW']<=old['IDSW'],'FN':new['FN']<=old['FN'],
          'improve':new['IDSW']<old['IDSW'] or new['HOTA']>old['HOTA'] or new['FN']<old['FN']}
    job.write(HERE/'dev_summary.json',{'baseline':old,'candidate':new,'seeds':scores,'gates':gate,'pass':all(gate.values())})
    job.write(HERE/'verification.json',{'status':'PASS','units':units,'hashes':len(frozen['sha256'])})
    print('SUMMARY',old,new,gate,flush=True)


if __name__=='__main__':
    if sys.argv[1]=='dev':job.dev(int(sys.argv[2]))
    else:{'preflight':preflight,'checks':checks,'summary':summary}[sys.argv[1]]()
