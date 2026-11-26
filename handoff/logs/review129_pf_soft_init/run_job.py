"""Development gate for covariance-weighted velocity initialization."""
from pathlib import Path
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review118_pf_shape'))
import run as job
import numpy as np
from soft import installed, fuse, SoftPF
from pf_ctrv import PFCTRV
import rbpf_ctrv

OLD=HERE.parent/'review123_pf100'
job.HERE=HERE
job.CFG=OLD/'rbpf100.yaml'
job.installed=installed


def preflight():
    assert not (HERE/'inputs.json').exists()
    prior=json.loads((OLD/'inputs.json').read_text())
    for p,h in prior['sha256'].items():assert job.base.sha(Path(p))==h,p
    paths=[Path(p) for p in prior['sha256']]
    paths += [Path(__file__),HERE/'soft.py',HERE/'00_protocol.md',
              HERE.parent/'review126_pf_efficiency/fast.py',HERE.parent/'review126_pf_efficiency/native_pf.py',
              HERE.parent/'review126_pf_efficiency/native/kernel.c',HERE.parent/'review126_pf_efficiency/native/kernel.dll']
    paths += [OLD/f'dev/evaluation/cartesian-s{s}/raw_fields.json' for s in range(3)]
    baseline=[job.base.boot_stats.combine(job.base.boot_stats.load_raw_fields(OLD/f'dev/evaluation/cartesian-s{s}/raw_fields.json'),job.base.DEV) for s in range(3)]
    job.write(HERE/'inputs.json',{'frames':prior['frames'],'baseline_seeds':baseline,
             'sha256':{str(p):job.base.sha(p) for p in paths}})


def checks():
    rng=np.random.default_rng(129);largest=0.
    for scale in (1e-6,1.,1e6):
        b=rng.normal(size=(250,2,2))*np.sqrt(scale)
        p=b@b.transpose(0,2,1)+np.eye(2)[None]*scale*.1
        r=np.diag([.7,1.2])*scale
        m=rng.normal(size=(250,2))*np.sqrt(scale);z=rng.normal(size=2)*np.sqrt(scale)
        actual,cov=fuse(m,p,z,r)
        inverse=np.linalg.inv(p)+np.linalg.inv(r)
        expected_cov=np.linalg.inv(inverse)
        information=np.einsum('nij,nj->ni',np.linalg.inv(p),m)+np.linalg.solve(r,z)
        expected=np.einsum('nij,nj->ni',expected_cov,information)
        assert np.allclose(actual,expected,atol=1e-10*np.sqrt(scale),rtol=1e-10)
        assert np.allclose(cov,expected_cov,atol=1e-10*scale,rtol=1e-10)
        assert np.linalg.eigvalsh(cov).min()>0
        largest=max(largest,float(abs(cov-expected_cov).max()/scale))
    for seed in range(3):
        pf=PFCTRV(num_particles=100)
        mean,cov=pf.initiate(np.array([10.,20.,1.,50.]))
        particles=pf.sample_initial(mean,cov,np.random.default_rng(seed))
        a=SoftPF(pf,mean,cov,particles,np.random.default_rng(seed+10))
        before=[a.means.copy(),a.covariances.copy(),a.omega.copy()]
        state=a.rng.bit_generator.state
        _,_,ok=a.initialize_motion(mean,cov,np.array([10.,20.,1.,50.]),1,(10.,20.))
        assert not ok and a.rng.bit_generator.state==state
        assert all(np.array_equal(x,y) for x,y in zip(before,[a.means,a.covariances,a.omega]))
        _,_,ok=a.initialize_motion(mean,cov,np.array([12.,24.,1.,50.]),2,(10.,20.))
        assert ok and a.rng.bit_generator.state==state
        assert (a.means[:,2:]>0).all() and (a.means[:,2:]<[1.,2.]).all()
        for t in range(60):
            a.predict(t%11<3)
            x,p=a.update(np.array([12+t,24+2*t,1.,50.]))
            assert np.isfinite(x).all() and np.isfinite(p).all()
            assert np.linalg.eigvalsh(p).min()>-1e-8
    old=rbpf_ctrv.ConditionalPF
    with installed():assert rbpf_ctrv.ConditionalPF is SoftPF
    assert rbpf_ctrv.ConditionalPF is old
    for seed in range(3):
        for video in job.base.SMOKE:
            observed,_=job.replay(video,seed,False,128)
            raw=np.loadtxt(OLD/f'dev/tracking/DETRAC-all/cartesian-s{seed}/data/{video}.txt',delimiter=',',ndmin=2)
            assert np.allclose(observed.to_numpy(float),raw[raw[:,0]<=128],atol=1e-9,rtol=0)
            x,_=job.replay(video,seed,True,128);y,_=job.replay(video,seed,True,128)
            assert np.array_equal(x.to_numpy(float),y.to_numpy(float))
    job.write(HERE/'smoke.json',{'status':'PASS','fusion_cases':750,'scaled_covariance_error':largest,
             'no_motion_no_draw':True,'PSD':True,'hook_restore':True,'parent_parity':6,'repeat':6})
    print('CHECKS PASS',largest,flush=True)


def summary():
    frozen=json.loads((HERE/'inputs.json').read_text())
    for p,h in frozen['sha256'].items():assert job.base.sha(Path(p))==h,p
    scores=[];units=0
    for seed in range(3):
        raw=job.base.boot_stats.load_raw_fields(HERE/f'dev/evaluation/cartesian-s{seed}/raw_fields.json')
        assert set(raw)==set(job.base.DEV)
        score=job.base.boot_stats.combine(raw,job.base.DEV);scores.append(score)
        saved=json.loads((HERE/f'dev/evaluation/cartesian-s{seed}/combined.json').read_text())
        assert all(abs(score[k]-saved[k])<1e-9 for k in job.base.METRICS)
        for v in job.base.DEV:
            folder=HERE/f'dev/tracking/DETRAC-all/cartesian-s{seed}'
            marker=json.loads((folder/f'meta/{v}.json').read_text())
            assert marker['frames']==frozen['frames'][v] and marker['mot_sha256']==job.base.sha(folder/f'data/{v}.txt');units+=1
    mean=lambda rows:{k:float(np.mean([r[k] for r in rows])) for k in job.base.METRICS}
    old,new=mean(frozen['baseline_seeds']),mean(scores)
    gates={'IDSW_vs_all':new['IDSW']<472,'two_seeds':sum(r['IDSW']<472 for r in scores)>=2,
           'HOTA':new['HOTA']>=old['HOTA']-.001,'FN':new['FN']<=old['FN']*1.002}
    job.write(HERE/'dev_summary.json',{'baseline':old,'candidate':new,'seeds':scores,'gates':gates,'pass':all(gates.values())})
    job.write(HERE/'verification.json',{'status':'PASS','units':units,'hashes':len(frozen['sha256'])})
    print('SUMMARY',new,gates,flush=True)


if __name__=='__main__':
    if sys.argv[1]=='dev':job.dev(int(sys.argv[2]))
    else:{'preflight':preflight,'checks':checks,'summary':summary}[sys.argv[1]]()
