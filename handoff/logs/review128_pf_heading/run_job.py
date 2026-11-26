"""UA development gate for heading diffusion, with frozen sources and R123 control."""
from pathlib import Path
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review118_pf_shape'))
import run as job
import numpy as np
from heading import installed, diffuse, HeadingPF, NativeCartesianPF
from pf_ctrv import PFCTRV

OLD=HERE.parent/'review123_pf100'
job.HERE=HERE
job.CFG=OLD/'rbpf100.yaml'
job.installed=installed


def preflight():
    assert not (HERE/'inputs.json').exists()
    prior=json.loads((OLD/'inputs.json').read_text())
    for p,h in prior['sha256'].items():assert job.base.sha(Path(p))==h,p
    paths=[Path(p) for p in prior['sha256']]
    paths += [Path(__file__),HERE/'heading.py',HERE/'00_protocol.md',
              HERE.parent/'review126_pf_efficiency/fast.py',
              HERE.parent/'review126_pf_efficiency/native_pf.py',
              HERE.parent/'review126_pf_efficiency/native/kernel.c',
              HERE.parent/'review126_pf_efficiency/native/kernel.dll']
    paths += [OLD/f'dev/evaluation/cartesian-s{s}/raw_fields.json' for s in range(3)]
    baseline=[job.base.boot_stats.combine(job.base.boot_stats.load_raw_fields(OLD/f'dev/evaluation/cartesian-s{s}/raw_fields.json'),job.base.DEV) for s in range(3)]
    job.write(HERE/'inputs.json',{'frames':prior['frames'],'baseline_seeds':baseline,
              'sha256':{str(p):job.base.sha(p) for p in paths}})


def checks():
    rng=np.random.default_rng(128)
    nodes,weights=np.polynomial.hermite.hermgauss(64);weights/=np.sqrt(np.pi)
    max_error=0.
    for sigma in (0.,.01,.3,1.):
        m=rng.normal(size=(16,4))*10
        b=rng.normal(size=(16,4,4));p=b@b.transpose(0,2,1)+np.eye(4)[None]*.01
        actual_m,actual_p=m.copy(),p.copy();diffuse(actual_m,actual_p,sigma)
        expected_m=np.zeros_like(m);expected_second=np.zeros_like(p)
        second=p+m[:,:,None]*m[:,None,:]
        for node,weight in zip(nodes,weights):
            angle=np.sqrt(2)*sigma*node;c,s=np.cos(angle),np.sin(angle)
            matrix=np.eye(4);matrix[2:,2:]=[[c,-s],[s,c]]
            expected_m+=weight*(m@matrix.T)
            expected_second+=weight*(matrix@second@matrix.T)
        expected_p=expected_second-expected_m[:,:,None]*expected_m[:,None,:]
        assert np.allclose(actual_m,expected_m,atol=1e-10,rtol=1e-10)
        assert np.allclose(actual_p,expected_p,atol=1e-10,rtol=1e-10)
        assert np.linalg.eigvalsh(actual_p).min()>-1e-10
        max_error=max(max_error,float(abs(actual_p-expected_p).max()))
    # Zero heading diffusion must recover the already validated native backend.
    for seed in range(3):
        pf=PFCTRV(num_particles=100,std_theta=0.)
        mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
        particles=pf.sample_initial(mean,cov,np.random.default_rng(seed))
        a=NativeCartesianPF(pf,mean,cov,particles,np.random.default_rng(seed+10))
        b=HeadingPF(pf,mean,cov,particles,np.random.default_rng(seed+10))
        for t in range(40):
            for method,args in [('predict',(t%11<3,)),('update',(np.array([50+2*t,80+10*np.sin(t/10),1.5,70.]),))]:
                x,p=getattr(a,method)(*args);y,q=getattr(b,method)(*args)
                assert np.array_equal(x,y) and np.array_equal(p,q)
                assert a.rng.bit_generator.state==b.rng.bit_generator.state
    for seed in range(3):
        for video in job.base.SMOKE:
            observed,_=job.replay(video,seed,False,128)
            raw=np.loadtxt(OLD/f'dev/tracking/DETRAC-all/cartesian-s{seed}/data/{video}.txt',delimiter=',',ndmin=2)
            assert np.allclose(observed.to_numpy(float),raw[raw[:,0]<=128],atol=1e-9,rtol=0)
            x,_=job.replay(video,seed,True,128);y,_=job.replay(video,seed,True,128)
            assert np.array_equal(x.to_numpy(float),y.to_numpy(float))
    job.write(HERE/'smoke.json',{'status':'PASS','quadrature_max_cov_error':max_error,
            'zero_noise_steps':240,'parent_parity':6,'repeat':6})
    print('CHECKS PASS',max_error,flush=True)


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
    gate={'IDSW_vs_all':new['IDSW']<472,'two_seeds':sum(r['IDSW']<472 for r in scores)>=2,
          'HOTA':new['HOTA']>=old['HOTA']-.001,'FN':new['FN']<=old['FN']*1.002}
    job.write(HERE/'dev_summary.json',{'baseline':old,'candidate':new,'seeds':scores,'gates':gate,'pass':all(gate.values())})
    job.write(HERE/'verification.json',{'status':'PASS','units':units,'hashes':len(frozen['sha256'])})
    print('SUMMARY',new,gate,flush=True)


if __name__=='__main__':
    if sys.argv[1]=='dev':job.dev(int(sys.argv[2]))
    else:{'preflight':preflight,'checks':checks,'summary':summary}[sys.argv[1]]()
