"""Reuse R129's verified scoring gate with a distinct frozen R130 process model."""
from pathlib import Path
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review129_pf_soft_init'))
spec=importlib.util.spec_from_file_location('r129_ua_harness',HERE.parent/'review129_pf_soft_init/run_job.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import numpy as np
from polar_noise import installed,direction_tensor,PolarNoisePF,NativeCartesianPF
from pf_ctrv import PFCTRV

base.HERE=HERE;base.job.HERE=HERE;base.job.installed=installed
job=base.job


def preflight():
    assert not (HERE/'inputs.json').exists()
    prior=json.loads((HERE.parent/'review129_pf_soft_init/inputs.json').read_text())
    for p,h in prior['sha256'].items():assert job.base.sha(Path(p))==h,p
    paths=[Path(p) for p in prior['sha256']]
    paths += [Path(__file__),HERE/'polar_noise.py',HERE/'00_protocol.md',
              HERE.parent/'review128_pf_heading/heading.py']
    job.write(HERE/'inputs.json',{'frames':prior['frames'],'baseline_seeds':prior['baseline_seeds'],
             'sha256':{str(p):job.base.sha(p) for p in paths}})


def checks():
    rng=np.random.default_rng(130)
    m=rng.normal(size=(1000,2));b=rng.normal(size=(1000,2,2));p=b@b.transpose(0,2,1)
    tensor=direction_tensor(m,p)
    assert np.allclose(np.trace(tensor,axis1=1,axis2=2),1.,atol=1e-14)
    assert np.linalg.eigvalsh(tensor).min()>-1e-14
    known=direction_tensor(np.array([[3.,4.]]),np.zeros((1,2,2)))
    assert np.allclose(known[0],np.array([[9.,12.],[12.,16.]])/25.)
    for scale in (0.,.1,100.):
        isotropic=direction_tensor(np.zeros((2,2)),np.broadcast_to(np.eye(2)*scale,(2,2,2)))
        assert np.array_equal(isotropic,np.broadcast_to(np.eye(2)*.5,(2,2,2)))
    for seed in range(3):
        pf=PFCTRV(num_particles=100,std_theta=0.)
        pf._std_weight_velocity=0.
        mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
        particles=pf.sample_initial(mean,cov,np.random.default_rng(seed))
        a=NativeCartesianPF(pf,mean,cov,particles,np.random.default_rng(seed+10))
        b=PolarNoisePF(pf,mean,cov,particles,np.random.default_rng(seed+10))
        for t in range(40):
            for method,args in [('predict',(t%11<3,)),('update',(np.array([50+t,80+np.sin(t/10),1.5,70.]),))]:
                x,p=getattr(a,method)(*args);y,q=getattr(b,method)(*args)
                assert np.array_equal(x,y) and np.array_equal(p,q)
                assert a.rng.bit_generator.state==b.rng.bit_generator.state
    # Non-zero model remains finite/PSD through repeated missing observations.
    pf=PFCTRV(num_particles=100)
    mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
    particles=pf.sample_initial(mean,cov,rng)
    a=PolarNoisePF(pf,mean,cov,particles,rng)
    for t in range(100):
        x,p=a.predict(t%11<3)
        assert np.isfinite(x).all() and np.linalg.eigvalsh(p).min()>-1e-8
        x,p=a.update(np.array([50+t,80+np.sin(t/10),1.5,70.]))
        assert np.isfinite(x).all() and np.linalg.eigvalsh(p).min()>-1e-8
    for seed in range(3):
        for video in job.base.SMOKE:
            observed,_=job.replay(video,seed,False,128)
            ref=np.loadtxt(base.OLD/f'dev/tracking/DETRAC-all/cartesian-s{seed}/data/{video}.txt',delimiter=',',ndmin=2)
            assert np.allclose(observed.to_numpy(float),ref[ref[:,0]<=128],atol=1e-9,rtol=0)
            x,_=job.replay(video,seed,True,128);y,_=job.replay(video,seed,True,128)
            assert np.array_equal(x.to_numpy(float),y.to_numpy(float))
    job.write(HERE/'smoke.json',{'status':'PASS','direction_cases':1000,'trace_PSD_limits':True,
             'zero_noise_native_steps':240,'nonzero_PSD_steps':200,'parent_parity':6,'repeat':6})
    print('CHECKS PASS',flush=True)


if __name__=='__main__':
    if sys.argv[1]=='dev':job.dev(int(sys.argv[2]))
    else:{'preflight':preflight,'checks':checks,'summary':base.summary}[sys.argv[1]]()
