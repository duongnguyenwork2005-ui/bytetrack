"""Research R-height variant using the existing same-input UA development gate."""
from pathlib import Path
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review129_pf_soft_init'))
spec=importlib.util.spec_from_file_location('r129_ua_harness',HERE.parent/'review129_pf_soft_init/run_job.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import numpy as np
from predicted_noise import installed,PredictedNoisePF,NativeCartesianPF
from pf_ctrv import PFCTRV
import rbpf_ctrv

base.HERE=HERE;base.job.HERE=HERE;base.job.installed=installed
job=base.job


def preflight():
    assert not (HERE/'inputs.json').exists()
    prior=json.loads((HERE.parent/'review129_pf_soft_init/inputs.json').read_text())
    for p,h in prior['sha256'].items():assert job.base.sha(Path(p))==h,p
    paths=[Path(p) for p in prior['sha256']]+[Path(__file__),HERE/'predicted_noise.py',HERE/'00_protocol.md']
    job.write(HERE/'inputs.json',{'frames':prior['frames'],'baseline_seeds':prior['baseline_seeds'],
             'effective_measurement_noise_height':'prediction','sha256':{str(p):job.base.sha(p) for p in paths}})


def checks():
    for seed in range(3):
        pf=PFCTRV(num_particles=100)
        mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
        particles=pf.sample_initial(mean,cov,np.random.default_rng(seed))
        a=NativeCartesianPF(pf,mean,cov,particles,np.random.default_rng(seed+10))
        b=PredictedNoisePF(pf,mean,cov,particles,np.random.default_rng(seed+10))
        for t in range(40):
            a.predict(t%11<3);b.predict(t%11<3)
            observation=np.array([50+t,80+np.sin(t/10),1.5,float(a.shape_mean[1,0])])
            x,p=a.update(observation);y,q=b.update(observation)
            assert np.array_equal(x,y) and np.array_equal(p,q)
            assert a.rng.bit_generator.state==b.rng.bit_generator.state
            assert '_measurement_noise' not in pf.__dict__
            assert np.linalg.eigvalsh(q).min()>-1e-8
        try:b.update(np.array([1.,2.]))
        except IndexError:pass
        else:raise AssertionError('invalid observation should raise')
        assert '_measurement_noise' not in pf.__dict__
        for height in (1.,140.,20.,70.):
            x,p=b.update(np.array([50.,80.,1.5,height]))
            assert np.isfinite(x).all() and np.linalg.eigvalsh(p).min()>-1e-8
            assert '_measurement_noise' not in pf.__dict__
    previous=rbpf_ctrv.ConditionalPF
    with installed():assert rbpf_ctrv.ConditionalPF is PredictedNoisePF
    assert rbpf_ctrv.ConditionalPF is previous
    for seed in range(3):
        for video in job.base.SMOKE:
            observed,_=job.replay(video,seed,False,128)
            ref=np.loadtxt(base.OLD/f'dev/tracking/DETRAC-all/cartesian-s{seed}/data/{video}.txt',delimiter=',',ndmin=2)
            assert np.allclose(observed.to_numpy(float),ref[ref[:,0]<=128],atol=1e-9,rtol=0)
            x,_=job.replay(video,seed,True,128);y,_=job.replay(video,seed,True,128)
            assert np.array_equal(x.to_numpy(float),y.to_numpy(float))
    job.write(HERE/'smoke.json',{'status':'PASS','equal_height_steps':240,'PSD':True,'restore_on_exception':True,
             'hook_restore':True,'parent_parity':6,'repeat':6})
    print('CHECKS PASS',flush=True)


if __name__=='__main__':
    if sys.argv[1]=='dev':job.dev(int(sys.argv[2]))
    else:{'preflight':preflight,'checks':checks,'summary':base.summary}[sys.argv[1]]()
