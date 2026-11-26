"""Fixed OU calibration transfer into the existing UA development harness."""
from pathlib import Path
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review129_pf_soft_init'))
spec=importlib.util.spec_from_file_location('r129_ua_harness',HERE.parent/'review129_pf_soft_init/run_job.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import numpy as np
from turn_prior import installed,coefficients,TurnPriorPF,NativeCartesianPF,PARAMETERS
from pf_ctrv import PFCTRV

base.HERE=HERE;base.job.HERE=HERE;base.job.installed=installed
job=base.job


def preflight():
    assert not (HERE/'inputs.json').exists()
    prior=json.loads((HERE.parent/'review129_pf_soft_init/inputs.json').read_text())
    calibration=json.loads((HERE/'calibration_inputs.json').read_text())
    hashes={**prior['sha256'],**calibration['sha256']}
    for p,h in hashes.items():assert job.base.sha(Path(p))==h,p
    paths=[Path(__file__),HERE/'turn_prior.py',HERE/'turn_prior.json',HERE/'calibration_inputs.json']
    hashes.update({str(p):job.base.sha(p) for p in paths})
    job.write(HERE/'inputs.json',{'frames':prior['frames'],'baseline_seeds':prior['baseline_seeds'],'sha256':hashes})


def checks():
    variance=PARAMETERS['stationary_variance_rad2_per_second2']
    r5,q5,i5=coefficients(.2);r25,q25,i25=coefficients(.04)
    for dt in (.04,.2):
        rho,q,initial=coefficients(dt)
        assert np.isclose(rho*rho*variance+(q/dt)**2,variance,rtol=1e-12)
        assert np.isclose(initial/dt,np.sqrt(variance),rtol=1e-12)
    assert np.isclose(r5,r25**5,rtol=1e-12)
    assert np.isclose((q5/.2)**2,(q25/.04)**2*sum(r25**(2*k) for k in range(5)),rtol=1e-12)
    for seed in range(3):
        pf=PFCTRV(num_particles=100)
        mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
        particles=pf.sample_initial(mean,cov,np.random.default_rng(seed))
        a=NativeCartesianPF(pf,mean,cov,particles,np.random.default_rng(seed+10))
        b=TurnPriorPF(pf,mean,cov,particles,np.random.default_rng(seed+10))
        assert np.allclose(b.omega,particles[:,pf.OMEGA]*i25/pf._init_std_omega)
        # Recover the native random walk to check the unchanged conditional kernel.
        b.omega=a.omega.copy();b.rho=1.;b.noise_std=pf._std_omega
        for t in range(40):
            for method,args in [('predict',(t%11<3,)),('update',(np.array([50+t,80+np.sin(t/10),1.5,70.]),))]:
                x,p=getattr(a,method)(*args);y,q=getattr(b,method)(*args)
                assert np.array_equal(x,y) and np.array_equal(p,q)
                assert a.rng.bit_generator.state==b.rng.bit_generator.state
    for seed in range(3):
        for video in job.base.SMOKE:
            observed,_=job.replay(video,seed,False,128)
            ref=np.loadtxt(base.OLD/f'dev/tracking/DETRAC-all/cartesian-s{seed}/data/{video}.txt',delimiter=',',ndmin=2)
            assert np.allclose(observed.to_numpy(float),ref[ref[:,0]<=128],atol=1e-9,rtol=0)
            x,_=job.replay(video,seed,True,128);y,_=job.replay(video,seed,True,128)
            assert np.array_equal(x.to_numpy(float),y.to_numpy(float))
    job.write(HERE/'smoke.json',{'status':'PASS','OU_moments_and_semigroup':True,
             'native_kernel_steps':240,'parent_parity':6,'repeat':6})
    print('CHECKS PASS',flush=True)


if __name__=='__main__':
    if sys.argv[1]=='dev':job.dev(int(sys.argv[2]))
    else:{'preflight':preflight,'checks':checks,'summary':base.summary}[sys.argv[1]]()
