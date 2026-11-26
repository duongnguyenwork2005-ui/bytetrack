from pathlib import Path
import sys
import json
import copy
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'src'))
from deferred import DeferredOmegaPF,PredictedNoisePF
from pf_ctrv import PFCTRV,systematic_resample


def make(cls,seed,ratio=.5,std=.005):
    pf=PFCTRV(num_particles=100,resample_ess_ratio=ratio,std_omega=std)
    mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
    particles=pf.sample_initial(mean,cov,np.random.default_rng(seed))
    model=cls(pf,mean,cov,particles,np.random.default_rng(seed+10))
    model.initialize_motion(mean,cov,np.array([58.,82.,1.5,70.]),1,np.array([50.,80.]))
    return model


def compare(a,b,result_a,result_b):
    assert all(np.array_equal(x,y) for x,y in zip(result_a,result_b))
    for field in ('means','covariances','logw','shape_mean','shape_cov'):
        assert np.array_equal(getattr(a,field),getattr(b,field)),field
    assert np.array_equal(a.full_particles(),b.full_particles())
    assert a.rng.bit_generator.state==b.rng.bit_generator.state
    assert a.updates==b.updates and a.resamples==b.resamples


def main():
    steps=lost_steps=zero_steps=0;zero_resamples=0
    for seed in range(3):
        a=make(PredictedNoisePF,seed,ratio=0);b=make(DeferredOmegaPF,seed,ratio=0)
        for t in range(60):
            compare(a,b,a.predict(t%11<3),b.predict(t%11<3));steps+=1
            z=np.array([50+8*t,80+10*np.sin(t/8),1.5,70.+np.sin(t/9)])
            compare(a,b,a.update(z),b.update(z));steps+=1
            assert np.array_equal(a.omega,b.omega) and b._pending_noise is None
        a=make(PredictedNoisePF,seed);b=make(DeferredOmegaPF,seed)
        for _ in range(40):
            compare(a,b,a.predict(True),b.predict(True));lost_steps+=1
        a=make(PredictedNoisePF,seed,ratio=.9,std=0);b=make(DeferredOmegaPF,seed,ratio=.9,std=0)
        for t in range(80):
            compare(a,b,a.predict(False),b.predict(False));zero_steps+=1
            z=np.array([50+8*t,80+10*np.sin(t/8),1.5,70.+np.sin(t/9)])
            compare(a,b,a.update(z),b.update(z));zero_steps+=1
            assert np.array_equal(a.omega,b.omega)
        zero_resamples+=a.resamples
    # One forced-selection event, independently reconstruct the expected descendants.
    a=make(PredictedNoisePF,71,ratio=0);b=make(DeferredOmegaPF,71,ratio=1)
    old=a.omega.copy()
    compare(a,b,a.predict(False),b.predict(False))
    noise=b._pending_noise.copy()
    selection_rng=np.random.default_rng();selection_rng.bit_generator.state=copy.deepcopy(a.rng.bit_generator.state)
    observation=np.array([500.,100.,1.5,70.])
    expected_result=a.update(observation)
    selected=systematic_resample(np.exp(a.logw),float(selection_rng.random()))
    assert len(set(selected.tolist()))<100
    actual=b.update(observation)
    assert b.resamples==1
    assert all(np.array_equal(x,y) for x,y in zip(expected_result,actual))
    assert np.array_equal(b.omega,old[selected]+noise)
    assert not np.array_equal(b.omega,(old+noise)[selected])
    assert np.array_equal(b.means,a.means[selected]) and np.array_equal(b.covariances,a.covariances[selected])
    assert b.rng.bit_generator.state==selection_rng.bit_generator.state
    assert b._pending_noise is None and np.all(b.logw==-np.log(100))
    detached=b.full_particles();detached[:]=0
    assert not np.array_equal(detached,b.full_particles())
    assert np.linalg.eigvalsh(b.covariances).min()>-1e-9
    assert zero_resamples>0
    report={'status':'PASS','no_resample_parity_steps':steps,'lost_no_observation_steps':lost_steps,
            'zero_noise_parity_steps':zero_steps,'zero_noise_resamples':zero_resamples,
            'forced_selection_formula':True,'RNG_order_unchanged':True,'ownership':True,'PSD':True,
            'forced_ancestors':len(set(selected.tolist()))}
    (HERE/'numerical.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NUMERICAL PASS',report,flush=True)


if __name__=='__main__':main()
