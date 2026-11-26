from pathlib import Path
import sys
import json
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'src'))
from extent import ExtentTrendPF,PredictedNoisePF,damped_transition
from pf_ctrv import PFCTRV


def make(cls,seed,horizon=30):
    pf=PFCTRV(num_particles=100)
    mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
    particles=pf.sample_initial(mean,cov,np.random.default_rng(seed))
    model=cls(pf,mean,cov,particles,np.random.default_rng(seed+10))
    model.horizon=horizon
    model.initialize_motion(mean,cov,np.array([58.,82.,1.5,70.]),1,np.array([50.,80.]))
    return model


def parity(a,b,aa,bb):
    assert all(np.array_equal(x,y) for x,y in zip(aa,bb))
    for field in ('means','covariances','omega','logw','shape_mean','shape_cov'):
        assert np.array_equal(getattr(a,field),getattr(b,field)),field
    assert a.rng.bit_generator.state==b.rng.bit_generator.state


def main():
    rng=np.random.default_rng(139);error=0.
    for _ in range(500):
        dt,s=rng.uniform(.01,2,2);horizon=float(np.exp(rng.normal()))
        f=damped_transition(dt,horizon);g=damped_transition(s,horizon)
        error=max(error,float(np.max(np.abs(damped_transition(dt+s,horizon)-f@g))))
        assert np.allclose(damped_transition(dt+s,horizon),f@g,atol=1e-12,rtol=1e-12)
    assert np.array_equal(damped_transition(1,0),np.diag([1.,0.]))
    assert np.allclose(damped_transition(1,1e-14),np.diag([1.,0.]),atol=1e-12)
    assert np.allclose(damped_transition(1,1e15),np.array([[1.,1.],[0.,1.]]),atol=1e-12)
    not_lost=zero_steps=lost_steps=0
    for seed in range(3):
        a=make(PredictedNoisePF,seed);b=make(ExtentTrendPF,seed)
        for t in range(80):
            parity(a,b,a.predict(False),b.predict(False));not_lost+=1
            z=np.array([50+8*t,80+10*np.sin(t/8),1.5,70+.5*t])
            parity(a,b,a.update(z),b.update(z));not_lost+=1
        for horizon in (6.,30.):
            c=make(ExtentTrendPF,seed,horizon)
            c.shape_mean[1,1]=1.
            m=c.shape_mean[1].copy();p=c.shape_cov[1].copy()
            for _ in range(12):
                height=max(float(m[0]),1e-3);matrix=damped_transition(c.pf.dt,horizon)
                m=matrix@m;m[0]=max(m[0],1e-3)
                p=matrix@p@matrix.T+np.diag([(c.pf._std_weight_position*height)**2,
                                            (c.pf._std_weight_velocity*height)**2])
                _,cov=c.predict(True)
                assert np.allclose(c.shape_mean[1],m,atol=1e-12,rtol=0)
                assert np.allclose(c.shape_cov[1],p,atol=1e-10,rtol=1e-12)
                assert np.linalg.eigvalsh(cov).min()>-1e-6 and np.linalg.eigvalsh(p).min()>0
                lost_steps+=1
        a=make(PredictedNoisePF,seed);b=make(ExtentTrendPF,seed,0.)
        for t in range(60):
            parity(a,b,a.predict(t%3==0),b.predict(t%3==0));zero_steps+=1
            z=np.array([50+8*t,80+10*np.sin(t/8),1.5,70+.5*t])
            parity(a,b,a.update(z),b.update(z));zero_steps+=1
        detached=b.full_particles();detached[:]=0
        assert not np.array_equal(detached,b.full_particles())
    report={'status':'PASS','semigroup_cases':500,'max_F_semigroup_error':error,
            'not_lost_parity_steps':not_lost,'zero_horizon_parity_steps':zero_steps,
            'lost_oracle_steps':lost_steps,'RNG_unchanged':True,'PSD':True,'ownership':True,
            'horizons':[6,30],'limits_zero_infinity':True}
    (HERE/'numerical.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NUMERICAL PASS',report,flush=True)


if __name__=='__main__':main()
