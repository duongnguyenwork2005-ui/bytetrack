from pathlib import Path
import sys
import json
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'src'))
from lost_shape import LostShapePF,PredictedNoisePF
from pf_ctrv import PFCTRV


def make(cls,seed):
    pf=PFCTRV(num_particles=100)
    mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
    particles=pf.sample_initial(mean,cov,np.random.default_rng(seed))
    model=cls(pf,mean,cov,particles,np.random.default_rng(seed+10))
    model.initialize_motion(mean,cov,np.array([58.,82.,1.5,70.]),1,np.array([50.,80.]))
    return model


def full_parity(a,b,aa,bb):
    assert all(np.array_equal(x,y) for x,y in zip(aa,bb))
    for field in ('means','covariances','omega','logw','shape_mean','shape_cov'):
        assert np.array_equal(getattr(a,field),getattr(b,field)),field
    assert a.rng.bit_generator.state==b.rng.bit_generator.state
    assert a.resamples==b.resamples and a.updates==b.updates


def main():
    steps=lost_steps=0;maxerr=0.;rows=[]
    for seed in range(3):
        a=make(PredictedNoisePF,seed);b=make(LostShapePF,seed)
        for t in range(100):
            full_parity(a,b,a.predict(False),b.predict(False));steps+=1
            z=np.array([50+8*t,80+10*np.sin(t/8),1.5,70.+.5*t])
            full_parity(a,b,a.update(z),b.update(z));steps+=1
        assert b.shape_cov[1,1,1]>0
        mean=b.shape_mean[1].copy();cov=b.shape_cov[1].copy()
        matrix=np.array([[1.,b.pf.dt],[0.,1.]])
        for _ in range(12):
            height=max(float(mean[0]),1e-3);mean[1]=0.;mean=matrix@mean
            cov=matrix@cov@matrix.T+np.diag([(b.pf._std_weight_position*height)**2,
                                          (b.pf._std_weight_velocity*height)**2])
            aa,ap=a.predict(True);bb,bp=b.predict(True)
            maxerr=max(maxerr,float(np.max(np.abs(b.shape_cov[1]-cov))))
            assert np.allclose(b.shape_mean[1],mean,atol=1e-12,rtol=0)
            assert np.allclose(b.shape_cov[1],cov,atol=1e-10,rtol=1e-12)
            assert np.array_equal(aa,bb)
            for field in ('means','covariances','omega','logw'):
                assert np.array_equal(getattr(a,field),getattr(b,field)),field
            assert a.rng.bit_generator.state==b.rng.bit_generator.state
            assert np.linalg.eigvalsh(b.shape_cov[1]).min()>0
            assert np.linalg.eigvalsh(bp).min()>-1e-6
            lost_steps+=1
        legacy_var=float(a.shape_cov[1,0,0]);retained_var=float(b.shape_cov[1,0,0])
        assert retained_var>legacy_var
        z=np.array([50+8*112,80.,1.5,float(b.shape_mean[1,0])+20.])
        a.update(z);b.update(z)
        rows.append({'seed':seed,'legacy_lost_h_var':legacy_var,'retained_lost_h_var':retained_var,
                     'legacy_h_error':abs(float(a.shape_mean[1,0])-z[3]),
                     'retained_h_error':abs(float(b.shape_mean[1,0])-z[3])})
        assert rows[-1]['retained_h_error']<rows[-1]['legacy_h_error']
        detached=b.full_particles();detached[:]=0
        assert not np.array_equal(detached,b.full_particles())
    report={'status':'PASS','not_lost_parity_steps':steps,'lost_oracle_steps':lost_steps,
            'max_shape_covariance_error':maxerr,'RNG_unchanged':True,'PSD':True,'ownership':True,
            'toy_reacquisition':rows,'note':'Toy height recovery is not a tracking superiority claim.'}
    (HERE/'numerical.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NUMERICAL PASS',report,flush=True)


if __name__=='__main__':main()
