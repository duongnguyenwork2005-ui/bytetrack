from pathlib import Path
import sys
import json
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'src'))
from moment_resample import corrected_cloud,MomentResamplePF,PredictedNoisePF
from pf_ctrv import PFCTRV,systematic_resample


def independent(weights,omega,means,covs):
    weights=weights/weights.sum();points=np.column_stack((omega,means))
    mu=sum(float(w)*point for w,point in zip(weights,points))
    covariance=np.zeros((5,5))
    for w,point,cov in zip(weights,points,covs):
        delta=point-mu;term=np.outer(delta,delta);term[1:,1:]+=cov
        covariance+=w*term
    return mu,covariance


def make(cls,seed,ratio=.9):
    pf=PFCTRV(num_particles=100,resample_ess_ratio=ratio)
    mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
    particles=pf.sample_initial(mean,cov,np.random.default_rng(seed))
    return cls(pf,mean,cov,particles,np.random.default_rng(seed+10))


def main():
    rng=np.random.default_rng(135);maxmean=maxcov=0.
    for _ in range(500):
        weights=np.exp(rng.normal(0,2,100));weights/=weights.sum()
        omega=rng.normal(0,.2,100)
        means=rng.normal(size=(100,4))*np.array([10,10,3,3])
        means+=omega[:,None]*np.array([10,15,3,-2])
        matrix=rng.normal(size=(100,4,4))
        covs=matrix@matrix.transpose(0,2,1)+np.eye(4)*.1
        original=[a.copy() for a in (weights,omega,means,covs)]
        selected=systematic_resample(weights,float(rng.random()))
        ow,om,op,applied=corrected_cloud(weights,omega,means,covs,selected)
        assert applied
        mu,cov=independent(weights,omega,means,covs)
        new_mu,new_cov=independent(np.ones(100),ow,om,op)
        maxmean=max(maxmean,float(np.max(np.abs(mu-new_mu))))
        maxcov=max(maxcov,float(np.max(np.abs(cov-new_cov))))
        assert np.allclose(mu,new_mu,atol=1e-11,rtol=1e-12)
        assert np.allclose(cov,new_cov,atol=1e-10,rtol=1e-12)
        assert np.linalg.eigvalsh(op).min()>0
        assert all(np.array_equal(a,b) for a,b in zip(original,(weights,omega,means,covs)))
        assert all(a.flags.c_contiguous for a in (ow,om,op))
    for omega in (np.zeros(100),np.ones(100)*.2):
        selected=np.arange(100)[::-1]
        a,b,c,applied=corrected_cloud(weights,omega,means,covs,selected)
        assert not applied and np.array_equal(a,omega[selected]) and np.array_equal(b,means[selected]) and np.array_equal(c,covs[selected])
    selected=np.zeros(100,dtype=int)
    assert not corrected_cloud(weights,rng.normal(size=100),means,covs,selected)[3]
    total_resamples=corrected_steps=0
    for seed in range(3):
        a=make(PredictedNoisePF,seed);b=make(MomentResamplePF,seed);b.correct_moments=False
        for t in range(120):
            z=np.array([50+6*t,80+10*np.sin(t/8),1.5,70.+np.sin(t/9)])
            for method,args in [('predict',(t%11<3,)),('update',(z,))]:
                x,p=getattr(a,method)(*args);y,q=getattr(b,method)(*args)
                assert np.array_equal(x,y) and np.array_equal(p,q)
                for field in ('omega','means','covariances','shape_mean','shape_cov','logw'):
                    assert np.array_equal(getattr(a,field),getattr(b,field)),field
                assert a.rng.bit_generator.state==b.rng.bit_generator.state
                assert a.pf.resample_ess_ratio==b.pf.resample_ess_ratio==.9
            assert a.resamples==b.resamples and a.updates==b.updates
        total_resamples+=a.resamples
        c=make(MomentResamplePF,seed)
        for t in range(120):
            c.predict(t%11<3)
            mean,cov=c.update(np.array([50+6*t,80+10*np.sin(t/8),1.5,70.+np.sin(t/9)]))
            assert np.isfinite(mean).all() and np.linalg.eigvalsh(cov).min()>-1e-7
            assert np.linalg.eigvalsh(c.covariances).min()>-1e-9
        corrected_steps+=c.corrected_resamples
    assert total_resamples>0 and corrected_steps>0
    bad=make(MomentResamplePF,0)
    try:bad.update(np.array([1.,2.,3.]))
    except (ValueError,IndexError):pass
    else:raise AssertionError('Bad shape was accepted')
    assert bad.pf.resample_ess_ratio==.9
    report={'status':'PASS','mixtures':500,'max_mean_error':maxmean,'max_covariance_error':maxcov,
            'PSD':True,'ownership':True,'degenerate_cases':3,'control_steps':720,
            'parent_resamples':total_resamples,'corrected_resamples_toy':corrected_steps,
            'no_extra_rng_control':True,'restore_on_exception':True}
    (HERE/'numerical.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NUMERICAL PASS',report,flush=True)


if __name__=='__main__':main()
