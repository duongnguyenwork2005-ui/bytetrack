"""Independent finite-mixture posterior, one-component and safe-fallback checks."""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'src'))
from mixture import MixturePF,WEIGHTS,SCALES,PredictedNoisePF,lib
from pf_ctrv import PFCTRV


def build(cls,seed=0):
    pf=PFCTRV(num_particles=100,resample_ess_ratio=1e-12)
    mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
    points=pf.sample_initial(mean,cov,np.random.default_rng(seed))
    return cls(pf,mean,cov,points,np.random.default_rng(seed+10))


def main():
    assert (WEIGHTS>0).all() and (SCALES>0).all()
    assert abs(WEIGHTS.sum()-1)<1e-14 and abs(WEIGHTS@SCALES-1)<1e-14
    kurtosis=float(3*(WEIGHTS@(SCALES*SCALES)));assert kurtosis>3
    rng=np.random.default_rng(133)
    a=build(MixturePF);n=a.n
    means=rng.normal(size=(n,4))*2+np.array([50.,80.,1.,-2.])
    b=rng.normal(size=(n,4,4));cov=b@b.transpose(0,2,1)+np.eye(4)[None]*.3
    a.means=means.copy();a.covariances=cov.copy()
    z=np.array([52.,79.,1.5,70.]);noise=a.pf._measurement_noise(70.,None)
    all_m=[];all_p=[];terms=[]
    for weight,scale in zip(WEIGHTS,SCALES):
        rr=noise[:2,:2]*scale;s=cov[:,:2,:2]+rr
        gain=np.linalg.solve(s,cov[:,:,:2].transpose(0,2,1)).transpose(0,2,1)
        updated=means+np.einsum('nij,nj->ni',gain,z[:2]-means[:,:2])
        pp=cov-gain@s@gain.transpose(0,2,1)
        all_m.append(updated);all_p.append(pp)
        terms.append([np.log(weight)+multivariate_normal.logpdf(z[:2],mean=means[i,:2],cov=s[i]) for i in range(n)])
    terms=np.array(terms);ll=logsumexp(terms,axis=0);responsibility=np.exp(terms-ll)
    expected_m=np.zeros((n,4));expected_p=np.zeros((n,4,4))
    for i in range(n):
        expected_m[i]=sum(responsibility[j,i]*all_m[j][i] for j in range(3))
        expected_p[i]=sum(responsibility[j,i]*(all_p[j][i]+np.outer(all_m[j][i]-expected_m[i],all_m[j][i]-expected_m[i])) for j in range(3))
    expected_logw=ll-logsumexp(ll)
    state=a.rng.bit_generator.state
    x,p=a.update(z)
    assert np.allclose(a.means,expected_m,atol=1e-10,rtol=1e-10)
    assert np.allclose(a.covariances,expected_p,atol=1e-10,rtol=1e-10)
    assert np.allclose(a.logw,expected_logw,atol=1e-10,rtol=1e-10)
    assert a.rng.bit_generator.state==state and a.mixture_fallbacks==0
    assert np.linalg.eigvalsh(a.covariances).min()>-1e-8 and np.linalg.eigvalsh(p).min()>-1e-8
    assert a.means.flags['C_CONTIGUOUS'] and a.covariances.flags['C_CONTIGUOUS']
    error=float(np.max(abs(a.covariances-expected_p)))
    class One(MixturePF):
        weights=np.array([1.]);scales=np.array([1.])
    for seed in range(3):
        a=build(PredictedNoisePF,seed);b=build(One,seed)
        for t in range(40):
            for method,args in [('predict',(t%11<3,)),('update',(np.array([50+t,80+np.sin(t/10),1.5,70.+np.sin(t/9)]),))]:
                x,p=getattr(a,method)(*args);y,q=getattr(b,method)(*args)
                assert np.allclose(x,y,atol=1e-9,rtol=1e-9)
                assert np.allclose(p,q,atol=1e-8,rtol=1e-8)
                assert a.rng.bit_generator.state==b.rng.bit_generator.state
                assert np.linalg.eigvalsh(q).min()>-1e-8
    # Reject the second component after the first has written scratch only.
    a=build(PredictedNoisePF);b=build(MixturePF);z=np.array([51.,81.,1.5,70.])
    expected=a.update(z);original=lib.update;calls=0
    def force_late_fallback(*args):
        nonlocal calls
        calls+=1
        return 1 if calls==2 else original(*args)
    lib.update=force_late_fallback
    try:actual=b.update(z)
    finally:lib.update=original
    assert b.mixture_fallbacks==1 and calls==3
    assert all(np.allclose(x,y,atol=1e-10,rtol=1e-10) for x,y in zip(expected,actual))
    assert np.allclose(a.means,b.means,atol=1e-10,rtol=1e-10)
    for bad in (np.array([1.,2.]),np.array([1.,2.,1.,np.nan])):
        try:b.update(bad)
        except ValueError:pass
        else:raise AssertionError('invalid observation reached the C API')
    data={'status':'PASS','conditional_cases':100,'max_covariance_error':error,'PSD':True,
          'one_component_steps':240,'late_component_fallback':True,'C_shape_guard':True,
          'mixture_weights':WEIGHTS.tolist(),'mixture_scales':SCALES.tolist(),'marginal_kurtosis':kurtosis}
    (HERE/'numerical.json').write_text(json.dumps(data,indent=2)+'\n')
    print('NUMERICAL PASS',error,kurtosis,flush=True)


if __name__=='__main__':main()
