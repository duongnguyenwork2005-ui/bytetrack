from pathlib import Path
import sys
import json
import math
import numpy as np
from scipy.special import ndtri

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'src'))
from median import marginal_median,MedianPF,PredictedNoisePF
from pf_ctrv import PFCTRV


def make(cls,seed):
    pf=PFCTRV(num_particles=100)
    mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
    particles=pf.sample_initial(mean,cov,np.random.default_rng(seed))
    return cls(pf,mean,cov,particles,np.random.default_rng(seed+10))


def main():
    rng=np.random.default_rng(136);error=0.
    for _ in range(500):
        weights=np.exp(rng.normal(size=100));weights/=weights.sum()
        means=rng.normal(0,20,100);variance=np.exp(rng.normal(size=100))
        original=[a.copy() for a in (weights,means,variance)]
        value=marginal_median(weights,means,variance)
        cdf=sum(float(w)*(.5+.5*math.erf((value-m)/math.sqrt(2*v)))
                for w,m,v in zip(weights,means,variance))
        error=max(error,abs(cdf-.5));assert abs(cdf-.5)<1e-10
        assert means.min()<=value<=means.max()
        assert all(np.array_equal(a,b) for a,b in zip(original,(weights,means,variance)))
    assert marginal_median([1],[3],[2])==3
    assert abs(marginal_median([.5,.5],[-3,3],[1,1]))<1e-10
    expected=float(ndtri(.5/.99))
    assert abs(marginal_median([.99,.01],[0,100],[1,1])-expected)<1e-10
    assert marginal_median([.75,.25],[0,10],[0,0])==0
    assert abs(marginal_median([.25,.75],[0,10],[0,0])-10)<1e-12
    steps=0;different=0;max_risk_error=0.
    for seed in range(3):
        a=make(PredictedNoisePF,seed);b=make(MedianPF,seed)
        for t in range(60):
            for method,args in [('predict',(t%11<3,)),('update',(np.array([50+3*t,80+10*np.sin(t/8),1.5,70.+np.sin(t/9)]),))]:
                mean,cov=getattr(a,method)(*args);action,risk=getattr(b,method)(*args)
                for field in ('means','covariances','omega','logw','shape_mean','shape_cov'):
                    assert np.array_equal(getattr(a,field),getattr(b,field)),field
                assert a.rng.bit_generator.state==b.rng.bit_generator.state
                assert np.array_equal(mean[2:],action[2:])
                shift=mean-action
                discrepancy=float(np.max(np.abs(risk-cov-np.outer(shift,shift))))
                max_risk_error=max(max_risk_error,discrepancy);assert discrepancy<1e-10
                assert np.isfinite(action).all() and np.linalg.eigvalsh(risk).min()>-1e-7
                different+=int(np.max(np.abs(shift[:2]))>1e-8);steps+=1
    assert different>0
    report={'status':'PASS','mixtures':500,'max_independent_cdf_error':error,'kernel_parity_steps':steps,
            'different_actions':different,'max_risk_error':max_risk_error,'PSD':True,'ownership':True,
            'symmetric_single_outlier_dirac':True,'RNG_unchanged':True}
    (HERE/'numerical.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NUMERICAL PASS',report,flush=True)


if __name__=='__main__':main()
