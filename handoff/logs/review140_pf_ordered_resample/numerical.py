from pathlib import Path
import sys
import json
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'src'))
from ordered import ordered_selection,OrderedPF,PredictedNoisePF
from pf_ctrv import PFCTRV


def scalar(weights,omega,uniforms):
    n=len(weights);order=sorted(range(n),key=lambda i:omega[i]);total=sum(weights)
    output=[]
    for child,u in enumerate(uniforms):
        target=(child+u)/n;mass=0.
        for index in order:
            mass+=weights[index]/total
            if mass>target:
                output.append(index);break
        else:output.append(order[-1])
    return np.array(output)


def make(cls,seed):
    pf=PFCTRV(num_particles=100,resample_ess_ratio=.9)
    mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
    particles=pf.sample_initial(mean,cov,np.random.default_rng(seed))
    return cls(pf,mean,cov,particles,np.random.default_rng(seed+10))


def main():
    rng=np.random.default_rng(140);max_expected_error=0.
    for _ in range(500):
        n=int(rng.integers(2,101));weights=np.exp(rng.normal(0,2,n));omega=rng.normal(size=n)
        uniforms=rng.uniform(.001,.999,n);original=[a.copy() for a in (weights,omega,uniforms)]
        selected=ordered_selection(weights,omega,uniforms)
        assert np.array_equal(selected,scalar(weights,omega,uniforms))
        permutation=rng.permutation(n)
        assert np.array_equal(selected,permutation[ordered_selection(weights[permutation],omega[permutation],uniforms)])
        assert all(np.array_equal(a,b) for a,b in zip(original,(weights,omega,uniforms)))
        order=np.argsort(omega);w=weights[order]/weights.sum();bounds=np.r_[0,np.cumsum(w)];bounds[-1]=1
        expected=np.zeros(n)
        for particle in range(n):
            for child in range(n):
                expected[particle]+=n*max(0.,min(bounds[particle+1],(child+1)/n)-max(bounds[particle],child/n))
        discrepancy=float(np.max(np.abs(expected-n*w)));max_expected_error=max(max_expected_error,discrepancy)
        assert discrepancy<1e-11
    assert np.array_equal(ordered_selection([0,0,1],[0,1,2],[0,0,0]),[2,2,2])
    for n in (2,7,100):
        omega=rng.normal(size=n);selected=ordered_selection(np.ones(n),omega,np.full(n,.5))
        assert np.array_equal(np.sort(selected),np.arange(n))
    steps=resamples=0
    for seed in range(3):
        a=make(PredictedNoisePF,seed);b=make(OrderedPF,seed);b.ordered_stratified=False
        for t in range(100):
            for method,args in [('predict',(t%11<3,)),('update',(np.array([50+6*t,80+10*np.sin(t/8),1.5,70.+np.sin(t/9)]),))]:
                x,p=getattr(a,method)(*args);y,q=getattr(b,method)(*args)
                assert np.array_equal(x,y) and np.array_equal(p,q)
                for field in ('means','covariances','omega','logw','shape_mean','shape_cov'):
                    assert np.array_equal(getattr(a,field),getattr(b,field)),field
                assert a.rng.bit_generator.state==b.rng.bit_generator.state;steps+=1
            assert a.resamples==b.resamples and a.updates==b.updates
        resamples+=a.resamples
        c=make(OrderedPF,seed)
        for t in range(100):
            c.predict(t%11<3)
            mean,cov=c.update(np.array([50+6*t,80+10*np.sin(t/8),1.5,70.+np.sin(t/9)]))
            assert np.isfinite(mean).all() and np.linalg.eigvalsh(cov).min()>-1e-6
            assert np.linalg.eigvalsh(c.covariances).min()>-1e-9
    assert resamples>0
    report={'status':'PASS','selection_cases':500,'permutation_equivariance':True,
            'max_expected_offspring_error':max_expected_error,'control_steps':steps,
            'control_resamples':resamples,'uniform_and_zero_weights':True,'ownership':True,'PSD':True}
    (HERE/'numerical.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NUMERICAL PASS',report,flush=True)


if __name__=='__main__':main()
