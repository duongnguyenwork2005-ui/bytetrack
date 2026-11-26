"""Reuse R146's cohort/scoring safeguards, replacing only the PF factory."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;HELPER=HERE.parent/'review146_pf_shared_ordered'
sys.path.insert(0,str(HELPER))
spec=importlib.util.spec_from_file_location('r146_harness',HELPER/'run_job.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
spec=importlib.util.spec_from_file_location('r147_factory',HERE/'filter.py')
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
base.HERE=HERE;base.j.HERE=HERE;base.j.pf_installed=factory.installed
j=base.j
from moment_resample import MomentResamplePF,PredictedNoisePF,corrected_cloud
from pf_ctrv import PFCTRV,systematic_resample


def independent(w,omega,means,covs):
    w=w/w.sum();points=np.column_stack((omega,means));mu=sum(float(x)*p for x,p in zip(w,points))
    covariance=np.zeros((5,5))
    for weight,p,c in zip(w,points,covs):
        d=p-mu;term=np.outer(d,d);term[1:,1:]+=c;covariance+=weight*term
    return mu,covariance


def make(cls,seed):
    pf=PFCTRV(num_particles=100,resample_ess_ratio=.9)
    m,p=pf.initiate(np.array([50.,80.,1.5,70.]));particles=pf.sample_initial(m,p,np.random.default_rng(seed))
    return cls(pf,m,p,particles,np.random.default_rng(seed+10))


def numerical():
    parent=HERE.parent/'review135_pf_moment_resample';proof=j.read(parent/'final_verification.json')
    frozen=parent/'inputs.json';source=parent/'moment_resample.py'
    assert proof['sha256'][str(frozen)]==j.sha(frozen)
    assert j.read(frozen)['sha256'][str(source)]==j.sha(source)
    rng=np.random.default_rng(147);meanerror=coverror=0.;steps=resamples=corrected=0
    for _ in range(200):
        w=np.exp(rng.normal(0,2,100));w/=w.sum();omega=rng.normal(0,.2,100)
        means=rng.normal(size=(100,4));a=rng.normal(size=(100,4,4));covs=a@a.transpose(0,2,1)+np.eye(4)*.1
        selected=systematic_resample(w,float(rng.random()))
        ow,om,op,applied=corrected_cloud(w,omega,means,covs,selected);assert applied
        m,p=independent(w,omega,means,covs);x,q=independent(np.ones(100),ow,om,op)
        assert np.allclose(m,x,atol=1e-11,rtol=1e-12) and np.allclose(p,q,atol=1e-10,rtol=1e-12)
        assert np.linalg.eigvalsh(op).min()>0
        meanerror=max(meanerror,float(abs(m-x).max()));coverror=max(coverror,float(abs(p-q).max()))
    for seed in range(3):
        a=make(PredictedNoisePF,seed);b=make(MomentResamplePF,seed);b.correct_moments=False;c=make(MomentResamplePF,seed)
        for t in range(120):
            z=np.array([50+6*t,80+10*np.sin(t/8),1.5,70.]);confidence=(t%10)/10
            for method,args in [('predict',(t%11<3,)),('update',(z,confidence))]:
                x,p=getattr(a,method)(*args);y,q=getattr(b,method)(*args)
                assert np.array_equal(x,y) and np.array_equal(p,q)
                for field in ('means','covariances','omega','logw','shape_mean','shape_cov'):
                    assert np.array_equal(getattr(a,field),getattr(b,field))
                assert a.rng.bit_generator.state==b.rng.bit_generator.state;steps+=1
            c.predict(t%11<3);x,p=c.update(z,confidence)
            assert np.isfinite(x).all() and np.linalg.eigvalsh(p).min()>-1e-6
            assert np.linalg.eigvalsh(c.covariances).min()>-1e-9
        resamples+=a.resamples
        corrected+=c.corrected_resamples
    assert resamples>0 and corrected>0
    j.write(HERE/'numerical.json',{'status':'PASS','mixtures':200,'scored_control_steps':steps,'resamples':resamples,
          'max_mean_error':meanerror,'max_covariance_error':coverror,'PSD_RNG':True,
          'enabled_corrected_resamples':corrected,'source_R135_verified':True})
    print('NUMERICAL PASS',steps,resamples,flush=True)


def preflight(stage):
    base.preflight(stage)
    path=HERE/stage/'inputs.json';frozen=j.read(path)
    paths=[Path(__file__),HERE/'filter.py',HELPER/'run_job.py',
           HERE.parent/'review135_pf_moment_resample/moment_resample.py']
    frozen['sha256'].update({str(p):j.sha(p) for p in paths})
    frozen['PF_factory']='MomentResamplePF with current-score R; no ordered resampling'
    j.write(path,frozen)


def smoke(stage):
    base.smoke(stage)
    p=HERE/stage/'smoke.json';record=j.read(p)
    record['moment_candidate_repeat']=record.pop('ordered_candidate_repeat')
    j.write(p,record)


if __name__=='__main__':
    action=sys.argv[1]
    if action=='numerical':numerical()
    elif action=='worker':base.worker(sys.argv[2],int(sys.argv[3]))
    else:{'preflight':preflight,'smoke':smoke,'summary':base.summary}[action](sys.argv[2])
