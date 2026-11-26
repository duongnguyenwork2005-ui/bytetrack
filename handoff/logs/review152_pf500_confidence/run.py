"""Single preregistered count ablation, reusing frozen cohort safeguards."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;HELPER=HERE.parent/'review146_pf_shared_ordered'
# Load the generic historical runner and explicitly replace its factory below.
# Its ordered-resampling factory is never used in this experiment.
sys.path.insert(0,str(HELPER))
spec=importlib.util.spec_from_file_location('r152_generic_cohort',HELPER/'run_job.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
spec=importlib.util.spec_from_file_location('r152_particle_guard',HERE/'filter.py')
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
base.HERE=HERE;base.j.HERE=HERE;base.j.pf_installed=factory.installed
j=base.j;ORIGINAL_CONFIG=j.CONFIGS['pf'];j.CONFIGS['pf']=HERE/'rbpf500.yaml'
from pf_ctrv import PFCTRV
from predicted_noise import PredictedNoisePF


def make(cls,n,seed,zero_turn=False):
    pf=PFCTRV(num_particles=n,resample_ess_ratio=.5,**({'std_omega':0.,'init_std_omega':0.} if zero_turn else {}))
    m,p=pf.initiate(np.array([50.,80.,1.5,70.]));rng=np.random.default_rng(seed)
    return cls(pf,m,p,pf.sample_initial(m,p,rng),np.random.default_rng(seed+10))


def numerical():
    steps=0;resamples=0;maxcv=0.
    for seed in range(3):
        a=make(PredictedNoisePF,500,seed);b=make(factory.PF500,500,seed)
        for t in range(100):
            for method,args in [('predict',(t%13<3,)),('update',(np.array([50+4*t,80+15*np.sin(t/12),1.5,70.]),(t%11)/10))]:
                x,p=getattr(a,method)(*args);y,q=getattr(b,method)(*args)
                assert np.array_equal(x,y) and np.array_equal(p,q)
                for f in ('means','covariances','omega','logw','shape_mean','shape_cov'):
                    assert np.array_equal(getattr(a,f),getattr(b,f))
                assert a.rng.bit_generator.state==b.rng.bit_generator.state
                assert np.isfinite(y).all() and np.linalg.eigvalsh(q).min()>-1e-6;steps+=1
        resamples+=a.resamples
    a=make(PredictedNoisePF,100,10,True);b=make(factory.PF500,500,10,True)
    for t in range(80):
        for method,args in [('predict',(False,)),('update',(np.array([50+t,80+.5*t,1.5,70.]),.8))]:
            x,p=getattr(a,method)(*args);y,q=getattr(b,method)(*args)
            maxcv=max(maxcv,float(abs(x-y).max()),float(abs(p-q).max()))
            assert np.allclose(x,y,atol=1e-9,rtol=1e-11) and np.allclose(p,q,atol=1e-9,rtol=1e-11)
    try:make(factory.PF500,100,0)
    except ValueError:pass
    else:raise AssertionError('Particle-count guard failed')
    j.write(HERE/'numerical.json',{'status':'PASS','N500_direct_parity_steps':steps,'resamples':resamples,
             'CV_identical_component_steps':160,'max_CV_error':maxcv,'count_guard':True,'PSD_RNG':True})
    print('N500 NUMERICAL PASS',flush=True)


def preflight():
    base.preflight('dev');path=HERE/'dev/inputs.json';frozen=j.read(path)
    paths=[Path(__file__),HERE/'filter.py',HERE/'rbpf500.yaml',HELPER/'run_job.py']
    frozen['sha256'].update({str(p):j.sha(p) for p in paths})
    frozen['PF_factory']='Unchanged R142 PredictedNoisePF with N=500 guard; no ordered resampling'
    j.write(path,frozen)


def smoke():
    saved=j.pf_installed;j.pf_installed=base.parent_installed;j.CONFIGS['pf']=ORIGINAL_CONFIG
    try:j.smoke('dev')
    finally:j.pf_installed=saved;j.CONFIGS['pf']=HERE/'rbpf500.yaml'
    c=j.case('dev')
    for v in (c['videos'][0],c['videos'][-1]):
        for seed in range(3):
            a,ac,_=j.replay(c,v,'pf',seed,length=128);b,bc,_=j.replay(c,v,'pf',seed,length=128)
            assert a.equals(b) and ac.equals(bc)
    proof=j.read(HERE/'dev/smoke.json');proof['N500_repeat']=6;j.write(HERE/'dev/smoke.json',proof)
    j.check('dev');print('N500 SMOKE PASS',flush=True)


if __name__=='__main__':
    action=sys.argv[1]
    if action=='worker':
        for seed in range(3):base.full_pf('dev',seed)
    elif action=='summary':base.summary('dev')
    else:{'numerical':numerical,'preflight':preflight,'smoke':smoke}[action]()
