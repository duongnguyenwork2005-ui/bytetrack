"""Frozen marginal LHS ablation; do not weaken the Gaussian references."""
from pathlib import Path
import importlib.util,sys
import numpy as np
HERE=Path(__file__).resolve().parent;TEMPLATE=HERE.parent/'review156_guided_shape/run.py'
spec=importlib.util.spec_from_file_location('r168_strong_cohort',TEMPLATE)
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('r168_lhs_factory',HERE/'filter.py')
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
import lhs
from guided import GuidedTurnPF,log_prior_over_proposal
from pf_ctrv import PFCTRV
t.HERE=HERE;t.factory=factory;t.base.HERE=HERE;t.base.j.HERE=HERE;t.j.pf_installed=factory.installed;j=t.j
def pair(seed,zero=False):
    objects=[]
    for cls in (GuidedTurnPF,factory.StratifiedGuidedPF):
        pf=PFCTRV(num_particles=100,resample_ess_ratio=.9,**({'std_omega':0.,'init_std_omega':0.} if zero else {}))
        m,p=pf.initiate(np.array([50.,80.,1.5,70.]));rng=np.random.default_rng(seed)
        objects.append(cls(pf,m,p,pf.sample_initial(m,p,rng),np.random.default_rng(seed+10)))
    return objects
def same(a,b,x,p,y,q):
    assert np.array_equal(x,y) and np.array_equal(p,q)
    for field in ('means','covariances','omega','logw','shape_mean','shape_cov','incoming_mu'):
        assert np.array_equal(getattr(a,field),getattr(b,field)),field
    assert a.rng.bit_generator.state==b.rng.bit_generator.state
def numerical():
    from scipy.special import ndtr
    from scipy.stats import norm,kstest
    rng=np.random.default_rng(168);steps=0;inits=0
    for seed in range(3):
        a,b=pair(seed);b.lhs=False
        z=np.array([50.,80.,1.5,70.]);x,p=a.update(z,None);y,q=b.update(z,None);same(a,b,x,p,y,q);steps+=1
        for tick in range(80):
            x,p=a.predict(tick%13<4);y,q=b.predict(tick%13<4);same(a,b,x,p,y,q);steps+=1
            z=np.array([50+4*tick,80+5*np.sin(tick/7),1.5,70.])
            if tick in (2,40):
                ra=a.initialize_motion(x,p,z,n_frames=1,ref_pos=(z[0]-5,z[1]-1))
                rb=b.initialize_motion(y,q,z,n_frames=1,ref_pos=(z[0]-5,z[1]-1))
                assert ra[2] and rb[2];same(a,b,ra[0],ra[1],rb[0],rb[1]);inits+=1
            if tick%11>=3:
                score=None if tick%17==0 else (tick%11)/10
                x,p=a.update(z,score);y,q=b.update(z,score);same(a,b,x,p,y,q);steps+=1
    for n in (1,3,17,100):
        for _ in range(100):
            draw=lhs.normal_lhs(rng,n);assert np.isfinite(draw).all()
            assert np.array_equal(np.sort(np.floor(ndtr(draw)*n).astype(int)),np.arange(n))
    samples=np.array([lhs.normal_lhs(rng,100) for _ in range(4000)])
    tests=[kstest(samples[:,i],'norm').statistic for i in (0,33,66,99)];assert max(tests)<.05
    iid=rng.standard_normal(samples.shape);vlhs=float(samples.mean(axis=1).var(ddof=1));viid=float(iid.mean(axis=1).var(ddof=1))
    assert vlhs<.2*viid
    # Exact Gaussian evidence, with a deliberately nonoptimal q and varied means.
    mu=np.linspace(-.5,.5,100);var=.3;qm=mu+.2;qv=np.full(100,.39);z=.4;obs_std=.5
    values=qm+np.sqrt(qv)*samples
    weights=np.exp(norm.logpdf(z,values,obs_std)+log_prior_over_proposal(values,mu,var,qm,qv))
    estimate=weights.mean(axis=1);expected=float(norm.pdf(z,mu,np.sqrt(var+obs_std**2)).mean())
    se=float(estimate.std(ddof=1)/np.sqrt(len(estimate)));assert abs(estimate.mean()-expected)<6*se+1e-5
    model=pair(75)[1];before=dict(lhs.ACTIVITY)
    for tick in range(80):
        model.predict(lost=tick%11<3);x,p=model.update(np.array([50+3*tick,80+8*np.sin(tick/9),1.5,70.]),.8)
        assert np.isfinite(x).all() and np.linalg.eigvalsh(p).min()>-1e-6 and isinstance(model.rng,np.random.Generator)
    assert lhs.ACTIVITY['calls']>before['calls']
    a,b=pair(82,zero=True)
    for tick in range(40):
        for name,args in [('predict',(tick%9<2,)),('update',(np.array([50+3*tick,80.,1.5,70.]),.8))]:
            x,p=getattr(a,name)(*args);y,q=getattr(b,name)(*args);same(a,b,x,p,y,q)
    original=model.rng
    try:
        with model.normal_scope():raise RuntimeError('controlled probe')
    except RuntimeError:pass
    assert model.rng is original
    old=lhs.rb.ConditionalPF
    with factory.installed():assert issubclass(lhs.rb.ConditionalPF,factory.StratifiedGuidedPF)
    assert lhs.rb.ConditionalPF is old
    j.write(HERE/'numerical.json',{'status':'PASS','disabled_parent_steps':steps,'motion_initializations':inits,
        'strata_checks':400,'marginal_batches':4000,'marginal_KS_D':tests,'LHS_mean_variance':vlhs,'IID_mean_variance':viid,
        'evidence_estimate':float(estimate.mean()),'evidence_oracle':expected,'evidence_MC_SE':se,
        'native_PSD_steps':80,'zero_variance_parent_steps':80,'RNG_restore_on_exception':True,'hook_restore':True})
    print('R168 NUMERICAL PASS',vlhs,viid,flush=True)
def preflight():
    t.preflight();path=HERE/'dev/inputs.json';record=j.read(path)
    record['candidate']='R153 N100 + randomized marginal LHS predict/guided innovations, same Gaussian marginals'
    record['sha256'].update({str(p):j.sha(p) for p in (Path(__file__),HERE/'lhs.py',TEMPLATE)})
    j.write(path,record)
def worker():
    for seed in range(3):
        before=dict(lhs.ACTIVITY);t.base.full_pf('dev',seed)
        j.write(HERE/'dev'/('seed%d_lhs_activity.json'%seed),{key:lhs.ACTIVITY[key]-before[key] for key in before})
if __name__=='__main__':
    {'numerical':numerical,'preflight':preflight,'smoke':t.smoke,'worker':worker,'summary':t.summary}[sys.argv[1]]()
