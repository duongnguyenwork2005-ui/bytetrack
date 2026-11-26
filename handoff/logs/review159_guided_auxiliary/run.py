"""Strong-reference dev harness and independent auxiliary importance oracles."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;TEMPLATE=HERE.parent/'review156_guided_shape/run.py'
spec=importlib.util.spec_from_file_location('r159_strong_cohort',TEMPLATE)
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
spec=importlib.util.spec_from_file_location('r159_auxiliary_factory',HERE/'filter.py')
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
t.HERE=HERE;t.factory=factory;t.base.HERE=HERE;t.base.j.HERE=HERE;t.j.pf_installed=factory.installed;j=t.j

def numerical():
    from scipy.stats import multivariate_normal,norm
    from guided import normal_proposal,log_prior_over_proposal
    rng=np.random.default_rng(159);steps=0;max_constant=0.;max_predictive=0.
    for seed in range(3):
        a=t.make(t.GuidedTurnPF,seed);b=t.make(factory.AuxiliaryGuidedPF,seed);b.auxiliary=False
        for tick in range(80):
            z=np.array([50+3*tick,80+5*np.sin(tick/10),1.5,70.])
            for name,args in [('predict',(tick%13<4,)),('update',(z,.8))]:
                x,p=getattr(a,name)(*args);y,q=getattr(b,name)(*args)
                assert np.array_equal(x,y) and np.array_equal(p,q)
                assert a.rng.bit_generator.state==b.rng.bit_generator.state;steps+=1
    for _ in range(100):
        n=7;means=rng.normal(size=(n,4));a=rng.normal(size=(n,4,4))
        covs=a@a.transpose(0,2,1)+np.eye(4)*.2;jac=rng.normal(size=(n,2));variance=float(rng.uniform(.01,2))
        centre=rng.normal(size=n);z=rng.normal(size=4);noise=np.diag([.5,.8,.1,.2])
        evidence=factory.log_predictive(means,covs,jac,variance,z,noise)
        qm,qv=normal_proposal(centre,variance,means,covs,jac,z,noise)
        for ix in range(n):
            s=covs[ix,:2,:2]+noise[:2,:2];marginal=s+variance*np.outer(jac[ix],jac[ix])
            expected=multivariate_normal.logpdf(z[:2],mean=means[ix,:2],cov=marginal)+np.log(2*np.pi)
            max_predictive=max(max_predictive,float(abs(evidence[ix]-expected)))
            assert abs(evidence[ix]-expected)<1e-10
            draws=rng.normal(qm[ix],np.sqrt(qv[ix]),size=10)
            target_mean=means[ix,:2]+(draws-centre[ix])[:,None]*jac[ix]
            likelihood=np.array([multivariate_normal.logpdf(z[:2],mean=m,cov=s)+np.log(2*np.pi) for m in target_mean])
            ratios=norm.logpdf(draws,centre[ix],np.sqrt(variance))-norm.logpdf(draws,qm[ix],np.sqrt(qv[ix]))
            direct=ratios+likelihood-evidence[ix]
            assert np.allclose(ratios,log_prior_over_proposal(draws,centre[ix],variance,qm[ix],qv[ix]))
            max_constant=max(max_constant,float(abs(direct).max()));assert abs(direct).max()<1e-10
    model=t.make(factory.AuxiliaryGuidedPF,91);ancestry=0
    # Deliberately separated conditional priors exercise nonuniform ancestry;
    # tracking measurements and prior/q retain their actual CT implementation.
    model.means[:,:2]+=rng.normal(size=(model.n,2))*30
    for tick in range(80):
        model.predict(lost=tick%11<3);old=model._proposal_cache
        recorded=[];original=model._predict_from_cache
        def record(cache,omega):
            recorded.append(cache)
            return original(cache,omega)
        model._predict_from_cache=record
        try:x,p=model.update(np.array([50+3*tick,80+3*np.sin(tick/7),1.5,70.]),.8)
        finally:model._predict_from_cache=original
        selected=model.last_auxiliary_indices
        if selected is not None:
            ancestry+=1
            for key in ('means','covs','incoming_mu','eta'):
                assert np.array_equal(recorded[-1][key],old[key][selected])
            assert np.array_equal(recorded[-1]['shape_mean'],old['shape_mean'])
        assert np.isfinite(p).all() and np.linalg.eigvalsh(p).min()>-1e-6
    assert ancestry>0 and model.auxiliary_updates==ancestry
    # Zero incoming variance uses the unchanged parent path, including RNG.
    a=t.make(t.GuidedTurnPF,92);b=t.make(factory.AuxiliaryGuidedPF,92)
    a.pf._std_omega=b.pf._std_omega=0.;a.incoming_var=b.incoming_var=0.
    for tick in range(40):
        for name,args in [('predict',(False,)),('update',(np.array([50+2*tick,80.,1.5,70.]),.8))]:
            x,p=getattr(a,name)(*args);y,q=getattr(b,name)(*args)
            assert np.array_equal(x,y) and np.array_equal(p,q) and a.rng.bit_generator.state==b.rng.bit_generator.state
    old=factory.rb.ConditionalPF
    with factory.installed():assert issubclass(factory.rb.ConditionalPF,factory.AuxiliaryGuidedPF)
    assert factory.rb.ConditionalPF is old
    j.write(HERE/'numerical.json',{'status':'PASS','disabled_parent_steps':steps,'linear_Gaussian_components':700,
          'constant_second_stage_draws':7000,'max_predictive_error':max_predictive,'max_second_stage_error':max_constant,
          'actual_CT_PSD_steps':80,'auxiliary_ancestry_checks':ancestry,'zero_variance_parent_steps':80,'hook_restore':True})
    print('GUIDED AUXILIARY NUMERICAL PASS',flush=True)

def preflight():
    t.preflight();path=HERE/'dev/inputs.json';record=j.read(path)
    record['candidate']='R153 guided + auxiliary ancestry ESS resampling with second-stage correction'
    record['sha256'].update({str(p):j.sha(p) for p in (Path(__file__),TEMPLATE)})
    j.write(path,record)

if __name__=='__main__':
    action=sys.argv[1]
    if action=='worker':
        for seed in range(3):t.base.full_pf('dev',seed)
    else:{'numerical':numerical,'preflight':preflight,'smoke':t.smoke,'summary':t.summary}[action]()
