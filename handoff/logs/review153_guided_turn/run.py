"""Independent numerical oracles before any guided-turn tracking scores."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;HELPER=HERE.parent/'review146_pf_shared_ordered'
sys.path.insert(0,str(HELPER))
spec=importlib.util.spec_from_file_location('r153_generic_cohort',HELPER/'run_job.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
spec=importlib.util.spec_from_file_location('r153_guided_factory',HERE/'filter.py')
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
base.HERE=HERE;base.j.HERE=HERE;base.j.pf_installed=factory.installed;j=base.j
import guided
from predicted_noise import PredictedNoisePF
from pf_ctrv import PFCTRV
from cartesian import transition


def make(cls,seed,zero=False,ratio=.9):
    pf=PFCTRV(num_particles=100,resample_ess_ratio=ratio,**({'std_omega':0.,'init_std_omega':0.} if zero else {}))
    m,p=pf.initiate(np.array([50.,80.,1.5,70.]));rng=np.random.default_rng(seed)
    return cls(pf,m,p,pf.sample_initial(m,p,rng),np.random.default_rng(seed+10))


def same(a,b,x,p,y,q):
    assert np.array_equal(x,y) and np.array_equal(p,q)
    for field in ('means','covariances','omega','logw','shape_mean','shape_cov'):
        assert np.array_equal(getattr(a,field),getattr(b,field)),field
    assert a.rng.bit_generator.state==b.rng.bit_generator.state
    assert np.isfinite(y).all() and np.linalg.eigvalsh(q).min()>-1e-6


def numerical():
    from scipy.linalg import expm
    from scipy.integrate import quad
    from scipy.stats import norm,multivariate_normal
    rng=np.random.default_rng(153);steps=0;initializations=0;maxderivative=0.;maxexpm=0.
    for seed in range(3):
        a=make(PredictedNoisePF,seed);b=make(factory.GuidedTurnPF,seed);b.guided=False
        z=np.array([50.,80.,1.5,70.]);x,p=a.update(z,.8);y,q=b.update(z,.8);same(a,b,x,p,y,q);steps+=1
        for t in range(100):
            x,p=a.predict(t%13<3);y,q=b.predict(t%13<3);same(a,b,x,p,y,q);steps+=1
            z=np.array([50+5*t,80+10*np.sin(t/8),1.5,70.])
            if t in (2,60):
                ra=a.initialize_motion(x,p,z,n_frames=1,ref_pos=(z[0]-5,z[1]-1))
                rb=b.initialize_motion(y,q,z,n_frames=1,ref_pos=(z[0]-5,z[1]-1))
                assert ra[2] and rb[2] and b._proposal_cache is None
                same(a,b,ra[0],ra[1],rb[0],rb[1]);initializations+=1
            if t%11>=3:
                score=None if t%17==0 else (t%11)/10
                x,p=a.update(z,score);y,q=b.update(z,score);same(a,b,x,p,y,q);steps+=1
        assert a.resamples==b.resamples and a.resamples>0
    for dt in (.04,.2,1.):
        omega=np.array([0.,1e-9,-1e-6,.00099,.00101,-.1,.5,-2.])
        means=rng.normal(size=(len(omega),4));eps=1e-5
        fd=(np.einsum('nij,nj->ni',transition(omega+eps,dt),means)-
            np.einsum('nij,nj->ni',transition(omega-eps,dt),means))/(2*eps)
        derivative=guided.position_derivative(omega,dt,means)
        maxderivative=max(maxderivative,float(abs(fd[:,:2]-derivative).max()))
        assert np.allclose(fd[:,:2],derivative,atol=1e-8,rtol=1e-6)
        for w,f in zip(omega,transition(omega,dt)):
            a=np.array([[0,0,1,0],[0,0,0,1],[0,0,0,-w],[0,0,w,0]],dtype=float)
            error=float(abs(f-expm(a*dt)).max());maxexpm=max(maxexpm,error);assert error<1e-12
    maxconstant=0.
    for _ in range(100):
        mu=float(rng.normal());var=float(np.exp(rng.normal(-1,.5)))
        centre=rng.normal(size=4);a=rng.normal(size=(4,4));cov=a@a.T+np.eye(4)*.1
        jac=rng.normal(size=2);z=rng.normal(size=4);noise=np.diag([.3,.4,.1,.2])
        qm,qv=guided.normal_proposal(np.array([mu]),var,centre[None],cov[None],jac[None],z,noise)
        draws=qm[0]+np.sqrt(qv[0])*np.linspace(-3,3,101)
        s=cov[:2,:2]+noise[:2,:2];means=centre[None,:2]+(draws-mu)[:,None]*jac
        ll=np.array([multivariate_normal.logpdf(z[:2],mean=m,cov=s) for m in means])
        increments=ll+guided.log_prior_over_proposal(draws,mu,var,qm[0],qv[0])
        evidence=multivariate_normal.logpdf(z[:2],mean=centre[:2],cov=s+var*np.outer(jac,jac))
        error=float(abs(increments-evidence).max());maxconstant=max(maxconstant,error);assert error<1e-9
    # Nonlinear CT oracle uses matrix exponentials in scalar quadrature, not T().
    mu=.25;var=.04;mean=np.array([0.,0.,8.,2.]);cov=np.diag([.4,.3,.1,.1]);q=np.diag([.09,.09,.02,.02]);r=np.diag([.2,.2])
    z=np.array([7.,3.4])
    def likelihood(w):
        a=np.array([[0,0,1,0],[0,0,0,1],[0,0,0,-w],[0,0,w,0]],dtype=float);f=expm(a)
        return multivariate_normal.pdf(z,mean=(f@mean)[:2],cov=(f@cov@f.T+q)[:2,:2]+r)
    lo,hi=mu-10*np.sqrt(var),mu+10*np.sqrt(var)
    evidence=quad(lambda w:likelihood(w)*norm.pdf(w,mu,np.sqrt(var)),lo,hi,epsabs=1e-11)[0]
    oracle=quad(lambda w:w*likelihood(w)*norm.pdf(w,mu,np.sqrt(var)),lo,hi,epsabs=1e-11)[0]/evidence
    f=transition(np.array([mu]),1.);mp=np.einsum('nij,j->ni',f,mean);pp=f@cov@f.transpose(0,2,1)+q
    jac=guided.position_derivative(np.array([mu]),1.,mean[None]);noise=np.eye(4);noise[:2,:2]=r
    qm,qv=guided.normal_proposal(np.array([mu]),var,mp,pp,jac,np.r_[z,0.,0.],noise)
    draws=qm[0]+np.sqrt(qv[0])*rng.standard_normal(20000)
    fs=transition(draws,1.);means=np.einsum('nij,j->ni',fs,mean);ps=fs@cov@fs.transpose(0,2,1)+q
    s=ps[:,:2,:2]+r;delta=z-means[:,:2];sign,logdet=np.linalg.slogdet(s);assert (sign>0).all()
    ll=-.5*(np.sum(delta*np.linalg.solve(s,delta[...,None])[...,0],axis=1)+logdet+2*np.log(2*np.pi))
    unnormalized=np.exp(ll+guided.log_prior_over_proposal(draws,mu,var,qm[0],qv[0]))
    weights=unnormalized/unnormalized.sum();estimate=float(weights@draws)
    se=float(np.sqrt(np.sum((weights*(draws-estimate))**2)));evse=float(unnormalized.std(ddof=1)/np.sqrt(len(draws)))
    assert abs(estimate-oracle)<=6*se+1e-4 and abs(unnormalized.mean()-evidence)<=6*evse+1e-5
    # Audit the Gaussian incoming ancestry at the actual resampling boundary.
    active=make(factory.GuidedTurnPF,77,ratio=1.);selected=[];original=guided.systematic_resample
    def audited(w,u):
        ix=original(w,u);selected.append(active.incoming_mu[ix].copy());return ix
    guided.systematic_resample=audited
    try:
        active.update(np.array([50.,80.,1.5,70.]),.9)
        assert active._proposal_cache is None
        for t in range(80):
            active.predict(t%11<3)
            x,p=active.update(np.array([50+4*t,80+15*np.sin(t/10),1.5,70.]),.9)
            assert active._proposal_cache is None and np.isfinite(x).all() and np.linalg.eigvalsh(p).min()>-1e-6
            if active.last_selected is not None:assert np.array_equal(active.incoming_mu,selected[-1])
    finally:guided.systematic_resample=original
    assert active.guided_updates>0 and active.resamples>0 and selected
    a=make(PredictedNoisePF,12,True);b=make(factory.GuidedTurnPF,12,True)
    for t in range(50):
        x,p=a.predict();y,q=b.predict();same(a,b,x,p,y,q)
        z=np.array([50+t,80+.5*t,1.5,70.]);x,p=a.update(z,.7);y,q=b.update(z,.7);same(a,b,x,p,y,q)
    assert b.guided_updates==0 and b.skipped_guides==50
    j.write(HERE/'numerical.json',{'status':'PASS','disabled_parent_steps':steps,'motion_initializations':initializations,
        'max_derivative_error':maxderivative,'max_matrix_exponential_error':maxexpm,
        'linear_Gaussian_cases':100,'max_constant_increment_error':maxconstant,
        'CT_quadrature':{'mean':oracle,'MC_mean':estimate,'MC_SE':se,'samples':20000,'evidence':evidence,
                         'MC_evidence':float(unnormalized.mean()),'evidence_SE':evse},
        'guided_updates':active.guided_updates,'resample_ancestry_cases':len(selected),'zero_variance_steps':100,'PSD_RNG':True})
    print('GUIDED TURN NUMERICAL PASS',flush=True)


def preflight():
    base.preflight('dev');path=HERE/'dev/inputs.json';frozen=j.read(path)
    paths=[Path(__file__),HERE/'guided.py',HERE/'filter.py',HELPER/'run_job.py']
    frozen['sha256'].update({str(p):j.sha(p) for p in paths})
    frozen['PF_factory']='N100 guided incoming turn innovation with explicit log prior/q; unchanged R142 temporal model'
    j.write(path,frozen)


def smoke():
    base.smoke('dev');saved=j.pf_installed;j.pf_installed=lambda:factory.installed(False)
    try:
        c=j.case('dev')
        for video in (c['videos'][0],c['videos'][-1]):
            for seed in range(3):
                mot,_,_=j.replay(c,video,'pf',seed,length=128)
                j.equal_prefix(mot.to_numpy(float),j.pf_mot(c,seed,video),128)
    finally:j.pf_installed=saved
    record=j.read(HERE/'dev/smoke.json');record['guided_repeat']=record.pop('ordered_candidate_repeat')
    record['guided_disabled_parent_parity']=6;j.write(HERE/'dev/smoke.json',record)
    j.check('dev');print('GUIDED TURN SMOKE PASS',flush=True)


if __name__=='__main__':
    action=sys.argv[1]
    if action=='worker':
        for seed in range(3):base.full_pf('dev',seed)
    elif action=='summary':base.summary('dev')
    else:{'numerical':numerical,'preflight':preflight,'smoke':smoke}[action]()
