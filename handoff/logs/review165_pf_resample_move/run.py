"""Bounded resample-move study against the unchanged strongest dev references."""
from pathlib import Path
import importlib.util,sys
import numpy as np
HERE=Path(__file__).resolve().parent;TEMPLATE=HERE.parent/'review156_guided_shape/run.py'
spec=importlib.util.spec_from_file_location('r165_strong_cohort',TEMPLATE)
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('r165_move_factory',HERE/'filter.py')
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
import rejuvenate as move
from guided import GuidedTurnPF,normal_proposal,position_derivative,log_prior_over_proposal
from cartesian import transition
from pf_ctrv import PFCTRV
t.HERE=HERE;t.factory=factory;t.base.HERE=HERE;t.base.j.HERE=HERE;t.j.pf_installed=factory.installed;j=t.j

def pair(seed,ratio=.9,zero=False):
    objects=[]
    for cls in (GuidedTurnPF,factory.ResampleMovePF):
        pf=PFCTRV(num_particles=100,resample_ess_ratio=ratio,**({'std_omega':0.,'init_std_omega':0.} if zero else {}))
        m,p=pf.initiate(np.array([50.,80.,1.5,70.]));rng=np.random.default_rng(seed)
        objects.append(cls(pf,m,p,pf.sample_initial(m,p,rng),np.random.default_rng(seed+10)))
    return objects
def same(a,b,x,p,y,q):
    assert np.array_equal(x,y) and np.array_equal(p,q)
    for field in ('means','covariances','omega','logw','shape_mean','shape_cov','incoming_mu'):
        assert np.array_equal(getattr(a,field),getattr(b,field)),field
    assert a.rng.bit_generator.state==b.rng.bit_generator.state
    assert np.isfinite(y).all() and np.linalg.eigvalsh(q).min()>-1e-6
def numerical():
    from scipy.linalg import expm
    from scipy.integrate import quad,cumulative_trapezoid
    from scipy.stats import norm,multivariate_normal
    rng=np.random.default_rng(165);steps=0;inits=0
    for seed in range(3):
        a,b=pair(seed);b.rejuvenate=False
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
    # Gaussian likelihood and MH ratio are compared to separate SciPy densities.
    max_ll=0.;max_balance=0.;optimal=0
    for _ in range(80):
        n=7;means=rng.normal(size=(n,4));a=rng.normal(size=(n,4,4));covs=a@a.transpose(0,2,1)+np.eye(4)*.2
        z=rng.normal(size=4);noise=np.diag([.2,.3,.1,.2]);actual=move.likelihood(means,covs,z,noise)
        expected=np.array([multivariate_normal.logpdf(z[:2],mean=m[:2],cov=p[:2,:2]+noise[:2,:2])+np.log(2*np.pi)
            for m,p in zip(means,covs)])
        error=float(abs(actual-expected).max());max_ll=max(max_ll,error);assert error<1e-10
        mu=float(rng.normal());var=float(rng.uniform(.01,1));jvec=rng.normal(size=2)
        centre=means[:1];cov=covs[:1];qm,qv=normal_proposal(np.array([mu]),var,centre,cov,jvec[None],z,noise)
        draws=qm[0]+np.sqrt(qv[0])*np.linspace(-4,4,51)
        shifted=np.repeat(centre,len(draws),axis=0);shifted[:,:2]+=(draws-mu)[:,None]*jvec
        weights=move.likelihood(shifted,np.repeat(cov,len(draws),axis=0),z,noise)+log_prior_over_proposal(draws,mu,var,qm[0],qv[0])
        assert np.ptp(weights)<1e-9
        assert move.acceptance(weights[1:],weights[:-1],np.full(len(draws)-1,.999)).all();optimal+=len(draws)-1
        # Detailed balance for a nonlinear scalar target, arbitrary Gaussian q.
        x,y=rng.normal(size=2);lp=lambda w:norm.logpdf(w,mu,np.sqrt(var))-.5*(np.sin(w)-.4)**2/.03
        lq=lambda w:norm.logpdf(w,float(qm[0]),float(np.sqrt(qv[0])))
        ratio=lp(y)-lq(y)-lp(x)+lq(x)
        flow_xy=lp(x)+lq(y)+min(0.,ratio);flow_yx=lp(y)+lq(x)+min(0.,-ratio)
        error=float(abs(flow_xy-flow_yx));max_balance=max(max_balance,error);assert error<1e-9
        u=np.array([.01,.3,.99]);assert np.array_equal(move.acceptance(np.full(3,lp(y)-lq(y)),np.full(3,lp(x)-lq(x)),u),np.log(u)<min(0.,ratio))
    # Nonlinear CT stationary-kernel oracle: target from scalar expm/quadrature.
    mu=.25;var=.04;m=np.array([0.,0.,8.,2.]);p=np.diag([.4,.3,.1,.1]);qnoise=np.diag([.09,.09,.02,.02]);r=np.diag([.2,.2,.1,.2]);z=np.array([7.,3.4,1.,1.])
    def density(w):
        a=np.array([[0,0,1,0],[0,0,0,1],[0,0,0,-w],[0,0,w,0]],dtype=float);f=expm(a)
        return norm.pdf(w,mu,np.sqrt(var))*multivariate_normal.pdf(z[:2],mean=(f@m)[:2],cov=(f@p@f.T+qnoise+r)[:2,:2])
    lo,hi=mu-8*np.sqrt(var),mu+8*np.sqrt(var);evidence=quad(density,lo,hi,epsabs=1e-12)[0]
    mean=quad(lambda w:w*density(w),lo,hi,epsabs=1e-12)[0]/evidence
    second=quad(lambda w:w*w*density(w),lo,hi,epsabs=1e-12)[0]/evidence
    grid=np.linspace(lo,hi,2001);cdf=cumulative_trapezoid(np.array([density(w) for w in grid]),grid,initial=0);cdf/=cdf[-1]
    current=np.interp(rng.random(50000),cdf,grid)
    f=transition(np.array([mu]),1.);mp=np.einsum('nij,j->ni',f,m);pp=f@p@f.transpose(0,2,1)+qnoise
    qm,qv=normal_proposal(np.array([mu]),var,mp,pp,position_derivative(np.array([mu]),1.,m[None]),z,r)
    proposal=rng.normal(qm[0],np.sqrt(qv[0]),size=len(current))
    def logratio(values):
        f=transition(values,1.);means=np.einsum('nij,j->ni',f,m);covs=f@p@f.transpose(0,2,1)+qnoise
        return move.likelihood(means,covs,z,r)+log_prior_over_proposal(values,mu,var,qm[0],qv[0])
    accept=move.acceptance(logratio(proposal),logratio(current),rng.random(len(current)));after=np.where(accept,proposal,current)
    se=np.sqrt((second-mean*mean)/len(current));assert abs(current.mean()-mean)<6*se+1e-4 and abs(after.mean()-mean)<6*se+1e-4
    # Actual native posterior, shared shape, and ancestry under repeated moves.
    model=pair(71,ratio=1.)[1];model.audit=True;audited=0;max_posterior=0.
    for tick in range(60):
        model.predict(lost=tick%11<3);cache=model._proposal_cache
        sm=model.shape_mean.copy();sp=model.shape_cov.copy();z=np.array([50+4*tick,80+8*np.sin(tick/9),1.5,70.])
        noise=model.pf._measurement_noise(max(float(sm[1,0]),1e-3),.8)
        for dim in range(2):
            gain=sp[dim,:,0]/(sp[dim,0,0]+noise[dim+2,dim+2]);sm[dim]+=gain*(z[dim+2]-sm[dim,0])
            sp[dim]-=np.outer(gain,sp[dim,0].copy())
        x,c=model.update(z,.8)
        assert np.allclose(model.shape_mean,sm,atol=1e-10) and np.allclose(model.shape_cov,sp,atol=1e-10)
        assert np.isfinite(c).all() and np.linalg.eigvalsh(c).min()>-1e-6
        audit=model.last_move
        if audit is None:continue
        audited+=1;ix=model.last_selected
        for key in ('means','covs','incoming_mu','eta'):assert np.array_equal(audit['cache'][key],cache[key][ix])
        expected=np.where(audit['accepted'],audit['new'],audit['old'])
        assert np.array_equal(model.incoming_mu,expected) and np.array_equal(model.omega,expected+cache['eta'][ix])
        assert np.array_equal(audit['accepted'],move.acceptance(audit['logratio'],np.zeros(model.n),audit['uniform']))
        for child in np.flatnonzero(audit['accepted'])[:5]:
            omega=audit['new'][child];dt=model.pf.dt
            a=np.array([[0,0,1,0],[0,0,0,1],[0,0,0,-omega],[0,0,omega,0]],float);f=expm(a*dt)
            h=cache['height'];qq=np.diag([(model.pf._std_weight_position*h)**2]*2+[(model.pf._std_weight_velocity*h)**2]*2)
            pm=f@cache['means'][ix[child]];pc=f@cache['covs'][ix[child]]@f.T+qq
            gain=np.linalg.solve(pc[:2,:2]+noise[:2,:2],pc[:2,:]).T
            em=pm+gain@(z[:2]-pm[:2]);correction=np.eye(4);correction[:,:2]-=gain
            ec=correction@pc@correction.T+gain@noise[:2,:2]@gain.T
            error=max(float(abs(em-model.means[child]).max()),float(abs(ec-model.covariances[child]).max()))
            max_posterior=max(max_posterior,error);assert error<1e-7
    assert audited>0 and model.move_accepted>0
    # No resampling or a Dirac incoming prior must leave parent arrays/RNG exact.
    for ratio,zero in ((0.,False),(.9,True)):
        a,b=pair(82,ratio=ratio,zero=zero)
        for tick in range(40):
            for name,args in [('predict',(tick%9<2,)),('update',(np.array([50+3*tick,80.,1.5,70.]),.8))]:
                x,p=getattr(a,name)(*args);y,q=getattr(b,name)(*args);same(a,b,x,p,y,q)
        assert b.move_steps==0
    old=factory.rb.ConditionalPF if hasattr(factory,'rb') else move.rb.ConditionalPF
    with factory.installed():assert issubclass(move.rb.ConditionalPF,factory.ResampleMovePF)
    assert move.rb.ConditionalPF is old
    j.write(HERE/'numerical.json',{'status':'PASS','disabled_parent_steps':steps,'motion_initializations':inits,
        'SciPy_likelihood_components':560,'max_likelihood_error':max_ll,'detailed_balance_pairs':80,'max_flow_error':max_balance,
        'optimal_linear_acceptance_draws':optimal,'CT_stationary_draws':50000,'CT_mean_oracle':mean,
        'CT_mean_before':float(current.mean()),'CT_mean_after':float(after.mean()),'CT_mean_SE':float(se),
        'native_move_steps':audited,'native_accepted':model.move_accepted,'max_posterior_error':max_posterior,
        'zero_variance_no_resample_parent_steps':160,'hook_restore':True})
    print('RESAMPLE MOVE NUMERICAL PASS',audited,model.move_accepted,flush=True)
def preflight():
    t.preflight();path=HERE/'dev/inputs.json';record=j.read(path)
    record['candidate']='R153 guided N100 + one conditional independent MH after ESS resampling'
    record['sha256'].update({str(p):j.sha(p) for p in (Path(__file__),HERE/'rejuvenate.py',TEMPLATE)})
    j.write(path,record)
def worker():
    for seed in range(3):
        before=dict(move.ACTIVITY);t.base.full_pf('dev',seed)
        activity={key:move.ACTIVITY[key]-before[key] for key in before}
        j.write(HERE/'dev'/('seed%d_move_activity.json'%seed),activity)
        print('R165 MOVE ACTIVITY',seed,activity,flush=True)
if __name__=='__main__':
    {'numerical':numerical,'preflight':preflight,'smoke':t.smoke,'worker':worker,'summary':t.summary}[sys.argv[1]]()
