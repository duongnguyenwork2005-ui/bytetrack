"""One invariant conditional MH move after R153's ordinary ESS resampling."""
from contextlib import contextmanager
import numpy as np
from guided import GuidedTurnPF,normal_proposal,position_derivative,log_prior_over_proposal,installed as guided_installed
from native_pf import lib
import rbpf_ctrv as rb

ACTIVITY={'steps':0,'proposals':0,'accepted':0}
def likelihood(means,covs,z,noise):
    s=covs[:,:2,:2]+noise[None,:2,:2]
    a=np.sqrt(s[:,0,0]);b=s[:,1,0]/a;d2=s[:,1,1]-b*b
    if not (d2>0).all():raise FloatingPointError('Non-SPD resample-move likelihood')
    d=np.sqrt(d2);delta=np.asarray(z)[None,:2]-means[:,:2]
    x=delta[:,0]/a;y=(delta[:,1]-b*x)/d
    value=-.5*(x*x+y*y)-np.log(a)-np.log(d)
    if not np.isfinite(value).all():raise FloatingPointError('Non-finite MH likelihood')
    return value
def acceptance(log_target_over_q_new,log_target_over_q_old,uniform):
    with np.errstate(divide='ignore'):logu=np.log(uniform)
    return logu<np.minimum(0.,log_target_over_q_new-log_target_over_q_old)

class ResampleMovePF(GuidedTurnPF):
    rejuvenate=True
    audit=False
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.move_steps=self.move_proposals=self.move_accepted=0;self.last_move=None
    def update(self,observation,confidence=None):
        cache=self._proposal_cache;self.last_move=None
        potential=self.rejuvenate and self.guided and cache is not None and cache['incoming_var']>0
        noise=self.pf._measurement_noise(max(float(self.shape_mean[1,0]),1e-3),confidence) if potential else None
        result=super().update(observation,confidence)
        if not (self.rejuvenate and self.guided and cache is not None and cache['incoming_var']>0
                and self.last_selected is not None):return result
        selected=self.last_selected;cache=dict(cache)
        for key in ('means','covs','incoming_mu','eta'):cache[key]=cache[key][selected].copy()
        z=np.ascontiguousarray(observation,dtype=np.float64);mu=cache['incoming_mu'];var=cache['incoming_var']
        m,p=self._predict_from_cache(cache,mu)
        qm,qv=normal_proposal(mu,var,m,p,position_derivative(mu,self.pf.dt,cache['means']),z,noise)
        current=self.incoming_mu.copy();proposal=qm+np.sqrt(qv)*self.rng.standard_normal(self.n)
        old_m,old_p=self._predict_from_cache(cache,current);new_m,new_p=self._predict_from_cache(cache,proposal)
        old=likelihood(old_m,old_p,z,noise)+log_prior_over_proposal(current,mu,var,qm,qv)
        new=likelihood(new_m,new_p,z,noise)+log_prior_over_proposal(proposal,mu,var,qm,qv)
        uniform=self.rng.random(self.n);accepted=acceptance(new,old,uniform)
        posterior_m=np.empty_like(new_m);posterior_p=np.empty_like(new_p)
        status=lib.update(self.n,new_m,new_p,z,noise,posterior_m,posterior_p,self._white,self._chol)
        if status:raise FloatingPointError('Native covariance failure during MH move')
        self.means[accepted]=posterior_m[accepted];self.covariances[accepted]=posterior_p[accepted]
        self.incoming_mu[accepted]=proposal[accepted];self.omega[accepted]=proposal[accepted]+cache['eta'][accepted]
        self._points_cache=None;count=int(accepted.sum())
        self.move_steps+=1;self.move_proposals+=self.n;self.move_accepted+=count
        ACTIVITY['steps']+=1;ACTIVITY['proposals']+=self.n;ACTIVITY['accepted']+=count
        if self.audit:self.last_move={'cache':cache,'old':current,'new':proposal,'qmean':qm,'qvar':qv,
            'accepted':accepted,'noise':noise,'logratio':new-old,'uniform':uniform,'z':z}
        return result

@contextmanager
def installed(enabled=True):
    with guided_installed():
        previous=rb.ConditionalPF
        rb.ConditionalPF=type('ConfiguredResampleMovePF',(ResampleMovePF,),{'rejuvenate':bool(enabled)})
        try:yield
        finally:rb.ConditionalPF=previous
