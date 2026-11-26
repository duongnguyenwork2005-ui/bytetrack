"""First-stage ancestry selection with the indispensable second-stage ratio."""
from contextlib import contextmanager
import numpy as np
from guided import GuidedTurnPF,position_derivative
from confidence import installed as confidence_installed
from pf_ctrv import systematic_resample
import rbpf_ctrv as rb

def log_predictive(means,covariances,jacobian,variance,observation,noise):
    s=covariances[:,:2,:2]+noise[None,:2,:2]+variance*jacobian[:,:,None]*jacobian[:,None,:]
    a=np.sqrt(s[:,0,0]);b=s[:,1,0]/a;d2=s[:,1,1]-b*b
    if not (d2>0).all():raise FloatingPointError('Non-SPD auxiliary predictive covariance')
    d=np.sqrt(d2);r=np.asarray(observation)[None,:2]-means[:,:2]
    u=r[:,0]/a;v=(r[:,1]-b*u)/d
    return -.5*(u*u+v*v)-np.log(a)-np.log(d)

class AuxiliaryGuidedPF(GuidedTurnPF):
    auxiliary=True

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.auxiliary_updates=0;self.last_auxiliary_indices=None

    def update(self,observation,confidence=None):
        cache=self._proposal_cache;self.last_auxiliary_indices=None
        if self.auxiliary and self.guided and cache is not None and cache['incoming_var']>0:
            means,covs=self._predict_from_cache(cache,cache['incoming_mu'])
            jac=position_derivative(cache['incoming_mu'],self.pf.dt,cache['means'])
            noise=self.pf._measurement_noise(max(float(self.shape_mean[1,0]),1e-3),confidence)
            lookahead=log_predictive(means,covs,jac,cache['incoming_var'],observation,noise)
            first=self.logw+lookahead;first-=first.max();first-=np.log(np.exp(first).sum())
            if self.pf.ess(first)<self.pf.resample_ess_ratio*self.n:
                selected=systematic_resample(np.exp(first),float(self.rng.random()))
                # Shape priors are common, so only body/omega/eta ancestry moves.
                updated=cache.copy()
                for key in ('means','covs','incoming_mu','eta'):updated[key]=cache[key][selected].copy()
                self._proposal_cache=updated
                self.logw=-lookahead[selected].copy()
                self.auxiliary_updates+=1;self.resamples+=1;self.pf.n_resample+=1
                self.last_auxiliary_indices=selected.copy()
        return super().update(observation,confidence)

@contextmanager
def installed(enabled=True):
    with confidence_installed():
        previous=rb.ConditionalPF
        rb.ConditionalPF=type('ConfiguredAuxiliaryGuidedPF',(AuxiliaryGuidedPF,),{'auxiliary':bool(enabled)})
        try:yield
        finally:rb.ConditionalPF=previous
