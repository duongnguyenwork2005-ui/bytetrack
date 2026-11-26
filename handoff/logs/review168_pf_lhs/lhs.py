"""Randomized one-dimensional marginal stratification of R153 innovations."""
from contextlib import contextmanager
import numpy as np
from scipy.special import ndtri
from guided import GuidedTurnPF,installed as guided_installed
import rbpf_ctrv as rb
ACTIVITY={'calls':0,'normal_draws':0}
def normal_lhs(rng,n):
    if int(n)!=n or n<=0:raise ValueError('Positive integer LHS size required')
    u=(rng.permutation(n)+rng.random(n))/n
    u=np.clip(u,np.nextafter(0.,1.),np.nextafter(1.,0.))
    return ndtri(u)
class Proxy:
    def __init__(self,rng,n):self.original,self.n=rng,n
    def __getattr__(self,name):return getattr(self.original,name)
    def standard_normal(self,size=None):
        if not isinstance(size,(int,np.integer)) or size!=self.n:raise ValueError('Only the frozen N angular draw is stratified')
        ACTIVITY['calls']+=1;ACTIVITY['normal_draws']+=self.n
        return normal_lhs(self.original,self.n)
class StratifiedGuidedPF(GuidedTurnPF):
    lhs=True
    @contextmanager
    def normal_scope(self):
        old=self.rng;self.rng=Proxy(old,self.n)
        try:yield
        finally:self.rng=old
    def predict(self,lost=False):
        if not self.lhs or self.pf._std_omega<=0:return super().predict(lost)
        with self.normal_scope():return super().predict(lost)
    def update(self,observation,confidence=None):
        cache=self._proposal_cache
        if not (self.lhs and self.guided and cache is not None and cache['incoming_var']>0):
            return super().update(observation,confidence)
        with self.normal_scope():return super().update(observation,confidence)
@contextmanager
def installed(enabled=True):
    with guided_installed():
        previous=rb.ConditionalPF;rb.ConditionalPF=type('ConfiguredLhsPF',(StratifiedGuidedPF,),{'lhs':bool(enabled)})
        try:yield
        finally:rb.ConditionalPF=previous
