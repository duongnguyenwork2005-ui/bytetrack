"""Ordered stratified selection on the sampled omega coordinate."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'review132_pf_predicted_noise'))
from predicted_noise import PredictedNoisePF
from pf_ctrv import systematic_resample
import rbpf_ctrv as rb


def ordered_selection(weights,omega,uniforms):
    weights=np.asarray(weights,dtype=float);omega=np.asarray(omega,dtype=float)
    uniforms=np.asarray(uniforms,dtype=float);n=len(weights)
    if n<1 or weights.shape!=omega.shape or weights.shape!=uniforms.shape:
        raise ValueError('Matching nonempty vectors required')
    if not all(np.isfinite(a).all() for a in (weights,omega,uniforms)):
        raise ValueError('Finite weights, omega and uniforms required')
    if np.any(weights<0) or weights.sum()<=0 or np.any((uniforms<0)|(uniforms>=1)):
        raise ValueError('Nonnegative weights and uniforms in [0,1) required')
    order=np.argsort(omega,kind='stable')
    cdf=np.cumsum(weights[order]/weights.sum());cdf[-1]=1.
    targets=(np.arange(n)+uniforms)/n
    return order[np.searchsorted(cdf,targets,side='right')]


class OrderedPF(PredictedNoisePF):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ordered_stratified=True

    def update(self,observation,confidence=None):
        pf=self.pf;ratio=pf.resample_ess_ratio
        pf.resample_ess_ratio=0.
        try:result=super().update(observation,confidence)
        finally:pf.resample_ess_ratio=ratio
        if pf.ess(self.logw)<ratio*self.n:
            weights=np.exp(self.logw)
            if self.ordered_stratified:
                selected=ordered_selection(weights,self.omega,self.rng.random(self.n))
            else:selected=systematic_resample(weights,float(self.rng.random()))
            self.means=self.means[selected].copy();self.covariances=self.covariances[selected].copy()
            self.omega=self.omega[selected].copy();self.logw.fill(-np.log(self.n))
            self._points_cache=None;self.resamples+=1;pf.n_resample+=1
        return result


@contextmanager
def installed(enabled=True):
    previous=rb.ConditionalPF
    rb.ConditionalPF=OrderedPF if enabled else PredictedNoisePF
    try:yield
    finally:rb.ConditionalPF=previous
