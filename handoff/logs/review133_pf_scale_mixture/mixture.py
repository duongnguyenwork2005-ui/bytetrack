"""A covariance-matched finite Gaussian scale mixture for position measurements."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np
from scipy.special import roots_genlaguerre, logsumexp

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review132_pf_predicted_noise'))
from predicted_noise import PredictedNoisePF
from native_pf import lib
from pf_ctrv import systematic_resample
import rbpf_ctrv as rb

NODES,WEIGHTS=roots_genlaguerre(3,1.)
WEIGHTS=WEIGHTS/WEIGHTS.sum()
SCALES=1/NODES
SCALES=SCALES/(WEIGHTS@SCALES)
WEIGHTS.setflags(write=False);SCALES.setflags(write=False)


def collapse(means,covariances,log_terms):
    likelihood=logsumexp(log_terms,axis=0)
    responsibility=np.exp(log_terms-likelihood[None,:])
    mean=np.sum(responsibility[:,:,None]*means,axis=0)
    delta=means-mean[None,:,:]
    cov=np.sum(responsibility[:,:,None,None]*(covariances+delta[:,:,:,None]*delta[:,:,None,:]),axis=0)
    return mean,.5*(cov+cov.transpose(0,2,1)),likelihood


class MixturePF(PredictedNoisePF):
    weights=WEIGHTS;scales=SCALES
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        k=len(self.weights)
        self._component_m=np.empty((k,self.n,4));self._component_p=np.empty((k,self.n,4,4))
        self._terms=np.empty((k,self.n));self.mixture_fallbacks=0

    def update(self,observation,confidence=None):
        self._points_cache=None;pf=self.pf
        z=np.ascontiguousarray(observation,dtype=float)
        if z.shape!=(4,) or not np.isfinite(z).all():
            raise ValueError('observation must contain four finite coordinates')
        noise=pf._measurement_noise(max(float(self.shape_mean[1,0]),1e-3),confidence)
        for index,(weight,scale) in enumerate(zip(self.weights,self.scales)):
            component_noise=noise.copy();component_noise[:2,:2]*=scale
            status=lib.update(self.n,self.means,self.covariances,z,component_noise,
                              self._component_m[index],self._component_p[index],self._white,self._chol)
            if status:
                self.mixture_fallbacks+=1
                return super().update(observation,confidence)
            self._terms[index]=np.log(weight)-.5*np.sum(self._white**2,axis=1)-np.log(self._chol).sum(axis=1)
        self.means,self.covariances,likelihood=collapse(self._component_m,self._component_p,self._terms)
        self.logw+=likelihood;self.logw-=self.logw.max();self.logw-=np.log(np.exp(self.logw).sum())
        lib.shapes(self.shape_mean,self.shape_cov,z,noise)
        result=self.estimate();self.updates+=1;pf.n_update+=1
        if pf.ess(self.logw)<pf.resample_ess_ratio*self.n:
            selected=systematic_resample(np.exp(self.logw),float(self.rng.random()))
            self.means=self.means[selected].copy();self.covariances=self.covariances[selected].copy()
            self.omega=self.omega[selected].copy();self.logw.fill(-np.log(self.n))
            self._points_cache=None;self.resamples+=1;pf.n_resample+=1
        return result


@contextmanager
def installed(enabled=True):
    previous=rb.ConditionalPF
    rb.ConditionalPF=MixturePF if enabled else PredictedNoisePF
    try:yield
    finally:rb.ConditionalPF=previous
