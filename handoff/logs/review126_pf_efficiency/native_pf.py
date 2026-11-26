"""Model-preserving C/float64 kernels; RNG and mixture mean stay in NumPy."""
from contextlib import contextmanager
from pathlib import Path
import ctypes as ct
import os
import numpy as np
from fast import FastCartesianPF, CartesianPF
from cartesian import transition
from pf_ctrv import systematic_resample
import rbpf_ctrv as rb

LIB=Path(__file__).resolve().parent/'native'/('kernel.dll' if os.name=='nt' else 'kernel.so')
lib=ct.CDLL(str(LIB))
Array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
lib.predict.argtypes=[ct.c_int,Array,Array,Array,ct.c_double,ct.c_double,Array,Array,ct.c_double,ct.c_int]
lib.update.argtypes=[ct.c_int,Array,Array,Array,Array,Array,Array,Array,Array]
lib.shapes.argtypes=[Array,Array,Array,Array]
lib.within.argtypes=[ct.c_int,Array,Array,Array,Array,Array]
for fn in (lib.predict,lib.update,lib.shapes,lib.within):fn.restype=ct.c_int


class NativeCartesianPF(FastCartesianPF):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._next_m=np.empty_like(self.means);self._next_p=np.empty_like(self.covariances)
        self._white=np.empty((self.n,2));self._chol=np.empty((self.n,2));self._within=np.empty((4,4))
        self.fallbacks=0

    def estimate(self):
        points=self._points()
        mean,cov=self.pf.estimate_state(points,self.logw)
        weights=np.exp(self.logw)
        lib.within(self.n,self.means,self.covariances,points,weights,self._within)
        cov[:4,:4]+=self._within
        cov[self._shape_indices[0]]+=self.shape_cov[0]
        cov[self._shape_indices[1]]+=self.shape_cov[1]
        return mean,(cov+cov.T)*.5

    def predict(self,lost=False):
        self._points_cache=None
        pf=self.pf;height=max(self.shape_mean[1,0],1e-3)
        matrix=transition(self.omega,pf.dt)
        lib.predict(self.n,self.means,self.covariances,matrix,
                    (pf._std_weight_position*height)**2,(pf._std_weight_velocity*height)**2,
                    self.shape_mean,self.shape_cov,pf.dt,int(lost))
        self.omega+=self.rng.standard_normal(self.n)*pf._std_omega
        return self.estimate()

    def update(self,observation,confidence=None):
        self._points_cache=None;pf=self.pf
        z=np.ascontiguousarray(observation,dtype=float)
        noise=pf._measurement_noise(max(z[3],1e-3),confidence)
        status=lib.update(self.n,self.means,self.covariances,z,noise,
                          self._next_m,self._next_p,self._white,self._chol)
        if status:
            self.fallbacks+=1
            return super().update(observation,confidence)
        self.means,self._next_m=self._next_m,self.means
        self.covariances,self._next_p=self._next_p,self.covariances
        self.logw+=-.5*np.sum(self._white**2,axis=1)-np.log(self._chol).sum(axis=1)
        self.logw-=self.logw.max();self.logw-=np.log(np.exp(self.logw).sum())
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
    rb.ConditionalPF=NativeCartesianPF if enabled else CartesianPF
    try:yield
    finally:rb.ConditionalPF=previous
