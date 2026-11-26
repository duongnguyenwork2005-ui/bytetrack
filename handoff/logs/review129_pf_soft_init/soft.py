"""Covariance-weighted two-observation velocity initialization; research approximation."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'review126_pf_efficiency'))
from native_pf import NativeCartesianPF
import rbpf_ctrv as rb


def fuse(means,covariances,observation,noise):
    innovation=covariances+noise
    gain=np.linalg.solve(innovation,covariances.transpose(0,2,1)).transpose(0,2,1)
    result=means+np.einsum('nij,nj->ni',gain,observation-means)
    correction=np.eye(2)[None]-gain
    posterior=correction@covariances@correction.transpose(0,2,1)+gain@noise@gain.transpose(0,2,1)
    return result,.5*(posterior+posterior.transpose(0,2,1))


class SoftPF(NativeCartesianPF):
    def __init__(self,pf,mean,covariance,particles,rng):
        super().__init__(pf,mean,covariance,particles,rng)
        self.last_position_noise=pf._measurement_noise(max(mean[pf.H],1e-3),None)[:2,:2].copy()

    def update(self,observation,confidence=None):
        result=super().update(observation,confidence)
        self.last_position_noise=self.pf._measurement_noise(max(float(observation[3]),1e-3),confidence)[:2,:2].copy()
        return result

    def initialize_motion(self,mean,covariance,observation,n_frames,ref_pos):
        pf=self.pf
        _,_,ok=pf.try_initiate_from_motion(mean,covariance,observation,n_frames,ref_pos)
        if not ok:return mean,covariance,False
        reference=np.asarray(ref_pos if ref_pos is not None else mean[:2],dtype=float)
        elapsed=pf.dt*max(n_frames,1.)
        measured=(np.asarray(observation[:2],dtype=float)-reference)/elapsed
        current_noise=pf._measurement_noise(max(float(observation[3]),1e-3),None)[:2,:2]
        noise=(self.last_position_noise+current_noise)/(elapsed*elapsed)
        velocity,block=fuse(self.means[:,2:],self.covariances[:,2:,2:],measured,noise)
        self.means[:,2:]=velocity
        self.covariances[:,2:,:]=self.covariances[:,:,2:]=0.
        self.covariances[:,2:,2:]=block
        self.logw.fill(-np.log(self.n));self._points_cache=None
        mean,covariance=self.estimate()
        return mean,covariance,True


@contextmanager
def installed(enabled=True):
    previous=rb.ConditionalPF
    rb.ConditionalPF=SoftPF if enabled else NativeCartesianPF
    try:yield
    finally:rb.ConditionalPF=previous
