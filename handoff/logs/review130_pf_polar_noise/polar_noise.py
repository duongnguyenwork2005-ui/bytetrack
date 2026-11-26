"""A directional speed-noise approximation plus marginalized heading diffusion."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parent.parent/'review126_pf_efficiency'),
             str(Path(__file__).resolve().parent.parent/'review128_pf_heading')]
from native_pf import NativeCartesianPF, lib
from cartesian import transition
from heading import diffuse
import rbpf_ctrv as rb


def direction_tensor(means,covariances):
    second=covariances+means[:,:,None]*means[:,None,:]
    trace=np.trace(second,axis1=1,axis2=2)
    result=np.broadcast_to(np.eye(2)*.5,second.shape).copy()
    np.divide(second,trace[:,None,None],out=result,where=trace[:,None,None]>0)
    return result


class PolarNoisePF(NativeCartesianPF):
    def predict(self,lost=False):
        self._points_cache=None;pf=self.pf
        height=max(self.shape_mean[1,0],1e-3)
        qv=(pf._std_weight_velocity*height)**2
        lib.predict(self.n,self.means,self.covariances,transition(self.omega,pf.dt),
                    (pf._std_weight_position*height)**2,0.,self.shape_mean,self.shape_cov,pf.dt,int(lost))
        # C receives qvel=0 for Cartesian velocity; restore the unchanged height noise.
        self.shape_cov[1,1,1]+=qv
        if qv:
            self.covariances[:,2:,2:]+=qv*direction_tensor(self.means[:,2:],self.covariances[:,2:,2:])
        diffuse(self.means,self.covariances,pf._std_theta)
        self.omega+=self.rng.standard_normal(self.n)*pf._std_omega
        return self.estimate()


@contextmanager
def installed(enabled=True):
    previous=rb.ConditionalPF
    rb.ConditionalPF=PolarNoisePF if enabled else NativeCartesianPF
    try:yield
    finally:rb.ConditionalPF=previous
