"""Conservative measurement-derived velocity covariance floor at motion init."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'review132_pf_predicted_noise'))
from predicted_noise import PredictedNoisePF
import rbpf_ctrv as rb


def covariance_floor(prior,noise):
    prior=np.asarray(prior,dtype=float);noise=np.asarray(noise,dtype=float)
    if prior.shape!=(2,2) or noise.shape!=(2,2) or not np.isfinite(prior).all() or not np.isfinite(noise).all():
        raise ValueError('Two finite 2x2 covariance matrices required')
    difference=(prior-noise);difference=(difference+difference.T)*.5
    values,vectors=np.linalg.eigh(difference)
    if values.min()>=0:return prior.copy()
    result=noise+(vectors*np.maximum(values,0.))@vectors.T
    return (result+result.T)*.5


class MotionUncertaintyPF(PredictedNoisePF):
    def __init__(self,pf,mean,covariance,particles,rng):
        super().__init__(pf,mean,covariance,particles,rng)
        self.previous_noise_height=max(float(mean[pf.H]),1e-3)
        self.previous_confidence=None
        self.floor_velocity=True
        self.velocity_covariance_floors=0

    def update(self,observation,confidence=None):
        # Save the exact arguments used by R132's predicted-height noise maker;
        # computing R later avoids an extra noise-matrix allocation every frame.
        height=max(float(self.shape_mean[1,0]),1e-3)
        result=super().update(observation,confidence)
        self.previous_noise_height=height
        self.previous_confidence=None if confidence is None else float(confidence)
        return result

    def initialize_motion(self,mean,covariance,observation,n_frames,ref_pos):
        result=super().initialize_motion(mean,covariance,observation,n_frames,ref_pos)
        if not result[2] or not self.floor_velocity:return result
        pf=self.pf;elapsed=pf.dt*max(float(n_frames),1.)
        if not np.isfinite(elapsed) or elapsed<=0:raise ValueError('Positive elapsed motion time required')
        previous=pf._measurement_noise(self.previous_noise_height,self.previous_confidence)[:2,:2]
        current=pf._measurement_noise(max(float(self.shape_mean[1,0]),1e-3),None)[:2,:2]
        noise=(previous+current)/(elapsed*elapsed)
        prior=self.covariances[0,2:,2:].copy()
        assert np.array_equal(self.covariances[:,2:,2:],np.broadcast_to(prior,(self.n,2,2)))
        updated=covariance_floor(prior,noise)
        if np.array_equal(prior,updated):return result
        self.covariances[:,2:,2:]=updated
        self._points_cache=None;self.velocity_covariance_floors+=1
        mean,covariance=self.estimate()
        return mean,covariance,True


@contextmanager
def installed(enabled=True):
    previous=rb.ConditionalPF
    rb.ConditionalPF=MotionUncertaintyPF if enabled else PredictedNoisePF
    try:yield
    finally:rb.ConditionalPF=previous
