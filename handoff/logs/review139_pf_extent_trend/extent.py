"""A damped linear Gaussian extent trend while the track is Lost."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'review132_pf_predicted_noise'))
from predicted_noise import PredictedNoisePF
from native_pf import lib
from cartesian import transition
import rbpf_ctrv as rb


def damped_transition(dt,horizon):
    if not np.isfinite(dt) or dt<=0 or np.isnan(horizon) or horizon<0:
        raise ValueError('Positive dt and nonnegative damping horizon required')
    if horizon==0:return np.array([[1.,0.],[0.,0.]])
    if np.isinf(horizon):return np.array([[1.,dt],[0.,1.]])
    phi=float(np.exp(-dt/horizon))
    integrated=float(-horizon*np.expm1(-dt/horizon))
    return np.array([[1.,integrated],[0.,phi]])


class ExtentTrendPF(PredictedNoisePF):
    horizon=30.
    def predict(self,lost=False):
        self._points_cache=None
        pf=self.pf;height=max(self.shape_mean[1,0],1e-3)
        qpos=(pf._std_weight_position*height)**2;qvel=(pf._std_weight_velocity*height)**2
        if lost:
            shape_m=self.shape_mean[1].copy();shape_p=self.shape_cov[1].copy()
        lib.predict(self.n,self.means,self.covariances,transition(self.omega,pf.dt),qpos,qvel,
                    self.shape_mean,self.shape_cov,pf.dt,0)
        if lost:
            matrix=damped_transition(pf.dt,self.horizon)
            self.shape_mean[1]=matrix@shape_m
            self.shape_cov[1]=matrix@shape_p@matrix.T+np.diag([qpos,qvel])
            self.shape_mean[1,0]=max(self.shape_mean[1,0],1e-3)
        self.omega+=self.rng.standard_normal(self.n)*pf._std_omega
        return self.estimate()


@contextmanager
def installed(enabled=True,lost_horizon_frames=30):
    previous=rb.ConditionalPF
    class PeriodPF(ExtentTrendPF):pass
    PeriodPF.horizon=float(lost_horizon_frames)
    rb.ConditionalPF=PeriodPF if enabled else PredictedNoisePF
    try:yield
    finally:rb.ConditionalPF=previous
