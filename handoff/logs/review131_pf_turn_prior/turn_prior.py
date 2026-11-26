"""Physical-time OU turn prior, fixed by BDD train calibration before tracking."""
from contextlib import contextmanager
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review126_pf_efficiency'))
from native_pf import NativeCartesianPF,lib
from cartesian import transition
import rbpf_ctrv as rb

PARAMETERS=json.loads((HERE/'turn_prior.json').read_text())


def coefficients(seconds):
    rate=PARAMETERS['lambda_per_second'];variance=PARAMETERS['stationary_variance_rad2_per_second2']
    rho=np.exp(-rate*seconds)
    return float(rho),float(np.sqrt(variance*(-np.expm1(-2*rate*seconds)))*seconds),float(np.sqrt(variance)*seconds)


class TurnPriorPF(NativeCartesianPF):
    seconds=1/25
    def __init__(self,pf,mean,covariance,particles,rng):
        assert pf.dt==1.,'This calibrated state represents angular increment per released frame.'
        super().__init__(pf,mean,covariance,particles,rng)
        self.rho,self.noise_std,initial_std=coefficients(self.seconds)
        self.omega*=initial_std/pf._init_std_omega
        self._points_cache=None

    def predict(self,lost=False):
        self._points_cache=None;pf=self.pf;height=max(self.shape_mean[1,0],1e-3)
        lib.predict(self.n,self.means,self.covariances,transition(self.omega,pf.dt),
                    (pf._std_weight_position*height)**2,(pf._std_weight_velocity*height)**2,
                    self.shape_mean,self.shape_cov,pf.dt,int(lost))
        self.omega=self.rho*self.omega+self.noise_std*self.rng.standard_normal(self.n)
        return self.estimate()


@contextmanager
def installed(enabled=True,frame_seconds=1/25):
    previous=rb.ConditionalPF
    class PeriodPF(TurnPriorPF):
        seconds=frame_seconds
    rb.ConditionalPF=PeriodPF if enabled else NativeCartesianPF
    try:yield
    finally:rb.ConditionalPF=previous
