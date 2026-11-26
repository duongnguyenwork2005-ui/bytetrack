"""Mean-only height-velocity reset during Lost; retain conditional uncertainty."""
from contextlib import contextmanager
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'review132_pf_predicted_noise'))
from predicted_noise import PredictedNoisePF
from native_pf import lib
from cartesian import transition
import rbpf_ctrv as rb


class LostShapePF(PredictedNoisePF):
    def predict(self,lost=False):
        self._points_cache=None
        pf=self.pf;height=max(self.shape_mean[1,0],1e-3)
        matrix=transition(self.omega,pf.dt)
        if lost:self.shape_mean[1,1]=0.
        # Native lost=1 also erases the h-vh covariance. Mean was reset above;
        # lost=0 retains that uncertainty while applying the identical F/Q.
        lib.predict(self.n,self.means,self.covariances,matrix,
                    (pf._std_weight_position*height)**2,(pf._std_weight_velocity*height)**2,
                    self.shape_mean,self.shape_cov,pf.dt,0)
        self.omega+=self.rng.standard_normal(self.n)*pf._std_omega
        return self.estimate()


@contextmanager
def installed(enabled=True):
    previous=rb.ConditionalPF
    rb.ConditionalPF=LostShapePF if enabled else PredictedNoisePF
    try:yield
    finally:rb.ConditionalPF=previous
