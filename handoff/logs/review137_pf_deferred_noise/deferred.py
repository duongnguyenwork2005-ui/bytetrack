"""Apply the existing, observation-independent omega innovation after selection."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'review132_pf_predicted_noise'))
from predicted_noise import PredictedNoisePF
from native_pf import lib
from cartesian import transition
import rbpf_ctrv as rb


class DeferredOmegaPF(PredictedNoisePF):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._pending_noise=None

    def _points(self):
        if self._points_cache is None:
            points=super()._points()
            if self._pending_noise is not None:
                points[:,self.pf.OMEGA]+=self._pending_noise
        return self._points_cache

    def _commit(self):
        if self._pending_noise is not None:
            self.omega+=self._pending_noise
            self._pending_noise=None
            self._points_cache=None

    def predict(self,lost=False):
        self._commit()
        self._points_cache=None
        pf=self.pf;height=max(self.shape_mean[1,0],1e-3)
        matrix=transition(self.omega,pf.dt)
        lib.predict(self.n,self.means,self.covariances,matrix,
                    (pf._std_weight_position*height)**2,(pf._std_weight_velocity*height)**2,
                    self.shape_mean,self.shape_cov,pf.dt,int(lost))
        # Same number and order of RNG calls as Native.predict; attach after selection.
        self._pending_noise=self.rng.standard_normal(self.n)*pf._std_omega
        return self.estimate()

    def update(self,observation,confidence=None):
        result=super().update(observation,confidence)
        # Native selection gathered oldomega, not pending_noise. Each new slot keeps
        # its independent draw, so duplicate ancestors no longer duplicate that draw.
        self._commit()
        return result


@contextmanager
def installed(enabled=True):
    previous=rb.ConditionalPF
    rb.ConditionalPF=DeferredOmegaPF if enabled else PredictedNoisePF
    try:yield
    finally:rb.ConditionalPF=previous
