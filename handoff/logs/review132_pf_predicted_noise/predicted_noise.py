"""Use the predicted common shape height for Gaussian measurement noise."""
from contextlib import contextmanager
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'review126_pf_efficiency'))
from native_pf import NativeCartesianPF
import rbpf_ctrv as rb


class PredictedNoisePF(NativeCartesianPF):
    def update(self,observation,confidence=None):
        pf=self.pf
        height=max(float(self.shape_mean[1,0]),1e-3)
        maker=pf._measurement_noise
        had_local='_measurement_noise' in pf.__dict__
        local=pf.__dict__.get('_measurement_noise')
        pf._measurement_noise=lambda ignored_height,confidence=None:maker(height,confidence)
        try:
            return super().update(observation,confidence)
        finally:
            if had_local:pf._measurement_noise=local
            else:del pf._measurement_noise


@contextmanager
def installed(enabled=True):
    previous=rb.ConditionalPF
    rb.ConditionalPF=PredictedNoisePF if enabled else NativeCartesianPF
    try:yield
    finally:rb.ConditionalPF=previous
