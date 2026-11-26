"""Research-only innovation-based R inflation for box shape, not position."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'review118_pf_shape'))
from cartesian import CartesianPF
import rbpf_ctrv as rb


def inflate(residual, variance):
    return np.maximum(1., (4. + np.square(residual) / variance) / 5.)


class ShapeRobustPF(CartesianPF):
    def update(self, observation, confidence=None):
        pf = self.pf
        original = pf._measurement_noise
        noise = original(max(float(observation[3]), 1e-3), confidence).copy()
        residual = np.asarray(observation)[2:] - self.shape_mean[:, 0]
        factors = inflate(residual, self.shape_cov[:, 0, 0] + np.diag(noise)[2:])
        noise[2, 2] *= factors[0]
        noise[3, 3] *= factors[1]
        # pf is shared across tracks, but all updates are synchronous. Restore
        # the exact instance dictionary, including when the parent raises.
        had_override = '_measurement_noise' in pf.__dict__
        old_override = pf.__dict__.get('_measurement_noise')
        pf._measurement_noise = lambda *args, **kwargs: noise.copy()
        try:
            return super().update(observation, confidence)
        finally:
            if had_override:
                pf._measurement_noise = old_override
            else:
                del pf._measurement_noise


@contextmanager
def installed(enabled=True):
    previous = rb.ConditionalPF
    rb.ConditionalPF = ShapeRobustPF if enabled else CartesianPF
    try:
        yield
    finally:
        rb.ConditionalPF = previous
