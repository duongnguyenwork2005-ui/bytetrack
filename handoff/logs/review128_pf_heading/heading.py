"""Marginalize independent heading diffusion into Cartesian velocity moments."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'review126_pf_efficiency'))
from native_pf import NativeCartesianPF, lib
from cartesian import transition
import rbpf_ctrv as rb


def diffuse(means, covariances, sigma):
    if sigma == 0:
        return
    a, b = np.exp(-.5*sigma*sigma), np.exp(-2*sigma*sigma)
    velocity = means[:, 2:].copy()
    outer = velocity[:, :, None]*velocity[:, None, :]
    second = covariances[:, 2:, 2:] + outer
    isotropic = (1-b)*np.trace(second, axis1=1, axis2=2)*.5
    result = b*second-a*a*outer
    result[:, 0, 0] += isotropic
    result[:, 1, 1] += isotropic
    covariances[:, 2:, 2:] = .5*(result+result.transpose(0, 2, 1))
    covariances[:, :2, 2:] *= a
    covariances[:, 2:, :2] *= a
    means[:, 2:] *= a


class HeadingPF(NativeCartesianPF):
    def predict(self, lost=False):
        self._points_cache = None
        pf = self.pf
        height = max(self.shape_mean[1, 0], 1e-3)
        matrix = transition(self.omega, pf.dt)
        lib.predict(self.n, self.means, self.covariances, matrix,
                    (pf._std_weight_position*height)**2,
                    (pf._std_weight_velocity*height)**2,
                    self.shape_mean, self.shape_cov, pf.dt, int(lost))
        diffuse(self.means, self.covariances, pf._std_theta)
        self.omega += self.rng.standard_normal(self.n)*pf._std_omega
        return self.estimate()


@contextmanager
def installed(enabled=True):
    previous = rb.ConditionalPF
    rb.ConditionalPF = HeadingPF if enabled else NativeCartesianPF
    try:
        yield
    finally:
        rb.ConditionalPF = previous
