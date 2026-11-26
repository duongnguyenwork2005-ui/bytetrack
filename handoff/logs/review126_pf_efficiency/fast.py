"""Same Cartesian RBPF model with reused views and a guarded 2x2 solver."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'review118_pf_shape'))
from cartesian import CartesianPF
import rbpf_ctrv as rb
from pf_ctrv import systematic_resample


def solve2(innovation, residual, cross):
    """Cholesky whitening and general 2x2 solve, with ill-condition fallback."""
    a, b = innovation[:, 0, 0], innovation[:, 0, 1]
    c, d = innovation[:, 1, 0], innovation[:, 1, 1]
    determinant = a*d-b*c
    if np.any((a <= 0) | ~np.isfinite(determinant) |
              (determinant <= 32*np.finfo(float).eps*(np.abs(a*d)+np.abs(b*c)))):
        chol = np.linalg.cholesky(innovation)
        white = np.linalg.solve(chol, residual[..., None])[..., 0]
        gain = np.linalg.solve(innovation, cross.transpose(0, 2, 1)).transpose(0, 2, 1)
        return white, np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1), gain
    l0 = np.sqrt(a)
    l10 = c/l0
    pivot = d-l10*l10
    if np.any(pivot <= 0):
        chol = np.linalg.cholesky(innovation)
        white = np.linalg.solve(chol, residual[..., None])[..., 0]
        gain = np.linalg.solve(innovation, cross.transpose(0, 2, 1)).transpose(0, 2, 1)
        return white, np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1), gain
    l1 = np.sqrt(pivot)
    white = np.empty_like(residual)
    white[:, 0] = residual[:, 0]/l0
    white[:, 1] = (residual[:, 1]-l10*white[:, 0])/l1
    gain = np.empty_like(cross)
    gain[:, :, 0] = (d[:, None]*cross[:, :, 0]-b[:, None]*cross[:, :, 1])/determinant[:, None]
    gain[:, :, 1] = (a[:, None]*cross[:, :, 1]-c[:, None]*cross[:, :, 0])/determinant[:, None]
    return white, np.log(l0)+np.log(l1), gain


class FastCartesianPF(CartesianPF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pf = self.pf
        self._points_cache = None
        self._jacobian = np.zeros((self.n, 4, 4))
        self._jacobian[:, 0, 0] = self._jacobian[:, 1, 1] = 1
        self._shape_indices = (np.ix_([pf.A, pf.VA], [pf.A, pf.VA]),
                               np.ix_([pf.H, pf.VH], [pf.H, pf.VH]))
        assert [pf.CX, pf.CY, pf.V, pf.THETA] == list(range(4))
        self._eye4 = np.broadcast_to(np.eye(4), (self.n, 4, 4))
        self._eye2 = np.broadcast_to(np.eye(2), (2, 2, 2))

    def _points(self):
        if self._points_cache is None:
            self._points_cache = super().full_particles()
        return self._points_cache

    def full_particles(self):
        # Preserve the original public method's ownership: callers can mutate
        # the returned view without corrupting a future estimate.
        return self._points().copy()

    def estimate(self):
        pf = self.pf
        points = self._points()
        mean, covariance = pf.estimate_state(points, self.logw)
        weights = np.exp(self.logw)
        jacobian = self._jacobian
        speed = np.maximum(points[:, pf.V], 1e-6)
        jacobian[:, 2, 2:] = self.means[:, 2:]/speed[:, None]
        jacobian[:, 3, 2] = -self.means[:, 3]/speed**2
        jacobian[:, 3, 3] = self.means[:, 2]/speed**2
        within = jacobian @ self.covariances @ jacobian.transpose(0, 2, 1)
        covariance[:4, :4] += np.einsum('n,nij->ij', weights, within)
        covariance[self._shape_indices[0]] += self.shape_cov[0]
        covariance[self._shape_indices[1]] += self.shape_cov[1]
        return mean, (covariance+covariance.T)*.5

    def predict(self, lost=False):
        self._points_cache = None
        return super().predict(lost)

    def update(self, observation, confidence=None):
        self._points_cache = None
        pf = self.pf
        observation = np.asarray(observation, dtype=float)
        noise = pf._measurement_noise(max(observation[3], 1e-3), confidence)
        innovation = self.covariances[:, :2, :2]+noise[None, :2, :2]
        residual = observation[None, :2]-self.means[:, :2]
        white, log_chol, gain = solve2(innovation, residual, self.covariances[:, :, :2])
        self.logw += -.5*np.sum(white**2, axis=1)-log_chol
        self.logw -= self.logw.max()
        self.logw -= np.log(np.exp(self.logw).sum())
        self.means += np.einsum('nij,nj->ni', gain, residual)
        correction = self._eye4.copy()
        correction[:, :, :2] -= gain
        self.covariances = (correction @ self.covariances @ correction.transpose(0, 2, 1)
                            + gain @ noise[:2, :2] @ gain.transpose(0, 2, 1))
        self.covariances = (self.covariances+self.covariances.transpose(0, 2, 1))*.5
        shape_noise = np.diag(noise)[2:]
        shape_gain = self.shape_cov[:, :, 0]/(self.shape_cov[:, 0, 0]+shape_noise)[:, None]
        self.shape_mean += shape_gain*(observation[2:]-self.shape_mean[:, 0])[:, None]
        correction = self._eye2.copy()
        correction[:, :, 0] -= shape_gain
        self.shape_cov = (correction @ self.shape_cov @ correction.transpose(0, 2, 1)
                          + shape_gain[:, :, None]*shape_gain[:, None, :]*shape_noise[:, None, None])
        self.shape_cov = (self.shape_cov+self.shape_cov.transpose(0, 2, 1))*.5
        self.shape_mean[1, 0] = max(self.shape_mean[1, 0], 1e-3)
        result = self.estimate()
        self.updates += 1
        pf.n_update += 1
        if pf.ess(self.logw) < pf.resample_ess_ratio*self.n:
            selected = systematic_resample(np.exp(self.logw), float(self.rng.random()))
            self.means = self.means[selected].copy()
            self.covariances = self.covariances[selected].copy()
            self.omega = self.omega[selected].copy()
            self.logw.fill(-np.log(self.n))
            self._points_cache = None
            self.resamples += 1
            pf.n_resample += 1
        return result

    def initialize_motion(self, *args, **kwargs):
        self._points_cache = None
        return super().initialize_motion(*args, **kwargs)


@contextmanager
def installed(fast=True):
    previous = rb.ConditionalPF
    rb.ConditionalPF = FastCartesianPF if fast else CartesianPF
    try:
        yield
    finally:
        rb.ConditionalPF = previous
