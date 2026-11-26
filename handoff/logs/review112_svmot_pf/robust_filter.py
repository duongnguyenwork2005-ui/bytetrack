"""Approximate robust RBPF with a four-degree-of-freedom Student-t innovation."""

from contextlib import contextmanager
import numpy as np
import rbpf
from pf_ctrv import systematic_resample


class RobustConditionalPF(rbpf.ConditionalPF):
    def update(self, observation, confidence=None):
        z = np.asarray(observation, dtype=float)
        measurement = self.pf._measurement_noise(max(z[3], 1e-3), confidence)
        obs = self.observed
        cross = self.covariances[:, :, obs]
        marginal = self.covariances[:, obs, :][:, :, obs]
        innovation_cov = marginal + measurement[None]
        residual = z[None] - self.means[:, obs]
        chol = np.linalg.cholesky(innovation_cov)
        whitened = np.linalg.solve(chol, residual[..., None])[..., 0]
        mahalanobis = np.sum(whitened ** 2, axis=1)
        degrees = 4.0
        likelihood = (
            -np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
            -0.5 * (degrees + 4) * np.log1p(mahalanobis / degrees)
        )
        inflation = np.maximum(1.0, (degrees + mahalanobis) / (degrees + 4))
        robust_measurement = inflation[:, None, None] * measurement[None]
        robust_innovation = marginal + robust_measurement
        gain = np.linalg.solve(robust_innovation, cross.transpose(0, 2, 1)).transpose(0, 2, 1)
        self.means += np.einsum("nij,nj->ni", gain, residual)
        self.means[:, 4] = np.maximum(self.means[:, 4], 1e-3)
        correction = np.repeat(np.eye(7)[None], self.n, axis=0)
        correction[:, :, obs] -= gain
        self.covariances = (
            correction @ self.covariances @ correction.transpose(0, 2, 1)
            + gain @ robust_measurement @ gain.transpose(0, 2, 1)
        )
        self.covariances = (self.covariances + self.covariances.transpose(0, 2, 1)) * 0.5
        self.logw += likelihood
        self.logw -= np.max(self.logw)
        self.logw -= np.log(np.exp(self.logw).sum())
        result = self.estimate()
        self.updates += 1
        self.pf.n_update += 1
        weights = np.exp(self.logw)
        if 1.0 / np.sum(weights ** 2) < self.pf.resample_ess_ratio * self.n:
            indices = systematic_resample(weights, float(self.rng.random()))
            self.means = self.means[indices].copy()
            self.covariances = self.covariances[indices].copy()
            self.angles = self.angles[indices].copy()
            self.logw = np.full(self.n, -np.log(self.n))
            self.resamples += 1
            self.pf.n_resample += 1
        return result


@contextmanager
def installed(enabled=True):
    original = rbpf.ConditionalPF
    if enabled:
        rbpf.ConditionalPF = RobustConditionalPF
    try:
        with rbpf.installed(enabled) as events:
            yield events
    finally:
        rbpf.ConditionalPF = original
