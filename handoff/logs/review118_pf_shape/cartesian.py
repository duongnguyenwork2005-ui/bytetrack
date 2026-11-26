"""Marginalized coordinated-turn filter with one sampled angular coordinate."""

from contextlib import contextmanager

import numpy as np
import rbpf_ctrv as rb
from pf_ctrv import systematic_resample


def transition(omega, dt):
    angle = omega * dt
    sine = dt * np.sinc(angle / np.pi)
    cosine = .5 * omega * dt**2 * np.sinc(angle / (2*np.pi))**2
    matrix = np.repeat(np.eye(4)[None], len(omega), axis=0)
    matrix[:, 0, 2], matrix[:, 0, 3] = sine, -cosine
    matrix[:, 1, 2], matrix[:, 1, 3] = cosine, sine
    matrix[:, 2, 2] = matrix[:, 3, 3] = np.cos(angle)
    matrix[:, 2, 3], matrix[:, 3, 2] = -np.sin(angle), np.sin(angle)
    return matrix


class CartesianPF:
    def __init__(self, pf, mean, covariance, initial_particles, rng):
        self.pf, self.rng, self.n = pf, rng, len(initial_particles)
        self.omega = initial_particles[:, pf.OMEGA].copy()
        self.means = np.zeros((self.n, 4))
        self.means[:, :2] = mean[:2]
        self.covariances = np.zeros((self.n, 4, 4))
        self.covariances[:, :2, :2] = covariance[:2, :2]
        self.covariances[:, 2, 2] = self.covariances[:, 3, 3] = covariance[pf.V, pf.V]
        self.shape_mean = np.array([[mean[pf.A], mean[pf.VA]], [mean[pf.H], mean[pf.VH]]])
        self.shape_cov = np.array([covariance[np.ix_([pf.A, pf.VA], [pf.A, pf.VA])],
                                   covariance[np.ix_([pf.H, pf.VH], [pf.H, pf.VH])]])
        self.logw = np.full(self.n, -np.log(self.n))
        self.updates = self.resamples = 0

    def full_particles(self):
        pf = self.pf
        particles = np.zeros((self.n, 9))
        particles[:, :2] = self.means[:, :2]
        particles[:, pf.V] = np.hypot(self.means[:, 2], self.means[:, 3])
        particles[:, pf.THETA] = np.arctan2(self.means[:, 3], self.means[:, 2])
        particles[:, pf.OMEGA] = self.omega
        particles[:, pf.A], particles[:, pf.VA] = self.shape_mean[0]
        particles[:, pf.H], particles[:, pf.VH] = self.shape_mean[1]
        return particles

    def estimate(self):
        pf = self.pf
        points = self.full_particles()
        mean, covariance = pf.estimate_state(points, self.logw)
        weights = np.exp(self.logw)
        # Delta-method covariance for the polar compatibility view.
        jacobian = np.zeros((self.n, 4, 4))
        jacobian[:, 0, 0] = jacobian[:, 1, 1] = 1
        speed = np.maximum(points[:, pf.V], 1e-6)
        jacobian[:, 2, 2:] = self.means[:, 2:] / speed[:, None]
        jacobian[:, 3, 2] = -self.means[:, 3] / speed**2
        jacobian[:, 3, 3] = self.means[:, 2] / speed**2
        within = jacobian @ self.covariances @ jacobian.transpose(0, 2, 1)
        indices = [pf.CX, pf.CY, pf.V, pf.THETA]
        covariance[np.ix_(indices, indices)] += np.einsum("n,nij->ij", weights, within)
        covariance[np.ix_([pf.A, pf.VA], [pf.A, pf.VA])] += self.shape_cov[0]
        covariance[np.ix_([pf.H, pf.VH], [pf.H, pf.VH])] += self.shape_cov[1]
        return mean, (covariance + covariance.T) * .5

    def predict(self, lost=False):
        pf = self.pf
        if lost:
            self.shape_mean[1, 1] = 0
            self.shape_cov[1, 1, :] = self.shape_cov[1, :, 1] = 0
        matrix = transition(self.omega, pf.dt)
        self.means = np.einsum("nij,nj->ni", matrix, self.means)
        self.covariances = matrix @ self.covariances @ matrix.transpose(0, 2, 1)
        height = max(self.shape_mean[1, 0], 1e-3)
        process_std = np.array([pf._std_weight_position*height]*2 + [pf._std_weight_velocity*height]*2)
        indexes = np.arange(4)
        self.covariances[:, indexes, indexes] += process_std**2
        shape_transition = np.array([[1, pf.dt], [0, 1]])
        self.shape_mean = self.shape_mean @ shape_transition.T
        self.shape_cov = shape_transition @ self.shape_cov @ shape_transition.T
        self.shape_cov[:, np.arange(2), np.arange(2)] += np.array(
            [[1e-2, 1e-5], [pf._std_weight_position*height, pf._std_weight_velocity*height]])**2
        self.shape_mean[1, 0] = max(self.shape_mean[1, 0], 1e-3)
        self.omega += self.rng.standard_normal(self.n) * pf._std_omega
        return self.estimate()

    def update(self, observation, confidence=None):
        pf = self.pf
        observation = np.asarray(observation, dtype=float)
        noise = pf._measurement_noise(max(observation[3], 1e-3), confidence)
        innovation = self.covariances[:, :2, :2] + noise[None, :2, :2]
        residual = observation[None, :2] - self.means[:, :2]
        chol = np.linalg.cholesky(innovation)
        whitened = np.linalg.solve(chol, residual[..., None])[..., 0]
        self.logw += -.5 * np.sum(whitened**2, axis=1) - np.log(
            np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
        self.logw -= self.logw.max()
        self.logw -= np.log(np.exp(self.logw).sum())
        cross = self.covariances[:, :, :2]
        gain = np.linalg.solve(innovation, cross.transpose(0, 2, 1)).transpose(0, 2, 1)
        self.means += np.einsum("nij,nj->ni", gain, residual)
        correction = np.repeat(np.eye(4)[None], self.n, axis=0)
        correction[:, :, :2] -= gain
        self.covariances = correction @ self.covariances @ correction.transpose(0, 2, 1) + gain @ noise[:2, :2] @ gain.transpose(0, 2, 1)
        self.covariances = (self.covariances + self.covariances.transpose(0, 2, 1)) * .5
        shape_noise = np.diag(noise)[2:]
        shape_gain = self.shape_cov[:, :, 0] / (self.shape_cov[:, 0, 0] + shape_noise)[:, None]
        self.shape_mean += shape_gain * (observation[2:] - self.shape_mean[:, 0])[:, None]
        correction = np.repeat(np.eye(2)[None], 2, axis=0)
        correction[:, :, 0] -= shape_gain
        self.shape_cov = (correction @ self.shape_cov @ correction.transpose(0, 2, 1) +
                          shape_gain[:, :, None] * shape_gain[:, None, :] * shape_noise[:, None, None])
        self.shape_cov = (self.shape_cov + self.shape_cov.transpose(0, 2, 1)) * .5
        self.shape_mean[1, 0] = max(self.shape_mean[1, 0], 1e-3)
        result = self.estimate()
        self.updates += 1
        pf.n_update += 1
        if pf.ess(self.logw) < pf.resample_ess_ratio * self.n:
            selected = systematic_resample(np.exp(self.logw), float(self.rng.random()))
            self.means = self.means[selected].copy()
            self.covariances = self.covariances[selected].copy()
            self.omega = self.omega[selected].copy()
            self.logw.fill(-np.log(self.n))
            self.resamples += 1
            pf.n_resample += 1
        return result

    def initialize_motion(self, mean, covariance, observation, n_frames, ref_pos):
        pf = self.pf
        updated, _, ok = pf.try_initiate_from_motion(mean, covariance, observation, n_frames, ref_pos)
        if not ok:
            return mean, covariance, False
        angle, speed = updated[pf.THETA], updated[pf.V]
        direction = np.array([np.cos(angle), np.sin(angle)])
        across = np.array([-direction[1], direction[0]])
        self.means[:, 2:] = speed * direction
        self.covariances[:, 2:, :] = self.covariances[:, :, 2:] = 0
        speed_variance = (2 * pf._std_weight_velocity * max(mean[pf.H], 1e-3) * 10)**2
        self.covariances[:, 2:, 2:] = (speed_variance * np.outer(direction, direction) +
                                      (speed * pf._heading_std_after_init)**2 * np.outer(across, across))
        self.logw.fill(-np.log(self.n))
        mean, covariance = self.estimate()
        return mean, covariance, True


@contextmanager
def installed(enabled=True):
    previous = rb.ConditionalPF
    if enabled:
        rb.ConditionalPF = CartesianPF
    try:
        yield
    finally:
        rb.ConditionalPF = previous
