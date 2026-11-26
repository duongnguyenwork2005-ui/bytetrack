"""Experimental CTRV RBPF with per-particle conditional Gaussian states.

Only heading and turn rate are sampled. Conditional on their history, the other
seven state coordinates have a linear Gaussian transition and observation.
Installed only inside a context manager in an isolated replay process.
"""

from contextlib import contextmanager
import numpy as np
from pf_ctrv import PFTrackView, systematic_resample
from ekf_ctrv import normalize_angle
from tracker_ctrv import PFSTrack
from ul_backend import TrackState


class ConditionalPF:
    def __init__(self, pf, mean, covariance, initial_particles, rng):
        self.pf, self.rng = pf, rng
        self.linear = [pf.CX, pf.CY, pf.V, pf.A, pf.H, pf.VA, pf.VH]
        self.angular = [pf.THETA, pf.OMEGA]
        self.observed = [0, 1, 3, 4]
        self.n = len(initial_particles)
        self.angles = initial_particles[:, self.angular].copy()
        self.means = np.repeat(mean[None, self.linear], self.n, axis=0)
        block = covariance[np.ix_(self.linear, self.linear)]
        self.covariances = np.repeat(block[None, :, :], self.n, axis=0)
        self.logw = np.full(self.n, -np.log(self.n))
        self.updates = self.resamples = 0

    def full_particles(self):
        particles = np.zeros((self.n, 9))
        particles[:, self.linear] = self.means
        particles[:, self.angular] = self.angles
        return particles

    def estimate(self):
        particles = self.full_particles()
        mean, covariance = self.pf.estimate_state(particles, self.logw)
        weights = np.exp(self.logw)
        within = np.einsum("n,nij->ij", weights, self.covariances)
        covariance[np.ix_(self.linear, self.linear)] += within
        return mean, (covariance + covariance.T) * 0.5

    def predict(self, lost=False):
        pf, dt = self.pf, self.pf.dt
        if lost:
            self.means[:, 6] = 0.0
            self.covariances[:, 6, :] = 0.0
            self.covariances[:, :, 6] = 0.0
        theta, omega = self.angles.T
        turning = np.abs(omega) > pf.omega_eps
        safe = np.where(turning, omega, 1.0)
        after = theta + omega * dt
        dx = np.where(turning, (np.sin(after) - np.sin(theta)) / safe, dt * np.cos(theta))
        dy = np.where(turning, (np.cos(theta) - np.cos(after)) / safe, dt * np.sin(theta))
        transition = np.repeat(np.eye(7)[None, :, :], self.n, axis=0)
        transition[:, 0, 2], transition[:, 1, 2] = dx, dy
        transition[:, 3, 5], transition[:, 4, 6] = dt, dt
        std = pf.process_noise_std_batch(np.maximum(self.means[:, 4], 1e-3))
        self.means = np.einsum("nij,nj->ni", transition, self.means)
        self.covariances = transition @ self.covariances @ transition.transpose(0, 2, 1)
        diagonal = np.arange(7)
        self.covariances[:, diagonal, diagonal] += std[:, self.linear] ** 2
        self.means[:, 4] = np.maximum(self.means[:, 4], 1e-3)
        angular_noise = self.rng.standard_normal((self.n, 2)) * std[:, self.angular]
        self.angles[:, 0] = normalize_angle(after + angular_noise[:, 0])
        self.angles[:, 1] += angular_noise[:, 1]
        return self.estimate()

    def update(self, observation, confidence=None):
        z = np.asarray(observation, dtype=float)
        measurement = self.pf._measurement_noise(max(z[3], 1e-3), confidence)
        obs = self.observed
        cross = self.covariances[:, :, obs]
        innovation_cov = self.covariances[:, obs, :][:, :, obs] + measurement[None]
        residual = z[None] - self.means[:, obs]
        # Positive measurement variance keeps the innovation covariance SPD.
        chol = np.linalg.cholesky(innovation_cov)
        whitened = np.linalg.solve(chol, residual[..., None])[..., 0]
        likelihood = -0.5 * (
            np.sum(whitened ** 2, axis=1)
            + 2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
        )
        gain = np.linalg.solve(innovation_cov, cross.transpose(0, 2, 1)).transpose(0, 2, 1)
        self.means += np.einsum("nij,nj->ni", gain, residual)
        self.means[:, 4] = np.maximum(self.means[:, 4], 1e-3)
        # Joseph form retains a positive semidefinite conditional covariance.
        correction = np.repeat(np.eye(7)[None], self.n, axis=0)
        correction[:, :, obs] -= gain
        self.covariances = (
            correction @ self.covariances @ correction.transpose(0, 2, 1)
            + gain @ measurement @ gain.transpose(0, 2, 1)
        )
        self.covariances = (self.covariances + self.covariances.transpose(0, 2, 1)) * 0.5
        self.logw += likelihood
        self.logw -= np.max(self.logw)
        self.logw -= np.log(np.exp(self.logw).sum())
        result = self.estimate()
        self.updates += 1
        self.pf.n_update += 1
        weights = np.exp(self.logw)
        ess = 1.0 / np.sum(weights ** 2)
        if ess < self.pf.resample_ess_ratio * self.n:
            indices = systematic_resample(weights, float(self.rng.random()))
            self.means = self.means[indices].copy()
            self.covariances = self.covariances[indices].copy()
            self.angles = self.angles[indices].copy()
            self.logw = np.full(self.n, -np.log(self.n))
            self.resamples += 1
            self.pf.n_resample += 1
        return result

    def initialize_motion(self, mean, covariance, observation, n_frames, ref_pos):
        updated, _, ok = self.pf.try_initiate_from_motion(
            mean, covariance, observation, n_frames=n_frames, ref_pos=ref_pos
        )
        if not ok:
            return mean, covariance, False
        pf = self.pf
        height = max(mean[pf.H], 1e-3)
        speed_std = 2 * pf._std_weight_velocity * height * 10
        self.means[:, 2] = updated[pf.V]
        self.covariances[:, 2, :] = 0.0
        self.covariances[:, :, 2] = 0.0
        self.covariances[:, 2, 2] = speed_std ** 2
        self.angles[:, 0] = normalize_angle(
            updated[pf.THETA] + self.rng.standard_normal(self.n) * pf._heading_std_after_init
        )
        self.logw = np.full(self.n, -np.log(self.n))
        mean, covariance = self.estimate()
        return mean, covariance, True


def sync(track):
    track.particles = track.rbpf.full_particles()
    track.logw = track.rbpf.logw.copy()
    track._k1_prior = None


@contextmanager
def installed(enabled=True):
    events = []
    if not enabled:
        yield events
        return
    original = {
        "activate": PFSTrack.activate,
        "predict": PFTrackView.predict,
        "update": PFTrackView.update,
        "motion": PFTrackView.try_initiate_from_motion,
        "multi_predict": PFSTrack.__dict__["multi_predict"],
    }
    instances = []

    def activate(track, pf, frame_id):
        original["activate"](track, pf, frame_id)
        track.rbpf = ConditionalPF(pf, track.mean, track.covariance, track.particles, track.rng)
        instances.append(track.rbpf)
        sync(track)

    def predict(view, mean, covariance, lost=None):
        track = view.track
        result = track.rbpf.predict(track.state != TrackState.Tracked if lost is None else lost)
        sync(track)
        return result

    def multi_predict(cls, tracks):
        for track in tracks:
            track.mean, track.covariance = predict(
                track.kalman_filter, track.mean, track.covariance,
                lost=track.state != TrackState.Tracked
            )

    def update(view, mean, covariance, measurement, confidence=None):
        result = view.track.rbpf.update(measurement, confidence)
        sync(view.track)
        return result

    def motion(view, mean, covariance, measurement, n_frames=1.0, ref_pos=None):
        result = view.track.rbpf.initialize_motion(
            mean, covariance, measurement, n_frames, ref_pos
        )
        sync(view.track)
        return result

    PFSTrack.activate = activate
    PFSTrack.multi_predict = classmethod(multi_predict)
    PFTrackView.predict, PFTrackView.update = predict, update
    PFTrackView.try_initiate_from_motion = motion
    try:
        yield events
    finally:
        events.append({"tracks": len(instances),
                       "updates": sum(x.updates for x in instances),
                       "resamples": sum(x.resamples for x in instances)})
        PFSTrack.activate = original["activate"]
        PFSTrack.multi_predict = original["multi_predict"]
        PFTrackView.predict, PFTrackView.update = original["predict"], original["update"]
        PFTrackView.try_initiate_from_motion = original["motion"]
