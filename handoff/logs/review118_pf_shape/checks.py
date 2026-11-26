"""Check CV equivalence, coordinated turns and numerical covariance stability."""

from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from pf_ctrv import PFCTRV
from cartesian import CartesianPF, transition, installed
import rbpf_ctrv


def run():
    dt = .4
    matrix = transition(np.array([0, 1e-10, -1e-10]), dt)
    reference = np.eye(4)
    reference[0, 2] = reference[1, 3] = dt
    assert np.allclose(matrix, reference, atol=1e-10)
    turn = transition(np.array([np.pi/2]), 1)[0]
    state = turn @ np.array([0., 0., 1., 0.])
    assert np.allclose(state, [2/np.pi, 2/np.pi, 0, 1], atol=1e-12)
    pf = PFCTRV(num_particles=20)
    pf._std_omega = 0
    mean, covariance = pf.initiate(np.array([50, 70, 1.5, 80]))
    particles = np.tile(mean, (20, 1))
    filter = CartesianPF(pf, mean, covariance, particles, np.random.default_rng(3))
    filter.means[:, 2:] = [2, -.5]
    linear_mean = filter.means[0].copy()
    linear_cov = filter.covariances[0].copy()
    for i in range(12):
        height = filter.shape_mean[1, 0]
        f = transition(np.array([0.]), pf.dt)[0]
        q = np.diag(np.square([pf._std_weight_position*height]*2 + [pf._std_weight_velocity*height]*2))
        linear_mean = f @ linear_mean
        linear_cov = f @ linear_cov @ f.T + q
        filter.predict()
        assert np.allclose(filter.means, linear_mean) and np.allclose(filter.covariances, linear_cov)
        z = np.array([52+2*i, 70-.5*i, 1.5, 80.])
        r = pf._measurement_noise(80)[:2, :2]
        gain = np.linalg.solve(linear_cov[:2, :2]+r, linear_cov[:2, :]).T
        linear_mean += gain @ (z[:2]-linear_mean[:2])
        correction = np.eye(4)
        correction[:, :2] -= gain
        linear_cov = correction @ linear_cov @ correction.T + gain @ r @ gain.T
        m, p = filter.update(z)
        assert np.allclose(filter.means, linear_mean) and np.allclose(filter.covariances, linear_cov)
        assert np.linalg.eigvalsh(p).min() >= -1e-8 and np.isfinite(m).all()
    filter.predict(lost=True)
    assert filter.shape_mean[1, 1] == 0 and filter.shape_cov[1, 1, 1] > 0
    original = rbpf_ctrv.ConditionalPF
    with installed():
        assert rbpf_ctrv.ConditionalPF is CartesianPF
    assert rbpf_ctrv.ConditionalPF is original
    print("PASS CV Kalman equivalence, analytic circular motion, zero-rate continuity, PSD and hook restoration")


if __name__ == "__main__":
    run()
