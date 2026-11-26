"""Numerical regression tests for the new conditional Gaussian representation."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
settings = Path(__file__).resolve().parent / "ultralytics_settings"
settings.mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(settings)
import numpy as np
from pf_ctrv import PFCTRV
from rbpf import ConditionalPF, installed
from tracker_ctrv import PFSTrack


def test_conditional_kalman():
    pf = PFCTRV(num_particles=32)
    pf._std_theta = pf._std_omega = 0.0
    mean, cov = pf.initiate(np.array([30.0, 50.0, 1.2, 40.0]))
    mean[pf.THETA], mean[pf.OMEGA], mean[pf.V] = 0.4, 0.03, 2.0
    cov[pf.THETA, pf.THETA] = cov[pf.OMEGA, pf.OMEGA] = 0.0
    rng = np.random.default_rng(112)
    initial = np.repeat(mean[None], 32, axis=0)
    filt = ConditionalPF(pf, mean, cov, initial, rng)
    m, p = mean[filt.linear].copy(), cov[np.ix_(filt.linear, filt.linear)].copy()
    theta, omega = mean[pf.THETA], mean[pf.OMEGA]
    observation_matrix = np.eye(7)[filt.observed]
    for step in range(30):
        std = pf.process_noise_std_batch(np.array([max(m[4], 1e-3)]))[0, filt.linear]
        transition = np.eye(7)
        transition[0, 2] = (np.sin(theta + omega) - np.sin(theta)) / omega
        transition[1, 2] = (np.cos(theta) - np.cos(theta + omega)) / omega
        transition[3, 5] = transition[4, 6] = 1.0
        m, p = transition @ m, transition @ p @ transition.T + np.diag(std ** 2)
        theta += omega
        actual_m, actual_p = filt.predict()
        np.testing.assert_allclose(actual_m[filt.linear], m, atol=1e-10)
        np.testing.assert_allclose(actual_p[np.ix_(filt.linear, filt.linear)], p, atol=1e-9)
        z = observation_matrix @ m + np.array([0.2, -0.1, 0.01, 0.4])
        r = pf._measurement_noise(z[3], None)
        s = observation_matrix @ p @ observation_matrix.T + r
        gain = np.linalg.solve(s, observation_matrix @ p).T
        m += gain @ (z - observation_matrix @ m)
        c = np.eye(7) - gain @ observation_matrix
        p = c @ p @ c.T + gain @ r @ gain.T
        actual_m, actual_p = filt.update(z)
        np.testing.assert_allclose(actual_m[filt.linear], m, atol=1e-9)
        np.testing.assert_allclose(actual_p[np.ix_(filt.linear, filt.linear)], p, atol=1e-9)
        assert np.linalg.eigvalsh(actual_p).min() > -1e-8
    print("PASS conditional Kalman reference, 30 prediction/update pairs")


def test_angles_lost_and_repeat():
    outputs = []
    for _ in range(2):
        pf = PFCTRV(num_particles=64)
        mean, cov = pf.initiate(np.array([30.0, 50.0, 1.0, 40.0]))
        rng = np.random.default_rng(113)
        particles = pf.sample_initial(mean, cov, rng)
        particles[:, pf.THETA] = np.tile([np.pi - 0.01, -np.pi + 0.01], 32)
        filt = ConditionalPF(pf, mean, cov, particles, rng)
        assert abs(abs(filt.estimate()[0][pf.THETA]) - np.pi) < 0.03
        for step in range(12):
            m, p = filt.predict(lost=step > 5)
            if step <= 5:
                m, p = filt.update(np.array([30.0 + step, 50.0, 1.0, 40.0]))
            assert np.isfinite(m).all() and np.isfinite(p).all()
            assert np.linalg.eigvalsh(p).min() > -1e-8
        outputs.append((m, p, filt.means.copy(), filt.covariances.copy()))
    for first, second in zip(*outputs):
        np.testing.assert_array_equal(first, second)
    print("PASS angle wrap, missing observations, covariance PSD, exact repeat")


def test_plugin_restore():
    method = PFSTrack.__dict__["multi_predict"]
    try:
        with installed():
            assert PFSTrack.__dict__["multi_predict"] is not method
            raise RuntimeError("test restore")
    except RuntimeError:
        pass
    assert PFSTrack.__dict__["multi_predict"] is method
    print("PASS plugin restore on exception")


if __name__ == "__main__":
    test_conditional_kalman()
    test_angles_lost_and_repeat()
    test_plugin_restore()
