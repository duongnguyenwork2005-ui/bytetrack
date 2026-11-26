"""R70 regression checks for the selectable PF measurement-height reference.

Run with the khoaluan-mot Python from the repository root. Uses synthetic boxes only.
The observation/default path is compared with the exact pre-R70 Git implementation.
"""
from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import tracker_ctrv
from pf_ctrv import PFCTRV
from run_tracking import load_tracker_cfg, make_tracker
from test_pf_ctrv import scene

BASE_COMMIT = "8f3b63df15ff68943befebbf4e31739f90032349"
PF_YAML = ROOT / "configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml"
MEASUREMENT_INDICES = [PFCTRV.CX, PFCTRV.CY, PFCTRV.A, PFCTRV.H]


def load_baseline(name):
    source = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:src/{name}.py"], cwd=ROOT).decode("utf-8")
    module = types.ModuleType(f"r70_baseline_{name}")
    module.__file__ = str(ROOT / "src" / f"{name}.py")
    exec(compile(source, f"{BASE_COMMIT}:src/{name}.py", "exec"), module.__dict__)
    return module


def filter_trace(pf):
    rng = pf.make_rng(7)
    mean, covariance = pf.initiate([100., 150., 1.2, 40.])
    particles = pf.sample_initial(mean, covariance, rng)
    logw = np.full(pf.num_particles, -np.log(pf.num_particles))
    rows = []
    for frame in range(1, 45):
        particles, prior = pf.propagate(particles, rng, return_prior=True)
        if 20 <= frame <= 25:
            continue
        measurement = np.array([100. + 3*frame, 150. + frame, 1.2, 40. + 2*np.sin(frame)])
        if frame == 40:
            measurement[:2] += 900
        particles, logw, mean, covariance = pf.update_particles(
            particles, logw, measurement, None, rng, prior=prior)
        rows.append(tuple(array.copy() for array in (particles, logw, mean, covariance)))
    counters = (pf.n_update, pf.n_resample, pf.n_degenerate, pf.n_k1_update, pf.n_k1_fallback)
    return rows, counters, copy.deepcopy(rng.bit_generator.state)


def tracker_trace(registry, overrides):
    tracker = make_tracker(load_tracker_cfg(PF_YAML, overrides), registry=registry)
    tracker.set_seed(3, "r70-measurement-height")
    rows = []
    for frame in range(1, 36):
        output = tracker.update(scene(frame, gap_every=5).cpu().numpy(), None)
        tracks = sorted(tracker.tracked_stracks + tracker.lost_stracks, key=lambda t: t.track_id)
        rows.append(np.asarray(output).copy().tobytes())
        for track in tracks:
            rows.append((track.track_id, track.particles.tobytes(), track.logw.tobytes(),
                         track.mean.tobytes(), track.covariance.tobytes(),
                         copy.deepcopy(track.rng.bit_generator.state)))
    return rows, tracker


class MeasurementHeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_pf = load_baseline("pf_ctrv")
        with patch.dict(sys.modules, {"pf_ctrv": cls.old_pf}):
            cls.old_tracker = load_baseline("tracker_ctrv")

    def test_default_and_explicit_observation_match_prechange_filter(self):
        for optimal_proposal in (False, True):
            for roughening in (0., 1.):
                kwargs = dict(num_particles=100, seed=5, optimal_proposal=optimal_proposal,
                              roughening=roughening)
                expected, counters, rng = filter_trace(self.old_pf.PFCTRV(**kwargs))
                self.assertGreater(counters[1], 0)
                self.assertGreater(counters[2], 0)
                for option in ({}, {"measurement_height": "observation"}):
                    with self.subTest(k1=optimal_proposal, k2=roughening, option=option):
                        actual, new_counters, new_rng = filter_trace(PFCTRV(**kwargs, **option))
                        self.assertEqual(counters, new_counters)
                        self.assertEqual(rng, new_rng)
                        self.assertEqual(len(actual), len(expected))
                        for first, second in zip(actual, expected):
                            for a, b in zip(first, second):
                                np.testing.assert_array_equal(a, b)

    def test_default_and_explicit_observation_match_prechange_tracker(self):
        old_registry = {"bytetrack_pf_ctrv": self.old_tracker.PFByteTracker}
        for k1, k2 in ((False, 0.), (True, 0.), (True, 1.)):
            overrides = {"pf_optimal_proposal": k1, "pf_roughening": k2}
            expected, _ = tracker_trace(old_registry, overrides)
            for option in ({}, {"pf_measurement_height": "observation"}):
                with self.subTest(k1=k1, k2=k2, option=option):
                    actual, _ = tracker_trace(None, overrides | option)
                    self.assertEqual(actual, expected)

    def test_prediction_bootstrap_uses_weighted_height_before_update(self):
        pf = PFCTRV(num_particles=3, measurement_height="prediction", estimate="map",
                    resample_ess_ratio=0., degenerate_maha2=1e6)
        particles = np.zeros((3, 9))
        particles[:, MEASUREMENT_INDICES] = [[100., 120., 1.2, 20.],
                                             [101., 121., 1.2, 40.],
                                             [99., 119., 1.2, 80.]]
        weights = np.array([.1, .2, .7])
        measurement = np.array([101., 119., 1.2, 40.])
        expected_height = 66.
        expected_R = np.diag(np.square([expected_height / 20, expected_height / 20,
                                        .1, expected_height / 20]))
        scores = np.log(weights) + multivariate_normal.logpdf(
            particles[:, MEASUREMENT_INDICES], mean=measurement, cov=expected_R)
        expected_weights = np.exp(scores - logsumexp(scores))
        with patch.object(pf, "_measurement_noise", wraps=pf._measurement_noise) as measurement_noise:
            _, actual_logw, _, _ = pf.update_particles(
                particles.copy(), np.log(weights), measurement, None, np.random.default_rng(4))
        measurement_noise.assert_called_once()
        self.assertAlmostEqual(measurement_noise.call_args.args[0], expected_height)
        np.testing.assert_allclose(np.exp(actual_logw), expected_weights, rtol=1e-12, atol=1e-15)

    def test_prediction_k1_weights_and_samples_use_same_predicted_R(self):
        pf = PFCTRV(num_particles=3, measurement_height="prediction", optimal_proposal=True,
                    resample_ess_ratio=0., degenerate_maha2=1e6)
        transition_mean = np.zeros((3, 9))
        transition_mean[:, MEASUREMENT_INDICES] = [[100., 120., 1.2, 20.],
                                                   [101., 121., 1.2, 40.],
                                                   [99., 119., 1.2, 80.]]
        transition_std = np.repeat(np.array([[1.], [2.], [3.]]), 9, axis=1)
        predicted_particles = transition_mean + .25 * transition_std
        weights = np.array([.1, .2, .7])
        measurement = np.array([101., 119., 1.2, 35.])
        predicted_height = weights @ predicted_particles[:, pf.H]
        R = np.diag(np.square([predicted_height / 20, predicted_height / 20,
                               .1, predicted_height / 20]))
        scores = []
        expected_particles = predicted_particles.copy()
        sample_noise = np.random.default_rng(8).standard_normal((3, 4))
        for i in range(3):
            Q = np.diag(transition_std[i, MEASUREMENT_INDICES] ** 2)
            mu = transition_mean[i, MEASUREMENT_INDICES]
            scores.append(np.log(weights[i]) + multivariate_normal.logpdf(measurement, mu, Q + R))
            gain = Q @ np.linalg.inv(Q + R)
            conditional_mean = mu + gain @ (measurement - mu)
            conditional_cov = Q - gain @ Q
            expected_particles[i, MEASUREMENT_INDICES] = (
                conditional_mean + sample_noise[i] * np.sqrt(np.diag(conditional_cov)))
        scores = np.asarray(scores)
        expected_weights = np.exp(scores - logsumexp(scores))
        with patch.object(pf, "log_likelihood", wraps=pf.log_likelihood) as likelihood:
            with patch.object(pf, "optimal_proposal_step", wraps=pf.optimal_proposal_step) as proposal:
                actual_particles, logw, _, _ = pf.update_particles(
                    predicted_particles.copy(), np.log(weights), measurement, None,
                    np.random.default_rng(8), prior=(transition_mean, transition_std))
        np.testing.assert_array_equal(likelihood.call_args.args[2], np.diag(R))
        np.testing.assert_array_equal(proposal.call_args.args[3], np.diag(R))
        np.testing.assert_allclose(np.exp(logw), expected_weights, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(actual_particles, expected_particles, rtol=1e-12, atol=1e-12)
        self.assertEqual((pf.n_k1_update, pf.n_k1_fallback), (1, 0))

    def test_prediction_R_controls_degeneracy_threshold(self):
        particles = np.zeros((2, 9))
        particles[:, MEASUREMENT_INDICES] = [100., 100., 1., 80.]
        logw = np.full(2, -np.log(2))
        measurement = np.array([100., 100., 1., 40.])
        for mode, expected in (("observation", 1), ("prediction", 0)):
            with self.subTest(mode=mode):
                pf = PFCTRV(num_particles=2, measurement_height=mode, degenerate_maha2=200.)
                pf.update_particles(particles.copy(), logw.copy(), measurement, None,
                                    np.random.default_rng(3))
                self.assertEqual(pf.n_degenerate, expected)

    def test_prediction_height_floor_and_nonfinite_failure(self):
        particles = np.zeros((2, 9))
        particles[:, PFCTRV.H] = -1.
        pf = PFCTRV(num_particles=2, measurement_height="prediction")
        with patch.object(pf, "_measurement_noise", wraps=pf._measurement_noise) as measurement_noise:
            pf.update_particles(particles.copy(), np.zeros(2), np.array([0., 0., 0., 1.]),
                                None, np.random.default_rng(1))
        self.assertEqual(measurement_noise.call_args.args[0], 1e-3)
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                particles[:, PFCTRV.H] = value
                with self.assertRaisesRegex(ValueError, "finite weighted particle height"):
                    pf.update_particles(particles.copy(), np.zeros(2), np.array([0., 0., 0., 1.]),
                                        None, np.random.default_rng(1))

    def test_prediction_option_reaches_normal_tracker_update(self):
        overrides = {"pf_measurement_height": "prediction", "pf_optimal_proposal": True}
        tracker = make_tracker(load_tracker_cfg(PF_YAML, overrides))
        tracker.set_seed(4, "r70-routing")
        self.assertEqual(tracker.kalman_filter.measurement_height, "prediction")
        tracker.update(scene(1, gap_every=0).cpu().numpy(), None)
        kf = tracker.kalman_filter
        with patch.object(kf, "_measurement_noise", wraps=kf._measurement_noise) as measurement_noise:
            tracker.update(scene(2, gap_every=0).cpu().numpy(), None)
        self.assertGreater(measurement_noise.call_count, 0)
        self.assertGreater(kf.n_k1_update, 0)
        tracker.reset()
        self.assertEqual(tracker.kalman_filter.measurement_height, "prediction")

    def test_unknown_mode_is_rejected_at_constructor_and_yaml_entry(self):
        with self.assertRaisesRegex(ValueError, "measurement_height"):
            PFCTRV(measurement_height="wrong")
        with self.assertRaisesRegex(ValueError, "measurement_height"):
            make_tracker(load_tracker_cfg(PF_YAML, {"pf_measurement_height": "wrong"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
