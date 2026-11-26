"""R75 resampling recentering: preserve mean, RNG stream, and old default."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ekf_ctrv import normalize_angle
from pf_ctrv import PFCTRV
from run_tracking import load_tracker_cfg, make_tracker


class Recenter(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.X = np.tile([100., 80., 4., 3.10, .02, .8, 40., .01, .04], (500, 1))
        self.X += rng.normal(size=(500, 9)) * [.6, .6, .3, .08, .01, .02, .6, .001, .01]
        self.X[:, 3] = normalize_angle(self.X[:, 3])
        weights = np.linspace(.01, 1.0, 500) ** 8
        self.logw = np.log(weights / weights.sum())

    def test_recenter_preserves_posterior_mean_and_rng_draw_count(self):
        plain = PFCTRV(num_particles=500, roughening=1., roughening_motion_scale=2.)
        centered = PFCTRV(num_particles=500, roughening=1., roughening_motion_scale=2.,
                          recenter_resample=True)
        a, b = np.random.default_rng(9), np.random.default_rng(9)
        x0, w0, did0 = plain.resample_if_needed(self.X, self.logw, a)
        x1, w1, did1 = centered.resample_if_needed(self.X, self.logw, b)
        self.assertTrue(did0 and did1)
        np.testing.assert_array_equal(plain.last_resample_idx, centered.last_resample_idx)
        np.testing.assert_array_equal(w0, w1)
        self.assertEqual(a.random(), b.random())
        target, _ = centered.estimate_state(self.X, self.logw)
        achieved, _ = centered.estimate_state(x1, w1)
        np.testing.assert_allclose(achieved[[0, 1, 2, 4, 5, 6, 7, 8]],
                                   target[[0, 1, 2, 4, 5, 6, 7, 8]], atol=1e-10)
        self.assertLess(abs(normalize_angle(achieved[3] - target[3])), 1e-10)
        self.assertGreater(np.max(np.abs(x0 - x1)), 0.0)

    def test_default_parity_and_yaml(self):
        a = PFCTRV(num_particles=500, roughening=1., roughening_motion_scale=2.)
        b = PFCTRV(num_particles=500, roughening=1., roughening_motion_scale=2.,
                   recenter_resample=False)
        x0, w0, _ = a.resample_if_needed(self.X, self.logw, np.random.default_rng(9))
        x1, w1, _ = b.resample_if_needed(self.X, self.logw, np.random.default_rng(9))
        np.testing.assert_array_equal(x0, x1)
        np.testing.assert_array_equal(w0, w1)
        cfg = load_tracker_cfg("configs/phase2/trackers/r75/recenter.yaml", {})
        self.assertTrue(make_tracker(cfg).kalman_filter.recenter_resample)
        with self.assertRaises(ValueError):
            PFCTRV(recenter_resample=True, estimate="map")


if __name__ == "__main__":
    unittest.main()
