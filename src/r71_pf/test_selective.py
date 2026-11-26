"""R71 targeted tests for selectable box roughening."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pf_ctrv import PFCTRV
from run_tracking import load_tracker_cfg, make_tracker

class SelectiveRoughening(unittest.TestCase):
    def test_default_preserves_full_roughening_and_rng(self):
        filters = [PFCTRV(num_particles=8, roughening=1.0),
                   PFCTRV(num_particles=8, roughening=1.0, roughening_box_scale=1.0)]
        X = np.tile(np.array([100., 90., 3., .2, .03, 1., 40., 0., 0.]), (8, 1))
        X[:, 0] += np.arange(8)
        logw = np.log(np.array([.8] + [.2 / 7] * 7))
        outputs = []
        for pf in filters:
            rng = np.random.default_rng(7)
            outputs.append((*pf.resample_if_needed(X, logw, rng), rng.random()))
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
        self.assertEqual(outputs[0][2:], outputs[1][2:])

    def test_box_zero_keeps_resampled_boxes_and_motion_jitter(self):
        X = np.tile(np.array([100., 90., 3., .2, .03, 1., 40., 0., 0.]), (8, 1))
        X[:, 0] += np.arange(8)
        logw = np.log(np.array([.8] + [.2 / 7] * 7))
        pf = PFCTRV(num_particles=8, roughening=1.0, roughening_box_scale=0.0)
        out, _, did = pf.resample_if_needed(X, logw, np.random.default_rng(7))
        self.assertTrue(did)
        np.testing.assert_array_equal(out[:, [pf.CX, pf.CY, pf.A, pf.H]],
                                      X[pf.last_resample_idx][:, [pf.CX, pf.CY, pf.A, pf.H]])
        self.assertTrue(np.any(out[:, pf.V] != X[pf.last_resample_idx, pf.V]))

    def test_yaml_and_invalid_scale(self):
        cfg = load_tracker_cfg("configs/phase2/trackers/r71/k2box050.yaml", {})
        tracker = make_tracker(cfg)
        self.assertEqual(tracker.kalman_filter.roughening_box_scale, 0.5)
        for value in (-0.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                PFCTRV(roughening_box_scale=value)

if __name__ == "__main__":
    unittest.main()
