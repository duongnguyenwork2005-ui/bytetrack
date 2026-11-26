"""R76 opt-in bound; old R72 bound and default remain intact."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pf_ctrv import PFCTRV
from run_tracking import load_tracker_cfg, make_tracker


class HighMotion(unittest.TestCase):
    def test_bound_opt_in(self):
        self.assertEqual(PFCTRV().roughening_motion_scale, 1.)
        with self.assertRaises(ValueError):
            PFCTRV(roughening_motion_scale=2.1)
        for value in (3., 4., 6., 8.):
            self.assertEqual(PFCTRV(roughening_motion_scale=value,
                                    allow_high_motion_roughening=True).roughening_motion_scale, value)
        with self.assertRaises(ValueError):
            PFCTRV(roughening_motion_scale=8.1, allow_high_motion_roughening=True)

    def test_yaml_integration(self):
        for scale in (300, 400):
            cfg = load_tracker_cfg(f"configs/phase2/trackers/r76/motion{scale}.yaml", {})
            pf = make_tracker(cfg).kalman_filter
            self.assertTrue(pf.allow_high_motion_roughening)
            self.assertEqual(pf.roughening_motion_scale, scale / 100)
            self.assertFalse(pf.recenter_resample)


if __name__ == "__main__":
    unittest.main()
