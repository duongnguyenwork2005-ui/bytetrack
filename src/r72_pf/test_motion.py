"""R72 motion roughening and ancestor diagnostic checks."""
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from pf_ctrv import PFCTRV
from run_tracking import load_tracker_cfg,make_tracker

class MotionRoughening(unittest.TestCase):
    def test_default_is_bit_identical_and_ancestor_count_is_real(self):
        X=np.tile(np.array([100.,90.,3.,.2,.03,1.,40.,0.,0.]),(8,1))
        X[:,0]+=np.arange(8)
        logw=np.log(np.array([.8]+[.2/7]*7))
        a=PFCTRV(num_particles=8,roughening=1.)
        b=PFCTRV(num_particles=8,roughening=1.,roughening_motion_scale=1.)
        for pf in (a,b): pf.enable_diag()
        ra,rb=np.random.default_rng(8),np.random.default_rng(8)
        xa,wa,sa=a.resample_if_needed(X,logw,ra)
        xb,wb,sb=b.resample_if_needed(X,logw,rb)
        np.testing.assert_array_equal(xa,xb)
        np.testing.assert_array_equal(wa,wb)
        self.assertEqual(sa,sb)
        self.assertEqual(ra.random(),rb.random())
        self.assertLess(len(np.unique(a.last_resample_idx)),len(xa))
        self.assertEqual(len(np.unique(xa[:,a.CX])),len(xa))

    def test_motion_zero_keeps_resampled_motion_and_yaml(self):
        X=np.tile(np.array([100.,90.,3.,.2,.03,1.,40.,0.,0.]),(8,1))
        logw=np.log(np.array([.8]+[.2/7]*7))
        pf=PFCTRV(num_particles=8,roughening=1.,roughening_motion_scale=0.)
        out,_,did=pf.resample_if_needed(X,logw,np.random.default_rng(8))
        self.assertTrue(did)
        ix=[pf.V,pf.THETA,pf.OMEGA,pf.VA,pf.VH]
        np.testing.assert_allclose(out[:,ix],X[pf.last_resample_idx][:,ix],rtol=0,atol=1e-15)
        cfg=load_tracker_cfg("configs/phase2/trackers/r72/motion200.yaml",{})
        self.assertEqual(make_tracker(cfg).kalman_filter.roughening_motion_scale,2.)
        for value in (-.1,2.1,np.nan):
            with self.assertRaises(ValueError): PFCTRV(roughening_motion_scale=value)

if __name__=="__main__": unittest.main()
