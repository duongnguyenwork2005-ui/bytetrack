import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(OUT / "ultralytics_settings"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from pf_ctrv import PFCTRV, PFTrackView
from pf_postgap import installed, recenter_velocity
import run_tracking as rt


class PostGapTests(unittest.TestCase):
    def test_weighted_velocity_and_preservation(self):
        pf = PFCTRV(num_particles=30)
        rng = np.random.default_rng(10)
        particles = rng.normal(size=(30, 9))
        particles[:, pf.H] = 40
        logw = np.linspace(-10, 0, 30)
        before = particles.copy()
        updated = recenter_velocity(pf, particles, logw, [3, -7])
        w = np.exp(logw - logw.max()); w /= w.sum()
        def velocity(x):
            return x[:, pf.V, None] * np.column_stack((np.cos(x[:, pf.THETA]), np.sin(x[:, pf.THETA])))
        np.testing.assert_allclose(w @ velocity(updated), [3, -7], atol=1e-12)
        np.testing.assert_allclose(velocity(updated) - w @ velocity(updated),
                                   velocity(before) - w @ velocity(before), atol=1e-12)
        np.testing.assert_array_equal(particles, before)
        np.testing.assert_array_equal(updated[:, [0, 1, 4, 5, 6, 7, 8]], before[:, [0, 1, 4, 5, 6, 7, 8]])
        m, cov = pf.estimate_state(updated, logw)
        assert np.isfinite(m).all() and np.isfinite(cov).all()
        assert np.linalg.eigvalsh(cov).min() > -1e-10

    def test_gap_timing_one_update_no_rng_or_weight_changes(self):
        pf = PFCTRV(num_particles=30, dt=.5)
        rng = np.random.default_rng(0)
        x = rng.normal(size=(30, 9)); x[:, pf.H] = 30
        track = SimpleNamespace(frame_id=1, track_id=5, particles=x, logw=np.linspace(-2, 0, 30),
                                rng=rng, _last_obs_xy=(0., 0.), _motion_initialized=True)
        original = PFTrackView.update
        calls = []
        def identity_update(view, mean, covariance, measurement, confidence=None):
            calls.append(1)
            view.track._k1_prior = None
            return pf.estimate_state(view.track.particles, view.track.logw)
        PFTrackView.update = identity_update
        try:
            state = copy.deepcopy(rng.bit_generator.state)
            weights = track.logw.copy()
            with installed() as events:
                view = PFTrackView(pf, track)
                def step(frame, position):
                    track._pending_frame_id = frame
                    view.update(None, None, np.array([*position, 1, 30]))
                    track._last_obs_xy = position
                step(2, (2., 0.))             # ordinary consecutive: no effect
                assert len(events) == 0
                step(6, (10., 0.))            # gap: wait for next real observation
                assert len(events) == 0
                step(7, (13., 4.))            # delta=(3,4)/.5
                assert len(events) == 1 and events[0]["vx"] == 6 and events[0]["vy"] == 8
                step(8, (16., 8.))            # one event only, no repeated reupdate
                step(12, (24., 8.))
                step(13, (24., 8.))           # stationary: no heading reset
                assert len(events) == 1
                step(18, (26., 8.))
                step(20, (27., 8.))           # another gap instead of consecutive
                assert len(events) == 1
            assert len(calls) == 8
            assert rng.bit_generator.state == state
            np.testing.assert_array_equal(weights, track.logw)
            assert PFTrackView.update is identity_update
        finally:
            PFTrackView.update = original

    def test_restore_on_exception(self):
        init, update = PFTrackView.__init__, PFTrackView.update
        with self.assertRaises(RuntimeError):
            with installed():
                raise RuntimeError("fixture")
        assert PFTrackView.__init__ is init and PFTrackView.update is update
        with installed(False):
            assert PFTrackView.__init__ is init and PFTrackView.update is update

    def test_real_tracker_fixture_and_reproducibility(self):
        import test_pf_ctrv as old
        class Synthetic:
            orig_shape = (540, 960)
            def iter_frames(self):
                for f in range(1, 41):
                    yield f, (f - 1) / 25., False, np.asarray(old.scene(f).data)
        cfg = rt.load_tracker_cfg(ROOT / "configs/phase2/trackers/r72/motion200.yaml", {"track_buffer": 30})
        def run(enabled):
            sink = {}
            with installed(enabled) as events:
                result = rt.track_video(Synthetic(), cfg, .04, "skip", 0, "r107_fixture", log_sink=sink)
            tracker = sink["tracker"]
            for track in tracker.tracked_stracks + tracker.lost_stracks:
                np.testing.assert_allclose(track.mean, track.kalman_filter.pf.estimate_state(track.particles, track.logw)[0], atol=1e-10)
                if track.frame_id == tracker.frame_id:
                    assert track._k1_prior is None
            return result[0].to_numpy(float), events
        b1, _ = run(False)
        c1, events = run(True)
        c2, repeated = run(True)
        b2, _ = run(False)
        np.testing.assert_array_equal(b1, b2)
        np.testing.assert_array_equal(c1, c2)
        assert events == repeated and len(events) > 0
        assert np.isfinite(c1).all()


if __name__ == "__main__":
    if "--regression" in sys.argv:
        import test_pf_ctrv as old
        tests = [getattr(old, name) for name in dir(old) if name.startswith("test_") and not name.startswith("test_11_")]
        for test in tests:
            test()
        assert all(passed for _, passed in old._RESULTS)
        print("REGRESSION", len(tests), "functions", len(old._RESULTS), "assertions")
    else:
        unittest.main()
