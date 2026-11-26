"""R94 one-step straight transition retains latent omega, including K1 prior."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from run_tracking import load_tracker_cfg, make_tracker
from test_pf_ctrv import scene
from tracker_ctrv import PFSTrack
from ul_backend import TrackState


def main():
    cfg = load_tracker_cfg("configs/phase2/trackers/r94/lost_cv50_preserve_omega.yaml", {})
    tracker = make_tracker(cfg)
    tracker.set_seed(0, "r94-fixture")
    for frame in (1, 2, 3):
        tracker.update(scene(frame, gap_every=0).cpu().numpy(), None)
    tracks = list(tracker.tracked_stracks)
    assert len(tracks) == 3

    a, b, c = (copy.deepcopy(tracks) for _ in range(3))
    b.reverse()
    for group in (a, b, c):
        for st in group:
            st.state = TrackState.Lost
    PFSTrack.multi_predict(a)
    PFSTrack.multi_predict(b)
    for st in c:
        st.predict()
    for x, y, z in zip(a, reversed(b), c):
        np.testing.assert_array_equal(x.particles, y.particles)
        np.testing.assert_array_equal(x.particles, z.particles)
        np.testing.assert_array_equal(x._k1_prior[0], z._k1_prior[0])
        np.testing.assert_array_equal(x._k1_prior[1], z._k1_prior[1])

    retained, reset = (copy.deepcopy(tracks) for _ in range(2))
    for group, keep in ((retained, True), (reset, False)):
        for st in group:
            st.state = TrackState.Lost
            st.kalman_filter.pf.lost_cv_fraction = 1.0
            st.kalman_filter.pf.lost_cv_preserve_omega = keep
    original_omega = [st.particles[:, st.kalman_filter.pf.OMEGA].copy() for st in tracks]
    PFSTrack.multi_predict(retained)
    PFSTrack.multi_predict(reset)
    for keep, drop, omega in zip(retained, reset, original_omega):
        ix = keep.kalman_filter.pf.OMEGA
        np.testing.assert_array_equal(keep.particles[:, :ix], drop.particles[:, :ix])
        np.testing.assert_allclose(keep.particles[:, ix] - drop.particles[:, ix], omega, atol=1e-12)
        np.testing.assert_allclose(keep._k1_prior[0][:, ix] - drop._k1_prior[0][:, ix], omega, atol=1e-12)
        np.testing.assert_array_equal(keep._k1_prior[1], drop._k1_prior[1])
    print("R94 preserve-turn fixtures passed", flush=True)


if __name__ == "__main__":
    main()
