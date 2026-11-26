"""R92 transition-mixture fixtures, including per-track RNG and K1 prior."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pf_ctrv import PFCTRV
from run_tracking import load_tracker_cfg, make_tracker
from test_pf_ctrv import scene
from tracker_ctrv import PFSTrack
from ul_backend import TrackState


def main():
    for value in (-.1, 1.1, float("nan")):
        try:
            PFCTRV(lost_cv_fraction=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {value}")
    state = np.array([[10., 20., 4., .2, .4, 1., 30., 0., 0.],
                      [10., 20., 4., .2, .4, 1., 30., 0., 0.]])
    straight = PFCTRV(num_particles=2, lost_cv_fraction=1.)
    original = state.copy()
    mixed = straight.lost_transition_input(state, np.random.default_rng(2))
    np.testing.assert_array_equal(state, original)
    assert np.all(mixed[:, straight.OMEGA] == 0)
    predicted = straight.f_batch(mixed)
    np.testing.assert_allclose(predicted[:, straight.CX], 10 + 4 * np.cos(.2) * straight.dt)
    np.testing.assert_allclose(predicted[:, straight.CY], 20 + 4 * np.sin(.2) * straight.dt)
    turn = PFCTRV(num_particles=2, lost_cv_fraction=0.)
    rng = np.random.default_rng(9)
    before = copy.deepcopy(rng.bit_generator.state)
    assert turn.lost_transition_input(state, rng) is state
    assert rng.bit_generator.state == before
    assert not np.array_equal(turn.f_batch(state)[:, :2], predicted[:, :2])

    cfg = load_tracker_cfg("configs/phase2/trackers/r92/lost_cv50.yaml", {})
    tracker = make_tracker(cfg)
    tracker.set_seed(0, "r92-fixture")
    for frame in (1, 2, 3):
        tracker.update(scene(frame, gap_every=0).cpu().numpy(), None)
    tracks = list(tracker.tracked_stracks)
    assert len(tracks) == 3
    a, b, c = (copy.deepcopy(tracks) for _ in range(3))
    b.reverse()
    for group in (a, b, c):
        for t in group:
            t.state = TrackState.Lost
    PFSTrack.multi_predict(a)
    PFSTrack.multi_predict(b)
    for t in c:
        t.predict()
    for x, y, z in zip(a, reversed(b), c):
        np.testing.assert_array_equal(x.particles, y.particles)
        np.testing.assert_array_equal(x.particles, z.particles)
        np.testing.assert_array_equal(x._k1_prior[0], z._k1_prior[0])
        np.testing.assert_array_equal(x._k1_prior[1], z._k1_prior[1])
    print("R92 transition mixture fixtures passed", flush=True)


if __name__ == "__main__":
    main()
