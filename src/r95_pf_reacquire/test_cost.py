"""Fixtures for conservative PF rescue and no-op ambiguous associations."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from r95_pf_reacquire.cost import posterior_similarity, rescue_costs
from ul_backend import TrackState


def track(cx=50., cy=50., state=TrackState.Lost, end_frame=5):
    x = np.zeros((8, 9))
    x[:, 0], x[:, 1], x[:, 5], x[:, 6] = cx, cy, 1., 20.
    return SimpleNamespace(particles=x, logw=np.full(8, -np.log(8)),
                           state=state, end_frame=end_frame)


def det(cx=50., cy=50., score=.9):
    return SimpleNamespace(tlwh=np.array([cx - 10, cy - 10, 20., 20.]), score=score)


def main():
    t, d = track(), det()
    assert abs(posterior_similarity(t, d) - 1.) < 1e-12
    original = np.array([[.99]], dtype=np.float32)
    before = original.copy()
    result, rescued = rescue_costs([t], [d], original, frame_id=6)
    assert len(rescued) == 1 and result[0, 0] < .8
    np.testing.assert_array_equal(original, before)

    # Already matchable rows/columns and old lost tracks must remain untouched.
    for tracks, dets, base, frame in (
        ([t], [d], np.array([[.2]]), 6),
        ([t, track(state=TrackState.Tracked)], [d], np.array([[.99], [.2]]), 6),
        ([track(end_frame=-10)], [d], np.array([[.99]]), 6),
        ([t], [d, det(200.)], np.array([[.2, .99]]), 6),
    ):
        out, selected = rescue_costs(tracks, dets, base, frame)
        assert not selected
        np.testing.assert_array_equal(out, base)

    # Ambiguous second detection or competing Lost track fails the 1.5 margin.
    for tracks, dets, base in (
        ([t], [d, det(51.)], np.full((1, 2), .99)),
        ([t, track(51.)], [d], np.full((2, 1), .99)),
    ):
        out, selected = rescue_costs(tracks, dets, base, 6)
        assert not selected
        np.testing.assert_array_equal(out, base)

    assert posterior_similarity(t, det(score=.9, cx=200.)) < .01
    print("R95 conservative PF rescue fixtures passed", flush=True)


if __name__ == "__main__":
    main()
