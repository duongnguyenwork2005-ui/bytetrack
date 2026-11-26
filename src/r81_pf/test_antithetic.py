"""R81: check process-noise marginals, reproducibility, and tracker prediction paths."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pf_ctrv import PFCTRV
from tracker_ctrv import PFSTrack
from run_tracking import load_tracker_cfg, make_tracker
from test_pf_ctrv import scene


def main() -> None:
    n = 500
    std = np.full((n, 9), 3.0)
    pf = PFCTRV(num_particles=n, antithetic_process_noise=True)
    draws = [pf.sample_process_noise(std, np.random.default_rng(seed)) for seed in range(200)]
    for noise in draws:
        assert np.array_equal(noise[:n // 2], -noise[n // 2:])
        assert np.max(np.abs(noise.mean(axis=0))) < 1e-13
    # Marginal variance of each of the first 500 particles is still 3^2.
    observed = np.var(np.stack(draws)[:, :n // 2, :], axis=(0, 1)).mean()
    assert abs(observed - 9.0) < 0.15, observed
    odd = pf.sample_process_noise(np.ones((5, 9)), np.random.default_rng(2))
    assert np.array_equal(odd[:2], -odd[2:4]) and odd.shape == (5, 9)

    cfg = load_tracker_cfg("configs/phase2/trackers/r72/motion200.yaml",
                           {"pf_antithetic_process_noise": True})
    tr = make_tracker(cfg)
    tr.set_seed(0, "r81-fixture")
    for frame in (1, 2, 3):
        tr.update(scene(frame, gap_every=0).cpu().numpy(), None)
    tracks = list(tr.tracked_stracks)
    assert len(tracks) == 3
    a, b, c = copy.deepcopy(tracks), copy.deepcopy(tracks)[::-1], copy.deepcopy(tracks)
    PFSTrack.multi_predict(a)
    PFSTrack.multi_predict(b)
    for track in c:
        track.predict()
    assert all(np.array_equal(x.particles, y.particles) and
               np.array_equal(x.particles, z.particles) for x, y, z in zip(a, b[::-1], c))
    print("R81 antithetic: 4/4 checks passed", flush=True)


if __name__ == "__main__":
    main()
