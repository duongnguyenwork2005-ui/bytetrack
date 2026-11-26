"""Synthetic audit. Run with the project Python. Does not open dataset files."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from pf_ctrv import PFCTRV, PFTrackView
from test_pf_ctrv import boxes
from run_tracking import make_tracker, load_tracker_cfg
from ul_backend import backend_name


def init_probe():
    pf = PFCTRV(num_particles=500)
    m, P = pf.initiate([100., 100., 1., 40.])
    cloud = np.tile(m, (500, 1))
    cloud[:250, pf.CX] -= 4
    cloud[250:, pf.CX] += 4
    weights = np.log(np.repeat([0.75 / 250, 0.25 / 250], 250))
    track = SimpleNamespace(particles=cloud, logw=weights.copy(), rng=np.random.default_rng(0))
    before, covariance = pf.estimate_state(cloud, weights)
    after, _, ok = PFTrackView(pf, track).try_initiate_from_motion(
        before, covariance, [103., 100., 1., 40.], ref_pos=(100., 100.))
    assert ok and np.array_equal(cloud[:, pf.CX], track.particles[:, pf.CX])
    return dict(ess_before=pf.ess(weights), ess_after=pf.ess(track.logw),
                cx_before=before[pf.CX], cx_after=after[pf.CX])


def k1_reference():
    # Heterogeneous Q, nonuniform prior weights. Disable resampling to inspect
    # posterior weights and compare with the predictive Gaussian density.
    pf = PFCTRV(num_particles=500, optimal_proposal=True, resample_ess_ratio=0)
    rng = np.random.default_rng(91)
    z = np.array([100., 120., 1., 40.])
    m, P = pf.initiate(z)
    X = pf.sample_initial(m, P, rng)
    X, prior = pf.propagate(X, rng, return_prior=True)
    idx = [pf.CX, pf.CY, pf.A, pf.H]
    mu = prior[0][:, idx]
    Q = prior[1][:, idx] ** 2
    R = np.diag(pf._measurement_noise(z[3], None))
    w = rng.uniform(0.5, 1.5, len(X))
    w /= w.sum()
    evidence = np.prod(np.exp(-0.5 * (z-mu)**2/(Q+R))/np.sqrt(2*np.pi*(Q+R)), axis=1)
    posterior_w = w * evidence
    posterior_w /= posterior_w.sum()
    new, lw, _, _ = pf.update_particles(X.copy(), np.log(w), z, None, rng, prior=prior)
    standard = (new[:,idx] - (mu + Q/(Q+R)*(z-mu))) / np.sqrt(Q*R/(Q+R))
    delta = float(np.max(np.abs(posterior_w - np.exp(lw))))
    assert delta < 1e-12 and pf.n_k1_update == 1
    return dict(max_weight_error=delta, residual_mean=standard.mean(axis=0).tolist(),
                residual_std=standard.std(axis=0).tolist())


def delayed_start():
    """Observe the unmodified PF500 wrapper when motion init is delayed."""
    records = []
    original = PFTrackView.try_initiate_from_motion

    def observe(view, *args, **kwargs):
        before = view.track.logw.copy()
        cx = view.pf.estimate_state(view.track.particles, before)[0][view.pf.CX]
        result = original(view, *args, **kwargs)
        if result[2]:
            records.append(dict(seed=seed, frame=frame, ess=view.pf.ess(before),
                                uniform=bool(np.ptp(before) == 0),
                                dx=float(result[0][view.pf.CX]-cx)))
        return result

    path = ROOT / "configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml"
    with patch.object(PFTrackView, "try_initiate_from_motion", observe):
        for seed in range(30):
            tr = make_tracker(load_tracker_cfg(path))
            tr.set_seed(seed, "r69-stop")
            for frame in range(1, 9):
                cx = 150 + (3 if frame == 8 else 0)
                tr.update(boxes([[cx-20, 165, cx+20, 195, .9, 3]]).cpu().numpy(), None)
    return records


def run(path, overrides, seed, turn, observation_noise=None):
    yaml = path if path == "bytetrack.yaml" else ROOT / "configs" / path
    tr = make_tracker(load_tracker_cfg(yaml, overrides))
    if hasattr(tr, "set_seed"):
        tr.set_seed(seed, "r69")
    cx, cy, angle = 150., 180., 0.
    prediction = []
    prior_spread_px = []
    distinct_ancestors = []
    initial_id = None
    original_predict = tr.multi_predict

    def observe(pool):
        original_predict(pool)
        for track in pool:
            if track.track_id == initial_id and frame == 29:
                x, y, w, h = track.tlwh
                prediction.append(float(np.hypot(x+w/2-cx, y+h/2-cy)))
                if observation_noise is not None and hasattr(track, "particles"):
                    prior_spread_px.append(float(np.sqrt(np.trace(track.covariance[:2, :2]))))

    tr.multi_predict = observe
    for frame in range(1, 30):
        if turn and frame >= 10:
            angle += 0.06
        cx += 3 * np.cos(angle)
        cy += 3 * np.sin(angle)
        mx, my, width, height = cx, cy, 40., 30.
        if observation_noise is not None:
            dx, dy, dw, dh = observation_noise[frame-1]
            mx, my, width, height = cx+dx, cy+dy, width+dw, height+dh
        rows = [] if 23 <= frame <= 28 else [[mx-width/2, my-height/2, mx+width/2, my+height/2, .9, 3]]
        out = tr.update(boxes(rows).cpu().numpy(), None)
        if frame == 1:
            initial_id = int(out[0,4])
        if observation_noise is not None and 2 <= frame <= 22 and hasattr(tr.kalman_filter, "last_resample_idx"):
            assert any(int(row[4]) == initial_id for row in out)
            indices = tr.kalman_filter.last_resample_idx
            n = tr.kalman_filter.num_particles
            distinct_ancestors.append(1. if indices is None else np.unique(indices).size / n)
    record = dict(error_px=prediction, same_id=any(int(row[4]) == initial_id for row in out),
                k1=getattr(tr.kalman_filter,"n_k1_update",0),
                fallback=getattr(tr.kalman_filter,"n_k1_fallback",0))
    assert len(prediction) == 1
    if prior_spread_px:
        record.update(prior_spread_px=prior_spread_px[0],
                      median_ancestor_fraction=float(np.median(distinct_ancestors)))
    return record


if __name__ == "__main__":
    pf_path = "phase2/trackers/bytetrack_pf_ctrv_n500.yaml"
    cases = {"KF": ("bytetrack.yaml", {}),
             "EKF": ("bytetrack_ekf_ctrv.yaml", {}), "UKF": ("bytetrack_ukf_ctrv.yaml", {}),
             "PF500": (pf_path, {}), "PF500_K1": (pf_path, {"pf_optimal_proposal": True}),
             "PF500_K2": (pf_path, {"pf_roughening": 0.25}),
             "PF1000": (pf_path, {"pf_num_particles": 1000})}
    if sys.argv[1:] == ["--noisy"]:
        # Same noisy measurements for both paths and every tracker/seed.
        noise = np.random.default_rng(6901).normal(size=(29, 4)) * [1., 1., 1.5, 1.5]
        trials = {}
        for turn, scene in ((False, "straight"), (True, "turn")):
            trials[scene] = {}
            for name in ("KF", "EKF", "UKF", "PF500", "PF500_K1", "PF1000"):
                path, overrides = cases[name]
                seeds = range(30) if name.startswith("PF") else [0]
                trials[scene][name] = [run(path, overrides, seed, turn, noise) for seed in seeds]
        print(json.dumps({"backend": backend_name(), "observation_noise_seed": 6901,
                          "pf_seeds": list(range(30)), "scenes": trials}, indent=2))
        sys.exit(0)
    result = {"backend": backend_name(), "init": init_probe(), "k1_math": k1_reference(),
              "delayed_start": delayed_start(), "scenes": {}}
    for turn, scene in ((False,"straight"),(True,"turn")):
        result["scenes"][scene] = {name: [run(path, ov, seed, turn) for seed in
                                  (range(3) if name.startswith("PF") else [0])]
                                  for name, (path, ov) in cases.items()}
    print(json.dumps(result, indent=2))
