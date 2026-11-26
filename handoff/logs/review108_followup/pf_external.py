"""Compare the upstream pfilter SIR update with the existing PF on the same ByteTrack inputs."""
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "pf_external"
SETTINGS = OUT / "ultralytics"
SETTINGS.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(SETTINGS)
sys.path[:0] = [str(HERE / "vendor"), str(ROOT / "src")]

import numpy as np
import pandas as pd
from pfilter import ParticleFilter
import pfilter.pfilter as upstream
import trackeval
import boot_stats
import det_cache
import run_tracking as rt
import run_trackeval as rte
from pf_ctrv import PFCTRV, systematic_resample

CACHE = "yolov8n-coco_conf0.05_all"
SPLIT = "DETRAC-all"
CFG = ROOT / "configs/phase2/trackers/r72/motion200.yaml"
VIDEOS = json.loads((ROOT / "results/phase2/manifests/detrac_dev.json").read_text())["videos"]
METRICS = ("HOTA", "IDF1", "IDSW", "FN", "FP")


@contextmanager
def use_external():
    original = PFCTRV.update_particles

    def update(pf, particles, logw, observation, confidence, rng, diag_key=None, prior=None):
        # ByteTrack has already predicted with the shared CTRV motion model.
        # Upstream performs the entire SIR correction, including its ESS decision.
        mean_likelihood = pf.log_likelihood(
            particles, observation, np.diag(pf._measurement_noise(max(observation[3], 1e-3), confidence)))
        joint = logw + mean_likelihood
        if not np.isfinite(joint).any():
            raise FloatingPointError("All SIR particle likelihoods are non-finite")
        likelihood = np.exp(np.maximum(joint - np.max(joint), -30))
        filt = getattr(pf, "external_sir", None)
        if filt is None:
            filt = ParticleFilter(prior_fn=lambda count: np.zeros((count, particles.shape[1])),
                                  n_particles=len(particles), n_eff_threshold=pf.resample_ess_ratio,
                                  resample_proportion=0)
            pf.external_sir = filt
        filt.particles = particles
        filt.weights.fill(1 / len(particles))
        filt.weight_fn = lambda hypotheses, observed: likelihood
        filt.resample_fn = lambda weights: systematic_resample(weights, float(rng.random()))
        filt.update(np.asarray(observation))
        mean, cov = pf.estimate_state(filt.original_particles, np.log(filt.original_weights))
        pf.n_update += 1
        pf.n_resample += int(filt.n_eff < filt.n_eff_threshold)
        return filt.particles, np.log(filt.weights), mean, cov

    PFCTRV.update_particles = update
    try:
        yield
    finally:
        PFCTRV.update_particles = original


def replay(video, seed, external, limit=None):
    original = det_cache.open_cache(CACHE, video, .25, [2, 3, 5, 7], 300)

    class Timeline:
        orig_shape = original.orig_shape

        def iter_frames(self):
            import itertools
            yield from itertools.islice(original.iter_frames(), limit)

    start = time.perf_counter()
    with use_external() if external else nullcontext():
        mot, _, timing = rt.track_video(Timeline(), rt.load_tracker_cfg(CFG, {"track_buffer": 30}),
                                       dt_nominal=.04, elapsed_mode="skip", seed=seed, video=video)
    assert len(timing) == (limit if limit is not None else len(original.frame_idx))
    assert not mot.duplicated(["frame", "id"]).any() and np.isfinite(mot.to_numpy(float)).all()
    return mot, len(timing), time.perf_counter() - start


def score(name):
    ec, dc, mc = rte.build_configs(SPLIT, name, VIDEOS)
    ec.update(PRINT_RESULTS=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, LOG_ON_ERROR=str(OUT / "error.log"))
    dc["TRACKERS_FOLDER"] = str(OUT / "trackers")
    metrics = [trackeval.metrics.HOTA(mc), trackeval.metrics.CLEAR(mc), trackeval.metrics.Identity(mc)]
    result, messages = trackeval.Evaluator(ec).evaluate([trackeval.datasets.MotChallenge2DBox(dc)], metrics)
    assert all(m == "Success" for m in next(iter(messages.values())).values())
    rte.flatten_results(result, name).to_csv(OUT / f"{name}_metrics.csv", index=False)
    raw = boot_stats.save_raw_fields(result, name, OUT / f"{name}_raw.json")
    return {k: boot_stats.combine(raw, VIDEOS)[k] for k in METRICS}


def main():
    metadata = {"package_version": "0.2.5", "package_source": "https://github.com/johnhw/pfilter",
                "package_module_sha256": hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest(),
                "seeds": [0, 1, 2], "clips": VIDEOS, "configuration": str(CFG),
                "cutoff": .25, "classes": [2, 3, 5, 7], "particles": 500,
                "started": datetime.now(timezone.utc).isoformat()}
    (OUT / "inputs.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for video in VIDEOS[:2]:
        base, _, _ = replay(video, 0, False, 128)
        sample, _, _ = replay(video, 0, True, 128)
        repeated, _, _ = replay(video, 0, True, 128)
        assert len(sample) and not sample.equals(base) and sample.equals(repeated)
        print("SMOKE", video, "deterministic external SIR", flush=True)
    combined = {}
    perf = []
    for seed in range(3):
        original = ROOT / f"results/trackeval/{SPLIT}/p2-kc48_r72_pf_dev-A-motion200-s{seed}/raw_fields.json"
        base = boot_stats.combine(boot_stats.load_raw_fields(original), VIDEOS)
        name = f"pfilter-s{seed}"
        folder = OUT / "trackers" / SPLIT / name / "data"
        folder.mkdir(parents=True, exist_ok=True)
        for video in VIDEOS:
            mot, frames, secs = replay(video, seed, True)
            mot.to_csv(folder / f"{video}.txt", index=False, header=False, float_format="%.10f")
            perf.append(dict(seed=seed, video=video, frames=frames, seconds=secs, rows=len(mot)))
            print("FULL", seed, video, frames, round(secs, 2), flush=True)
        combined[str(seed)] = {"baseline": {k: base[k] for k in METRICS}, "external": score(name)}
        (OUT / "metrics.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
        pd.DataFrame(perf).to_csv(OUT / "runtime.csv", index=False)
    print("PF_EXTERNAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
