"""Dev evaluation using the established R107 cache and metric functions."""

from pathlib import Path
import argparse
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / "ultralytics_settings"))
sys.path[:0] = [str(ROOT / "src"), str(HERE.parent / "review107_filters")]
import numpy as np
import run_r107 as previous
import trackeval
from apf import installed


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    previous.write(path, data)


def replay(video, seed, mode, length=None):
    cache = previous.det_cache.open_cache(previous.CACHE, video, .25, [2, 3, 5, 7], 300)
    if length is not None:
        cache = previous.Prefix(cache, length)
    with installed(mode) as events:
        mot, classes, timing = previous.rt.track_video(cache, previous.config(), .04, elapsed_mode="skip",
                                                      seed=seed, video=video)
    assert len(timing) == (length or len(cache)) and np.isfinite(mot.to_numpy(float)).all()
    assert not mot.duplicated(["frame", "id"]).any() and (mot[["bb_width", "bb_height"]] > 0).all().all()
    return mot, classes, timing, events


def preflight():
    assert not (HERE / "inputs.json").exists()
    paths = [Path(__file__), HERE / "apf.py", HERE / "checks.py", HERE / "00_protocol.md", previous.CFG]
    paths += [ROOT / "src" / p for p in ("pf_ctrv.py", "tracker_ctrv.py", "ekf_ctrv.py", "run_tracking.py",
                                         "ul_backend.py", "det_cache.py", "time_units.py")]
    counts = {}
    for video in previous.DEV:
        cache = previous.det_cache.open_cache(previous.CACHE, video, .25, [2, 3, 5, 7], 300)
        counts[video] = len(cache)
        paths += [ROOT / "data/interim/det_cache" / previous.CACHE / f"{video}{suffix}"
                  for suffix in (".json", ".npz")]
        paths += [ROOT / "data/processed/DETRAC-all" / video / "gt/gt.txt"]
    base = []
    for seed in range(3):
        path = previous.source_baseline(seed) / "raw_fields.json"
        raw = previous.boot_stats.load_raw_fields(path)
        assert set(raw) == set(previous.DEV)
        base.append(previous.boot_stats.combine(raw, previous.DEV))
        paths.append(path)
    write(HERE / "inputs.json", {"frames": counts, "config": previous.rt.cfg_to_dict(previous.config()),
                                "baseline_seeds": base, "sha256": {str(p): previous.sha(p) for p in paths}})
    print("FROZEN", sum(counts.values()), flush=True)


def smoke():
    result = []
    for seed in range(3):
        for video in previous.SMOKE:
            base = replay(video, seed, "base", 128)
            old = np.loadtxt(previous.reference(seed) / "data" / f"{video}.txt", delimiter=",")
            assert np.array_equal(base[0].to_numpy(float), old[old[:, 0] <= 128])
            for mode in ("apf", "apf_k2"):
                candidate = replay(video, seed, mode, 128)
                if seed == 0:
                    duplicate = replay(video, seed, mode, 128)
                    assert np.array_equal(candidate[0].to_numpy(float), duplicate[0].to_numpy(float))
                result.append({"video": video, "seed": seed, "mode": mode, "events": candidate[3]})
    write(HERE / "smoke.json", result)
    print("SMOKE", len(result), flush=True)


def evaluate_seed(mode, seed):
    tracker = f"{mode}-s{seed}"
    ec, dc, mc = previous.rte.build_configs(previous.SPLIT, tracker, previous.DEV)
    output = HERE / "dev/evaluation" / tracker
    output.mkdir(parents=True, exist_ok=True)
    ec.update(PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
              LOG_ON_ERROR=str(output / "error.log"))
    dc.update(TRACKERS_FOLDER=str(HERE / "dev/tracking"), OUTPUT_FOLDER=str(output))
    dataset = trackeval.datasets.MotChallenge2DBox(dc)
    metrics = [trackeval.metrics.HOTA(mc), trackeval.metrics.CLEAR(mc), trackeval.metrics.Identity(mc)]
    result, messages = trackeval.Evaluator(ec).evaluate([dataset], metrics)
    assert all(value == "Success" for value in next(iter(messages.values())).values())
    raw = previous.boot_stats.save_raw_fields(result, tracker, output / "raw_fields.json")
    combined = previous.boot_stats.combine(raw, previous.DEV)
    write(output / "combined.json", combined)
    print("SCORE", tracker, combined, flush=True)


def dev(mode, seed):
    assert (HERE / "smoke.json").is_file()
    frozen = json.loads((HERE / "inputs.json").read_text())
    for video in previous.DEV:
        folder = HERE / "dev/tracking" / previous.SPLIT / f"{mode}-s{seed}"
        mot_path = folder / "data" / f"{video}.txt"
        meta = folder / "meta" / f"{video}.json"
        if meta.is_file():
            assert mot_path.is_file()
            continue
        mot, classes, timing, events = replay(video, seed, mode)
        assert len(timing) == frozen["frames"][video]
        mot_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path = folder / "timing" / f"{video}.csv"
        timing_path.parent.mkdir(exist_ok=True)
        mot.to_csv(mot_path, index=False, header=False, float_format="%.10f")
        timing.to_csv(timing_path, index=False)
        write(meta, {"mot_sha256": previous.sha(mot_path), "frames": len(timing), "events": events,
                     "update_mean_ms": float(timing.t_update_ms.mean())})
        print("DONE", mode, seed, video, len(timing), flush=True)
    evaluate_seed(mode, seed)


def summary(modes=("apf", "apf_k2"), suffix=""):
    frozen = json.loads((HERE / "inputs.json").read_text())
    if suffix:
        amended = json.loads((HERE / "ess_inputs.json").read_text())
        assert all(previous.sha(Path(path)) == digest for path, digest in amended["sha256"].items())
        assert all(previous.sha(Path(path)) == digest for path, digest in frozen["sha256"].items()
                   if Path(path).name not in ("run.py", "apf.py"))
    else:
        assert all(previous.sha(Path(path)) == digest for path, digest in frozen["sha256"].items())
    baseline = {k: float(np.mean([row[k] for row in frozen["baseline_seeds"]])) for k in previous.METRICS}
    candidates = {}
    for mode in modes:
        records = [json.loads((HERE / f"dev/evaluation/{mode}-s{s}/combined.json").read_text()) for s in range(3)]
        mean = {k: float(np.mean([row[k] for row in records])) for k in previous.METRICS}
        deltas = {}
        for video in previous.DEV:
            changed, reference = [], []
            for seed in range(3):
                raw = previous.boot_stats.load_raw_fields(HERE / f"dev/evaluation/{mode}-s{seed}/raw_fields.json")
                base = previous.boot_stats.load_raw_fields(previous.source_baseline(seed) / "raw_fields.json")
                changed.append(raw[video]["IDSW"])
                reference.append(base[video]["IDSW"])
                folder = HERE / "dev/tracking" / previous.SPLIT / f"{mode}-s{seed}"
                meta = json.loads((folder / "meta" / f"{video}.json").read_text())
                assert previous.sha(folder / "data" / f"{video}.txt") == meta["mot_sha256"]
            deltas[video] = float(np.mean(changed) - np.mean(reference))
        rng = np.random.default_rng(117)
        draws = rng.choice(list(deltas.values()), (10000, len(deltas))).sum(axis=1)
        gate = {"IDSW": mean["IDSW"] <= .9 * baseline["IDSW"], "HOTA": mean["HOTA"] >= baseline["HOTA"] - .005,
                "FN": mean["FN"] <= 1.005 * baseline["FN"],
                "seeds": all(a["IDSW"] <= b["IDSW"] for a, b in zip(records, frozen["baseline_seeds"]))}
        candidates[mode] = {"mean": mean, "seeds": records, "gate": gate, "pass": all(gate.values()),
                            "IDSW_delta_by_video": deltas, "bootstrap95": np.quantile(draws, [.025, .975]).tolist()}
    write(HERE / f"dev_summary{suffix}.json", {"baseline": baseline, "candidates": candidates})
    print(baseline, candidates, flush=True)


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "dev":
        dev(sys.argv[2], int(sys.argv[3]))
    elif action == "summary-ess":
        summary(("apf_ess_k2",), "_ess")
    else:
        {"preflight": preflight, "smoke": smoke, "summary": summary}[action]()
