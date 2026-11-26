"""Evaluate the Cartesian coordinated-turn RBPF against the R112 parent."""

from pathlib import Path
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / "ultralytics_settings"))
sys.path[:0] = [str(ROOT / "src"), str(HERE.parent / "review107_filters")]
import numpy as np
import run_r107 as base
import trackeval
from cartesian import installed

CFG = ROOT / "configs/phase2/trackers/r112/rbpf500.yaml"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    base.write(path, value)


def parent(seed):
    return HERE.parent / f"review112_svmot_pf/dev/evaluation/candidate-s{seed}/raw_fields.json"


def replay(name, seed, candidate=True, length=None, cache_id=None, fps=25, classes=None, buffer=30):
    cache = base.det_cache.open_cache(cache_id or base.CACHE, name, .25, classes or [2, 3, 5, 7], 300)
    if length:
        cache = base.Prefix(cache, length)
    cfg = base.rt.load_tracker_cfg(CFG, {"track_buffer": buffer})
    with installed(candidate):
        mot, _, timing = base.rt.track_video(cache, cfg, 1/fps, elapsed_mode="skip", seed=seed, video=name)
    assert np.isfinite(mot.to_numpy(float)).all() and not mot.duplicated(["frame", "id"]).any()
    assert (mot[["bb_width", "bb_height"]] > 0).all().all()
    assert len(timing) == (length or len(cache))
    return mot, timing


def preflight():
    assert not (HERE / "inputs.json").exists()
    paths = [Path(__file__), HERE / "cartesian.py", HERE / "checks.py", HERE / "00_protocol.md", CFG,
             ROOT / "src/rbpf_ctrv.py", ROOT / "results/phase2/manifests/detrac_dev.json"]
    prior = json.loads((HERE.parent / "review117_apf/inputs.json").read_text())
    paths += [Path(p) for p in prior["sha256"] if "/src/" in str(p).replace("\\", "/")
              or "/data/" in str(p).replace("\\", "/")]
    paths += [parent(seed) for seed in range(3)]
    summaries = [base.boot_stats.combine(base.boot_stats.load_raw_fields(parent(seed)), base.DEV) for seed in range(3)]
    write(HERE / "inputs.json", {"frames": prior["frames"], "baseline_seeds": summaries,
                                "config": base.rt.cfg_to_dict(base.rt.load_tracker_cfg(CFG, {"track_buffer": 30})),
                                "sha256": {str(p): base.sha(p) for p in paths}})


def smoke():
    rows = []
    for seed in range(3):
        for video in base.SMOKE:
            observed, _ = replay(video, seed, False, 128)
            old_path = HERE.parent / f"review112_svmot_pf/dev/tracking/DETRAC-all/candidate-s{seed}/data/{video}.txt"
            old = np.loadtxt(old_path, delimiter=",")
            assert np.allclose(observed.to_numpy(float), old[old[:, 0] <= 128], atol=1e-9, rtol=0)
            a, _ = replay(video, seed, True, 128)
            b, _ = replay(video, seed, True, 128)
            assert np.array_equal(a.to_numpy(float), b.to_numpy(float))
            rows.append({"video": video, "seed": seed, "parent_parity": True, "repeat": True})
    write(HERE / "smoke.json", rows)
    print("SMOKE", len(rows), flush=True)


def score(seed, split, videos, mode="cartesian", stage="dev"):
    folder = HERE / stage
    tracker = f"{mode}-s{seed}"
    ec, dc, mc = base.rte.build_configs(split, tracker, None)
    ec.update(PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
              LOG_ON_ERROR=str(folder / "error.log"))
    dc.update(TRACKERS_FOLDER=str(folder / "tracking"), SEQ_INFO={name: None for name in videos},
              OUTPUT_FOLDER=str(folder / "evaluation"))
    dataset = trackeval.datasets.MotChallenge2DBox(dc)
    metrics = [trackeval.metrics.HOTA(mc), trackeval.metrics.CLEAR(mc), trackeval.metrics.Identity(mc)]
    result, message = trackeval.Evaluator(ec).evaluate([dataset], metrics)
    assert all(value == "Success" for value in next(iter(message.values())).values())
    target = folder / "evaluation" / tracker
    target.mkdir(parents=True, exist_ok=True)
    raw = base.boot_stats.save_raw_fields(result, tracker, target / "raw_fields.json")
    combined = base.boot_stats.combine(raw, videos)
    write(target / "combined.json", combined)
    print("SCORE", tracker, combined, flush=True)


def dev(seed):
    assert (HERE / "smoke.json").is_file()
    frozen = json.loads((HERE / "inputs.json").read_text())
    for video in base.DEV:
        folder = HERE / f"dev/tracking/DETRAC-all/cartesian-s{seed}"
        marker = folder / "meta" / f"{video}.json"
        if marker.is_file():
            continue
        mot, timing = replay(video, seed)
        assert len(timing) == frozen["frames"][video]
        output = folder / "data" / f"{video}.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        mot.to_csv(output, index=False, header=False, float_format="%.10f")
        timing_folder = folder / "timing"
        timing_folder.mkdir(exist_ok=True)
        timing.to_csv(timing_folder / f"{video}.csv", index=False)
        write(marker, {"mot_sha256": base.sha(output), "frames": len(timing)})
        print("DONE", seed, video, len(timing), flush=True)
    score(seed, "DETRAC-all", base.DEV)


def summary():
    frozen = json.loads((HERE / "inputs.json").read_text())
    assert all(base.sha(Path(p)) == digest for p, digest in frozen["sha256"].items())
    baseline = {k: float(np.mean([r[k] for r in frozen["baseline_seeds"]])) for k in base.METRICS}
    scores = [json.loads((HERE / f"dev/evaluation/cartesian-s{s}/combined.json").read_text()) for s in range(3)]
    candidate = {k: float(np.mean([r[k] for r in scores])) for k in base.METRICS}
    diff = {}
    for name in base.DEV:
        changes = []
        for seed in range(3):
            current = base.boot_stats.load_raw_fields(HERE / f"dev/evaluation/cartesian-s{seed}/raw_fields.json")
            old = base.boot_stats.load_raw_fields(parent(seed))
            changes.append(current[name]["IDSW"] - old[name]["IDSW"])
        diff[name] = float(np.mean(changes))
    rng = np.random.default_rng(118)
    bootstrap = rng.choice(list(diff.values()), (10000, len(diff))).sum(axis=1)
    conditions = {"IDSW": candidate["IDSW"] < baseline["IDSW"],
                  "HOTA": candidate["HOTA"] >= baseline["HOTA"] - .005,
                  "FN": candidate["FN"] <= 1.005 * baseline["FN"],
                  "seeds": sum(c["IDSW"] < b["IDSW"] for c, b in zip(scores, frozen["baseline_seeds"])) >= 2}
    write(HERE / "dev_summary.json", {"baseline": baseline, "candidate": candidate, "seeds": scores,
                                      "conditions": conditions, "pass": all(conditions.values()),
                                      "IDSW_delta_by_video": diff, "bootstrap95": np.quantile(bootstrap, [.025, .975]).tolist()})
    print(baseline, candidate, conditions, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "dev":
        dev(int(sys.argv[2]))
    else:
        {"preflight": preflight, "smoke": smoke, "summary": summary}[sys.argv[1]]()
