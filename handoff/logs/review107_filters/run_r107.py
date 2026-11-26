"""CPU replay/evaluation confined to this directory; no CLI tracking writer is called."""
import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(OUT / "ultralytics_settings"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import boot_stats
import det_cache
import run_tracking as rt
import run_trackeval as rte
from pf_postgap import installed

SPLIT = "DETRAC-all"
CACHE = "yolov8n-coco_conf0.05_all"
CFG = ROOT / "configs/phase2/trackers/r72/motion200.yaml"
DEV = json.loads((ROOT / "results/phase2/manifests/detrac_dev.json").read_text())["videos"]
SMOKE = ["MVI_20052", "MVI_20062"]
BASE = ROOT / "results/trackeval/DETRAC-all"
METRICS = ["HOTA", "IDF1", "IDSW", "FN", "FP", "MOTA"]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_baseline(seed):
    return BASE / f"p2-kc48_r72_pf_dev-A-motion200-s{seed}"


def reference(seed):
    return ROOT / f"data/processed/trackers/DETRAC-all/r95-ua-smoke-base-s{seed}"


def config():
    return rt.load_tracker_cfg(CFG, {"track_buffer": 30})


def preflight():
    assert not (OUT / "inputs.json").exists(), "Do not replace frozen preflight"
    paths = [CFG, Path(__file__), OUT / "pf_postgap.py", OUT / "00_protocol.md",
             ROOT / "results/phase2/manifests/detrac_dev.json",
             ROOT / "data/interim/det_cache" / CACHE / "meta.json",
             ROOT / "handoff/logs/review102_ridesafe_pf_nano/pf_association_replay.json",
             ROOT / "handoff/logs/review96_pf_nano/10_ket_qua.md"]
    paths += [ROOT / "src" / name for name in ["pf_ctrv.py", "tracker_ctrv.py", "ekf_ctrv.py", "ukf_ctrv.py",
              "run_tracking.py", "det_cache.py", "run_trackeval.py", "ul_backend.py", "boot_stats.py",
              "lost_association.py", "lost_recovery.py", "assoc_log.py", "time_units.py"]]
    totals = []
    for seed in (0, 1, 2):
        meta = json.loads((reference(seed) / "run_meta.json").read_text())
        assert rt.cfg_to_dict(config()) == meta["tracker_cfg"]
        assert (meta["conf_min"], meta["classes"], meta["max_det"], meta["fps"],
                meta["elapsed_mode"], meta["seed"]) == (.25, [2, 3, 5, 7], 300, 25., "skip", seed)
        current_meta = det_cache.load_meta(ROOT / "data/interim/det_cache" / CACHE)
        assert det_cache.meta_gen_spec(current_meta) == det_cache.meta_gen_spec(meta["cache_meta"])
        paths += [reference(seed) / "run_meta.json"]
        paths += [reference(seed) / "data" / f"{v}.txt" for v in SMOKE]
        path = source_baseline(seed)
        paths += [path / "raw_fields.json", path / "combined_metrics.csv", path / "per_video_metrics.csv"]
        raw = boot_stats.load_raw_fields(path / "raw_fields.json")
        assert set(raw) == set(DEV)
        combined = boot_stats.combine(raw, DEV)
        recorded = pd.read_csv(path / "combined_metrics.csv").iloc[0]
        assert all(abs(combined[k] - recorded[k]) < 1e-9 for k in METRICS)
        totals.append(combined)
    frames = {}
    for video in DEV:
        cache = det_cache.open_cache(CACHE, video, .25, [2, 3, 5, 7], 300)
        count = sum(1 for _ in cache.iter_frames())
        paths += [ROOT / "data/interim/det_cache" / CACHE / f"{video}{suffix}" for suffix in (".json", ".npz")]
        paths += [ROOT / "data/processed" / SPLIT / video / "gt/gt.txt",
                  ROOT / "data/processed" / SPLIT / video / "seqinfo.ini"]
        frames[video] = count
    paths += [ROOT / ".git" / name for name in ("HEAD", "index") if (ROOT / ".git" / name).is_file()]
    import pf_ctrv, tracker_ctrv, ul_backend, trackeval
    from ul_backend import BYTETracker, STrack
    import inspect
    modules = {"BYTETracker": inspect.getfile(BYTETracker), "STrack": inspect.getfile(STrack),
               "trackeval": inspect.getfile(trackeval)}
    paths += [Path(p) for p in modules.values()]
    paths = sorted(set(p.resolve() for p in paths))
    write(OUT / "inputs.json", {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
          "backend": ul_backend.backend_name(), "python": sys.executable,
          "numpy": np.__version__, "loaded_modules": modules,
          "cfg": rt.cfg_to_dict(config()), "frames": frames, "baseline_seed_combined": totals,
          "sha256": {str(p): sha(p) for p in paths},
          "deadline_epoch": time.time() + 5400})
    print("PREFLIGHT", sum(frames.values()), "frames; baseline IDSW", [r["IDSW"] for r in totals], flush=True)


class Prefix:
    def __init__(self, cache, length):
        self.cache, self.length, self.orig_shape = cache, length, cache.orig_shape

    def iter_frames(self):
        for index, row in enumerate(self.cache.iter_frames()):
            if index >= self.length:
                break
            yield row


def replay(video, seed, enabled, length=None):
    cache = det_cache.open_cache(CACHE, video, .25, [2, 3, 5, 7], 300)
    if length:
        cache = Prefix(cache, length)
    frozen = json.loads((OUT / "inputs.json").read_text(encoding="utf-8"))
    expected = length or frozen["frames"][video]
    seen = []

    def frame_check(frame, output):
        if time.time() > frozen["deadline_epoch"]:
            raise TimeoutError("R107 deadline reached; preserve partial output, no next job")
        seen.append(frame)

    with installed(enabled) as events:
        mot, classes, timing = rt.track_video(cache, config(), dt_nominal=.04, elapsed_mode="skip",
                                             seed=seed, video=video, on_frame=frame_check)
    assert len(timing) == expected == len(seen)
    assert np.isfinite(mot.to_numpy(float)).all()
    assert not mot.duplicated(["frame", "id"]).any()
    assert (mot[["bb_width", "bb_height"]].to_numpy() > 0).all()
    assert set(mot.frame).issubset(seen)
    return mot, classes, timing, events


def save_replay(scope, case, seed, video, result):
    mot, classes, timing, events = result
    path = OUT / scope / "tracking" / SPLIT / f"{case}-s{seed}"
    for folder in ("data", "data_cls", "timing", "events"):
        (path / folder).mkdir(parents=True, exist_ok=True)
    assert not (path / "data" / f"{video}.txt").exists(), "No silent overwrite"
    mot.to_csv(path / "data" / f"{video}.txt", index=False, header=False, float_format="%.10f")
    classes.to_csv(path / "data_cls" / f"{video}.csv", index=False)
    timing.to_csv(path / "timing" / f"{video}.csv", index=False)
    write(path / "events" / f"{video}.json", events)
    write(path / "run_meta.json", {"cfg": rt.cfg_to_dict(config()), "cache": CACHE,
          "cutoff": .25, "classes": [2, 3, 5, 7], "max_det": 300, "fps": 25,
          "elapsed": "skip", "seed": seed, "plugin_enabled": case == "candidate",
          "plugin_sha256": sha(OUT / "pf_postgap.py"), "runner_sha256": sha(Path(__file__)),
          "protocol_sha256": sha(OUT / "00_protocol.md"), "inputs_sha256": sha(OUT / "inputs.json")})
    print(scope, case, seed, video, "frames", len(timing), "events", len(events), flush=True)


def smoke():
    rows = []
    for seed in (0, 1, 2):
        for video in SMOKE:
            base = replay(video, seed, False, 128)
            old = np.loadtxt(reference(seed) / "data" / f"{video}.txt", delimiter=",")
            old = old[old[:, 0] <= 128]
            assert np.array_equal(base[0].to_numpy(float), old), (seed, video, "disabled parity")
            candidate = replay(video, seed, True, 128)
            if seed == 0:
                repeat = replay(video, seed, True, 128)
                assert np.array_equal(candidate[0].to_numpy(float), repeat[0].to_numpy(float))
                assert candidate[3] == repeat[3]
            save_replay("smoke", "base", seed, video, base)
            save_replay("smoke", "candidate", seed, video, candidate)
            rows.append({"seed": seed, "video": video, "exact_disabled_mot": True,
                         "repeat_candidate": seed == 0, "events": len(candidate[3])})
    write(OUT / "smoke_checks.json", rows)


def dev():
    assert (OUT / "smoke_checks.json").is_file()
    for seed in (0, 1, 2):
        for video in DEV:
            save_replay("dev", "candidate", seed, video, replay(video, seed, True))


def evaluate():
    rows, per_video, raw_paths = [], [], []
    for seed in (0, 1, 2):
        tracker = f"candidate-s{seed}"
        path = OUT / "dev/evaluation" / tracker
        assert not path.exists(), "No silent rescore"
        path.mkdir(parents=True)
        ec, dc, mc = rte.build_configs(SPLIT, tracker, DEV)
        ec.update({"LOG_ON_ERROR": str(OUT / "trackeval_error.log"), "PRINT_RESULTS": False})
        dc.update({"TRACKERS_FOLDER": str(OUT / "dev/tracking"), "OUTPUT_FOLDER": str(path),
                   "OUTPUT_SUB_FOLDER": "trackeval"})
        import trackeval
        dataset = trackeval.datasets.MotChallenge2DBox(dc)
        evaluator = trackeval.Evaluator(ec)
        metrics = [trackeval.metrics.HOTA(mc), trackeval.metrics.CLEAR(mc), trackeval.metrics.Identity(mc)]
        result, messages = evaluator.evaluate([dataset], metrics)
        assert all(m == "Success" for m in next(iter(messages.values())).values()), messages
        df = rte.flatten_results(result, tracker)
        df.to_csv(path / "metrics.csv", index=False)
        raw = boot_stats.save_raw_fields(result, tracker, path / "raw_fields.json")
        combined = boot_stats.combine(raw, DEV)
        recorded = df[df.video == "COMBINED_SEQ"].iloc[0]
        assert all(abs(combined[k] - recorded[k]) < 1e-9 for k in METRICS)
        rows.append(combined)
        raw_paths.append(str(path / "raw_fields.json"))
        per_video.append(raw)
        print("EVAL", seed, {k: combined[k] for k in METRICS}, flush=True)
    frozen = json.loads((OUT / "inputs.json").read_text(encoding="utf-8"))
    base_rows = frozen["baseline_seed_combined"]
    base = {k: float(np.mean([r[k] for r in base_rows])) for k in METRICS}
    cand = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
    diff = {}
    for video in DEV:
        b = [boot_stats.load_raw_fields(source_baseline(s) / "raw_fields.json")[video]["IDSW"] for s in range(3)]
        diff[video] = float(np.mean([r[video]["IDSW"] for r in per_video]) - np.mean(b))
    limits = {"IDSW_max": .90 * base["IDSW"], "HOTA_min": base["HOTA"] - .005,
              "FN_max": 1.005 * base["FN"]}
    conditions = {"IDSW": cand["IDSW"] <= limits["IDSW_max"],
                  "HOTA": cand["HOTA"] >= limits["HOTA_min"], "FN": cand["FN"] <= limits["FN_max"],
                  "all_seeds": all(c["IDSW"] <= b["IDSW"] for c, b in zip(rows, base_rows)),
                  "videos": sum(v < 0 for v in diff.values()) > sum(v > 0 for v in diff.values())}
    values = np.array(list(diff.values()))
    rng = np.random.default_rng(107)
    boots = values[rng.integers(0, len(values), (5000, len(values)))].sum(axis=1)
    write(OUT / "dev_gate.json", {"baseline": base, "candidate": cand, "candidate_seeds": rows,
          "raw_paths": raw_paths, "limits": limits, "conditions": conditions,
          "gate_pass": all(conditions.values()), "per_video_IDSW_delta": diff,
          "bootstrap_video_IDSW_delta95": np.quantile(boots, [.025, .975]).tolist(),
          "limits_inference": "Exploratory on repeatedly viewed dev; seed mean of per-seed COMBINED"})
    print("GATE", conditions, flush=True)


def integrity():
    frozen = json.loads((OUT / "inputs.json").read_text(encoding="utf-8"))
    bad = [p for p, h in frozen["sha256"].items() if sha(Path(p)) != h]
    assert not bad, bad
    records = []
    for seed in range(3):
        path = OUT / "dev/tracking" / SPLIT / f"candidate-s{seed}"
        for video in DEV:
            mot = pd.read_csv(path / "data" / f"{video}.txt", header=None)
            timing = pd.read_csv(path / "timing" / f"{video}.csv")
            assert len(timing) == frozen["frames"][video]
            assert np.isfinite(mot.to_numpy()).all() and not mot.duplicated([0, 1]).any()
            assert (mot[[4, 5]].to_numpy() > 0).all()
            events = json.loads((path / "events" / f"{video}.json").read_text())
            records.append({"seed": seed, "video": video, "frames": len(timing), "mot_rows": len(mot),
                            "events": len(events), "mot_sha256": sha(path / "data" / f"{video}.txt"),
                            "update_mean_ms": float(timing.t_update_ms.mean())})
    write(OUT / "integrity_verified.json", {"frozen_inputs_unchanged": True,
          "hashes_checked": len(frozen["sha256"]), "units": records, "total_frames": sum(r["frames"] for r in records)})
    print("INTEGRITY", len(records), "units;", len(frozen["sha256"]), "hashes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "smoke", "dev", "eval", "integrity"))
    args = parser.parse_args()
    {"preflight": preflight, "smoke": smoke, "dev": dev, "eval": evaluate, "integrity": integrity}[args.stage]()
