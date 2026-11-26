"""Recompute live speed and compare MOT with the equivalent saved PC replay."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, median
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
R110 = HERE.parent / "review110_nano_filters/replay_full"
R113 = HERE.parent / "review113_vehicle_scenarios/nano80"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def require_close(observed, expected):
    assert math.isclose(observed, expected, rel_tol=0, abs_tol=1e-8), (observed, expected)


def verify(root, expected_clips):
    manifest, items = read(root / "inputs.json"), read(root / "runs.json")
    picked = {(r["dataset"], r["name"]) for r in manifest["sample"]}
    methods = {"kf", "ekf", "ukf", "rbpf"}
    assert len(picked) == expected_clips and len(items) == 4 * len(picked)
    assert {(r["dataset"], r["name"], r["method"]) for r in items} == {
        (dataset, video, method) for dataset, video in picked for method in methods}
    historic = read(R110 / "scores.json")["capture"]
    expected_cache = {(r["dataset"], r["name"]): r["cache_sha256"] for r in historic}
    grouped, parity, peak = defaultdict(list), defaultdict(list), defaultdict(list)
    for item in items:
        ds, name, method = item["dataset"], item["name"], item["method"]
        base = root / "runs" / ds / name / method
        data = rows(base / "pipeline" / f"{name}.csv")
        assert len(data) == 150 and [int(r["frame"]) for r in data] == list(range(1, 151))
        assert not any(r["unreadable"] != "False" or r["infer_skipped"] != "False" for r in data)
        finish = [float(r["t_xong_cha"]) for r in data]
        assert all(b > a for a, b in zip(finish, finish[1:]))
        after = data[25:]
        for frame in data:
            require_close(float(frame["tre_ms"]), 1000 * (float(frame["t_xong_cha"]) - float(frame["t_bat_dau_con"])))
        latencies = sorted(float(r["tre_ms"]) for r in after)
        fps = 125 / (finish[-1] - finish[24])
        p90 = latencies[round(.9 * (len(after)-1))]
        assert item["det_cache_sha256"] == expected_cache[ds, name] == sha(base / "det_capture" / name / f"{name}.npz")
        meta = read(base / "run_meta.json")
        steady = meta["thoi_gian"]["tom_tat_sau_warmup"]
        require_close(fps, item["fps"])
        require_close(fps, steady["fps_on_dinh"])
        require_close(p90, item["latency_p90_ms"])
        require_close(p90, steady["tre_ms"]["p90"])
        timing = rows(base / "timing" / f"{name}.csv")[25:]
        update = [float(r["t_update_ms"]) for r in timing]
        assert len(update) == 125
        require_close(median(update), item["update_median_ms"])
        host = item["hardware_after"]
        assert host["sd_errors"] == "0" and host["boot"] == manifest["hardware_start"]["boot"]
        split = "BDDCOCOVID-val" if ds == "bdd" else "DETRAC-test"
        previous = (R113 / "trackers" / split / "rbpf_s0" if method == "rbpf"
                    else R110 / "trackers" / split / method) / "data" / f"{name}.txt"
        mot = base / "data" / f"{name}.txt"
        assert sha(mot) == item["mot_sha256"] and sha(base / "data_cls" / f"{name}.csv") == item["classes_sha256"]
        a, b = [np.loadtxt(p, delimiter=",", ndmin=2) if p.stat().st_size else np.empty((0, 7)) for p in (mot, previous)]
        identical = a.shape == b.shape and np.array_equal(a[:, [0, 1, 6]], b[:, [0, 1, 6]])
        identical = identical and bool(np.allclose(a, b, rtol=0, atol=1e-3))
        parity[ds, method].append(identical)
        stats = (base / "tegrastats.log").read_text(encoding="utf-8")
        temps = re.findall(r"\b(?:CPU|GPU|thermal)@([\d.]+)C\b", stats)
        if temps:
            peak[ds, method].append(max(map(float, temps)))
        grouped[ds, method].append({**item, "gpu_exec_median_ms": median(float(r["exec_ms"]) for r in after),
                                    "tracked_per_frame_median": median(int(r["n_tracks"]) for r in timing),
                                    "startup_s": meta["thoi_gian"]["khoi_tao_s"]})
    summary = {ds: {} for ds in ["bdd", "ua"]}
    for (ds, method), group in grouped.items():
        fps = [r["fps"] for r in group]
        worst = max(group, key=lambda r: r["latency_p90_ms"])
        summary[ds][method] = {"clips": len(group), "fps_mean": mean(fps), "fps_median": median(fps),
                               "fps_min": min(fps), "fps_max": max(fps), "clips_at_15": sum(v >= 15 for v in fps),
                               "median_clip_latency_p90_ms": median(r["latency_p90_ms"] for r in group),
                               "max_clip_latency_p90_ms": worst["latency_p90_ms"], "max_latency_clip": worst["name"],
                               "median_clip_update_ms": median(r["update_median_ms"] for r in group),
                               "median_clip_gpu_exec_ms": median(r["gpu_exec_median_ms"] for r in group),
                               "parity_with_pc_mot": sum(parity[ds, method]),
                               "max_cpu_gpu_thermal_c": max(peak[ds, method], default=None)}
    sample_rows = [r for per_group in grouped.values() for r in per_group]
    result = {"clips": len(picked), "runs": len(items), "frames": sum(row["n_frames"] for row in items),
              "summary": summary, "per_run": sample_rows, "hardware_start": manifest["hardware_start"],
              "hardware_end": read(root / "hardware_after.json")}
    (root / "verified_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    verify(Path(sys.argv[1]), int(sys.argv[2]))
