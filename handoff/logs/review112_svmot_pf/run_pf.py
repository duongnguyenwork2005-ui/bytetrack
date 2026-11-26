"""Reuse the R107 cache/evaluation entry points with an isolated RBPF plugin."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SETTINGS = OUT / "ultralytics_settings"
SETTINGS.mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(SETTINGS)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path[:0] = [str(ROOT / "src"), str(OUT.parent / "review107_filters")]
import numpy as np
import run_r107 as runner
from rbpf import installed

runner.OUT = OUT
runner.installed = installed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def preflight():
    assert not (OUT / "inputs.json").exists()
    old = json.loads((OUT.parent / "review107_filters/inputs.json").read_text(encoding="utf-8"))
    paths = [OUT / "00_protocol.md", OUT / "rbpf.py", Path(__file__),
             OUT / "test_rbpf.py", runner.CFG]
    paths += [Path(path) for path in old["sha256"] if Path(path).is_file()]
    for path in paths:
        assert path.is_file(), path
    for path, expected in old["sha256"].items():
        if Path(path).is_file():
            assert sha(path) == expected, ("Input changed since R107", path)
    data = {
        "backend": "PC CPU NumPy", "python": sys.executable,
        "frames": old["frames"], "baseline_seed_combined": old["baseline_seed_combined"],
        "sha256": {str(path.resolve()): sha(path) for path in sorted(set(paths))},
        "deadline_epoch": time.time() + 5400,
        "candidate": "conditional Gaussian 7D + 500 angular CTRV particles",
        "cfg": runner.rt.cfg_to_dict(runner.config()),
        "baseline_source": "R72 full metrics, R95 smoke MOT for parity",
    }
    write(OUT / "inputs.json", data)
    print("PREFLIGHT", sum(data["frames"].values()), "frames/seed", flush=True)


def save_replay(scope, case, seed, video, result):
    mot, classes, timing, events = result
    path = OUT / scope / "tracking" / runner.SPLIT / f"{case}-s{seed}"
    for folder in ("data", "data_cls", "timing", "events"):
        (path / folder).mkdir(parents=True, exist_ok=True)
    assert not (path / "data" / f"{video}.txt").exists()
    mot.to_csv(path / "data" / f"{video}.txt", index=False, header=False, float_format="%.10f")
    classes.to_csv(path / "data_cls" / f"{video}.csv", index=False)
    timing.to_csv(path / "timing" / f"{video}.csv", index=False)
    write(path / "events" / f"{video}.json", events)
    write(path / "run_meta.json", {
        "candidate": "RBPF" if case == "candidate" else "PF500 K1/K2 motion2",
        "cfg": runner.rt.cfg_to_dict(runner.config()), "seed": seed,
        "plugin_sha256": sha(OUT / "rbpf.py"), "inputs_sha256": sha(OUT / "inputs.json"),
        "cache": runner.CACHE, "cutoff": 0.25, "fps": 25, "max_det": 300,
    })
    print(scope, case, seed, video, "frames", len(timing), events, flush=True)


runner.save_replay = save_replay
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "smoke", "dev", "eval", "integrity"])
    args = parser.parse_args()
    {"preflight": preflight, "smoke": runner.smoke, "dev": runner.dev,
     "eval": runner.evaluate, "integrity": runner.integrity}[args.stage]()
