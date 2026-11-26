"""Three independent seed processes for the single registered robust candidate."""

from concurrent.futures import ProcessPoolExecutor
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")


def load():
    import run_pf as setup
    import robust_filter
    from rbpf import installed as ordinary
    out = HERE / "robust"
    setup.OUT = out
    setup.runner.OUT = out
    setup.runner.installed = robust_filter.installed
    setup.runner.save_replay = save_replay
    return setup


def replay_seed(seed):
    setup = load()
    for video in setup.runner.DEV:
        result = setup.runner.replay(video, seed, True)
        save_replay("dev", "candidate", seed, video, result)
    return seed


def save_replay(scope, case, seed, video, result):
    setup = load()
    mot, classes, timing, events = result
    path = HERE / "robust" / scope / "tracking/DETRAC-all" / f"{case}-s{seed}"
    for folder in ["data", "data_cls", "timing", "events"]:
        (path / folder).mkdir(parents=True, exist_ok=True)
    assert not (path / "data" / f"{video}.txt").exists()
    mot.to_csv(path / "data" / f"{video}.txt", index=False, header=False, float_format="%.10f")
    classes.to_csv(path / "data_cls" / f"{video}.csv", index=False)
    timing.to_csv(path / "timing" / f"{video}.csv", index=False)
    setup.write(path / "events" / f"{video}.json", events)
    setup.write(path / "run_meta.json", {
        "candidate": "robust RBPF nu4", "cfg": setup.runner.rt.cfg_to_dict(setup.runner.config()),
        "seed": seed, "robust_source_sha256": setup.sha(HERE / "robust_filter.py"),
        "conditional_source_sha256": setup.sha(HERE / "rbpf.py"),
    })
    print("ROBUST", scope, case, seed, video, len(timing), flush=True)


def main(stage):
    setup = load()
    out = HERE / "robust"
    if stage == "preflight":
        source = json.loads((HERE / "inputs.json").read_text())
        source["deadline_epoch"] = time.time() + 5400
        source["candidate"] = "Student-t robust conditional PF, nu=4"
        paths = [HERE / "robust_filter.py", Path(__file__), HERE / "test_robust.py",
                 out / "00_protocol.md"]
        source["sha256"].update({str(p): setup.sha(p) for p in paths})
        setup.write(out / "inputs.json", source)
        print("ROBUST PREFLIGHT", flush=True)
    elif stage == "dev":
        assert (out / "smoke_checks.json").is_file()
        with ProcessPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(replay_seed, range(3)))
        assert results == [0, 1, 2]
    elif stage == "eval":
        setup.runner.evaluate()
    elif stage == "integrity":
        setup.runner.integrity()
    elif stage == "smoke":
        setup.runner.smoke()
    else:
        raise ValueError(stage)


if __name__ == "__main__":
    main(sys.argv[1])
