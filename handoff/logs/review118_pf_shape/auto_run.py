"""Use the R118 harness for the automatic velocity initialization ablation."""

from pathlib import Path
import json
import sys
import run as job
from automatic import installed

TOP = Path(__file__).resolve().parent
job.HERE = TOP / "automatic"
job.installed = installed
job.parent = lambda seed: TOP.parent / f"review112_svmot_pf/dev/evaluation/candidate-s{seed}/raw_fields.json"


def preflight():
    assert not (job.HERE / "inputs.json").exists()
    frozen = json.loads((TOP / "inputs.json").read_text())
    paths = [TOP / "automatic.py", Path(__file__), TOP / "01_initialization.md"]
    frozen["sha256"].update({str(path): job.base.sha(path) for path in paths})
    job.write(job.HERE / "inputs.json", frozen)
    rows = []
    for seed in range(3):
        for video in job.base.SMOKE:
            a, _ = job.replay(video, seed, True, 128)
            b, _ = job.replay(video, seed, True, 128)
            assert job.np.array_equal(a.to_numpy(float), b.to_numpy(float))
            rows.append({"seed": seed, "video": video, "repeat": True})
    job.write(job.HERE / "smoke.json", rows)
    print("AUTOMATIC SMOKE", len(rows), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "preflight":
        preflight()
    elif sys.argv[1] == "dev":
        job.dev(int(sys.argv[2]))
    else:
        job.summary()
