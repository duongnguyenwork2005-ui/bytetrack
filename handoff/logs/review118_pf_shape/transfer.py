"""Paired SVMOT transfer for the best dev-qualified Cartesian representation."""

from pathlib import Path
import json
import sys
import run as job

TOP = job.HERE
VIDEOS = [f"seq_{i}_{'morning' if i < 26 else 'evening'}" for i in range(22, 31)]


def selected():
    records = {"two_observations": TOP / "dev_summary.json",
               "automatic": TOP / "automatic/dev_summary.json"}
    valid = {name: json.loads(path.read_text()) for name, path in records.items()}
    valid = {name: record for name, record in valid.items() if record["pass"]}
    assert valid, "No candidate passed the predeclared dev conditions"
    return min(valid, key=lambda name: valid[name]["candidate"]["IDSW"])


def run_seed(seed):
    assert (TOP / "transfer_inputs.json").is_file()
    name = selected()
    if name == "automatic":
        from automatic import installed
        job.installed = installed
    for video in VIDEOS:
        for mode, enabled in (("parent", False), ("candidate", True)):
            folder = TOP / f"transfer/tracking/SVMOTSYS-test/{mode}-s{seed}"
            output = folder / "data" / f"{video}.txt"
            assert not output.exists()
            mot, timing = job.replay(video, seed, enabled, cache_id="r64_svmot_yolov8n_coco_c0.05",
                                     fps=30, classes=[1, 2, 3, 5, 7], buffer=36)
            output.parent.mkdir(parents=True, exist_ok=True)
            mot.to_csv(output, header=False, index=False, float_format="%.10f")
            timing_path = folder / "timing"
            timing_path.mkdir(exist_ok=True)
            timing.to_csv(timing_path / f"{video}.csv", index=False)
            job.write(folder / "meta" / f"{video}.json",
                      {"selected": name, "frames": len(timing), "mot_sha256": job.base.sha(output)})
            if not enabled:
                reference = TOP.parent / f"review112_svmot_pf/transfer_svmot/tracking/SVMOTSYS-test/candidate-s{seed}/data/{video}.txt"
                old = job.np.loadtxt(reference, delimiter=",")
                assert job.np.allclose(mot.to_numpy(float), old, atol=1e-9, rtol=0), (seed, video)
            print("TRANSFER", seed, video, mode, len(timing), flush=True)
    for mode in ("parent", "candidate"):
        job.score(seed, "SVMOTSYS-test", VIDEOS, mode=mode, stage="transfer")


def summary():
    inputs = json.loads((TOP / "transfer_inputs.json").read_text())
    assert all(job.base.sha(Path(path)) == digest for path, digest in inputs["sha256"].items())
    means, seeds = {}, {}
    for mode in ("parent", "candidate"):
        data = [json.loads((TOP / f"transfer/evaluation/{mode}-s{s}/combined.json").read_text()) for s in range(3)]
        seeds[mode] = data
        means[mode] = {k: float(job.np.mean([row[k] for row in data])) for k in job.base.METRICS}
    job.write(TOP / "transfer_summary.json", {"selected": selected(), "mean": means, "seeds": seeds,
                                             "videos": VIDEOS, "units": 54, "baseline_full_mot_parity": 27})
    print(means, flush=True)


def preflight():
    assert not (TOP / "transfer_inputs.json").exists()
    paths = [Path(__file__), TOP / "cartesian.py", TOP / "automatic.py", TOP / "run.py"]
    for video in VIDEOS:
        paths += [job.ROOT / "data/interim/det_cache/r64_svmot_yolov8n_coco_c0.05" / f"{video}{suffix}"
                  for suffix in (".npz", ".json")]
        paths += [job.ROOT / "data/processed/SVMOTSYS-test" / video / "gt/gt.txt"]
    job.write(TOP / "transfer_inputs.json", {"selected": selected(),
                                           "sha256": {str(p): job.base.sha(p) for p in paths}})


if __name__ == "__main__":
    if sys.argv[1] == "preflight":
        preflight()
    elif sys.argv[1] == "summary":
        summary()
    else:
        run_seed(int(sys.argv[1]))
