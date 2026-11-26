"""Transfer the dev-selected auxiliary PF to frozen FastTrack detections."""

from pathlib import Path
import json
import sys

from run import HERE, ROOT, previous, write
from apf import installed
import numpy as np

sys.path[:0] = [str(HERE.parent / "review115_new_datasets")]
import detect
import score

R116 = HERE.parent / "review116_fasttrack_full"
PRIOR_DATA = detect.DATA
NEW_DATA = Path("D:/LABOTARY/test dataset/FastTrack-R116")
NEW_WORK = NEW_DATA / "evaluation_r116"
SCORE_ROOT = HERE / "fasttrack"
WORK_TAG = "evaluation_r117"


def metadata(archive, name):
    import configparser
    parser = configparser.ConfigParser()
    parser.read_string(archive.read(f"{name}/seqinfo.ini").decode())
    sequence = parser["Sequence"]
    frames = sequence.getint("seqLength")
    names = [f"{name}/img1/{i:06d}.jpg" for i in range(1, frames + 1)]
    assert sorted(n for n in archive.namelist() if n.startswith(f"{name}/img1/") and n.endswith(".jpg")) == names
    return names, sequence.getint("frameRate"), (sequence.getint("imHeight"), sequence.getint("imWidth"))


def selection():
    gate = json.loads((HERE / "dev_summary.json").read_text())
    assert (HERE / "01_transfer_extension.md").is_file()
    allowed = {name: row for name, row in gate["candidates"].items()
               if row["mean"]["IDSW"] < gate["baseline"]["IDSW"]
               and all(row["gate"][k] for k in ("HOTA", "FN", "seeds"))}
    assert allowed, "No APF variant improved IDSW while preserving HOTA, FN and seed consistency"
    return min(allowed, key=lambda name: allowed[name]["mean"]["IDSW"])


def records():
    manifest = json.loads((R116 / "sources.json").read_text())
    return [Path(row["path"]).stem for row in manifest["archives"]]


def reference(name):
    old = name in detect.FASTTRACK
    folder = HERE.parent / "review115_new_datasets/scores" if old else R116 / "scores"
    root = PRIOR_DATA if old else NEW_DATA
    record = json.loads((folder / f"{name}.json").read_text())
    assert previous.sha(root / f"{name}_yolov8n.npz") == record["cache_sha256"]
    return root, record


def seed_run(seed):
    selected = selection()
    for name in records():
        root, base = reference(name)
        score.DATA = root
        score.WORK = NEW_DATA / WORK_TAG / f"seed{seed}"
        score.METHODS = [("pf", seed)]
        output = SCORE_ROOT / f"seed{seed}"
        output.mkdir(parents=True, exist_ok=True)
        assert not (output / f"{name}.json").exists()
        with installed(selected) as events:
            score.score_sequence(name, output, fps=base["fps"], fasttrack_metadata=metadata)
        assert events["auxiliary_updates"] > 0
        candidate = json.loads((output / f"{name}.json").read_text())
        for field in ("frames", "cache_sha256", "gt_sha256"):
            assert candidate[field] == base[field], (name, field)
        write(output / f"{name}_events.json", {"selected": selected, **events})
        print("TRANSFER", seed, name, flush=True)


def summarize():
    all_scores = {}
    selected = selection()
    for seed in range(3):
        variants = {}
        for name in records():
            root, base = reference(name)
            record = json.loads((SCORE_ROOT / f"seed{seed}/{name}.json").read_text())
            variant = record["variants"][f"pf-s{seed}"]
            mot = NEW_DATA / WORK_TAG / f"seed{seed}/trackers/pf-s{seed}/data/{name}.txt"
            assert previous.sha(mot) == variant["mot_sha256"]
            variants[name] = variant
        combined = {}
        for metric in score.METRICS:
            key = metric.get_name()
            raw = {name: {k: np.asarray(v) if isinstance(v, list) else v for k, v in row["raw"][key].items()}
                   for name, row in variants.items()}
            combined[key] = metric.combine_sequences(raw)
        all_scores[str(seed)] = score.concise(combined)
    mean = {key: float(np.mean([result[key] for result in all_scores.values()])) for key in all_scores["0"]}
    old = json.loads((R116 / "scores/scores.json").read_text())["all12"]["results"]
    baseline = {key: float(np.mean([old[f"pf-s{s}"]["metrics"][key] for s in range(3)]))
                for key in mean}
    write(HERE / ("transfer_ess_summary.json" if WORK_TAG.endswith("_ess") else "transfer_summary.json"),
          {"selected": selected, "candidate_mean": mean, "baseline_mean": baseline,
                                          "seeds": all_scores, "units": 36, "frames_per_seed": 19480})
    print(baseline, mean, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summarize()
    else:
        seed_run(int(sys.argv[1]))
