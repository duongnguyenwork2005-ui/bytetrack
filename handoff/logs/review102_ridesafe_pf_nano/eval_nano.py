"""Score the matched 150-frame Nano outputs with the existing project evaluators."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import trackeval

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import run_trackeval as rte  # noqa: E402

SPLITS = {"bdd": ("BDDCOCOVID-val", "mot_crowd"), "ua": ("DETRAC-test", "mot")}


def clipped_gt(source, first, last):
    rows = []
    with source.open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if not row:
                continue
            frame = int(row[0])
            if first <= frame <= last:
                rows.append([frame - first + 1, *row[1:]])
    assert rows, source
    return rows


def stage_inputs(logs, output):
    videos = json.loads((logs / "videos.json").read_text())
    stats = json.loads((logs / "pipeline_pairs.json").read_text())
    assert len(videos) == 8 and len(stats) == 16
    checks = {(r["dataset"], r["name"], r["version"]): r for r in stats}
    expected_paths = {f"{ds}/{name}/{version}.txt" for ds, name, version in checks}
    with tarfile.open(logs / "mot_outputs.tar.gz", "r:gz") as archive:
        assert set(archive.getnames()) == expected_paths
        for video in videos:
            ds, name = video["dataset"], video["name"]
            split, _ = SPLITS[ds]
            assert video["last"] - video["first"] + 1 == video["n_frames"] == 150
            source = ROOT / "data/processed" / split / name / "gt/gt.txt"
            gt_file = output / "gt" / split / name / "gt/gt.txt"
            gt_file.parent.mkdir(parents=True, exist_ok=True)
            with gt_file.open("w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerows(clipped_gt(source, video["first"], video["last"]))
            for version in "AB":
                payload = archive.extractfile(f"{ds}/{name}/{version}.txt").read()
                row = checks[(ds, name, version)]
                assert row["frames"] == 150 and hashlib.sha256(payload).hexdigest() == row["mot_sha256"]
                reader = csv.reader(io.StringIO(payload.decode("utf-8")))
                entries = [(int(x[0]), int(x[1])) for x in reader if x]
                assert len(entries) == row["mot_rows"] == len(set(entries))
                assert all(1 <= frame <= 150 for frame, _ in entries)
                tracker = output / "trackers" / split / version / "data" / f"{name}.txt"
                tracker.parent.mkdir(parents=True, exist_ok=True)
                tracker.write_bytes(payload)
    return videos


def scalars(raw):
    h, c, i = raw["HOTA"], raw["CLEAR"], raw["Identity"]
    return {"hota": float(h["HOTA"].mean()), "deta": float(h["DetA"].mean()), "assa": float(h["AssA"].mean()),
            "idf1": float(i["IDF1"]), "idsw": int(c["IDSW"]), "fn": int(c["CLR_FN"]), "fp": int(c["CLR_FP"])}


def score(output, videos):
    summary = {}
    for ds, (split, evaluator) in SPLITS.items():
        names = [v["name"] for v in videos if v["dataset"] == ds]
        assert len(names) == 4
        summary[ds] = {}
        for version in "AB":
            _, cfg, _ = rte.build_configs(split, version, None)
            cfg.update({"GT_FOLDER": str(output / "gt"), "TRACKERS_FOLDER": str(output / "trackers"),
                        "SEQ_INFO": dict.fromkeys(names, 150)})
            dataset = rte.dataset_class(evaluator)(cfg)
            metrics = (trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
                       trackeval.metrics.Identity({"PRINT_CONFIG": False}))
            raw = {}
            for name in names:
                data = dataset.get_preprocessed_seq_data(dataset.get_raw_seq_data(version, name), "pedestrian")
                assert data["num_timesteps"] == 150
                raw[name] = {m.get_name(): m.eval_sequence(data) for m in metrics}
            combined = {m.get_name(): m.combine_sequences({seq: item[m.get_name()] for seq, item in raw.items()})
                        for m in metrics}
            summary[ds][version] = {"combined": scalars(combined), "videos": {seq: scalars(item) for seq, item in raw.items()}}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    videos = stage_inputs(args.logs, args.stage)
    result = score(args.stage, videos)
    (args.logs / "nano_quality.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: {a: r["combined"] for a, r in v.items()} for k, v in result.items()}), flush=True)


if __name__ == "__main__":
    main()
