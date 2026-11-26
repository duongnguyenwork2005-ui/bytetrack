"""Evaluate all completed R103 Nano comparisons with the R102 GT/evaluators."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import tarfile

import numpy as np
from scipy.optimize import linear_sum_assignment

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "review102_ridesafe_pf_nano"))
import eval_nano as evaluation  # noqa: E402

SPLITS = evaluation.SPLITS.copy()
FPS = {"bdd": 5, "ua": 25}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def speed_stats(samples, mode):
    results = {}
    for dataset in SPLITS:
        group = [row for row in samples.values() if row["dataset"] == dataset and row["mode"] == mode]
        if not group:
            continue
        base = [samples[(dataset, row["name"], "full")] for row in group]
        out = {key: [median(row[key] for row in base), median(row[key] for row in group)]
               for key in ("fps_steady", "p90_latency_ms", "decode_ms", "infer_ms", "n_inferred", "mot_rows")}
        out["paired_fps_gain_pct"] = median(100 * (b["fps_steady"] / a["fps_steady"] - 1) for a, b in zip(base, group))
        out["mot_equal"] = sum(a["mot_sha256"] == b["mot_sha256"] for a, b in zip(base, group))
        out["class_equal"] = sum(a["cls_sha256"] == b["cls_sha256"] for a, b in zip(base, group))
        out["clips"] = len(group)
        results[dataset] = out
    return results


def stage(stage_path, videos, samples, contents, mode):
    relevant = [v for v in videos if (v["dataset"], v["name"], mode) in samples]
    coverage = defaultdict(int)
    for video in relevant:
        ds, name = video["dataset"], video["name"]
        split, _ = SPLITS[ds]
        source = evaluation.ROOT / "data/processed" / split / name / "gt/gt.txt"
        gt_file = stage_path / "gt" / split / name / "gt/gt.txt"
        gt_file.parent.mkdir(parents=True, exist_ok=True)
        gt = evaluation.clipped_gt(source, video["first"], video["last"])
        with gt_file.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerows(gt)
        if mode == "roi256":
            top = samples[(ds, name, mode)]["roi_y"]
            for row in gt:
                if int(float(row[6])) != 1 or int(float(row[7])) != 1:
                    continue
                y, height = float(row[3]), float(row[5])
                assert height > 0
                overlap = max(0, min(y + height, top + 384) - max(y, top))
                coverage["gt_boxes"] += 1
                coverage["gt_outside"] += overlap == 0
                coverage["gt_half_or_more_inside"] += overlap >= .5 * height
        for version, mode_name in (("A", "full"), ("B", mode)):
            payload = contents[f"{ds}/{name}/{mode_name}.txt"]
            row = samples[(ds, name, mode_name)]
            assert len(payload.splitlines()) == row["mot_rows"] and sha(payload) == row["mot_sha256"]
            keys = [tuple(map(int, line.split(b",")[:2])) for line in payload.splitlines()]
            assert len(set(keys)) == len(keys) and all(1 <= frame <= 150 for frame, _ in keys)
            tracker = stage_path / "trackers" / split / version / "data" / f"{name}.txt"
            tracker.parent.mkdir(parents=True, exist_ok=True)
            tracker.write_bytes(payload)
    return relevant, dict(coverage)


def first_matches(data):
    count = data["num_gt_ids"]
    first_seen = np.full(count, -1, dtype=int)
    first_match = np.full(count, -1, dtype=int)
    previous_id = np.full(count, -1, dtype=int)
    eps = np.finfo(float).eps
    for step, (gt, trackers) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
        unset = gt[first_seen[gt] < 0]
        first_seen[unset] = step
        if not len(gt) or not len(trackers):
            continue
        similarity = data["similarity_scores"][step]
        score = 1000 * (trackers == previous_id[gt, None]) + similarity
        score[similarity < .5 - eps] = 0
        rows, cols = linear_sum_assignment(-score)
        keep = score[rows, cols] > eps
        matched, identities = gt[rows[keep]], trackers[cols[keep]]
        unset = matched[first_match[matched] < 0]
        first_match[unset] = step
        previous_id.fill(-1)
        previous_id[matched] = identities
    return first_seen, first_match


def start_delays(stage_path, relevant):
    results = {}
    for ds in evaluation.SPLITS:
        split, evaluator = SPLITS[ds]
        names = [v["name"] for v in relevant if v["dataset"] == ds]
        runs = {}
        for version in "AB":
            _, cfg, _ = evaluation.rte.build_configs(split, version, None)
            cfg.update({"GT_FOLDER": str(stage_path / "gt"), "TRACKERS_FOLDER": str(stage_path / "trackers"),
                        "SEQ_INFO": dict.fromkeys(names, 150)})
            dataset = evaluation.rte.dataset_class(evaluator)(cfg)
            runs[version] = {name: first_matches(dataset.get_preprocessed_seq_data(
                dataset.get_raw_seq_data(version, name), "pedestrian")) for name in names}
        comparable, matched_a, matched_b, entering = [], 0, 0, 0
        for name in names:
            (first_a, match_a), (first_b, match_b) = (runs[v][name] for v in "AB")
            assert np.array_equal(first_a, first_b)
            new = first_a > 0
            entering += int(new.sum())
            matched_a += int((new & (match_a >= 0)).sum())
            matched_b += int((new & (match_b >= 0)).sum())
            common = new & (match_a >= 0) & (match_b >= 0)
            comparable.extend((match_b[common] - match_a[common]).tolist())
        results[ds] = {"entering_gt_ids": entering, "matched_at_least_once": [matched_a, matched_b],
                       "matched_by_both": len(comparable), "candidate_later": sum(n > 0 for n in comparable),
                       "candidate_earlier": sum(n < 0 for n in comparable),
                       "median_added_ms": None if not comparable else median(comparable) * 1000 / FPS[ds]}
    return results


def main():
    logs = Path(sys.argv[1])
    stage_root = Path(sys.argv[2]).resolve()
    videos = json.loads((logs / "videos.json").read_text())
    rows = json.loads((logs / "pairs.json").read_text())
    samples = {(r["dataset"], r["name"], r["mode"]): r for r in rows}
    assert len(videos) == 8 and len(samples) == len(rows) and len(rows) in (16, 24)
    expected = {f"{ds}/{name}/{mode}.txt" for ds, name, mode in samples}
    with tarfile.open(logs / "mot_outputs.tar.gz", "r:gz") as file:
        assert set(file.getnames()) == expected
        contents = {name: file.extractfile(name).read() for name in expected}
    stats = {}
    for mode in sorted({mode for _, _, mode in samples} - {"full"}):
        stats[mode] = {"speed": speed_stats(samples, mode)}
        if mode == "early" and all(x["mot_equal"] == x["class_equal"] == x["clips"] for x in stats[mode]["speed"].values()):
            stats[mode]["exact_output_parity"] = True
            continue
        stage_path = stage_root / mode
        relevant, coverage = stage(stage_path, videos, samples, contents, mode)
        evaluation.SPLITS = {ds: item for ds, item in SPLITS.items() if any(r["dataset"] == ds for r in relevant)}
        if mode == "early":
            stats[mode]["exact_output_parity"] = False
            continue
        stats[mode]["quality"] = evaluation.score(stage_path, relevant)
        stats[mode]["first_match_delay"] = start_delays(stage_path, relevant)
        if coverage:
            stats[mode]["roi_coverage"] = coverage
    (logs / "assessment.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({name: {"speed": data["speed"], "quality": {
        ds: {version: x["combined"] for version, x in groups.items()} for ds, groups in data.get("quality", {}).items()}}
        for name, data in stats.items()}))


if __name__ == "__main__":
    main()
