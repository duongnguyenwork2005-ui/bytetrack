"""Evaluate audited before/after events from cached MOT results."""

from collections import Counter
import json

import numpy as np

from run import HERE, OUT, PREVIOUS, score
from check_occlusion import candidate_coverage
from detect import DATA


def assignments(name, recorded):
    gt = np.loadtxt(score.WORK / "gt" / name / "gt" / "gt.txt", delimiter=",")
    frames = recorded["frames"]
    ids = np.unique(gt[:, 1]).astype(int)
    present = np.zeros((frames, len(ids)), dtype=bool)
    present[gt[:, 0].astype(int) - 1, np.searchsorted(ids, gt[:, 1])] = True
    result = {}
    for method, variant in recorded["variants"].items():
        dataset = score.trackeval.datasets.MotChallenge2DBox(
            {"PRINT_CONFIG": False, "TRACKERS_TO_EVAL": [method], "BENCHMARK": "FastTrack",
             "SPLIT_TO_EVAL": "full", "SKIP_SPLIT_FOL": True, "DO_PREPROC": False,
             "GT_FOLDER": str(score.WORK / "gt"), "TRACKERS_FOLDER": str(score.WORK / "trackers"),
             "SEQ_INFO": {name: frames}})
        raw = dataset.get_raw_seq_data(method, name)
        data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
        assert all(np.array_equal(ids[a], b) for a, b in zip(data["gt_ids"], raw["gt_ids"]))
        assigned, switched = score.clear_assignments(data, present.shape)
        assert int(switched.sum()) == variant["metrics"]["IDSW"]
        assert int((present & (assigned < 0)).sum()) == variant["metrics"]["FN"]
        result[method] = assigned
    return ids, present, result


def count(clips):
    totals, jointly_matched = {}, []
    for sequence, rows in clips.items():
        for event in rows:
            for method, status in event["filters"].items():
                totals.setdefault(method, Counter())[status] += 1
            if all(s in ("same_id", "different_id") for s in event["filters"].values()):
                jointly_matched.append({"sequence": sequence, "gt_car": event["gt_car"],
                                        "gt_blocker": event["gt_blocker"], "before": event["before"],
                                        "filters": event["filters"]})
    denominator = sum(map(len, clips.values()))
    jointly_kept = {method: sum(row["filters"][method] == "same_id" for row in jointly_matched)
                    for method in totals}
    assert all(sum(v.values()) == denominator for v in totals.values())
    return {"all_events": denominator, "blocker_ids_per_sequence":
            len({(s, e["gt_blocker"]) for s, rows in clips.items() for e in rows}),
            "counts": totals, "jointly_matched": len(jointly_matched), "same_id_on_joint": jointly_kept}


def score_events(name, record, candidates, keep):
    ids, present, assignment = assignments(name, record)
    result = []
    for event in candidates:
        if event["number"] not in keep:
            continue
        car, blocker = event["car"], event["blocker"]
        before, middle, after = event["before"], event["peak"], event["after"]
        column = np.searchsorted(ids, car)
        assert ids[column] == car and present[[before-1, middle-1, after-1], column].all()
        tags = {}
        for method, assigned in assignment.items():
            start, end = assigned[before-1, column], assigned[after-1, column]
            if start < 0:
                state = "before_missing"
            elif end < 0:
                state = "after_missing"
            else:
                state = "same_id" if start == end else "different_id"
            tags[method] = state
        result.append({"gt_car": car, "gt_blocker": blocker, "before": before, "peak": middle,
                       "after": after, "filters": tags})
    assert len(result) == len(keep)
    return result


def evaluate():
    selection = json.loads((HERE / "verified_events.json").read_text())
    events, coverage = {}, {}
    for path in sorted((HERE / "candidates").glob("*.json")):
        name = path.stem
        assert name in selection, f"Missing review for {name}"
        candidates = json.loads(path.read_text())
        keep = set(selection[name])
        assert len(keep) == len(selection[name]) and keep <= {e["number"] for e in candidates}
        events[name] = []
        record = json.loads((OUT / f"{name}.json").read_text())
        gt = np.loadtxt(score.WORK / "gt" / name / "gt" / "gt.txt", delimiter=",")
        coverage[name] = candidate_coverage(gt, score.Cache(DATA / f"{name}_yolov8n.npz", record["fps"]))
        if keep:
            events[name] = score_events(name, record, candidates, keep)
    assert set(events) == set(selection) and len(events) == 9
    prior = json.loads((PREVIOUS / "scores/fasttrack_verified_occlusion.json").read_text())
    total = prior["clips"] | events
    coverage.update(prior["detector_candidate_coverage"])
    score.write_json(OUT / "occlusion.json", {"new9": count(events), "all12": count(total), "clips": total,
                                            "candidate_coverage": coverage})
    print(count(events), count(total), flush=True)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["fixed"]:
        source = json.loads((HERE / "fixed_candidates.json").read_text())
        name = source["sequence"]
        record = json.loads((OUT / f"{name}.json").read_text())
        accepted = set(json.loads((HERE / "fixed_verified.json").read_text()))
        items = score_events(name, record, source["events"], accepted)
        totals = count({name: items})
        by_blocker = {blocker: count({name: [event for event in items if event["gt_blocker"] == blocker]})
                      for blocker in source["regions"]}
        score.write_json(OUT / "fixed_occlusion.json", {"totals": totals, "by_blocker": by_blocker,
                                                      "clip": name, "events": items})
        print(totals, by_blocker, flush=True)
    else:
        evaluate()
