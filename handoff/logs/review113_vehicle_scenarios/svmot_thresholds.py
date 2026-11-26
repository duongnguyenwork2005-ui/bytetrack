"""Check detection candidates and compare both preexisting SVMOT cutoffs."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "review112_svmot_pf"))
from diagnose_svmot import MANIFEST, ORIGINAL, ROOT, _BaseDataset, det_cache, np, pd, reconstruct, rte


def counts(gt, frames, cutoff):
    present_low = gt[:, 8] <= .5
    hits = Counter({"low_gt": int(present_low.sum()), "other_gt": int((~present_low).sum())})
    by_frame = np.split(gt, np.flatnonzero(np.diff(gt[:, 0])) + 1)
    for rows in by_frame:
        source = frames[int(rows[0, 0]) - 1]
        boxes = source[(source[:, 4] > cutoff) & np.isin(source[:, 5], [1, 2, 3, 5, 7]), :4][:300].copy()
        if not len(boxes):
            continue
        boxes[:, 2:4] -= boxes[:, :2]
        hit = _BaseDataset._calculate_box_ious(rows[:, 2:6], boxes, box_format="xywh").max(axis=1) >= .5
        hits.update(low_hits=int((hit & (rows[:, 8] <= .5)).sum()), other_hits=int((hit & (rows[:, 8] > .5)).sum()))
    return hits


def main():
    thresholds = [.25, .10, .05]
    coverage_all, tracking_all = defaultdict(Counter), defaultdict(Counter)
    rows, hashes = [], {}
    for video, info in sorted(MANIFEST.items()):
        official = info["split"] == "official_test"
        split = "SVMOTSYS-test" if official else "SVMOT-release21"
        base = ROOT / "data/processed" if official else ORIGINAL / "ground_truth"
        gtpath = base / split / video / "gt/gt.txt"
        gt = np.loadtxt(gtpath, delimiter=",", ndmin=2)
        hashes[str(gtpath)] = hashlib.sha256(gtpath.read_bytes()).hexdigest()
        length = info["frames"]
        if official:
            cache = det_cache.CACHE_ROOT / "r64_svmot_yolov8n_coco_c0.05" / f"{video}.npz"
            frames = [row[-1] for row in det_cache.open_cache(cache.parent.name, video, .05, [1, 2, 3, 5, 7], 300).iter_frames()]
        else:
            cache = ORIGINAL / "cache" / f"{video}.npz"
            with np.load(cache) as archive:
                frames = [archive[str(index)] for index in range(length)]
        assert len(frames) == length
        hashes[str(cache)] = hashlib.sha256(cache.read_bytes()).hexdigest()
        for cutoff in thresholds:
            result = counts(gt, frames, cutoff)
            coverage_all[(info["split"], cutoff)].update(result)
            rows.append({"video": video, "cutoff": cutoff, "split": info["split"], **result})
        if not official:
            continue
        ids = np.unique(gt[:, 1])
        present = np.zeros((length, len(ids)), bool)
        low = present.copy()
        index = gt[:, 0].astype(int) - 1, np.searchsorted(ids, gt[:, 1])
        present[index] = True
        low[index] = gt[:, 8] <= .5
        recent = low.copy()
        for lag in range(1, 31):
            recent[lag:] |= low[:-lag]
        for cutoff in ["c25", "c10"]:
            for method in ["cv", "ekf", "ukf"]:
                name = f"r64-yolov8n_coco-{cutoff}-{method}"
                _, dc, _ = rte.build_configs(split, name, None)
                dc.update(PRINT_CONFIG=False, SEQ_INFO={video: length})
                ds = rte.dataset_class("mot")(dc)
                source = ds.get_raw_seq_data(name, video)
                data = ds.get_preprocessed_seq_data(source, "pedestrian")
                assert all(np.array_equal(a, ids[b]) for a, b in zip(source["gt_ids"], data["gt_ids"]))
                matched, events = reconstruct(data, present.shape)
                ref = pd.read_csv(ROOT / "results/trackeval" / split / name / "per_video_metrics.csv").set_index("video").loc[video]
                missed = present & ~matched
                assert len(events) == ref.IDSW and missed.sum() == ref.FN
                tracking_all[(cutoff, method)].update(IDSW=int(ref.IDSW), FN=int(ref.FN), FP=int(ref.FP),
                    low_IDSW=sum(bool(low[t, g]) for t, g, _ in events), low_FN=int((missed & low).sum()),
                    recent_IDSW=sum(bool(recent[t, g]) for t, g, _ in events), recent_FN=int((missed & recent).sum()))
        print(video, flush=True)
    report = {"coverage": {f"{g}/{c}": dict(v) for (g, c), v in coverage_all.items()},
              "tracking_test9": {f"{c}/{m}": dict(v) for (c, m), v in tracking_all.items()},
              "coverage_by_video": rows, "gt_cache_sha256": hashes}
    (HERE / "svmot_thresholds.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["coverage", "tracking_test9"]}), flush=True)


if __name__ == "__main__":
    main()
