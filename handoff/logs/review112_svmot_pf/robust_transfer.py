"""Transfer the fixed robust candidate, reusing the already replayed PF baseline."""

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
VIDEOS = [f"seq_{i}_{'morning' if i < 26 else 'evening'}" for i in range(22, 31)]
DEST = HERE / "robust/transfer_svmot"


def worker(seed):
    import run_pf as setup
    from robust_filter import installed
    import numpy as np
    rt, cache_module = setup.runner.rt, setup.runner.det_cache
    for video in VIDEOS:
        cfg = rt.load_tracker_cfg(setup.runner.CFG, {"track_buffer": 36})
        cache = cache_module.open_cache("r64_svmot_yolov8n_coco_c0.05", video, .25, [1, 2, 3, 5, 7], 300)
        with installed() as events:
            mot, classes, timing = rt.track_video(cache, cfg, 1/30, elapsed_mode="steps", seed=seed, video=video)
        assert len(timing) == len(cache) and np.isfinite(mot.to_numpy(float)).all()
        assert not mot.duplicated(["frame", "id"]).any()
        target = DEST / "tracking/SVMOTSYS-test" / f"candidate-s{seed}"
        for folder in ["data", "timing"]:
            (target / folder).mkdir(parents=True, exist_ok=True)
        assert not (target / "data" / f"{video}.txt").exists()
        mot.to_csv(target / "data" / f"{video}.txt", index=False, header=False, float_format="%.10f")
        timing.to_csv(target / "timing" / f"{video}.csv", index=False)
        print("ROBUST TRANSFER", seed, video, len(timing), flush=True)
    return seed


def score():
    import run_pf as setup
    import numpy as np
    import pandas as pd
    import trackeval
    from diagnose_svmot import reconstruct
    root, runner = setup.ROOT, setup.runner
    baseline = json.loads((HERE / "transfer_svmot/verified_summary.json").read_text())
    baseline_low = json.loads((HERE / "verification.json").read_text())["transfer_low_visibility"]["base"]
    values, low_rows = [], []
    for seed in range(3):
        tracker = f"candidate-s{seed}"
        ec, dc, mc = runner.rte.build_configs("SVMOTSYS-test", tracker, VIDEOS)
        ec.update(PRINT_RESULTS=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
                  LOG_ON_ERROR=str(DEST / "error.log"))
        dc["TRACKERS_FOLDER"] = str(DEST / "tracking")
        dataset = trackeval.datasets.MotChallenge2DBox(dc)
        result, messages = trackeval.Evaluator(ec).evaluate([dataset], [
            trackeval.metrics.HOTA(mc), trackeval.metrics.CLEAR(mc), trackeval.metrics.Identity(mc)])
        assert all(v == "Success" for v in next(iter(messages.values())).values())
        path = DEST / "evaluation" / tracker
        path.mkdir(parents=True)
        raw = runner.boot_stats.save_raw_fields(result, tracker, path / "raw_fields.json")
        combined = runner.boot_stats.combine(raw, VIDEOS)
        runner.rte.flatten_results(result, tracker).to_csv(path / "metrics.csv", index=False)
        values.append(combined)
        for video in VIDEOS:
            gt = np.loadtxt(root / "data/processed/SVMOTSYS-test" / video / "gt/gt.txt", delimiter=",")
            ids = np.unique(gt[:, 1]).astype(int)
            present = np.zeros((dataset.seq_lengths[video], len(ids)), bool)
            low = present.copy()
            at = gt[:, 0].astype(int) - 1, np.searchsorted(ids, gt[:, 1])
            present[at] = True
            low[at] = gt[:, 8] <= .5
            raw_data = dataset.get_raw_seq_data(tracker, video)
            data = dataset.get_preprocessed_seq_data(raw_data, "pedestrian")
            assert all(np.array_equal(a, ids[b]) for a, b in zip(raw_data["gt_ids"], data["gt_ids"]))
            matched, events = reconstruct(data, present.shape)
            assert len(events) == raw[video]["IDSW"]
            missed = present & ~matched
            assert missed.sum() == raw[video]["CLR_FN"]
            low_rows.append({"seed": seed, "video": video,
                             "IDSW": sum(bool(low[f, g]) for f, g, _ in events),
                             "FN": int((missed & low).sum())})
    candidate = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
    low = {k: sum(row[k] for row in low_rows) / 3 for k in ["IDSW", "FN"]}
    base = baseline["summary"]["base"]
    conditions = {"IDSW": candidate["IDSW"] <= base["IDSW"],
                  "low_IDSW": low["IDSW"] <= baseline_low["IDSW"],
                  "HOTA": candidate["HOTA"] >= base["HOTA"] - .005,
                  "FN": candidate["FN"] <= 1.005 * base["FN"],
                  "all_seeds": all(values[s]["IDSW"] <= baseline["per_seed"][f"base-s{s}"]["IDSW"]
                                   for s in range(3))}
    result = {"baseline": base, "candidate": candidate, "candidate_seeds": values,
              "baseline_low": baseline_low, "candidate_low": low,
              "conditions": conditions, "gate_pass": all(conditions.values()),
              "new_replay_units": 27, "reused_baseline_units": 27,
              "baseline_source": str(HERE / "transfer_svmot/verified_summary.json")}
    (DEST / "verified_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(low_rows).to_csv(DEST / "visibility.csv", index=False)
    print("ROBUST TRANSFER COMPLETE", json.dumps(result), flush=True)


def main(stage):
    gate = json.loads((HERE / "robust/dev_gate.json").read_text())
    assert gate["gate_pass"]
    DEST.mkdir(exist_ok=True)
    if stage == "run":
        with ProcessPoolExecutor(max_workers=3) as pool:
            assert list(pool.map(worker, range(3))) == [0, 1, 2]
    elif stage == "score":
        score()
    else:
        raise ValueError(stage)


if __name__ == "__main__":
    main(sys.argv[1])
