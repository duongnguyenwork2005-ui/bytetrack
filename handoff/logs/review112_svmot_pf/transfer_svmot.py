"""Paired transfer comparison on all nine original SVMOT test clips."""

import hashlib
import json
from pathlib import Path
import run_pf as setup
import numpy as np
import trackeval
from rbpf import installed

runner, OUT, ROOT = setup.runner, setup.OUT, setup.ROOT
DEST = OUT / "transfer_svmot"
SPLIT = "SVMOTSYS-test"
VIDEOS = [f"seq_{i}_{'morning' if i < 26 else 'evening'}" for i in range(22, 31)]


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    gate = json.loads((OUT / "dev_gate.json").read_text(encoding="utf-8"))
    assert gate["gate_pass"], "Transfer starts only after the registered dev screen passes"
    DEST.mkdir(exist_ok=True)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [OUT / "rbpf.py", Path(__file__), runner.CFG, OUT / "dev_gate.json"]}
    for seed in range(3):
        for video in VIDEOS:
            for case in ["base", "candidate"]:
                path = DEST / "tracking" / SPLIT / f"{case}-s{seed}"
                for folder in ["data", "data_cls", "timing"]:
                    (path / folder).mkdir(parents=True, exist_ok=True)
                target = path / "data" / f"{video}.txt"
                assert not target.exists()
                cache = runner.det_cache.open_cache(
                    "r64_svmot_yolov8n_coco_c0.05", video, .25, [1, 2, 3, 5, 7], 300
                )
                cfg = runner.rt.load_tracker_cfg(runner.CFG, {"track_buffer": 36})
                with installed(case == "candidate") as counters:
                    mot, classes, timing = runner.rt.track_video(
                        cache, cfg, dt_nominal=1 / 30, elapsed_mode="steps", seed=seed, video=video
                    )
                assert len(timing) == len(cache)
                assert np.isfinite(mot.to_numpy(float)).all()
                assert (mot[["bb_width", "bb_height"]].to_numpy() > 0).all()
                assert not mot.duplicated(["frame", "id"]).any()
                mot.to_csv(target, index=False, header=False, float_format="%.10f")
                classes.to_csv(path / "data_cls" / f"{video}.csv", index=False)
                timing.to_csv(path / "timing" / f"{video}.csv", index=False)
                write(path / "run_meta.json", {"case": case, "seed": seed, "fps": 30,
                                              "cfg": runner.rt.cfg_to_dict(cfg),
                                              "candidate": "RBPF" if case == "candidate" else "PF500 K1/K2 motion2"})
                print("TRANSFER", case, seed, video, len(timing), counters, flush=True)
    summaries = {}
    for seed in range(3):
        for case in ["base", "candidate"]:
            tracker = f"{case}-s{seed}"
            path = DEST / "evaluation" / tracker
            path.mkdir(parents=True)
            ec, dc, mc = runner.rte.build_configs(SPLIT, tracker, VIDEOS)
            ec.update(PRINT_RESULTS=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
                      LOG_ON_ERROR=str(path / "error.log"))
            dc.update(TRACKERS_FOLDER=str(DEST / "tracking"), OUTPUT_FOLDER=str(path))
            metrics = [trackeval.metrics.HOTA(mc), trackeval.metrics.CLEAR(mc), trackeval.metrics.Identity(mc)]
            dataset = trackeval.datasets.MotChallenge2DBox(dc)
            results, messages = trackeval.Evaluator(ec).evaluate([dataset], metrics)
            assert all(m == "Success" for m in next(iter(messages.values())).values())
            raw = runner.boot_stats.save_raw_fields(results, tracker, path / "raw_fields.json")
            combined = runner.boot_stats.combine(raw, VIDEOS)
            table = runner.rte.flatten_results(results, tracker)
            table.to_csv(path / "metrics.csv", index=False)
            combined_row = table[table.video == "COMBINED_SEQ"].iloc[0]
            assert all(abs(combined[k] - combined_row[k]) < 1e-9 for k in runner.METRICS)
            summaries[tracker] = combined
    summary = {case: {k: float(np.mean([summaries[f"{case}-s{s}"][k] for s in range(3)]))
                      for k in runner.METRICS} for case in ["base", "candidate"]}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in before.items())
    write(DEST / "verified_summary.json", {"units": 54, "videos": VIDEOS,
          "summary": summary, "per_seed": summaries, "frozen_sha256": before,
          "interpretation": "Transfer on repeatedly observed SVMOT test scenes; exploratory, no parameter selection"})
    print("TRANSFER_COMPLETE", summary, flush=True)


if __name__ == "__main__":
    main()
