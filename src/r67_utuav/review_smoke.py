"""Independent technical gate for UTUAV Q3 smoke; performance is never a gate."""
import datetime as dt
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval as run


def main():
    for test in ("test_utuav.py", "test_metric_fixture.py"):
        result = subprocess.run([sys.executable, "-B", str(HERE/test)], cwd=run.ROOT,
                                capture_output=True, text=True)
        run.require(result.returncode == 0, f"fixture {test} failed: {result.stdout} {result.stderr}")
    summary_path = run.prep.LOG/"smoke/summary.json"
    repeat_path = run.prep.LOG/"smoke/repeat_pf.json"
    recall_path = run.prep.LOG/"smoke/detector_upper_bound_recall.json"
    visual_path = run.prep.LOG/"gt_visual_first_mid_last.jpg"
    summary = run.read(summary_path)
    repeat = run.read(repeat_path)
    recall = run.read(recall_path)
    run.require(summary["status"] == "complete" and summary["runs"] == 36
                and repeat["status"] == "PASS" and len(recall["rows"]) == 18
                and visual_path.is_file(), "smoke evidence incomplete")
    phase = run.Phase("smoke", (dt.datetime.now().astimezone()+dt.timedelta(minutes=110)).isoformat())
    run.require(summary["protocol_hash"] == phase.fingerprint
                and repeat["protocol_hash"] == phase.fingerprint
                and summary["source_manifest_sha256"] == phase.source_sha,
                "smoke protocol/source differ")
    run.require(summary["csv_sha256"] == run.prep.digest(phase.log/"combined_tung_run.csv"),
                "smoke summary stale")
    cache = []
    units = 0
    for detector in run.r64.DETECTOR:
        for video in phase.videos:
            cache.append(run.check_cache(phase, detector, video))
    for detector in run.r64.DETECTOR:
        for cond in run.r64.DIEU_KIEN:
            for loc, (_, seeds) in run.r64.BO_LOC.items():
                for seed in seeds or [None]:
                    name = run.run_name(detector, cond, loc, seed)
                    run.require(all((phase.log/"units"/name/video/"unit.json").is_file()
                                    for video in phase.videos), f"{name}: missing technical unit")
                    run.require((run.config.RESULTS_DIR/"trackeval"/phase.split/name/"unit.json").is_file(),
                                f"{name}: missing score unit")
                    metric = run.tracking(phase, detector, cond, loc, seed)
                    run.require(all(math.isfinite(float(value)) for value in metric.values()),
                                f"{name}: nonfinite metric")
                    units += 2
    repeat_name = run.run_name("yolov8n_mixed5", "c25", "pf500", 0, True)
    run.require(all((phase.log/"units"/repeat_name/v/"unit.json").is_file()
                    for v in phase.videos), "repeat PF units missing")
    run.repeat_pf(phase)
    run.require(units == 72 and len(set(cache)) == 6, "smoke coverage mismatch")
    path = phase.log/"review.json"
    decision = {"decision": "PASS", "reason": "technical correctness; no score or recall threshold",
        "protocol_hash": phase.fingerprint, "cache_units": 6, "tracking_units": units,
        "score_runs": 36, "pf_repeat": "byte-identical MOT and score",
        "fixtures": ["converter: four cases", "metric: oracle+distractor+empty"],
        "source_boundary_boxes": {"B": 3, "C": 42, "maximum_excursion_px": 4,
            "handling": "unclipped source XML"},
        "summary_sha256": run.prep.digest(summary_path),
        "repeat_sha256": run.prep.digest(repeat_path),
        "recall_sha256": run.prep.digest(recall_path),
        "visual_sha256": run.prep.digest(visual_path)}
    run.write(path, decision)
    print(path, "PASS", units)


if __name__ == "__main__":
    main()
