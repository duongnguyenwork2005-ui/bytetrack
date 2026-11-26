"""Reconstruct and certify all UTUAV Q3 full scores from sealed run artifacts."""
from __future__ import annotations

import csv
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval as run

METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "FP", "FN")


def main():
    log = run.prep.LOG / "full"
    summary = run.read(log / "summary.json")
    run.require(summary["status"] == "complete" and summary["runs"] == 36
                and summary["csv_sha256"] == run.prep.digest(log / "combined_tung_run.csv"),
                "full summary incomplete/stale")
    phase = run.Phase("full", (dt.datetime.now().astimezone()+dt.timedelta(minutes=110)).isoformat())
    run.require(summary["protocol_hash"] == phase.fingerprint and
                summary["source_manifest_sha256"] == phase.source_sha, "full fingerprint mismatch")
    cache = {f"{detector}/{video}": run.check_cache(phase, detector, video)
             for detector in run.r64.DETECTOR for video in phase.videos}
    rows, per_video, deltas, certificates = [], [], [], {}
    for detector in run.r64.DETECTOR:
        for condition in run.r64.DIEU_KIEN:
            for loc, (_, seeds) in run.r64.BO_LOC.items():
                for seed in seeds or [None]:
                    name = run.run_name(detector, condition, loc, seed)
                    units = [log/"units"/name/v/"unit.json" for v in phase.videos]
                    run.require(all(p.is_file() for p in units), f"{name}: tracking unit missing")
                    for video, path in zip(phase.videos, units):
                        m = run.read(path)
                        run.check_hashes(path.parent, m["sha256"])
                        run.validate_track(path.parent, video, phase.n(video))
                    track = run.config.PROCESSED_DIR/"trackers"/phase.split/name
                    meta = run.read(track/"run_meta.json")
                    run.require(meta["settings"]["protocol_hash"] == phase.fingerprint
                                and meta["videos"] == phase.videos, f"{name}: run meta differs")
                    for video in phase.videos:
                        for filename in run.paths(video):
                            unit = run.read(log/"units"/name/video/"unit.json")
                            run.require(run.prep.digest(track/filename) == unit["sha256"][filename],
                                        f"{name}/{video}: published track changed")
                    score = run.config.RESULTS_DIR/"trackeval"/phase.split/name
                    cert = run.read(score/"unit.json")
                    run.require(cert["validated"] and cert["fingerprint"] == run.fp.sha256_obj(meta),
                                f"{name}: score marker differs")
                    run.check_hashes(score, cert["sha256"])
                    raw = run.read(score/"raw_fields.json")
                    run.require(set(raw) == set(phase.videos), f"{name}: raw set differs")
                    combined = pd.read_csv(score/"combined_metrics.csv")
                    seq = pd.read_csv(score/"per_video_metrics.csv")
                    run.require(len(combined) == 1 and combined.iloc[0]["video"] == "COMBINED_SEQ"
                                and len(seq) == 2 and set(seq["video"]) == set(phase.videos),
                                f"{name}: metric rows differ")
                    recalc = run.combine(raw, phase.videos)
                    delta = max(abs(float(recalc[k])-float(combined.iloc[0][k])) for k in METRICS)
                    run.require(delta <= 1e-9 and all(math.isfinite(float(recalc[k])) for k in METRICS),
                                f"{name}: metric recomposition differs")
                    deltas.append(delta)
                    base = {"run": name, "detector": detector, "condition": condition,
                            "filter": loc, "seed": "" if seed is None else seed}
                    rows.append({**base, **{k: recalc[k] for k in METRICS}})
                    for _, item in seq.iterrows():
                        per_video.append({**base, "video": item["video"],
                            **{k: float(item[k]) for k in METRICS}})
                    certificates[name] = run.prep.digest(score/"unit.json")
    run.require(len(rows) == 36 and len(per_video) == 72 and len(cache) == 6,
                "full coverage differs")
    prior = pd.read_csv(log/"combined_tung_run.csv").set_index("run")
    run.require(set(prior.index) == set(certificates), "matrix CSV run set differs")
    for row in rows:
        old = prior.loc[row["run"]]
        run.require(all(abs(float(old[k])-float(row[k])) <= 1e-9 for k in METRICS),
                    f"{row['run']}: matrix CSV differs from scores")
    per_path = log/"per_video.csv"
    pd.DataFrame(per_video).to_csv(per_path, index=False)
    combined_summary = []
    for detector in run.r64.DETECTOR:
        for condition in run.r64.DIEU_KIEN:
            for loc in run.r64.BO_LOC:
                subset = [row for row in rows if row["detector"] == detector
                          and row["condition"] == condition and row["filter"] == loc]
                run.require(len(subset) == (3 if loc == "pf500" else 1),
                            f"{detector}/{condition}/{loc}: seed coverage")
                item = {"detector": detector, "condition": condition, "filter": loc,
                        "n_seeds": len(subset)}
                item.update({k: float(np.mean([row[k] for row in subset])) for k in METRICS})
                if loc == "pf500":
                    item["HOTA_min"] = min(row["HOTA"] for row in subset)
                    item["HOTA_max"] = max(row["HOTA"] for row in subset)
                combined_summary.append(item)
    filter_path = log/"filter_summary.csv"
    pd.DataFrame(combined_summary).to_csv(filter_path, index=False)
    evidence = {"decision": "PASS", "protocol_hash": phase.fingerprint,
        "source_manifest_sha256": phase.source_sha,
        "xml_audit_sha256": phase.audit_sha,
        "cache_units": 6, "tracking_units": 72, "score_runs": 36,
        "max_recomposition_delta": max(deltas),
        "matrix_sha256": run.prep.digest(log/"combined_tung_run.csv"),
        "filter_summary_sha256": run.prep.digest(filter_path),
        "per_video_sha256": run.prep.digest(per_path),
        "score_certificates": certificates, "cache_certificates": cache}
    run.write(log/"final_review.json", evidence)
    print("PASS", evidence["cache_units"], evidence["tracking_units"],
          evidence["score_runs"], evidence["max_recomposition_delta"])


if __name__ == "__main__":
    main()
