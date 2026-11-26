"""Tier 1 audit of collected R105 evidence; runs no detector or remote job."""
import argparse
import csv
import difflib
import inspect
import io
import json
from pathlib import Path
from statistics import median
import tarfile

import eval_benchmark as assess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", type=Path)
    parser.add_argument("stage", type=Path, help="Existing, scored staging directory")
    args = parser.parse_args()
    smoke_videos, smoke, _ = assess.load_evidence(args.host / "smoke", phase="smoke")
    videos, samples, _ = assess.load_evidence(args.host / "full")
    for v in smoke_videos:
        name = v["dataset"] + "_" + v["name"] + "_images.json"
        assert (args.host / "smoke" / name).read_bytes() == (args.host / "full" / name).read_bytes()
    host_exit = json.loads((args.host / "host_exit.json").read_text())
    assert all(host_exit[key] == 0 for key in ("smoke_exit", "ssh_probe_exit", "full_exit", "collection_exit"))
    assert host_exit["active_r105_jobs"] == []
    assert "R105_BENCH_COMPLETE full" in (args.host / "full.log").read_text()
    assert "R105_BENCH_COMPLETE smoke" in (args.host / "smoke.log").read_text()
    assessment = json.loads((args.host / "full/assessment.json").read_text())
    assert assessment["_validation"]["status"] == "verified_outputs_and_sources"

    # Test that the preserved copies are identical to both their source records
    # and the original reviewed drivers, without changing any original file.
    drivers = json.loads((HERE / "driver_manifest.json").read_text())
    for name, record in drivers.items():
        assert assess.sha(ROOT / record["source"]) == record["sha256"]
        assert assess.sha(HERE / "drivers" / name) == record["sha256"]
    diffs = []
    for name in ("bench_nano.py", "prepare_nano.py", "cadence.py"):
        executed = (args.host / "executed_sources" / name).read_text().splitlines(True)
        current = (HERE / name).read_text().splitlines(True)
        diffs.extend(difflib.unified_diff(executed, current, fromfile="executed/"+name, tofile="current/"+name))
    (HERE / "executed_to_current.diff").write_text("".join(diffs), encoding="utf-8")

    per_run = []
    input_manifest = []
    package_hashes = {}
    with tarfile.open(args.host / "full/evidence.tar.gz") as archive:
        for v in videos:
            ds, name = v["dataset"], v["name"]
            split, _ = assess.SPLITS[ds]
            gt = ROOT / "data/processed" / split / name / "gt/gt.txt"
            input_manifest.append(dict(dataset=ds, name=name, first=v["first"], last=v["last"],
                gt_source=str(gt.relative_to(ROOT)), gt_sha256=assess.sha(gt),
                images_manifest_sha256=assess.sha(args.host / "full" / (ds+"_"+name+"_images.json"))))
            for mode in assess.modes_for(v, videos.index(v)):
                row = samples[(ds, name, mode)]
                prefix = "raw/"+ds+"/"+name+"/"+mode+"/"
                frames = list(csv.DictReader(io.StringIO(archive.extractfile(prefix+"pipeline/"+name+".csv").read().decode())))
                inferred = [r for r in frames[25:] if r.get("infer_skipped", "False") != "True"]
                meta = json.loads(archive.extractfile(prefix+"run_meta.json").read())
                version = "A" if mode == "v21combo" else "B"
                package_hashes[version] = meta["provenance"]["file_ma_sha256_lf"]
                if mode in ("full", "v21combo"):
                    q = assessment["v21combo"]["quality_full_A_candidate_B"][ds]["A" if mode == "full" else "B"]["videos"][name]
                else:
                    q = assessment[mode]["quality_full_A_candidate_B"][ds]["B"]["videos"][name]
                base = samples[(ds, name, "full")]
                deployed = samples[(ds, name, "v21combo")]
                per_run.append(dict(dataset=ds, name=name, mode=mode, frames=150,
                    fps_steady=row["fps_steady"], p90_latency_ms=row["p90_latency_ms"],
                    fps_gain_vs_v21combo_pct=100*(row["fps_steady"]/deployed["fps_steady"]-1),
                    fps_gain_vs_full_pct=100*(row["fps_steady"]/base["fps_steady"]-1),
                    n_inferred=row["n_inferred"],
                    scheduler_total_ms=(row["adaptive"] or {}).get("cpu_ms", 0),
                    scheduler_mean_ms_per_frame=(row["adaptive"] or {}).get("cpu_ms", 0)/150,
                    infer_called_median_ms=median(float(r["infer_ms"]) for r in inferred),
                    gpu_exec_called_median_ms=median(float(r["exec_ms"]) for r in inferred),
                    decode_median_ms=row["decode_ms"],
                    post_all_median_ms=median(float(r["post_ms"]) for r in frames[25:]),
                    max_temp_c=row["max_temp_c"], regions_sha256=meta["hello_con"]["regions_sha256"], **q))
    write_csv(HERE / "per_run_verified.csv", per_run)

    # Preserve per-ID first-match data as well as the R103 aggregate. IDs here
    # are the evaluator's normalized indices, not the dataset's original IDs.
    delays = []
    for mode in ("v21combo", "stride2", "motion", "guarded"):
        for ds in assess.SPLITS:
            relevant = [v for v in videos if v["dataset"] == ds and (ds, v["name"], mode) in samples]
            if not relevant:
                continue
            split, evaluator = assess.SPLITS[ds]
            for v in relevant:
                runs = {}
                for version in "AB":
                    _, cfg, _ = assess.evaluation.rte.build_configs(split, version, None)
                    cfg.update(GT_FOLDER=str(args.stage / mode / "gt"),
                               TRACKERS_FOLDER=str(args.stage / mode / "trackers"), SEQ_INFO={v["name"]: 150})
                    dataset = assess.evaluation.rte.dataset_class(evaluator)(cfg)
                    data = dataset.get_preprocessed_seq_data(dataset.get_raw_seq_data(version, v["name"]), "pedestrian")
                    runs[version] = assess.previous.first_matches(data)
                first_a, match_a = runs["A"]
                first_b, match_b = runs["B"]
                assert (first_a == first_b).all()
                for index in range(len(first_a)):
                    if first_a[index] <= 0:
                        continue
                    a, b = int(match_a[index]), int(match_b[index])
                    delays.append(dict(dataset=ds, name=v["name"], mode=mode, normalized_gt_index=index,
                        first_seen_frame=int(first_a[index])+1,
                        full_first_match_frame=a+1 if a >= 0 else "", candidate_first_match_frame=b+1 if b >= 0 else "",
                        added_ms=(b-a)*1000/assess.previous.FPS[ds] if min(a, b) >= 0 else ""))
            group = [r for r in delays if r["dataset"] == ds and r["mode"] == mode]
            common = [r["added_ms"] for r in group if r["added_ms"] != ""]
            summary = assessment[mode]["first_match_delay_full_A_candidate_B"][ds]
            assert len(common) == summary["matched_by_both"]
            assert median(common) == summary["median_added_ms"]
    write_csv(HERE / "first_match_verified.csv", delays)

    regression_path = HERE.parent / "review102_ridesafe_pf_nano/pipeline_pairs.json"
    regression = {"available": regression_path.is_file()}
    if regression["available"]:
        old = json.loads(regression_path.read_text())
        matches = []
        for r in old:
            mode = "v21combo" if r["version"] == "A" else "full"
            current = samples[(r["dataset"], r["name"], mode)]
            matches.append(current["mot_sha256"] == r["mot_sha256"])
        assert len(matches) == 16 and all(matches)
        regression.update(r102_mot_hash_equal=sum(matches), compared=len(matches), source_sha256=assess.sha(regression_path))
    evaluator_sources = [Path(assess.previous.__file__), Path(assess.evaluation.__file__), Path(assess.evaluation.rte.__file__)]
    evaluator_sources += [Path(inspect.getfile(assess.evaluation.rte.dataset_class(e))) for _, e in assess.SPLITS.values()]
    output = dict(smoke_runs=len(smoke), full_runs=len(samples), frame_rows_checked=(len(smoke)+len(samples))*150,
        timing_recalculated_runs=38, bdd_fallback_equal=2, host_exit_record=host_exit,
        provenance=assessment["_provenance"], r102_regression=regression,
        per_version_six_source_hashes_lf=package_hashes,
        gt_and_images=input_manifest, evaluator_sha256={str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): assess.sha(p) for p in evaluator_sources},
        artifact_sha256={name: assess.sha(HERE / name) for name in ("per_run_verified.csv", "first_match_verified.csv", "host_evidence/full/assessment.json")},
        files={p.name: assess.sha(p) for p in HERE.glob("*.py")},
        limitations=["Tier 1 implementation audit, not the required separate Codex review.",
                     "Host exit/process status is preserved bridge evidence; this audit does not open SSH.",
                     "Early deployment did not record complete baseline/experimental package hashes.",
                     "JPEGs are represented by recorded per-file hashes, not retained in the archive.",
                     "No additional NMS/decode measurement is present in the completed bridge evidence."])
    (HERE / "verification.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("smoke_runs", "full_runs", "frame_rows_checked", "timing_recalculated_runs", "bdd_fallback_equal", "r102_regression")}))


if __name__ == "__main__":
    main()
