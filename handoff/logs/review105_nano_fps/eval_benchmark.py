"""Validate and score R105 evidence. No Nano speed is inferred from fixtures."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
from statistics import median
import sys
import tarfile

import numpy as np
import trackeval

HERE = Path(__file__).resolve().parent
import importlib.util
_spec = importlib.util.spec_from_file_location("r103_evaluation", HERE.parent / "review103_nano_optimization/eval_benchmark.py")
previous = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(previous)
from bench_nano import EXPECTED_ENGINE, EXPECTED_ONNX, frame_checks, modes_for, sha

evaluation = previous.evaluation
SPLITS = previous.SPLITS.copy()


def verify_timing(frames, summary):
    """Recalculate speed/latency from timestamps, rather than trusting metadata."""
    assert len(frames) == 150 and summary["warmup_bo"] == 25
    assert summary["n_frame_tong_ket"] == 125
    done = [float(r["t_xong_cha"]) for r in frames]
    assert all(math.isfinite(t) for t in done)
    assert all(b > a for a, b in zip(done, done[1:]))
    fps = 125 / (done[-1] - done[24])
    assert math.isclose(fps, summary["fps_on_dinh"], rel_tol=1e-10)
    for r in frames:
        delay = (float(r["t_xong_cha"]) - float(r["t_bat_dau_con"])) * 1000
        assert math.isfinite(delay) and delay >= 0
        assert math.isclose(delay, float(r["tre_ms"]), rel_tol=1e-9, abs_tol=1e-7)
        if r.get("infer_skipped", "False") == "True":
            assert float(r["infer_ms"]) == 0 and int(r["n_det_raw"]) == 0
        else:
            assert float(r["infer_ms"]) > 0
    ordered = sorted(float(r["tre_ms"]) for r in frames[25:])
    # Match the deployed nearest-order-statistic definition, without interpolation.
    p90 = ordered[round(.9 * (len(ordered) - 1))]
    assert math.isclose(p90, summary["tre_ms"]["p90"], rel_tol=1e-10)
    for key in ("decode_ms", "infer_ms"):
        assert math.isclose(median(float(r[key]) for r in frames[25:]),
                            summary[key]["trung_vi"], rel_tol=1e-10, abs_tol=1e-9)
    return {"fps_recalculated": fps, "p90_recalculated_ms": p90}


def verify_run_config(meta, command, row, inputs):
    ds, mode = row["dataset"], row["mode"]
    version = "A" if mode == "v21combo" else "B"
    cfg, hello = meta["cau_hinh"], meta["hello_con"]
    fps = 25 if ds == "ua" else 5
    buffer = 30 if ds == "ua" else 6
    assert row["version"] == version and row["frames"] == 150
    assert meta["ma_thoat"] == 0 and meta["hien_thi"]["bat"] is False
    assert not any(meta["che_do"][k] for k in ("gia_lap", "render", "hien_thi", "hien_thi_an"))
    assert meta["con"]["returncode"] == 0 and meta["con"]["bi_kill_vi_treo"] is False
    assert meta["provenance"]["tracker_backend"] == "np"
    assert cfg["tracker_cfg"] == dict(tracker_type="bytetrack", track_high_thresh=.25,
        track_low_thresh=.1, new_track_thresh=.25, track_buffer=buffer,
        match_thresh=.8, fuse_score=True)
    assert cfg["conf_min"] == .25 and cfg["det_conf"] == .05
    assert cfg["det_iou"] == .7 and cfg["max_det"] == 300 and cfg["det_max_det"] == 3000
    assert cfg["fps"] == hello["fps"] == fps and cfg["hang_doi"] == hello["hang_doi"] == 4
    assert cfg["classes"] == inputs["classes"][version][ds]
    assert cfg["class_map"] == (None if version == "A" else {"0": 2, "1": 7, "2": 5, "3": 3, "4": 1})
    assert cfg["elapsed_mode"] == ("skip" if ds == "ua" else "steps")
    assert cfg["track_buffer_frames"] == buffer and cfg["track_buffer_sec"] == 1.2
    assert not cfg["allow_truncated"] and not cfg["cho_phep_frame_hong"]
    assert cfg["gia_lap_cham_tracker_ms"] == 0
    assert hello["engine_sha256"] == EXPECTED_ENGINE[version]
    assert hello["onnx_sha256"] == EXPECTED_ONNX[version]
    assert hello["detector"] == "tensorrt" and hello["letterbox_hw"] == [384, 640]
    assert hello["nguon"]["loai"] == "image_dir"
    assert hello["nguon"]["duong_dan"].startswith("/dev/shm/r105_")
    assert (hello["regions_sha256"] is not None) == (ds == "ua")
    assert command[1] == "src/nano_live.py"
    for flag, value in (("--tracker", "cv"), ("--warmup-frames", "25"),
                        ("--hang-doi", "4"), ("--fps", str(fps)),
                        ("--source", hello["nguon"]["duong_dan"]),
                        ("--engine", hello["engine"]), ("--onnx-sha256", EXPECTED_ONNX[version])):
        assert command[command.index(flag)+1] == value
    assert "--show" not in command
    adaptive = row["adaptive"]
    if version == "B":
        enabled = mode in ("motion", "guarded") and ds == "ua"
        assert adaptive["enabled"] is enabled
        if enabled:
            assert adaptive["mode"] == mode
            assert sum(adaptive["counts"].values()) == 150
            assert adaptive["counts"]["skipped"] == len(row["skipped_frames"])
    return {"regions_sha256": hello["regions_sha256"], "tracker_cfg": cfg["tracker_cfg"]}


def quality(stage, relevant):
    result = {}
    for ds, (split, evaluator) in SPLITS.items():
        names = [v["name"] for v in relevant if v["dataset"] == ds]
        if not names:
            continue
        result[ds] = {}
        for version in "AB":
            _, cfg, _ = evaluation.rte.build_configs(split, version, None)
            cfg.update(GT_FOLDER=str(stage / "gt"), TRACKERS_FOLDER=str(stage / "trackers"),
                       SEQ_INFO=dict.fromkeys(names, 150))
            dataset = evaluation.rte.dataset_class(evaluator)(cfg)
            metrics = (trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
                       trackeval.metrics.Identity({"PRINT_CONFIG": False}))
            raw = {}
            for name in names:
                data = dataset.get_preprocessed_seq_data(dataset.get_raw_seq_data(version, name), "pedestrian")
                assert data["num_timesteps"] == 150
                raw[name] = {m.get_name(): m.eval_sequence(data) for m in metrics}
            combined = {m.get_name(): m.combine_sequences({n: x[m.get_name()] for n, x in raw.items()})
                        for m in metrics}
            result[ds][version] = {"combined": evaluation.scalars(combined),
                                  "videos": {n: evaluation.scalars(x) for n, x in raw.items()}}
    return result


def paired_speed(samples, mode, reference):
    output = {}
    for ds in SPLITS:
        group = [r for r in samples.values() if r["dataset"] == ds and r["mode"] == mode]
        if not group:
            continue
        base = [samples[(ds, r["name"], reference)] for r in group]
        output[ds] = {"clips": len(group), "reference": reference,
                      "paired_fps_gain_pct": median(100 * (b["fps_steady"] / a["fps_steady"] - 1)
                                                    for a, b in zip(base, group)),
                      "per_clip_fps_gain_pct": {b["name"]: 100 * (b["fps_steady"] / a["fps_steady"] - 1)
                                                for a, b in zip(base, group)}}
        for key in ("fps_steady", "p90_latency_ms", "n_inferred", "decode_ms", "infer_ms"):
            output[ds][key] = [median(r[key] for r in base), median(r[key] for r in group)]
        for key in ("mot_sha256", "cls_sha256"):
            output[ds][key + "_equal"] = sum(a[key] == b[key] for a, b in zip(base, group))
    return output


LEGACY_RUNNER_SHA = "94be03b3e12ca8d3da1f9f16db0c136a2a6ee16b92ff333554155ae271f96ef1"


def verify_legacy_snapshot(logs, inputs, complete, require_sources=True):
    # Explicit older deployment schema: raw JSON remains unchanged.
    assert inputs["code_sha256"]["bench_nano.py"] == LEGACY_RUNNER_SHA
    assert inputs["code_sha256"] == complete["code_sha256"]
    snapshot = logs.parent / "executed_sources"
    missing = []
    for name, expected in inputs["code_sha256"].items():
        if not (snapshot / name).is_file():
            missing.append(str(snapshot / name))
        else:
            assert sha(snapshot / name) == expected
    drivers = json.loads((HERE / "driver_manifest.json").read_text())
    for name, record in drivers.items():
        assert inputs["code_sha256"][name] == record["sha256"]
    for name, record in inputs["package_sources"].items():
        source = snapshot / name if name in drivers else snapshot / "package" / name
        if not source.is_file():
            missing.append(str(source))
        else:
            assert sha(source) == record["sha256"]
    assert inputs["package_sources"]["r105_cadence.py"]["sha256"] == inputs["code_sha256"]["cadence.py"]
    smoke_root = logs.parent / "smoke"
    smoke = json.loads((smoke_root / "complete.json").read_text())
    smoke_inputs = json.loads((smoke_root / "inputs.json").read_text())
    assert smoke["runs"] == 8 and smoke["engine_sha256"] == complete["engine_sha256"]
    assert smoke["code_sha256"] == inputs["code_sha256"]
    assert smoke_inputs["package_sources"] == inputs["package_sources"]
    assert sha(smoke_root / "evidence.tar.gz") == smoke["archive_sha256"]
    if missing and require_sources:
        raise FileNotFoundError("Missing execution sources: " + ", ".join(missing))
    unique_sources = set(inputs["code_sha256"]) | {name if name in drivers else "package/"+name
                                                   for name in inputs["package_sources"]}
    return {"missing_sources": sorted(set(missing)), "schema": "early_r105_bridge", "runner_sha256": LEGACY_RUNNER_SHA,
            "verified_source_files": len(unique_sources) - len(set(missing)),
            "verified_source_references": len(inputs["code_sha256"]) + len(inputs["package_sources"]),
            "limitation": "No full baseline-package hash was recorded by the early runner."}


def load_evidence(logs, phase="full", outputs_only=False):
    assert phase in ("smoke", "full")
    wanted = 30 if phase == "full" else 8
    complete = json.loads((logs / "complete.json").read_text())
    assert complete["runs"] == wanted and sha(logs / "evidence.tar.gz") == complete["archive_sha256"]
    inputs = json.loads((logs / "inputs.json").read_text())
    assert inputs["engine_sha256"] == complete["engine_sha256"] == EXPECTED_ENGINE
    assert inputs["onnx_sha256"] == EXPECTED_ONNX
    assert inputs["code_sha256"] == complete["code_sha256"]
    if "package_sha256" in inputs:
        assert inputs["package_sha256"] == complete["package_sha256"]
        assert inputs["baseline_package_sha256"] == complete["baseline_package_sha256"]
        assert inputs["driver_manifest_sha256"] == complete["driver_manifest_sha256"]
    else:
        verify_legacy_snapshot(logs, inputs, complete, require_sources=not outputs_only)
    after = json.loads((logs / "after.json").read_text())
    assert after["boot"] == inputs["boot"] and after["errors"] == inputs["errors"] == "0"
    rows = json.loads((logs / "pairs.json").read_text())
    videos = json.loads((logs / "videos.json").read_text())
    assert [r["dataset"] for r in videos] == (["bdd"]*4 + ["ua"]*4 if phase == "full" else ["bdd", "ua"])
    samples = {(r["dataset"], r["name"], r["mode"]): r for r in rows}
    expected = {(v["dataset"], v["name"], m) for i, v in enumerate(videos) for m in modes_for(v, i)}
    assert len(samples) == len(rows) == len(expected) == wanted and set(samples) == expected
    contents = {}
    clip_configs = {}
    package_configs = {}
    with tarfile.open(logs / "evidence.tar.gz") as archive:
        names = archive.getnames()
        assert len(names) == len(set(names))
        for ds, name, mode in expected:
            prefix = ds + "/" + name + "/" + mode
            row = samples[(ds, name, mode)]
            for suffix, key in ((".txt", "mot_sha256"), ("_cls.csv", "cls_sha256")):
                data = archive.extractfile(prefix + suffix).read()
                assert previous.sha(data) == row[key]
                contents[prefix+suffix] = data
            frames = archive.extractfile("raw/"+prefix+"/pipeline/"+name+".csv").read()
            assert previous.sha(frames) == row["pipeline_sha256"]
            frame_rows = list(csv.DictReader(io.StringIO(frames.decode())))
            skipped = frame_checks(frame_rows, mode, ds)
            assert skipped == row["skipped_frames"] and row["n_inferred"] == 150-len(skipped)
            metadata = archive.extractfile("raw/"+prefix+"/run_meta.json").read()
            assert previous.sha(metadata) == row["meta_sha256"]
            meta = json.loads(metadata)
            assert meta["hoan_tat"] and meta["dem"]["n_frame_da_xu_ly"] == 150
            assert meta["dem"]["n_unreadable"] == meta["dem"]["n_truncated"] == 0
            assert meta["hello_con"]["n_frames_khai_bao"] == 150
            assert meta["eos_con"].get("adaptive") == row["adaptive"]
            timing = meta["thoi_gian"]["tom_tat_sau_warmup"]
            verify_timing(frame_rows, timing)
            command = json.loads(archive.extractfile("raw/"+prefix+"/command.json").read())
            invariants = verify_run_config(meta, command, row, inputs)
            assert clip_configs.setdefault((ds, name), invariants) == invariants
            version = "A" if mode == "v21combo" else "B"
            source_hashes = meta["provenance"]["file_ma_sha256_lf"]
            assert set(source_hashes) == {"nano_live.py", "nano_detect_stream.py", "nano_stream_proto.py",
                                         "nano_detector.py", "run_tracking.py", "det_cache.py"}
            assert package_configs.setdefault(version, source_hashes) == source_hashes
            if version == "B":
                for filename in ("nano_live.py", "nano_detect_stream.py", "run_tracking.py"):
                    snapshot = logs.parent / "executed_sources/package" / filename
                    if snapshot.is_file():
                        assert previous.sha(snapshot.read_bytes().replace(b"\r\n", b"\n")) == source_hashes[filename]
            assert timing["fps_on_dinh"] == row["fps_steady"] > 0
            assert timing["tre_ms"]["p90"] == row["p90_latency_ms"]
            assert row["max_temp_c"] < 75
            assert row["engine_sha256"] == EXPECTED_ENGINE["A" if mode == "v21combo" else "B"]
            assert row["source_sha256"] == sha(logs / (ds+"_"+name+"_images.json"))
            images = json.loads((logs / (ds+"_"+name+"_images.json")).read_text())
            video = next(v for v in videos if v["dataset"] == ds and v["name"] == name)
            assert len(images) == video["n_frames"] == video["last"]-video["first"]+1 == 150
            assert sorted(images)[0] == video["first_file"] and sorted(images)[-1] == video["last_file"]
            assert all(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest) for digest in images.values())
        for row in rows:
            if row["dataset"] == "bdd" and row["mode"] == "guarded":
                base = samples[("bdd", row["name"], "full")]
                assert row["adaptive"]["enabled"] is False
                assert all(base[k] == row[k] for k in ("mot_sha256", "cls_sha256"))
    return videos, samples, contents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--outputs-only", action="store_true", help="Write provisional scores and exit 2 if execution sources are missing; all raw-output gates still apply")
    args = parser.parse_args()
    videos, samples, contents = load_evidence(args.logs, outputs_only=args.outputs_only)
    args.stage.mkdir(exist_ok=False, parents=True)
    result = {"full": {"speed_vs_v21combo": paired_speed(samples, "full", "v21combo")}}
    for mode in ("v21combo", "stride2", "motion", "guarded"):
        stage = args.stage / mode
        relevant, _ = previous.stage(stage, videos, samples, contents, mode)
        evaluation.SPLITS = {ds: s for ds, s in SPLITS.items() if any(v["dataset"] == ds for v in relevant)}
        result[mode] = {"speed_vs_v21combo": paired_speed(samples, mode, "v21combo"),
                        "speed_vs_full": paired_speed(samples, mode, "full"),
                        "quality_full_A_candidate_B": quality(stage, relevant),
                        "first_match_delay_full_A_candidate_B": previous.start_delays(stage, relevant)}
    inputs = json.loads((args.logs / "inputs.json").read_text())
    if "package_sha256" not in inputs:
        provenance = verify_legacy_snapshot(args.logs, inputs,
                     json.loads((args.logs / "complete.json").read_text()), require_sources=False)
        result["_provenance"] = provenance
    pending = result.get("_provenance", {}).get("missing_sources", [])
    result["_validation"] = {"status": "blocked_missing_sources" if pending else "verified_outputs_and_sources",
                             "real_runs": 30, "source_files_pending": len(pending)}
    (args.logs / ("provisional_assessment.json" if pending else "assessment.json")).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if pending:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
