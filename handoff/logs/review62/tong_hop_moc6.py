"""Read-only R62 evidence aggregation on PC; writes only a new JSON report.

Usage: python tong_hop_moc6.py --index index.json --comparison-group paired300 --out NEW.json
Index is a list of {group,pair,tracker,variant,dir}; variant is baseline/candidate,
tracker is cv/ekf, dir is absolute or relative to INDEX's directory. Optional
expected_frames defaults to 300 (use 1470 for long runs). Include EVERY attempt;
put historical/diagnostic/incomplete attempts in their own groups if they are not
part of the prespecified paired comparison. Duplicate pair variants fail closed.

Each dir contains result.json, before.json, after.json, parity.json and live/
(run_meta.json, pipeline/MVI_39031.csv, data/, data_cls/, tegrastats.log).
Optional detections_array_sha256.json: {"detections_array_sha256":"<64 hex>"}.
Missing detection evidence remains explicit and blocks acceptance; this script
does not open NPZ files. The hash-generation method must be identical across runs.

Optional index throttling_review: {"no_throttling":true,"evidence":"log path"}.
The evidence path is relative to INDEX, is hashed, and is an EXTERNAL review,
not an inference from low schedutil clock samples. Missing review blocks overall
short-comparison acceptance; numeric/parity gates are also reported separately.

FPS uses ALL frames / processing seconds. p90 uses CSV after 25 warmup frames,
with nano_live's nearest-rank index round(.9*(n-1)), not linear interpolation.
Pair gains are 100*(candidate/baseline-1); distribution quartiles are linear.
At least four complete pairs per tracker are required; every included pair is
checked, and both CV/EKF medians must pass. No outlier deletion. Each pair p90
increase and its group median must be <=5%; FPS gain median must be >=5%.
This is a short-run comparison gate, NOT completion/deployment of milestone 6.
Exit 0: short gate passes; 2: evidence/gate incomplete or failed; 1: invalid input.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


OUTPUTS = ("data/MVI_39031.txt", "data_cls/MVI_39031.csv")
BOOT = "/proc/sys/kernel/random/boot_id"
STAGES = ("decode_ms", "pre_ms", "infer_ms", "post_ms", "tre_ms")
SHA = re.compile(r"^[0-9a-fA-F]{64}$")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def number(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (not positive or value > 0))


def close(a, b):
    return number(a) and number(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)


def valid_sha(value):
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def quantile(values, q):
    s = sorted(values)
    p = (len(s) - 1) * q
    lo, hi = int(math.floor(p)), int(math.ceil(p))
    return s[lo] + (s[hi] - s[lo]) * (p - lo)


def distribution(values):
    if not values:
        return None
    m = statistics.median(values)
    return {"n": len(values), "median": m, "min": min(values), "max": max(values),
            "p25": quantile(values, .25), "p75": quantile(values, .75),
            "mad": statistics.median([abs(x - m) for x in values])}


def native_stats(values):
    s = sorted(values)
    return {"n": len(s), "median": statistics.median(s),
            "p90": s[min(len(s) - 1, int(round(.9 * (len(s) - 1))))]}


def relative_path(value, base):
    p = Path(value)
    return (p if p.is_absolute() else base / p).resolve()


def read_json(path, issues, evidence, optional=False):
    if optional and not path.exists():
        return None
    try:
        evidence[str(path)] = sha(path)
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(obj, dict):
            raise ValueError("expected a JSON object")
        return obj
    except (OSError, ValueError) as exc:
        issues.append("%s: %s" % (path.name, exc))
        return None


def swap_kb(snap):
    text = snap.get("/proc/meminfo", "")
    found = dict(re.findall(r"^(SwapTotal|SwapFree):\s+(\d+)\s+kB", text, re.M))
    return int(found["SwapTotal"]) - int(found["SwapFree"]) if len(found) == 2 else None


def inspect_run(entry, base, run_id):
    out = relative_path(entry["dir"], base)
    issues, evidence, checks = [], {}, {}
    result = read_json(out / "result.json", issues, evidence) or {}
    meta = read_json(out / "live/run_meta.json", issues, evidence) or {}
    parity = read_json(out / "parity.json", issues, evidence) or {}
    before = read_json(out / "before.json", issues, evidence) or {}
    after = read_json(out / "after.json", issues, evidence) or {}
    detection = read_json(out / "detections_array_sha256.json", issues, evidence, optional=True)
    if detection is None:
        issues.append("missing detections_array_sha256.json (no NPZ inspection performed)")
    expected = entry.get("expected_frames", 300)
    counts, times = meta.get("dem", {}), meta.get("thoi_gian", {})
    summary = times.get("tom_tat_sau_warmup", {})
    mode, hello = meta.get("che_do", {}), meta.get("hello_con", {})
    show = meta.get("hien_thi") or {}
    resource = meta.get("thiet_bi") or {}
    telemetry = resource.get("tegrastats") or {}
    n, duration = counts.get("n_frame_da_xu_ly"), times.get("xu_ly_s")
    fps = n / duration if number(n) and number(duration, True) else None
    checks["completed_successfully"] = (meta.get("hoan_tat") is True and meta.get("ma_thoat") == 0
                                        and result.get("live_exit") == 0 and meta.get("con", {}).get("returncode") == 0)
    checks["tracker_matches_index"] = result.get("tracker") == entry["tracker"]
    checks["frame_counts_complete"] = (result.get("frames") == expected and
        all(counts.get(k) == expected for k in ("n_frames_khai_bao", "n_frame_nhan", "n_frame_da_xu_ly", "n_frame_vao_tracker_doc_duoc")))
    checks["no_processing_loss"] = all(counts.get(k) == 0 for k in
        ("n_unreadable", "n_truncated", "n_frame_bay_khi_dung_khong_xu_ly"))
    checks["real_tensorrt"] = mode.get("gia_lap") is False and mode.get("detector") == "tensorrt"
    checks["fps_recomputed_matches"] = close(fps, times.get("fps_xu_ly")) and close(fps, result.get("fps"))
    checks["warmup_25"] = summary.get("warmup_bo") == 25
    checks["boot_unchanged"] = isinstance(before.get(BOOT), str) and bool(before.get(BOOT)) and before.get(BOOT) == after.get(BOOT)
    checks["live_replay_parity_recorded"] = (result.get("parity_byte") is True and parity.get("n_video") == 1
        and parity.get("n_dat") == 1 and len(parity.get("ket_qua", [])) == 1
        and parity["ket_qua"][0].get("dat") is True)
    actual_hashes = {}
    for name in OUTPUTS:
        try:
            actual_hashes[name] = sha(out / "live" / name)
            evidence[str(out / "live" / name)] = actual_hashes[name]
        except OSError as exc:
            issues.append("%s: %s" % (name, exc))
    checks["live_hashes_match_helper"] = (len(actual_hashes) == len(OUTPUTS) and
        all(actual_hashes[k] == result.get("hashes", {}).get(k) for k in OUTPUTS))
    det_hash = (detection or {}).get("detections_array_sha256")
    checks["detection_hash_present"] = valid_sha(det_hash)
    raw_stats, csv_count = {}, None
    try:
        path = out / "live/pipeline/MVI_39031.csv"
        evidence[str(path)] = sha(path)
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        csv_count = len(rows)
        checks["csv_frames_1_to_N"] = [int(r["frame"]) for r in rows] == list(range(1, expected + 1))
        checks["csv_all_readable"] = all(r["unreadable"].lower() in ("false", "0") for r in rows)
        usable = [r for r in rows if r["unreadable"].lower() in ("false", "0") and r.get("t_xong_cha")]
        for stage in STAGES:
            vals = [float(r[stage]) for r in usable[25:]]
            if not vals or not all(number(x) and x >= 0 for x in vals):
                raise ValueError("invalid/missing post-warmup " + stage)
            raw_stats[stage] = native_stats(vals)
        checks["csv_latency_matches_metadata"] = (raw_stats["tre_ms"]["n"] == expected - 25
            and close(raw_stats["tre_ms"]["median"], summary.get("tre_ms", {}).get("trung_vi"))
            and close(raw_stats["tre_ms"]["p90"], summary.get("tre_ms", {}).get("p90")))
        checks["csv_latency_matches_timestamps"] = all(close(float(r["tre_ms"]),
            (float(r["t_xong_cha"]) - float(r["t_bat_dau_con"])) * 1000.0) for r in usable)
        window = times.get("moc_monotonic", {})
        start, end = window.get("gui_bat_dau"), window.get("ket_thuc_xu_ly")
        checks["csv_inside_measurement_window"] = (number(start) and number(end) and
            all(start <= float(r["t_bat_dau_con"]) <= float(r["t_xong_cha"]) <= end for r in usable))
    except (OSError, ValueError, KeyError) as exc:
        issues.append("pipeline CSV: %s" % exc)
        checks["csv_valid"] = False
    if not raw_stats:
        checks["csv_valid"] = False
    display = result.get("display")
    checks["display_metadata_consistent"] = (display in ("on", "off") and
        (mode.get("hien_thi") is (display == "on")) and mode.get("hien_thi_an") is False)
    drops = shown = None
    if display == "on":
        dem = show.get("dem", {})
        drops, shown = dem.get("n_bo_hien_thi"), dem.get("n_hien_thi")
        checks["display_frames_accounted"] = (show.get("bat") is True and show.get("an") is False
            and number(drops) and number(shown) and drops >= 0 and shown + drops == expected
            and all(dem.get(k) == 0 for k in ("n_khong_ghep", "n_anh_bo_khong_ket_qua", "n_khong_hien_thi_cua_so_dong"))
            and not show.get("loi") and show.get("luong_con_song_sau_dong") is False)
    checks["telemetry_present"] = (telemetry.get("co_tegrastats") is True and number(telemetry.get("n_mau"), True)
        and number(telemetry.get("ram_dung_MB_max")) and number(telemetry.get("swap_dung_MB_max"))
        and bool(telemetry.get("nhiet_max_C")))
    try:
        path = out / "live/tegrastats.log"
        evidence[str(path)] = sha(path)
        checks["telemetry_raw_present"] = path.stat().st_size > 0
    except OSError as exc:
        issues.append("tegrastats.log: %s" % exc)
        checks["telemetry_raw_present"] = False
    thermal = entry.get("throttling_review") or {}
    thermal_review = {"source": "external review, not inferred from DVFS frequency", "no_throttling": None}
    if thermal.get("evidence"):
        try:
            p = relative_path(thermal["evidence"], base)
            thermal_review.update(evidence=str(p), evidence_sha256=sha(p), no_throttling=thermal.get("no_throttling"))
        except OSError as exc:
            issues.append("throttling evidence: %s" % exc)
    checks["no_throttling_reviewed"] = thermal_review["no_throttling"] is True
    frame_start = swap_kb(before)
    frame_end = swap_kb(after)
    run = {"id": run_id, "group": entry["group"], "pair": entry["pair"], "tracker": entry["tracker"],
           "variant": entry["variant"], "dir": str(out), "expected_frames": expected,
           "complete": checks["completed_successfully"] and checks["frame_counts_complete"],
           "state": meta.get("trang_thai", "missing_metadata"), "display": display,
           "frames_processed": n, "csv_frames": csv_count, "fps": fps,
           "fps_display": times.get("fps_hien_thi"), "startup_s": times.get("khoi_tao_s"),
           "processing_s": duration, "total_s": times.get("tong_s"),
           "latency_ms": raw_stats.get("tre_ms"), "stages_ms": raw_stats,
           "preview_drops": drops, "preview_shown": shown, "hashes": actual_hashes,
           "detections_array_sha256": det_hash,
           "detection_digest_method": (detection or {}).get("method"),
           "resources": {"ram_max_MB": telemetry.get("ram_dung_MB_max"),
                         "swap_max_MB": telemetry.get("swap_dung_MB_max"),
                         "swap_before_kB": frame_start, "swap_after_kB": frame_end,
                         "swap_delta_kB": frame_end - frame_start if frame_end is not None and frame_start is not None else None,
                         "temperature_max_C": telemetry.get("nhiet_max_C"),
                         "cpu_MHz_min_max": telemetry.get("cpu_MHz_min_max"),
                         "gpu_load_mean_max": telemetry.get("gr3d_pct_tb_max"),
                         "clocks_before": resource.get("truoc", {}).get("xung"),
                         "clocks_after": resource.get("sau", {}).get("xung"),
                         "source": "run_meta.thiet_bi.tegrastats; before/after meminfo"},
           "throttling_review": thermal_review, "checks": checks, "issues": issues,
           "failed_checks": [k for k, v in checks.items() if v is not True], "evidence_sha256": evidence,
           "comparability": {"config": meta.get("cau_hinh"),
               "detector": {k: hello.get(k) for k in ("engine_sha256", "onnx_sha256", "versions", "bo_dem",
                    "bindings", "imgsz", "conf", "iou", "max_det", "fps", "hang_doi", "lam_nong", "regions_sha256", "n_regions", "orig_shape")},
               "code": meta.get("provenance", {}).get("file_ma_sha256_lf", {}),
               "tracker_backend": meta.get("provenance", {}).get("tracker_backend"),
               "power_mode": resource.get("truoc", {}).get("nvpmodel"),
               "cpu_governor": resource.get("truoc", {}).get("cpu_governor"),
               "boot_id": before.get(BOOT)}}
    return run


def inspect_pair(key, members):
    group, tracker, pair = key
    out = {"group": group, "tracker": tracker, "pair": pair, "run_ids": [r["id"] for r in members],
           "fps_gain_percent": None, "p90_increase_percent": None, "checks": {}}
    variants = collections.defaultdict(list)
    for r in members:
        variants[r["variant"]].append(r)
    c = out["checks"]
    c["exactly_one_each_variant"] = len(variants["baseline"]) == len(variants["candidate"]) == 1
    if not c["exactly_one_each_variant"]:
        out["issues"] = ["missing or duplicate baseline/candidate; all attempts retained"]
        return out
    b, a = variants["baseline"][0], variants["candidate"][0]
    c["run_evidence_valid_except_thermal_review"] = all(
        value is True for r in (b, a) for k, value in r["checks"].items() if k != "no_throttling_reviewed")
    c["both_real_display_on"] = b["display"] == a["display"] == "on"
    c["both_300_frames"] = b["expected_frames"] == a["expected_frames"] == 300
    bc, ac = b["comparability"], a["comparability"]
    c["same_config"] = bool(bc["config"]) and bc["config"] == ac["config"]
    c["same_engine_and_detector_settings"] = (valid_sha(bc["detector"].get("engine_sha256"))
        and bc["detector"] == ac["detector"])
    c["same_power_governor_backend"] = all(bc[k] is not None and bc[k] == ac[k] for k in
        ("power_mode", "cpu_governor", "tracker_backend"))
    c["same_boot_within_pair"] = bool(bc["boot_id"]) and bc["boot_id"] == ac["boot_id"]
    cb, ca = bc["code"], ac["code"]
    c["only_detector_code_differs"] = (bool(cb) and cb.keys() == ca.keys()
        and valid_sha(cb.get("nano_detector.py")) and valid_sha(ca.get("nano_detector.py"))
        and cb["nano_detector.py"] != ca["nano_detector.py"]
        and all(cb[k] == ca[k] for k in cb if k != "nano_detector.py"))
    c["mot_and_class_bytes_identical"] = bool(b["hashes"]) and b["hashes"] == a["hashes"]
    c["detections_identical"] = (valid_sha(b["detections_array_sha256"])
        and b["detections_array_sha256"] == a["detections_array_sha256"]
        and b["detection_digest_method"] == a["detection_digest_method"])
    c["no_extra_preview_drops"] = number(b["preview_drops"]) and number(a["preview_drops"]) and a["preview_drops"] <= b["preview_drops"]
    bs, cs = b["resources"]["swap_max_MB"], a["resources"]["swap_max_MB"]
    c["no_extra_peak_swap"] = number(bs) and number(cs) and cs <= bs
    c["no_throttling_reviewed"] = b["checks"]["no_throttling_reviewed"] and a["checks"]["no_throttling_reviewed"]
    if number(b["fps"], True) and number(a["fps"], True):
        out["fps_gain_percent"] = 100.0 * (a["fps"] / b["fps"] - 1.0)
    bl, cl = (b["latency_ms"] or {}).get("p90"), (a["latency_ms"] or {}).get("p90")
    if number(bl, True) and number(cl, True):
        out["p90_increase_percent"] = 100.0 * (cl / bl - 1.0)
    c["p90_increase_le_5_percent"] = number(out["p90_increase_percent"]) and out["p90_increase_percent"] <= 5.0
    out["baseline"] = {k: b[k] for k in ("id", "fps", "latency_ms", "preview_drops", "resources")}
    out["candidate"] = {k: a[k] for k in ("id", "fps", "latency_ms", "preview_drops", "resources")}
    out["failed_checks"] = [k for k, value in c.items() if value is not True]
    return out


def aggregate(entries, base, comparison_group):
    runs = [inspect_run(e, base, i + 1) for i, e in enumerate(entries)]
    members = collections.defaultdict(list)
    for r in runs:
        members[(r["group"], r["tracker"], str(r["pair"]))].append(r)
    pairs = [inspect_pair(k, v) for k, v in sorted(members.items())]
    selected_runs = [r for r in runs if r["group"] == comparison_group]
    selected_pairs = [p for p in pairs if p["group"] == comparison_group]
    paths = [r["dir"] for r in runs]
    duplicate_dirs = sorted(p for p, n in collections.Counter(paths).items() if n > 1)
    groups = {}
    for tracker in ("cv", "ekf"):
        ps = [p for p in selected_pairs if p["tracker"] == tracker]
        gains = [p["fps_gain_percent"] for p in ps if number(p["fps_gain_percent"])]
        latencies = [p["p90_increase_percent"] for p in ps if number(p["p90_increase_percent"])]
        gain_stat, latency_stat = distribution(gains), distribution(latencies)
        checks = {"at_least_4_pairs": len(ps) >= 4,
                  "all_pairs_have_numbers": len(ps) == len(gains) == len(latencies) and bool(ps),
                  "median_fps_gain_ge_5_percent": bool(gain_stat) and gain_stat["median"] >= 5.0,
                  "median_p90_increase_le_5_percent": bool(latency_stat) and latency_stat["median"] <= 5.0,
                  "every_pair_p90_increase_le_5_percent": bool(latencies) and all(x <= 5.0 for x in latencies),
                  "all_evidence_and_parity_checks": bool(ps) and all(
                      v is True for p in ps for k, v in p["checks"].items() if k != "no_throttling_reviewed"),
                  "no_throttling_reviewed": bool(ps) and all(p["checks"].get("no_throttling_reviewed") is True for p in ps)}
        groups[tracker] = {"n_pairs": len(ps), "fps_gain_percent": gain_stat,
                           "p90_increase_percent": latency_stat, "checks": checks,
                           "pass": all(v is True for v in checks.values())}
    numerical = bool(selected_runs) and not duplicate_dirs and all(
        v is True for g in groups.values() for k, v in g["checks"].items() if k != "no_throttling_reviewed")
    accepted = numerical and all(g["pass"] for g in groups.values())
    return {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "comparison_group": comparison_group, "runs": runs, "pairs": pairs, "tracker_summary": groups,
            "duplicate_run_directories": duplicate_dirs,
            "incomplete_run_ids": [r["id"] for r in runs if not r["complete"]],
            "acceptance": {"numeric_evidence_parity_gate": numerical, "short_comparison_pass": accepted,
                           "milestone_6_complete": False, "deployment_authorized_by_this_report": False},
            "limitations": ["All indexed attempts are reported; only comparison_group enters acceptance.",
                "Index must enumerate every planned run; missing pairs beyond indexed records cannot be discovered.",
                "Resource maxima come from run_meta telemetry; low DVFS clocks do not prove throttling.",
                "Detection digest is externally supplied; NPZ payloads are not opened here.",
                "Live MOT/class hashes are recomputed; live/replay byte equality is the helper's recorded assertion.",
                "Full 1470-frame validation and launcher/HDMI checks remain separate milestone gates."]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--comparison-group", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    entries = json.loads(a.index.read_text(encoding="utf-8-sig"))
    if not isinstance(entries, list) or not entries:
        ap.error("index must be a nonempty list")
    for e in entries:
        if not isinstance(e, dict) or any(k not in e for k in ("group", "pair", "tracker", "variant", "dir")):
            ap.error("each entry needs group,pair,tracker,variant,dir")
        if e["variant"] not in ("baseline", "candidate") or e["tracker"] not in ("cv", "ekf"):
            ap.error("variant=baseline/candidate; tracker=cv/ekf")
        if not isinstance(e["group"], str) or not isinstance(e["dir"], str) or not e["dir"]:
            ap.error("group and nonempty dir must be strings")
        if type(e.get("expected_frames", 300)) is not int or e.get("expected_frames", 300) <= 25:
            ap.error("expected_frames must be integer >25")
    report = aggregate(entries, a.index.resolve().parent, a.comparison_group)
    report["index"] = {"path": str(a.index.resolve()), "sha256": sha(a.index)}
    report["script_sha256"] = sha(Path(__file__))
    with a.out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(report["acceptance"], ensure_ascii=False))
    return 0 if report["acceptance"]["short_comparison_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
