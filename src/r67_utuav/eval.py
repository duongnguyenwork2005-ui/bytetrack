"""UTUAV Q3: immutable per-sequence cache/tracking units, smoke gate, full matrix."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/r64_ngoai"), str(ROOT / "src/r66_ngoai"), str(HERE)]
import boot_stats  # noqa: E402
import config  # noqa: E402
import danh_gia_ngoai as r64  # noqa: E402
import det_cache  # noqa: E402
import prepare as prep  # noqa: E402
import run_fingerprint as fp  # noqa: E402
import run_tracking as rt  # noqa: E402
import run_trackeval as rte  # noqa: E402
from mot_datasets.base import FrameRef  # noqa: E402
from sher_metric import combine  # noqa: E402

FPS = prep.FPS
BUFFER = round(1.2 * FPS)
CLASSES = [1, 2, 3, 5, 7]
CODE = ["src/r67_utuav/eval.py", "src/r67_utuav/prepare.py", "src/r64_ngoai/danh_gia_ngoai.py",
        "src/det_cache.py", "src/detector_classes.py", "src/run_tracking.py", "src/run_trackeval.py",
        "src/boot_stats.py", "src/r66_ngoai/sher_metric.py"]
FILES = ("data/{v}.txt", "data_cls/{v}.csv", "timing/{v}.csv")


def require(ok, msg):
    if not ok:
        raise r64.LoiHopDong(msg)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    prep.write(Path(path), obj)


def hashes(folder, names):
    return {name: prep.digest(folder / name) for name in names}


def check_hashes(folder, locked):
    require(hashes(folder, locked) == locked, f"artifact changed: {folder}")


def publish(source, target, expected):
    if target.exists():
        require(prep.digest(target) == expected, f"published artifact differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    shutil.copyfile(source, temp)
    require(prep.digest(temp) == expected, f"copy differs: {target}")
    temp.replace(target)


class Phase:
    def __init__(self, phase, deadline):
        self.phase = phase
        self.deadline = deadline
        self.split = prep.SMOKE if phase == "smoke" else prep.SPLIT
        self.log = prep.LOG / phase
        self.log.mkdir(parents=True, exist_ok=True)
        self.videos = list(prep.RANGES)
        manifest = read(prep.LOG / "source_manifest.json")
        self.source_sha = prep.digest(prep.LOG / "source_manifest.json")
        audit = read(prep.LOG / "source_xml_audit.json")
        require({k: (audit[k]["frames"], audit[k]["out_of_bounds"], audit[k]["max_excursion_px"])
                 for k in self.videos} == {"B_validation": (738, 3, 2), "C_validation": (751, 42, 4)}
                and all(audit[k][field] == 0 for k in self.videos for field in
                    ("negative_or_zero_size", "nonfinite", "duplicate_ids", "unknown_class")),
                "source XML audit differs from locked boundary finding")
        self.audit_sha = prep.digest(prep.LOG / "source_xml_audit.json")
        image_manifest = read(prep.LOG / "images_verified.json")
        require(manifest["source"] == image_manifest["source"], "source manifests differ")
        require(manifest["source"]["images"]["sha256_from_transfer_manifest"] == prep.IMAGE_SHA,
                "archive source provenance changed")
        self.inputs = {}
        for v, (scene, first, last, _) in prep.RANGES.items():
            n = 128 if phase == "smoke" else last - first + 1
            root = config.PROCESSED_DIR / self.split / v
            expected = manifest["sequences"][v][self.split]
            gt = root / "gt/gt.txt"
            require(prep.digest(gt) == expected["gt_sha256"] and expected["frames"] == n,
                    f"{v}: GT differs from converter")
            require(prep.digest(root / "seqinfo.ini") == expected["seqinfo_sha256"], f"{v}: seqinfo changed")
            arr = np.loadtxt(gt, delimiter=",", ndmin=2)
            r64.kiem_gt(arr, n, v)
            require(arr.shape[1] == 9 and set(np.unique(arr[:, 7])).issubset({1, 8})
                    and (arr[:, 2:4] >= -4).all() and (arr[:, 2] + arr[:, 4] <= 3844).all()
                    and (arr[:, 3] + arr[:, 5] <= 2164).all(), f"{v}: invalid GT")
            import configparser
            ini = configparser.ConfigParser()
            ini.read(root / "seqinfo.ini", encoding="utf-8")
            s = ini["Sequence"]
            require(s["name"] == v and s.getint("seqLength") == n
                    and abs(s.getfloat("frameRate") - FPS) < 1e-9
                    and s.getint("imWidth") == 3840 and s.getint("imHeight") == 2160
                    and Path(s["imDir"]).resolve() == (prep.IMAGES / v).resolve(), f"{v}: invalid seqinfo")
            source_rows = [image_manifest["files"][f"{scene}:{i}"] for i in range(first, first+n)]
            for i, item in enumerate(source_rows, 1):
                path = prep.IMAGES / v / f"{i:04d}.jpg"
                require(path.is_file() and path.stat().st_size == item["bytes"] and
                        prep.digest(path) == item["sha256"], f"image differs from archive {path}")
            self.inputs[v] = {"n": n, "source_range": [first, first+n-1],
                "gt_sha256": expected["gt_sha256"], "seqinfo_sha256": expected["seqinfo_sha256"],
                "images_sha256": fp.sha256_obj([row["sha256"] for row in source_rows])}
        self.code = hashes(ROOT, CODE)
        import inspect
        import trackeval
        from trackeval.datasets import MotChallenge2DBox
        self.metric_code = {c.__name__: prep.digest(inspect.getfile(c)) for c in
            (MotChallenge2DBox, trackeval.metrics.HOTA, trackeval.metrics.CLEAR,
             trackeval.metrics.Identity, trackeval.Evaluator)}
        self.backend = fp.code_fingerprint()
        require(self.backend["backend"]["backend"] == "ultralytics", "tracker backend differs")
        self.fingerprint = fp.sha256_obj({"protocol": "UTUAV-Q3-v1", "phase": phase,
            "fps_fraction": [30000, 1001], "stride": 1, "buffer": BUFFER,
            "detectors": list(r64.DETECTOR), "filters": list(r64.BO_LOC), "conditions": r64.DIEU_KIEN,
            "classes": CLASSES, "class_policy": prep.CLASSES, "inputs": self.inputs,
            "source_manifest": self.source_sha, "source_xml_audit": self.audit_sha, "code": self.code,
            "metric_code": self.metric_code, "backend": self.backend})
        record = {"split": self.split, "protocol_hash": self.fingerprint, "inputs": self.inputs,
                  "source_manifest_sha256": self.source_sha, "source_xml_audit_sha256": self.audit_sha,
                  "code": self.code,
                  "metric_code": self.metric_code, "backend": self.backend}
        locked = self.log / "phase_inputs.json"
        if locked.exists():
            require(read(locked) == record, "locked phase inputs changed")
        else:
            write(locked, record)

    def n(self, video):
        return self.inputs[video]["n"]

    def cache_id(self, detector):
        return f"r67_utuav_{self.phase}_{self.fingerprint[:12]}_{detector}_c005"

    def gate(self, label, seconds, bytes_needed=5_000_000):
        remaining = dt.datetime.fromisoformat(self.deadline).timestamp() - time.time()
        free_d = shutil.disk_usage(ROOT).free
        free_f = shutil.disk_usage(prep.IMAGES.anchor).free
        ok = remaining >= seconds + 1800 and free_d >= bytes_needed + 5*1024**3 and free_f >= 5*1024**3
        with (self.log / "ngan_sach.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"unit": label, "at": r64.bay_gio(), "deadline": self.deadline,
                "remaining_s": remaining, "estimated_s": seconds, "free_d": free_d, "free_f": free_f,
                "bytes_needed": bytes_needed, "ok": ok}) + "\n")
        if not ok:
            raise r64.HetNganSach(label)


def check_arrays(phase, video, arrays, mapping):
    n = phase.n(video)
    a = arrays["data"]
    counts = arrays["n_per_frame"]
    require(np.array_equal(arrays["frame_idx"], np.arange(1, n+1))
            and np.allclose(arrays["time_s"], np.arange(n) * 1001 / 30000, atol=1e-9, rtol=0)
            and tuple(arrays["orig_shape"]) == (2160, 3840), f"{video}: cache timeline/shape")
    require(len(counts) == n and (counts >= 0).all() and (counts < 3000).all()
            and counts.sum() == len(a) and not arrays["unreadable"].any()
            and not arrays["truncated"].any(), f"{video}: cache frame coverage")
    require(a.shape == (int(counts.sum()), 6) and np.isfinite(a).all()
            and (a[:, 2:4] > a[:, :2]).all() and (a[:, :2] >= 0).all()
            and (a[:, 2] <= 3840).all() and (a[:, 3] <= 2160).all()
            and (a[:, 4] >= 0.05-1e-6).all() and (a[:, 4] <= 1).all(), f"{video}: cache boxes")
    require((a[:, 5] == np.floor(a[:, 5])).all() and ((a[:, 5] >= 0) & (a[:, 5] <= 79)).all()
            and arrays["cls_goc"].shape == (len(a),), f"{video}: cache classes")
    original = a.copy(); original[:, 5] = arrays["cls_goc"]
    require(np.array_equal(r64.doi_id_sang_coco(original, mapping), a), f"{video}: class remap")


def check_cache(phase, detector, video):
    root = det_cache.CACHE_ROOT / phase.cache_id(detector)
    marker = root / f"{video}.xong.json"
    m = read(marker)
    require((m["detector"], m["video"], m["protocol_hash"]) ==
            (detector, video, phase.fingerprint), f"{video}: cache identity")
    check_hashes(root, m["sha256"])
    meta = read(root / "meta.json")
    c = r64.detector_r54(detector)
    require(all(meta.get(k) == x for k, x in {"backend": "ultralytics", "weights_sha256": c["sha256"],
        "imgsz": 640, "conf": 0.05, "iou": 0.7, "max_det": 3000, "half": False,
        "rect": True, "dataset": "utuav"}.items()), f"{video}: cache spec")
    info = read(root / f"{video}.json")
    require(info["spec_sha256"] == meta["spec_sha256"] and info["protocol_hash"] == phase.fingerprint
            and info["source"] == phase.inputs[video], f"{video}: cache provenance")
    mapping = r64.bang_anh_xa_coco({int(k): v for k, v in info["names"].items()}, c["loai_checkpoint"])
    with np.load(root / f"{video}.npz") as z:
        check_arrays(phase, video, {k: z[k] for k in z.files}, mapping)
    return prep.digest(marker)


def cache(phase, detector):
    root = det_cache.CACHE_ROOT / phase.cache_id(detector)
    missing = []
    for v in phase.videos:
        if (root / f"{v}.xong.json").exists():
            check_cache(phase, detector, v)
        else:
            require(not any((root / f"{v}.{ext}").exists() for ext in ("npz", "json")),
                    f"{detector}/{v}: incomplete published cache")
            missing.append(v)
    if not missing:
        return
    from ultralytics import YOLO
    c = r64.detector_r54(detector)
    checkpoint = ROOT / c["checkpoint"]
    require(prep.digest(checkpoint) == c["sha256"], f"{detector}: checkpoint changed")
    spec = det_cache.DetectorSpec(backend="ultralytics", weights=str(checkpoint),
        weights_sha256=c["sha256"], imgsz=640, conf=0.05, iou=0.7, classes=None,
        max_det=3000, half=False, agnostic_nms=False, augment=False, batch=1, device="0",
        mask_ignored_regions=False, dataset="utuav", part="validation").fill_env()
    phase.gate(f"init cache {detector}", 60)
    det_cache.write_meta(root, spec, mode="append" if root.joinpath("meta.json").exists() else "create")
    model = YOLO(str(checkpoint))
    mapping = r64.bang_anh_xa_coco(model.names, c["loai_checkpoint"])
    for v in missing:
        n = phase.n(v)
        phase.gate(f"cache {detector}/{v}", max(90, n/4), n * 4096)
        frames = [FrameRef(index=i, path=prep.IMAGES / v / f"{i:04d}.jpg",
                           time_s=(i-1) * 1001 / 30000) for i in range(1, n+1)]
        start = time.monotonic()
        arrays = det_cache.build_video_yolo(model, frames, [], spec, (2160, 3840))
        original = arrays["data"]
        arrays["cls_goc"] = original[:, 5].astype(np.int16)
        arrays["data"] = r64.doi_id_sang_coco(original, mapping)
        check_arrays(phase, v, arrays, mapping)
        extra = {"detector": detector, "dataset": "utuav", "split": phase.split,
            "checkpoint_sha256": c["sha256"], "protocol_hash": phase.fingerprint,
            "source": phase.inputs[v], "timeline": {"n": n, "fps": FPS, "stride": 1},
            "names": {str(k): str(x) for k, x in model.names.items()},
            "anh_xa_coco": None if mapping is None else {str(k): x for k, x in mapping.items()}}
        staging = root / "_attempts" / f"{v}_{uuid.uuid4().hex}"
        det_cache.save_video(staging, v, arrays, extra=extra, spec=spec)
        marker = {"detector": detector, "video": v, "protocol_hash": phase.fingerprint,
            "seconds": time.monotonic()-start,
            "sha256": {**hashes(staging, [f"{v}.npz", f"{v}.json"]),
                       "meta.json": prep.digest(root / "meta.json")}}
        write(staging / f"{v}.xong.json", marker)
        for f in (f"{v}.npz", f"{v}.json", f"{v}.xong.json"):
            publish(staging / f, root / f, prep.digest(staging / f))
        check_cache(phase, detector, v)
        print(f"cache {detector}/{v}: {n} frames, {len(arrays['data'])} boxes, {marker['seconds']:.1f}s", flush=True)


def paths(video):
    return [name.format(v=video) for name in FILES]


def run_name(detector, cond, loc, seed, repeat=False):
    suffix = "" if seed is None else f"-s{seed}"
    return f"r67utuav{'-repeat' if repeat else ''}-{detector}-{cond}-{loc}{suffix}"


def validate_track(folder, video, n):
    tm = pd.read_csv(folder / f"timing/{video}.csv")
    require(np.array_equal(tm["frame"], np.arange(1, n+1)) and (tm["n_steps"] == 1).all(),
            f"{video}: incomplete/irregular tracking timeline")
    p = folder / f"data/{video}.txt"
    a = np.loadtxt(p, delimiter=",", ndmin=2) if p.stat().st_size else np.empty((0, 7))
    require(a.shape[1] >= 7 and np.isfinite(a).all() and (a[:, 4:6] > 0).all()
            and (a[:, 0] >= 1).all() and (a[:, 0] <= n).all() and not r64.dong_trung(a).any(),
            f"{video}: invalid tracker output")
    cls = pd.read_csv(folder / f"data_cls/{video}.csv")
    require(len(cls) == len(a) and np.array_equal(cls[["frame", "id"]].values, a[:, :2]),
            f"{video}: class file differs")


def score(phase, name, run_meta):
    final = config.RESULTS_DIR / "trackeval" / phase.split / name
    files = ["raw_fields.json", "combined_metrics.csv", "per_video_metrics.csv"]
    fingerprint = fp.sha256_obj(run_meta)
    if final.exists():
        m = read(final / "unit.json")
        require(m["fingerprint"] == fingerprint and m["validated"], f"{name}: stale score")
        check_hashes(final, m["sha256"])
        root = final
    else:
        phase.gate(f"score {name}", 90)
        root = final.parent / "_staging" / f"{name}_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        evaluated, _ = rte.run(phase.split, name, phase.videos, "mot")
        df = rte.flatten_results(evaluated, name)
        df[df["video"] != "COMBINED_SEQ"].sort_values("video").to_csv(root / "per_video_metrics.csv", index=False)
        df[df["video"] == "COMBINED_SEQ"].to_csv(root / "combined_metrics.csv", index=False)
        boot_stats.save_raw_fields(evaluated, name, root / "raw_fields.json")
    raw = read(root / "raw_fields.json")
    require(set(raw) == set(phase.videos), f"{name}: raw sequence set")
    combined = pd.read_csv(root / "combined_metrics.csv")
    per = pd.read_csv(root / "per_video_metrics.csv")
    require(len(combined) == 1 and combined.iloc[0]["video"] == "COMBINED_SEQ"
            and set(per["video"]) == set(phase.videos) and len(per) == 2, f"{name}: score rows")
    calculated = combine(raw, phase.videos)
    delta = max(abs(float(calculated[k])-float(combined.iloc[0][k])) for k in calculated)
    require(delta <= 1e-9, f"{name}: recomposition delta {delta}")
    if not final.exists():
        write(root / "unit.json", {"fingerprint": fingerprint, "validated": True,
            "sha256": hashes(root, files), "max_delta": delta})
        root.rename(final)
    return calculated


def tracking(phase, detector, cond, loc, seed, repeat=False):
    yaml, _ = r64.BO_LOC[loc]
    cfg = rt.load_tracker_cfg(yaml, {"track_buffer": BUFFER})
    require(r64.max_time_lost_thuc(yaml, BUFFER) == BUFFER, "effective buffer differs")
    if loc == "pf500":
        require(rt.cfg_to_dict(cfg).get("pf_num_particles") == 500, "PF N differs")
    name = run_name(detector, cond, loc, seed, repeat)
    settings = {"protocol_hash": phase.fingerprint, "tracker_cfg": rt.cfg_to_dict(cfg),
        "yaml_sha256": prep.digest(rt.resolve_tracker_yaml(yaml)), "backend": phase.backend,
        "detector": detector, "condition": cond, "conf_min": r64.DIEU_KIEN[cond],
        "classes": CLASSES, "max_det": 300, "seed": seed, "fps": FPS,
        "dt_nominal_s": 1001/30000, "elapsed": "steps", "buffer": BUFFER}
    run_root = config.PROCESSED_DIR / "trackers" / phase.split / name
    certificates = {}
    for v in phase.videos:
        cache_marker = check_cache(phase, detector, v)
        unit_fp = fp.sha256_obj({"settings": settings, "cache": cache_marker, "input": phase.inputs[v]})
        unit = phase.log / "units" / name / v
        marker = unit / "unit.json"
        if not unit.exists():
            require(not any((run_root / f).exists() for f in paths(v)), f"{name}/{v}: orphan output")
            phase.gate(f"track {name}/{v}", max(60, phase.n(v)*(0.15 if loc == "pf500" else 0.02)))
            stage = phase.log / "staging" / name / f"{v}_{uuid.uuid4().hex}"
            vd = det_cache.open_cache(phase.cache_id(detector), v, r64.DIEU_KIEN[cond], CLASSES, 300, None)
            start = time.monotonic()
            rt.run_video_to_dir(stage, v, vd, cfg, 1001/30000, "steps", seed, rt.tracker_registry())
            validate_track(stage, v, phase.n(v))
            write(stage / "unit.json", {"fingerprint": unit_fp, "settings": settings,
                "video": v, "cache_marker_sha256": cache_marker,
                "seconds": time.monotonic()-start, "sha256": hashes(stage, paths(v))})
            unit.parent.mkdir(parents=True, exist_ok=True)
            stage.rename(unit)
        m = read(marker)
        require(m["fingerprint"] == unit_fp and m["settings"] == settings
                and m["video"] == v and m["cache_marker_sha256"] == cache_marker,
                f"{name}/{v}: stale tracker unit")
        check_hashes(unit, m["sha256"])
        validate_track(unit, v, phase.n(v))
        for path in paths(v):
            publish(unit / path, run_root / path, m["sha256"][path])
        certificates[v] = prep.digest(marker)
    require({p.stem for p in (run_root / "data").glob("*.txt")} == set(phase.videos),
            f"{name}: published sequence set differs")
    meta = {"settings": settings, "units": certificates, "videos": phase.videos,
            "track_buffer_frames": BUFFER, "elapsed_mode": "steps", "fps": FPS}
    meta_path = run_root / "run_meta.json"
    if meta_path.exists():
        require(read(meta_path) == meta, f"{name}: stale run metadata")
    else:
        write(meta_path, meta)
    r64.hau_kiem_tracking({"split": phase.split, "videos": phase.videos,
        "n_goc": {v: phase.n(v) for v in phase.videos}, "n_frame": None, "fps": FPS}, name, BUFFER, BUFFER)
    return score(phase, name, meta)


def matrix(phase):
    rows = []
    for detector in r64.DETECTOR:
        for cond in r64.DIEU_KIEN:
            for loc, (_, seeds) in r64.BO_LOC.items():
                for seed in seeds or [None]:
                    metric = tracking(phase, detector, cond, loc, seed)
                    name = run_name(detector, cond, loc, seed)
                    rows.append({"run": name, "detector": detector, "condition": cond,
                                 "filter": loc, "seed": seed, **metric})
                    print(f"{name}: HOTA {100*metric['HOTA']:.3f}, IDF1 {100*metric['IDF1']:.3f}", flush=True)
    require(len(rows) == 36, "matrix incomplete")
    out = phase.log / "combined_tung_run.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    write(phase.log / "summary.json", {"status": "complete", "runs": 36,
        "protocol_hash": phase.fingerprint, "source_manifest_sha256": phase.source_sha,
        "csv_sha256": prep.digest(out)})


def repeat_pf(phase):
    require(phase.phase == "smoke", "repeat only smoke")
    det, cond, loc, seed = "yolov8n_mixed5", "c25", "pf500", 0
    original = run_name(det, cond, loc, seed)
    repeated = run_name(det, cond, loc, seed, True)
    tracking(phase, det, cond, loc, seed, True)
    base = config.PROCESSED_DIR / "trackers" / phase.split
    result = config.RESULTS_DIR / "trackeval" / phase.split
    for v in phase.videos:
        require(prep.digest(base/original/"data"/f"{v}.txt") ==
                prep.digest(base/repeated/"data"/f"{v}.txt"), f"PF repeat MOT differs: {v}")
    for file in ("raw_fields.json", "combined_metrics.csv", "per_video_metrics.csv"):
        require(prep.digest(result/original/file) == prep.digest(result/repeated/file),
                f"PF repeat score differs: {file}")
    write(phase.log / "repeat_pf.json", {"status": "PASS", "original": original,
        "repeat": repeated, "protocol_hash": phase.fingerprint})


def full_gate():
    smoke = prep.LOG / "smoke"
    summary = read(smoke / "summary.json")
    repeat = read(smoke / "repeat_pf.json")
    review = read(smoke / "review.json")
    require(summary["status"] == "complete" and summary["runs"] == 36
            and repeat["status"] == "PASS" and summary["protocol_hash"] == repeat["protocol_hash"]
            and summary["csv_sha256"] == prep.digest(smoke / "combined_tung_run.csv")
            and review["decision"] == "PASS" and review["summary_sha256"] == prep.digest(smoke / "summary.json")
            and review["repeat_sha256"] == prep.digest(smoke / "repeat_pf.json"),
            "smoke technical review missing/stale")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["cache", "matrix", "repeat-pf", "all"])
    parser.add_argument("--phase", choices=["smoke", "full"], required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    deadline = dt.datetime.fromisoformat(args.deadline)
    require(deadline.tzinfo is not None, "deadline needs timezone")
    remaining = deadline.timestamp() - time.time()
    require(0 < remaining <= 7200, "session deadline must be within 2 hours")
    if args.phase == "full":
        full_gate()
    if not args.worker:
        budget = remaining - 1800
        require(budget > 0, "no time outside 30-minute review reserve")
        try:
            return subprocess.run([sys.executable, "-B", __file__, *sys.argv[1:], "--worker"],
                                  timeout=budget).returncode
        except subprocess.TimeoutExpired:
            write(prep.LOG / args.phase / "partial.json", {"status": "PARTIAL", "step": args.step,
                "deadline": args.deadline})
            return 10
    phase = Phase(args.phase, args.deadline)
    if args.step in ("cache", "all"):
        for detector in r64.DETECTOR:
            cache(phase, detector)
    if args.step in ("matrix", "all"):
        matrix(phase)
    if args.step in ("repeat-pf", "all") and args.phase == "smoke":
        repeat_pf(phase)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (r64.LoiHopDong, r64.HetNganSach, ValueError, OSError, KeyError) as error:
        print(f"[STOP] {error}", flush=True)
        raise SystemExit(10 if isinstance(error, r64.HetNganSach) else 2)
