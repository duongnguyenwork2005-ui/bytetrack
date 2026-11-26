"""Run the locked R66 HELMET motorcycle evaluation. Run one phase at a time.

    python -B src/r66_ngoai/helmet_eval.py plan --phase smoke
    python -B src/r66_ngoai/helmet_eval.py cache --phase smoke --deadline <ISO-with-timezone>
    python -B src/r66_ngoai/helmet_eval.py matrix --phase smoke --deadline <ISO-with-timezone>
    python -B src/r66_ngoai/helmet_eval.py repeat-pf --phase smoke --deadline <ISO-with-timezone>

Full cache/matrix additionally require --review FILE (a recorded smoke PASS decision).
"""
from __future__ import annotations

import argparse
import configparser
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

import danh_gia_r66 as r66
import helmet_prep as prep
import run_fingerprint as fp
import run_tracking as rt
from danh_gia_r66 import config, det_cache, r64
from mot_datasets.base import FrameRef


def require(ok, message):
    if not ok:
        raise r64.LoiHopDong(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    r66.ghi_json(path, obj)


def hashes(folder, names):
    return {name: prep.sha256(folder / name) for name in names}


def check_files(folder, names, locked):
    require(set(locked) == set(names), f"{folder}: marker artifact set differs")
    require(hashes(folder, names) == locked, f"{folder}: artifact content changed")


def publish_file(source, target, digest):
    if target.exists():
        require(prep.sha256(target) == digest, f"published content differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
    shutil.copyfile(source, temp)
    require(prep.sha256(temp) == digest, f"copy differs: {target}")
    temp.rename(target)


def videos_for(phase):
    rows = prep.manifest()["videos"]
    clips = [r["video_id"] for r in rows if phase == "full" or r["rank_in_group"] == 1]
    require(len(set(clips)) == len(clips) == (36 if phase == "full" else 12), "invalid phase sample")
    return clips


class Phase:
    """Resolve the prepared split, hashes and input paths exactly once per invocation."""

    def __init__(self, phase, deadline):
        self.phase = phase
        self.clips = videos_for(phase)
        self.split = prep.SPLIT if phase == "full" else prep.SPLIT_SMOKE
        self.log = prep.LOG / phase
        self.log.mkdir(parents=True, exist_ok=True)
        self.deadline = deadline
        self.inputs = {}
        root = config.PROCESSED_DIR / self.split
        gt_log = read(root / "gt_log.json")
        mh = prep.sha256(prep.OUTPUT)
        require(gt_log["manifest_sha256"] == mh and set(gt_log["videos"]) == set(self.clips), "GT manifest mismatch")
        seqmap = config.PROCESSED_DIR / "seqmaps" / f"{self.split}.txt"
        require(seqmap.read_text(encoding="utf-8").splitlines() == ["name", *self.clips], "seqmap mismatch")
        extraction = read(prep.LOG / "giai_nen_test36.json")
        verification = read(prep.LOG / "nguon_kiem_sha_crc.json")
        for name, expected in prep.listed_files().items():
            entry = verification[name]
            require(entry["sha256"] == expected["sha256"] and entry["size"] == expected["size"]
                    and (not name.endswith(".zip") or entry["crc_ok"]), f"unverified HELMET source {name}")
        for v in self.clips:
            p = root / v
            gt = prep.sha256(p / "gt/gt.txt")
            require(gt == gt_log["videos"][v]["gt_sha256"], f"{v}: GT differs from converter")
            a = np.loadtxt(p / "gt/gt.txt", delimiter=",", ndmin=2)
            r64.kiem_gt(a, 100, v)
            require(len(a) == gt_log["videos"][v]["gt_rows"] and a.shape[1] == 9
                    and (a[:, 6:9] == 1).all(), f"{v}: invalid MOT classes/rows")
            ini = configparser.ConfigParser()
            ini.read(p / "seqinfo.ini", encoding="utf-8")
            seq = ini["Sequence"]
            require([seq.getint(k) for k in ("frameRate", "seqLength", "imWidth", "imHeight")]
                    == [10, 100, 1920, 1080] and seq["name"] == v
                    and Path(seq["imDir"]).resolve() == (prep.IMAGES / v).resolve(), f"{v}: incorrect seqinfo")
            paths = self.images(v)
            require(extraction[v]["frames"] == 100 and len(list((prep.IMAGES / v).glob("*.jpg"))) == 100,
                    f"{v}: incomplete image set")
            live_hashes = {str(i):prep.sha256(path) for i,path in enumerate(paths,1)}
            require(live_hashes == extraction[v]["sha256"] and extraction[v]["manifest_sha256"] == mh,
                    f"{v}: images differ from verified archive entries")
            source = fp.sha256_obj(list(live_hashes.values()))
            self.inputs[v] = {"gt_sha256": gt, "seqinfo_sha256": prep.sha256(p / "seqinfo.ini"),
                              "image_sha256": source}
        self.implementation = hashes(r66.GOC, [*r66.MA_THUC_DUNG, "src/r66_ngoai/helmet_eval.py",
            "src/r66_ngoai/helmet_manifest.py", "src/r66_ngoai/helmet_prep.py"])
        import trackeval
        from trackeval.datasets import MotChallenge2DBox
        import inspect
        self.evaluator = {c.__name__: prep.sha256(Path(inspect.getfile(c))) for c in
                          [MotChallenge2DBox, trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity,
                           trackeval.Evaluator]}
        self.fingerprint = fp.sha256_obj({"protocol": "HELMET-R66-v1", "phase": phase,
            "split": self.split, "manifest": mh, "archives": verification, "extraction": extraction,
            "seqmap_sha256": prep.sha256(seqmap), "class_mapping": "mixed5 via model.names -> COCO; track [3]",
            "gt_class": 1, "visibility": 1, "roi": None, "fps": 10, "track_buffer": 12,
            "inputs": self.inputs, "code": self.implementation, "evaluator": self.evaluator})
        self.backend = fp.code_fingerprint()
        require(self.backend["backend"]["backend"] == "ultralytics", "R66 requires the locked ultralytics backend")
        self.cache_checked = {}
        self.tracker_registry = rt.tracker_registry()
        record = {"phase": phase, "split": self.split, "clips": self.clips, "protocol_hash": self.fingerprint,
                  "cache_ids": {d: self.cache_id(d) for d in r64.DETECTOR},
                  "manifest_sha256": mh, "inputs": self.inputs, "implementation": self.implementation,
                  "evaluator": self.evaluator, "tracker_backend": self.backend}
        record_path = self.log / "phase_inputs.json"
        if record_path.exists():
            require(read(record_path) == record, f"{record_path}: previously locked inputs differ")
        else:
            write(record_path, record)

    @staticmethod
    def images(video):
        return [prep.IMAGES / video / f"{frame}.jpg" for frame in range(1, 101)]

    def cache_id(self, detector):
        phase = "smoke12" if self.phase == "smoke" else "test36"
        return f"r66_helmet_{phase}_{self.fingerprint[:12]}_{detector}_c005"

    def gate(self, label, seconds, byte_count):
        require(prep.IMAGES.is_dir() and prep.SOURCE.is_dir(), "HELMET input volume unavailable")
        free = {p:shutil.disk_usage(p).free for p in ("D:/","F:/")}
        remaining = dt.datetime.fromisoformat(self.deadline).timestamp() - time.time()
        ok = remaining >= seconds + 1800 and free["D:/"] >= byte_count + 5*1024**3 and free["F:/"] >= 5*1024**3
        with (self.log/"ngan_sach.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(dict(unit=label,seconds=seconds,bytes=byte_count,free=free,
                        at=r64.bay_gio(),deadline=self.deadline,remaining=remaining,ok=ok))+"\n")
        if not ok:
            raise r64.HetNganSach(label)

    def recheck_sources(self):
        live = Phase(self.phase, self.deadline)
        require(live.fingerprint == self.fingerprint and live.backend == self.backend,
                "HELMET inputs or implementation changed during this run")
        for (detector, video), digest in self.cache_checked.items():
            require(check_cache(live, detector, video) == digest, f"{detector}/{video}: cache changed")


@contextmanager
def exclusive_worker():
    import msvcrt
    prep.LOG.mkdir(parents=True, exist_ok=True)
    with (prep.LOG / "worker.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def check_arrays(arrays, mapping):
    a = arrays["data"]
    counts = arrays["n_per_frame"]
    require(np.array_equal(arrays["frame_idx"], np.arange(1, 101))
            and np.allclose(arrays["time_s"], np.arange(100) / 10, atol=1e-9, rtol=0)
            and tuple(arrays["orig_shape"]) == (1080, 1920), "cache timeline or shape differs")
    require(counts.shape == (100,) and (counts >= 0).all() and (counts < 3000).all()
            and counts.sum() == len(a) and not arrays["unreadable"].any() and not arrays["truncated"].any(),
            "invalid frame count, unreadable or truncated cache")
    require(a.shape == (int(counts.sum()), 6) and np.isfinite(a).all()
            and (a[:, 2:4] > a[:, :2]).all() and (a[:, 4] >= 0.05 - 1e-6).all() and (a[:, 4] <= 1).all(),
            "invalid detection geometry/confidence")
    require((a[:, 5] == np.floor(a[:, 5])).all() and ((0 <= a[:, 5]) & (a[:, 5] <= 79)).all(),
            "invalid COCO category")
    original = a.copy()
    require(arrays["cls_goc"].shape == (len(a),), "unaligned cls_goc")
    original[:, 5] = arrays["cls_goc"]
    require(np.array_equal(r64.doi_id_sang_coco(original, mapping), a), "COCO class remapping differs")


def check_cache(phase, detector, video):
    key = (detector, video)
    if key in phase.cache_checked:
        return phase.cache_checked[key]
    root = det_cache.CACHE_ROOT / phase.cache_id(detector)
    marker = root / f"{video}.xong.json"
    m = read(marker)
    require((m["detector"], m["video"], m["split"], m["protocol_hash"]) ==
            (detector, video, phase.split, phase.fingerprint), f"{marker}: wrong identity")
    check_files(root, [f"{video}.npz", f"{video}.json", "meta.json"], m["sha256"])
    meta = read(root / "meta.json")
    c = r64.detector_r54(detector)
    expected = {"backend": "ultralytics", "weights_sha256": c["sha256"], "imgsz": 640,
                "conf": 0.05, "iou": 0.7, "max_det": 3000, "classes": None, "half": False,
                "agnostic_nms": False, "augment": False, "batch": 1, "mask_ignored_regions": False,
                "rect": True, "device": "0", "dataset": "helmet"}
    require(all(meta.get(k) == val for k, val in expected.items()) and
            meta["spec_sha256"] == fp.sha256_obj(det_cache.meta_gen_spec(meta)), f"{root}: detector spec mismatch")
    info = read(root / f"{video}.json")
    expected_info = {"protocol_hash": phase.fingerprint, "split": phase.split, "dataset": "helmet",
                     "detector": detector, "checkpoint_sha256": c["sha256"], "ma_sha256": phase.implementation,
                     "nguon": phase.inputs[video], "gt_luc_sinh_sha256": phase.inputs[video]["gt_sha256"],
                     "spec_sha256": meta["spec_sha256"], "video": video, "n_frames": 100}
    require(all(info.get(k) == val for k, val in expected_info.items()), f"{video}: clip provenance mismatch")
    mapping = r64.bang_anh_xa_coco({int(k): v for k, v in info["names"].items()}, c["loai_checkpoint"])
    require(info["anh_xa_coco"] == (None if mapping is None else {str(k): v for k, v in mapping.items()}),
            f"{video}: class mapping changed")
    with np.load(root / f"{video}.npz") as z:
        check_arrays(z, mapping)
    digest = prep.sha256(marker)
    phase.cache_checked[key] = digest
    return digest


def publish_cache(root, video, folder):
    marker_name = f"{video}.xong.json"
    certificate = read(folder / marker_name)
    expected = [f"{video}.npz", f"{video}.json"]
    require(set(certificate["sha256"]) == {"meta.json", *expected}
            and prep.sha256(root / "meta.json") == certificate["sha256"]["meta.json"],
            f"{video}: stale cache completion")
    for name in expected + [marker_name]:
        src, dst = folder / name, root / name
        h = prep.sha256(src)
        require(name == marker_name or h == certificate["sha256"][name], f"{video}: unit changed")
        publish_file(src, dst, h)


def cache(phase, detector):
    root = det_cache.CACHE_ROOT / phase.cache_id(detector)
    print("cache:", root.name, flush=True)
    missing = []
    for v in phase.clips:
        staged = root / "_committed" / v
        if staged.exists() and not (root / f"{v}.xong.json").exists():
            publish_cache(root, v, staged)
        if (root / f"{v}.xong.json").exists():
            check_cache(phase, detector, v)
        else:
            require(not any((root / f"{v}.{s}").exists() for s in ("npz", "json")), f"{root}/{v}: orphan cache")
            missing.append(v)
    if not missing:
        return
    phase.gate(f"init cache {detector}", 60, 5_000_000)
    from ultralytics import YOLO
    c = r64.detector_r54(detector)
    checkpoint = r66.GOC / c["checkpoint"]
    spec = det_cache.DetectorSpec(weights=str(checkpoint), weights_sha256=c["sha256"], mask_ignored_regions=False,
                                  dataset="helmet", part="test").fill_env()
    det_cache.write_meta(root, spec, mode="append")
    model = YOLO(str(checkpoint))
    mapping = r64.bang_anh_xa_coco(model.names, c["loai_checkpoint"])
    for v in missing:
        estimate_s = 75
        if phase.phase == "full":
            estimates = [read(p)["seconds"] for p in det_cache.CACHE_ROOT.glob(
                f"r66_helmet_smoke12_*_{detector}_c005/*.xong.json")]
            if estimates:
                estimate_s = max(estimate_s, max(estimates) * 2 + 30)
        phase.gate(f"cache {detector}/{v}", estimate_s, 9_000_000)
        frames = [FrameRef(index=i, path=p, time_s=(i - 1) / 10) for i, p in enumerate(phase.images(v), 1)]
        start = time.monotonic()
        arrays = det_cache.build_video_yolo(model, frames, [], spec, (1080, 1920))
        original = arrays["data"]
        arrays["cls_goc"] = original[:, 5].astype(np.int16)
        arrays["data"] = r64.doi_id_sang_coco(original, mapping)
        check_arrays(arrays, mapping)
        extra = {"detector": detector, "dataset": "helmet", "split": phase.split, "checkpoint_sha256": c["sha256"],
                 "protocol_hash": phase.fingerprint, "timeline": {"n": 100, "fps": 10}, "nguon": phase.inputs[v],
                 "ma_sha256": phase.implementation, "gt_luc_sinh_sha256": phase.inputs[v]["gt_sha256"],
                 "anh_xa_coco": None if mapping is None else {str(k): val for k, val in mapping.items()},
                 "names": {str(k): s for k, s in model.names.items()}}
        attempt = root / "_attempts" / f"{v}_{uuid.uuid4().hex}"
        det_cache.save_video(attempt, v, arrays, extra=extra, spec=spec)
        write(attempt / f"{v}.xong.json", {"detector": detector, "video": v, "split": phase.split,
              "protocol_hash": phase.fingerprint, "seconds": time.monotonic() - start,
              "sha256": {**hashes(attempt, [f"{v}.npz", f"{v}.json"]), **hashes(root, ["meta.json"])}})
        committed = root / "_committed" / v
        committed.parent.mkdir(parents=True, exist_ok=True)
        attempt.rename(committed)
        publish_cache(root, v, committed)
        check_cache(phase, detector, v)
        print(f"cache {detector}/{v} OK", flush=True)


def clip_paths(video):
    return (f"data/{video}.txt", f"data_cls/{video}.csv", f"timing/{video}.csv")


def validate_track(folder, video):
    timing = pd.read_csv(folder / f"timing/{video}.csv")
    require(np.array_equal(timing["frame"], np.arange(1, 101)) and (timing["n_steps"] == 1).all()
            and np.isfinite(timing.to_numpy(dtype=float)).all(), f"{video}: invalid timing/steps")
    path = folder / f"data/{video}.txt"
    a = np.loadtxt(path, delimiter=",", ndmin=2) if path.stat().st_size else np.empty((0, 7))
    require(a.shape[1] >= 7 and np.isfinite(a).all() and (a[:, 4:6] > 0).all()
            and ((a[:, 0] >= 1) & (a[:, 0] <= 100)).all()
            and np.equal(a[:, :2], np.floor(a[:, :2])).all() and not r64.dong_trung(a).any(),
            f"{video}: invalid MOT coordinates or IDs")
    cls = pd.read_csv(folder / f"data_cls/{video}.csv")
    require(len(cls) == len(a) and (cls["cls"] == 3).all()
            and np.array_equal(cls[["frame", "id"]].values, a[:, :2]), f"{video}: wrong tracker class")


def publish_clip(folder, run_root, video, certificate):
    for name in clip_paths(video):
        publish_file(folder / name, run_root / name, certificate["sha256"][name])


def run_name(detector, cond, loc, seed, repeat=False):
    suffix = "" if seed is None else f"-s{seed}"
    return f"r66helmet{'-repeat' if repeat else ''}-{detector}-{cond}-{loc}{suffix}"


def estimate(phase, name, key, fallback):
    path = prep.LOG / "smoke" / "units" / name
    if phase.phase == "full" and path.exists():
        times = [read(p)["seconds"] for p in path.glob(f"*/{key}")]
        if times:
            return max(fallback, 2 * max(times) + 5)
    return fallback


def tracking(phase, detector, cond, loc, seed, repeat=False):
    yaml, _ = r64.BO_LOC[loc]
    cfg = rt.load_tracker_cfg(yaml, {"track_buffer": 12})
    cfgdict = rt.cfg_to_dict(cfg)
    require(r64.max_time_lost_thuc(yaml, 12) == 12, "effective lifecycle buffer differs")
    if loc == "pf500":
        require(cfgdict.get("pf_num_particles") == 500, "PF particle count differs")
        require(not cfgdict.get("pf_optimal_proposal", False) and not cfgdict.get("pf_roughening", 0),
                "PF experimental options enabled")
    name = run_name(detector, cond, loc, seed, repeat)
    units = phase.log / "units" / name
    certificates = {}
    run_root = config.PROCESSED_DIR / "trackers" / phase.split / name
    settings = {"protocol_hash": phase.fingerprint, "tracker_cfg": cfgdict,
                "yaml_sha256": prep.sha256(Path(rt.resolve_tracker_yaml(yaml))),
                "backend": phase.backend, "detector": detector, "threshold": r64.DIEU_KIEN[cond],
                "classes": [3], "max_det": 300, "seed": seed, "elapsed_mode": "steps", "fps": 10,
                "max_time_lost_thuc": 12, "track_buffer_frames": 12, "dt_nominal_s": 0.1}
    settings_fp = fp.sha256_obj(settings)
    for v in phase.clips:
        cache_marker = check_cache(phase, detector, v)
        fingerprint = fp.sha256_obj({"settings": settings_fp, "cache_marker": cache_marker,
                                     "clip": v, "inputs": phase.inputs[v]})
        folder = units / v
        marker = folder / "unit.json"
        if not folder.exists():
            require(not any((run_root / f).exists() for f in clip_paths(v)), f"{name}/{v}: orphan tracker output")
            phase.gate(f"track {name}/{v}", estimate(phase, name, "unit.json", 35), 5_000_000)
            attempt = phase.log / "staging" / name / f"{v}_{uuid.uuid4().hex}"
            start = time.monotonic()
            vd = det_cache.open_cache(phase.cache_id(detector), v, r64.DIEU_KIEN[cond], [3], 300, None)
            rt.run_video_to_dir(attempt, v, vd, cfg, 0.1, "steps", seed, phase.tracker_registry)
            validate_track(attempt, v)
            write(attempt / "unit.json", {"protocol_hash": phase.fingerprint, "fingerprint": fingerprint,
                  "video": v, "seed": seed, "settings": settings, "cache_marker_sha256": cache_marker,
                  "sha256": hashes(attempt, clip_paths(v)), "seconds": time.monotonic() - start})
            units.mkdir(parents=True, exist_ok=True)
            attempt.rename(folder)  # immutable completion, abandoned attempts stay in staging
        certificate = read(marker)
        require(certificate["fingerprint"] == fingerprint and certificate["settings"] == settings
                and certificate["video"] == v and certificate["seed"] == seed, f"{name}/{v}: marker mismatch")
        check_files(folder, clip_paths(v), certificate["sha256"])
        validate_track(folder, v)
        publish_clip(folder, run_root, v, certificate)
        certificates[v] = prep.sha256(marker)
    require({p.stem for p in (run_root / "data").glob("*.txt")} == set(phase.clips), f"{name}: output set differs")
    run_meta = {**settings, "tracker_backend": phase.backend["backend"]["backend"], "status": "complete",
                "input_fingerprint": settings_fp, "videos": phase.clips, "units": certificates}
    meta_path = run_root / "run_meta.json"
    if meta_path.exists():
        require(read(meta_path) == run_meta, f"{name}: run metadata differs")
    else:
        write(meta_path, run_meta)
    r64.hau_kiem_tracking({"split": phase.split, "videos": phase.clips, "n_frame": 100, "fps": 10}, name, 12, 12)
    return score(phase, name, run_meta)


def score(phase, name, run_meta):
    from sher_metric import combine
    final = config.RESULTS_DIR / "trackeval" / phase.split / name
    names = ["raw_fields.json", "combined_metrics.csv", "per_video_metrics.csv"]
    fingerprint = fp.sha256_obj({"run": run_meta, "metric_helper": prep.sha256(Path(__file__).with_name("sher_metric.py"))})
    if final.exists():
        result = final
        certificate = final / "cong_r66.json"
        require(certificate.is_file(), f"{result}: orphan/incomplete score, preserve for review")
        evidence = read(certificate)
        require(evidence["input_fingerprint"] == fingerprint and evidence["validated"] is True
                and evidence["units"] == run_meta["units"], f"{name}: score fingerprint differs")
        check_files(result, names, evidence["sha256"])
    else:
        import run_trackeval as rte
        import boot_stats
        phase.gate(f"score {name}", 60 + 1.5 * len(phase.clips), 5_000_000)
        result = final.parent / "_staging" / f"{name}_{uuid.uuid4().hex}"
        result.mkdir(parents=True)
        certificate = result / "cong_r66.json"
        evaluated, _ = rte.run(phase.split, name, phase.clips, "mot")
        df = rte.flatten_results(evaluated, name)
        df[df["video"] != "COMBINED_SEQ"].sort_values("video").to_csv(result / "per_video_metrics.csv", index=False)
        df[df["video"] == "COMBINED_SEQ"].to_csv(result / "combined_metrics.csv", index=False)
        boot_stats.save_raw_fields(evaluated, name, result / "raw_fields.json")
    raw = read(result / "raw_fields.json")
    require(set(raw) == set(phase.clips), f"{name}: raw clip set differs")
    per_clip = pd.read_csv(result / "per_video_metrics.csv").set_index("video")
    combined = pd.read_csv(result / "combined_metrics.csv")
    require(len(combined) == 1 and combined.iloc[0]["video"] == "COMBINED_SEQ" and
            not per_clip.index.duplicated().any() and set(per_clip.index) == set(phase.clips), "score rows differ")
    row = combined.iloc[0]
    # A sequence-level row can have a perfectly empty GT/prediction. Use the
    # evaluator's sequence result there; COMBINED always uses combine_sequences.
    require(np.isfinite(per_clip.select_dtypes(include="number").to_numpy()).all(), "nonfinite per-clip scores")
    calculated = combine(raw, phase.clips)
    max_delta = max(abs(float(calculated[k]) - float(row[k]))
                    for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW"))
    require(max_delta <= 1e-9, f"{name}: raw recombination differs by {max_delta}")
    if not certificate.exists():
        write(certificate, {"input_fingerprint": fingerprint, "protocol_hash": phase.fingerprint,
                            "units": run_meta["units"], "sha256": hashes(result, names),
                            "max_recomposition_delta": max_delta, "validated": True})
        result.rename(final)
    return {k: float(row[k]) for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW")}


def matrix(phase):
    rows = []
    for det in r64.DETECTOR:
        for cond in r64.DIEU_KIEN:
            for loc, (_, seeds) in r64.BO_LOC.items():
                for seed in seeds or [None]:
                    metric = tracking(phase, det, cond, loc, seed)
                    name = run_name(det, cond, loc, seed)
                    print(f"{name}: HOTA {100 * metric['HOTA']:.2f} ({len(phase.clips)} clips)", flush=True)
                    rows.append({"run": name, "detector": det, "cond": cond, "loc": loc, "seed": seed, **metric})
    require(len(rows) == 36, "incomplete matrix")
    phase.recheck_sources()
    csv = phase.log / "combined_tung_run.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    write(phase.log / "tong_hop_trang_thai.json", {"status": "complete", "runs": 36, "phase": phase.phase,
          "clips": phase.clips, "protocol_hash": phase.fingerprint, "combined_sha256": prep.sha256(csv),
          "manifest_sha256": prep.sha256(prep.OUTPUT)})


def repeat_pf(phase):
    require(phase.phase == "smoke", "repeat-pf is for the smoke phase")
    original = run_name("yolov8n_mixed5", "c25", "pf500", 0)
    repeated = run_name("yolov8n_mixed5", "c25", "pf500", 0, True)
    base = config.PROCESSED_DIR / "trackers" / phase.split
    results = config.RESULTS_DIR / "trackeval" / phase.split
    # Verify both runs, then compare final MOT, raw fields and metric tables.
    for again in (False, True):
        tracking(phase, "yolov8n_mixed5", "c25", "pf500", 0, again)
    mot_hashes = hashes(base / original / "data", [f"{v}.txt" for v in phase.clips])
    check_files(base / repeated / "data", mot_hashes.keys(), mot_hashes)
    score_hashes = hashes(results / original, ["raw_fields.json", "combined_metrics.csv", "per_video_metrics.csv"])
    check_files(results / repeated, score_hashes.keys(), score_hashes)
    phase.recheck_sources()
    write(phase.log / "repeat_pf.json", {"status": "PASS", "original": original, "repeat": repeated,
          "protocol_hash": phase.fingerprint, "mot_sha256": mot_hashes, "score_sha256": score_hashes})


def full_gate(review):
    require(review is not None and review.is_file(), "full requires --review PATH to the recorded smoke decision")
    smoke = prep.LOG / "smoke"
    summary = read(smoke / "tong_hop_trang_thai.json")
    repeat = read(smoke / "repeat_pf.json")
    decision = read(review)
    require(summary["status"] == "complete" and summary["runs"] == 36 and repeat["status"] == "PASS"
            and summary["protocol_hash"] == repeat["protocol_hash"] and summary["manifest_sha256"] == prep.sha256(prep.OUTPUT)
            and summary["combined_sha256"] == prep.sha256(smoke / "combined_tung_run.csv"), "smoke proof incomplete")
    require(decision["decision"] == "PASS" and decision["summary_sha256"] == prep.sha256(smoke / "tong_hop_trang_thai.json")
            and decision["repeat_pf_sha256"] == prep.sha256(smoke / "repeat_pf.json"), "smoke review not tied to these runs")


def missing_units(phase_name, step):
    clips = videos_for(phase_name)
    log = prep.LOG / phase_name
    if step == "repeat-pf":
        runs = [run_name("yolov8n_mixed5", "c25", "pf500", 0, True)]
    else:
        runs = [run_name(det, cond, loc, s) for det in r64.DETECTOR for cond in r64.DIEU_KIEN
                for loc, (_, seeds) in r64.BO_LOC.items() for s in seeds or [None]]
    tracking = {name: [v for v in clips if not (log / "units" / name / v / "unit.json").is_file()] for name in runs}
    record = log / "phase_inputs.json"
    cache_ids = read(record)["cache_ids"] if record.exists() else {}
    cache = {det: [v for v in clips if det not in cache_ids or
             not (det_cache.CACHE_ROOT / cache_ids[det] / f"{v}.xong.json").is_file()] for det in r64.DETECTOR}
    split = prep.SPLIT if phase_name == "full" else prep.SPLIT_SMOKE
    scores = [name for name in runs if not (config.RESULTS_DIR / "trackeval" / split / name / "cong_r66.json").is_file()]
    return {"cache": cache, "tracking": tracking, "scores": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["plan", "cache", "matrix", "repeat-pf", "all"])
    parser.add_argument("--phase", choices=["smoke", "full"], required=True)
    parser.add_argument("--deadline", help="absolute ISO timestamp with timezone; cap 120 minutes")
    parser.add_argument("--review", type=Path, help="JSON smoke PASS decision for full")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.step == "plan":
        record = prep.LOG / args.phase / "phase_inputs.json"
        print(json.dumps({"phase": args.phase, "clips": videos_for(args.phase), "runs": 36, "fps": 10,
                          "frames_per_clip": 100, "buffer": 12, "class": 3,
                          "cache_ids": read(record)["cache_ids"] if record.exists() else None}, indent=2))
        return 0
    require(args.deadline is not None, "deadline is required")
    deadline = dt.datetime.fromisoformat(args.deadline)
    require(deadline.tzinfo is not None, "deadline requires a timezone")
    remaining = deadline.timestamp() - time.time()
    require(remaining <= 7200, "deadline exceeds 120 minute cap")
    if args.phase == "full":
        full_gate(args.review)
    if not args.worker:
        # Kill the computation itself at the absolute cutoff and preserve 30 min
        # for review. Completed clip folders survive a timeout.
        budget = remaining - r66.DU_PHONG_GIAY
        if budget <= 0:
            raise r64.HetNganSach("deadline already in review reserve")
        try:
            result = subprocess.run([sys.executable, "-B", __file__, *sys.argv[1:], "--worker"], timeout=budget)
            rc = result.returncode
        except subprocess.TimeoutExpired:
            rc = 10
        if rc == 10:
            report = prep.LOG / args.phase / "partial.json"
            write(report, {"status": "PARTIAL", "step": args.step, "deadline": args.deadline,
                  "missing_clip_markers": missing_units(args.phase, args.step)})
        return rc
    with exclusive_worker():
        phase = Phase(args.phase, args.deadline)
        if args.step in ("cache", "all"):
            for detector in r64.DETECTOR:
                cache(phase, detector)
            phase.recheck_sources()
        if args.step in ("matrix", "all"):
            matrix(phase)
        if args.step == "repeat-pf" or (args.step == "all" and args.phase == "smoke"):
            repeat_pf(phase)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except r64.HetNganSach as exc:
        print(f"[PARTIAL] {exc}", flush=True)
        raise SystemExit(10)
    except (r64.LoiHopDong, ValueError, OSError, KeyError) as exc:
        print(f"[STOP] {exc}", flush=True)
        raise SystemExit(2)
