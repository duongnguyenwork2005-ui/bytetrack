"""Capture two existing UA samples and profile only PF tracker.update on Nano."""
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import time

import numpy as np

HOME = Path.home()
BASE = HOME / "nano_run/r63_20261030/v21combo/nano_transfer"
DATA = HOME / "nano_run/r86_multivideo_20261106/sample/ua"
OUT = HOME / "nano_run/r98_pf_hotspots_20261110"
ENGINE = HOME / "nano_run/r63_20261030/g3_c1_384x640_toc_do_01/engine/yolov8n_384x640_op12_fp16.engine"
ONNX = HOME / "nano_run/r63_20261030/ung_vien_onnx/384x640/yolov8n_384x640_op12.onnx"
BOOT = Path("/proc/sys/kernel/random/boot_id")
ERRORS = Path("/sys/fs/ext4/mmcblk1p1/errors_count")
VIDEOS = ("MVI_39031", "MVI_40855")


def swap_mb():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("SwapTotal:", "SwapFree:")):
            key, value = line.split(":")
            fields[key] = int(value.strip().split()[0])
    return (fields["SwapTotal"] - fields["SwapFree"]) / 1024


def main():
    boot, errors, swap0 = BOOT.read_text(), ERRORS.read_text(), swap_mb()
    assert swap0 < 100 and errors.strip() == "0"
    OUT.mkdir(exist_ok=True)
    cache_root = OUT / "cache"
    cache_root.mkdir(exist_ok=True)
    onnx_sha = hashlib.sha256(ONNX.read_bytes()).hexdigest()
    for video in VIDEOS:
        assert BOOT.read_text() == boot and ERRORS.read_text() == errors and swap_mb() - swap0 <= 100
        source = DATA / video
        assert len(list(source.glob("*.jpg"))) == 150
        dest = OUT / video / "live"
        cache_id = "r98_" + video
        if not (dest / "run_meta.json").exists():
            assert not dest.exists() and not (cache_root / cache_id).exists()
            dest.parent.mkdir(exist_ok=True)
            cmd = [sys.executable, str(HOME / "nano_run/r96_nano_live_pf_experiment.py"),
                   "--source", str(source), "--fps", "25", "--video-name", video,
                   "--engine", str(ENGINE), "--onnx-sha256", onnx_sha,
                   "--tracker-yaml", "configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml",
                   "--out", str(dest), "--warmup-frames", "25", "--hang-doi", "4",
                   "--elapsed-mode", "skip", "--regions", str(source / "ignored_regions.txt"),
                   "--classes", "2", "3", "5", "7", "--save-cache", cache_id,
                   "--cache-root", str(cache_root)]
            dest.mkdir()
            (dest / "command.json").write_text(json.dumps(cmd, indent=2))
            with (dest / "stdout.log").open("w") as f:
                rc = subprocess.run(cmd, cwd=BASE, env=dict(os.environ, BYTETRACK_BACKEND="np"),
                                    stdout=f, stderr=subprocess.STDOUT).returncode
            if rc:
                raise RuntimeError(f"capture {video} failed exit {rc}: {dest}/stdout.log")
        meta = json.loads((dest / "run_meta.json").read_text())
        assert meta["hoan_tat"] and meta["dem"]["n_frame_da_xu_ly"] == 150
        assert BOOT.read_text() == boot and ERRORS.read_text() == errors and swap_mb() - swap0 <= 100
        print(video, "CACHE_OK", flush=True)

        if video == VIDEOS[0]:
            sys.path.insert(0, str(BASE / "src"))
            os.environ["BYTETRACK_BACKEND"] = "np"
            import det_cache
            import run_tracking
            import bytetrack_np
            import tracker_ctrv
            old_registry = bytetrack_np.tracker_registry_np
            def registry():
                reg = old_registry()
                reg["bytetrack_pf_ctrv"] = tracker_ctrv.PFByteTracker
                return reg
            bytetrack_np.tracker_registry_np = registry
        det_cache.CACHE_ROOT = cache_root
        vd = det_cache.open_cache(cache_id, video, .25, [2, 3, 5, 7])
        cfg = run_tracking.load_tracker_cfg(BASE / "configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml",
                                            {"track_buffer": 30})
        tr = run_tracking.make_tracker(cfg)
        profiler = cProfile.Profile()
        frame_rows = []
        for frame, _t, unreadable, arr in vd.iter_frames():
            assert not unreadable
            det = det_cache.to_boxes_np(arr, vd.orig_shape)
            start = time.perf_counter()
            profiler.enable()
            tr.update(det, None, n_steps=1)
            profiler.disable()
            frame_rows.append({"frame": frame, "detections": len(arr),
                               "tracked": len(tr.tracked_stracks), "lost": len(tr.lost_stracks),
                               "update_ms": (time.perf_counter() - start) * 1000})
        assert len(frame_rows) == 150
        stats = pstats.Stats(profiler)
        funcs = [{"file": key[0], "line": key[1], "function": key[2], "calls": val[1],
                  "self_s": val[2], "cum_s": val[3]} for key, val in stats.stats.items()]
        funcs.sort(key=lambda x: x["cum_s"], reverse=True)
        summary = {"video": video, "n_frames": len(frame_rows),
                   "update_median_ms": float(np.median([r["update_ms"] for r in frame_rows])),
                   "detections_median": float(np.median([r["detections"] for r in frame_rows])),
                   "tracked_median": float(np.median([r["tracked"] for r in frame_rows])),
                   "lost_median": float(np.median([r["lost"] for r in frame_rows])),
                   "top_functions": funcs[:60], "frames": frame_rows}
        (OUT / video / "profile.json").write_text(json.dumps(summary, indent=2))
        print(video, "PROFILE_OK", summary["update_median_ms"], flush=True)
    print("R98_DONE", flush=True)


if __name__ == "__main__":
    main()
