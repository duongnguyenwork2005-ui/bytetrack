"""Create an isolated Nano package with an explicit skipped-inference frame flag."""
from pathlib import Path
import shutil

HOME = Path.home()
BASE = HOME / "nano_run/r63_20261030/v21combo/nano_transfer"
PACKAGE = HOME / "nano_run/r103_optimization/nano_transfer"


def replace(name, original, substitute):
    path = PACKAGE / "src" / name
    text = path.read_text()
    if text.count(original) != 1:
        raise ValueError(f"Anchor in {name} expected exactly once")
    path.write_text(text.replace(original, substitute, 1))


def main():
    PACKAGE.mkdir(exist_ok=False)
    shutil.copytree(BASE / "src", PACKAGE / "src")
    for entry in BASE.iterdir():
        if entry.name != "src":
            (PACKAGE / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())

    replace("nano_stream_proto.py", "CO_UNREADABLE, CO_TRUNCATED = 1, 2",
            "CO_UNREADABLE, CO_TRUNCATED, CO_INFER_SKIPPED = 1, 2, 4")
    replace("nano_detect_stream.py", '    n_gui, ly_do = 0, "het_nguon"',
            '    cadence = int(os.environ.get("R103_CADENCE", "1"))\n'
            '    if cadence not in (1, 2):\n'
            '        raise ValueError("Unsupported detection cadence")\n'
            '    n_gui, ly_do = 0, "het_nguon"')
    replace("nano_detect_stream.py", "                co |= proto.CO_UNREADABLE\n            else:\n                t1 = time.perf_counter()",
            "                co |= proto.CO_UNREADABLE\n"
            "            elif (idx - 1) % cadence:\n"
            "                d = np.zeros((0, 6), np.float32)\n"
            "                co |= proto.CO_INFER_SKIPPED\n"
            "            else:\n                t1 = time.perf_counter()")

    replace("nano_live.py", '            row = {"frame": idx, "time_s": float(msg["time_s"]), "unreadable": unreadable,',
            '            infer_skipped = bool(msg["co"] & proto.CO_INFER_SKIPPED)\n'
            '            if infer_skipped and (unreadable or len(det)):\n'
            '                raise ValueError("Invalid skipped-inference frame")\n'
            '            row = {"frame": idx, "time_s": float(msg["time_s"]), "unreadable": unreadable,\n'
            '                   "infer_skipped": infer_skipped,')
    replace("nano_live.py", '                yield idx, float(msg["time_s"]), unreadable, arr',
            '                yield idx, float(msg["time_s"]), unreadable, (None if infer_skipped else arr)')
    replace("nano_live.py", "    a = ap.parse_args()\n",
            '    a = ap.parse_args()\n'
            '    if os.environ.get("R103_EARLY_CUTOFF") == "1":\n'
            '        a.det_conf = a.conf_min\n'
            '    if int(os.environ.get("R103_CADENCE", "1")) != 1 and a.save_cache:\n'
            '        raise ValueError("Scheduled skip requires an explicit flag; cache export is unavailable")\n')

    replace("run_tracking.py", 'def _vong_lap_tracker(gen, vd, tr, timer, elapsed_mode, dt_nominal, rows, cls_rows, timing, on_frame):',
            'def _predict_without_observation(tracker, steps):\n'
            '    """Advance time without treating a scheduled skip as a detector miss."""\n'
            '    _, active = tracker._split_tracked()\n'
            '    pool = {}\n'
            '    for track in active + tracker.lost_stracks:\n'
            '        pool.setdefault(track.track_id, track)\n'
            '    tracker.frame_id += steps\n'
            '    for _ in range(steps):\n'
            '        tracker.multi_predict(list(pool.values()))\n'
            '    return tracker._format_output()\n\n\n'
            'def _vong_lap_tracker(gen, vd, tr, timer, elapsed_mode, dt_nominal, rows, cls_rows, timing, on_frame):')
    replace("run_tracking.py",
            '        dets = (det_cache.to_boxes_np(arr, vd.orig_shape) if _BACKEND_NP\n'
            '                else det_cache.to_boxes(arr, vd.orig_shape).cpu().numpy())\n'
            '        tr.log_frame = int(frame_idx)      # chi so frame cua video (cho assoc_log / pf_diag)\n'
            '        t0 = time.perf_counter()\n        out = tr.update(dets, None, n_steps=n_steps)',
            '        tr.log_frame = int(frame_idx)\n'
            '        if arr is None:\n'
            '            t0 = time.perf_counter()\n'
            '            out = _predict_without_observation(tr, n_steps)\n'
            '        else:\n'
            '            dets = (det_cache.to_boxes_np(arr, vd.orig_shape) if _BACKEND_NP\n'
            '                    else det_cache.to_boxes(arr, vd.orig_shape).cpu().numpy())\n'
            '            t0 = time.perf_counter()\n'
            '            out = tr.update(dets, None, n_steps=n_steps)')
    replace("run_tracking.py", "        timing.append((frame_idx, n_steps, n_tracks, len(arr),",
            "        timing.append((frame_idx, n_steps, n_tracks, 0 if arr is None else len(arr),")

    replace("nano_detector.py", '        self.t = {"pre_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0, "n": 0}',
            '        self.roi_y = int(os.environ["R103_ROI_Y"]) if "R103_ROI_Y" in os.environ else None\n'
            '        self.t = {"pre_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0, "n": 0}')
    replace("nano_detector.py", "        lb, r, pad = letterbox(img_bgr, self.imgsz)",
            "        if self.roi_y is not None:\n"
            "            if tuple(self.imgsz) != (256, 640) or img_bgr.shape[:2] != (540, 960):\n"
            "                raise ValueError(\"Unexpected engine or source shape for fixed ROI\")\n"
            "            if not 0 <= self.roi_y <= 156:\n"
            "                raise ValueError(\"ROI outside source frame\")\n"
            "            img_bgr = img_bgr[self.roi_y:self.roi_y + 384, :]\n"
            "        lb, r, pad = letterbox(img_bgr, self.imgsz)")
    replace("nano_detector.py", "        det = scale_boxes(det, lb.shape[:2], img_bgr.shape[:2])\n        t3 = time.perf_counter()",
            "        det = scale_boxes(det, lb.shape[:2], img_bgr.shape[:2])\n"
            "        if self.roi_y is not None:\n"
            "            det[:, [1, 3]] += self.roi_y\n"
            "        t3 = time.perf_counter()")
    print(f"Isolated Nano package ready: {PACKAGE}")


if __name__ == "__main__":
    main()
