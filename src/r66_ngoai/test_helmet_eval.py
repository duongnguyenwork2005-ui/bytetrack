"""CPU-only checks of R66 per-clip resumption and parity with run_tracking.py."""
import sys
import tempfile
import unittest
import datetime as dt
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import helmet_eval as live


def fixture_arrays():
    # Motorcycle describes a turn, disappears for four frames, and reappears.
    rows = []
    counts = []
    for i in range(100):
        start = len(rows)
        if i not in range(41, 45):
            x = 120 + min(55, i) * 2
            y = 160 + max(0, i - 55) * 2
            rows.append([x, y, x + 60, y + 80, .9, 3])
        if i % 8 == 0:
            rows.append([800, 400, 830, 460, .95, 2])
        counts.append(len(rows) - start)
    data = np.asarray(rows, np.float32)
    return {"frame_idx": np.arange(1, 101), "time_s": np.arange(100) / 10,
            "unreadable": np.zeros(100, bool), "truncated": np.zeros(100, bool),
            "n_per_frame": np.array(counts), "data": data, "orig_shape": np.array([1080, 1920]),
            "cls_goc": data[:, 5].copy()}


class HelmetEvalTest(unittest.TestCase):
    def test_deadline_refuses_oversized_or_exhausted_window(self):
        for minutes, error in ((121, live.r64.LoiHopDong), (10, live.r64.HetNganSach)):
            when = (dt.datetime.now().astimezone() + dt.timedelta(minutes=minutes)).isoformat()
            with patch.object(sys, "argv", ["runner", "cache", "--phase", "smoke", "--deadline", when]), self.assertRaises(error):
                live.main()

    def test_cache_guards(self):
        data = fixture_arrays()
        live.check_arrays(data, None)
        for name, idx, value in [("data", (0, 0), float("nan")), ("data", (0, 5), 2),
                                 ("time_s", 1, 0.12), ("n_per_frame", 0, 3000),
                                 ("unreadable", 21, True)]:
            with self.subTest(name=name):
                changed = dict(data, **{name: data[name].copy()})
                changed[name][idx] = value
                with self.assertRaises(live.r64.LoiHopDong):
                    live.check_arrays(changed, None)

    def test_backend_parity_and_resume(self):
        fixture_root = live.r66.GOC / "tmp"
        fixture_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="helmet_cpu_fixture_", dir=fixture_root) as tmp:
            root = Path(tmp)
            cache_root = root / "interim" / "det_cache"
            cache_id = "cpu-helmet-fixture"
            clip = "cpu_clip"
            folder = cache_root / cache_id
            spec = live.det_cache.DetectorSpec(backend="gt-synthetic", dataset="helmet")
            live.det_cache.write_meta(folder, spec, "create")
            live.det_cache.save_video(folder, clip, fixture_arrays(), spec=spec)
            phase = SimpleNamespace(phase="smoke", split="HELMET-fixture", log=root / "log", clips=[clip],
                fingerprint="fixture", backend=live.fp.code_fingerprint(), inputs={clip: {}},
                tracker_registry=live.rt.tracker_registry(), cache_id=lambda _: cache_id, gate=lambda *args: None,
                recheck_sources=lambda: None)
            sequence = root / "processed" / phase.split / clip
            (sequence / "gt").mkdir(parents=True)
            (sequence / "gt/gt.txt").write_text("".join(
                f"{i},1,{120+2*min(55,i-1)},{160+2*max(0,i-1-55)},60,80,1,1,1\n" for i in range(1, 101)),
                encoding="utf-8")
            (sequence / "seqinfo.ini").write_text("[Sequence]\nseqLength=100\n", encoding="utf-8")
            with (patch.object(live.config, "INTERIM_DIR", cache_root.parent),
                  patch.object(live.det_cache, "CACHE_ROOT", cache_root),
                  patch.object(live.config, "PROCESSED_DIR", root / "processed"),
                  patch.object(live.config, "RESULTS_DIR", root / "results"),
                  patch.object(live, "check_cache", return_value="fixture-cache"),
                  contextlib.redirect_stdout(io.StringIO())):
                trackers = live.config.PROCESSED_DIR / "trackers" / phase.split
                for loc, (yaml, seeds) in live.r64.BO_LOC.items():
                    seed = 0 if seeds else None
                    live.tracking(phase, "yolov8n_mixed5", "c25", loc, seed)
                    name = live.run_name("yolov8n_mixed5", "c25", loc, seed)
                    args = ["run_tracking.py", "--cache-id", cache_id, "--split-name", phase.split,
                        "--tracker-yaml", yaml, "--tracker-name", "control-" + loc, "--videos", clip,
                        "--conf-min", "0.25", "--classes", "3", "--max-det", "300", "--fps", "10",
                        "--track-buffer-sec", "1.2", "--elapsed-mode", "steps"]
                    if seeds:
                        args += ["--seeds", "0"]
                    with patch.object(sys, "argv", args):
                        self.assertEqual(live.rt.main(), 0)
                    control = trackers / ("control-" + loc + ("-s0" if seeds else ""))
                    self.assertEqual((trackers / name / "data" / f"{clip}.txt").read_bytes(),
                                     (control / "data" / f"{clip}.txt").read_bytes())
                    with patch.object(live.rt, "run_video_to_dir", side_effect=AssertionError("re-ran clip")):
                        live.tracking(phase, "yolov8n_mixed5", "c25", loc, seed)
                live.repeat_pf(phase)
                self.assertEqual(live.read(phase.log / "repeat_pf.json")["status"], "PASS")
                with (trackers / name / "data" / f"{clip}.txt").open("ab") as f:
                    f.write(b"invalid")
                with self.assertRaises(live.r64.LoiHopDong):
                    live.tracking(phase, "yolov8n_mixed5", "c25", loc, seed)


if __name__ == "__main__":
    unittest.main()
