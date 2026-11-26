"""CPU-only protocol/integration tests; never substitutes for a Nano benchmark."""
import ast
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import subprocess
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRATCH = HERE / "pc_test_scratch"
SCRATCH.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import cadence
import prepare_nano
import bench_nano as bench
import eval_benchmark as assess


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CadenceTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((90, 160, 3), np.uint8)
        self.box = np.array([[40, 30, 60, 50, .9, 0]], np.float32)

    def test_static_schedule_150(self):
        scheduler = cadence.AdaptiveCadence((90, 160), [], False)
        skipped = []
        for i in range(1, 151):
            if scheduler.should_skip(i, self.image):
                skipped.append(i)
            else:
                scheduler.observe(self.box)
        self.assertEqual(skipped, list(range(2, 151, 2)))
        self.assertEqual(sum(scheduler.counts.values()), 150)

    def test_motion_threshold_and_known_padding(self):
        for pixels, expected in ((11, True), (12, False)):
            scheduler = cadence.AdaptiveCadence((90, 160), [], False)
            scheduler.should_skip(1, self.image)
            changed = self.image.copy()
            changed[0, :6] = 255
            changed[1, :pixels-6] = 255
            self.assertEqual(scheduler.should_skip(2, changed), expected)
        scheduler = cadence.AdaptiveCadence((90, 160), [], False)
        scheduler.should_skip(1, self.image)
        scheduler.observe(self.box)
        changed = self.image.copy()
        changed[26:54, 36:64] = 255
        self.assertTrue(scheduler.should_skip(2, changed))

    def test_ignore_region(self):
        scheduler = cadence.AdaptiveCadence((90, 160), [(0, 0, 10, 10)], False)
        scheduler.should_skip(1, self.image)
        changed = self.image.copy()
        changed[:10, :10] = 255
        self.assertTrue(scheduler.should_skip(2, changed))

    def test_lost_internal_boundary_and_cutoff(self):
        scheduler = cadence.AdaptiveCadence((90, 160), [], True)
        scheduler.should_skip(1, self.image)
        scheduler.observe(self.box)
        scheduler.observe(np.empty((0, 6), np.float32))
        self.assertFalse(scheduler.should_skip(2, self.image))
        scheduler.reset()
        scheduler.observe(np.array([[0, 0, 10, 10, .9, 0]], np.float32))
        scheduler.observe(np.empty((0, 6), np.float32))
        self.assertFalse(scheduler.lost)
        scheduler.observe(self.box * np.array([1, 1, 1, 1, .25/.9, 1]))
        self.assertEqual(len(scheduler.observed), 0)

    def test_low_fps_and_invalid_combinations(self):
        for mode in ("motion", "guarded"):
            self.assertIsNone(cadence.make_scheduler(mode, 1, 5, (90, 160), []))
            self.assertIsNotNone(cadence.make_scheduler(mode, 1, 25, (90, 160), []))
        for mode, stride in (("unknown", 1), ("guarded", 2), ("off", 3)):
            with self.assertRaises(ValueError):
                cadence.make_scheduler(mode, stride, 25, (90, 160), [])


class RunnerTests(unittest.TestCase):
    def test_evaluator_cold_cli_import(self):
        result = subprocess.run([sys.executable, "-B", str(HERE / "eval_benchmark.py"), "--help"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_matrix_and_hash_lengths(self):
        videos = json.loads((HERE.parent / "review103_nano_optimization/pipeline/videos.json").read_text())
        matrix = [(v["dataset"], v["name"], m) for i, v in enumerate(videos) for m in bench.modes_for(v, i)]
        self.assertEqual(len(matrix), 30)
        self.assertEqual(sum(ds == "bdd" and m == "guarded" for ds, _, m in matrix), 2)
        self.assertEqual(sum(ds == "ua" for ds, _, _ in matrix), 20)
        self.assertTrue(all(len(h) == 64 for h in [*bench.EXPECTED_ONNX.values(), *bench.EXPECTED_ENGINE.values()]))

    def test_frame_gates_reject_corruption(self):
        rows = [dict(frame=str(i), infer_skipped=str(i % 2 == 0), unreadable="False") for i in range(1, 151)]
        self.assertEqual(len(bench.frame_checks(rows, "stride2", "ua")), 75)
        for mode, ds in (("full", "ua"), ("guarded", "bdd")):
            with self.assertRaises(ValueError):
                bench.frame_checks(rows, mode, ds)
        duplicate = copy.deepcopy(rows)
        duplicate[-1]["frame"] = "149"
        with self.assertRaises(ValueError):
            bench.frame_checks(duplicate, "stride2", "ua")
        odd = copy.deepcopy(rows)
        odd[0]["infer_skipped"] = "True"
        with self.assertRaises(ValueError):
            bench.frame_checks(odd, "motion", "ua")

    def test_paired_ratio_not_ratio_of_medians(self):
        samples = {}
        for name, a, b in (("x", 10, 30), ("y", 20, 20), ("z", 100, 110)):
            for mode, fps in (("full", a), ("motion", b)):
                samples[("ua", name, mode)] = dict(dataset="ua", name=name, mode=mode, fps_steady=fps,
                    p90_latency_ms=2, n_inferred=150, decode_ms=1, infer_ms=1, mot_sha256="a", cls_sha256="b")
        self.assertAlmostEqual(assess.paired_speed(samples, "motion", "full")["ua"]["paired_fps_gain_pct"], 10)

    def test_live_driver_mode_wiring_and_fallback(self):
        previous = {k: os.environ.get(k) for k in bench.ENV_KEYS}
        try:
            with TemporaryDirectory(prefix="r105_wiring_", dir=str(SCRATCH)) as tmp:
                root = Path(tmp)
                source = root / "images"
                source.mkdir()
                (source / "r105_images.json").write_text("{}")
                engine = root / "engine"
                engine.write_bytes(b"fixture")
                row = dict(dataset="bdd", name="fixture", n_frames=150)
                guard = SimpleNamespace(PACKAGES={"B": root / "deployed-v21combo"})
                live = SimpleNamespace(ENGINES={"A": engine, "B": engine})
                enabled = [False]
                def run_one(row, version, source, hashes):
                    self.assertEqual(version, "B")
                    self.assertEqual(os.environ["R105_MODE"], "guarded")
                    self.assertNotIn("R103_ROI_Y", os.environ)
                    dest = live.OUT / "fixture"
                    (dest / "data").mkdir(parents=True)
                    mot = dest / "data/fixture.txt"
                    mot.write_bytes(b"1,1,0,0,10,10,1,-1,-1,-1\n")
                    (dest / "data_cls").mkdir()
                    (dest / "data_cls/fixture.csv").write_text("fixture")
                    (dest / "pipeline").mkdir()
                    (dest / "pipeline/fixture.csv").write_text("frame,infer_skipped,unreadable\n"+
                        "".join(str(i)+",False,False\n" for i in range(1, 151)))
                    (dest / "run_meta.json").write_text(json.dumps(dict(eos_con=dict(adaptive=dict(
                        enabled=enabled[0], counts={}, mode="guarded", cpu_ms=0)))))
                    return mot, dict(dataset="bdd", name="fixture")
                live.run_one = run_one
                os.environ["R103_ROI_Y"] = "77"
                _, _, _, record = bench.run_mode(root / "valid", "guarded", row, source, live, guard, {})
                self.assertEqual(record["n_inferred"], 150)
                self.assertFalse(record["adaptive"]["enabled"])
                enabled[0] = True
                with self.assertRaises(ValueError):
                    bench.run_mode(root / "bad", "guarded", row, source, live, guard, {})
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_driver_copies(self):
        manifest = json.loads((HERE / "driver_manifest.json").read_text())
        for name, record in manifest.items():
            self.assertEqual(bench.sha(HERE / "drivers" / name), record["sha256"])
            self.assertEqual(bench.sha(ROOT / record["source"]), record["sha256"])


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory(prefix="r105_test_", dir=str(SCRATCH))
        cls.root = Path(cls.temp.name)
        base = cls.root / "v21fixture"
        (base / "src").mkdir(parents=True)
        for name in ("nano_stream_proto.py", "nano_detect_stream.py", "nano_live.py", "run_tracking.py", "nano_detector.py"):
            (base / "src" / name).write_bytes((ROOT / "src" / name).read_bytes())
        cls.original = {p.name: bench.sha(p) for p in (base / "src").iterdir()}
        r103 = load("r103_prepare", HERE.parent / "review103_nano_optimization/prepare_nano.py")
        r103.BASE, r103.PACKAGE = base, cls.root / "r103"
        with redirect_stdout(io.StringIO()):
            r103.main()
        prepare_nano.BASE, prepare_nano.PACKAGE = r103.PACKAGE, cls.root / "r105"
        with redirect_stdout(io.StringIO()):
            prepare_nano.main()
        cls.source = (prepare_nano.PACKAGE / "src/nano_detect_stream.py").read_text()
        cls.base = base
        cls.old_env = {k: os.environ.get(k) for k in bench.ENV_KEYS}
        cls.old_module = sys.modules.get("r105_cadence")
        sys.modules["r105_cadence"] = cadence

    @classmethod
    def tearDownClass(cls):
        for key, value in cls.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if cls.old_module is None:
            sys.modules.pop("r105_cadence", None)
        else:
            sys.modules["r105_cadence"] = cls.old_module
        cls.temp.cleanup()

    def run_stream(self, mode, fps, detections=None, motion=False):
        for key in bench.ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["R105_MODE"] = mode
        if mode == "stride2":
            os.environ["R105_MODE"] = "off"
            os.environ["R103_CADENCE"] = "2"
        class Source:
            orig_shape = (90, 160)
            n_frames_khai_bao = 150
            def __iter__(self):
                for i in range(1, 151):
                    image = np.zeros((90, 160, 3), np.uint8)
                    if motion and i >= 2:
                        image[:10, :10] = 255
                    yield i, i/fps, image, 0., time.monotonic()
            def dong(self):
                pass
        source = Source()
        source.fps = fps
        channel = SimpleNamespace(cho_bat_dau=lambda: 0, dung=False, cha_dong=False,
            kiem_khong_chan=lambda: None, cho_cho_trong=lambda: 0, dang_bay=0, max_bay=0,
            n_ket_qua=0, bo_doc=SimpleNamespace(n_byte_la=0))
        calls = []
        box = np.array([[40, 30, 60, 50, .9, 0]], np.float32)
        def detect(img):
            calls.append(len(calls)+1)
            return (detections(len(calls)) if detections else box.copy()), {}
        messages = []
        proto = SimpleNamespace(CO_UNREADABLE=1, CO_TRUNCATED=2, CO_INFER_SKIPPED=4, EOS=9, ERROR=10,
            dong_goi_frame=lambda idx, t_s, d, co, tm: dict(frame=idx, det=d.copy(), flags=co),
            dong_goi_json=lambda kind, obj: dict(kind=kind, obj=obj))
        args = SimpleNamespace(gia_lap_loi_o_frame=0, max_det=300, gia_lap_cham_ms=0,
            gia_lap_bo_frame=0, max_frames=0, hang_doi=4)
        namespace = dict(os=os, np=np, time=time, json=json, log=lambda *args: None, proto=proto,
                         nd=SimpleNamespace(mask_regions=lambda image, regions: image))
        function = next(n for n in ast.parse(self.source).body if isinstance(n, ast.FunctionDef) and n.name == "_vong_lap")
        exec(compile(ast.Module(body=[function], type_ignores=[]), "patched_stream", "exec"), namespace)
        code = namespace["_vong_lap"](args, source, channel, SimpleNamespace(detect=detect), None,
                      None, [], messages.append, SimpleNamespace(close=lambda: None), None, {})
        self.assertEqual(code, 0)
        frames = [m for m in messages if "frame" in m]
        self.assertEqual([m["frame"] for m in frames], list(range(1, 151)))
        return calls, frames, messages[-1]["obj"]

    def test_worker_static_motion_and_fixed_schedule(self):
        for mode in ("motion", "guarded", "stride2"):
            calls, frames, eos = self.run_stream(mode, 25)
            self.assertEqual(len(calls), 75)
            self.assertEqual([m["frame"] for m in frames if m["flags"] & 4], list(range(2, 151, 2)))
            self.assertTrue(all(len(m["det"]) == 0 for m in frames if m["flags"] & 4))
        calls, frames, eos = self.run_stream("motion", 25, motion=True)
        self.assertEqual(len(calls), 76)
        self.assertFalse(frames[1]["flags"] & 4)

    def test_worker_low_fps_exact_detection_parity(self):
        base_calls, base, _ = self.run_stream("off", 5)
        calls, fallback, eos = self.run_stream("guarded", 5)
        self.assertEqual(len(calls), len(base_calls))
        self.assertEqual(len(calls), 150)
        self.assertFalse(eos["adaptive"]["enabled"])
        for a, b in zip(base, fallback):
            self.assertEqual(a["flags"], b["flags"])
            np.testing.assert_array_equal(a["det"], b["det"])

    def test_worker_lost_guard_forces_even_inference(self):
        box = np.array([[40, 30, 60, 50, .9, 0]], np.float32)
        _, frames, eos = self.run_stream("guarded", 25,
                            detections=lambda n: np.empty((0, 6), np.float32) if n == 2 else box)
        self.assertFalse(frames[3]["flags"] & 4)
        self.assertGreater(eos["adaptive"]["counts"]["recently_lost"], 0)

    def test_stale_package_is_rejected(self):
        prepare_nano.validate_package()
        path = prepare_nano.PACKAGE / "src/r105_cadence.py"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n# stale copy\n")
            with self.assertRaises(ValueError):
                prepare_nano.validate_package()
        finally:
            path.write_bytes(original)

    def test_source_isolation_and_syntax(self):
        self.assertEqual({p.name: bench.sha(p) for p in (self.base / "src").iterdir()}, self.original)
        for name in ("nano_detect_stream.py", "r105_cadence.py"):
            ast.parse((prepare_nano.PACKAGE / "src" / name).read_text(), feature_version=(3, 6))
        with self.assertRaises(ValueError):
            prepare_nano.patch_stream("invalid integration source")


class MetricTests(unittest.TestCase):
    def test_timestamp_speed_and_latency_gate(self):
        frames = [dict(t_xong_cha=str(10+i*.1), t_bat_dau_con=str(10+i*.1-.02),
                       tre_ms="20", infer_ms="40", decode_ms="5", n_det_raw="1") for i in range(150)]
        summary = dict(warmup_bo=25, n_frame_tong_ket=125, fps_on_dinh=10,
                       tre_ms=dict(p90=20), infer_ms=dict(trung_vi=40), decode_ms=dict(trung_vi=5))
        self.assertAlmostEqual(assess.verify_timing(frames, summary)["fps_recalculated"], 10)
        # A metadata/CSV pair that agrees on a fabricated FPS must still be rejected.
        summary["fps_on_dinh"] = 11
        with self.assertRaises(AssertionError):
            assess.verify_timing(frames, summary)
        summary["fps_on_dinh"] = 10
        frames[80]["tre_ms"] = "21"
        with self.assertRaises(AssertionError):
            assess.verify_timing(frames, summary)

    def test_legacy_evidence_requires_exact_source_snapshot(self):
        with TemporaryDirectory(prefix="r105_legacy_", dir=str(SCRATCH)) as tmp:
            inputs = dict(code_sha256={"bench_nano.py": "unknown"})
            with self.assertRaises(AssertionError):
                assess.verify_legacy_snapshot(Path(tmp), inputs, inputs)
            inputs["code_sha256"]["bench_nano.py"] = assess.LEGACY_RUNNER_SHA
            inputs["code_sha256"]["cadence.py"] = bench.sha(HERE / "cadence.py")
            drivers = json.loads((HERE / "driver_manifest.json").read_text())
            inputs["code_sha256"].update({name: x["sha256"] for name, x in drivers.items()})
            inputs["package_sources"] = {"r105_cadence.py": dict(sha256=inputs["code_sha256"]["cadence.py"])}
            with self.assertRaises(FileNotFoundError):
                assess.verify_legacy_snapshot(Path(tmp), inputs, inputs)

    def test_first_match_new_id_delay(self):
        data = dict(num_gt_ids=2, gt_ids=[np.array([0]), np.array([0, 1]), np.array([0, 1])],
            tracker_ids=[np.array([8]), np.array([8]), np.array([8, 9])],
            similarity_scores=[np.array([[1.]]), np.array([[1.], [0.]]), np.eye(2)])
        seen, matched = assess.previous.first_matches(data)
        np.testing.assert_array_equal(seen, [0, 1])
        np.testing.assert_array_equal(matched, [0, 2])

    def test_trackeval_oracle_two_clip_fallback_subset(self):
        # R102 score() requires four clips; R105 must also score a two-clip subset.
        with TemporaryDirectory(prefix="r105_metric_", dir=str(SCRATCH)) as tmp:
            root = Path(tmp)
            videos = [dict(dataset="bdd", name="fixture1"), dict(dataset="bdd", name="fixture2")]
            for v in videos:
                split = assess.SPLITS["bdd"][0]
                gt = root / "gt" / split / v["name"] / "gt/gt.txt"
                gt.parent.mkdir(parents=True)
                gt.write_text("".join(str(i)+",1,10,10,20,20,1,1,1\n" for i in range(1, 151)))
                for version in "AB":
                    path = root / "trackers" / split / version / "data" / (v["name"]+".txt")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("".join(str(i)+",1,10,10,20,20,1,-1,-1,-1\n" for i in range(1, 151)))
            with redirect_stdout(io.StringIO()):
                result = assess.quality(root, videos)
            for version in "AB":
                self.assertEqual(result["bdd"][version]["combined"]["hota"], 1)
                self.assertEqual(result["bdd"][version]["combined"]["idsw"], 0)
                self.assertEqual(result["bdd"][version]["combined"]["fn"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
