"""Run the deployed engine, YOLO11n 50/50, and early cutoff on the same clips."""
from argparse import ArgumentParser
from collections import defaultdict
import json
import os
from pathlib import Path
from shutil import copytree, disk_usage
from statistics import median
import sys
from tempfile import TemporaryDirectory


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("r105_source", type=Path, help="Folder containing the R105 benchmark and drivers")
    parser.add_argument("output", type=Path, help="New output directory on the Nano")
    parser.add_argument("--clips-per-dataset", type=int, choices=(2, 8, 40), default=2)
    parser.add_argument("--matrix-package", type=Path)
    parser.add_argument("--decode-package", type=Path, help="Compare matrix NMS with and without JPEG overlap")
    parser.add_argument("--compare-v21combo", action="store_true", help="Compare v21combo directly against JPEG overlap")
    args = parser.parse_args()
    if args.compare_v21combo and not args.decode_package:
        parser.error("--compare-v21combo requires --decode-package")
    source_dir = args.r105_source.resolve(strict=True)
    sys.path[:0] = [str(source_dir), str(source_dir / "drivers")]
    import bench_nano as benchmark
    import r86_driver as guard
    import r101_driver as tools
    import r102_driver as live

    assert Path(benchmark.__file__).parent == source_dir
    package = benchmark.OUT / "nano_transfer/src"
    detector = package / "nano_detect_stream.py"
    pipeline = package / "nano_live.py"
    assert "R103_EARLY_CUTOFF" in pipeline.read_text()
    files = [detector, pipeline, package / "run_tracking.py"]
    modes = ("v21combo", "full", "early")
    if args.matrix_package:
        matrix = args.matrix_package.resolve(strict=True)
        detector_matrix = matrix / "src/nano_detector.py"
        assert "def nms_matrix(" in detector_matrix.read_text()
        files.append(detector_matrix)
        modes += ("matrix",)
    if args.decode_package:
        assert args.matrix_package
        decode = args.decode_package.resolve(strict=True)
        files.append(decode / "src/nano_detect_stream.py")
        assert benchmark.sha(decode / "src/nano_detector.py") == benchmark.sha(detector_matrix)
        modes = ("v21combo" if args.compare_v21combo else "matrix", "overlap")
    code_hashes = {str(path): benchmark.sha(path) for path in files}
    live.ENGINES = {"A": guard.ENGINES["B"], "B": benchmark.ANCHOR / "yolo11n_interp50_fp16.engine"}
    onnx = {"A": guard.ONNX["B"], "B": benchmark.ANCHOR / "yolo11n_interp50.onnx"}
    onnx_hash = {key: benchmark.sha(path) for key, path in onnx.items()}
    engine_hash = {key: benchmark.sha(path) for key, path in live.ENGINES.items()}
    assert onnx_hash == benchmark.EXPECTED_ONNX
    assert engine_hash == benchmark.EXPECTED_ENGINE

    manifest = json.loads((benchmark.SAMPLES / "sample_manifest.json").read_text())["rows"]
    count = args.clips_per_dataset
    indices = (0, 10) if count == 2 else range(0, 40, 40 // count)
    samples = []
    for dataset in ("bdd", "ua"):
        group = [row for row in manifest if row["dataset"] == dataset]
        assert len(group) == 40
        samples.extend(group[i] for i in indices)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    boot = guard.read("/proc/sys/kernel/random/boot_id")
    swap = guard.swap_mb()
    card_errors = "/sys/fs/ext4/mmcblk1p1/errors_count"
    errors = guard.read(card_errors)
    assert errors == "0"
    benchmark.write(root, "inputs.json", {"videos": samples, "modes": modes, "engine_sha256": engine_hash,
                    "onnx_sha256": onnx_hash, "package_sha256": code_hashes, "boot": boot,
                    "swap_mb": swap, "swap_activity": tools.swap_activity()})

    run_one = live.run_one
    def enable_cutoff(*params):
        os.environ["R103_EARLY_CUTOFF"] = "1"
        return run_one(*params)

    def enable_matrix(*params):
        live.PACKAGE = matrix
        return enable_cutoff(*params)

    def enable_overlap(*params):
        live.PACKAGE = decode
        return enable_cutoff(*params)

    run_mode = {"v21combo": run_one, "full": run_one, "early": enable_cutoff,
                "matrix": enable_matrix, "overlap": enable_overlap}
    reference_mode = modes[0]
    parity_reference = "matrix" if args.decode_package else "full"
    # Only expect byte parity for two modes using the same detector checkpoint.
    candidates = ("overlap",) if args.decode_package and not args.compare_v21combo else modes[2:]
    results = []
    clip_pairs = defaultdict(list)
    try:
        for index, row in enumerate(samples):
            dataset, name = row["dataset"], row["name"]
            original = benchmark.SAMPLES / dataset / name
            needed = sum(path.stat().st_size for path in original.iterdir() if path.is_file())
            assert needed < disk_usage("/dev/shm").free / 4
            # Rotate the first runner to spread warmup and ordering effects.
            first = index % len(modes)
            with TemporaryDirectory(prefix="r108_nano_", dir="/dev/shm") as tmp:
                source = Path(tmp) / name
                copytree(original, source)
                images = sorted(source.glob("*.jpg"))
                assert len(images) == 150 and images[0].name == row["first_file"] and images[-1].name == row["last_file"]
                hashes = {path.name: benchmark.sha(path) for path in images}
                benchmark.write(source, "r105_images.json", hashes)
                benchmark.write(root, dataset + "_" + name + "_images.json", hashes)
                measured = {}
                for mode in modes[first:] + modes[:first]:
                    guard.check_guard(boot, swap, errors)
                    mode_root = root / mode
                    live.run_one = run_mode[mode]
                    _, _, destination, item = benchmark.run_mode(
                        mode_root, "v21combo" if mode == "v21combo" else "full", row, source, live, guard, onnx_hash)
                    assert item["n_inferred"] == 150
                    meta = json.loads((destination / "run_meta.json").read_text())
                    command = meta["lenh_con"]
                    confidence = float(command[command.index("--conf") + 1])
                    assert confidence == (.25 if mode in ("early", "matrix", "overlap") else .05), (mode, confidence)
                    guard.check_guard(boot, swap, errors)
                    item.update(mode=mode, pre_nms_confidence=confidence)
                    measured[mode] = item
                    results.append(item)
                    benchmark.write(root, "pairs.json", results)
                    print(json.dumps(item), flush=True)
                parity = {mode: all(measured[parity_reference][field] == measured[mode][field]
                               for field in ("mot_sha256", "cls_sha256")) for mode in candidates}
                clip_pairs[dataset].append({"name": name, "parity": parity,
                    **{mode: measured[mode]["fps_steady"] for mode in modes}})
                benchmark.write(root, "per_clip.json", clip_pairs)
        summary = {}
        for dataset, clips in clip_pairs.items():
            summary[dataset] = {mode: {"median_fps": median(row[mode] for row in clips),
                "clips_at_15": sum(row[mode] >= 15 for row in clips),
                "paired_gain_pct": 100 * median(row[mode] / row[reference_mode] - 1 for row in clips)}
                for mode in modes}
            for mode in candidates:
                summary[dataset][mode + "_parity_clips"] = sum(row["parity"][mode] for row in clips)
        assert all(benchmark.sha(Path(path)) == value for path, value in code_hashes.items())
        benchmark.write(root, "summary.json", summary)
        print("NANO_15_COMPLETE", len(results), flush=True)
    finally:
        live.run_one = run_one
        for key in benchmark.ENV_KEYS:
            os.environ.pop(key, None)
        benchmark.write(root, "after.json", {"boot": guard.read("/proc/sys/kernel/random/boot_id"),
            "swap_mb": guard.swap_mb(), "swap_activity": tools.swap_activity(), "errors": guard.read(card_errors)})


if __name__ == "__main__":
    main()
