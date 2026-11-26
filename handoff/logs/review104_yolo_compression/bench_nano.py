"""Compare a compressed checkpoint's engine with the R102 engine on the same eight clips."""
import argparse
import json
from pathlib import Path
from shutil import copytree, disk_usage
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path.home() / "nano_run/r104_yolo_compression"
PRIOR = ROOT.parent / "r103_optimization"
sys.path.insert(0, str(PRIOR))
import bench_nano as previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    args = parser.parse_args()
    mode = args.candidate
    output = ROOT / f"bench_{mode}"
    output.mkdir(exist_ok=False)
    candidate = ROOT / f"{mode}.engine"
    engines = {"full": previous.engine("full"), mode: candidate}
    onnx_paths = {"full": previous.onnx("full"), mode: ROOT / f"{mode}.onnx"}
    previous.engine = engines.__getitem__
    previous.live.PACKAGE = PRIOR / "nano_transfer"
    previous.tools.OUT = output
    previous.guard.ENGINES = {"A": engines["full"], "B": candidate}
    rows = previous.sample_rows()
    guard = previous.guard
    onnx_hash = {name: guard.sha(path) for name, path in onnx_paths.items()}
    boot = guard.read("/proc/sys/kernel/random/boot_id")
    swap = guard.swap_mb()
    errors = guard.read("/sys/fs/ext4/mmcblk1p1/errors_count")
    assert errors == "0" and guard.sha(engines["full"]) == previous.BASELINE_ENGINE_SHA
    previous.write(output, "videos.json", rows)
    previous.write(output, "inputs.json", {"boot": boot, "swap_mb": swap,
                  "swap_activity": previous.tools.swap_activity(), "onnx_sha256": onnx_hash,
                  "engines": {key: {"sha256": guard.sha(path), "bytes": path.stat().st_size}
                              for key, path in engines.items()}})
    previous.tools.benchmark_engines()
    results, files = [], []
    for index, row in enumerate(rows):
        dataset, name = row["dataset"], row["name"]
        original = previous.SAMPLES / dataset / name
        needed = sum(path.stat().st_size for path in original.iterdir() if path.is_file())
        assert needed < disk_usage("/dev/shm").free / 4
        with TemporaryDirectory(prefix="r104_", dir="/dev/shm") as temp:
            source = Path(temp) / name
            copytree(original, source)
            images = sorted(source.glob("*.jpg"))
            assert len(images) == 150 and images[0].name == row["first_file"] and images[-1].name == row["last_file"]
            order = ("full", mode) if index % 2 == 0 else (mode, "full")
            for version in order:
                guard.check_guard(boot, swap, errors)
                mot, item = previous.run_mode(output, version, row, source, onnx_hash)
                guard.check_guard(boot, swap, errors)
                files.append((mot, f"{dataset}/{name}/{version}.txt"))
                results.append(item)
                previous.write(output, "pairs.json", results)
                print(json.dumps(item), flush=True)
    assert len(results) == 16
    with tarfile.open(output / "mot_outputs.tar.gz", "w:gz") as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
    previous.write(output, "after.json", {"boot": boot, "swap_mb": guard.swap_mb(),
                   "swap_activity": previous.tools.swap_activity(),
                   "errors": guard.read("/sys/fs/ext4/mmcblk1p1/errors_count")})


if __name__ == "__main__":
    main()
