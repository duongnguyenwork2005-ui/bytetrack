"""Benchmark the current and reduced YOLOv8n engines on the Jetson Nano."""
import json
from pathlib import Path
import re
from shutil import copytree, disk_usage
import subprocess
from tempfile import TemporaryDirectory

import r86_driver as r86

HOME = Path.home()
OUT = HOME / "nano_run/r101_slim_20261111"
SAMPLES = HOME / "nano_run/r86_multivideo_20261106/sample"
TRTEXEC = Path("/usr/src/tensorrt/bin/trtexec")
BASELINE_SHA256 = "a6912393829f8c811d3cc33008f3e4056a626434df327c85aae7beb33af01749"


def write(name, payload):
    (OUT / name).write_text(json.dumps(payload, indent=2))


def swap_activity():
    return {line.split()[0]: int(line.split()[1]) for line in Path("/proc/vmstat").read_text().splitlines()
            if line.startswith(("pswpin ", "pswpout "))}


def benchmark_engines():
    rows = []
    for i, version in enumerate("ABBA"):
        command = [str(TRTEXEC), "--loadEngine=" + str(r86.ENGINES[version]),
                   "--warmUp=500", "--duration=6", "--iterations=100"]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        (OUT / f"engine_{i}_{version}.log").write_text(result.stdout)
        median = re.search(r"GPU Compute Time:.*median = ([\d.]+) ms", result.stdout)
        throughput = re.search(r"Throughput: ([\d.]+) qps", result.stdout)
        assert median and throughput and "PASSED" in result.stdout
        rows.append({"version": version, "gpu_median_ms": float(median.group(1)), "qps": float(throughput.group(1))})
    write("engine_abba.json", rows)
    print(json.dumps(rows), flush=True)


def benchmark_pipeline(boot, swap0, mmc0):
    manifest = json.loads((SAMPLES / "sample_manifest.json").read_text())["rows"]
    rows = []
    for dataset in ("bdd", "ua"):
        group = [item for item in manifest if item["dataset"] == dataset]
        assert len(group) == 40
        rows.extend(group[i] for i in (0, 10, 20, 30))
    write("videos.json", rows)
    onnx_sha = {key: r86.sha(path) for key, path in r86.ONNX.items()}
    results = []
    for index, row in enumerate(rows):
        dataset, name = row["dataset"], row["name"]
        source = SAMPLES / dataset / name
        tmpfs = Path("/dev/shm")
        required = sum(path.stat().st_size for path in source.iterdir() if path.is_file())
        assert required < disk_usage(tmpfs).free / 4
        with TemporaryDirectory(prefix="r101_slim_", dir=tmpfs) as temp:
            r86.DATA = Path(temp)
            copytree(source, r86.DATA / dataset / name)
            for version in ("AB" if index % 2 == 0 else "BA"):
                r86.check_guard(boot, swap0, mmc0)
                result = r86.run_one(row, "cv", version, boot, swap0, mmc0, onnx_sha)
                meta = json.loads((r86.OUT / dataset / name / "cv" / version / "run_meta.json").read_text())
                result["infer_ms"] = meta["thoi_gian"]["tom_tat_sau_warmup"]["infer_ms"]["trung_vi"]
                mot = r86.OUT / dataset / name / "cv" / version / "data" / f"{name}.txt"
                result["mot_rows"] = len(mot.read_bytes().splitlines())
                result["swap_activity"] = swap_activity()
                results.append(result)
                write("pipeline_pairs.json", results)
                print(json.dumps(result), flush=True)
    assert len(results) == 16 and not any(item["skipped_complete"] for item in results)


def main():
    r86.ENGINES = {"A": r86.ENGINES["B"], "B": OUT / "yolov8n_c2f_slim_fp16.engine"}
    r86.PACKAGES = dict.fromkeys("AB", r86.PACKAGES["B"])
    r86.ONNX = {"A": r86.ONNX["B"], "B": OUT / "yolov8n_c2f_slim.onnx"}
    r86.OUT = OUT / "pipeline"
    boot = r86.read("/proc/sys/kernel/random/boot_id")
    mmc0 = r86.read("/sys/fs/ext4/mmcblk1p1/errors_count")
    swap0 = r86.swap_mb()
    assert mmc0 == "0" and r86.sha(r86.ENGINES["A"]) == BASELINE_SHA256
    activity_before = swap_activity()
    write("benchmark_inputs.json", {"boot": boot, "swap_mb": swap0,
                                   "swap_activity": activity_before,
                                   "engine_sha256": {key: r86.sha(path) for key, path in r86.ENGINES.items()},
                                   "onnx_sha256": {key: r86.sha(path) for key, path in r86.ONNX.items()}})
    benchmark_engines()
    r86.check_guard(boot, swap0, mmc0)
    benchmark_pipeline(boot, swap0, mmc0)
    write("benchmark_after.json", {"swap_mb": r86.swap_mb(), "swap_activity": swap_activity(),
                                  "mmc_errors": r86.read("/sys/fs/ext4/mmcblk1p1/errors_count")})
    print("R101_SLIM_PASS", flush=True)


if __name__ == "__main__":
    main()
