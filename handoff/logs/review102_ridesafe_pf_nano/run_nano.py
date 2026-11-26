"""Benchmark the five-class interpolation engine against the deployed COCO engine."""
import json
import os
from pathlib import Path
from shutil import copytree, disk_usage
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

import r86_driver as r86
import r101_driver as r101

HOME = Path.home()
OUT = HOME / "nano_run/r102_interp50_20261111"
SAMPLES = HOME / "nano_run/r86_multivideo_20261106/sample"
PACKAGE = r86.PACKAGES["B"]
ONNX = {"A": r86.ONNX["B"], "B": OUT / "yolo11n_interp50.onnx"}
ENGINES = {"A": r86.ENGINES["B"], "B": OUT / "yolo11n_interp50_fp16.engine"}
EXPECTED_ONNX = "a0ae9b42a0965876004eda1e56d4a4ccd801bb1a9cecd35051f47d5c8a794e48"
CLASSES = {"A": {"bdd": [1, 2, 3, 5, 7], "ua": [2, 3, 5, 7]},
           "B": {"bdd": [0, 1, 2, 3, 4], "ua": [0, 1, 2, 3]}}
COCO_MAP = "0:2,1:7,2:5,3:3,4:1"


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2))


def run_one(row, version, source, onnx_sha):
    dataset, name = row["dataset"], row["name"]
    ua = dataset == "ua"
    dest = OUT / "pipeline" / dataset / name / version
    dest.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, "src/nano_live.py", "--source", str(source), "--fps", "25" if ua else "5",
            "--video-name", name, "--engine", str(ENGINES[version]), "--onnx-sha256", onnx_sha[version],
            "--tracker", "cv", "--out", str(dest), "--warmup-frames", "25", "--hang-doi", "4",
            "--tegrastats-ms", "500", "--progress-every", "100", "--elapsed-mode", "skip" if ua else "steps",
            "--classes", *map(str, CLASSES[version][dataset])]
    if ua:
        argv += ["--regions", str(source / "ignored_regions.txt")]
    if version == "B":
        argv += ["--class-map", COCO_MAP]
    (dest / "command.json").write_text(json.dumps(argv, indent=2))
    environment = dict(os.environ, BYTETRACK_BACKEND="np", PYTHONDONTWRITEBYTECODE="1")
    with (dest / "stdout.log").open("w") as log:
        result = subprocess.run(argv, cwd=PACKAGE, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{dataset}/{name}/{version} failed, see {dest}/stdout.log")
    meta = json.loads((dest / "run_meta.json").read_text())
    count = meta["dem"]
    assert meta["hoan_tat"] and count["n_frame_da_xu_ly"] == row["n_frames"] == 150
    assert count["n_unreadable"] == count["n_truncated"] == 0
    assert meta["hello_con"]["n_frames_khai_bao"] == 150
    temperatures = meta.get("thiet_bi", {}).get("tegrastats", {}).get("nhiet_max_C", {})
    temp = max((float(value) for value in temperatures.values()), default=0)
    assert temp < 75
    times = meta["thoi_gian"]["tom_tat_sau_warmup"]
    mot = dest / "data" / f"{name}.txt"
    return mot, {"dataset": dataset, "name": name, "version": version, "frames": count["n_frame_da_xu_ly"],
                 "fps": meta["thoi_gian"]["fps_xu_ly"], "fps_steady": times["fps_on_dinh"],
                 "decode_ms": times["decode_ms"]["trung_vi"], "infer_ms": times["infer_ms"]["trung_vi"],
                 "p90_latency_ms": times["tre_ms"]["p90"], "max_temp_c": temp,
                 "mot_rows": len(mot.read_bytes().splitlines()), "mot_sha256": r86.sha(mot),
                 "swap_mb": r86.swap_mb(), "swap_activity": r101.swap_activity()}


def main():
    boot = r86.read("/proc/sys/kernel/random/boot_id")
    errors = r86.read("/sys/fs/ext4/mmcblk1p1/errors_count")
    swap = r86.swap_mb()
    assert errors == "0" and r86.sha(ENGINES["A"]) == r101.BASELINE_SHA256
    onnx_sha = {key: r86.sha(path) for key, path in ONNX.items()}
    assert onnx_sha["B"] == EXPECTED_ONNX
    write("inputs.json", {"boot": boot, "swap_mb": swap, "swap_activity": r101.swap_activity(),
                          "onnx_sha256": onnx_sha, "engine_sha256": {key: r86.sha(p) for key, p in ENGINES.items()},
                          "classes": CLASSES, "class_map_B": COCO_MAP})
    r101.OUT, r86.ENGINES = OUT, ENGINES
    r101.benchmark_engines()
    r86.check_guard(boot, swap, errors)
    manifest = json.loads((SAMPLES / "sample_manifest.json").read_text())["rows"]
    rows = []
    for dataset in ("bdd", "ua"):
        group = [r for r in manifest if r["dataset"] == dataset]
        assert len(group) == 40
        rows.extend(group[i] for i in (0, 10, 20, 30))
    write("videos.json", rows)
    result, files = [], []
    for i, row in enumerate(rows):
        dataset, name = row["dataset"], row["name"]
        original = SAMPLES / dataset / name
        needed = sum(path.stat().st_size for path in original.iterdir() if path.is_file())
        assert needed < disk_usage("/dev/shm").free / 4
        with TemporaryDirectory(prefix="r102_interp50_", dir="/dev/shm") as root:
            source = Path(root) / name
            copytree(original, source)
            jpg = sorted(source.glob("*.jpg"))
            assert len(jpg) == 150 and jpg[0].name == row["first_file"] and jpg[-1].name == row["last_file"]
            for version in ("AB" if i % 2 == 0 else "BA"):
                r86.check_guard(boot, swap, errors)
                mot, item = run_one(row, version, source, onnx_sha)
                r86.check_guard(boot, swap, errors)
                files.append((mot, f"{dataset}/{name}/{version}.txt"))
                result.append(item)
                write("pipeline_pairs.json", result)
                print(json.dumps(item), flush=True)
    assert len(result) == 16
    with tarfile.open(OUT / "mot_outputs.tar.gz", "w:gz") as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
    write("after.json", {"boot": boot, "swap_mb": r86.swap_mb(), "swap_activity": r101.swap_activity(),
                         "errors": r86.read("/sys/fs/ext4/mmcblk1p1/errors_count")})
    print("R102_NANO_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
