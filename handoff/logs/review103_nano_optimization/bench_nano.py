"""Measure isolated Nano candidates on exactly the R102 sample windows."""
import csv
import json
import os
from pathlib import Path
from shutil import copytree, disk_usage
import sys
import tarfile
from tempfile import TemporaryDirectory

import r102_driver as live
import r101_driver as tools
import r86_driver as guard

HOME = Path.home()
OUT = HOME / "nano_run/r103_optimization"
ANCHOR = HOME / "nano_run/r102_interp50_20261111"
SAMPLES = HOME / "nano_run/r86_multivideo_20261106/sample"
ROI_Y = {"MVI_39031": 86, "MVI_40701": 0, "MVI_40772": 156, "MVI_40855": 30}
PREFIX = {"full": ANCHOR / "yolo11n_interp50", "stride2": ANCHOR / "yolo11n_interp50",
          "early": ANCHOR / "yolo11n_interp50", "roi256": OUT / "yolo11n_roi256", "student": OUT / "yolo11n_student"}
ENV = {"stride2": "R103_CADENCE", "early": "R103_EARLY_CUTOFF", "roi256": "R103_ROI_Y"}
BASELINE_ENGINE_SHA = "13f8cd0f9777840052fc671a621dc0473a103eeecce71ce655b54c74a4ba9823"


def onnx(mode):
    return PREFIX[mode].with_suffix(".onnx")


def engine(mode):
    return PREFIX[mode].with_name(PREFIX[mode].name + "_fp16.engine")


def write(output, name, payload):
    (output / name).write_text(json.dumps(payload, indent=2))


def sample_rows():
    manifest = json.loads((SAMPLES / "sample_manifest.json").read_text())["rows"]
    result = []
    for dataset in ("bdd", "ua"):
        group = [r for r in manifest if r["dataset"] == dataset]
        assert len(group) == 40
        result.extend(group[i] for i in (0, 10, 20, 30))
    return result


def select_modes(phase, row, index):
    if phase == "student":
        modes = ["full", "student"]
    else:
        modes = ["full", "stride2"]
        if row["dataset"] == "ua":
            modes.append("roi256")
        if index % 4 < 2:
            modes.append("early")
    return modes if index % 2 == 0 else modes[::-1]


def run_mode(output, mode, row, source, onnx_hash):
    for key in ENV.values():
        os.environ.pop(key, None)
    if mode in ENV:
        os.environ[ENV[mode]] = str({"stride2": 2, "early": 1, "roi256": ROI_Y.get(row["name"])}[mode])
    live.OUT = output / "runs" / mode
    live.ENGINES = {"B": engine(mode)}
    mot, details = live.run_one(row, "B", source, {"B": onnx_hash[mode]})
    pipeline = mot.parent.parent / "pipeline" / f"{row['name']}.csv"
    with pipeline.open() as file:
        frame_rows = list(csv.DictReader(file))
    assert len(frame_rows) == 150
    marked = [int(f["frame"]) for f in frame_rows if f["infer_skipped"] == "True"]
    assert marked == (list(range(2, 151, 2)) if mode == "stride2" else [])
    label = mot.parent.parent / "data_cls" / f"{row['name']}.csv"
    return mot, dict(details, mode=mode, n_inferred=150 - len(marked), cls_sha256=guard.sha(label),
                     roi_y=ROI_Y[row["name"]] if mode == "roi256" else None)


def main():
    phase = sys.argv[1]
    assert phase in {"pipeline", "student"}
    output = OUT / phase
    output.mkdir(exist_ok=False)
    live.PACKAGE = OUT / "nano_transfer"
    rows = sample_rows()
    selected_modes = sorted({mode for i, row in enumerate(rows) for mode in select_modes(phase, row, i)})
    onnx_hash = {mode: guard.sha(onnx(mode)) for mode in selected_modes}
    engine_hash = {mode: guard.sha(engine(mode)) for mode in selected_modes}
    assert engine_hash["full"] == BASELINE_ENGINE_SHA
    boot = guard.read("/proc/sys/kernel/random/boot_id")
    swap = guard.swap_mb()
    errors = guard.read("/sys/fs/ext4/mmcblk1p1/errors_count")
    assert errors == "0"
    write(output, "inputs.json", {"boot": boot, "swap_mb": swap, "swap_activity": tools.swap_activity(),
                                  "onnx_sha256": onnx_hash, "engine_sha256": engine_hash, "roi_y": ROI_Y})
    write(output, "videos.json", rows)
    tools.OUT = output
    guard.ENGINES = {"A": engine("full"), "B": engine("student" if phase == "student" else "roi256")}
    tools.benchmark_engines()
    results, files = [], []
    for i, row in enumerate(rows):
        dataset, name = row["dataset"], row["name"]
        original = SAMPLES / dataset / name
        needed = sum(path.stat().st_size for path in original.iterdir() if path.is_file())
        assert needed < disk_usage("/dev/shm").free / 4
        with TemporaryDirectory(prefix="r103_nano_", dir="/dev/shm") as tmp:
            source = Path(tmp) / name
            copytree(original, source)
            jpg = sorted(source.glob("*.jpg"))
            assert len(jpg) == 150 and jpg[0].name == row["first_file"] and jpg[-1].name == row["last_file"]
            for mode in select_modes(phase, row, i):
                guard.check_guard(boot, swap, errors)
                mot, item = run_mode(output, mode, row, source, onnx_hash)
                guard.check_guard(boot, swap, errors)
                files.append((mot, f"{dataset}/{name}/{mode}.txt"))
                results.append(item)
                write(output, "pairs.json", results)
                print(json.dumps(item), flush=True)
    assert len(results) == (16 if phase == "student" else 24)
    with tarfile.open(output / "mot_outputs.tar.gz", "w:gz") as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
    write(output, "after.json", {"boot": boot, "swap_mb": guard.swap_mb(), "swap_activity": tools.swap_activity(),
                                 "errors": guard.read("/sys/fs/ext4/mmcblk1p1/errors_count")})
    print("R103_BENCH_COMPLETE", phase, flush=True)


if __name__ == "__main__":
    main()
