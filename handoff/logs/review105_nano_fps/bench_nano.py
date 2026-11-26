"""R105: fresh paired live runs; reuse the R102 driver without altering it."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from shutil import copytree, disk_usage
import tarfile
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
HOME = Path.home()
OUT = HOME / "nano_run/r105_nano_fps"
ANCHOR = HOME / "nano_run/r102_interp50_20261111"
SAMPLES = HOME / "nano_run/r86_multivideo_20261106/sample"
EXPECTED_ENGINE = {"A": "a6912393829f8c811d3cc33008f3e4056a626434df327c85aae7beb33af01749",
                   "B": "13f8cd0f9777840052fc671a621dc0473a103eeecce71ce655b54c74a4ba9823"}
EXPECTED_ONNX = {"A": "5f5b337d2978bd7b5bb3195f6d04e3725f536d5e4cd87d4d1cb7a1c8af984f49",
                 "B": "a0ae9b42a0965876004eda1e56d4a4ccd801bb1a9cecd35051f47d5c8a794e48"}
ENV_KEYS = ("R103_CADENCE", "R103_EARLY_CUTOFF", "R103_ROI_Y", "R105_MODE")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def modes_for(row, index):
    modes = ["v21combo", "full"]
    if row["dataset"] == "ua":
        modes += ["stride2", "motion", "guarded"]
    elif index < 2:
        modes += ["guarded"]
    return modes if index % 2 == 0 else modes[::-1]


def frame_checks(rows, mode, dataset):
    if [int(r["frame"]) for r in rows] != list(range(1, 151)):
        raise ValueError("Expected exactly the 150 consecutive frames")
    skipped = [int(r["frame"]) for r in rows if r.get("infer_skipped", "False") == "True"]
    if any(r["unreadable"] != "False" for r in rows):
        raise ValueError("Unreadable frame")
    if mode == "stride2":
        if skipped != list(range(2, 151, 2)):
            raise ValueError("Stride schedule mismatch")
    elif mode in ("full", "v21combo") or dataset == "bdd":
        if skipped:
            raise ValueError("Full-rate/fallback unexpectedly skipped inference")
    elif any(i % 2 for i in skipped):
        raise ValueError("Adaptive schedule skipped an odd frame")
    if mode != "v21combo" and any("infer_skipped" not in r for r in rows):
        raise ValueError("Missing explicit inference flags")
    return skipped


def write(root, name, value):
    (root / name).write_text(json.dumps(value, indent=2))


def run_mode(root, mode, row, source, live, guard, onnx_hash):
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    if mode == "stride2":
        os.environ["R103_CADENCE"] = "2"
    if mode in ("motion", "guarded"):
        os.environ["R105_MODE"] = mode
    version = "A" if mode == "v21combo" else "B"
    live.OUT = root / "runs" / mode
    # Direct deployed v21combo code for A; experiment copy for YOLO11n.
    live.PACKAGE = guard.PACKAGES["B"] if version == "A" else OUT / "nano_transfer"
    mot, details = live.run_one(row, version, source, onnx_hash)
    dest = mot.parent.parent
    with (dest / "pipeline" / (row["name"] + ".csv")).open() as file:
        frames = list(csv.DictReader(file))
    skipped = frame_checks(frames, mode, row["dataset"])
    meta_path = dest / "run_meta.json"
    meta = json.loads(meta_path.read_text())
    adaptive = meta["eos_con"].get("adaptive")
    if version == "B":
        enabled = mode in ("motion", "guarded") and row["dataset"] == "ua"
        if not adaptive or adaptive["enabled"] != enabled:
            raise ValueError("Adaptive scheduler enable/fallback mismatch")
        if enabled and (sum(adaptive["counts"].values()) != 150 or
                        adaptive["counts"]["skipped"] != len(skipped)):
            raise ValueError("Scheduler/frame counters disagree")
    label = dest / "data_cls" / (row["name"] + ".csv")
    details.update(mode=mode, n_inferred=150-len(skipped), skipped_frames=skipped,
                   cls_sha256=sha(label), pipeline_sha256=sha(dest / "pipeline" / (row["name"] + ".csv")),
                   meta_sha256=sha(meta_path), adaptive=adaptive,
                   engine_sha256=sha(live.ENGINES[version]), source_sha256=sha(source / "r105_images.json"))
    return mot, label, dest, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "full"))
    args = parser.parse_args()
    # These are byte-identical local copies of previously reviewed drivers.
    import sys
    sys.path.insert(0, str(HERE / "drivers"))
    driver_manifest = json.loads((HERE / "driver_manifest.json").read_text())
    assert all(sha(HERE / "drivers" / name) == info["sha256"] for name, info in driver_manifest.items())
    import r86_driver as guard
    import r101_driver as tools
    import r102_driver as live
    import prepare_nano
    if not (OUT / "nano_transfer").exists():
        OUT.mkdir(exist_ok=True)
        prepare_nano.main()
    prepare_nano.validate_package()
    live.ENGINES = {"A": guard.ENGINES["B"], "B": ANCHOR / "yolo11n_interp50_fp16.engine"}
    onnx = {"A": guard.ONNX["B"], "B": ANCHOR / "yolo11n_interp50.onnx"}
    onnx_hash = {k: sha(p) for k, p in onnx.items()}
    engine_hash = {k: sha(p) for k, p in live.ENGINES.items()}
    if onnx_hash != EXPECTED_ONNX or engine_hash != EXPECTED_ENGINE:
        raise ValueError("Engine/ONNX mismatch; do not rebuild or substitute")
    manifest = json.loads((SAMPLES / "sample_manifest.json").read_text())["rows"]
    videos = []
    for dataset in ("bdd", "ua"):
        group = [r for r in manifest if r["dataset"] == dataset]
        assert len(group) == 40
        videos.extend(group[i] for i in (0, 10, 20, 30))
    assert all(r["n_frames"] == r["last"]-r["first"]+1 == 150 for r in videos)
    baseline_package_hash = {p.name: sha(p) for p in (guard.PACKAGES["B"] / "src").glob("*.py")}
    assert baseline_package_hash
    package_hash = {p.name: sha(p) for p in (OUT / "nano_transfer/src").glob("*.py")}
    root = OUT / args.phase
    if args.phase == "full":
        smoke = json.loads((OUT / "smoke" / "complete.json").read_text())
        assert smoke["runs"] == 8 and smoke["engine_sha256"] == engine_hash
        assert smoke["code_sha256"] == {p.name: sha(p) for p in HERE.glob("*.py")}
        assert smoke["package_sha256"] == package_hash
        assert smoke["baseline_package_sha256"] == baseline_package_hash
        assert smoke["driver_manifest_sha256"] == sha(HERE / "driver_manifest.json")
    root.mkdir(exist_ok=False)
    boot = guard.read("/proc/sys/kernel/random/boot_id")
    swap = guard.swap_mb()
    errors = guard.read("/sys/fs/ext4/mmcblk1p1/errors_count")
    assert errors == "0"
    write(root, "inputs.json", {"boot": boot, "swap_mb": swap, "errors": errors,
          "swap_activity": tools.swap_activity(), "engine_sha256": engine_hash, "onnx_sha256": onnx_hash,
          "classes": live.CLASSES, "class_map_B": live.COCO_MAP,
          "code_sha256": {p.name: sha(p) for p in HERE.glob("*.py")},
          "package_sources": json.loads((OUT / "package_sources.json").read_text()),
          "package_sha256": package_hash, "baseline_package_sha256": baseline_package_hash,
          "driver_manifest_sha256": sha(HERE / "driver_manifest.json")})
    selected = videos if args.phase == "full" else [videos[0], videos[4]]
    write(root, "videos.json", selected)
    results, files = [], []
    try:
        for row in selected:
            index = videos.index(row)
            ds, name = row["dataset"], row["name"]
            original = SAMPLES / ds / name
            needed = sum(p.stat().st_size for p in original.iterdir() if p.is_file())
            assert needed < disk_usage("/dev/shm").free / 4
            with TemporaryDirectory(prefix="r105_", dir="/dev/shm") as tmp:
                source = Path(tmp) / name
                copytree(original, source)
                jpg = sorted(source.glob("*.jpg"))
                assert len(jpg) == 150 and jpg[0].name == row["first_file"] and jpg[-1].name == row["last_file"]
                image_hash = {p.name: sha(p) for p in jpg}
                write(source, "r105_images.json", image_hash)
                write(root, ds + "_" + name + "_images.json", image_hash)
                for mode in modes_for(row, index):
                    guard.check_guard(boot, swap, errors)
                    mot, label, dest, item = run_mode(root, mode, row, source, live, guard, onnx_hash)
                    guard.check_guard(boot, swap, errors)
                    prefix = ds + "/" + name + "/" + mode
                    files.extend([(mot, prefix+".txt"), (label, prefix+"_cls.csv")])
                    for path in dest.rglob("*"):
                        if path.is_file() and path not in (mot, label):
                            files.append((path, "raw/"+prefix+"/"+str(path.relative_to(dest))))
                    results.append(item)
                    write(root, "pairs.json", results)
                    print(json.dumps(item), flush=True)
                if ds == "bdd" and index < 2:
                    base, fallback = [next(r for r in results if r["name"] == name and r["mode"] == m)
                                      for m in ("full", "guarded")]
                    assert base["mot_sha256"] == fallback["mot_sha256"]
                    assert base["cls_sha256"] == fallback["cls_sha256"]
        assert len(results) == (30 if args.phase == "full" else 8)
        assert {k: sha(p) for k, p in live.ENGINES.items()} == engine_hash
        assert {p.name: sha(p) for p in (OUT / "nano_transfer/src").glob("*.py")} == package_hash
        assert {p.name: sha(p) for p in (guard.PACKAGES["B"] / "src").glob("*.py")} == baseline_package_hash
        with tarfile.open(root / "evidence.tar.gz", "w:gz") as archive:
            for path, name in files:
                archive.add(path, arcname=name, recursive=False)
        write(root, "complete.json", {"runs": len(results), "engine_sha256": engine_hash,
              "code_sha256": {p.name: sha(p) for p in HERE.glob("*.py")},
              "archive_sha256": sha(root / "evidence.tar.gz"), "package_sha256": package_hash,
              "baseline_package_sha256": baseline_package_hash,
              "driver_manifest_sha256": sha(HERE / "driver_manifest.json")})
        print("R105_BENCH_COMPLETE", args.phase, flush=True)
    finally:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        write(root, "after.json", {"boot": guard.read("/proc/sys/kernel/random/boot_id"),
              "swap_mb": guard.swap_mb(), "swap_activity": tools.swap_activity(),
              "errors": guard.read("/sys/fs/ext4/mmcblk1p1/errors_count")})


if __name__ == "__main__":
    main()
