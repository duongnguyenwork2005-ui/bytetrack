"""Compare four filters with the locked R109 TensorRT live pipeline on Nano."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
NANO = ROOT.parent
REFERENCE = NANO / "r109_full80_20261117/results"
OVERLAP = NANO / "r108_15fps_20261117/overlap_package"
SAMPLES = NANO / "r86_multivideo_20261106/sample"
R110 = NANO / "r110_filters_20261117/results"
PACKAGE = ROOT / "pipeline_package"
METHODS = ("kf", "ekf", "ukf", "rbpf")
CONFIGS = {"ekf": "bytetrack_ekf_ctrv.yaml", "ukf": "bytetrack_ukf_ctrv.yaml", "rbpf": "rbpf500.yaml"}
BOOT = Path("/proc/sys/kernel/random/boot_id")
ERRORS = Path("/sys/fs/ext4/mmcblk1p1/errors_count")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hardware():
    mem = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
           if line.startswith(("SwapTotal:", "SwapFree:"))}
    return {"boot": BOOT.read_text().strip(), "sd_errors": ERRORS.read_text().strip(),
            "swap_mib": (mem["SwapTotal"] - mem["SwapFree"]) / 1024}


def prepare():
    PACKAGE.mkdir(exist_ok=False)
    shutil.copytree(OVERLAP / "src", PACKAGE / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("configs", "third_party", "models", "data"):
        (PACKAGE / name).symlink_to((OVERLAP / name).resolve(), target_is_directory=True)
    for name in ("pf_ctrv.py", "tracker_ctrv.py", "rbpf_ctrv.py"):
        shutil.copy2(ROOT / "overlay" / name, PACKAGE / "src" / name)
    runner = PACKAGE / "src/run_tracking.py"
    text = runner.read_text()
    line = "        return bytetrack_np.tracker_registry_np()"
    assert text.count(line) == 1
    runner.write_text(text.replace(line, "        from rbpf_ctrv import RBPFByteTracker\n"
              "        reg = bytetrack_np.tracker_registry_np()\n"
              "        reg.update(bytetrack_ukf_ctrv=tracker_ctrv.UKFByteTracker, bytetrack_rbpf_ctrv=RBPFByteTracker)\n"
              "        return reg"))
    live = PACKAGE / "src/nano_live.py"
    text = live.read_text()
    old = "track_video(vd, cfg, dt_nominal, a.elapsed_mode, None, ten,"
    new = "track_video(vd, cfg, dt_nominal, a.elapsed_mode, getattr(cfg, 'pf_seed', None), ten,"
    assert text.count(old) == 1
    live.write_text(text.replace(old, new))


def run_process(args, logfile):
    env = dict(os.environ, BYTETRACK_BACKEND="np", R103_EARLY_CUTOFF="1", PYTHONDONTWRITEBYTECODE="1")
    for name in ("R103_CADENCE", "R103_ROI_Y", "R105_MODE"):
        env.pop(name, None)
    with logfile.open("w") as log:
        with subprocess.Popen(args, cwd=PACKAGE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as child:
            try:
                code = child.wait(timeout=240)
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                raise
            if code != 0:
                raise RuntimeError(f"Nano live exited with {code}; see {logfile}")


def one(item, mounted, method, root, pair, capture):
    ds, name = item["dataset"], item["name"]
    baseline = REFERENCE / "overlap/runs/full/pipeline" / ds / name / "B"
    dest = root / "runs" / ds / name / method
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert not dest.exists()
    argv = read(baseline / "command.json")
    assert argv[1] == "src/nano_live.py" and argv[argv.index("--tracker") + 1] == "cv"
    assert sha(Path(argv[argv.index("--engine") + 1])) == pair["engine_sha256"]
    argv[argv.index("--source") + 1] = str(mounted)
    argv[argv.index("--out") + 1] = str(dest)
    if ds == "ua":
        argv[argv.index("--regions") + 1] = str(mounted / "ignored_regions.txt")
    if method != "kf":
        idx = argv.index("--tracker")
        argv[idx:idx+2] = ["--tracker-yaml", str(ROOT / "overlay" / CONFIGS[method])]
    argv.extend(["--save-cache", name, "--cache-root", str(dest / "det_capture")])
    write(dest.parent / (method + "_command.json"), argv)
    run_process(argv, dest.parent / (method + ".log"))
    meta = read(dest / "run_meta.json")
    assert meta["hoan_tat"] and meta["dem"]["n_frame_da_xu_ly"] == 150 and meta["dem"]["n_unreadable"] == 0
    assert meta["hello_con"]["engine_sha256"] == pair["engine_sha256"]
    assert meta["cau_hinh"]["conf_min"] == meta["cau_hinh"]["det_conf"] == .25 and not meta["hien_thi"]["bat"]
    times = meta["thoi_gian"]["thong_ke_steady"] if "thong_ke_steady" in meta["thoi_gian"] else meta["thoi_gian"]
    with (dest / "pipeline" / (name + ".csv")).open() as stream:
        frames = list(csv.DictReader(stream))
    assert len(frames) == 150 and [int(r["frame"]) for r in frames] == list(range(1, 151))
    completion = [float(row["t_xong_cha"]) for row in frames]
    assert all(b > a for a, b in zip(completion, completion[1:]))
    fps = 125 / (completion[-1] - completion[24])
    latency = sorted(float(r["tre_ms"]) for r in frames[25:])
    p90 = latency[round(.9 * (len(latency)-1))]
    mot = dest / "data" / (name + ".txt")
    cls = dest / "data_cls" / (name + ".csv")
    cache = dest / "det_capture" / name / (name + ".npz")
    cache_key = sha(cache)
    assert cache_key == capture["cache_sha256"], (name, method, "detector cache drift", cache_key)
    if method == "kf":
        assert sha(mot) == pair["mot_sha256"] and sha(cls) == pair["cls_sha256"], (name, "KF parity")
    with (dest / "timing" / (name + ".csv")).open() as stream:
        updates = [float(row["t_update_ms"]) for row in csv.DictReader(stream)][25:]
    assert len(updates) == 125
    return {"dataset": ds, "name": name, "method": method, "fps": fps, "latency_p90_ms": p90,
            "update_median_ms": statistics.median(updates), "det_cache_sha256": cache_key,
            "mot_sha256": sha(mot), "classes_sha256": sha(cls), "n_frames": 150,
            "hardware_after": hardware(), "path": str(dest)}


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("stage", choices=("prepare", "smoke", "measure"))
    stage = args.parse_args().stage
    if stage == "prepare":
        prepare()
        return
    root = ROOT / stage
    root.mkdir(exist_ok=False)
    old = read(REFERENCE / "inputs.json")
    assert old["videos"] == read(SAMPLES / "sample_manifest.json")["rows"]
    pairs = {(r["dataset"], r["name"]): r for r in read(REFERENCE / "pairs.json") if r["mode"] == "overlap"}
    capture = {(r["dataset"], r["name"]): r for r in read(R110 / "captures.json")}
    assert len(pairs) == len(capture) == 80
    indices = (0,) if stage == "smoke" else tuple(range(0, 40, 5))
    grouped = [[r for r in old["videos"] if r["dataset"] == ds] for ds in ("bdd", "ua")]
    sample = [g[i] for i in indices for g in grouped]
    start = hardware()
    assert start["sd_errors"] == "0" and start["swap_mib"] < 100, start
    paths = [PACKAGE / "src" / name for name in ("nano_live.py", "nano_detector.py", "nano_detect_stream.py",
              "run_tracking.py", "pf_ctrv.py", "rbpf_ctrv.py", "tracker_ctrv.py", "ekf_ctrv.py", "ukf_ctrv.py")]
    inputs = {"sample": sample, "methods": METHODS, "warmup": 25, "rbpf_seed": 0,
              "hardware_start": start, "code_sha256": {str(p): sha(p) for p in paths},
              "config_sha256": {str(ROOT / "overlay" / q): sha(ROOT / "overlay" / q) for q in CONFIGS.values()},
              "source_run": str(REFERENCE), "numpy": subprocess.check_output([sys.executable, "-c", "import numpy;print(numpy.__version__)"], text=True).strip()}
    write(root / "inputs.json", inputs)
    runs = []
    try:
        for j, row in enumerate(sample):
            ds, name = row["dataset"], row["name"]
            prior = read(REFERENCE / (ds + "_" + name + "_images.json"))
            start_method = ((j // 2) + 2 * (j % 2)) % len(METHODS)
            with TemporaryDirectory(prefix="r114_nano_", dir="/dev/shm") as tmp:
                mounted = Path(tmp) / name
                shutil.copytree(SAMPLES / ds / name, mounted)
                assert len(prior) == 150 and {p.name: sha(p) for p in mounted.glob("*.jpg")} == prior
                for method in METHODS[start_method:] + METHODS[:start_method]:
                    current = hardware()
                    assert current["boot"] == start["boot"] and current["sd_errors"] == "0" and current["swap_mib"] - start["swap_mib"] < 100, current
                    result = one(row, mounted, method, root, pairs[ds, name], capture[ds, name])
                    runs.append(result)
                    write(root / "runs.json", runs)
                    print(len(runs), ds, name, method, f'{result["fps"]:.3f}', f'{result["latency_p90_ms"]:.1f}', flush=True)
        assert len(runs) == len(sample) * len(METHODS)
        assert all(sha(Path(path)) == digest for key in ("code_sha256", "config_sha256") for path, digest in inputs[key].items())
        print("COMPLETE", len(runs), flush=True)
    finally:
        write(root / "hardware_after.json", hardware())


if __name__ == "__main__":
    main()
