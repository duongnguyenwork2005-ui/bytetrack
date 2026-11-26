"""Run fixed BDD/UA Nano A/B speed pairs without changing installed launchers.

Copy with the sample archive into ~/nano_run/r86_multivideo_20261106 and invoke
using /home/jetson/miniforge3/envs/nano/bin/python on Jetson.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path("/home/jetson/nano_run/r63_20261030")
ROOT = Path("/home/jetson/nano_run/r86_multivideo_20261106")
DATA = ROOT / "sample"
OUT = ROOT / "runs"
ENGINES = {
    "A": BASE / "engine_v16_board_hien_tai_01/yolov8n_640_op12_fp16.engine",
    "B": BASE / "g3_c1_384x640_toc_do_01/engine/yolov8n_384x640_op12_fp16.engine",
}
PACKAGES = {
    "A": BASE / "baseline/nano_transfer",
    "B": BASE / "v21combo/nano_transfer",
}
ONNX = {
    "A": PACKAGES["A"] / "models/export/yolov8n_640_op12.onnx",
    "B": BASE / "ung_vien_onnx/384x640/yolov8n_384x640_op12.onnx",
}


def read(path: str) -> str:
    return Path(path).read_text().strip()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def swap_mb() -> float:
    x = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("SwapTotal:", "SwapFree:")):
            k, v = line.split(":")
            x[k] = int(v.strip().split()[0])
    return (x["SwapTotal"] - x["SwapFree"]) / 1024


def check_guard(boot: str, swap0: float, mmc0: str) -> None:
    if read("/proc/sys/kernel/random/boot_id") != boot:
        raise RuntimeError("Nano rebooted")
    if read("/sys/fs/ext4/mmcblk1p1/errors_count") != mmc0:
        raise RuntimeError("mmc errors_count changed")
    if swap_mb() - swap0 > 100:
        raise RuntimeError("swap increased >100 MB")


def run_one(row: dict, tracker: str, version: str, boot: str, swap0: float, mmc0: str, onnx_sha: dict) -> dict:
    ds, name = row["dataset"], row["name"]
    source = DATA / ds / name
    dest = OUT / ds / name / tracker / version
    meta_path = dest / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("hoan_tat") and meta.get("dem", {}).get("n_frame_da_xu_ly") == 150:
            return {"dataset": ds, "name": name, "tracker": tracker, "version": version,
                    "fps": meta["thoi_gian"]["fps_xu_ly"], "skipped_complete": True}
    if dest.exists():
        raise RuntimeError("Existing incomplete output: " + str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    pkg = PACKAGES[version]
    cmd = [sys.executable, "src/nano_live.py", "--source", str(source), "--fps", "5" if ds == "bdd" else "25",
           "--video-name", name, "--engine", str(ENGINES[version]), "--onnx-sha256", onnx_sha[version],
           "--tracker", tracker, "--out", str(dest), "--warmup-frames", "25", "--hang-doi", "4",
           "--tegrastats-ms", "500", "--progress-every", "100", "--elapsed-mode", "steps" if ds == "bdd" else "skip"]
    if ds == "ua":
        cmd += ["--regions", str(source / "ignored_regions.txt"), "--classes", "2", "3", "5", "7"]
    env = dict(os.environ, BYTETRACK_BACKEND="np", PYTHONDONTWRITEBYTECODE="1")
    dest.mkdir()
    (dest / "r86_command.json").write_text(json.dumps({"argv": cmd, "cwd": str(pkg)}, indent=2))
    with (dest / "r86_stdout.log").open("w") as log:
        rc = subprocess.run(cmd, cwd=pkg, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"{ds}/{name}/{tracker}/{version} rc={rc}; see {dest}/r86_stdout.log")
    meta = json.loads(meta_path.read_text())
    dem = meta["dem"]
    if not (meta["hoan_tat"] and dem["n_frame_da_xu_ly"] == 150 and dem["n_unreadable"] == dem["n_truncated"] == 0):
        raise RuntimeError(f"invalid completion: {ds}/{name}/{tracker}/{version}: {dem}")
    if meta["hello_con"]["n_frames_khai_bao"] != 150:
        raise RuntimeError("wrong source length")
    check_guard(boot, swap0, mmc0)
    therm = meta.get("thiet_bi", {}).get("tegrastats", {}).get("nhiet_max_C", {})
    if max((float(x) for x in therm.values()), default=0) >= 75:
        raise RuntimeError("thermal guard >=75 C")
    return {"dataset": ds, "name": name, "tracker": tracker, "version": version,
            "fps": meta["thoi_gian"]["fps_xu_ly"], "fps_steady": meta["thoi_gian"]["tom_tat_sau_warmup"]["fps_on_dinh"],
            "decode_ms": meta["thoi_gian"]["tom_tat_sau_warmup"]["decode_ms"]["trung_vi"],
            "p90_latency_ms": meta["thoi_gian"]["tom_tat_sau_warmup"]["tre_ms"]["p90"],
            "swap_mb": swap_mb(), "boot": boot, "skipped_complete": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    manifest = json.loads((DATA / "sample_manifest.json").read_text())
    rows = manifest["rows"]
    assert len(rows) == 80 and sum(r["dataset"] == "bdd" for r in rows) == 40 and sum(r["dataset"] == "ua" for r in rows) == 40
    if args.pilot:
        rows = rows[:2] + rows[40:42]
    boot = read("/proc/sys/kernel/random/boot_id")
    mmc0 = read("/sys/fs/ext4/mmcblk1p1/errors_count")
    swap0 = swap_mb()
    if swap0 >= 100:
        raise RuntimeError("Swap starts >=100 MB")
    onnx_sha = {v: sha(p) for v, p in ONNX.items()}
    input_hash = {"engines": {v: sha(p) for v, p in ENGINES.items()}, "onnx": onnx_sha,
                  "detector_code": {v: sha(PACKAGES[v] / "src/nano_detector.py") for v in "AB"},
                  "boot": boot, "mmc_errors_start": mmc0, "swap_mb_start": swap0}
    ROOT.mkdir(exist_ok=True)
    info = ROOT / "run_inputs.json"
    if info.exists() and json.loads(info.read_text())["engines"] != input_hash["engines"]:
        raise RuntimeError("Engine hash changed")
    info.write_text(json.dumps(input_hash, indent=2))
    outjsonl = ROOT / "progress.jsonl"
    for index, row in enumerate(rows):
        check_guard(boot, swap0, mmc0)
        for tracker in ("cv", "ekf") if index % 2 == 0 else ("ekf", "cv"):
            versions = ("A", "B") if index % 2 == 0 else ("B", "A")
            for version in versions:
                r = run_one(row, tracker, version, boot, swap0, mmc0, onnx_sha)
                with outjsonl.open("a") as f:
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                print(json.dumps(r), flush=True)
                time.sleep(1)
    print("R86_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
