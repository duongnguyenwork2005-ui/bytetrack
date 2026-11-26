"""Isolated CV/PF500 Nano pilot on two existing R86 150-frame samples."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path.home() / "nano_run/r63_20261030/v21combo/nano_transfer"
DATA = Path.home() / "nano_run/r86_multivideo_20261106/sample"
OUT = Path.home() / "nano_run/r96_pf_pilot_fix_20261110"
ENGINE = Path.home() / "nano_run/r63_20261030/g3_c1_384x640_toc_do_01/engine/yolov8n_384x640_op12_fp16.engine"
ONNX = Path.home() / "nano_run/r63_20261030/ung_vien_onnx/384x640/yolov8n_384x640_op12.onnx"
ERRORS = Path("/sys/fs/ext4/mmcblk1p1/errors_count")
BOOT = Path("/proc/sys/kernel/random/boot_id")


def swap_mb():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("SwapTotal:", "SwapFree:")):
            k, v = line.split(":")
            values[k] = int(v.strip().split()[0])
    return (values["SwapTotal"] - values["SwapFree"]) / 1024


def main():
    assert BASE.is_dir() and DATA.is_dir() and ENGINE.is_file() and ONNX.is_file()
    boot, errors, swap0 = BOOT.read_text(), ERRORS.read_text(), swap_mb()
    assert swap0 < 100, f"initial swap {swap0} MB"
    manifest = json.loads((DATA / "sample_manifest.json").read_text())
    rows = [next(r for r in manifest["rows"] if r["dataset"] == ds) for ds in ("bdd", "ua")]
    onnx_hash = hashlib.sha256(ONNX.read_bytes()).hexdigest()
    OUT.mkdir(exist_ok=True)
    report = []
    for row in rows:
        ds, name = row["dataset"], row["name"]
        source = DATA / ds / name
        assert len(list(source.glob("*.jpg"))) == 150, str(source)
        for tracker in (("cv", "pf") if ds == "bdd" else ("pf", "cv")):
            assert BOOT.read_text() == boot and ERRORS.read_text() == errors
            assert swap_mb() - swap0 <= 100
            dest = OUT / ds / name / tracker
            if dest.exists():
                raise RuntimeError("output exists: " + str(dest))
            dest.parent.mkdir(parents=True, exist_ok=True)
            entry = (str(Path.home() / "nano_run/r96_nano_live_pf_experiment.py")
                     if tracker == "pf" else "src/nano_live.py")
            cmd = [sys.executable, entry, "--source", str(source),
                   "--fps", "5" if ds == "bdd" else "25", "--video-name", name,
                   "--engine", str(ENGINE), "--onnx-sha256", onnx_hash,
                   "--tracker-yaml", ("configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml"
                                      if tracker == "pf" else "configs/phase2/trackers/bytetrack_cv.yaml"),
                   "--out", str(dest), "--warmup-frames", "25", "--hang-doi", "4",
                   "--tegrastats-ms", "500", "--elapsed-mode", "steps" if ds == "bdd" else "skip"]
            if ds == "ua":
                cmd += ["--regions", str(source / "ignored_regions.txt"),
                        "--classes", "2", "3", "5", "7"]
            dest.mkdir()
            (dest / "command.json").write_text(json.dumps(cmd, indent=2))
            env = dict(os.environ, BYTETRACK_BACKEND="np", PYTHONDONTWRITEBYTECODE="1")
            with (dest / "stdout.log").open("w") as f:
                rc = subprocess.run(cmd, cwd=BASE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            if rc:
                raise RuntimeError(f"{ds}/{tracker} exit {rc}; see {dest}/stdout.log")
            meta = json.loads((dest / "run_meta.json").read_text())
            assert meta["hoan_tat"] and meta["dem"]["n_frame_da_xu_ly"] == 150
            assert meta["dem"]["n_unreadable"] == meta["dem"]["n_truncated"] == 0
            assert BOOT.read_text() == boot and ERRORS.read_text() == errors
            assert swap_mb() - swap0 <= 100
            therm = meta.get("thiet_bi", {}).get("tegrastats", {}).get("nhiet_max_C", {})
            assert max((float(x) for x in therm.values()), default=0) < 75
            item = {"dataset": ds, "name": name, "tracker": tracker,
                    "fps": meta["thoi_gian"]["fps_xu_ly"],
                    "fps_steady": meta["thoi_gian"]["tom_tat_sau_warmup"]["fps_on_dinh"],
                    "timing": meta["thoi_gian"]["tom_tat_sau_warmup"],
                    "swap_mb": swap_mb(), "boot": boot, "mmc_errors": errors.strip()}
            report.append(item)
            (OUT / "pilot.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({k: item[k] for k in ("dataset", "tracker", "fps", "fps_steady")}), flush=True)
    print("R96_PILOT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
