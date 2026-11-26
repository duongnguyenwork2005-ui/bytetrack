"""Live A/B/B/A on two UA clips; use same engine and require byte-identical MOT."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HOME = Path.home()
BASE = HOME / "nano_run/r63_20261030/v21combo/nano_transfer"
DATA = HOME / "nano_run/r86_multivideo_20261106/sample/ua"
OUT = HOME / "nano_run/r98_lse_live_20261110"
ENGINE = HOME / "nano_run/r63_20261030/g3_c1_384x640_toc_do_01/engine/yolov8n_384x640_op12_fp16.engine"
ONNX = HOME / "nano_run/r63_20261030/ung_vien_onnx/384x640/yolov8n_384x640_op12.onnx"
BOOT = Path("/proc/sys/kernel/random/boot_id")
ERRORS = Path("/sys/fs/ext4/mmcblk1p1/errors_count")


def swap_mb():
    vals = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("SwapTotal:", "SwapFree:")):
            k, v = line.split(":")
            vals[k] = int(v.strip().split()[0])
    return (vals["SwapTotal"] - vals["SwapFree"]) / 1024


def main():
    boot, errors, swap0 = BOOT.read_text(), ERRORS.read_text(), swap_mb()
    assert errors.strip() == "0" and swap0 < 100
    onnx_sha = hashlib.sha256(ONNX.read_bytes()).hexdigest()
    results = []
    for video in ("MVI_39031", "MVI_40855"):
        source = DATA / video
        assert len(list(source.glob("*.jpg"))) == 150
        canonical = None
        for phase, case in enumerate(("baseline", "candidate", "candidate", "baseline")):
            assert BOOT.read_text() == boot and ERRORS.read_text() == errors and swap_mb() - swap0 <= 100
            entry = ("r96_nano_live_pf_experiment.py" if case == "baseline"
                     else "r98_fast_pf_live.py")
            dest = OUT / video / f"{phase}_{case}"
            assert not dest.exists()
            dest.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(HOME / "nano_run" / entry),
                   "--source", str(source), "--fps", "25", "--video-name", video,
                   "--engine", str(ENGINE), "--onnx-sha256", onnx_sha,
                   "--tracker-yaml", "configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml",
                   "--out", str(dest), "--warmup-frames", "25", "--hang-doi", "4",
                   "--elapsed-mode", "skip", "--regions", str(source / "ignored_regions.txt"),
                   "--classes", "2", "3", "5", "7", "--tegrastats-ms", "500"]
            dest.mkdir()
            (dest / "command.json").write_text(json.dumps(cmd, indent=2))
            with (dest / "stdout.log").open("w") as f:
                rc = subprocess.run(cmd, cwd=BASE, env=dict(os.environ, BYTETRACK_BACKEND="np"),
                                    stdout=f, stderr=subprocess.STDOUT).returncode
            if rc:
                raise RuntimeError(f"{video}/{case}/{phase} exit {rc}: {dest}/stdout.log")
            meta = json.loads((dest / "run_meta.json").read_text())
            assert meta["hoan_tat"] and meta["dem"]["n_frame_da_xu_ly"] == 150
            assert meta["dem"]["n_unreadable"] == meta["dem"]["n_truncated"] == 0
            data = (dest / "data" / (video + ".txt")).read_bytes()
            if canonical is None:
                canonical = data
            assert data == canonical, (video, case, phase)
            assert BOOT.read_text() == boot and ERRORS.read_text() == errors and swap_mb() - swap0 <= 100
            therm = meta.get("thiet_bi", {}).get("tegrastats", {}).get("nhiet_max_C", {})
            assert max((float(x) for x in therm.values()), default=0) < 75
            t = meta["thoi_gian"]["tom_tat_sau_warmup"]
            row = {"video": video, "phase": phase, "case": case,
                   "fps_steady": t["fps_on_dinh"], "update_median_ms": t["t_update_ms"]["trung_vi"],
                   "decode_median_ms": t["decode_ms"]["trung_vi"],
                   "infer_median_ms": t["infer_ms"]["trung_vi"],
                   "latency_p90_ms": t["tre_ms"]["p90"],
                   "mot_sha256": hashlib.sha256(data).hexdigest(), "swap_mb": swap_mb()}
            results.append(row)
            (OUT / "live_abba.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(row), flush=True)
    assert len(results) == 8
    print("R98_LIVE_ABBA_PASS", flush=True)


if __name__ == "__main__":
    main()
