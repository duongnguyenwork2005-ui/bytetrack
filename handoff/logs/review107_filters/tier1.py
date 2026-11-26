"""Capture concrete commands, exit statuses and logs; wait for every child process."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
PYTHON = "C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe"
ROOT = OUT.parents[2]
stage = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else stage
(OUT / "ultralytics_settings/Ultralytics").mkdir(parents=True, exist_ok=True)
command = ([PYTHON, "-B", str(OUT / "test_postgap.py")] + (["--regression"] if stage == "regression" else [])
           if stage in ("unit", "regression") else [PYTHON, "-B", str(OUT / "run_r107.py"), stage])
if stage == "verify":
    command = [PYTHON, "-B", str(OUT / "verify_results.py")]
env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1",
           YOLO_CONFIG_DIR=str(OUT / "ultralytics_settings"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1")
start = time.time()
log = OUT / f"tier1_{label}.log"
assert not log.exists(), "Keep previous evidence; choose an explicit new log for a correction"
with log.open("w", encoding="utf-8") as stream:
    try:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=5400)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
record = {"stage": stage, "command": command, "cwd": str(ROOT), "exit": code,
          "started_utc": dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
          "seconds": time.time() - start, "log": str(log), "child_waited": True,
          "environment_overrides": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE",
          "YOLO_CONFIG_DIR", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
path = OUT / "tier1_checks.json"
records = json.loads(path.read_text()) if path.exists() else []
records.append(record)
path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record), flush=True)
print(log.read_text(encoding="utf-8")[-3500:])
raise SystemExit(code)
