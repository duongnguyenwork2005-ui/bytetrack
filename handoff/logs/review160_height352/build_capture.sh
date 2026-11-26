#!/usr/bin/env bash
set -euo pipefail
cd /home/jetson/nano_run/r160_height352_20261120
test ! -e model.engine
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine \
    --fp16 --workspace=512 --buildOnly --verbose > build.log 2>&1
python3 - <<'PY'
from pathlib import Path
import hashlib,json
c=json.loads(Path('build_models.json').read_text());m=c['models']['r352x640']
assert hashlib.sha256(Path(m['onnx']).read_bytes()).hexdigest()==m['onnx_sha256']
m['engine_sha256']=hashlib.sha256(Path(m['engine']).read_bytes()).hexdigest()
Path('models.json').write_text(json.dumps(c,indent=2))
PY
check=/home/jetson/nano_run/r150_pruning_20261120/check_engine.py
if LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 "$check" models.json r352x640 > raw_parity.log 2>&1; then
    echo HEIGHT352_RAW_PARITY_PASS
else
    echo HEIGHT352_RAW_PARITY_FAIL
fi
audit=/home/jetson/nano_run/r155_runtime_accuracy_20261120
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 "$audit/capture.py" models.json r352x640 "$audit/dataset.json" "$audit/dev_images.tar" cache \
  > capture.log 2>&1
echo HEIGHT352_ACTUAL_AP_CAPTURE_COMPLETE
