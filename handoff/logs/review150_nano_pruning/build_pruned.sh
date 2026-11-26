#!/usr/bin/env bash
set -euo pipefail
cd /home/jetson/nano_run/r150_pruning_20261120
for mode in anchor control channel25_KD; do
    test ! -e "$mode/model.engine"
    /usr/src/tensorrt/bin/trtexec --onnx="$mode/model.onnx" \
        --saveEngine="$mode/model.engine" --fp16 --workspace=512 --buildOnly --verbose \
        > "${mode}_build.log" 2>&1
    test -s "$mode/model.engine"
    printf 'PRUNING_BUILD_COMPLETE %s\n' "$mode"
done
python3 - <<'PY'
from pathlib import Path
import hashlib,json
p=Path('build_models.json');c=json.loads(p.read_text())
for m in c['models'].values():
    for kind in ('onnx','engine'):
        h=hashlib.sha256(Path(m[kind]).read_bytes()).hexdigest()
        if kind=='onnx':assert h==m['onnx_sha256']
        m[kind+'_sha256']=h
Path('models.json').write_text(json.dumps(c,indent=2))
PY
for mode in anchor control channel25_KD; do
    # Preserve a failed raw-output check, then still permit the separately
    # preregistered actual-AP/trade-off audit. Never forge a parity marker.
    if LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 check_engine.py models.json "$mode" \
        > "${mode}_parity.log" 2>&1; then
        printf 'PRUNING_RAW_PARITY_PASS %s\n' "$mode"
    else
        printf 'PRUNING_RAW_PARITY_FAIL %s\n' "$mode"
    fi
done
