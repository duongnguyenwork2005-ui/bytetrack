#!/usr/bin/env bash
set -euo pipefail
cd /home/jetson/nano_run/r151_resolution_20261120
sha256sum --check onnx_expected.sha256
for mode in baseline r352x608 r320x576; do
    test ! -e "models/${mode}.engine"
    /usr/src/tensorrt/bin/trtexec \
        --onnx="models/${mode}.onnx" \
        --saveEngine="models/${mode}.engine" \
        --fp16 --workspace=512 --buildOnly --verbose \
        > "${mode}_build.log" 2>&1
    test -s "models/${mode}.engine"
    sha256sum "models/${mode}.engine" > "${mode}_engine.sha256"
    printf 'BUILD_COMPLETE %s\n' "$mode"
done
sha256sum --check onnx_expected.sha256
