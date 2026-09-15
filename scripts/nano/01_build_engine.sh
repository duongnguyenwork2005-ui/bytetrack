#!/usr/bin/env bash
# Build engine TensorRT tu ONNX tren Jetson Nano (TensorRT 8.2). FP16 mac dinh; bo --fp16 de FP32.
# Cach dung: bash scripts/nano/01_build_engine.sh <model.onnx> <out.engine> [--fp32]
set -eu
ONNX=$1; ENGINE=$2; PREC=${3:---fp16}
TRTEXEC=/usr/src/tensorrt/bin/trtexec
[ -x "$TRTEXEC" ] || { echo "Khong thay $TRTEXEC"; exit 1; }
FLAGS="--onnx=$ONNX --saveEngine=$ENGINE --workspace=1024 --verbose"
[ "$PREC" = "--fp16" ] && FLAGS="$FLAGS --fp16"
echo "$TRTEXEC $FLAGS"
$TRTEXEC $FLAGS 2>&1 | tee "${ENGINE}.build.log"
