#!/usr/bin/env bash
# R63 - build engine FP16 tren Nano tu ONNX ung vien, CUNG co voi engine v16 (scripts/nano/01_build_engine.sh:
# --workspace=1024 --fp16 --verbose). Chi chay SAU khi ke hoach R63 duoc duyet. Khong ghi de engine co san.
# Chay qua chay_co_giam_sat.sh (timeout 1800 s, nhiet/swap/kernel/boot_id). Cach dung:
#   bash build_engine_nano.sh <model.onnx> <thu_muc_ra_moi>
set -u
ONNX=${1:?can onnx}; OUT=${2:?can thu_muc_ra}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
DIR=$(cd "$(dirname "$0")" && pwd)
[ -x "$TRTEXEC" ] || { echo "khong thay $TRTEXEC"; exit 2; }
[ -f "$ONNX" ] || { echo "khong thay $ONNX"; exit 2; }
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$OUT khong rong"; exit 2; fi
mkdir -p "$OUT"
ENG="$OUT/$(basename "$ONNX" .onnx)_fp16.engine"
{
  echo "date=$(date -Is)"; echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "onnx=$(readlink -f "$ONNX")"; echo "onnx_sha256=$(sha256sum "$ONNX" | cut -d' ' -f1)"
  echo "lenh=$TRTEXEC --onnx=$ONNX --saveEngine=$ENG --workspace=1024 --fp16 --verbose"
  echo "nvpmodel=$(nvpmodel -q 2>/dev/null | tr '\n' ' ')"
} > "$OUT/build_dieu_kien.txt"
bash "$DIR/chay_co_giam_sat.sh" "$OUT/build" 1800 -- \
  "$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENG" --workspace=1024 --fp16 --verbose
rc=$?
echo "exit=$rc" >> "$OUT/build_dieu_kien.txt"
if [ $rc -eq 0 ] && [ -f "$ENG" ]; then
  echo "engine_sha256=$(sha256sum "$ENG" | cut -d' ' -f1) bytes=$(stat -c %s "$ENG")" >> "$OUT/build_dieu_kien.txt"
fi
grep -E "Engine built|PASSED|FAILED|Error" "$OUT/build/stdout.log" | tail -5
exit $rc
