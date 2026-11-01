#!/usr/bin/env bash
# R63 G3 (toc do nhanh, khong can sua src/): build engine FP16 ung vien tren Nano (cung co v16) -> trtexec A-E ->
# bench 300 anh 4 pass voi ung vien, roi bench lai baseline (engine v16 build tren board nay) cung boot de so.
# Dung chuoi khi loi/vi pham cong. Chay tu ~/nano_run/r63_20261030:
#   nohup setsid bash cong_cu_v4/g3_toc_do_ung_vien.sh <onnx> <thu_muc_ra_moi> [engine_da_build] > <log> 2>&1 < /dev/null &
# engine_da_build: bo buoc build, dung engine co san (vd build truoc do bi dung chuoi chi vi swap tang luc build).
set -u
ONNX=$(readlink -f "${1:?can onnx}"); OUT=${2:?can thu_muc_ra}; ENG_CO_SAN=${3:-}
BASE=/home/jetson/nano_run/r63_20261030/engine_v16_board_hien_tai_01/yolov8n_640_op12_fp16.engine
PY=/usr/bin/python3; GOI=baseline/nano_transfer; ANH=/home/jetson/nano_run/data_full/MVI_39031
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$OUT khong rong"; exit 2; fi
mkdir -p "$OUT"
{ echo "bat_dau=$(date -Is)"; echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "mmc1=$(dmesg | grep -m1 'mmc1: new')"; sha256sum "$ONNX" "$BASE"; } > "$OUT/dieu_kien.txt"
buoc() { echo "[$(date +%T)] BAT DAU $1"; shift; "$@"; rc=$?; echo "[$(date +%T)] exit=$rc"
         [ $rc -eq 0 ] || { echo "[DUNG] buoc loi"; exit $rc; }; sleep 30; }
if [ -n "$ENG_CO_SAN" ]; then
  ENG=$(readlink -f "$ENG_CO_SAN"); echo "engine_co_san=$ENG $(sha256sum "$ENG" | cut -d' ' -f1)" >> "$OUT/dieu_kien.txt"
else
  buoc build bash cong_cu_v4/build_engine_nano.sh "$ONNX" "$OUT/engine"
  ENG="$OUT/engine/$(basename "$ONNX" .onnx)_fp16.engine"
fi
buoc trtexec bash cong_cu_v4/do_trtexec.sh "$ENG" "$OUT/trtexec"
buoc bench_ung_vien bash cong_cu_v4/chay_co_giam_sat.sh "$OUT/bench_ung_vien" 1200 -- \
  $PY cong_cu_v4/bench_engine_pipeline.py --goi $GOI --engine "$ENG" --anh $ANH \
  --ra "$OUT/bench_ung_vien/ket_qua" --so-anh 300 --pass 4 --warmup 25
buoc bench_baseline bash cong_cu_v4/chay_co_giam_sat.sh "$OUT/bench_baseline" 1200 -- \
  $PY cong_cu_v4/bench_engine_pipeline.py --goi $GOI --engine "$BASE" --anh $ANH \
  --ra "$OUT/bench_baseline/ket_qua" --so-anh 300 --pass 4 --warmup 25
echo "ket_thuc=$(date -Is)" >> "$OUT/dieu_kien.txt"
echo "[XONG]"
