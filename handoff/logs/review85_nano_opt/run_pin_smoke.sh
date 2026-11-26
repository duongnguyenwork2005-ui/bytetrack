#!/usr/bin/env bash
# R85: output-pinned with copy back to cacheable host memory; optional pinned input.
# Paired four-run no-display diagnostic; same C1 engine and 300 JPG source.
set -eu
R=/home/jetson/nano_run/r63_20261030
E=$R/g3_c1_384x640_toc_do_01/engine/yolov8n_384x640_op12_fp16.engine
O=$R/ung_vien_onnx/384x640/yolov8n_384x640_op12.onnx
P=/home/jetson/miniforge3/envs/nano/bin/python
run() {
  name=$1 pkg=$2 pin=$3
  target=$R/r85_pin_${name}_20261106
  [ ! -e "$target" ] || { echo "existing target $target" >&2; exit 2; }
  (cd "$R/$pkg/nano_transfer" && bash "$R/cong_cu_v4/chay_co_giam_sat.sh" \
    "$target" 900 -- env NANO_PIN_INPUT="$pin" "$P" "$R/cong_cu_v4/chay_mot_luot_r63.py" \
    --tracker cv --display off --frames 300 --engine "$E" --onnx "$O" \
    --out "$target/luot")
  echo "DONE $name"
  sleep 20
}
run base1 v21combo 0
run out v23b_pin_copy 0
run inout v23b_pin_copy 1
run base2 v21combo 0
echo COMPLETE
