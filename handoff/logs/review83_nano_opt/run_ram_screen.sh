#!/usr/bin/env bash
# Same 300 JPG source in tmpfs for every package; isolates code from SD decode variance.
set -eu
R=/home/jetson/nano_run/r63_20261030
E=$R/g3_c1_384x640_toc_do_01/engine/yolov8n_384x640_op12_fp16.engine
O=$R/ung_vien_onnx/384x640/yolov8n_384x640_op12.onnx
P=/home/jetson/miniforge3/envs/nano/bin/python
run() {
  name=$1
  pkg=$2
  (cd "$R/r83ram/$pkg" && "$P" "$R/cong_cu_v4/chay_mot_luot_r63.py" \
    --tracker cv --display off --frames 300 --engine "$E" --onnx "$O" \
    --out "$R/r83_ram_${name}_20261104")
}
run base1 base
run combo combo
run lut lut
run base2 base
