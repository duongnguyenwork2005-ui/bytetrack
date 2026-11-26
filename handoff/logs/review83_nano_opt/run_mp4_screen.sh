#!/usr/bin/env bash
set -eu
R=/home/jetson/nano_run/r63_20261030
SRC=/home/jetson/nano_run/v16/nano_transfer/data/smoke/MVI_39031_f0001-0300.mp4
E=$R/g3_c1_384x640_toc_do_01/engine/yolov8n_384x640_op12_fp16.engine
P=/home/jetson/miniforge3/envs/nano/bin/python
export BYTETRACK_BACKEND=np DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority PYTHONDONTWRITEBYTECODE=1
for v in combo base; do
  pkg=v17c
  [ "$v" = combo ] && pkg=v21combo
  out=$R/r83_mp4_${v}_20261105b
  [ ! -e "$out" ] || { echo "existing $out" >&2; exit 2; }
  (cd "$R/$pkg/nano_transfer" && "$P" src/nano_live.py --source "$SRC" --fps 25 \
   --video-name MVI_39031 --engine "$E" \
   --onnx-sha256 5f5b337d2978bd7b5bb3195f6d04e3725f536d5e4cd87d4d1cb7a1c8af984f49 \
   --tracker cv --out "$out" --save-cache detections --cache-root "$out/cache" \
   --warmup-frames 25 --hang-doi 4 --tegrastats-ms 500 --progress-every 100)
  echo "DONE $v"
  sleep 20
done
