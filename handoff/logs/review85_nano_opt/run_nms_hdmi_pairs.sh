#!/usr/bin/env bash
# R85: balanced v21combo/v22nms pairs; same C1 engine, frames and tracker config.
set -eu
R=/home/jetson/nano_run/r63_20261030
OUT=$R/r85_nms_hdmi_20261105
E=$R/g3_c1_384x640_toc_do_01/engine/yolov8n_384x640_op12_fp16.engine
O=$R/ung_vien_onnx/384x640/yolov8n_384x640_op12.onnx
P=/home/jetson/miniforge3/envs/nano/bin/python
mkdir -p "$OUT"
sha256sum "$E" "$O" "$R/v21combo/nano_transfer/src/nano_detector.py" \
  "$R/v22nms/nano_transfer/src/nano_detector.py" > "$OUT/inputs.sha256"
run() {
  name=$1 pkg=$2 tracker=$3
  target=$OUT/$name
  [ ! -e "$target" ] || { echo "existing target: $target" >&2; exit 2; }
  (cd "$R/$pkg/nano_transfer" && bash "$R/cong_cu_v4/chay_co_giam_sat.sh" \
    "$target" 900 -- "$P" "$R/cong_cu_v4/chay_mot_luot_r63.py" \
    --tracker "$tracker" --display on --frames 300 --engine "$E" --onnx "$O" \
    --out "$target/luot")
  echo "DONE $name"
  sleep 20
}
for pair in 1 2 3 4; do
  for tracker in cv ekf; do
    if [ $((pair % 2)) -eq 1 ]; then
      run "p${pair}_${tracker}_A" v21combo "$tracker"
      run "p${pair}_${tracker}_B" v22nms "$tracker"
    else
      run "p${pair}_${tracker}_B" v22nms "$tracker"
      run "p${pair}_${tracker}_A" v21combo "$tracker"
    fi
  done
done
echo COMPLETE
