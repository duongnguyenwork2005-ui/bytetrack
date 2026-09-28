#!/usr/bin/env bash
# LUONG TRUC TIEP tren Nano: video / thu muc anh -> TensorRT FP16 (Python 3.6 he thong) -> tracker CV/EKF (env >= 3.8,
# backend NumPy), xu ly TUNG frame qua pipe co gioi han (src/nano_live.py). Chay tu GOC GOI.
#
#   KHOI DONG:  bash scripts/nano/04_live_nano.sh <cv|ekf> <thu_muc_ket_qua_MOI> [them tham so nano_live.py ...]
#     vd:       bash scripts/nano/04_live_nano.sh cv  results_nano/live/thu-cv
#               bash scripts/nano/04_live_nano.sh ekf results_nano/live/video-ekf --source data/smoke/MVI_39031_f0001-0300.mp4
#   DUNG:       touch <thu_muc_ket_qua>/STOP     (hoac Ctrl+C) -> hoan tat frame dang xu ly, dung tien trinh con qua
#               giao thuc, ghi DU phan da xu ly; ma thoat 4 (da_dung). 0 = hoan tat, 2 = loi truoc vong lap,
#               3 = loi giua luong (con chet / mat frame / anh hong) -- ket qua ghi du phan da xu ly, KHONG "hoan tat".
#
# Bien moi truong (tuy chon): PYV (python >= 3.8, mac dinh ~/miniforge3/envs/nano/bin/python -- goi TRUC TIEP, khong
# `conda run`), ENGINE (yolov8n_fp16.engine), SOURCE (data/smoke/MVI_39031_f0001-0300), FPS (25), VIDEO (MVI_39031).
set -u
T=${1:?cv|ekf}; OUT=${2:?thu muc ket qua MOI}; shift 2
PYV=${PYV:-$HOME/miniforge3/envs/nano/bin/python}
ENGINE=${ENGINE:-yolov8n_fp16.engine}
SOURCE=${SOURCE:-data/smoke/MVI_39031_f0001-0300}
ONNX=models/export/yolov8n_640_op12.onnx
EXTRA=()
[ -f "$ONNX" ] && EXTRA+=(--onnx-sha256 "$(sha256sum "$ONNX" | cut -d' ' -f1)")
export BYTETRACK_BACKEND=np
exec "$PYV" src/nano_live.py --source "$SOURCE" --fps "${FPS:-25}" --video-name "${VIDEO:-MVI_39031}" \
     --engine "$ENGINE" "${EXTRA[@]}" --tracker "$T" --out "$OUT" "$@"
