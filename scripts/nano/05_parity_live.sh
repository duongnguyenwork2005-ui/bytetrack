#!/usr/bin/env bash
# KIEM CHUNG luong truc tiep tren Nano TRUOC khi do toc do (src/nano_live_kiem.py parity). Chay tu GOC GOI.
#   A. gia lap tren 300 JPG that + cache smoke == pc-ref smoke   B. TensorRT that: live == replay tu cache (CV, EKF)
#   C. mo ta: TRT vs cache PC, TRT lan CV vs lan EKF              D. bien: anh den, mp4, stop-file, SIGINT, kill con, engine sai
#
#   bash scripts/nano/05_parity_live.sh [engine] [thu_muc_ket_qua_MOI]
set -u
ENGINE=${1:-yolov8n_fp16.engine}
OUT=${2:-results_nano/live_v6/parity}
PYV=${PYV:-$HOME/miniforge3/envs/nano/bin/python}
mkdir -p "$(dirname "$OUT")"
export BYTETRACK_BACKEND=np
"$PYV" src/nano_live_kiem.py parity --engine "$ENGINE" --out "$OUT" 2>&1 | tee "${OUT%/}.log"
exit "${PIPESTATUS[0]}"
