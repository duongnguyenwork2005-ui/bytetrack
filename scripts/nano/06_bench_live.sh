#!/usr/bin/env bash
# DO THOI GIAN TOAN LUONG tren Nano (src/nano_live_kiem.py bench) -- CHI chay SAU khi 05_parity_live.sh DAT va dieu
# kien do da duoc CONG BO TRUOC trong file <dieu_kien>. Chay tu GOC GOI. Khong doi nvpmodel / jetson_clocks.
#
#   bash scripts/nano/06_bench_live.sh <dieu_kien.md> [engine] [thu_muc_ket_qua_MOI]
set -u
DK=${1:?file dieu kien do da cong bo truoc}
ENGINE=${2:-yolov8n_fp16.engine}
OUT=${3:-results_nano/live_v6/bench}
PYV=${PYV:-$HOME/miniforge3/envs/nano/bin/python}
mkdir -p "$(dirname "$OUT")"
export BYTETRACK_BACKEND=np
"$PYV" src/nano_live_kiem.py bench --engine "$ENGINE" --out "$OUT" --dieu-kien "$DK" 2>&1 | tee "${OUT%/}.log"
exit "${PIPESTATUS[0]}"
