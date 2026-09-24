#!/usr/bin/env bash
# Benchmark tracker CV <-> EKF tu CUNG det cache tren Nano -- LAN 2 (lan 1 that bai: thieu BYTETRACK_BACKEND=np,
# warm-up --pairs 2 bi cong cu tu choi). KHONG sua code. Chay tu goc goi v5:
#   bash bench_v5_lan2.sh [CR]      CR mac dinh: ~/miniforge3/bin/conda run -n nano python
# Output MOI (khong ghi de): results_nano/bench_cv_ekf/lan2/, split NANO-bench-warmup / NANO-bench.
set -u
CR=${1:-"$HOME/miniforge3/bin/conda run -n nano python"}
export BYTETRACK_BACKEND=np                         # BAT BUOC (README scripts/nano muc 187)
OUT=results_nano/bench_cv_ekf/lan2
mkdir -p "$OUT"
[ -e data/processed/trackers/NANO-bench ] && { echo "[LOI] NANO-bench da ton tai -> dung, khong ghi de"; exit 8; }
[ -e data/processed/trackers/NANO-bench-warmup ] && { echo "[LOI] NANO-bench-warmup da ton tai -> dung"; exit 8; }
DK="--videos MVI_39031 --conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --fps 25 --track-buffer-sec 1.2 --elapsed-mode steps"

echo "### WARM-UP (tach khoi so lieu chinh): 1 luot CV + 1 luot EKF, split NANO-bench-warmup $(date -Is)"
for T in cv ekf; do
  Y=bytetrack.yaml; [ "$T" = ekf ] && Y=configs/bytetrack_ekf_ctrv.yaml
  S=$(date +%s%N)
  $CR src/run_tracking.py --cache-id yolov8n-coco_conf0.05_all --split-name NANO-bench-warmup \
      --tracker-yaml "$Y" --tracker-name "warmup-$T" $DK
  RC=$?; echo "warmup $T exit=$RC wall=$(( ($(date +%s%N) - S) / 1000000 )) ms"
done

echo "### BENCH CHINH: 6 cap (3 AB + 3 BA) = 12 luot, nghi 20 s giua luot $(date -Is)"
$CR src/nano_bench_tracker.py --cache-id yolov8n-coco_conf0.05_all --split-name NANO-bench \
  --a cv:bytetrack.yaml --b ekf:configs/bytetrack_ekf_ctrv.yaml $DK \
  --pairs 6 --warmup-frames 50 --nghi-giay 20 --tegrastats-ms 500 --tag bench --out "$OUT"
echo "bench exit=$?"
echo "### XONG $(date -Is)"
