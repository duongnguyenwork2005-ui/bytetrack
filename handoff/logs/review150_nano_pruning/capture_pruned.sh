#!/usr/bin/env bash
set -euo pipefail
cd /home/jetson/nano_run/r150_pruning_20261120
audit=/home/jetson/nano_run/r155_runtime_accuracy_20261120
for mode in anchor control channel25_KD; do
    LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      python3 "$audit/capture.py" models.json "$mode" "$audit/dataset.json" \
      "$audit/dev_images.tar" cache > "capture_${mode}.log" 2>&1
    printf 'PRUNING_AP_CAPTURE_COMPLETE %s\n' "$mode"
done
