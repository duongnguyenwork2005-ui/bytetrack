#!/usr/bin/env bash
set -euo pipefail
cd /home/jetson/nano_run/r155_runtime_accuracy_20261120
for mode in original anchor_rebuilt r352x608 r320x576 r352x608_psa32 r320x576_psa32; do
    LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 capture.py \
        models.json "$mode" dataset.json dev_images.tar cache \
        > "capture_${mode}.log" 2>&1
    tail -1 "capture_${mode}.log"
done
