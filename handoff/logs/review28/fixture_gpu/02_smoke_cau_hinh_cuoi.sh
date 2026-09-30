#!/usr/bin/env bash
# Vong 28 -- chay lai smoke 1 epoch (FIXTURE) sau khi sua CHU THICH batch trong train_bdd5.yaml (chi comment),
# de bang chung gan voi sha256 cau hinh cuoi. Dataset v2 da xuat o 01b (khong xuat lai).
set -u
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
SP="C:/Users/Mikuno/AppData/Local/Temp/claude/d--LABOTARY-bytetrack/725dbe02-11c7-4caa-9d6c-980457a66fad/scratchpad/r27"
L=handoff/logs/review28/fixture_gpu
D="$SP/yolo_fixture_bdd5_r28_smoke/data.yaml"
echo "cau hinh: $(sha256sum < configs/phase2/detector/train_bdd5.yaml | cut -c1-16)"
for M in yolov8n yolo11n; do
  TEN=fixture_smoke_r28_cuoi_$M
  echo "### $(date -Is) $TEN | RAM trong truoc: $($P -c 'import psutil;print(round(psutil.virtual_memory().available/2**20))') MiB"
  $P -u src/train_detector.py --model $M --che-do smoke --ten $TEN --data "$D" --project "$SP/runs_r28" --ket-qua-dir $L/ket_qua \
     > $L/02_$TEN.log 2>&1
  echo "$TEN exit=$? | $(tail -1 $L/02_$TEN.log | cut -c1-300)"
done
echo "### XONG $(date -Is)"
