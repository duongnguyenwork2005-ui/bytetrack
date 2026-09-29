#!/usr/bin/env bash
# Vong 27 -- smoke CUOI tren FIXTURE voi DUNG cau hinh de xuat configs/phase2/detector/train_bdd5.yaml
# (batch 8, workers 2, khong ghi de gi). Moi model: kiem-cau-hinh (khong train) roi smoke 1 epoch.
set -u
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
SP="C:/Users/Mikuno/AppData/Local/Temp/claude/d--LABOTARY-bytetrack/725dbe02-11c7-4caa-9d6c-980457a66fad/scratchpad/r27"
L=handoff/logs/review27/fixture_gpu
D="$SP/yolo_fixture_bdd5_r27_smoke/data.yaml"
echo "cau hinh: $(sha256sum < configs/phase2/detector/train_bdd5.yaml | cut -c1-16)"
for M in yolov8n yolo11n; do
  TEN=fixture_smoke_cuoi_$M
  $P -u src/train_detector.py --model $M --dry-run --ten $TEN --data "$D" --project "$SP/runs" --ket-qua-dir $L/ket_qua \
     > $L/07_kiem_cau_hinh_$M.log 2>&1
  echo "$M kiem-cau-hinh exit=$? | $(tail -1 $L/07_kiem_cau_hinh_$M.log | cut -c1-200)"
  echo "### $(date -Is) $TEN | RAM trong truoc: $($P -c 'import psutil;print(round(psutil.virtual_memory().available/2**20))') MiB"
  $P -u src/train_detector.py --model $M --che-do smoke --ten $TEN --data "$D" --project "$SP/runs" --ket-qua-dir $L/ket_qua \
     > $L/07_$TEN.log 2>&1
  echo "$TEN exit=$? | $(tail -1 $L/07_$TEN.log | cut -c1-300)"
done
echo "### XONG $(date -Is)"
