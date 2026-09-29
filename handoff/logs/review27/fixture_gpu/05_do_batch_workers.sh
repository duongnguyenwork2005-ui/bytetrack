#!/usr/bin/env bash
# Vong 27 -- smoke duong chay + do VRAM / toc do / RAM tren FIXTURE 1280x720 (khong phai BDD that).
# Moi luot: 1 epoch, cung dataset smoke fixture (719 train / 90 dev), cung chinh sach train_bdd5.yaml;
# chi ghi de batch / workers (ghi vao ket qua). Chay tuan tu, khong song song.
set -u
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
SP="C:/Users/Mikuno/AppData/Local/Temp/claude/d--LABOTARY-bytetrack/725dbe02-11c7-4caa-9d6c-980457a66fad/scratchpad/r27"
L=handoff/logs/review27/fixture_gpu
D="$SP/yolo_fixture_bdd5_r27_smoke/data.yaml"
for X in yolov8n:16:2 yolov8n:8:2 yolov8n:8:0 yolo11n:16:2 yolo11n:8:2; do
  M=${X%%:*}; R=${X#*:}; B=${R%%:*}; W=${R#*:}
  TEN=fixture_smoke_${M}_b${B}_w${W}
  echo "### $(date -Is) $TEN | RAM trong truoc: $($P -c 'import psutil;print(round(psutil.virtual_memory().available/2**20))') MiB"
  $P -u src/train_detector.py --model $M --che-do smoke --ten $TEN --batch $B --workers $W --data "$D" \
     --project "$SP/runs" --ket-qua-dir $L/ket_qua > $L/05_${TEN}.log 2>&1
  echo "$TEN exit=$? | $(tail -1 $L/05_${TEN}.log | cut -c1-300)"
done
echo "### XONG $(date -Is)"
