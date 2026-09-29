#!/usr/bin/env bash
# Vong 27 -- do bo sung tren FIXTURE (sau 05): them lay mau utilization.gpu.
#   a) yolov8n b8 w2 lap lai (deterministic=true, chinh sach de xuat) -> do lap + muc dung GPU
#   b) yolov8n b8 w2 deterministic=false (ban sao cau hinh, CHI de do chi phi tat dinh)
#   c) yolo11n b12 w2 -> batch 12 co vua VRAM cho model lon hon khong
set -u
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
SP="C:/Users/Mikuno/AppData/Local/Temp/claude/d--LABOTARY-bytetrack/725dbe02-11c7-4caa-9d6c-980457a66fad/scratchpad/r27"
L=handoff/logs/review27/fixture_gpu
D="$SP/yolo_fixture_bdd5_r27_smoke/data.yaml"
chay() {  # ten model batch workers config
  echo "### $(date -Is) $1 | RAM trong truoc: $($P -c 'import psutil;print(round(psutil.virtual_memory().available/2**20))') MiB"
  $P -u src/train_detector.py --config $5 --model $2 --che-do smoke --ten $1 --batch $3 --workers $4 --data "$D" \
     --project "$SP/runs" --ket-qua-dir $L/ket_qua > $L/06_$1.log 2>&1
  echo "$1 exit=$? | $(tail -1 $L/06_$1.log | cut -c1-300)"
}
chay fixture_smoke_yolov8n_b8_w2_lap yolov8n 8 2 configs/phase2/detector/train_bdd5.yaml
chay fixture_smoke_yolov8n_b8_w2_khongtatdinh yolov8n 8 2 $L/train_bdd5_khong_tat_dinh.yaml
chay fixture_smoke_yolo11n_b12_w2 yolo11n 12 2 configs/phase2/detector/train_bdd5.yaml
echo "### XONG $(date -Is)"
