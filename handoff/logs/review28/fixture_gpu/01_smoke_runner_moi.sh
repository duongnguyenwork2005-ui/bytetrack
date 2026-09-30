#!/usr/bin/env bash
# Vong 28 -- smoke GPU tren FIXTURE voi runner DA SUA (khong phai BDD that, khong phai training dai):
#  a) runner moi --dry-run tren dataset fixture VONG 27 (manifest v1) -> phai TU CHOI (khong bo sung hash)
#  b) xuat lai tu cung nguon fixture + split da khoa vong 27 -> dataset v2 (tap con smoke)
#  c) smoke 1 epoch YOLOv8n + YOLO11n, cau hinh de xuat khong ghi de -> kiem hop dong + callback du lieu da nap trong trainer THAT
set -u
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
SP="C:/Users/Mikuno/AppData/Local/Temp/claude/d--LABOTARY-bytetrack/725dbe02-11c7-4caa-9d6c-980457a66fad/scratchpad/r27"
L=handoff/logs/review28/fixture_gpu
echo "### a) runner moi tren dataset fixture v1 (vong 27): $SP/yolo_fixture_bdd5_r27_smoke"
$P -u src/train_detector.py --model yolov8n --dry-run --data "$SP/yolo_fixture_bdd5_r27_smoke/data.yaml" \
   --project "$SP/runs_r28" --ket-qua-dir $L/ket_qua > $L/01a_dry_run_dataset_v1.log 2>&1
echo "a exit=$? | $(tail -1 $L/01a_dry_run_dataset_v1.log | cut -c1-400)"
echo "### b) xuat lai tap con smoke (manifest v2)"
$P -u src/prepare_mixed_yolo_dataset.py --config $L/bdd5_fixture_r28.yaml --buoc xuat --tap-con smoke > $L/01b_xuat_v2.log 2>&1
echo "b exit=$? | $(grep -c . $L/01b_xuat_v2.log) dong"
D="$SP/yolo_fixture_bdd5_r28_smoke/data.yaml"
for M in yolov8n yolo11n; do
  TEN=fixture_smoke_r28_$M
  echo "### c) $(date -Is) $TEN | RAM trong truoc: $($P -c 'import psutil;print(round(psutil.virtual_memory().available/2**20))') MiB"
  $P -u src/train_detector.py --model $M --che-do smoke --ten $TEN --data "$D" --project "$SP/runs_r28" --ket-qua-dir $L/ket_qua \
     > $L/01c_$TEN.log 2>&1
  echo "$TEN exit=$? | $(tail -1 $L/01c_$TEN.log | cut -c1-300)"
done
echo "### XONG $(date -Is)"
