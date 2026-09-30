#!/usr/bin/env bash
# Vong 29 -- runner THAT (khong spy) tren dataset FIXTURE v2 cua vong 28 (du lieu tong hop, khong phai BDD that):
#  a) .npy anh den canh 1 anh train -> runner --che-do smoke phai exit 2 TRUOC khi nap model / tao thu muc luot
#  b) .npy anh den canh 1 anh dev   -> nhu (a)
#  c) dataset sach -> smoke 1 epoch YOLOv8n THAT; callback ghi nguon pixel (cache_anh None, npy 0) + du lieu nap khop
# File .npy tao/xoa trong thu muc fixture cua CHINH phien lam viec (scratchpad), khong dung du lieu nguoi dung.
set -u
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
SP="C:/Users/Mikuno/AppData/Local/Temp/claude/d--LABOTARY-bytetrack/725dbe02-11c7-4caa-9d6c-980457a66fad/scratchpad/r27"
DS="$SP/yolo_fixture_bdd5_r28_smoke"
L=handoff/logs/review29/fixture_gpu
echo "cau hinh: $(sha256sum < configs/phase2/detector/train_bdd5.yaml | cut -c1-16) | HEAD $(git rev-parse --short HEAD)"
echo "npy trong dataset truoc khi chay: $(find "$DS/images" -name '*.npy' | wc -l)"
for S in train dev; do
  A=$(ls "$DS/images/$S"/*.jpg | head -1); N="${A%.jpg}.npy"
  $P -c "import cv2,numpy as np,sys;np.save(sys.argv[2], np.zeros_like(cv2.imread(sys.argv[1])))" "$A" "$N"
  TEN=r29_npy_$S
  $P -u src/train_detector.py --model yolov8n --che-do smoke --ten $TEN --data "$DS/data.yaml" \
     --project "$SP/runs_r29" --ket-qua-dir $L/ket_qua > $L/01_$TEN.log 2>&1
  echo "$S exit=$? | run dir ton tai: $([ -e "$SP/runs_r29/$TEN" ] && echo CO || echo khong) | kq ton tai: $([ -e "$L/ket_qua/$TEN.json" ] && echo CO || echo khong) | $(tail -1 $L/01_$TEN.log | cut -c1-330)"
  rm -f "$N"
done
echo "npy con lai: $(find "$DS/images" -name '*.npy' | wc -l)"
TEN=r29_sach_yolov8n
echo "### $(date -Is) $TEN | RAM trong truoc: $($P -c 'import psutil;print(round(psutil.virtual_memory().available/2**20))') MiB"
$P -u src/train_detector.py --model yolov8n --che-do smoke --ten $TEN --data "$DS/data.yaml" \
   --project "$SP/runs_r29" --ket-qua-dir $L/ket_qua > $L/01c_$TEN.log 2>&1
echo "$TEN exit=$? | $(tail -1 $L/01c_$TEN.log | cut -c1-300)"
echo "### XONG $(date -Is)"
