#!/usr/bin/env bash
# Vong 35 -- smoke TUAN TU YOLOv8n roi YOLO11n (khong chay dong thoi), chay tu goc repo:
#   bash handoff/logs/review35/chay_smoke.sh
# Moi model: (1) --dry-run (kiem cau hinh, khong train); (2) do RAM tien trinh ben ngoai (nen) + smoke 1 epoch, stdout gan
# moc gio. Runner va cau hinh giu nguyen (batch 8, workers 2, imgsz 640, seed 0 ... theo train_bdd5.yaml); khong ghi de luot cu.
set -u
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
L=handoff/logs/review35
ram() { "$P" -c "import psutil; v=psutil.virtual_memory(); print('%.0f MiB con trong / %.0f MiB (dung %.0f%%)' % (v.available/2**20, v.total/2**20, v.percent))"; }
for M in yolov8n yolo11n; do
  D="$L/07_dry_run_$M.log"
  { echo "# $(date '+%Y-%m-%d %H:%M:%S %z') HEAD $(git rev-parse --short HEAD)"; echo "# lenh: python src/train_detector.py --model $M --dry-run"; } > "$D"
  "$P" src/train_detector.py --model "$M" --dry-run >> "$D" 2>&1
  echo "# exit=$?" >> "$D"

  META="$L/08_smoke_$M.meta.txt"
  { echo "# $(date '+%Y-%m-%d %H:%M:%S %z') HEAD $(git rev-parse --short HEAD)"; echo "# lenh: python -u src/train_detector.py --model $M --che-do smoke"
    echo "ram_truoc: $(ram)"; echo "gpu_truoc: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"; } > "$META"
  "$P" "$L/do_tai_nguyen_ngoai.py" ram --mau train_detector.py --ra "$L/08_ram_tien_trinh_$M.json" > /dev/null 2>&1 &
  RAMPID=$!
  T0=$(date +%s)
  "$P" -u src/train_detector.py --model "$M" --che-do smoke 2>&1 | "$P" "$L/do_tai_nguyen_ngoai.py" moc-gio --ra "$L/08_smoke_$M.log"
  RC=${PIPESTATUS[0]}
  T1=$(date +%s)
  wait "$RAMPID"
  { echo "exit=$RC"; echo "wall_s=$((T1 - T0))"; echo "ram_sau: $(ram)"; echo "# xong $(date '+%H:%M:%S')"; } >> "$META"
done
