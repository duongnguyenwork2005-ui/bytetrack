#!/usr/bin/env bash
# Vong 35 -- smoke LAN 2, TUAN TU YOLOv8n roi YOLO11n (khong chay dong thoi), chay tu goc repo:
#   bash handoff/logs/review35/chay_smoke_lan2.sh
# Khac lan 1 (chay_smoke.sh, giu nguyen): (a) bo gan moc gio da sua (bf7c0a2, khong con lam dut pipe);
# (b) --dry-run them --data <dataset smoke> vi kiem-cau-hinh mac dinh doc dataset DAY DU (chua xuat vong nay);
# (c) ten luot moi *_smoke_lan2 -- runner tu choi ghi de luot/ket qua lan 1 (giu nguyen lam bang chung).
# Runner, cau hinh train_bdd5.yaml (batch 8, workers 2, imgsz 640, seed 0, amp false, cache false ...), model,
# checkpoint va dataset KHONG doi.
set -u
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
L=handoff/logs/review35
DS=data/yolo_bdd5_v1_smoke/data.yaml
ram() { "$P" -c "import psutil; v=psutil.virtual_memory(); print('%.0f MiB con trong / %.0f MiB (dung %.0f%%)' % (v.available/2**20, v.total/2**20, v.percent))"; }
dia() { "$P" -c "import shutil; print('%.2f GB con trong tren D:' % (shutil.disk_usage('D:/').free/1e9))"; }
for M in yolov8n yolo11n; do
  D="$L/11_dry_run_$M.log"
  { echo "# $(date '+%Y-%m-%d %H:%M:%S %z') HEAD $(git rev-parse --short HEAD)"; echo "# lenh: python src/train_detector.py --model $M --dry-run --data $DS"; } > "$D"
  "$P" src/train_detector.py --model "$M" --dry-run --data "$DS" >> "$D" 2>&1
  echo "# exit=$?" >> "$D"

  TEN="train_bdd5_v1_${M}_smoke_lan2"
  META="$L/12_smoke_${M}_lan2.meta.txt"
  { echo "# $(date '+%Y-%m-%d %H:%M:%S %z') HEAD $(git rev-parse --short HEAD)"
    echo "# lenh: python -u src/train_detector.py --model $M --che-do smoke --ten $TEN 2>&1 | python $L/do_tai_nguyen_ngoai.py moc-gio --ra $L/12_smoke_${M}_lan2.log"
    echo "ram_truoc: $(ram)"; echo "dia_truoc: $(dia)"
    echo "gpu_truoc: $(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader)"; } > "$META"
  "$P" "$L/do_tai_nguyen_ngoai.py" ram --mau train_detector.py --ra "$L/12_ram_tien_trinh_${M}_lan2.json" > /dev/null 2>&1 &
  RAMPID=$!
  T0=$(date +%s)
  "$P" -u src/train_detector.py --model "$M" --che-do smoke --ten "$TEN" 2>&1 | "$P" "$L/do_tai_nguyen_ngoai.py" moc-gio --ra "$L/12_smoke_${M}_lan2.log"
  RC=("${PIPESTATUS[@]}")
  T1=$(date +%s)
  wait "$RAMPID"
  { echo "exit_runner=${RC[0]}"; echo "exit_moc_gio=${RC[1]}"; echo "wall_s=$((T1 - T0))"; echo "ram_sau: $(ram)"; echo "dia_sau: $(dia)"
    echo "gpu_sau: $(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader)"
    echo "run_dir: $(du -sb "models/finetune_phase2/$TEN" 2>/dev/null)"
    ls -l "models/finetune_phase2/$TEN/weights" 2>/dev/null | sed 's/^/weights: /'
    echo "# xong $(date '+%H:%M:%S')"; } >> "$META"
done
