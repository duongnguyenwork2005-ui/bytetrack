#!/usr/bin/env bash
# Vong 37 -- PILOT CHAN DOAN (nguoi dung duyet 2026-10-02): YOLOv8n ROI YOLO11n, moi model MOT luot 2 epoch, TUAN TU tren
# GPU PC, dataset smoke da khoa. Chay tu goc repo:  bash handoff/logs/review37/chay_pilot.sh
# Moi model:
#   (1) kiem-pilot (KHONG train) ngay truoc khi chay, so khop voi file KHOA da commit 01_khoa_cau_hinh_pilot_<M>.json;
#       khac khoa -> DUNG, khong train (exit 4).
#   (2) bo do RAM ben ngoai (psutil, 1 s, mau tho CSV) + pilot; stdout gan moc gio (moc-gio doc byte, khong dut pipe).
# Runner tu choi ghi de luot / ket qua cu. Khong doi model, batch, anh, epoch, du lieu; khong chay them luot.
set -u
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
L=handoff/logs/review37
CFG=configs/phase2/detector/train_bdd5_pilot.yaml
ram() { "$P" -c "import psutil; v=psutil.virtual_memory(); print('%.0f MiB con trong / %.0f MiB (dung %.0f%%)' % (v.available/2**20, v.total/2**20, v.percent))"; }
dia() { "$P" -c "import shutil; print('%.2f GB con trong tren D:' % (shutil.disk_usage('D:/').free/1e9))"; }
for M in yolov8n yolo11n; do
  K="$L/05_kiem_pilot_truoc_chay_$M.log"
  { echo "# $(date '+%Y-%m-%d %H:%M:%S %z') HEAD $(git rev-parse --short HEAD)"; echo "# lenh: python src/train_detector.py --config $CFG --model $M --che-do kiem-pilot"; } > "$K"
  "$P" src/train_detector.py --config "$CFG" --model "$M" --che-do kiem-pilot >> "$K" 2>&1
  echo "# exit=$?" >> "$K"
  S="$L/05_so_khop_khoa_$M.log"
  "$P" "$L/so_khop_khoa.py" "$L/01_khoa_cau_hinh_pilot_$M.json" "$K" > "$S" 2>&1
  SK=$?
  echo "# exit=$SK" >> "$S"
  if [ "$SK" != 0 ]; then echo "[DUNG] $M: cau hinh khac khoa da commit -- khong chay pilot (xem $S)"; exit 4; fi

  META="$L/06_pilot_${M}.meta.txt"
  { echo "# $(date '+%Y-%m-%d %H:%M:%S %z') HEAD $(git rev-parse --short HEAD)"
    echo "# lenh: python -u src/train_detector.py --config $CFG --model $M --che-do pilot --xac-nhan-pilot 2>&1 | python $L/do_tai_nguyen_ngoai.py moc-gio --ra $L/06_pilot_$M.log"
    echo "ram_truoc: $(ram)"; echo "dia_truoc: $(dia)"
    echo "gpu_truoc: $(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader)"; } > "$META"
  "$P" "$L/do_tai_nguyen_ngoai.py" ram --mau train_detector.py --ra "$L/06_ram_ngoai_$M.json" --tho "$L/06_ram_ngoai_${M}_mau_tho.csv" > /dev/null 2>&1 &
  RAMPID=$!
  T0=$(date +%s)
  "$P" -u src/train_detector.py --config "$CFG" --model "$M" --che-do pilot --xac-nhan-pilot 2>&1 | "$P" "$L/do_tai_nguyen_ngoai.py" moc-gio --ra "$L/06_pilot_$M.log"
  RC=("${PIPESTATUS[@]}")
  T1=$(date +%s)
  wait "$RAMPID"
  { echo "exit_runner=${RC[0]}"; echo "exit_moc_gio=${RC[1]}"; echo "wall_s=$((T1 - T0))"; echo "ram_sau: $(ram)"; echo "dia_sau: $(dia)"
    echo "gpu_sau: $(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader)"
    echo "run_dir: $(du -sb "models/finetune_phase2/train_bdd5_v1_${M}_pilot" 2>/dev/null)"
    ls -l "models/finetune_phase2/train_bdd5_v1_${M}_pilot/weights" "models/finetune_phase2/train_bdd5_v1_${M}_pilot/chan_doan" 2>/dev/null | sed 's/^/  /'
    echo "# xong $(date '+%H:%M:%S')"; } >> "$META"
done
