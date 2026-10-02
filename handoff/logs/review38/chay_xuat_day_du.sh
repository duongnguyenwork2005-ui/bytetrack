#!/usr/bin/env bash
# Vong 38 -- XUAT dataset YOLO DAY DU tu BDD100K MOT train theo split da khoa, buoc frame 5 (nguoi dung duyet).
# Chay tu goc repo:  bash handoff/logs/review38/chay_xuat_day_du.sh
# Dung NGUYEN cong cu da nghiem thu (prepare_mixed_yolo_dataset.py --buoc xuat, khong --tap-con): kiem trung tap danh
# gia truoc khi ghi, ghi vao <dau_ra>.dang_tao roi doi ten, kiem hop dong v2 sau khi xuat. Tu choi neu dich da ton tai.
# Do kem: thoi gian (date), RAM (bo do ngoai vong 37, mau tho 1 s), dia truoc/sau. KHONG train, KHONG sua du lieu nguon.
set -u
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
L=handoff/logs/review38
C=configs/phase2/detector/bdd5_train.yaml
ram() { "$P" -c "import psutil; v=psutil.virtual_memory(); print('%.0f MiB con trong / %.0f MiB (dung %.0f%%)' % (v.available/2**20, v.total/2**20, v.percent))"; }
dia() { "$P" -c "import shutil; print('%.2f GB con trong tren D:' % (shutil.disk_usage('D:/').free/1e9))"; }
META="$L/03_xuat_day_du.meta.txt"
{ echo "# $(date '+%Y-%m-%d %H:%M:%S %z') HEAD $(git rev-parse --short HEAD)"
  echo "# lenh: python -u src/prepare_mixed_yolo_dataset.py --config $C --buoc xuat"
  echo "ram_truoc: $(ram)"; echo "dia_truoc: $(dia)"; } > "$META"
"$P" handoff/logs/review37/do_tai_nguyen_ngoai.py ram --mau prepare_mixed_yolo_dataset.py --ra "$L/03_ram_xuat.json" --tho "$L/03_ram_xuat_mau_tho.csv" > /dev/null 2>&1 &
RAMPID=$!
T0=$(date +%s)
{ echo "# $(date '+%Y-%m-%d %H:%M:%S %z') HEAD $(git rev-parse --short HEAD)"
  echo "# lenh: python -u src/prepare_mixed_yolo_dataset.py --config $C --buoc xuat"
  PYTHONIOENCODING=utf-8 "$P" -u src/prepare_mixed_yolo_dataset.py --config "$C" --buoc xuat
  echo "# exit=$?"
  echo "# xong $(date '+%Y-%m-%d %H:%M:%S %z')"; } > "$L/03_xuat_day_du.log" 2>&1
T1=$(date +%s)
wait "$RAMPID"
{ echo "wall_s=$((T1 - T0))"; echo "ram_sau: $(ram)"; echo "dia_sau: $(dia)"
  echo "dataset: $(du -sb data/yolo_bdd5_v1 2>/dev/null)"
  echo "so_file: $(find data/yolo_bdd5_v1 -type f 2>/dev/null | wc -l)"
  echo "# xong $(date '+%H:%M:%S')"; } >> "$META"
