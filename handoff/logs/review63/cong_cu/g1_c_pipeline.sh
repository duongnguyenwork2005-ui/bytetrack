#!/usr/bin/env bash
# R63 G1(c): baseline pipeline that 300 JPG MVI_39031 voi engine v16 build tren board hien tai, hien thi bat,
# 4 luot CV + 4 luot EKF thu tu can bang (cv ekf ekf cv cv ekf ekf cv), nghi 20 s, cung boot. Moi luot qua
# chay_co_giam_sat.sh (timeout 900 s); loi/vi pham cong -> DUNG chuoi. Chay tu ~/nano_run/r63_20261030:
#   nohup setsid bash cong_cu_v4/g1_c_pipeline.sh <thu_muc_ra_moi> [engine] > <log> 2>&1 < /dev/null &
set -u
OUT=$(readlink -f "${1:?can thu_muc_ra}")
ENG=${2:-/home/jetson/nano_run/r63_20261030/engine_v16_board_hien_tai_01/yolov8n_640_op12_fp16.engine}
R=/home/jetson/nano_run/r63_20261030
PY=/home/jetson/miniforge3/envs/nano/bin/python
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$OUT khong rong"; exit 2; fi
mkdir -p "$OUT"
{ echo "bat_dau=$(date -Is)"; echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "mmc1=$(dmesg | grep -m1 'mmc1: new')"; sha256sum "$ENG" "$R/cong_cu_v4/chay_mot_luot_r63.py"; } > "$OUT/dieu_kien.txt"
cd "$R/baseline/nano_transfer" || exit 2
i=0
for t in cv ekf ekf cv cv ekf ekf cv; do
  i=$((i+1)); ten=$(printf "%02d_%s_on" $i $t)
  echo "[$(date +%T)] BAT DAU $ten"
  bash "$R/cong_cu_v4/chay_co_giam_sat.sh" "$OUT/$ten" 900 -- \
    "$PY" "$R/cong_cu_v4/chay_mot_luot_r63.py" --tracker $t --display on --frames 300 --engine "$ENG" --out "$OUT/$ten/luot"
  rc=$?
  echo "[$(date +%T)] exit=$rc $(tail -1 "$OUT/$ten/stdout.log" 2>/dev/null)"
  [ $rc -eq 0 ] || { echo "[DUNG] $ten exit=$rc"; exit $rc; }
  sleep 20
done
echo "ket_thuc=$(date -Is)" >> "$OUT/dieu_kien.txt"
echo "[XONG]"
