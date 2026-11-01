#!/usr/bin/env bash
# R63 G3: so cap pipeline that 300 JPG MVI_39031, hien thi bat, cung boot. A = baseline (ban sao v16 + engine v16
# build tren board nay), B = ung vien (goi v17-C + engine ung vien). 4 cap CV + 4 cap EKF, thu tu trong cap dao chieu
# xen ke (AB, BA, ...), CV/EKF xen nhau, nghi 20 s. Moi luot qua chay_co_giam_sat.sh (900 s); loi/vi pham -> DUNG.
#   nohup setsid bash cong_cu_v4/g3_so_cap.sh <thu_muc_ra_moi> <engine_ung_vien> <onnx_ung_vien> > <log> 2>&1 < /dev/null &
# Sua 01/11 (review GPT, provenance): truyen --onnx cua ung vien cho nhanh B (luot g3_c1_so_cap_01 ghi nham ONNX v16,
# dinh chinh o g3/04_dinh_chinh_provenance.md); nhanh A dung ONNX v16 trong goi (mac dinh runner).
set -u
OUT=$(readlink -f "${1:?can thu_muc_ra}"); ENG_B=$(readlink -f "${2:?can engine ung vien}")
ONNX_B=$(readlink -f "${3:?can onnx ung vien}")
[ -f "$ONNX_B" ] || { echo "khong thay $ONNX_B"; exit 2; }
R=/home/jetson/nano_run/r63_20261030
PY=/home/jetson/miniforge3/envs/nano/bin/python
GOI_A=$R/baseline/nano_transfer; ENG_A=$R/engine_v16_board_hien_tai_01/yolov8n_640_op12_fp16.engine
GOI_B=$R/v17c/nano_transfer
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$OUT khong rong"; exit 2; fi
mkdir -p "$OUT"
{ echo "bat_dau=$(date -Is)"; echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "mmc1=$(dmesg | grep -m1 'mmc1: new')"; sha256sum "$ENG_A" "$ENG_B" "$ONNX_B" "$R/cong_cu_v4/chay_mot_luot_r63.py" \
  "$GOI_A/src/nano_detector.py" "$GOI_B/src/nano_detector.py" "$GOI_B/src/nano_detect_stream.py"; } > "$OUT/dieu_kien.txt"
luot() {  # luot <ten> <A|B> <tracker>
  local ten=$1 v=$2 t=$3 goi=$GOI_A eng=$ENG_A onnx=models/export/yolov8n_640_op12.onnx
  [ "$v" = B ] && { goi=$GOI_B; eng=$ENG_B; onnx=$ONNX_B; }
  echo "[$(date +%T)] BAT DAU $ten"
  (cd "$goi" && bash "$R/cong_cu_v4/chay_co_giam_sat.sh" "$OUT/$ten" 900 -- \
     "$PY" "$R/cong_cu_v4/chay_mot_luot_r63.py" --tracker "$t" --display on --frames 300 --engine "$eng" --onnx "$onnx" --out "$OUT/$ten/luot")
  local rc=$?
  echo "[$(date +%T)] exit=$rc $(tail -1 "$OUT/$ten/stdout.log" 2>/dev/null)"
  [ $rc -eq 0 ] || { echo "[DUNG] $ten exit=$rc"; exit $rc; }
  sleep 20
}
for cap in 1 2 3 4; do
  for t in cv ekf; do
    k=0; [ $t = ekf ] && k=1
    if [ $(( (cap + k) % 2 )) -eq 1 ]; then thu="A B"; else thu="B A"; fi
    for v in $thu; do luot "cap${cap}_${t}_${v}" "$v" "$t"; done
  done
done
echo "ket_thuc=$(date -Is)" >> "$OUT/dieu_kien.txt"
echo "[XONG]"
