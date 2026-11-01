#!/usr/bin/env bash
# R63 G1(a0): do engine cu (c32690cd, build board R62) va engine moi (build board hien tai) CUNG boot:
# trtexec A-E roi bench 300 anh 4 pass (xen ke dong_bo/su_kien). Thu tu xen ke cu->moi->moi->cu de can bang troi
# nhiet. Dung chuoi khi bat ky buoc nao loi/vi pham cong. Chay tu ~/nano_run/r63_20261030:
#   nohup setsid bash cong_cu_v4/g1_a0_hai_engine.sh <thu_muc_ra_moi> > <log> 2>&1 &
set -u
OUT=${1:?can thu_muc_ra}
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$OUT khong rong"; exit 2; fi
mkdir -p "$OUT"
CU=/home/jetson/nano_run/engine/yolov8n_fp16.engine
MOI=/home/jetson/nano_run/r63_20261030/engine_v16_board_hien_tai_01/yolov8n_640_op12_fp16.engine
PY=/usr/bin/python3
GOI=baseline/nano_transfer
ANH=/home/jetson/nano_run/data_full/MVI_39031
{ echo "bat_dau=$(date -Is)"; echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "mmc1=$(dmesg | grep -m1 'mmc1: new')"; sha256sum "$CU" "$MOI"; } > "$OUT/dieu_kien.txt"
buoc() {  # buoc <ten> <lenh...>
  echo "[$(date +%T)] BAT DAU $1"; shift
  "$@"; rc=$?
  echo "[$(date +%T)] exit=$rc"
  [ $rc -eq 0 ] || { echo "[DUNG] buoc loi"; exit $rc; }
  sleep 30
}
buoc trtexec_cu  bash cong_cu_v4/do_trtexec.sh "$CU"  "$OUT/trtexec_cu"
buoc trtexec_moi bash cong_cu_v4/do_trtexec.sh "$MOI" "$OUT/trtexec_moi"
for t in moi cu; do
  E=$CU; [ $t = moi ] && E=$MOI
  buoc bench_$t bash cong_cu_v4/chay_co_giam_sat.sh "$OUT/bench_$t" 1200 -- \
    $PY cong_cu_v4/bench_engine_pipeline.py --goi $GOI --engine "$E" --anh $ANH \
    --ra "$OUT/bench_$t/ket_qua" --so-anh 300 --pass 4 --warmup 25
done
echo "ket_thuc=$(date -Is)" >> "$OUT/dieu_kien.txt"
echo "[XONG]"
