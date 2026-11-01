#!/usr/bin/env bash
# R63 G3: duong toc do cho cac muc C2 352x576, C3 288x512, C4 256x448 (chi toc do). Moi muc: build_engine_nano.sh ->
# g3_toc_do_ung_vien.sh <onnx> <ra> <engine_da_build> (trtexec A-E, bench ung vien, bench baseline cung boot).
# Sua 01/11 (review GPT): BO co che tu mien cong — build vi pham BAT KY cong nao (ke ca "swap tang") -> DUNG chuoi va
# giu bang chung; quyet dinh dung engine do (neu co) la quyet dinh rieng, ghi cong khai. Ban truoc (luot
# g3_duong_toc_do_01, 01/11) da tu tiep tuc khi C2/C3 chi vi pham swap — ghi la sai lech o g3/03_duong_toc_do_c2_c4.md.
# Luu y lich: theo ke hoach (01_ke_hoach_r63.md, G3/G4 xen ke) chi xuong muc nho hon sau khi muc truoc dat chat luong dev.
#   nohup setsid bash cong_cu_v4/g3_duong_toc_do.sh <tien_to_ra> [muc ...] > <log> 2>&1 < /dev/null &
set -u
TT=${1:?can tien_to_ra}; shift
MUC=${*:-352x576 288x512 256x448}
for hw in $MUC; do
  ONNX=ung_vien_onnx/$hw/yolov8n_${hw}_op12.onnx
  [ -f "$ONNX" ] || { echo "[DUNG] thieu $ONNX"; exit 2; }
  B=${TT}_${hw}_build; O=${TT}_${hw}_toc_do
  echo "[$(date +%T)] BUILD $hw"
  bash cong_cu_v4/build_engine_nano.sh "$ONNX" "$B" || { echo "[DUNG] build $hw exit=$?"; exit 10; }
  ENG="$B/yolov8n_${hw}_op12_fp16.engine"
  sleep 30
  echo "[$(date +%T)] DO $hw"
  bash cong_cu_v4/g3_toc_do_ung_vien.sh "$ONNX" "$O" "$ENG" || { echo "[DUNG] do $hw"; exit 3; }
  sleep 30
done
echo "[XONG]"
