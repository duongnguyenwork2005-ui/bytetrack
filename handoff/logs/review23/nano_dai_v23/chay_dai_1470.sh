#!/usr/bin/env bash
# Vong 23 -- chay DAY DU 1470 frame MVI_39031 qua luong truc tiep da nghiem thu (c4b39fe; goi v8 tren Nano):
#   doc anh -> TensorRT FP16 -> tracker, KHONG render, 4 luot CV -> EKF -> EKF -> CV (hai cap AB/BA), nghi giua luot.
# Moi luot: live (--save-cache) -> replay bang CHINH replay_argv trong run_meta -> nano_parity -> so byte MOT -> kiem tai
# nguyen (tien trinh con con song?, MemAvailable, swap, nhiet truoc/sau). Giu MOI luot, ke ca that bai.
# Script nay CHI goi entry point da nghiem thu (nano_live.py, run_tracking/nano_replay qua replay_argv, nano_parity.py).
#
#   cd ~/nano_run/v8/nano_transfer && bash chay_dai_1470.sh <thu_muc_ket_qua_MOI>
set -u
PYV=${PYV:-$HOME/miniforge3/envs/nano/bin/python}
ENGINE=${ENGINE:-$HOME/nano_run/engine/yolov8n_fp16.engine}
SRC=${SRC:-$HOME/nano_run/data_full/MVI_39031}
OUT=${1:?thu muc ket qua MOI}
NGHI=${NGHI:-20}
THU_TU=${THU_TU:-"cv ekf ekf cv"}
export BYTETRACK_BACKEND=np
[ -e "$OUT" ] && { echo "[LOI] $OUT da ton tai -> dung, khong ghi de"; exit 8; }
mkdir -p "$OUT"
ONNX_SHA=$(sha256sum models/export/yolov8n_640_op12.onnx | cut -d' ' -f1)
TAG=$(date +%Y%m%d%H%M%S)
SPLIT=NANO-dai-$TAG
TN="$OUT/tai_nguyen.csv"
echo "luot,cau_hinh,moc,thoi_diem,uptime_s,boot,mem_available_kB,swap_used_MB,cpu_therm_C,gpu_therm_C,so_tien_trinh_con_dang_chay,ext4_errors" > "$TN"

anh_chup() {  # $1 luot $2 cau_hinh $3 moc
  local ma=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  local sw=$(free -m | awk '/Swap/ {print $3}')
  local ct=$(( $(cat /sys/devices/virtual/thermal/thermal_zone1/temp) / 1000 ))
  local gt=$(( $(cat /sys/devices/virtual/thermal/thermal_zone2/temp) / 1000 ))
  local nc=$(pgrep -fc "src/nano_detect_stream.py")
  echo "$1,$2,$3,$(date -Is),$(cut -d. -f1 /proc/uptime),$(uptime -s | tr ' ' 'T'),$ma,$sw,$ct,$gt,$nc,$(cat /sys/fs/ext4/mmcblk1p1/errors_count)" >> "$TN"
}

echo "### chay_dai_1470 $(date -Is) | nguon $SRC ($(ls "$SRC" | wc -l) anh) | engine $(sha256sum "$ENGINE" | cut -c1-16) | thu tu: $THU_TU | nghi $NGHI s"
echo "### zone1=$(cat /sys/devices/virtual/thermal/thermal_zone1/type) zone2=$(cat /sys/devices/virtual/thermal/thermal_zone2/type)"
i=0
for T in $THU_TU; do
  i=$((i+1)); L=$(printf "luot%02d_%s" "$i" "$T"); O="$OUT/$L"; CID="dai_${L}_$TAG"
  if [ "$i" -gt 1 ]; then echo "(nghi $NGHI s)"; sleep "$NGHI"; fi
  anh_chup "$i" "$T" truoc
  echo "=== luot $i $T $(date -Is)"
  "$PYV" src/nano_live.py --source "$SRC" --fps 25 --video-name MVI_39031 --engine "$ENGINE" --onnx-sha256 "$ONNX_SHA" \
      --tracker "$T" --out "$O" --save-cache "$CID" --warmup-frames 25 --hang-doi 4 --tegrastats-ms 500 --progress-every 250
  RC=$?
  echo "luot $i $T live exit=$RC"
  PID=$(cat "$O/detector.pid" 2>/dev/null || echo 0)
  if kill -0 "$PID" 2>/dev/null; then echo "luot $i con pid $PID: CON SONG sau khi cha thoat"; else echo "luot $i con pid $PID: da thoat"; fi
  anh_chup "$i" "$T" sau
  "$PYV" - "$O" "$SPLIT" "replay-$L" <<'PY'
import json, subprocess, sys
o, split, ten = sys.argv[1:4]
m = json.load(open(o + "/run_meta.json"))
argv = [sys.executable if x == "python" else split if x == "<split>" else ten if x == "<ten>" else x for x in m["replay_argv"]]
print("$ " + " ".join(argv), flush=True)
sys.exit(subprocess.run(argv).returncode)
PY
  RR=$?
  echo "luot $i replay exit=$RR"
  "$PYV" src/nano_parity.py --ref "data/processed/trackers/$SPLIT/replay-$L" --test "$O" --json "$OUT/parity_$L.json" | tail -3
  echo "luot $i parity exit=${PIPESTATUS[0]}"
  cp -r "data/processed/trackers/$SPLIT/replay-$L" "$OUT/${L}_replay" 2>/dev/null
  if cmp -s "$O/data/MVI_39031.txt" "$OUT/${L}_replay/data/MVI_39031.txt"; then
    echo "luot $i MOT live vs replay: BYTE GIONG HET"
  else
    echo "luot $i MOT live vs replay: KHAC BYTE"
  fi
  cp -r "data/interim/det_cache/$CID" "$OUT/${L}_cache" 2>/dev/null
done
echo "### XONG $(date -Is) | boot $(uptime -s) | ext4 errors $(cat /sys/fs/ext4/mmcblk1p1/errors_count) | con dang chay $(pgrep -fc src/nano_detect_stream.py)"
