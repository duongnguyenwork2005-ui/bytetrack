#!/usr/bin/env bash
# Vong 25 -- chay TensorRT THAT qua luong truc tiep (nano_live) VOI / KHONG hien thi tren HDMI, cung dieu kien:
#   doc anh -> TensorRT FP16 -> tracker CV/EKF, c25, K = 4, --save-cache; hien thi = cua so OpenCV that tren man hinh :0.
# Moi luot: live -> (neu hien thi) chup man hinh -> kiem tien trinh/cua so sau khi thoat -> replay bang CHINH replay_argv
# -> nano_parity -> so byte MOT live vs replay -> (neu luu mau) so anh xem truoc voi ve_frame(MOT da ghi). Giu MOI luot.
# Script CHI goi entry point da nghiem thu (nano_live.py, nano_replay qua replay_argv, nano_parity.py, render_mot_video).
#
#   cd ~/nano_run/v12/nano_transfer && [SRC=... LUOT="cv:bat cv:tat ..." MAU=50 CHUP=20] bash 02_hien_thi_trt.sh <OUT_MOI>
set -u
PYV=${PYV:-$HOME/miniforge3/envs/nano/bin/python}
ENGINE=${ENGINE:-$HOME/nano_run/engine/yolov8n_fp16.engine}
SRC=${SRC:-data/smoke/MVI_39031_f0001-0300}
OUT=${1:?thu muc ket qua MOI}
NGHI=${NGHI:-10}
LUOT=${LUOT:-"cv:bat cv:tat cv:tat cv:bat ekf:bat ekf:tat ekf:tat ekf:bat"}
MAU=${MAU:-0}                  # luu anh xem truoc moi MAU frame (0 = khong)
CHUP=${CHUP:-15}               # chup man hinh sau CHUP giay (chi luot bat hien thi; 0 = khong)
export BYTETRACK_BACKEND=np DISPLAY=${DISPLAY:-:0} XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority}
[ -e "$OUT" ] && { echo "[LOI] $OUT da ton tai -> dung, khong ghi de"; exit 8; }
mkdir -p "$OUT"
SRC=$(cd "$SRC" && pwd)
ONNX_SHA=$(sha256sum models/export/yolov8n_640_op12.onnx | cut -d' ' -f1)
TAG=$(date +%Y%m%d%H%M%S)
SPLIT=NANO-ht-$TAG
TN="$OUT/tai_nguyen.csv"
echo "luot,cau_hinh,hien_thi,moc,thoi_diem,uptime_s,boot,mem_available_kB,swap_used_MB,cpu_therm_C,gpu_therm_C,con_detector_dang_chay,cua_so_xem,ext4_errors" > "$TN"

anh_chup() {  # $1 luot $2 tracker $3 hien_thi $4 moc
  local ma=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  local sw=$(free -m | awk '/Swap/ {print $3}')
  local ct=$(( $(cat /sys/devices/virtual/thermal/thermal_zone1/temp) / 1000 ))
  local gt=$(( $(cat /sys/devices/virtual/thermal/thermal_zone2/temp) / 1000 ))
  local nc=$(pgrep -fc "^/usr/bin/python3 .*nano_detect_stream")
  local cs=$(xdotool search --name "Xem truc tiep" 2>/dev/null | wc -l)
  echo "$1,$2,$3,$4,$(date -Is),$(cut -d. -f1 /proc/uptime),$(uptime -s | tr ' ' 'T'),$ma,$sw,$ct,$gt,$nc,$cs,$(cat /sys/fs/ext4/mmcblk1p1/errors_count)" >> "$TN"
}

echo "### 02_hien_thi_trt $(date -Is) | nguon $SRC ($(ls "$SRC" | wc -l) anh) | engine $(sha256sum "$ENGINE" | cut -c1-16) | luot: $LUOT | nghi $NGHI s | mau $MAU | chup $CHUP s"
echo "### goi $(python3 -c 'import json;print(json.load(open("PACKAGE_INFO.json"))["nguon"]["HEAD_short"])') | DISPLAY=$DISPLAY | $(xrandr 2>/dev/null | grep ' connected' | head -1)"
i=0
for X in $LUOT; do
  T=${X%%:*}; H=${X##*:}
  i=$((i+1)); L=$(printf "luot%02d_%s_%s" "$i" "$T" "$H"); O="$OUT/$L"; CID="ht_${L}_$TAG"
  if [ "$i" -gt 1 ]; then echo "(nghi $NGHI s)"; sleep "$NGHI"; fi
  anh_chup "$i" "$T" "$H" truoc
  EXTRA=()
  if [ "$H" = bat ]; then
    EXTRA+=(--hien-thi --hien-thi-kich-thuoc 960 540)
    [ "$MAU" -gt 0 ] && EXTRA+=(--hien-thi-luu-mau "$O/mau" --hien-thi-moi "$MAU")
  fi
  echo "=== luot $i $T hien_thi=$H $(date -Is)"
  if [ "$H" = bat ] && [ "$CHUP" -gt 0 ]; then
    ( sleep "$CHUP"; scrot "$OUT/${L}_man_hinh.png" && echo "luot $i: da chup man hinh sau ${CHUP}s" ) &
  fi
  "$PYV" src/nano_live.py --source "$SRC" --fps 25 --video-name MVI_39031 --engine "$ENGINE" --onnx-sha256 "$ONNX_SHA" \
      --tracker "$T" --out "$O" --save-cache "$CID" --cache-root "$OUT/cache" --warmup-frames 25 --hang-doi 4 \
      --tegrastats-ms 500 --progress-every 100 "${EXTRA[@]}"
  RC=$?
  wait
  echo "luot $i $T hien_thi=$H live exit=$RC"
  PID=$(cat "$O/detector.pid" 2>/dev/null || echo 0)
  if kill -0 "$PID" 2>/dev/null; then echo "luot $i con pid $PID: CON SONG sau khi cha thoat"; else echo "luot $i con pid $PID: da thoat"; fi
  echo "luot $i cua so 'Xem truc tiep' con mo: $(xdotool search --name 'Xem truc tiep' 2>/dev/null | wc -l)"
  anh_chup "$i" "$T" "$H" sau
  "$PYV" - "$O" "$SPLIT" "replay-$L" <<'PY'
import json, subprocess, sys
o, split, ten = sys.argv[1:4]
m = json.load(open(o + "/run_meta.json"))
argv = [sys.executable if x == "python" else split if x == "<split>" else ten if x == "<ten>" else x for x in m["replay_argv"]]
print("$ " + " ".join(argv), flush=True)
sys.exit(subprocess.run(argv).returncode)
PY
  echo "luot $i replay exit=$?"
  "$PYV" src/nano_parity.py --ref "data/processed/trackers/$SPLIT/replay-$L" --test "$O" --json "$OUT/parity_$L.json" | tail -2
  echo "luot $i parity exit=${PIPESTATUS[0]}"
  cp -r "data/processed/trackers/$SPLIT/replay-$L" "$OUT/${L}_replay" 2>/dev/null
  if cmp -s "$O/data/MVI_39031.txt" "$OUT/${L}_replay/data/MVI_39031.txt" && cmp -s "$O/data_cls/MVI_39031.csv" "$OUT/${L}_replay/data_cls/MVI_39031.csv"; then
    echo "luot $i MOT + lop live vs replay: BYTE GIONG HET"
  else
    echo "luot $i MOT + lop live vs replay: KHAC BYTE"
  fi
  if [ "$H" = bat ] && [ "$MAU" -gt 0 ]; then
    /usr/bin/python3 "$(dirname "$0")/so_mau_hien_thi.py" "$O/mau" "$SRC" "$O/data/MVI_39031.txt" "$O/data_cls/MVI_39031.csv" \
        "$PWD/src" "$OUT/${L}_mau.json" | cut -c1-200
    echo "luot $i anh xem truoc vs ve_frame(MOT da ghi) exit=${PIPESTATUS[0]}"
  fi
done
echo "### so sanh giua cac luot (cung detection -> bat/tat hien thi khong doi tracking)"
"$PYV" - "$OUT" <<'PY'
import glob, hashlib, json, os, sys
import numpy as np
out = sys.argv[1]
luot = sorted(d for d in os.listdir(out) if d.startswith("luot") and os.path.isdir(os.path.join(out, d)) and "_replay" not in d)
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
bang = {}
for L in luot:
    o = os.path.join(out, L)
    npz = glob.glob(os.path.join(out, "cache", "ht_%s_*" % L, "MVI_39031.npz"))
    z = np.load(npz[0]) if npz else None
    det = hashlib.sha256(b"".join(np.ascontiguousarray(z[k]).tobytes() for k in ("frame_idx", "n_per_frame", "data"))).hexdigest()[:16] if z is not None else None
    bang[L] = (h(os.path.join(o, "data", "MVI_39031.txt")), h(os.path.join(o, "data_cls", "MVI_39031.csv")), det)
    print("%-22s MOT %s  lop %s  detection %s" % (L, *bang[L]))
for tr in ("cv", "ekf"):
    ls = [L for L in luot if "_%s_" % tr in L]
    if ls:
        print("%s: %d luot | MOT giong nhau: %s | lop giong nhau: %s | detection giong nhau: %s" % (
            tr, len(ls), len({bang[L][0] for L in ls}) == 1, len({bang[L][1] for L in ls}) == 1, len({bang[L][2] for L in ls}) == 1))
PY
echo "### XONG $(date -Is) | boot $(uptime -s) | ext4 errors $(cat /sys/fs/ext4/mmcblk1p1/errors_count) | con dang chay $(pgrep -fc "^/usr/bin/python3 .*nano_detect_stream") | cua so xem $(xdotool search --name 'Xem truc tiep' 2>/dev/null | wc -l)"
