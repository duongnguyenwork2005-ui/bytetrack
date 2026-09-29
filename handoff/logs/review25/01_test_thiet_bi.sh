#!/bin/bash
# Vong 25: test thiet bi tren Nano (goi v12): cha = env nano (Python 3.8), con = /usr/bin/python3 (3.6, TensorRT/OpenCV).
# DISPLAY=:0 cua phien desktop -> phep 9 (Tk) mo cua so that tren HDMI trong giay lat; phep 7 tu bo DISPLAY cho tien trinh cua no.
#   setsid nohup bash 01_test_thiet_bi.sh > 01.out 2>&1 < /dev/null &      (xong -> $OUT/XONG)
set -u
GOI=${GOI:-v12}
G=$HOME/nano_run/$GOI/nano_transfer
OUT=$G/results_nano/v25_test
PYV=$HOME/miniforge3/envs/nano/bin/python
mkdir -p "$OUT"
cd "$G" || exit 2
export BYTETRACK_BACKEND=np DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority
HEAD=$(python3 -c 'import json;print(json.load(open("PACKAGE_INFO.json"))["nguon"]["HEAD_short"])')
: > "$OUT/XONG.tmp"
for T in test_nano_hien_thi test_nano_stream; do
    {
        echo "# $(date -Is) | $T | goi $GOI ($HEAD) | boot $(uptime -s) | cha $($PYV --version 2>&1) | con $(/usr/bin/python3 --version 2>&1)"
        echo "\$ $PYV src/$T.py --python-detector /usr/bin/python3"
    } > "$OUT/$T.log"
    t0=$(date +%s)
    "$PYV" "src/$T.py" --python-detector /usr/bin/python3 >> "$OUT/$T.log" 2>&1
    rc=$?
    echo "# exit=$rc wall_s=$(( $(date +%s) - t0 ))" >> "$OUT/$T.log"
    echo "$T exit=$rc" >> "$OUT/XONG.tmp"
done
{ echo "\$ bash scripts/nano/07_giao_dien.sh --kiem-tra"; bash scripts/nano/07_giao_dien.sh --kiem-tra; echo "# exit=$?"; } \
    > "$OUT/kiem_tra_trien_khai.log" 2>&1
tail -1 "$OUT/kiem_tra_trien_khai.log" | sed 's/^# /kiem_tra_trien_khai /' >> "$OUT/XONG.tmp"
mv "$OUT/XONG.tmp" "$OUT/XONG"
