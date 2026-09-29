#!/usr/bin/env bash
# Vong 26 -- chay LAI test_nano_hien_thi (ban cuoi: 12e kiem winfo_exists, moi ca 12a-e bat ngoai le) tren Nano:
#   B2. ma DA SUA: ban sao goi v16 (ma chay == v16) + test moi
#   C2. ma CU: ban sao v14 (nano_hien_thi/nano_launcher == 20e66e9) + CUNG test moi
# Goi v16 giu nguyen (khong ghi de file trong goi). Kiem tien trinh bang tien_trinh.py (/proc exe + argv).
#   bash 07_test_lai_v26.sh <thu_muc_ket_qua_MOI> <duong_dan_test_moi.py>
set -u
KQ=${1:?thu muc ket qua MOI}; TEST=${2:?test moi}
PYV=$HOME/miniforge3/envs/nano/bin/python
export DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority BYTETRACK_BACKEND=np
[ -e "$KQ" ] && { echo "[LOI] $KQ da ton tai"; exit 8; }
mkdir -p "$KQ"
TT() { /usr/bin/python3 "$HOME/nano_run/v16/tien_trinh.py" "$@"; }
dem() { echo "  [$1] detector $(TT python nano_detect_stream.py | cut -d' ' -f1) | nano_live $(TT python nano_live.py | cut -d' ' -f1) |" \
             "launcher $(TT python nano_launcher.py | cut -d' ' -f1) | zenity gia $(TT python zenity_gia picker.py | cut -d' ' -f1) |" \
             "zenity $(TT zenity | cut -d' ' -f1)"; }
MOI=$HOME/nano_run/v26_test_moi
CU=$HOME/nano_run/v26_ma_cu
[ -e "$MOI" ] || { mkdir -p "$MOI" && (cd "$HOME/nano_run/v16/nano_transfer" && tar cf - --exclude=./results_nano .) | (cd "$MOI" && tar xf -); }
cp "$TEST" "$MOI/src/test_nano_hien_thi.py"; cp "$TEST" "$CU/src/test_nano_hien_thi.py"
echo "### 07_test_lai_v26 $(date -Is) | test moi $(sha256sum < "$TEST" | cut -c1-16)"
echo "  ma DA SUA (ban sao v16): nano_hien_thi $(sha256sum < "$MOI/src/nano_hien_thi.py" | cut -c1-16) nano_launcher $(sha256sum < "$MOI/src/nano_launcher.py" | cut -c1-16)"
echo "  ma CU (ban sao v14):     nano_hien_thi $(sha256sum < "$CU/src/nano_hien_thi.py" | cut -c1-16) nano_launcher $(sha256sum < "$CU/src/nano_launcher.py" | cut -c1-16)"
dem dau
for X in "B2_ma_da_sua:$MOI" "C2_ma_cu:$CU"; do
  TEN=${X%%:*}; D=${X#*:}
  (cd "$D" && "$PYV" src/test_nano_hien_thi.py --python-detector /usr/bin/python3) > "$KQ/$TEN.log" 2>&1
  echo "$TEN exit=$? | $(grep 'KET QUA' "$KQ/$TEN.log" | tail -1)"
  grep -E "\[(FAIL|SKIP)\]" "$KQ/$TEN.log" | cut -c1-160
  dem "sau $TEN"
done
echo "### XONG $(date -Is)"
