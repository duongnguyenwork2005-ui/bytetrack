#!/usr/bin/env bash
# R63 - do engine TensorRT co san bang trtexec (TRT 8.2), batch 1, input shape cua engine (engine tinh,
# explicit batch: KHONG dung --batch). Muc dich: tach thoi gian GPU compute that su khoi chi phi H2D/D2H,
# enqueue va dong bo de doi chieu voi exec_ms ~69 ms do bang host wall trong pipeline.
# Moi luot chay qua chay_co_giam_sat.sh (timeout, nhiet, swap, kernel, boot_id); luot loi hoac vi pham cong ->
# DUNG chuoi, giu bang chung, khong chay lai tu dong. Khong sudo, khong doi clocks/nvpmodel. Cach dung:
#   bash do_trtexec.sh <engine> <thu_muc_ra_moi> [duration_s=30] [nghi_s=20]
set -u
ENGINE=${1:?can engine}; OUT=${2:?can thu_muc_ra}; DUR=${3:-30}; NGHI=${4:-20}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
DIR=$(cd "$(dirname "$0")" && pwd)
[ -x "$TRTEXEC" ] || { echo "khong thay $TRTEXEC"; exit 2; }
[ -f "$ENGINE" ] || { echo "khong thay engine $ENGINE"; exit 2; }
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$OUT khong rong"; exit 2; fi
mkdir -p "$OUT"
{
  echo "date=$(date -Is)"; echo "host=$(hostname)"; echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "engine=$(readlink -f "$ENGINE")"; echo "engine_sha256=$(sha256sum "$ENGINE" | cut -d' ' -f1)"
  echo "trtexec=$TRTEXEC"; echo "nvpmodel=$(nvpmodel -q 2>/dev/null | tr '\n' ' ')"
} > "$OUT/dieu_kien.txt"

CHUNG="--loadEngine=$ENGINE --warmUp=3000 --duration=$DUR --iterations=300 --avgRuns=50 --percentile=90"
# Thu tu co dinh; luot A gan pipeline nhat (execute dong bo, co H2D/D2H, khong CUDA graph).
LUOT=(
  "A_mac_dinh|"
  "B_khong_truyen|--noDataTransfers"
  "C_spin_wait|--useSpinWait"
  "D_cuda_graph_khong_truyen|--useCudaGraph --noDataTransfers"
  "E_profile_lop|--noDataTransfers --dumpProfile --separateProfileRun"
)
LOGS=()
for l in "${LUOT[@]}"; do
  ten=${l%%|*}; co=${l#*|}
  # shellcheck disable=SC2086
  bash "$DIR/chay_co_giam_sat.sh" "$OUT/$ten" $((DUR * 3 + 120)) -- \
    "$TRTEXEC" $CHUNG $co --exportTimes="$OUT/$ten/trtexec.times.json"
  rc=$?
  # log phan tich: dong dau la lenh, cuoi la exit (giu dinh dang phan_tich_trtexec doc)
  { echo "== $ten: $TRTEXEC $CHUNG $co"; cat "$OUT/$ten/stdout.log" "$OUT/$ten/stderr.log"; echo "[exit=$rc]"; } > "$OUT/$ten/trtexec.log"
  LOGS+=("$OUT/$ten/trtexec.log")
  if [ $rc -ne 0 ]; then
    echo "[DUNG] luot $ten exit=$rc (10 = vi pham cong) - xem $OUT/$ten/giam_sat.json; khong chay cac luot sau"
    python3 "$DIR/phan_tich_trtexec.py" "${LOGS[@]}" --json "$OUT/tom_tat_trtexec.json"
    exit $rc
  fi
  sleep "$NGHI"
done
python3 "$DIR/phan_tich_trtexec.py" "${LOGS[@]}" --json "$OUT/tom_tat_trtexec.json"
exit $?
