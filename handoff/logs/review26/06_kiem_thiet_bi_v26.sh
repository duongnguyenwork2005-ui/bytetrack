#!/usr/bin/env bash
# Vong 26 -- kiem tren Nano (goi v16) sau khi sua N25-F1/F2 (review 20e66e9). Chay duoi gnome-session-inhibit:
#   A. script Codex (KHONG sua) tren ma da sua: F1 bang /usr/bin/python3 (OpenCV 3.2), F2 bang env nano (POSIX)
#   B. test_nano_hien_thi (co 12a-e voi tien trinh chon file that, 12e cua so Tk that) + test_nano_stream
#   C. CUNG test moi chay tren MA CU (ban sao v14 == 20e66e9 cho nano_hien_thi/nano_launcher)
#   -- cho man hinh MO KHOA (D-F can thao tac cua so bang xdotool) --
#   D. TensorRT that CV 300 anh, hien thi that; dong cua so xem (Alt+F4) giua chung -> metadata + live/replay + tien to MOT
#   E. zenity THAT qua launcher: (1) mo -> huy (Esc) -> mo lai -> go duong dan + Enter; (2) mo -> dong launcher (chua chay);
#      (3) tracker dang chay + hop chon mo -> dong launcher
#   F. mo tu bieu tuong desktop (gtk-launch, v16), 2 luot lien tiep, tu dong, thoat sach
# Kiem tien trinh bang tien_trinh.py (/proc exe + argv), KHONG pgrep -f.
#   bash 06_kiem_thiet_bi_v26.sh <thu_muc_ket_qua_MOI>       (tu ~/nano_run/v16)
set -u
V=$HOME/nano_run/v16
G=$V/nano_transfer
KQ=${1:?thu muc ket qua MOI}
PYV=$HOME/miniforge3/envs/nano/bin/python
export DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority BYTETRACK_BACKEND=np
B=$(tr "\0" "\n" < /proc/$(pgrep -u "$USER" -x gnome-session-b | head -1)/environ | grep DBUS_SESSION_BUS_ADDRESS)
export "$B"
[ -e "$KQ" ] && { echo "[LOI] $KQ da ton tai"; exit 8; }
mkdir -p "$KQ/anh"
SMOKE=$G/data/smoke/MVI_39031_f0001-0300
MP4=$G/data/smoke/MVI_39031_f0001-0300.mp4
TT() { /usr/bin/python3 "$V/tien_trinh.py" "$@"; }
khoa() { gdbus call --session --dest com.canonical.Unity --object-path /com/canonical/Unity/Session \
         --method com.canonical.Unity.Session.IsLocked 2>/dev/null | tr -d '(),' ; }
chup() { import -window root "$KQ/anh/$1.png" 2>/dev/null && echo "  chup anh/$1.png"; }
dem() { echo "  [$1] detector $(TT python nano_detect_stream.py | cut -d' ' -f1) | nano_live $(TT python nano_live.py | cut -d' ' -f1) |" \
             "launcher $(TT python nano_launcher.py | cut -d' ' -f1) | zenity $(TT zenity | cut -d' ' -f1) |" \
             "cua so xem $(xdotool search --name 'Xem truc tiep' 2>/dev/null | wc -l) | man hinh khoa $(khoa)"; }
song() { [ -n "$1" ] && [ -e "/proc/$1" ] && [ "$(awk '{print $3}' /proc/$1/stat 2>/dev/null)" != Z ] && echo song || echo "da thoat"; }
cho() { local t=$1; shift; local k; for k in $(seq 1 $((t * 5))); do "$@" && return 0; sleep 0.2; done; return 1; }
co_zenity() { [ "$(TT zenity | cut -d' ' -f1)" -ge 1 ]; }
het_zenity() { [ "$(TT zenity | cut -d' ' -f1)" -eq 0 ]; }
nhat_ky_phien_moi() { ls -t "$HOME/.cache/nano_tracking"/launcher_phien_*.log 2>/dev/null | head -1; }

echo "### 06_kiem_thiet_bi_v26 $(date -Is) | goi v16 $(python3 -c 'import json;print(json.load(open("'"$G"'/PACKAGE_INFO.json"))["nguon"]["HEAD_short"])') | boot $(uptime -s) | khoa $(khoa)"
dem dau

echo "=== A. script Codex (KHONG sua) tren ma DA SUA (v16)"
cd "$V" || exit 2
echo "\$ /usr/bin/python3 chay_repro_tung_phan.py CODEX_REPRO_20e66e9_ui.py closed_preview $G $KQ/repro_sau_f1.json"
/usr/bin/python3 chay_repro_tung_phan.py CODEX_REPRO_20e66e9_ui.py closed_preview "$G" "$KQ/repro_sau_f1.json" 2>&1 | tail -12
echo "A F1 exit=${PIPESTATUS[0]}"
echo "\$ $PYV chay_repro_tung_phan.py CODEX_REPRO_20e66e9_ui.py picker_close $G $KQ/repro_sau_f2.json"
"$PYV" chay_repro_tung_phan.py CODEX_REPRO_20e66e9_ui.py picker_close "$G" "$KQ/repro_sau_f2.json" 2>&1 | tail -12
echo "A F2 exit=${PIPESTATUS[0]}"
echo "  fixture picker con lai (theo exe): $(TT python codex_n25_picker_ picker.py)"

echo "=== B. test tren ma da sua (v16)"
cd "$G" || exit 2
for T in test_nano_hien_thi test_nano_stream; do
  "$PYV" "src/$T.py" --python-detector /usr/bin/python3 > "$KQ/B_$T.log" 2>&1
  echo "B $T exit=$? | $(grep 'KET QUA' "$KQ/B_$T.log" | tail -1)"
  grep -E "\[(FAIL|SKIP)\]" "$KQ/B_$T.log" | cut -c1-150
done
dem sau_B

echo "=== C. CUNG test moi tren MA CU (ban sao v14 + test moi)"
CU=$HOME/nano_run/v26_ma_cu
if [ ! -e "$CU" ]; then mkdir -p "$CU" && (cd "$HOME/nano_run/v14/nano_transfer" && tar cf - --exclude=./results_nano .) | (cd "$CU" && tar xf -); fi
cp "$G/src/test_nano_hien_thi.py" "$CU/src/test_nano_hien_thi.py"
echo "  ma cu: nano_hien_thi $(sha256sum < "$CU/src/nano_hien_thi.py" | cut -c1-16) nano_launcher $(sha256sum < "$CU/src/nano_launcher.py" | cut -c1-16) | test moi $(sha256sum < "$CU/src/test_nano_hien_thi.py" | cut -c1-16)"
(cd "$CU" && "$PYV" src/test_nano_hien_thi.py --python-detector /usr/bin/python3 > "$KQ/C_test_moi_tren_ma_cu.log" 2>&1)
echo "C exit=$? | $(grep 'KET QUA' "$KQ/C_test_moi_tren_ma_cu.log" | tail -1)"
grep -E "\[(FAIL|SKIP)\]" "$KQ/C_test_moi_tren_ma_cu.log" | cut -c1-150
echo "  fixture zenity gia con lai (theo exe): $(TT python zenity_gia picker.py)"
dem sau_C

echo "=== cho man hinh MO KHOA cho D-F (toi da 3 gio)"
for k in $(seq 1 1080); do [ "$(khoa)" = false ] && break; sleep 10; done
echo "  khoa=$(khoa) luc $(date -Is)"
[ "$(khoa)" = false ] || { echo "[CHUA KIEM] man hinh van khoa sau 3 gio: D-F khong chay"; exit 0; }
sleep 15

echo "=== D. TensorRT that: dong cua so xem (Alt+F4) giua chung"
O=$KQ/D_dong_cua_so_trt
TAG=$(date +%Y%m%d%H%M%S)
( for k in $(seq 1 180); do W=$(xdotool search --name "Xem truc tiep" 2>/dev/null | head -1); [ -n "$W" ] && break; sleep 0.5; done
  sleep 10; chup D_truoc_khi_dong
  xdotool windowactivate --sync "$W" key alt+F4; echo "  da gui Alt+F4 toi cua so xem $W luc $(date +%T.%N)" ) &
"$PYV" src/nano_live.py --source "$SMOKE" --fps 25 --video-name MVI_39031 --engine "$HOME/nano_run/engine/yolov8n_fp16.engine" \
    --onnx-sha256 "$(sha256sum models/export/yolov8n_640_op12.onnx | cut -d' ' -f1)" --tracker cv --out "$O" \
    --save-cache "d_$TAG" --cache-root "$KQ/cache" --hien-thi --hien-thi-kich-thuoc 960 540 --tegrastats-ms 500 --progress-every 100
echo "D live exit=$?"
wait
sleep 1; chup D_sau_khi_dung
echo "  con detector (pid $(cat "$O/detector.pid")): $(song "$(cat "$O/detector.pid")")"
dem sau_D
"$PYV" - "$O" "NANO-v26-$TAG" "replay-D" <<'PY'
import json, subprocess, sys
o, split, ten = sys.argv[1:4]
m = json.load(open(o + "/run_meta.json"))
h = m["hien_thi"]
print("  run_meta: %s ma %s | %s | xu ly %s frame | hien thi %s, khong hien vi dong %s %s, bo %s | cuoi %s | fps_hien_thi %s | ly do %s" % (
    m["trang_thai"], m["ma_thoat"], m["chi_tiet"], m["dem"]["n_frame_da_xu_ly"], h["dem"]["n_hien_thi"],
    h["dem"]["n_khong_hien_thi_cua_so_dong"], h["frame_khong_hien_thi_cua_so_dong_dau"][:5], h["dem"]["n_bo_hien_thi"],
    h["frame_hien_thi_cuoi"], h["fps_hien_thi"], h["ly_do_dung_tu_cua_so"]))
argv = [sys.executable if x == "python" else split if x == "<split>" else ten if x == "<ten>" else x for x in m["replay_argv"]]
print("  $ " + " ".join(argv), flush=True)
sys.exit(subprocess.run(argv, stdout=subprocess.DEVNULL).returncode)
PY
echo "D replay exit=$?"
"$PYV" src/nano_parity.py --ref "data/processed/trackers/NANO-v26-$TAG/replay-D" --test "$O" --json "$KQ/D_parity.json" | tail -2
echo "D parity exit=${PIPESTATUS[0]}"
cmp -s "$O/data/MVI_39031.txt" "data/processed/trackers/NANO-v26-$TAG/replay-D/data/MVI_39031.txt" && \
cmp -s "$O/data_cls/MVI_39031.csv" "data/processed/trackers/NANO-v26-$TAG/replay-D/data_cls/MVI_39031.csv" && \
echo "D MOT + lop live vs replay: BYTE GIONG HET" || echo "D MOT + lop live vs replay: KHAC"
cp -r "data/processed/trackers/NANO-v26-$TAG/replay-D" "$KQ/D_replay"
REF=$HOME/nano_run/v13/nano_transfer/results_nano/v25_hdmi/D_anh_huong_300/luot02_cv_tat
"$PYV" - "$O" "$REF" <<'PY'
import json, sys
o, ref = sys.argv[1:3]
n = json.load(open(o + "/run_meta.json"))["dem"]["n_frame_da_xu_ly"]
def loc(p, hdr):
    ls = open(p, "rb").read().replace(b"\r\n", b"\n").split(b"\n")
    ls = ls[1:] if hdr else ls
    return [x for x in ls if x and int(x.split(b",")[0]) <= n]
a, b = loc(o + "/data/MVI_39031.txt", 0), loc(ref + "/data/MVI_39031.txt", 0)
c, d = loc(o + "/data_cls/MVI_39031.csv", 1), loc(ref + "/data_cls/MVI_39031.csv", 1)
print("  tien to %d frame so voi luot DU khong hien thi (v25 D luot02_cv_tat): MOT %s (%d hang), lop %s" % (n, a == b, len(a), c == d))
PY

echo "=== E1. zenity THAT: mo -> huy (Esc) -> mo lai -> go duong dan mp4 + Enter; roi dong launcher (Alt+F4)"
bash "$G/scripts/nano/07_giao_dien.sh" --ket-qua "$KQ/E1" --tu-dong-hop-chon video video &
LP=$!
cho 60 co_zenity; Z1=$(TT zenity | cut -d' ' -f2); W1=$(xdotool search --pid "$Z1" 2>/dev/null | tail -1)
sleep 2; chup E1_zenity_lan1; echo "  zenity lan 1 pid $Z1 cua so $W1"
xdotool windowactivate --sync "$W1" key Escape
cho 20 het_zenity; echo "  sau Esc: zenity pid $Z1 $(song "$Z1")"
cho 30 co_zenity; Z2=$(TT zenity | cut -d' ' -f2); W2=$(xdotool search --pid "$Z2" 2>/dev/null | tail -1)
sleep 2; echo "  zenity lan 2 pid $Z2 cua so $W2"
xdotool windowactivate --sync "$W2" key ctrl+l; sleep 0.5
xdotool type --delay 20 "$MP4"; sleep 0.5; xdotool key Return
cho 20 het_zenity; echo "  sau chon: zenity pid $Z2 $(song "$Z2")"
sleep 2; chup E1_da_chon_nguon
LW=$(xdotool search --pid "$LP" 2>/dev/null | head -1); [ -n "$LW" ] || LW=$(xdotool search --name "Theo v" 2>/dev/null | head -1)
xdotool windowactivate --sync "$LW" key alt+F4
cho 30 bash -c "[ ! -e /proc/$LP ]"; echo "  launcher pid $LP $(song "$LP")"
dem sau_E1
L1=$(nhat_ky_phien_moi); echo "  nhat ky phien $L1:"; grep -E "hop chon|zenity" "$L1" | sed 's/^/    /'

echo "=== E2. mo hop chon -> dong launcher khi CHUA chay tracker"
bash "$G/scripts/nano/07_giao_dien.sh" --ket-qua "$KQ/E2" --tu-dong-hop-chon video --tu-dong-dong-khi-hop-chon-mo-sau 6 &
LP=$!
cho 60 co_zenity; Z=$(TT zenity | cut -d' ' -f2); sleep 2; chup E2_hop_chon_mo; echo "  zenity pid $Z"
cho 60 bash -c "[ ! -e /proc/$LP ]"; echo "  launcher pid $LP $(song "$LP") | zenity pid $Z $(song "$Z")"
dem sau_E2
L2=$(nhat_ky_phien_moi); echo "  nhat ky phien $L2:"; grep -E "hop chon|zenity|tu dong" "$L2" | sed 's/^/    /'

echo "=== E3. tracker DANG CHAY + hop chon mo -> dong launcher"
bash "$G/scripts/nano/07_giao_dien.sh" --nguon "$SMOKE" --tracker cv --fps 25 --ket-qua "$KQ/E3" --tu-dong-bat-dau \
     --tu-dong-hop-chon video --tu-dong-dong-khi-hop-chon-mo-sau 6 &
LP=$!
cho 120 co_zenity; Z=$(TT zenity | cut -d' ' -f2); sleep 2; chup E3_tracker_chay_va_hop_chon; echo "  zenity pid $Z"
cho 120 bash -c "[ ! -e /proc/$LP ]"; echo "  launcher pid $LP $(song "$LP") | zenity pid $Z $(song "$Z")"
R3=$(ls -d "$KQ/E3"/*/ 2>/dev/null | head -1)
[ -n "$R3" ] && echo "  luot: $(python3 -c "import json;m=json.load(open('$R3/run_meta.json'));print(m['trang_thai'],m['ma_thoat'],m['dem']['n_frame_da_xu_ly'],m['chi_tiet'])") | detector pid $(cat "$R3/detector.pid") $(song "$(cat "$R3/detector.pid")")"
dem sau_E3
L3=$(nhat_ky_phien_moi); echo "  nhat ky phien $L3:"; grep -E "hop chon|zenity|tu dong|dung" "$L3" | sed 's/^/    /' | head -12

echo "=== F. bieu tuong desktop -> v16; mo bang gtk-launch (phien desktop), 2 luot EKF, tu dong dong"
bash "$G/scripts/nano/08_tao_loi_tat.sh" | sed 's/^/  /'
export NANO_UI_TU_DONG="{\"nguon\": \"$SMOKE\", \"tracker\": \"ekf\", \"fps\": \"25\", \"ket_qua\": \"$KQ/F\", \"bat_dau\": true, \"so_luot\": 2, \"dong_khi_xong\": true}"
setsid gtk-launch nano-tracking >/dev/null 2>&1 < /dev/null &
unset NANO_UI_TU_DONG
cho 30 bash -c "[ \$(/usr/bin/python3 $V/tien_trinh.py python nano_launcher.py | cut -d' ' -f1) -ge 1 ]"
FP=$(TT python nano_launcher.py | cut -d' ' -f2)
echo "  launcher pid $FP | ppid $(awk '{print $4}' /proc/$FP/stat 2>/dev/null) ($(cat /proc/$(awk '{print $4}' /proc/$FP/stat 2>/dev/null)/comm 2>/dev/null)) | sid $(ps -o sid= -p "$FP" | tr -d ' ')"
sleep 25; chup F_dang_chay
cho 300 bash -c "[ ! -e /proc/$FP ]"; echo "  launcher pid $FP $(song "$FP")"
for d in "$KQ/F"/*/; do python3 -c "import json;m=json.load(open('$d/run_meta.json'));l=json.load(open('$d/launcher.json'));h=m['hien_thi'];print('  ',m['trang_thai'],m['ma_thoat'],m['dem']['n_frame_da_xu_ly'],'hien thi',h['dem']['n_hien_thi'],'fps ht',round(h['fps_hien_thi'] or 0,2),'| launcher',l['trang_thai'])"; done
dem cuoi
echo "### XONG $(date -Is) | khoa $(khoa) | ext4 errors $(cat /sys/fs/ext4/mmcblk1p1/errors_count)"
