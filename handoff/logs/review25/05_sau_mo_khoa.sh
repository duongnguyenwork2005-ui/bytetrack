#!/usr/bin/env bash
# Vong 25 -- PHA KIEM TREN MAN HINH HDMI DA MO KHOA (tu chay khi nguoi dung mo khoa; xem 05_cho_mo_khoa.sh).
# Chay duoi gnome-session-inhibit --inhibit idle (khong khoa lai trong luc kiem). Thu tu:
#   A. tao loi tat (08_tao_loi_tat.sh) + chup desktop
#   B. mo launcher BANG LOI TAT DESKTOP (gtk-launch nano-tracking) -- tien trinh thuoc phien desktop, KHONG thuoc SSH;
#      tu dong (NANO_UI_TU_DONG) chay CV 300 anh co hien thi; chup man hinh trong luc chay va khi xong
#   C. kich ban cua so xem that: K4 phim q, K5 dong cua so (xdotool) -- 04_kich_ban_launcher.sh
#   D. do anh huong hien thi: 300 anh, CV/EKF, ABBA bat/tat (02_hien_thi_trt.sh), chup man hinh cac luot bat
#   E. day du 1470 anh CV + EKF co hien thi, luu mau anh xem truoc moi 100 frame (02_hien_thi_trt.sh)
#   bash 05_sau_mo_khoa.sh <thu_muc_ket_qua_MOI>     (tu thu muc ~/nano_run/v13)
set -u
V=$HOME/nano_run/v13
G=$V/nano_transfer
KQ=${1:?thu muc ket qua MOI}
export DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority
B=$(tr "\0" "\n" < /proc/$(pgrep -u "$USER" -x gnome-session-b | head -1)/environ | grep DBUS_SESSION_BUS_ADDRESS)
export "$B"
mkdir -p "$KQ/anh"
khoa() { gdbus call --session --dest com.canonical.Unity --object-path /com/canonical/Unity/Session \
         --method com.canonical.Unity.Session.IsLocked 2>/dev/null | tr -d '(),' ; }
chup() { import -window root "$KQ/anh/$1.png" 2>/dev/null && echo "  chup: anh/$1.png (khoa=$(khoa))"; }
phien() { echo "  phien dang nhap: $(who | tr -s ' ' | tr '\n' ';')"; }

echo "### 05_sau_mo_khoa $(date -Is) | khoa=$(khoa) | $(xrandr | grep ' connected' | head -1)"
phien

echo "=== A. loi tat desktop"
bash "$G/scripts/nano/08_tao_loi_tat.sh"
ls -la "$HOME/Desktop/nano-tracking.desktop" "$HOME/.local/share/applications/nano-tracking.desktop" | cut -c1-120
sleep 3; chup A_desktop_loi_tat

echo "=== B. mo launcher bang loi tat desktop (gtk-launch), tu dong chay CV 300 anh co hien thi"
KB=$KQ/B_loi_tat_desktop
export NANO_UI_TU_DONG="{\"nguon\": \"$G/data/smoke/MVI_39031_f0001-0300\", \"tracker\": \"cv\", \"fps\": \"25\", \"ket_qua\": \"$KB\", \"bat_dau\": true, \"dong_khi_xong\": true}"
setsid gtk-launch nano-tracking >/dev/null 2>&1 < /dev/null &
unset NANO_UI_TU_DONG
sleep 4
LP=$(pgrep -f "src/nano_launcher.py" | head -1)
echo "  launcher pid ${LP:-KHONG THAY} | ppid $(ps -o ppid= -p "${LP:-1}" | tr -d ' ') | sid $(ps -o sid= -p "${LP:-1}" | tr -d ' ') | cmd: $(ps -o args= -p "${LP:-1}" | cut -c1-100)"
echo "  chuoi tien trinh cha: $(p=${LP:-1}; for i in 1 2 3 4 5; do p=$(ps -o ppid= -p $p | tr -d ' '); [ -z "$p" ] || [ "$p" = 0 ] && break; printf '%s(%s) ' $p "$(ps -o comm= -p $p)"; done)"
sleep 6; chup B1_launcher_mo
for k in $(seq 1 120); do xdotool search --name "Xem truc tiep" >/dev/null 2>&1 && break; sleep 1; done
sleep 12; chup B2_dang_chay_launcher_va_cua_so_xem
phien
for k in $(seq 1 180); do d=$(ls -d "$KB"/*/ 2>/dev/null | head -1); [ -n "$d" ] && grep -q '"hoan_tat"\|"da_dung"\|"loi"' "$d/progress.json" 2>/dev/null && break; sleep 1; done
sleep 1; chup B3_xong_tom_tat
for k in $(seq 1 30); do pgrep -f "src/nano_launcher.py" >/dev/null || break; sleep 1; done
echo "  launcher con chay sau khi xong: $(pgrep -fc src/nano_launcher.py)"

echo "=== C. kich ban cua so xem that (K4 phim q, K5 dong cua so)"
KICH_BAN="k4 k5" GOI=v13 bash "$V/04_kich_ban_launcher.sh" "$KQ/C_cua_so_xem"

echo "=== D. anh huong hien thi: 300 anh, ABBA, chup man hinh luot bat"
( cd "$G" && LUOT="cv:bat cv:tat cv:tat cv:bat ekf:tat ekf:bat ekf:bat ekf:tat" MAU=0 CHUP=20 NGHI=10 bash "$V/02_hien_thi_trt.sh" "$KQ/D_anh_huong_300" )

echo "=== E. day du 1470 anh CV + EKF co hien thi"
( cd "$G" && SRC=$HOME/nano_run/data_full/MVI_39031 LUOT="cv:bat ekf:bat" MAU=100 CHUP=60 NGHI=20 bash "$V/02_hien_thi_trt.sh" "$KQ/E_day_du_1470" )

phien
echo "### XONG $(date -Is) | khoa=$(khoa) | detector $(pgrep -fc '^/usr/bin/python3 .*nano_detect_stream') | launcher $(pgrep -fc src/nano_launcher.py) | cua so xem $(xdotool search --name 'Xem truc tiep' 2>/dev/null | wc -l) | ext4 errors $(cat /sys/fs/ext4/mmcblk1p1/errors_count)"
