#!/usr/bin/env bash
# Vong 25 -- kich ban LAUNCHER tren Nano, mo bang CHINH scripts/nano/07_giao_dien.sh tren man hinh :0.
# Nut bam duoc "bam" bang tham so --tu-dong-* (goi CUNG ham voi nut); thao tac TRONG cua so xem (phim q, dong cua so)
# bang xdotool -- CAN man hinh Nano MO KHOA (Unity khoa sau 5 phut ranh, xdotool khong vao duoc cua so khi khoa).
# Sau moi kich ban: trang thai tung luot (run_meta/launcher.json), tien trinh/cua so con lai.
#
#   [GOI=v13 KICH_BAN="k1 k2 k3 k4 k5"] bash 04_kich_ban_launcher.sh <thu_muc_ket_qua_MOI>
set -u
GOI=${GOI:-v13}
G=$HOME/nano_run/$GOI/nano_transfer
KQ=${1:?thu muc ket qua MOI}
KICH_BAN=${KICH_BAN:-"k1 k2 k3"}
export DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority
[ -e "$KQ" ] && { echo "[LOI] $KQ da ton tai"; exit 8; }
mkdir -p "$KQ"
SMOKE=$G/data/smoke/MVI_39031_f0001-0300
MP4=$G/data/smoke/MVI_39031_f0001-0300.mp4
B=$(tr "\0" "\n" < /proc/$(pgrep -u "$USER" -x gnome-session-b | head -1)/environ | grep DBUS_SESSION_BUS_ADDRESS)
export "$B"

khoa() { gdbus call --session --dest com.canonical.Unity --object-path /com/canonical/Unity/Session \
         --method com.canonical.Unity.Session.IsLocked 2>/dev/null | tr -d '(),' ; }
kiem_sach() {
  echo "  [$1] detector $(pgrep -fc '^/usr/bin/python3 .*nano_detect_stream') | nano_live $(pgrep -fc 'nano_live.py --source') |" \
       "launcher $(pgrep -fc 'src/nano_launcher.py') | cua so xem $(xdotool search --name 'Xem truc tiep' 2>/dev/null | wc -l) |" \
       "chan khoa $(pgrep -fc 'gnome-session-inhibit --app-id nano-tracking') | man hinh khoa $(khoa)"
}
tom_tat() {  # $1 thu muc goc cua kich ban
  for d in "$1"/*/; do
    [ -f "$d/run_meta.json" ] || continue
    python3 - "$d" <<'PY'
import json, os, sys
d = sys.argv[1]
m = json.load(open(os.path.join(d, "run_meta.json")))
l = json.load(open(os.path.join(d, "launcher.json"))) if os.path.exists(os.path.join(d, "launcher.json")) else {}
h = m.get("hien_thi") or {}
dm = h.get("dem") or {}
print("  %-45s %-9s ma %s | frame %s/%s | hien thi %s (bo %s) ly do cua so %s | fps xu ly %.2f | tong nano_live %.1f s | launcher %s %s"
      % (os.path.basename(d.rstrip("/")), m.get("trang_thai"), m.get("ma_thoat"), m["dem"]["n_frame_da_xu_ly"],
         m["dem"]["n_frames_khai_bao"], dm.get("n_hien_thi"), dm.get("n_bo_hien_thi"), h.get("ly_do_dung_tu_cua_so"),
         (m["thoi_gian"].get("fps_xu_ly") or 0), m["thoi_gian"].get("tong_s") or 0, l.get("trang_thai"),
         "%.1f s" % l["tong_s_phia_launcher"] if l.get("tong_s_phia_launcher") else ""))
PY
  done
}
cho_cua_so() {  # doi cua so xem xuat hien (toi da $1 giay)
  local W=""
  for k in $(seq 1 $(( $1 * 2 ))); do W=$(xdotool search --name "Xem truc tiep" 2>/dev/null | head -1); [ -n "$W" ] && break; sleep 0.5; done
  echo "$W"
}
chay() {  # $1 ten kich ban, $2 thao tac xdotool (none|q|dong), con lai: tham so launcher
  local ten=$1 tt=$2; shift 2
  echo "=== $ten $(date -Is) | thao tac cua so: $tt"
  kiem_sach truoc
  bash "$G/scripts/nano/07_giao_dien.sh" --ket-qua "$KQ/$ten" "$@" &
  local P=$!
  if [ "$tt" != none ]; then
    local W; W=$(cho_cua_so 180)
    echo "  cua so xem: ${W:-KHONG THAY} | man hinh khoa: $(khoa)"
    sleep 8
    if [ -n "$W" ]; then
      xdotool windowactivate --sync "$W" 2>/dev/null
      if [ "$tt" = q ]; then xdotool key q; else xdotool key alt+F4; fi
      echo "  da gui '$tt' luc $(date +%T)"
    fi
  fi
  local t0=$(date +%s)
  while kill -0 $P 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 900 ]; do sleep 1; done
  if kill -0 $P 2>/dev/null; then echo "  [LOI] launcher chua thoat sau 900 s"; fi
  wait $P 2>/dev/null; echo "  launcher (07_giao_dien.sh) exit=$?"
  sleep 2
  kiem_sach sau
  tom_tat "$KQ/$ten"
}

echo "### 04_kich_ban_launcher $(date -Is) | goi $GOI ($(python3 -c 'import json;print(json.load(open("'"$G"'/PACKAGE_INFO.json"))["nguon"]["HEAD_short"])')) | kich ban: $KICH_BAN | man hinh khoa: $(khoa)"
for K in $KICH_BAN; do
  case $K in
    k1) chay k1_cv_anh_2_luot none --nguon "$SMOKE" --tracker cv --fps 25 --luu-cache --tu-dong-bat-dau --tu-dong-so-luot 2 \
            --tu-dong-dong-khi-xong ;;
    k2) chay k2_ekf_video_dung_roi_luot_moi none --nguon "$MP4" --tracker ekf --tu-dong-bat-dau --tu-dong-dung-sau 12 \
            --tu-dong-so-luot 2 --tu-dong-dong-khi-xong ;;
    k3) chay k3_dong_launcher_khi_dang_chay none --nguon "$SMOKE" --tracker cv --fps 25 --tu-dong-bat-dau --tu-dong-dong-sau 12 ;;
    k4) chay k4_phim_q_cua_so_xem q --nguon "$SMOKE" --tracker ekf --fps 25 --tu-dong-bat-dau --tu-dong-dong-khi-xong ;;
    k5) chay k5_dong_cua_so_xem dong --nguon "$SMOKE" --tracker cv --fps 25 --tu-dong-bat-dau --tu-dong-dong-khi-xong ;;
  esac
  sleep 5
done
echo "### XONG $(date -Is)"
kiem_sach cuoi
