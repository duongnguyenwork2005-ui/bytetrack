#!/usr/bin/env bash
# Vong 25 -- CHO nguoi dung mo khoa man hinh Nano (toi da 6 gio, kiem moi 10 s), roi chay 05_sau_mo_khoa.sh duoi
# gnome-session-inhibit --inhibit idle (khong khoa lai trong luc kiem). Chay tach khoi SSH:
#   setsid nohup bash 05_cho_mo_khoa.sh <thu_muc_ket_qua_MOI> > 05_cho.out 2>&1 < /dev/null &
set -u
V=$HOME/nano_run/v13
KQ=${1:?thu muc ket qua MOI}
export DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority
B=$(tr "\0" "\n" < /proc/$(pgrep -u "$USER" -x gnome-session-b | head -1)/environ | grep DBUS_SESSION_BUS_ADDRESS)
export "$B"
echo "# bat dau cho mo khoa $(date -Is)"
for k in $(seq 1 2160); do
  L=$(gdbus call --session --dest com.canonical.Unity --object-path /com/canonical/Unity/Session \
      --method com.canonical.Unity.Session.IsLocked 2>/dev/null | tr -d '(),')
  if [ "$L" = "false" ]; then
    echo "# man hinh MO KHOA luc $(date -Is) -> doi 20 s cho nguoi dung roi tay, bat dau pha kiem"
    sleep 20
    gnome-session-inhibit --app-id nano-tracking-kiem --reason "Kiem hien thi vong 25" --inhibit idle \
        bash "$V/05_sau_mo_khoa.sh" "$KQ" > "$V/05_sau_mo_khoa.out" 2>&1 < /dev/null
    echo "# pha kiem xong $(date -Is) exit=$?"
    exit 0
  fi
  sleep 10
done
echo "# het 6 gio van khoa $(date -Is)"
