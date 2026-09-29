#!/usr/bin/env bash
# MO GIAO DIEN TAI CHO tren Jetson Nano (man hinh HDMI + chuot + ban phim): chon video / thu muc anh, chon CV / EKF,
# chon thu muc luu, Bat dau / Dung, xem bbox/ID/lop trong luc tracking (src/nano_launcher.py).
#
#   Tu desktop:  bam dup bieu tuong "Theo vet phuong tien" (tao bang scripts/nano/08_tao_loi_tat.sh) hoac tim trong menu.
#   Tu terminal CUA DESKTOP:  bash scripts/nano/07_giao_dien.sh
#   Chi kiem cau hinh:        bash scripts/nano/07_giao_dien.sh --kiem-tra
#
# Chuong trinh chay doc lap trong phien desktop cua Nano, khong can SSH hay PC. Nhat ky: ~/.cache/nano_tracking/.
# Bien moi truong (tuy chon): PYV = Python cua env tracker (mac dinh ~/miniforge3/envs/nano/bin/python, goi TRUC TIEP).
set -u
GOC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYV=${PYV:-$HOME/miniforge3/envs/nano/bin/python}
NK="${XDG_CACHE_HOME:-$HOME/.cache}/nano_tracking"
mkdir -p "$NK"
LOG="$NK/giao_dien_$(date +%Y%m%d-%H%M%S).log"

bao_loi() {
    echo "[LOI] $1" | tee -a "$LOG" >&2
    if [ -n "${DISPLAY:-}" ] && command -v zenity >/dev/null 2>&1; then
        zenity --error --no-wrap --title "Theo vết phương tiện" --text "$1" >/dev/null 2>&1
    fi
    exit 2
}

[ -x "$PYV" ] || bao_loi "Không thấy Python của môi trường tracker: $PYV
Cách khắc phục: kiểm tra env 'nano' (~/miniforge3/envs/nano) hoặc đặt biến PYV trong lệnh mở."
[ -f "$GOC/src/nano_launcher.py" ] || bao_loi "Gói không đầy đủ: thiếu $GOC/src/nano_launcher.py. Giải nén lại gói."
if [ "${1:-}" != "--kiem-tra" ] && [ -z "${DISPLAY:-}" ]; then
    bao_loi "Không có màn hình (DISPLAY trống). Hãy mở từ desktop của Nano (biểu tượng hoặc terminal trên desktop)."
fi

export BYTETRACK_BACKEND=np
cd "$GOC" || bao_loi "Không vào được thư mục gói $GOC"
{
    echo "# mo giao dien $(date -Iseconds) | goc $GOC | PYV $PYV | DISPLAY=${DISPLAY:-} | tham so: $*"
} >> "$LOG"
if [ "${1:-}" = "--kiem-tra" ]; then
    "$PYV" src/nano_launcher.py "$@" 2>&1 | tee -a "$LOG"
    exit "${PIPESTATUS[0]}"
fi
exec "$PYV" src/nano_launcher.py "$@" >> "$LOG" 2>&1
