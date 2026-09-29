#!/usr/bin/env bash
# TAO LOI TAT cho giao dien tai cho (chi trong tai khoan nguoi dung, KHONG sudo):
#   ~/.local/share/applications/nano-tracking.desktop   (menu ung dung / tim kiem "Theo vet")
#   ~/Desktop/nano-tracking.desktop                     (bieu tuong tren desktop; neu co thu muc Desktop)
# Loi tat tro toi scripts/nano/07_giao_dien.sh CUA GOI NAY (duong dan tuyet doi). Chay lai sau khi doi goi.
#
#   bash scripts/nano/08_tao_loi_tat.sh          # tao / cap nhat
#   bash scripts/nano/08_tao_loi_tat.sh --xoa    # go bo hai file .desktop do chinh script nay tao
set -u
GOC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEN=nano-tracking.desktop
MENU="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"

if [ "${1:-}" = "--xoa" ]; then
    for f in "$MENU/$TEN" "$DESK/$TEN"; do
        if [ -f "$f" ] && grep -q "07_giao_dien.sh" "$f"; then rm -f "$f" && echo "da go: $f"; fi
    done
    exit 0
fi

NOI_DUNG="[Desktop Entry]
Type=Application
Version=1.0
Name=Theo vết phương tiện (Nano)
Name[vi]=Theo vết phương tiện (Nano)
Comment=Chọn video/thư mục ảnh, chạy tracking CV/EKF bằng TensorRT, xem bbox/ID/lớp trực tiếp
Exec=/bin/bash \"$GOC/scripts/nano/07_giao_dien.sh\"
Path=$GOC
Icon=camera-video
Terminal=false
Categories=AudioVideo;Video;
StartupNotify=true
"
mkdir -p "$MENU"
printf '%s' "$NOI_DUNG" > "$MENU/$TEN"
chmod +x "$MENU/$TEN"
echo "menu:    $MENU/$TEN"
if [ -d "$DESK" ]; then
    printf '%s' "$NOI_DUNG" > "$DESK/$TEN"
    chmod +x "$DESK/$TEN"
    # GNOME/Unity 18.04: danh dau tin cay de bam dup chay ngay (khong hoi); loi -> van dung duoc qua menu
    gio set "$DESK/$TEN" metadata::trusted true 2>/dev/null || true
    # Nautilus 3.26 chi doc lai thong tin khi file thay doi -> touch de bieu tuong hien ten + icon ngay (vong 25, do tren Nano)
    touch "$DESK/$TEN"
    echo "desktop: $DESK/$TEN"
else
    echo "khong co thu muc Desktop ($DESK): chi tao loi tat trong menu"
fi
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$MENU" 2>/dev/null || true
echo "tro toi: $GOC/scripts/nano/07_giao_dien.sh"
