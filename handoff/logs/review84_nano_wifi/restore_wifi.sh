#!/usr/bin/env bash
# Run in an interactive Jetson terminal. The Wi-Fi key is never written to this script or shell history.
set -euo pipefail
set +x
UUID=f6ce5c29-9a28-401d-bf3e-6dbaf549ef09
if ! ip -4 addr show dev eth0 | grep -q 'inet '; then
    echo 'LAN eth0 is not connected; stop to preserve remote access.' >&2
    exit 2
fi
sudo -v
printf 'VIETTEL_5G password (hidden): ' >&2
IFS= read -r -s WIFI_PSK
printf '\n' >&2
if [ ${#WIFI_PSK} -lt 8 ] || [ ${#WIFI_PSK} -gt 64 ]; then
    unset WIFI_PSK
    echo 'Password length is outside WPA2 limits.' >&2
    exit 2
fi
sudo nmcli connection modify uuid "$UUID" wifi-sec.psk "$WIFI_PSK" wifi-sec.psk-flags 0
unset WIFI_PSK
sudo nmcli --wait 20 connection down uuid "$UUID"
sudo nmcli --wait 30 connection up uuid "$UUID" ifname wlan0
echo 'Saved secret length (including newline):'
nmcli --show-secrets -g 802-11-wireless-security.psk connection show "$UUID" | wc -c
nmcli -t -f DEVICE,STATE,CONNECTION device status
ip -4 -br addr show wlan0
ping -I wlan0 -c 3 -W 2 192.168.1.1
