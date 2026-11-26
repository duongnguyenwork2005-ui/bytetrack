#!/usr/bin/env bash
# Run interactively with sudo; no credentials are recorded.
set -euo pipefail
test "$(id -u)" -eq 0
root=/home/jetson/nano_run/r155_runtime_accuracy_20261120
exec > >(tee -a "$root/timing_prepare.log") 2>&1
printf '\nPREPARE_TIMING %s\n' "$(date --iso-8601=seconds)"
# Do not interrupt the experimental builds or accuracy captures.
while pgrep -x trtexec >/dev/null || \
      pgrep -f '^python3 ([^ ]*/)?capture.py ' >/dev/null || \
      pgrep -f '^bash ([^ ]*/)?(build_pruned.sh|run_capture.sh|capture_pruned.sh|build_capture.sh)' >/dev/null || \
      pgrep -f '^bash /home/jetson/nano_run/r160_height352_20261120/queued.sh' >/dev/null; do
    sleep 5
done
available=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
swapped=$(awk 'NR>1{s+=$4}END{print s+0}' /proc/swaps)
test "$available" -gt "$((2*swapped+262144))"
cat /proc/swaps
restoring=''
restore_priority=''
trap 'if test -n "$restoring"; then swapon --priority "$restore_priority" "$restoring"; fi' EXIT
# Keep every active device and its original priority. Only used swap needs
# cycling; no formatting, file removal, service changes, or reboot.
while read -r device priority; do
    restoring=$device;restore_priority=$priority
    swapoff "$device"
    swapon --priority "$priority" "$device"
    restoring=''
done < <(awk 'NR>1 && $4>0{print $1,$5}' /proc/swaps)
free -m
cat /proc/swaps
used=$(awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{print t-f}' /proc/meminfo)
test "$used" -lt 102400
printf 'PREPARE_TIMING_PASS swap_kib=%s\n' "$used"
