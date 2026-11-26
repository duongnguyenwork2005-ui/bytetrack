#!/usr/bin/env bash
# Run only after actual capture/scoring, if the locked timing guard is waiting.
set -euo pipefail
task_root=/home/jetson/nano_run/r172_common7_pruning_20261121
if [[ $EUID -ne 0 ]]; then
    echo 'Run this script with sudo in the Nano terminal; never send passwords.' >&2
    exit 2
fi
exec 9>/run/r172_swap_maintenance.lock
flock -n 9 || { echo 'Another swap maintenance is active.' >&2; exit 2; }
python3 - "$task_root/recovered/complete.json" <<'PY'
import json,sys
with open(sys.argv[1]) as stream:
    value=json.load(stream)
assert value['status']=='PASS' and set(value['built'])=={'control','joint50'}
PY
if pgrep -f '[t]rtexec|[b]uild_recovered.py|[c]apture.py|[r]un_models_v[45].py|[r]un_pipeline.py' >/dev/null; then
    echo 'Experiment work is active. Leave swap unchanged and run after it ends.' >&2
    exit 2
fi
task_used=$(awk '/^SwapTotal:/{total=$2}/^SwapFree:/{free=$2}END{print total-free}' /proc/meminfo)
task_available=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
if (( task_used < 102400 )); then
    echo 'Timing swap guard already passes; no changes made.'
    exit 0
fi
if (( task_available <= task_used + 1048576 )); then
    echo 'Insufficient RAM headroom to drain swap safely; no changes made.' >&2
    exit 2
fi
mapfile -t task_rows < <(awk 'NR>1{print $1, $5}' /proc/swaps)
(( ${#task_rows[@]} > 0 ))
for task_row in "${task_rows[@]}"; do
    read -r task_device task_priority <<< "$task_row"
    case "$task_device" in
        /swapfile|/dev/zram0|/dev/zram1|/dev/zram2|/dev/zram3) ;;
        *) echo "Unexpected swap device: $task_device; no changes made." >&2; exit 2 ;;
    esac
done
task_log="$task_root/swap_maintenance_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee "$task_log") 2>&1
echo 'Before: clock, network, swap configuration files and services will not be changed.'
cat /proc/swaps
task_restore() {
    local task_row task_device task_priority task_failed=0
    for task_row in "${task_rows[@]}"; do
        read -r task_device task_priority <<< "$task_row"
        if ! awk -v device="$task_device" 'NR>1&&$1==device{found=1}END{exit !found}' /proc/swaps; then
            if (( task_priority >= 0 )); then
                if ! swapon --priority "$task_priority" -- "$task_device"; then
                    echo "Restore failed: $task_device; attempting remaining devices." >&2
                    task_failed=1
                fi
            else
                if ! swapon -- "$task_device"; then
                    echo "Restore failed: $task_device; attempting remaining devices." >&2
                    task_failed=1
                fi
            fi
        fi
    done
    return "$task_failed"
}
trap task_restore EXIT
swapoff -a
task_restore
trap - EXIT
echo 'After:'
cat /proc/swaps
awk '/^MemAvailable:|^SwapTotal:|^SwapFree:/{print}' /proc/meminfo
task_after=$(awk '/^SwapTotal:/{total=$2}/^SwapFree:/{free=$2}END{print total-free}' /proc/meminfo)
chown jetson:jetson "$task_log"
if (( task_after >= 102400 )); then
    echo 'Swap devices restored, but the locked timing guard is still not satisfied.' >&2
    exit 3
fi
echo 'Original swap devices restored; timing swap guard now passes.'
