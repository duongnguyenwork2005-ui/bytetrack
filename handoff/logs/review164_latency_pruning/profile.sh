#!/usr/bin/env bash
set -euo pipefail
root=/home/jetson/nano_run/r164_latency_pruning_20261121
engine=/home/jetson/nano_run/r102_interp50_20261111/yolo11n_interp50_fp16.engine
test ! -e "$root/profile.json"
test "$(sha256sum "$engine" | cut -d ' ' -f1)" = 13f8cd0f9777840052fc671a621dc0473a103eeecce71ce655b54c74a4ba9823
test "$(cat /sys/fs/ext4/mmcblk1p1/errors_count)" = 0
cat /proc/sys/kernel/random/boot_id > "$root/profile_boot.txt"
cat /proc/meminfo > "$root/profile_mem_before.txt"
/usr/src/tensorrt/bin/trtexec --loadEngine="$engine" --warmUp=500 --duration=6 --iterations=100 --separateProfileRun --dumpProfile --exportProfile="$root/profile.json" > "$root/profile.log" 2>&1
cat /proc/meminfo > "$root/profile_mem_after.txt"
test "$(cat /proc/sys/kernel/random/boot_id)" = "$(cat "$root/profile_boot.txt")"
test "$(cat /sys/fs/ext4/mmcblk1p1/errors_count)" = 0
printf 'R164_BASELINE_PROFILE_COMPLETE\n'
