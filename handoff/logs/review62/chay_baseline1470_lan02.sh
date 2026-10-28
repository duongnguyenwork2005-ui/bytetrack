#!/bin/bash
set -eu
root=/home/jetson/nano_run/r62_20261028
exec 9>"$root/measurement.lock"
flock -n 9 || exit 73
test ! -e "$root/baseline1470_lan02"
mkdir "$root/baseline1470_lan02"
echo $$ > "$root/baseline1470_lan02/runner.pid"
cat /proc/sys/kernel/random/boot_id > "$root/baseline1470_lan02/boot_id"
trap 'echo $? > "$root/baseline1470_lan02/exit_code"; sync' EXIT
cd "$root/baseline/nano_transfer"
for item in 01_cv_on 02_ekf_on 03_ekf_on 04_cv_on; do
  sleep 20
  rest=${item#*_}; tracker=${rest%_*}
  echo "START $(date -Is) $item"
  /home/jetson/miniforge3/envs/nano/bin/python -u "$root/chay_mot_luot.py" --tracker "$tracker" --display on --frames 1470 --out "$root/baseline1470_lan02/$item"
  sync
done
