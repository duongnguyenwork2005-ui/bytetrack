#!/bin/bash
set -eu
cd /home/jetson/nano_run/r62_20261028/baseline/nano_transfer
for item in 02_cv_off 03_cv_off 04_cv_on 05_ekf_off 06_ekf_on 07_ekf_on 08_ekf_off; do
  sleep 20
  rest=${item#*_}; tracker=${rest%_*}; display=${rest##*_}
  /home/jetson/miniforge3/envs/nano/bin/python /home/jetson/nano_run/r62_20261028/chay_mot_luot.py --tracker "$tracker" --display "$display" --out "/home/jetson/nano_run/r62_20261028/baseline300/$item"
done
