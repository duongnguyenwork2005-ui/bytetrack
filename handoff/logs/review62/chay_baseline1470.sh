#!/bin/bash
set -eu
cd /home/jetson/nano_run/r62_20261028/baseline/nano_transfer
for item in 01_cv_on 02_ekf_on 03_ekf_on 04_cv_on; do
  sleep 20
  rest=${item#*_}; tracker=${rest%_*}
  /home/jetson/miniforge3/envs/nano/bin/python /home/jetson/nano_run/r62_20261028/chay_mot_luot.py --tracker "$tracker" --display on --frames 1470 --out "/home/jetson/nano_run/r62_20261028/baseline1470/$item"
done
