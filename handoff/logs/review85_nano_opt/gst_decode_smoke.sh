#!/usr/bin/env bash
# R85 isolated hardware decode and BGR conversion smoke; no tracker changes.
set -eu
SRC=/home/jetson/nano_run/v16/nano_transfer/data/smoke/MVI_39031_f0001-0300.mp4
test -f "$SRC"
TIMEFORMAT='wall_seconds=%R'
time gst-launch-1.0 -q \
  filesrc location="$SRC" ! qtdemux ! mpeg4videoparse ! nvv4l2decoder ! \
  nvvidconv ! 'video/x-raw,format=BGRx' ! videoconvert ! \
  'video/x-raw,format=BGR' ! fakesink sync=false
