#!/usr/bin/env bash
# R85 MP4 decode to CPU BGR arrays: software OpenCV vs hardware GStreamer appsink.
set -eu
R=/home/jetson/nano_run/r85_nms_probe
for variant in cv2 gst gst cv2; do
  echo "variant=$variant"
  if [ "$variant" = cv2 ]; then
    python3 "$R/cv2_decode_smoke.py"
  else
    python3 "$R/gst_appsink_probe.py"
  fi
  sleep 5
done
