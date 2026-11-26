#!/usr/bin/env bash
set -euo pipefail
root=/home/jetson/nano_run/r160_height352_20261120
echo $$ > "$root/queue.pid"
prior=/home/jetson/nano_run/r150_pruning_20261120/cache/channel25_KD/complete.json
while test ! -f "$prior" || pgrep -f '^python3 ([^ ]*/)?capture.py ' >/dev/null || pgrep -x trtexec >/dev/null || \
      pgrep -f '^/home/jetson/miniforge3/envs/nano/bin/python src/nano_live.py ' >/dev/null; do
    sleep 10
done
echo HEIGHT352_START_AFTER_R150_CAPTURE
bash "$root/build_capture.sh"
echo HEIGHT352_QUEUE_COMPLETE
