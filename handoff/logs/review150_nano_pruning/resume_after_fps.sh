#!/usr/bin/env bash
set -euo pipefail
root=/home/jetson/nano_run/r150_pruning_20261120
audit=/home/jetson/nano_run/r155_runtime_accuracy_20261120
echo $$ > "$root/resume_queue.pid"
while test ! -f "$audit/runs_absolute/full/complete.json" || \
      test ! -f "$audit/runs_absolute/full/after.json" || \
      pgrep -f '^/home/jetson/miniforge3/envs/nano/bin/python timing_v2.py full ' >/dev/null || \
      pgrep -f '^/home/jetson/miniforge3/envs/nano/bin/python src/nano_live.py ' >/dev/null; do
    sleep 10
done
cd "$root"
python3 - <<'PY'
from pathlib import Path
import hashlib,json
p=Path('cache/channel25_KD/inputs.json');old=json.loads(p.read_text())
note={'old_boot':old['boot'],'new_boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
      'inputs_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
      'preserved_complete_chunks':sum(Path('cache/channel25_KD/%04d.json'%i).exists() for i in range(72)),
      'purpose':'Resume stateless per-image accuracy capture; no mixed-boot FPS aggregation'}
with Path('resume_20261120.json').open('x') as stream:json.dump(note,stream,indent=2)
PY
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 python3 "$audit/capture.py" models.json channel25_KD "$audit/dataset.json" "$audit/dev_images.tar" cache \
 > resume_channel25_20261120.log 2>&1
echo PRUNING_REMAINING_ACCURACY_CAPTURE_COMPLETE
