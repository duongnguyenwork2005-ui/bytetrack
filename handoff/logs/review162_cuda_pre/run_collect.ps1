$ErrorActionPreference='Stop'
$here=$PSScriptRoot
$remote='/home/jetson/nano_run/r162_cuda_pre_20261120'
$base='/home/jetson/nano_run/r155_runtime_accuracy_20261120'
scp "$here/capture.py" "$here/models.json" "nano:$remote/"
if ($LASTEXITCODE -ne 0) { throw 'CUDA capture staging failed' }
ssh nano "PYTHONUNBUFFERED=1 LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 $remote/capture.py $remote/models.json gpu_pre $base/dataset.json $base/dev_images.tar $remote/cache > $remote/capture.log 2>&1"
if ($LASTEXITCODE -ne 0) { throw 'CUDA actual detection capture failed' }
ssh nano "cd $remote && tar -cf capture_evidence.tar cache models.json capture.py capture.log oracle.json oracle.log compiled.sha256 gpu_pre.py fused_pre.cu fused_pre.so 00_protocol.md"
if ($LASTEXITCODE -ne 0) { throw 'CUDA evidence archive failed' }
scp "nano:$remote/capture_evidence.tar" "$here/capture_evidence.tar"
if ($LASTEXITCODE -ne 0) { throw 'CUDA evidence copy failed' }
$output=Join-Path $here 'capture_evidence'
New-Item -ItemType Directory -Path $output | Out-Null
tar -xf "$here/capture_evidence.tar" -C $output
if ($LASTEXITCODE -ne 0) { throw 'CUDA evidence extraction failed' }
$score=Join-Path $here '../review155_runtime_accuracy/score.py'
& 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe' -u $score score "$output/cache/gpu_pre" *> "$here/score.log"
if ($LASTEXITCODE -ne 0) { throw 'CUDA AP scoring failed' }
Write-Output 'CUDA_ACTUAL_AP_COMPLETE'
