$ErrorActionPreference = 'Stop'
$taskHere = Split-Path -Parent $MyInvocation.MyCommand.Path
$taskPython = 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
$taskRemote = '/home/jetson/nano_run/r169_paired_cuda_resume_20261121/recovered'
$taskSpeed = '/home/jetson/nano_run/r169_paired_cuda_resume_20261121/speed'
Set-Location 'D:/LABOTARY/bytetrack'
$env:YOLO_CONFIG_DIR = Join-Path $taskHere 'settings'
function Check-R169Exit([string]$taskStep) {
    if ($LASTEXITCODE -ne 0) { throw "R169 runtime failed at $taskStep" }
}
& $taskPython "$taskHere/prepare_recovered_stage.py" *> "$taskHere/recovered_stage_prepare.log"
Check-R169Exit 'prepare recovered stage'
ssh nano "mkdir -p $taskRemote"
Check-R169Exit 'create remote isolated folder'
scp "$taskHere/recovered_stage.tar" "nano:$taskRemote/recovered_stage.tar"
Check-R169Exit 'upload recovered stage'
$taskRemoteHash = (ssh nano "sha256sum $taskRemote/recovered_stage.tar").Split()[0]
Check-R169Exit 'hash recovered stage'
$taskExpectedHash = (Get-Content -Raw "$taskHere/recovered_stage_preflight.json" | ConvertFrom-Json).archive_sha256
if ($taskRemoteHash -ne $taskExpectedHash) { throw 'Recovered stage SHA256 mismatch' }
ssh nano "cd $taskRemote && tar -xf recovered_stage.tar && PYTHONUNBUFFERED=1 LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 build_recovered.py > runtime.log 2>&1"
Check-R169Exit 'build recovered engines and actual captures'
ssh nano "cd $taskRemote && tar -cf recovered_evidence.tar cache config.json runtime_models.json build_results.json before.json complete.json build_recovered.py runtime.log *_build.log *_capture.log *_raw_parity.json"
Check-R169Exit 'archive actual captures'
$taskRemoteHash = (ssh nano "sha256sum $taskRemote/recovered_evidence.tar").Split()[0]
Check-R169Exit 'hash actual captures'
scp "nano:$taskRemote/recovered_evidence.tar" "$taskHere/recovered_evidence.tar"
Check-R169Exit 'fetch actual captures'
if ((Get-FileHash -LiteralPath "$taskHere/recovered_evidence.tar" -Algorithm SHA256).Hash.ToLower() -ne $taskRemoteHash) { throw 'Actual captures SHA256 mismatch' }
$taskEvidence = Join-Path $taskHere 'recovered_evidence'
New-Item -ItemType Directory -Path $taskEvidence -ErrorAction Stop | Out-Null
tar -xf "$taskHere/recovered_evidence.tar" -C $taskEvidence
Check-R169Exit 'extract actual captures'
$taskModels = (Get-Content -Raw "$taskEvidence/runtime_models.json" | ConvertFrom-Json).models.PSObject.Properties.Name
foreach ($taskMode in $taskModels) {
    & $taskPython "$taskHere/../review155_runtime_accuracy/score.py" score "$taskEvidence/cache/$taskMode" *> "$taskHere/${taskMode}_actual_AP.log"
    Check-R169Exit "score actual AP $taskMode"
}
Write-Output 'R169 ACTUAL TENSORRT QUALITY COMPLETE'
& $taskPython "$taskHere/prepare_speed.py" *> "$taskHere/speed_prepare.log"
Check-R169Exit 'prepare speed stage'
ssh nano "mkdir -p $taskSpeed"
Check-R169Exit 'create speed folder'
scp -r "$taskHere/speed_stage/." "nano:$taskSpeed/"
Check-R169Exit 'upload speed stage'
scp "$taskHere/../review155_runtime_accuracy/collect_extra.py" "nano:$taskSpeed/collect_extra.py"
Check-R169Exit 'upload frozen collector'
$taskMemory = ssh nano "cat /proc/meminfo"
Check-R169Exit 'read final timing memory gate'
$taskTotal = [double](($taskMemory | Where-Object { $_ -match '^SwapTotal:' }) -split '\s+')[1]
$taskFree = [double](($taskMemory | Where-Object { $_ -match '^SwapFree:' }) -split '\s+')[1]
$taskSwap = ($taskTotal - $taskFree) / 1024.0
if ($taskSwap -ge 100) {
    @{ status='WAITING_FOR_TIMING_GUARD'; swap_MiB=$taskSwap; maximum_start_swap_MiB=100; actual_AP_complete=$true; speed_stage_uploaded=$true } |
        ConvertTo-Json | Set-Content -LiteralPath "$taskHere/timing_pending.json" -Encoding utf8
    Write-Output "R169 ACTUAL AP COMPLETE; TIMING WAITING FOR SWAP GUARD ($taskSwap MiB)"
    exit 0
}
& "$taskHere/finish_timing.ps1"
if ($LASTEXITCODE -ne 0) { throw 'R169 timing stage failed' }
