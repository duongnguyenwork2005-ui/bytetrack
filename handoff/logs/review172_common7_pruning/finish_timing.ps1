$ErrorActionPreference = 'Stop'
$taskHere = Split-Path -Parent $MyInvocation.MyCommand.Path
$taskPython = 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
$taskRemote = '/home/jetson/nano_run/r172_common7_pruning_20261121/recovered'
$taskSpeed = '/home/jetson/nano_run/r172_common7_pruning_20261121/speed'
Set-Location 'D:/LABOTARY/bytetrack'
$env:YOLO_CONFIG_DIR = Join-Path $taskHere 'settings'
function Check-R172Exit([string]$taskStep) {
    if ($LASTEXITCODE -ne 0) { throw "R172 runtime failed at $taskStep" }
}
ssh nano "cd $taskSpeed && mkdir -p runs_absolute && /home/jetson/miniforge3/envs/nano/bin/python timing_v4.py engine $taskSpeed/timing_models_v4.json $taskSpeed/runs_absolute/engine > engine.log 2>&1 && /home/jetson/miniforge3/envs/nano/bin/python timing_v4.py pilot $taskSpeed/timing_models_v4.json $taskSpeed/runs_absolute/pilot > pilot.log 2>&1 && python3 select_full.py $taskSpeed > selection.log 2>&1"
Check-R172Exit 'engine and pilot'
$taskSelection = (ssh nano "cat $taskSpeed/selection.json") | ConvertFrom-Json
Check-R172Exit 'read pilot selection'
if ($taskSelection.full_skipped) {
    scp "nano:$taskSpeed/selection.json" "$taskHere/negative_pilot.json"
    Check-R172Exit 'fetch negative pilot'
    Write-Output 'R172 NEGATIVE PIPELINE PILOT; FULL80 NOT STARTED'
    exit 0
}
ssh nano "cd $taskSpeed && /home/jetson/miniforge3/envs/nano/bin/python timing_v4.py full $taskSpeed/full_models.json $taskSpeed/runs_absolute/full > full.log 2>&1 && python3 collect_extra.py $taskSpeed > collection.log 2>&1"
Check-R172Exit 'full80 and evidence collection'
$taskArchive = 'full80_evidence_20261120.tar.gz'
$taskRemoteHash = (ssh nano "sha256sum $taskSpeed/$taskArchive").Split()[0]
Check-R172Exit 'hash full80'
scp "nano:$taskSpeed/$taskArchive" "$taskHere/full80_evidence.tar.gz"
Check-R172Exit 'fetch full80'
if ((Get-FileHash -LiteralPath "$taskHere/full80_evidence.tar.gz" -Algorithm SHA256).Hash.ToLower() -ne $taskRemoteHash) { throw 'Full80 SHA256 mismatch' }
$taskFullEvidence = Join-Path $taskHere 'full80_evidence'
New-Item -ItemType Directory -Path $taskFullEvidence -ErrorAction Stop | Out-Null
tar -xf "$taskHere/full80_evidence.tar.gz" -C $taskFullEvidence
Check-R172Exit 'extract full80'
& $taskPython "$taskHere/../review155_runtime_accuracy/score_full.py" "$taskFullEvidence/runs_absolute/full" *> "$taskHere/full80_score.log"
Check-R172Exit 'audit full80 timestamps and tracking'
Write-Output 'R172 FULL80 SPEED AND TRACKING COMPLETE'
