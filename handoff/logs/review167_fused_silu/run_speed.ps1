$ErrorActionPreference='Stop'
$taskHere=Split-Path -Parent $MyInvocation.MyCommand.Path
$taskPython='C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
$taskSpeed='/home/jetson/nano_run/r167_fused_silu_20261121/speed'
Set-Location 'D:/LABOTARY/bytetrack'
& $taskPython "$taskHere/bind_timing.py" *> "$taskHere/timing_bound_prepare.log"
if($LASTEXITCODE -ne 0){throw 'R167 timing binding failed'}
ssh nano "mkdir -p $taskSpeed/runs_absolute"
if($LASTEXITCODE -ne 0){throw 'R167 timing folder failed'}
scp -r "$taskHere/speed_stage/." "nano:$taskSpeed/"
if($LASTEXITCODE -ne 0){throw 'R167 timing inputs upload failed'}
ssh nano "cd $taskSpeed && python3 record_environment.py > environment.log 2>&1 && /home/jetson/miniforge3/envs/nano/bin/python timing_v5.py engine $taskSpeed/timing_models_v5_bound.json $taskSpeed/runs_absolute/engine > engine.log 2>&1 && /home/jetson/miniforge3/envs/nano/bin/python timing_v5.py pilot $taskSpeed/timing_models_v5_bound.json $taskSpeed/runs_absolute/pilot > pilot.log 2>&1 && python3 select_full_bound.py $taskSpeed > selection.log 2>&1"
if($LASTEXITCODE -ne 0){throw 'R167 engine/pilot failed'}
$taskSelection=(ssh nano "cat $taskSpeed/selection.json") | ConvertFrom-Json
if($LASTEXITCODE -ne 0){throw 'R167 pilot selection read failed'}
if($taskSelection.full_skipped){
    scp "nano:$taskSpeed/selection.json" "$taskHere/negative_pilot.json"
    if($LASTEXITCODE -ne 0){throw 'R167 negative pilot fetch failed'}
    Write-Output 'R167 NEGATIVE PILOT; FULL80 NOT STARTED'
    exit 0
}
Write-Output 'R167 FULL80 STARTED'
ssh nano "cd $taskSpeed && /home/jetson/miniforge3/envs/nano/bin/python timing_v5.py full $taskSpeed/full_models.json $taskSpeed/runs_absolute/full > full.log 2>&1 && python3 record_environment.py environment_after.json > environment_after.log 2>&1 && python3 collect_full.py $taskSpeed > collection.log 2>&1"
if($LASTEXITCODE -ne 0){throw 'R167 full80/collection failed'}
$taskArchive='full80_evidence_20261121.tar.gz'
$taskHash=(ssh nano "sha256sum $taskSpeed/$taskArchive").Split()[0]
if($LASTEXITCODE -ne 0){throw 'R167 full80 remote hash failed'}
scp "nano:$taskSpeed/$taskArchive" "$taskHere/full80_evidence.tar.gz"
if($LASTEXITCODE -ne 0){throw 'R167 full80 fetch failed'}
if((Get-FileHash "$taskHere/full80_evidence.tar.gz" -Algorithm SHA256).Hash.ToLower() -ne $taskHash){throw 'R167 full80 SHA mismatch'}
New-Item -ItemType Directory -Path "$taskHere/full80_evidence" | Out-Null
tar -xf "$taskHere/full80_evidence.tar.gz" -C "$taskHere/full80_evidence"
if($LASTEXITCODE -ne 0){throw 'R167 full80 extraction failed'}
& $taskPython "$taskHere/../review155_runtime_accuracy/score_full.py" "$taskHere/full80_evidence/runs_absolute/full" *> "$taskHere/full80_score.log"
if($LASTEXITCODE -ne 0){throw 'R167 full80 host audit failed'}
Write-Output 'R167 FULL80 SPEED AND TRACKING AUDIT COMPLETE'
