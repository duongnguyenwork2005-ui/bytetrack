$ErrorActionPreference='Stop'
$taskHere=Split-Path -Parent $MyInvocation.MyCommand.Path
$taskPython='C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
$taskRemote='/home/jetson/nano_run/r167_fused_silu_20261121'
$taskBase='/home/jetson/nano_run/r155_runtime_accuracy_20261120'
Set-Location 'D:/LABOTARY/bytetrack'
New-Item -ItemType Directory -Path "$taskHere/libraries" | Out-Null
scp "nano:$taskRemote/fused_silu_precise.so" "nano:$taskRemote/fused_silu_fast.so" "$taskHere/libraries/"
if($LASTEXITCODE -ne 0){throw 'R167 library fetch failed'}
& $taskPython "$taskHere/verify_screen.py" *> "$taskHere/screen_verification.log"
if($LASTEXITCODE -ne 0){throw 'R167 GPU verification failed'}
& $taskPython "$taskHere/prepare_actual.py" *> "$taskHere/actual_prepare.log"
if($LASTEXITCODE -ne 0){throw 'R167 actual AP preparation failed'}
scp "$taskHere/capture.py" "$taskHere/actual_models.json" "nano:$taskRemote/"
if($LASTEXITCODE -ne 0){throw 'R167 AP sources upload failed'}
$taskConfig=Get-Content -Raw "$taskHere/actual_models.json" | ConvertFrom-Json
foreach($taskMode in $taskConfig.models.PSObject.Properties.Name){
    Write-Output "R167 ACTUAL AP CAPTURE START $taskMode"
    ssh nano "cd $taskRemote && PYTHONUNBUFFERED=1 LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 capture.py $taskRemote/actual_models.json $taskMode $taskBase/dataset.json $taskBase/dev_images.tar $taskRemote/cache > ${taskMode}_capture.log 2>&1"
    if($LASTEXITCODE -ne 0){throw "R167 $taskMode actual capture failed"}
}
ssh nano "cd $taskRemote && tar -cf actual_evidence.tar cache actual_models.json capture.py *_capture.log"
if($LASTEXITCODE -ne 0){throw 'R167 AP archive failed'}
$taskHash=(ssh nano "sha256sum $taskRemote/actual_evidence.tar").Split()[0]
if($LASTEXITCODE -ne 0){throw 'R167 AP remote hash failed'}
scp "nano:$taskRemote/actual_evidence.tar" "$taskHere/actual_evidence.tar"
if($LASTEXITCODE -ne 0){throw 'R167 AP evidence fetch failed'}
if((Get-FileHash "$taskHere/actual_evidence.tar" -Algorithm SHA256).Hash.ToLower() -ne $taskHash){throw 'R167 AP evidence SHA mismatch'}
New-Item -ItemType Directory -Path "$taskHere/actual_evidence" | Out-Null
tar -xf "$taskHere/actual_evidence.tar" -C "$taskHere/actual_evidence"
if($LASTEXITCODE -ne 0){throw 'R167 AP extraction failed'}
foreach($taskMode in $taskConfig.models.PSObject.Properties.Name){
    & $taskPython "$taskHere/../review155_runtime_accuracy/score.py" score "$taskHere/actual_evidence/cache/$taskMode" *> "$taskHere/${taskMode}_AP.log"
    if($LASTEXITCODE -ne 0){throw "R167 $taskMode AP scoring failed"}
    Write-Output "R167 ACTUAL AP SCORED $taskMode"
}
Write-Output 'R167 ALL ACTUAL AP COMPLETE'
