$ErrorActionPreference='Stop'
& scp -q (Join-Path $PSScriptRoot 'queued.sh') nano:/home/jetson/nano_run/r160_height352_20261120/
if($LASTEXITCODE){throw 'R160 queue transfer failed'}
& ssh -o BatchMode=yes -o ConnectTimeout=8 nano 'bash /home/jetson/nano_run/r160_height352_20261120/queued.sh'
if($LASTEXITCODE){throw 'R160 build/capture failed'}
& ssh nano 'tar -cf /home/jetson/nano_run/r160_height352_20261120/evidence.tar -C /home/jetson/nano_run/r160_height352_20261120 cache models.json build.log raw_parity.log capture.log'
if($LASTEXITCODE){throw 'R160 evidence archive failed'}
& scp -q nano:/home/jetson/nano_run/r160_height352_20261120/evidence.tar $PSScriptRoot
if($LASTEXITCODE){throw 'R160 evidence transfer failed'}
& tar -xf (Join-Path $PSScriptRoot 'evidence.tar') -C $PSScriptRoot
if($LASTEXITCODE){throw 'R160 evidence extraction failed'}
& ssh nano 'test -f /home/jetson/nano_run/r160_height352_20261120/r352x640_parity.json'
if($LASTEXITCODE -eq 0) {
    & scp -q nano:/home/jetson/nano_run/r160_height352_20261120/r352x640_parity.json $PSScriptRoot
    if($LASTEXITCODE){throw 'R160 raw parity marker transfer failed'}
} elseif($LASTEXITCODE -ne 1){throw 'R160 raw parity marker check failed'}
& 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe' (Join-Path $PSScriptRoot '../review155_runtime_accuracy/score.py') score (Join-Path $PSScriptRoot 'cache/r352x640') *> (Join-Path $PSScriptRoot 'score.log')
if($LASTEXITCODE){Get-Content (Join-Path $PSScriptRoot 'score.log') -Tail 12;throw 'R160 scoring failed'}
Write-Output 'HEIGHT352_ACTUAL_AP_SCORED'
