$ErrorActionPreference='Stop'
& scp -q (Join-Path $PSScriptRoot 'collect_full.py') nano:/home/jetson/nano_run/r155_runtime_accuracy_20261120/
if($LASTEXITCODE){throw 'Collector transfer failed'}
& ssh -o BatchMode=yes -o ConnectTimeout=8 nano '/home/jetson/miniforge3/envs/nano/bin/python /home/jetson/nano_run/r155_runtime_accuracy_20261120/collect_full.py'
if($LASTEXITCODE){throw 'Full80 evidence collection failed'}
& scp -q nano:/home/jetson/nano_run/r155_runtime_accuracy_20261120/full80_evidence_20261120.tar.gz $PSScriptRoot
if($LASTEXITCODE){throw 'Full80 evidence transfer failed'}
$taskDestination=Join-Path $PSScriptRoot 'full80_evidence'
if(Test-Path -LiteralPath $taskDestination){throw 'Evidence destination already exists'}
New-Item -ItemType Directory -Path $taskDestination | Out-Null
& tar -xf (Join-Path $PSScriptRoot 'full80_evidence_20261120.tar.gz') -C $taskDestination
if($LASTEXITCODE){throw 'Full80 evidence extraction failed'}
& 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe' (Join-Path $PSScriptRoot 'score_full.py') (Join-Path $taskDestination 'runs_absolute/full') *> (Join-Path $PSScriptRoot 'score_full.log')
if($LASTEXITCODE){Get-Content (Join-Path $PSScriptRoot 'score_full.log') -Tail 16;throw 'Full80 speed/quality validation failed'}
Write-Output 'FULL80_SCORED_VERIFIED'
