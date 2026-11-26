$ErrorActionPreference='Stop'
while($true) {
    & ssh -o BatchMode=yes -o ConnectTimeout=8 nano 'test -f /home/jetson/nano_run/r150_pruning_20261120/cache/channel25_KD/complete.json'
    if($LASTEXITCODE -eq 0){break}
    if($LASTEXITCODE -ne 1){throw 'Nano disconnected while waiting for resumed accuracy capture'}
    Start-Sleep -Seconds 20
}
& ssh nano 'tar -cf /home/jetson/nano_run/r150_pruning_20261120/channel25_KD_cache.tar -C /home/jetson/nano_run/r150_pruning_20261120/cache channel25_KD'
if($LASTEXITCODE){throw 'Pruned cache archive failed'}
& scp -q nano:/home/jetson/nano_run/r150_pruning_20261120/channel25_KD_cache.tar $PSScriptRoot
if($LASTEXITCODE){throw 'Pruned cache transfer failed'}
& tar -xf (Join-Path $PSScriptRoot 'channel25_KD_cache.tar') -C (Join-Path $PSScriptRoot 'cache')
if($LASTEXITCODE){throw 'Pruned cache extraction failed'}
& 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe' (Join-Path $PSScriptRoot '../review155_runtime_accuracy/score.py') score (Join-Path $PSScriptRoot 'cache/channel25_KD') *> (Join-Path $PSScriptRoot 'score_channel25_KD.log')
if($LASTEXITCODE){Get-Content (Join-Path $PSScriptRoot 'score_channel25_KD.log') -Tail 15;throw 'Pruned actual AP scoring failed'}
& scp -q nano:/home/jetson/nano_run/r150_pruning_20261120/models.json (Join-Path $PSScriptRoot 'runtime_models.json')
if($LASTEXITCODE){throw 'Pruned model manifest transfer failed'}
foreach($mode in @('anchor','control','channel25_KD')) {
    & scp -q "nano:/home/jetson/nano_run/r150_pruning_20261120/${mode}_parity.log" $PSScriptRoot
    if($LASTEXITCODE){throw 'Raw parity log transfer failed'}
    & ssh nano "test -f /home/jetson/nano_run/r150_pruning_20261120/${mode}_parity.json"
    if($LASTEXITCODE -eq 0) {
        & scp -q "nano:/home/jetson/nano_run/r150_pruning_20261120/${mode}_parity.json" $PSScriptRoot
        if($LASTEXITCODE){throw 'Raw parity marker transfer failed'}
    } elseif($LASTEXITCODE -ne 1){throw 'Raw parity marker check failed'}
}
Write-Output 'PRUNED_RESUMED_ACTUAL_AP_SCORED'
