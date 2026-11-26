$ErrorActionPreference='Stop'
$taskRoot=Join-Path $PSScriptRoot 'cache'
New-Item -ItemType Directory -Path $taskRoot -ErrorAction SilentlyContinue | Out-Null
foreach($mode in @('anchor','control','channel25_KD')) {
    while($true) {
        & ssh -o BatchMode=yes -o ConnectTimeout=8 nano "test -f /home/jetson/nano_run/r150_pruning_20261120/cache/$mode/complete.json"
        if($LASTEXITCODE -eq 0){break}
        if($LASTEXITCODE -ne 1){throw 'Nano SSH failed while waiting for accuracy capture'}
        Start-Sleep -Seconds 20
    }
    & ssh nano "tar -cf /home/jetson/nano_run/r150_pruning_20261120/${mode}_cache.tar -C /home/jetson/nano_run/r150_pruning_20261120/cache $mode"
    if($LASTEXITCODE){throw 'Remote cache archive failed'}
    & scp -q "nano:/home/jetson/nano_run/r150_pruning_20261120/${mode}_cache.tar" $PSScriptRoot
    if($LASTEXITCODE){throw 'Cache transfer failed'}
    & tar -xf (Join-Path $PSScriptRoot "${mode}_cache.tar") -C $taskRoot
    if($LASTEXITCODE){throw 'Cache extraction failed'}
    & 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe' (Join-Path $PSScriptRoot '../review155_runtime_accuracy/score.py') score (Join-Path $taskRoot $mode) *> (Join-Path $PSScriptRoot "score_${mode}.log")
    if($LASTEXITCODE){Get-Content (Join-Path $PSScriptRoot "score_${mode}.log") -Tail 12;throw "Actual Nano AP scoring failed: $mode"}
    Write-Output "PRUNING_ACTUAL_AP_SCORED $mode"
}
& scp -q nano:/home/jetson/nano_run/r150_pruning_20261120/models.json (Join-Path $PSScriptRoot 'runtime_models.json')
if($LASTEXITCODE){throw 'Model manifest transfer failed'}
foreach($mode in @('anchor','control','channel25_KD')) {
    & scp -q "nano:/home/jetson/nano_run/r150_pruning_20261120/${mode}_parity.log" $PSScriptRoot
    if($LASTEXITCODE){throw 'Raw parity log transfer failed'}
    & ssh nano "test -f /home/jetson/nano_run/r150_pruning_20261120/${mode}_parity.json"
    if($LASTEXITCODE -eq 0) {
        & scp -q "nano:/home/jetson/nano_run/r150_pruning_20261120/${mode}_parity.json" $PSScriptRoot
        if($LASTEXITCODE){throw 'Raw parity marker transfer failed'}
    } elseif($LASTEXITCODE -ne 1){throw 'Raw parity marker SSH check failed'}
}
Write-Output 'PRUNING_ACTUAL_AP_BATCH_COMPLETE'
