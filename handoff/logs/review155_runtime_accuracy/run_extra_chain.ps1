$ErrorActionPreference='Stop'
$PSNativeCommandUseErrorActionPreference=$false
$here=$PSScriptRoot
$remote='/home/jetson/nano_run/r155_runtime_accuracy_20261120'
scp "$here/run_extra_chain.py" "$here/collect_extra.py" "nano:$remote/"
if ($LASTEXITCODE -ne 0) { throw 'Extra speed driver staging failed' }
ssh nano "/home/jetson/miniforge3/envs/nano/bin/python -u $remote/run_extra_chain.py"
$runExit=$LASTEXITCODE
if ($runExit -notin @(0,75)) { throw "Extra speed chain failed with exit $runExit; inspect per-phase log" }
scp "nano:$remote/extra_chain_state.json" "$here/extra_chain_state.json"
if ($LASTEXITCODE -ne 0) { throw 'Chain state copy failed' }
foreach ($item in @(
    @{Folder='review162_cuda_pre';Remote='r162_cuda_pre_20261120'},
    @{Folder='review150_nano_pruning';Remote='r150_pruning_20261120'},
    @{Folder='review160_height352';Remote='r160_height352_20261120'})) {
    $destination=Join-Path $here "../$($item.Folder)"
    $source="/home/jetson/nano_run/$($item.Remote)"
    ssh nano "test -f $source/full80_evidence_20261120.tar.gz"
    if ($LASTEXITCODE -ne 0) { continue }
    $archive=Join-Path $destination 'full80_evidence_20261120.tar.gz'
    if (-not (Test-Path -LiteralPath $archive)) {
        scp "nano:$source/full80_evidence_20261120.tar.gz" $archive
        if ($LASTEXITCODE -ne 0) { throw 'Full80 evidence copy failed' }
    }
    $remoteHash=@(ssh nano "sha256sum $source/full80_evidence_20261120.tar.gz")[0].Split(' ')[0]
    if ($LASTEXITCODE -ne 0 -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLower() -ne $remoteHash) { throw 'Evidence archive hash mismatch' }
    $output=Join-Path $destination 'full80_evidence'
    if (-not (Test-Path -LiteralPath $output)) {
        New-Item -ItemType Directory -Path $output | Out-Null
        tar -xf $archive -C $output
        if ($LASTEXITCODE -ne 0) { throw 'Evidence extraction failed' }
    }
    $full=Join-Path $output 'runs_absolute/full'
    if (-not (Test-Path -LiteralPath (Join-Path $full 'score_verified.json'))) {
        & 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe' -u "$here/score_full.py" $full *> "$destination/score_full.log"
        if ($LASTEXITCODE -ne 0) { throw "Full80 independent scoring failed for $($item.Folder)" }
    }
}
if ($runExit -eq 75) { Write-Output 'EXTRA_CHAIN_WAITING_FOR_SWAP_RESET' }
else { Write-Output 'EXTRA_CHAIN_FULLY_COLLECTED_AND_SCORED' }
