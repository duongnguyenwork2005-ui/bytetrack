$ErrorActionPreference = 'Stop'
$taskHere = Split-Path -Parent $MyInvocation.MyCommand.Path
$taskPython = 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
$taskSpeed = '/home/jetson/nano_run/r172_common7_pruning_20261121/speed'
$taskArchive = 'full80_evidence_20261120.tar.gz'
$taskLocalArchive = Join-Path $taskHere 'full80_evidence.tar.gz'
$taskEvidence = Join-Path $taskHere 'full80_evidence'
Set-Location 'D:/LABOTARY/bytetrack'
if (Test-Path -LiteralPath $taskEvidence) { throw 'Preserve existing evidence; review before recollection' }
$taskRemoteHashLine = ssh -o HostName=10.11.50.200 -o BatchMode=yes -o ConnectTimeout=10 nano "sha256sum $taskSpeed/$taskArchive"
if ($LASTEXITCODE -ne 0) { throw 'Remote archive unavailable' }
$taskRemoteHash = $taskRemoteHashLine.Split()[0]
if ($taskRemoteHash -ne 'a62cae8257d3a76a094fafa07e1f723fa595d94d467fed17ad0b7095d7d892ed') { throw 'Archive differs from completed collection log' }
if (-not (Test-Path -LiteralPath $taskLocalArchive)) {
    scp -o HostName=10.11.50.200 -o BatchMode=yes -o ConnectTimeout=10 "nano:$taskSpeed/$taskArchive" $taskLocalArchive
    if ($LASTEXITCODE -ne 0) { throw 'Archive transfer failed' }
}
if ((Get-FileHash -LiteralPath $taskLocalArchive -Algorithm SHA256).Hash.ToLower() -ne $taskRemoteHash) { throw 'Archive SHA256 mismatch' }
New-Item -ItemType Directory -Path $taskEvidence -ErrorAction Stop | Out-Null
tar -xf $taskLocalArchive -C $taskEvidence
if ($LASTEXITCODE -ne 0) { throw 'Evidence extraction failed' }
& $taskPython "$taskHere/../review155_runtime_accuracy/score_full.py" "$taskEvidence/runs_absolute/full" *> "$taskHere/full80_score.log"
if ($LASTEXITCODE -ne 0) { throw 'Full80 verification/scoring failed; preserve evidence and log' }
Write-Output 'R172 RECONNECTED COLLECTION AND SCORING COMPLETE'
