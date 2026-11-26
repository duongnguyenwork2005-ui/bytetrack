$ErrorActionPreference = 'Stop'
$taskHere = Split-Path -Parent $MyInvocation.MyCommand.Path
$taskRemote = '/home/jetson/nano_run/r174_silu_graph_20261126'
ssh -o HostName=10.11.50.200 -o BatchMode=yes -o ConnectTimeout=10 nano "cd $taskRemote && test -f complete.json && tar -czf evidence_v2.tar.gz -- 00_protocol.md screen.py bindings.json inputs.json rows.json complete.json sampleInference.cpp sampleDevice.h screen.log [0-9][0-9]_*.log"
if ($LASTEXITCODE -ne 0) { throw 'R174 collection v2 failed; do not rerun screen' }
$taskHash = (ssh -o HostName=10.11.50.200 -o BatchMode=yes -o ConnectTimeout=10 nano "sha256sum $taskRemote/evidence_v2.tar.gz").Split()[0]
if ($LASTEXITCODE -ne 0) { throw 'R174 v2 remote hash failed' }
scp -o HostName=10.11.50.200 -o BatchMode=yes -o ConnectTimeout=10 "nano:$taskRemote/evidence_v2.tar.gz" "$taskHere/evidence_v2.tar.gz"
if ($LASTEXITCODE -ne 0) { throw 'R174 v2 transfer failed' }
if ((Get-FileHash -LiteralPath "$taskHere/evidence_v2.tar.gz" -Algorithm SHA256).Hash.ToLower() -ne $taskHash) { throw 'R174 v2 archive SHA mismatch' }
New-Item -ItemType Directory -Path "$taskHere/evidence" -ErrorAction Stop | Out-Null
tar -xf "$taskHere/evidence_v2.tar.gz" -C "$taskHere/evidence"
if ($LASTEXITCODE -ne 0) { throw 'R174 v2 extract failed' }
Write-Output 'R174 EVIDENCE V2 COLLECTED; ORIGINAL 12 BENCHMARKS PRESERVED'
