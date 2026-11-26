$ErrorActionPreference = 'Stop'
$taskHere = Split-Path -Parent $MyInvocation.MyCommand.Path
$taskRemote = '/home/jetson/nano_run/r174_silu_graph_20261126'
$taskPython = 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
Set-Location 'D:/LABOTARY/bytetrack'
& $taskPython "$taskHere/prepare.py"
if ($LASTEXITCODE -ne 0) { throw 'R174 prepare failed' }
scp -o HostName=10.11.50.200 -o BatchMode=yes -o ConnectTimeout=10 "$taskHere/00_protocol.md" "$taskHere/screen.py" "$taskHere/bindings.json" "nano:$taskRemote/"
if ($LASTEXITCODE -ne 0) { throw 'R174 upload failed' }
ssh -o HostName=10.11.50.200 -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 nano "cd $taskRemote && python3 screen.py > screen.log 2>&1 && tar -czf evidence.tar.gz --exclude=evidence.tar.gz ."
if ($LASTEXITCODE -ne 0) { throw 'R174 screen failed; preserve logs' }
$taskHash = (ssh -o HostName=10.11.50.200 -o BatchMode=yes -o ConnectTimeout=10 nano "sha256sum $taskRemote/evidence.tar.gz").Split()[0]
if ($LASTEXITCODE -ne 0) { throw 'R174 remote hash failed' }
scp -o HostName=10.11.50.200 -o BatchMode=yes -o ConnectTimeout=10 "nano:$taskRemote/evidence.tar.gz" "$taskHere/evidence.tar.gz"
if ($LASTEXITCODE -ne 0) { throw 'R174 fetch failed' }
if ((Get-FileHash -LiteralPath "$taskHere/evidence.tar.gz" -Algorithm SHA256).Hash.ToLower() -ne $taskHash) { throw 'R174 archive SHA mismatch' }
New-Item -ItemType Directory -Path "$taskHere/evidence" -ErrorAction Stop | Out-Null
tar -xf "$taskHere/evidence.tar.gz" -C "$taskHere/evidence"
if ($LASTEXITCODE -ne 0) { throw 'R174 extract failed' }
Write-Output 'R174 GPU SCREEN COLLECTED; VERIFY BEFORE INTERPRETATION'
