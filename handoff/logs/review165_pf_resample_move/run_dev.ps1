$ErrorActionPreference='Stop'
$taskPython='C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
$taskHere=Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location 'D:/LABOTARY/bytetrack'
& $taskPython "$taskHere/prepare.py" *> "$taskHere/preflight.log"
if ($LASTEXITCODE -ne 0) { throw 'R165 preflight failed' }
foreach ($taskAction in @('smoke','worker','summary')) {
    Write-Output "R165 START $taskAction"
    & $taskPython "$taskHere/run.py" $taskAction *> "$taskHere/${taskAction}.log"
    if ($LASTEXITCODE -ne 0) { throw "R165 $taskAction failed" }
}
Write-Output 'R165 DEV36 COMPLETE'
