$ErrorActionPreference='Stop'
$taskPython='C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
$taskHere=Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location 'D:/LABOTARY/bytetrack'
foreach ($taskAction in @('numerical','preflight','smoke','worker','summary')) {
    Write-Output "R168 START $taskAction"
    & $taskPython "$taskHere/run.py" $taskAction *> "$taskHere/${taskAction}.log"
    if ($LASTEXITCODE -ne 0) { throw "R168 $taskAction failed" }
}
Write-Output 'R168 DEV36 COMPLETE'
