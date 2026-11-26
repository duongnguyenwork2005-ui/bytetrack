$ErrorActionPreference='Stop'
$taskHere=Split-Path -Parent $MyInvocation.MyCommand.Path
$taskPython='C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
Set-Location 'D:/LABOTARY/bytetrack'
& $taskPython "$taskHere/prepare.py" *> "$taskHere/prepare.log"
if($LASTEXITCODE -ne 0){throw 'R169 prepare failed'}
foreach($taskMode in @('control','joint50')){
    Write-Output "R169 RESUME START $taskMode"
    & $taskPython "$taskHere/resume.py" $taskMode *> "$taskHere/${taskMode}_train.log"
    if($LASTEXITCODE -ne 0){throw "R169 $taskMode resume failed"}
}
foreach($taskMode in @('control','joint50')){
    & $taskPython "$taskHere/run.py" select $taskMode *> "$taskHere/${taskMode}_monitor.log"
    if($LASTEXITCODE -ne 0){throw "R169 $taskMode monitor failed"}
}
foreach($taskMode in @('anchor','control','joint50')){
    & $taskPython "$taskHere/run.py" dev $taskMode *> "$taskHere/${taskMode}_dev.log"
    if($LASTEXITCODE -ne 0){throw "R169 $taskMode dev failed"}
}
& $taskPython "$taskHere/run.py" export *> "$taskHere/recovered_exports.log"
if($LASTEXITCODE -ne 0){throw 'R169 export failed'}
Write-Output 'R169 PAIRED RECOVERY AND PC QUALITY COMPLETE'
