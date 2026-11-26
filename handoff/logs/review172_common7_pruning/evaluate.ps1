$ErrorActionPreference='Stop'
$taskHere=Split-Path -Parent $MyInvocation.MyCommand.Path
$taskPython='C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
Set-Location 'D:/LABOTARY/bytetrack'
& $taskPython "$taskHere/prepare.py" *> "$taskHere/prepare.log"
if($LASTEXITCODE -ne 0){throw 'R172 checkpoint preparation failed'}
foreach($taskMode in @('anchor','control','joint50')){
    & $taskPython "$taskHere/evaluate.py" $taskMode *> "$taskHere/${taskMode}_dev.log"
    if($LASTEXITCODE -ne 0){throw "R172 dev evaluation failed: $taskMode"}
}
& $taskPython "$taskHere/export_recovered.py" *> "$taskHere/recovered_exports.log"
if($LASTEXITCODE -ne 0){throw 'R172 common7 export failed'}
Write-Output 'R172 COMMON7 PC QUALITY AND EXPORT COMPLETE; PRIMARY10 REMAINS INCOMPLETE'
