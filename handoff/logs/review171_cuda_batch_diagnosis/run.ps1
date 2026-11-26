$ErrorActionPreference='Stop'
$taskHere=Split-Path -Parent $MyInvocation.MyCommand.Path
$taskPython='C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
Set-Location 'D:/LABOTARY/bytetrack'
foreach($taskMode in @('control','joint50')){
    foreach($taskBackend in @('cudnn','nativeBN')){
        & $taskPython "$taskHere/probe.py" $taskMode $taskBackend *> "$taskHere/${taskMode}_${taskBackend}.log"
        Write-Output "R171 DIAGNOSTIC RETURN $taskMode $taskBackend $LASTEXITCODE"
    }
}
Write-Output 'R171 CAPTURED BATCH DIAGNOSTICS ENDED; REVIEW RESULTS BEFORE TRAINING'
