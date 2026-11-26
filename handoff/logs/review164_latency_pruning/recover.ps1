$ErrorActionPreference = 'Stop'
$taskPython = 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
$taskHere = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location 'D:/LABOTARY/bytetrack'
function Run-R164Python([string]$taskScript, [string[]]$taskArgs, [string]$taskLog) {
    & $taskPython (Join-Path $taskHere $taskScript) @taskArgs *> (Join-Path $taskHere $taskLog)
    if ($LASTEXITCODE -ne 0) { throw "R164 failed: $taskScript $taskArgs; see $taskLog" }
}
Run-R164Python 'verify_screen.py' @() 'screen_verification.log'
$taskSelection = Get-Content -Raw (Join-Path $taskHere 'screen_evidence/selection.json') | ConvertFrom-Json
if (@($taskSelection.selected_for_training).Count -eq 0) {
    Write-Output 'R164 NEGATIVE GPU SCREEN; RECOVERY NOT STARTED'
    exit 0
}
Run-R164Python 'train.py' @('prepare') 'training_prepare.log'
$taskContract = Get-Content -Raw (Join-Path $taskHere 'training_inputs.json') | ConvertFrom-Json
$taskModes = @($taskContract.models.PSObject.Properties.Name)
foreach ($taskMode in $taskModes) {
    Write-Output "R164 RECOVERY START $taskMode"
    Run-R164Python 'train.py' @('train', $taskMode) ($taskMode + '_train.log')
    Write-Output "R164 TRAINED $taskMode"
    Run-R164Python 'monitor.py' @('select', $taskMode) ($taskMode + '_monitor.log')
    Write-Output "R164 EPOCH SELECTED $taskMode"
}
# The 7190-image dev set is opened only after epoch selection is complete.
foreach ($taskMode in @('anchor') + $taskModes) {
    Run-R164Python 'monitor.py' @('dev', $taskMode) ($taskMode + '_dev.log')
    Write-Output "R164 DEV SCORED $taskMode"
}
Run-R164Python 'export_recovered.py' @() 'recovered_exports.log'
Write-Output 'R164 RECOVERY AND PC QUALITY COMPLETE'
