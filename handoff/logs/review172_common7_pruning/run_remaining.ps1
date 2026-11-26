# One-off experiment continuation, not a recurring app automation.
$ErrorActionPreference='Stop'
$taskHere=Split-Path -Parent $MyInvocation.MyCommand.Path
$taskDeadline=(Get-Date).AddHours(6)
Set-Location 'D:/LABOTARY/bytetrack'
Write-Output 'R172 WAITING FOR FIXED COMMON7 EXPORTS AND R167 COLLECTION'
while($true){
    if(Test-Path -LiteralPath "$taskHere/recovered_exports.json"){
        $taskExports=Get-Content -Raw "$taskHere/recovered_exports.json" | ConvertFrom-Json
        if($taskExports.status -ne 'PASS'){throw 'R172 exports did not pass'}
        ssh nano 'test -f /home/jetson/nano_run/r167_fused_silu_20261121/speed/collection.log && grep -q EXTRA_FULL80_EVIDENCE_READY /home/jetson/nano_run/r167_fused_silu_20261121/speed/collection.log'
        if($LASTEXITCODE -eq 0){break}
    }
    if((Get-Date) -gt $taskDeadline){throw 'R172 prerequisites not ready; runtime was not started'}
    Start-Sleep -Seconds 20
}
& "$taskHere/continue_runtime.ps1"
if($LASTEXITCODE -ne 0){throw 'R172 runtime chain failed'}
Write-Output 'R172 RUNTIME RETURNED; REVIEW FULL80, NEGATIVE_PILOT OR TIMING_PENDING'
