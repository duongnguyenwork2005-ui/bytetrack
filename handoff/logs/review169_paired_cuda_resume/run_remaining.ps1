# One-off continuation of this experiment, not a recurring app automation.
$ErrorActionPreference='Stop'
$taskHere=Split-Path -Parent $MyInvocation.MyCommand.Path
$taskDeadline=(Get-Date).AddHours(6)
Set-Location 'D:/LABOTARY/bytetrack'
Write-Output 'R169 WAITING FOR PAIRED EXPORTS AND R167 GPU RELEASE'
while($true){
    if(Get-ChildItem -LiteralPath $taskHere -Filter '*_failed_cpu_batch.pt'){
        throw 'R169 resume failed; preserve the captured batch and review before runtime work'
    }
    if(Test-Path -LiteralPath "$taskHere/recovered_exports.json"){
        $taskExports=Get-Content -Raw "$taskHere/recovered_exports.json" | ConvertFrom-Json
        if($taskExports.status -ne 'PASS'){throw 'R169 exports not verified'}
        ssh nano 'test -f /home/jetson/nano_run/r167_fused_silu_20261121/speed/collection.log && grep -q EXTRA_FULL80_EVIDENCE_READY /home/jetson/nano_run/r167_fused_silu_20261121/speed/collection.log'
        if($LASTEXITCODE -eq 0){break}
    }
    if((Get-Date) -gt $taskDeadline){throw 'R169 prerequisites not ready within the bounded wait; no runtime test started'}
    Start-Sleep -Seconds 20
}
Write-Output 'R169 RUNTIME START; R167 NANO TIMING HAS FINISHED'
& "$taskHere/continue_runtime.ps1"
if($LASTEXITCODE -ne 0){throw 'R169 runtime chain failed'}
Write-Output 'R169 RUNTIME CHAIN RETURNED; INSPECT FULL80 OR TIMING_PENDING'
