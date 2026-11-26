param([Parameter(Mandatory = $true)][int]$HelmetPid)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
Set-Location -LiteralPath $repo
$active = Get-CimInstance Win32_Process -Filter "ProcessId=$HelmetPid"
if ($active -and $active.CommandLine -match 'review108_followup[/\\]helmet182\.py') {
    Wait-Process -Id $HelmetPid
}
$py = 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
& $py -B -X utf8 (Join-Path $PSScriptRoot 'svmot30.py') *> (Join-Path $PSScriptRoot 'svmot30.log')
exit $LASTEXITCODE
