param([Parameter(Mandatory = $true)][int]$TrackingQueuePid)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
Set-Location -LiteralPath $repo
$active = Get-CimInstance Win32_Process -Filter "ProcessId=$TrackingQueuePid"
if ($active -and $active.CommandLine -match 'queue_svmot\.ps1') {
    Wait-Process -Id $TrackingQueuePid
}

$py = 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
$out = Join-Path $PSScriptRoot 'detection_interp50'
$settings = New-Item -Path $out -Name 'ultralytics' -ItemType Directory -Force
$env:YOLO_CONFIG_DIR = $settings.FullName
$common = @('--model', 'interp50=models/r101_ridesafe_interp50/yolo11n_mixed5_interp50.pt',
    '--sha256', 'interp50=d6aad8667ea7d344592c5aa170a7dbcda968c4aa30930cecc08d3541566daa82',
    '--hinh-dang', '384x640', '--conf-van-hanh', '0.25', '--batch', '16', '--device', '0')
$failed = $false
foreach ($dataset in 'bdd_val', 'detrac_test') {
    $adapter = "src/danh_gia_detector_$dataset.py"
    $log = Join-Path $out "$dataset.log"
    & $py -B -u -X utf8 $adapter @common --out (Join-Path $out $dataset) *> $log
    $code = $LASTEXITCODE
    Add-Content -LiteralPath $log -Value "EXIT $code"
    $failed = $failed -or ($code -ne 0)
}
exit [int]$failed
