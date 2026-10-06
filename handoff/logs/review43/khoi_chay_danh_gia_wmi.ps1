# Vong 43: khoi chay MOT luot danh gia BDD val (mot model) qua src/chay_giam_sat.py trong tien trinh DOC LAP voi phien
# Claude (WMI Win32_Process.Create: cha WmiPrvSE, cua so an; cung co che vong 42 da nghiem thu). Checkpoint + sha256 co
# dinh theo ke hoach khoa (logs/review43/04_ke_hoach_danh_gia.md muc 2); khong nhan duong dan tuy y.
#   powershell -NoProfile -ExecutionPolicy Bypass -File handoff\logs\review43\khoi_chay_danh_gia_wmi.ps1 -Ten yolov8n_coco
param(
    [Parameter(Mandatory = $true)][ValidateSet('yolov8n_coco', 'yolov8n_ft', 'yolo11n_ft')][string]$Ten,
    [ValidatePattern('^[A-Za-z0-9_]*$')][string]$HauTo = ''
)
$ErrorActionPreference = 'Stop'
$Repo = 'D:\LABOTARY\bytetrack'
$PY = 'C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe'
$Bang = @{
    'yolov8n_coco' = @('models/yolov8n.pt', 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36')
    'yolov8n_ft'   = @('models/finetune_phase2/train_bdd5_v1_yolov8n_day_du/weights/best.pt',
                       '5f66776d3202389ca484f3d9e2f107dddb354d21a9f1762d19401d690a4a0d8a')
    'yolo11n_ft'   = @('models/finetune_phase2/train_bdd5_v1_yolo11n_day_du/weights/best.pt',
                       'd7d1c7e86d03c69d913febe79e4cbb6c2e1bcf9f17f034724af4de2ccfba7923')
}
$m = $Bang[$Ten]
$Luot = "r43_$Ten$HauTo"
$Out = "results\phase2\detector_eval\bdd_val_cocovid\$Luot"
$Gs = "results\phase2\detector_eval\giam_sat\$Luot"
foreach ($d in @($Out, $Gs)) { if (Test-Path -LiteralPath (Join-Path $Repo $d)) { throw "$d da ton tai -- khong ghi de" } }
$dang = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
          Where-Object { $_.CommandLine -match 'danh_gia_detector_bdd_val\.py|train_detector\.py' })
if ($dang.Count -gt 0) { throw "dang co tien trinh danh gia / train (PID $($dang.ProcessId -join ', ')) -- khong chay chong" }
$cmd = "`"$PY`" -u src\chay_giam_sat.py chay --thu-muc $Gs --khong-in -- `"$PY`" -u src\danh_gia_detector_bdd_val.py " +
       "--model $Ten=$($m[0]) --sha256 $Ten=$($m[1]) --out $Out"
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmd; CurrentDirectory = $Repo; ProcessStartupInformation = $si }
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') WMI Win32_Process.Create ReturnValue=$($r.ReturnValue) PID_cong_cu=$($r.ProcessId)"
"cwd: $Repo"
"lenh: $cmd"
if ($r.ReturnValue -ne 0) { exit 3 }
