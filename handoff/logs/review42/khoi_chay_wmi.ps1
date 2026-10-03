# Vong 42: khoi chay luot training dai qua src/chay_giam_sat.py trong mot tien trinh DOC LAP voi phien Claude.
# Tao bang WMI Win32_Process.Create: tien trinh cha la WmiPrvSE.exe (ngoai cay tien trinh claude.exe / VSCode), cung
# session nguoi dung, moi truong nguoi dung chuan (nhu mot terminal moi), cua so an. KHONG phai tac vu nen cua Claude
# (bi cat sau ~2 h). Dung tien trinh: Task Manager (end process tree) -- khong co trong lenh nay.
#   powershell -NoProfile -ExecutionPolicy Bypass -File handoff\logs\review42\khoi_chay_wmi.ps1 -Model yolov8n -GiamSat yolov8n_lan0
#   powershell -NoProfile -ExecutionPolicy Bypass -File handoff\logs\review42\khoi_chay_wmi.ps1 -Model yolov8n -GiamSat yolov8n_tiep_tuc_1 -TiepTuc
# Truoc -TiepTuc phai chay va dat: python src\train_detector.py --model <M> --dry-run --tiep-tuc
param(
    [Parameter(Mandatory = $true)][ValidateSet('yolov8n', 'yolo11n')][string]$Model,
    [Parameter(Mandatory = $true)][ValidatePattern('^[A-Za-z0-9_]+$')][string]$GiamSat,
    [switch]$TiepTuc
)
$ErrorActionPreference = 'Stop'
$Repo = 'D:\LABOTARY\bytetrack'
$PY = 'C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe'
$Dir = "results\phase2\detector_train\giam_sat\$GiamSat"
if (Test-Path -LiteralPath (Join-Path $Repo $Dir)) { throw "$Dir da ton tai -- khong ghi de ket qua giam sat cu" }
$dang = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train_detector\.py' })
if ($dang.Count -gt 0) { throw "dang co tien trinh train_detector.py (PID $($dang.ProcessId -join ', ')) -- khong chay chong" }
$cmd = "`"$PY`" -u src\chay_giam_sat.py chay --thu-muc $Dir --khong-in -- `"$PY`" -u src\train_detector.py --model $Model --che-do day-du --xac-nhan-train-dai"
if ($TiepTuc) { $cmd += ' --tiep-tuc' }
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmd; CurrentDirectory = $Repo; ProcessStartupInformation = $si }
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') WMI Win32_Process.Create ReturnValue=$($r.ReturnValue) PID_cong_cu=$($r.ProcessId)"
"cwd: $Repo"
"lenh: $cmd"
if ($r.ReturnValue -ne 0) { exit 3 }
