# R52B: khoi chay luot training dai MIXED5 qua src/chay_giam_sat.py trong mot tien trinh DOC LAP voi phien Claude.
# Ban sao co chu dich cua handoff/logs/review42/khoi_chay_wmi.ps1 (vong 42, da dung that); CHI them --config / --project /
# --ket-qua-dir cua mixed5 va doi thu muc giam sat. CHUA CHAY trong R52B -- chi dung sau khi nguoi dung duyet luot dai.
# Tao bang WMI Win32_Process.Create: tien trinh cha la WmiPrvSE.exe (ngoai cay tien trinh claude.exe / VSCode), cung
# session nguoi dung, moi truong nguoi dung chuan, cua so an. Dung tien trinh: Task Manager (end process tree).
#   powershell -NoProfile -ExecutionPolicy Bypass -File handoff\logs\review52\khoi_chay_wmi_mixed5.ps1 -Model yolov8n -GiamSat yolov8n_lan0
#   powershell -NoProfile -ExecutionPolicy Bypass -File handoff\logs\review52\khoi_chay_wmi_mixed5.ps1 -Model yolov8n -GiamSat yolov8n_tiep_tuc_1 -TiepTuc
# Truoc -TiepTuc phai chay va dat:
#   python src\train_detector.py --config configs\phase2\detector\train_mixed5.yaml --model <M> --dry-run --tiep-tuc ^
#       --project models\finetune_mixed5 --ket-qua-dir results\phase2\mixed5\detector_train
param(
    [Parameter(Mandatory = $true)][ValidateSet('yolov8n', 'yolo11n')][string]$Model,
    [Parameter(Mandatory = $true)][ValidatePattern('^[A-Za-z0-9_]+$')][string]$GiamSat,
    [switch]$TiepTuc
)
$ErrorActionPreference = 'Stop'
$Repo = 'D:\LABOTARY\bytetrack'
$PY = 'C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe'
$Dir = "results\phase2\mixed5\detector_train\giam_sat\$GiamSat"
if (Test-Path -LiteralPath (Join-Path $Repo $Dir)) { throw "$Dir da ton tai -- khong ghi de ket qua giam sat cu" }
$dang = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train_detector\.py' })
if ($dang.Count -gt 0) { throw "dang co tien trinh train_detector.py (PID $($dang.ProcessId -join ', ')) -- khong chay chong" }
$cmd = "`"$PY`" -u src\chay_giam_sat.py chay --thu-muc $Dir --khong-in -- `"$PY`" -u src\train_detector.py " +
       "--config configs\phase2\detector\train_mixed5.yaml --model $Model --che-do day-du --xac-nhan-train-dai " +
       "--project models\finetune_mixed5 --ket-qua-dir results\phase2\mixed5\detector_train"
if ($TiepTuc) { $cmd += ' --tiep-tuc' }
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmd; CurrentDirectory = $Repo; ProcessStartupInformation = $si }
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') WMI Win32_Process.Create ReturnValue=$($r.ReturnValue) PID_cong_cu=$($r.ProcessId)"
"cwd: $Repo"
"lenh: $cmd"
if ($r.ReturnValue -ne 0) { exit 3 }
