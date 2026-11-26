$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
$root = 'handoff/logs/review85_nano_opt'
$manifest = 'results/phase2/manifests/detrac_dev.json'
$videos = @((Get-Content $manifest -Raw | ConvertFrom-Json).videos)
if ($videos.Count -ne 12) { throw "Expected 12 dev videos; got $($videos.Count)" }
$cache = 'r85_yolov8n-coco_c0.05_dev12_rect576'
if (-not (Test-Path "data/interim/det_cache/$cache/meta.json")) { throw 'C2 cache not ready' }
function Run-Step([string]$name, [string[]]$arguments) {
    $log = "$root/$name.log"
    if (Test-Path $log) { throw "Existing log: $log" }
    Write-Host "[$(Get-Date -Format HH:mm:ss)] $name"
    & $py @arguments *> $log
    if ($LASTEXITCODE -ne 0) { throw "Failed $name (exit $LASTEXITCODE); see $log" }
    Get-Content $log -Tail 2
}
Run-Step '02_recall_dev12' @('handoff/logs/review63/chat_luong/phan_tang_kich_thuoc.py',
    '--cache-id',$cache,'--so-voi','r63_yolov8n-coco_c0.05_dev12_rect640',
    '--manifest',$manifest,'--ra',"$root/recall_dev12_c2_vs_c1")
foreach ($filter in @('cv','ekf')) {
    $yaml = if ($filter -eq 'cv') { 'bytetrack.yaml' } else { 'configs/bytetrack_ekf_ctrv.yaml' }
    $name = "r85-c2-dev12-c25-$filter"
    Run-Step "03_track_dev12_$filter" @('src/run_tracking.py','--cache-id',$cache,
        '--split-name','DETRAC-all','--tracker-yaml',$yaml,'--tracker-name',$name,
        '--manifest',$manifest,'--conf-min','0.25','--classes','2','3','5','7',
        '--max-det','300','--fps','25','--track-buffer-sec','1.2','--elapsed-mode','skip')
    $scoreArgs = @('handoff/logs/review63/cong_cu/g4_cham.py',
        '--split-name','DETRAC-all','--tracker',$name,'--videos') + $videos + @('--dataset','detrac')
    Run-Step "04_score_dev12_$filter" $scoreArgs
}
