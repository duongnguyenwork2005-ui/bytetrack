# NOT RUN in R85: C2 failed the dev12 quality gate. Retained as a prepared command only.
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
$root = 'handoff/logs/review85_nano_opt'
$seqmap = 'data/processed/seqmaps/BDDDEV-train.txt'
$videos = @(Get-Content $seqmap | Select-Object -Skip 1 | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($videos.Count -ne 139) { throw "Expected 139 BDD dev videos; got $($videos.Count)" }
$cache = 'r85_yolov8n-coco_c0.05_bdddev_rect576'
function Run-Step([string]$name, [string[]]$arguments) {
    $log = "$root/$name.log"
    if (Test-Path $log) { throw "Existing log: $log" }
    Write-Host "[$(Get-Date -Format HH:mm:ss)] $name"
    & $py @arguments *> $log
    if ($LASTEXITCODE -ne 0) { throw "Failed $name (exit $LASTEXITCODE); see $log" }
    Get-Content $log -Tail 2
}
$cacheArgs = @('-u','src/det_cache.py','--dataset','bdd100k','--part','train','--videos') + $videos + @(
    '--cache-id',$cache,'--model','models/yolov8n.pt','--conf','0.05','--device','0','--imgsz','576')
Run-Step '05_cache_bdddev' $cacheArgs
$recallArgs = @('handoff/logs/review63/chat_luong/phan_tang_kich_thuoc.py',
    '--cache-id',$cache,'--so-voi','r63_yolov8n-coco_c0.05_bdddev_rect640',
    '--videos') + $videos + @('--split','BDDDEV-train','--classes','1,2,3,5,7','--nguon','bdd',
    '--ra',"$root/recall_bdddev_c2_vs_c1")
Run-Step '06_recall_bdddev' $recallArgs
foreach ($filter in @('cv','ekf')) {
    $yaml = if ($filter -eq 'cv') { 'bytetrack.yaml' } else { 'configs/bytetrack_ekf_ctrv.yaml' }
    $name = "r85-c2-bdddev-c25-$filter"
    $trackArgs = @('src/run_tracking.py','--cache-id',$cache,'--split-name','BDDDEV-train',
        '--tracker-yaml',$yaml,'--tracker-name',$name,'--videos') + $videos + @(
        '--conf-min','0.25','--classes','1','2','3','5','7','--max-det','300',
        '--fps','5','--track-buffer-sec','1.2','--elapsed-mode','steps')
    Run-Step "07_track_bdddev_$filter" $trackArgs
    $scoreArgs = @('handoff/logs/review63/cong_cu/g4_cham.py',
        '--split-name','BDDDEV-train','--tracker',$name,'--videos') + $videos + @('--dataset','bdd100k')
    Run-Step "08_score_bdddev_$filter" $scoreArgs
}
