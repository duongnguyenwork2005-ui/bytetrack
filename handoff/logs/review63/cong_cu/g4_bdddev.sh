#!/usr/bin/env bash
# R63 G4 (Q3): BDD dev 139 video (split BDDDEV-train, g4_bdd_dev_split.py). Buoc:
#   cache  : det_cache yolov8n COCO conf 0.05 cho vuong640 (--no-rect) va rect640, dataset bdd100k (box_track_20 native)
#   track  : tracking + cham (g4_cham.py, evaluator mot_crowd) cau hinh KHOA theo R61 (configs/phase2/r61_bdd/
#            giao_thuc.json): lop 1 2 3 5 7, max_det 300, fps 5, buffer 1,2 s (6 buoc), elapsed_mode steps, c25/c10, CV/EKF.
# Chay tu goc repo:  bash handoff/logs/review63/cong_cu/g4_bdddev.sh cache|track
set -u
PY=${PY:-C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe}
SPLIT=BDDDEV-train
VIDEOS=$(tail -n +2 data/processed/seqmaps/$SPLIT.txt | tr -d '\r' | tr '\n' ' ')
export PYTHONIOENCODING=utf-8
[ $(echo $VIDEOS | wc -w) -eq 139 ] || { echo "[LOI] seqmap khong du 139 video"; exit 2; }
case "${1:-}" in
cache)
  for cache in vuong640 rect640; do
    F=""; [ $cache = vuong640 ] && F="--no-rect"
    cid=r63_yolov8n-coco_c0.05_bdddev_$cache
    [ -e data/interim/det_cache/$cid ] && { echo "[DUNG] da ton tai $cid"; exit 2; }
    echo "[$(date +%T)] cache $cid"
    $PY -u src/det_cache.py --dataset bdd100k --part train --videos $VIDEOS --cache-id $cid --model models/yolov8n.pt \
      --conf 0.05 --device 0 $F || { echo "[DUNG] cache $cid"; exit 3; }
  done ;;
track)
  for cache in vuong640 rect640; do
    cid=r63_yolov8n-coco_c0.05_bdddev_$cache
    for cond in c25 c10; do
      conf=0.25; [ $cond = c10 ] && conf=0.10
      for loc in cv ekf; do
        yaml=bytetrack.yaml; [ $loc = ekf ] && yaml=configs/bytetrack_ekf_ctrv.yaml
        ten=r63g4b-$cache-$cond-$loc
        [ -e data/processed/trackers/$SPLIT/$ten ] && { echo "[DUNG] da ton tai $ten"; exit 2; }
        echo "[$(date +%T)] $ten"
        $PY src/run_tracking.py --cache-id $cid --split-name $SPLIT --tracker-yaml $yaml --tracker-name $ten \
          --videos $VIDEOS --conf-min $conf --classes 1 2 3 5 7 --max-det 300 --fps 5 --track-buffer-sec 1.2 \
          --elapsed-mode steps > /dev/null || { echo "[DUNG] tracking $ten"; exit 3; }
        $PY handoff/logs/review63/cong_cu/g4_cham.py --split-name $SPLIT --tracker $ten --videos $VIDEOS \
          --dataset bdd100k 2>&1 | tail -1
      done
    done
  done ;;
*) echo "dung: $0 cache|track"; exit 2 ;;
esac
echo "[XONG]"
