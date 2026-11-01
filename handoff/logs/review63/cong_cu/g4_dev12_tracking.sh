#!/usr/bin/env bash
# R63 G4: tracking + TrackEval tren UA-DETRAC dev12 cho hai cache cung detector yolov8n COCO:
#   vuong640 (letterbox 640x640, tai hien Nano v16) va rect640 (384x640 = ung vien C1 / duong PC R53-R61).
# Cau hinh tracker KHOA theo R54 (configs/phase2/r54_tracking/giao_thuc_test40.json): lop 2 3 5 7, max_det 300,
# fps 25, buffer 1,2 s (30 buoc), elapsed_mode skip, c25 / c10, CV bytetrack.yaml / EKF configs/bytetrack_ekf_ctrv.yaml.
# Chay tu goc repo. Ten tracker r63g4-<cache>-<cond>-<loc>; tu choi neu thu muc tracker da ton tai.
set -u
PY=${PY:-C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe}
MAN=results/phase2/manifests/detrac_dev.json
SPLIT=DETRAC-all
VIDEOS=$($PY -c "import json;print(' '.join(json.load(open('$MAN'))['videos']))")
export PYTHONIOENCODING=utf-8
for cache in vuong640 rect640; do
  cid=r63_yolov8n-coco_c0.05_dev12_$cache
  for cond in c25 c10; do
    conf=0.25; [ $cond = c10 ] && conf=0.10
    for loc in cv ekf; do
      yaml=bytetrack.yaml; [ $loc = ekf ] && yaml=configs/bytetrack_ekf_ctrv.yaml
      ten=r63g4-$cache-$cond-$loc
      [ -e data/processed/trackers/$SPLIT/$ten ] && { echo "[DUNG] da ton tai $ten"; exit 2; }
      echo "[$(date +%T)] $ten"
      $PY src/run_tracking.py --cache-id $cid --split-name $SPLIT --tracker-yaml $yaml --tracker-name $ten \
        --manifest $MAN --conf-min $conf --classes 2 3 5 7 --max-det 300 --fps 25 --track-buffer-sec 1.2 \
        --elapsed-mode skip || { echo "[DUNG] tracking $ten"; exit 3; }
      $PY handoff/logs/review63/cong_cu/g4_cham.py --split-name $SPLIT --tracker $ten --videos $VIDEOS \
        --dataset detrac 2>&1 | tail -2 || { echo "[DUNG] trackeval $ten"; exit 4; }
    done
  done
done
echo "[XONG]"
