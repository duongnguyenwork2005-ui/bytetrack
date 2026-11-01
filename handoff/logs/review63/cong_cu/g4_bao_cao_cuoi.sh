#!/usr/bin/env bash
# R63 G4 — BAO CAO CUOI (sau khi chot C1 theo GPT 01/11; CHI bao cao, khong doi quyet dinh, khong tinh chinh):
# doi chung vuong640 (letterbox 640x640 = Nano v16) vs ung vien rect640 (384x640 = C1), cung yolov8n.pt COCO.
#   test40 : UA-DETRAC test 40 video, giao thuc KHOA R54 (configs/phase2/r54_tracking/giao_thuc_test40.json):
#            lop 2 3 5 7, max_det 300, fps 25, buffer 1,2 s, elapsed skip, c25/c10, CV/EKF, evaluator mot.
#   bddval : BDD COCO-video val 200 video, giao thuc KHOA R61 (configs/phase2/r61_bdd/giao_thuc.json):
#            lop 1 2 3 5 7, fps 5, buffer 1,2 s (6 buoc), elapsed steps, c25/c10, CV/EKF, evaluator mot_crowd.
# Buoc (moi buoc mot lan goi, ~<= 1,5 h GPU, tu choi neu dau ra da ton tai):
#   cache40 | cachebdd | track40 | trackbdd | tang40 | tangbdd | ap40 | apbdd
# Chay tu goc repo:  bash handoff/logs/review63/cong_cu/g4_bao_cao_cuoi.sh <buoc>
set -uo pipefail   # sua 02/11 (review GPT A_sua): loi g4_cham qua pipe tail khong con bi che
PY=${PY:-C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe}
export PYTHONIOENCODING=utf-8
W=models/yolov8n.pt; SHA=f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36
MAN40=results/phase2/manifests/detrac_test.json
MANBDD=results/phase2/manifests/bdd100k_transfer_cocovid_val.json
NHAN_BDD=data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json
RA=handoff/logs/review63/bao_cao_cuoi
mkdir -p $RA
V40=$($PY -c "import json;print(' '.join(json.load(open('$MAN40'))['videos']))")
VBDD=$(tail -n +2 data/processed/seqmaps/BDDCOCOVID-val.txt | tr -d '\r' | tr '\n' ' ')
[ $(echo $V40 | wc -w) -eq 40 ] || { echo "[LOI] manifest test40 khong du 40 video"; exit 2; }
[ $(echo $VBDD | wc -w) -eq 200 ] || { echo "[LOI] seqmap BDD val khong du 200 video"; exit 2; }
echo "$W $SHA" | $PY -c "import hashlib,sys;w,s=sys.stdin.read().split();h=hashlib.sha256(open(w,'rb').read()).hexdigest();sys.exit(0 if h==s else print('[LOI] sha256 checkpoint',h) or 2)" || exit 2

track() {  # track <split> <tien_to> <mien> <videos...> -- cac co tracker rieng cua giao thuc o bien CO
  local split=$1 tt=$2 mien=$3; shift 3
  for cache in vuong640 rect640; do
    cid=r63_yolov8n-coco_c0.05_${mien}_$cache
    for cond in c25 c10; do
      conf=0.25; [ $cond = c10 ] && conf=0.10
      for loc in cv ekf; do
        yaml=bytetrack.yaml; [ $loc = ekf ] && yaml=configs/bytetrack_ekf_ctrv.yaml
        ten=$tt-$cache-$cond-$loc
        [ -e data/processed/trackers/$split/$ten ] && { echo "[DUNG] da ton tai $ten"; exit 2; }
        echo "[$(date +%T)] $ten"
        $PY src/run_tracking.py --cache-id $cid --split-name $split --tracker-yaml $yaml --tracker-name $ten \
          --videos "$@" --conf-min $conf $CO > /dev/null || { echo "[DUNG] tracking $ten"; exit 3; }
        $PY handoff/logs/review63/cong_cu/g4_cham.py --split-name $split --tracker $ten --videos "$@" \
          --dataset $DS 2>&1 | tail -1 || { echo "[DUNG] cham $ten"; exit 4; }
      done
    done
  done
}

case "${1:-}" in
cache40|cachebdd)
  for cache in vuong640 rect640; do
    F=""; [ $cache = vuong640 ] && F="--no-rect"
    if [ $1 = cache40 ]; then
      cid=r63_yolov8n-coco_c0.05_test40_$cache; DSA="--dataset detrac --part test --manifest $MAN40"
    else
      cid=r63_yolov8n-coco_c0.05_bddval_$cache
      DSA="--dataset bdd100k_coco --labels-json $NHAN_BDD --part val --manifest $MANBDD"
    fi
    [ -e data/interim/det_cache/$cid ] && { echo "[DUNG] da ton tai $cid"; exit 2; }
    echo "[$(date +%T)] cache $cid"
    $PY -u src/det_cache.py $DSA --cache-id $cid --model $W --conf 0.05 --device 0 $F || { echo "[DUNG] cache $cid"; exit 3; }
  done ;;
track40)
  DS=detrac CO="--classes 2 3 5 7 --max-det 300 --fps 25 --track-buffer-sec 1.2 --elapsed-mode skip"
  track DETRAC-test r63bc test40 $V40
  $PY handoff/logs/review63/cong_cu/g4_tong_hop_tracking.py --mien test40 --ra $RA/tracking_test40_so_cap.json ;;
trackbdd)
  DS=bdd100k_coco CO="--classes 1 2 3 5 7 --max-det 300 --fps 5 --track-buffer-sec 1.2 --elapsed-mode steps"
  track BDDCOCOVID-val r63bcb bddval $VBDD
  $PY handoff/logs/review63/cong_cu/g4_tong_hop_tracking.py --mien bddval --ra $RA/tracking_bddval_so_cap.json ;;
tang40)
  $PY handoff/logs/review63/chat_luong/phan_tang_kich_thuoc.py --cache-id r63_yolov8n-coco_c0.05_test40_rect640 \
    --so-voi r63_yolov8n-coco_c0.05_test40_vuong640 --manifest $MAN40 --split DETRAC-test --ra $RA/tang_test40_c25 ;;
tangbdd)
  # GT BDD COCO-video val: visibility = 1,0 cho 100 % hang xe (kiem 01/11: 382.903/382.903) -> KHONG co tang che;
  # chi bao tang kich thuoc. Precision chua tru vung bo qua lop 13 (chi recall dung cho doc cong).
  $PY handoff/logs/review63/chat_luong/phan_tang_kich_thuoc.py --cache-id r63_yolov8n-coco_c0.05_bddval_rect640 \
    --so-voi r63_yolov8n-coco_c0.05_bddval_vuong640 --videos $VBDD --split BDDCOCOVID-val --classes 1,2,3,5,7 \
    --nguon bdd --ra $RA/tang_bddval_c25 ;;
ap40|apbdd)
  for hd in 640x640 384x640; do
    if [ $1 = ap40 ]; then
      out=results/phase2/r63_g4/ap_test40_$hd; WR=src/danh_gia_detector_detrac_test.py
    else
      out=results/phase2/r63_g4/ap_bddval_$hd; WR=src/danh_gia_detector_bdd_val.py
    fi
    [ -e $out ] && { echo "[DUNG] da ton tai $out"; exit 2; }
    echo "== $hd $(date +%T)"
    $PY -u $WR --model yolov8n_coco=$W --sha256 yolov8n_coco=$SHA --out $out --hinh-dang $hd || { echo "[DUNG] ap $hd"; exit 3; }
  done ;;
*) echo "dung: $0 cache40|cachebdd|track40|trackbdd|tang40|tangbdd|ap40|apbdd"; exit 2 ;;
esac
echo "[XONG] $1"
