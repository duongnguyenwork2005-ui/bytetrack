#!/usr/bin/env bash
# R66: chay mot dataset theo thu tu GPT (cache 3 detector -> CV/EKF/UKF -> PF500 -> tong hop), dung o loi/het ngan sach.
#   HET_HAN=2026-11-02T18:00:00+07:00 bash src/r66_ngoai/chay.sh <ds> <cache|track|tat_ca|tong-hop> [--smoke]
# Ma thoat wrapper: 0 xong, 10 het ngan sach (PARTIAL -> dung chuoi), khac -> dung. kiem-run: 0 xong, 1 chua, khac -> dung.
set -uo pipefail
PY=${PY:-C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe}
export PYTHONIOENCODING=utf-8
DS=${1:?ds}; BUOC=${2:?buoc}; SM=${3:-}
HH=(); [ -n "${HET_HAN:-}" ] && HH=(--het-han "$HET_HAN")
W="$PY -B src/r66_ngoai/danh_gia_r66.py"
DET=${DET_CHON:-"yolov8n_coco yolov8n_mixed5 yolo11n_mixed5"}
goi() { "$@"; local rc=$?; [ $rc -eq 10 ] && { echo "[PARTIAL] $*"; exit 10; }; [ $rc -ne 0 ] && { echo "[DUNG rc=$rc] $*"; exit $rc; }; return 0; }
lam_cache() { for d in $DET; do echo "[$(date +%T)] cache $DS $d"; goi $W cache $DS $d $SM "${HH[@]}"; done; }
lam_track() {
  for d in $DET; do for c in c25 c10; do for l in "$@"; do
    $W kiem-run $DS $d $c $l $SM >/dev/null; rc=$?
    case $rc in
      0) echo "[bo qua] $d $c $l da xong + qua kiem"; continue ;;
      1) ;;
      *) echo "[DUNG] $d $c $l: don vi do/hong (kiem-run rc=$rc)"; exit 2 ;;
    esac
    echo "[$(date +%T)] track $DS $d $c $l"; goi $W track $DS $d $c $l $SM "${HH[@]}"
  done; done; done
}
case $BUOC in
cache) lam_cache ;;
track) lam_track cv ekf ukf pf500 ;;
tong-hop) goi $W tong-hop $DS $SM ;;
tat_ca) lam_cache; lam_track cv ekf ukf pf500; goi $W tong-hop $DS $SM ;;
*) echo "buoc?"; exit 2 ;;
esac
echo "[$(date +%T)] xong $DS $BUOC $SM"
