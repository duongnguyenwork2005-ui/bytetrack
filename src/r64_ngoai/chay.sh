#!/usr/bin/env bash
# R64: chay mot dataset theo thu tu GPT (cache 3 detector -> CV/EKF/UKF -> PF500 -> tong hop), dung o loi/het ngan sach.
#   HET_HAN=2026-11-02T07:00:00+07:00 bash src/r64_ngoai/chay.sh <svmot|roundabout> <cache|track|pf|tong-hop|tat_ca> [--smoke]
# Ma thoat cua wrapper: 0 xong, 10 het ngan sach (PARTIAL -> dung chuoi, chay lai se tiep tuc), khac 0/10 -> loi -> dung.
# Cache: don vi co marker phai qua cong hoan tat/noi dung moi duoc bo qua; track: run da ton tai -> DUNG (khong ghi de).
set -uo pipefail
PY=${PY:-C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe}
export PYTHONIOENCODING=utf-8
DS=${1:?ds}; BUOC=${2:?buoc}; SM=${3:-}
HH=(); [ -n "${HET_HAN:-}" ] && HH=(--het-han "$HET_HAN")
W="$PY -B src/r64_ngoai/danh_gia_ngoai.py"
DET=${DET_CHON:-"yolov8n_coco yolov8n_mixed5 yolo11n_mixed5"}   # DET_CHON: chay song song tung detector (chi PF, CPU)
goi() { "$@"; local rc=$?; [ $rc -eq 10 ] && { echo "[PARTIAL] $*"; exit 10; }; [ $rc -ne 0 ] && { echo "[DUNG rc=$rc] $*"; exit $rc; }; return 0; }
lam_cache() { for d in $DET; do echo "[$(date +%T)] cache $DS $d"; goi $W cache $DS $d $SM "${HH[@]}"; done; }
lam_track() {  # lam_track <bo loc...>; kiem-run (sua 02/11 theo review cuoi GPT): 0 xong -> bo qua (da xac thuc cong_r64,
  # tap video, hash GT, du seed), 1 chua -> chay, khac -> don vi do/hong -> DUNG, khong ghi de
  for d in $DET; do for c in c25 c10; do for l in "$@"; do
    $W kiem-run $DS $d $c $l $SM; rc=$?
    case $rc in
      0) echo "[bo qua] $d $c $l da xong + qua kiem"; continue ;;
      1) ;;
      *) echo "[DUNG] $d $c $l: don vi do/hong (kiem-run rc=$rc), can xu ly tay"; exit 2 ;;
    esac
    echo "[$(date +%T)] track $DS $d $c $l"; goi $W track $DS $d $c $l $SM "${HH[@]}"
  done; done; done
}
case $BUOC in
cache) lam_cache ;;
track) lam_track cv ekf ukf ;;
pf) lam_track pf500 ;;
tong-hop) goi $W tong-hop $DS $SM ;;
tat_ca) lam_cache; lam_track cv ekf ukf; lam_track pf500; goi $W tong-hop $DS $SM ;;
*) echo "buoc sai"; exit 2 ;;
esac
echo "[XONG] $DS $BUOC $SM"
