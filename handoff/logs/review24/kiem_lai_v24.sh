#!/bin/bash
# Vong 24 (review 1d5ffaa, N23-F1): kiem lai HAI video render hau ky vong 23 tren Nano bang bo kiem MOI (goi v10, 9b1805c),
# Python HE THONG (/usr/bin/python3 3.6.9, OpenCV 3.2.0). KHONG chay lai tracking, KHONG build engine, KHONG sua video cu.
#   A. hash video / MOT / lop so voi bang chung vong 23 (handoff/logs/review23/render_v23/nguon_sha256.txt + *.mp4.json)
#   B. bo kiem moi, CUNG tham so vong 23 (--buoc-lop 5 mac dinh) -> <luot>_kiem_v24.json/.log
#   C. bo sung: kiem ten lop TREN MOI box (--buoc-lop 1) -> <luot>_kiem_v24_lop_moi_box.json/.log
#   D. doi chieu refactor render: render lai bang render_mot_video v10 ra FILE MOI (render_v10_doi_chieu/), so sha256 voi
#      video vong 23 (video cu giu nguyen).
# Chay:  [GOI=v11] setsid nohup bash kiem_lai_v24.sh > kiem_lai_v24.out 2>&1 < /dev/null &   (xong -> $OUT/XONG co exit tung buoc)
set -u
GOI=${GOI:-v10}                                   # goi ma: v10 (9b1805c) hoac v11 (ma cuoi, chot phien ban OpenCV)
V10=$HOME/nano_run/$GOI/nano_transfer
V9R=$HOME/nano_run/v9/nano_transfer/results_nano/render_v23
V8D=$HOME/nano_run/v8/nano_transfer/results_nano/dai_v23
IMG=$HOME/nano_run/data_full/MVI_39031
OUT=$V10/results_nano/kiem_lai_v24
PY=/usr/bin/python3
mkdir -p "$OUT/render_v10_doi_chieu"
cd "$V10" || exit 2
{
  echo "# kiem lai video render vong 23 bang bo kiem vong 24 -- $(date -Is) | host $(hostname) | boot $(uptime -s)"
  echo "# goi: $($PY -c 'import json;n=json.load(open("PACKAGE_INFO.json"))["nguon"];print(n["HEAD_short"], n["branch"], "tracked_da_sua:", n["tracked_da_sua_ngoai_file_rieng"])')"
  echo "# python: $($PY --version 2>&1) | $($PY -c 'import cv2,numpy;print("cv2", cv2.__version__, "numpy", numpy.__version__)')"
  echo "# ma:"
  sha256sum src/kiem_video_render.py src/render_mot_video.py
  echo "## A. hash dau vao (ky vong: video CV 5f55fcb4..., EKF c7d40019...; MOT/lop theo *.mp4.json vong 23)"
  sha256sum "$V9R/luot01_cv_f0001-0300.mp4" "$V9R/luot02_ekf_f0001-0300.mp4" \
            "$V8D/luot01_cv/data/MVI_39031.txt" "$V8D/luot01_cv/data_cls/MVI_39031.csv" \
            "$V8D/luot02_ekf/data/MVI_39031.txt" "$V8D/luot02_ekf/data_cls/MVI_39031.csv"
  stat -c '%n %s B mtime %y' "$V9R/luot01_cv_f0001-0300.mp4" "$V9R/luot02_ekf_f0001-0300.mp4"
  echo "# anh: $(ls "$IMG" | wc -l) file trong $IMG; 300 anh dau: $(ls "$IMG" | sort | head -300 | tail -1)"
} > "$OUT/00_hash_va_phien_ban.txt" 2>&1

: > "$OUT/XONG.tmp"
for L in luot01_cv luot02_ekf; do
  VID="$V9R/${L}_f0001-0300.mp4"; MOT="$V8D/$L/data/MVI_39031.txt"; CLS="$V8D/$L/data_cls/MVI_39031.csv"
  for BIEN in "kiem_v24:5" "kiem_v24_lop_moi_box:1"; do
    TEN=${BIEN%%:*}; BUOC=${BIEN##*:}
    CMD="$PY src/kiem_video_render.py --video $VID --images $IMG --mot $MOT --cls $CLS --fps 25 --max-frames 300 --buoc-lop $BUOC --json $OUT/${L}_${TEN}.json"
    { echo "# $(date -Is) | cwd $V10"; echo "\$ $CMD"; } > "$OUT/${L}_${TEN}.log"
    t0=$(date +%s%N)
    $CMD >> "$OUT/${L}_${TEN}.log" 2>&1
    rc=$?
    t1=$(date +%s%N)
    echo "# exit=$rc wall_ms=$(( (t1 - t0) / 1000000 ))" >> "$OUT/${L}_${TEN}.log"
    echo "$L $TEN exit=$rc" >> "$OUT/XONG.tmp"
  done
  # D. doi chieu refactor render (file moi, khong ghi de video vong 23)
  RV="$OUT/render_v10_doi_chieu/${L}_f0001-0300.mp4"
  CMD="$PY src/render_mot_video.py --images $IMG --mot $MOT --cls $CLS --fps 25 --max-frames 300 --out $RV"
  { echo "\$ $CMD"; $CMD; echo "# exit=$?"; sha256sum "$RV" "$VID"; } > "$OUT/render_v10_doi_chieu/${L}.log" 2>&1
  if [ "$(sha256sum < "$RV" | cut -c1-64)" = "$(sha256sum < "$VID" | cut -c1-64)" ]; then
    echo "$L render_v10 == video vong 23 (sha256 giong)" >> "$OUT/XONG.tmp"
  else
    echo "$L render_v10 KHAC video vong 23" >> "$OUT/XONG.tmp"
  fi
done
mv "$OUT/XONG.tmp" "$OUT/XONG"
