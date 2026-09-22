#!/usr/bin/env bash
# Chuoi END-TO-END tren Jetson Nano: engine (build TREN Nano) -> detector TensorRT (Python 3.6 he thong) -> det cache
# -> tracker CV/EKF (venv >= 3.8, backend NumPy) -> so detection voi cache PC -> video minh hoa.
# Chay tu GOC GOI (thu muc nano_transfer). Yeu cau: da qua buoc 1-4 cua NANO_START_HERE.md (parity tracker DAT).
#
#   bash scripts/nano/03_e2e_nano.sh <python_venv38> [engine] [tag]
#   vd: bash scripts/nano/03_e2e_nano.sh ~/nano_env/bin/python yolov8n_fp16.engine nano-trt
#
# Ket qua: results_nano/<tag>/{engine.build.log, detector_timing.json, det_compare.json, run_meta.json...},
#          data/interim/det_cache/<tag>_MVI_39031_f0001-0300/, data/processed/trackers/NANO-e2e/<tag>-{cv,ekf}/
set -u
PYV=${1:?python cua venv >= 3.8}
ENGINE=${2:-yolov8n_fp16.engine}
TAG=${3:-nano-trt}
ONNX=models/export/yolov8n_640_op12.onnx
IMG=data/smoke/MVI_39031_f0001-0300
REG=data/smoke/MVI_39031/ignored_regions.txt
CID="${TAG//-/_}_MVI_39031_f0001-0300"
OUT=results_nano/$TAG
mkdir -p "$OUT"
LOG=$OUT/e2e.log
exec > >(tee -a "$LOG") 2>&1
echo "# 03_e2e_nano $(date -Is) tag=$TAG engine=$ENGINE venv=$PYV"
echo "# he thong: $(uname -m) $(cat /etc/nv_tegra_release 2>/dev/null | head -1)"; python3 -VV; "$PYV" -VV
ONNX_SHA=$(sha256sum "$ONNX" | cut -d' ' -f1); echo "# onnx sha256 $ONNX_SHA"

# 1. engine (chi build neu chua co)
if [ ! -f "$ENGINE" ]; then
  echo "== 1. build engine FP16 tren Nano (trtexec)"; bash scripts/nano/01_build_engine.sh "$ONNX" "$ENGINE" --fp16 || { echo "[LOI] build engine that bai"; exit 1; }
  cp "${ENGINE}.build.log" "$OUT/" 2>/dev/null || true
else
  echo "== 1. dung engine co san: $ENGINE ($(sha256sum "$ENGINE" | cut -d' ' -f1))"
fi
echo "== 1b. benchmark engine (latency GPU detector, KHONG phai pipeline)"; bash scripts/nano/02_benchmark_engine.sh "$ENGINE" 2>&1 | tail -15 | tee "$OUT/trtexec_bench.txt"

# 2. detector TensorRT -> det cache (Python 3.6 he thong: tensorrt + numpy + cv2 (+ torch lam bo dem))
echo "== 2. detector TensorRT -> cache $CID"
python3 -c "import tensorrt, numpy, cv2; print('tensorrt', tensorrt.__version__, 'numpy', numpy.__version__, 'cv2', cv2.__version__)" || { echo "[LOI] python3 he thong thieu tensorrt/numpy/cv2"; exit 1; }
rm -rf "data/interim/det_cache/$CID"
python3 src/nano_detector.py --engine "$ENGINE" --onnx-sha256 "$ONNX_SHA" --images "$IMG" --fps 25 --video-name MVI_39031 \
    --cache-id "$CID" --regions "$REG" --timing-json "$OUT/detector_timing.json" || { echo "[LOI] detector that bai"; exit 1; }

# 3. so detection voi cache PC (mo ta, khong dat/khong dat)
echo "== 3. det_compare: cache TRT vs cache PC (dieu kien vao tracker c25) va (tho conf 0.05)"
"$PYV" src/nano_det_compare.py --a nano_smoke_MVI_39031_f0001-0300 --b "$CID" --video MVI_39031 --conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --json "$OUT/det_compare_c25.json"
"$PYV" src/nano_det_compare.py --a nano_smoke_MVI_39031_f0001-0300 --b "$CID" --video MVI_39031 --json "$OUT/det_compare_raw.json"

# 4. tracker CV/EKF tu cache TRT (backend np, venv)
echo "== 4. tracker tu cache TRT"
for T in cv ekf; do
  YAML=bytetrack.yaml; [ "$T" = ekf ] && YAML=configs/bytetrack_ekf_ctrv.yaml
  BYTETRACK_BACKEND=np "$PYV" src/run_tracking.py --cache-id "$CID" --split-name NANO-e2e --tracker-yaml "$YAML" --tracker-name "$TAG-$T" \
      --videos MVI_39031 --conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --fps 25 --track-buffer-sec 1.2 --elapsed-mode steps || { echo "[LOI] tracker $T that bai"; exit 1; }
  cp "data/processed/trackers/NANO-e2e/$TAG-$T/run_meta.json" "$OUT/run_meta_$T.json"
  # so voi tracking tham chieu PC (cache PC): MO TA -- khac detector nen KHONG ky vong parity
  "$PYV" src/nano_parity.py --ref data/processed/trackers/NANO-smoke/pc-ref-$T --test "data/processed/trackers/NANO-e2e/$TAG-$T" --json "$OUT/so_tracking_vs_pc_ref_$T.json" || true
done

# 5. video minh hoa
echo "== 5. video minh hoa"
"$PYV" src/render_mot_video.py --images "$IMG" --mot "data/processed/trackers/NANO-e2e/$TAG-cv/data/MVI_39031.txt" \
    --cls "data/processed/trackers/NANO-e2e/$TAG-cv/data_cls/MVI_39031.csv" --fps 25 --out "$OUT/$TAG-cv.mp4" || echo "(render bo qua: can opencv trong venv)"
echo "== XONG: xem $OUT/ (e2e.log, detector_timing.json, det_compare_*.json, run_meta_*.json, so_tracking_vs_pc_ref_*.json)"
