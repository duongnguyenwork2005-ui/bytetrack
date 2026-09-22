#!/usr/bin/env bash
# CHI DE TAI TAO THAM CHIEU TREN PC. TREN NANO: dung `python src/nano_run_cache.py` (chi ghi nano-*, khong dung pc-ref-*).
# Lenh da tao ban tham chieu PC (chay tai goc goi). Tren Nano: doi --tracker-name thanh nano-cv / nano-ekf.
set -e
cd "$(dirname "$0")/.."
# --- NANO-smoke: cache nano_smoke_MVI_39031_f0001-0300 ---
BYTETRACK_BACKEND=np python3 src/run_tracking.py --cache-id nano_smoke_MVI_39031_f0001-0300 --split-name NANO-smoke --tracker-yaml bytetrack.yaml --tracker-name pc-ref-cv --videos MVI_39031 --conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --fps 25 --track-buffer-sec 1.2 --elapsed-mode steps
BYTETRACK_BACKEND=np python3 src/run_tracking.py --cache-id nano_smoke_MVI_39031_f0001-0300 --split-name NANO-smoke --tracker-yaml configs/bytetrack_ekf_ctrv.yaml --tracker-name pc-ref-ekf --videos MVI_39031 --conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --fps 25 --track-buffer-sec 1.2 --elapsed-mode steps
# --- NANO-full: cache yolov8n-coco_conf0.05_all ---
BYTETRACK_BACKEND=np python3 src/run_tracking.py --cache-id yolov8n-coco_conf0.05_all --split-name NANO-full --tracker-yaml bytetrack.yaml --tracker-name pc-ref-cv --videos MVI_39031 --conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --fps 25 --track-buffer-sec 1.2 --elapsed-mode steps
BYTETRACK_BACKEND=np python3 src/run_tracking.py --cache-id yolov8n-coco_conf0.05_all --split-name NANO-full --tracker-yaml configs/bytetrack_ekf_ctrv.yaml --tracker-name pc-ref-ekf --videos MVI_39031 --conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --fps 25 --track-buffer-sec 1.2 --elapsed-mode steps

# Parity (tren Nano, sau khi chay cac lenh tren voi --tracker-name nano-cv / nano-ekf):
python3 src/nano_parity.py --ref data/processed/trackers/NANO-smoke/pc-ref-cv  --test data/processed/trackers/NANO-smoke/nano-cv  --json reference/parity_nano_cv.json
python3 src/nano_parity.py --ref data/processed/trackers/NANO-smoke/pc-ref-ekf --test data/processed/trackers/NANO-smoke/nano-ekf --json reference/parity_nano_ekf.json
python3 src/nano_parity.py --ref data/processed/trackers/NANO-full/pc-ref-cv  --test data/processed/trackers/NANO-full/nano-cv  --json reference/parity_nano_full_cv.json
python3 src/nano_parity.py --ref data/processed/trackers/NANO-full/pc-ref-ekf --test data/processed/trackers/NANO-full/nano-ekf --json reference/parity_nano_full_ekf.json
