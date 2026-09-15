# Đầu vào lớn KHÔNG có trong ZIP (chỉ trên máy) — vị trí, cách nhận diện, lệnh dùng

| Đầu vào | Vị trí trên máy (tương đối gốc repo) | Nhận diện | Lệnh dùng / cách lấy |
|:--|:--|:--|:--|
| UA-DETRAC ảnh (100 video train+test) | `data/extracted/DETRAC-Images/<video>/img*.jpg` | 60 video train + 40 test (danh sách `results/phase2/manifests/detrac_{dev,test}.json`) | `README.md` mục "Tải bộ ảnh train"; `src/extract_verify.py` |
| UA-DETRAC annotation XML | `data/extracted/DETRAC-{Train,Test}-Annotations-XML/<video>.xml` | — | `src/parse_detrac_xml.py`, `src/to_motchallenge.py` → `data/processed/DETRAC-all/<video>/gt/gt.txt` |
| GT MOTChallenge + seqmaps | `data/processed/{DETRAC-all,DETRAC-test}/<video>/{gt/gt.txt,seqinfo.ini}`, `data/processed/seqmaps/` | sinh từ XML (tái lập được) | như trên |
| Đoạn che khuất GT | `data/interim/{occlusion_segments,full_occlusion_segments}.csv` (train), `*_test.csv` | sinh từ XML | `src/occlusion_segments.py` |
| Detector checkpoint | `models/yolov8n.pt` | sha256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` (ultralytics yolov8n COCO) | `from ultralytics import YOLO; YOLO("yolov8n.pt")` tự tải |
| **Detection cache** (dẫn xuất) | `data/interim/det_cache/yolov8n-coco_conf0.05_all/{meta.json,<video>.npz,<video>.json}` | `meta.json` ghi sha256 weights, conf 0,05, 80 lớp, max_det 3000, imgsz 640, iou 0,7, phiên bản torch/ultralytics/cuDNN, GPU; 52 video (12 dev + 40 test) | **ĐÃ GÓM vào ZIP nếu ≤ 60 MB** (xem `handoff/INDEX.md`). Tạo lại: `python src/det_cache.py --dataset detrac --part train --manifest results/phase2/manifests/detrac_dev.json --cache-id yolov8n-coco_conf0.05_all` (cần GPU + ảnh) |
| Output tracker từng frame | `data/processed/trackers/<split>/p2-*/data/<video>.txt` (+ `assoc_log/*.parquet`, `timing/*.csv`, `pf_diag/*.csv`) | `run_meta.json` (có trong ZIP) ghi cache_id, YAML sha, commit, seed | Tái tạo từ cache: `python src/run_experiment.py --exp configs/phase2/experiments/<exp>.yaml --stage track` (CPU, ~1 phút/tracker/12 video) |
| TrackEval chi tiết | `results/trackeval/<split>/p2-*/` | `per_video_metrics.csv`, `combined_metrics.csv`, `raw_fields.json` **có trong ZIP** | `--stage eval` |
| Output pre-fix/fixed của `verify_shared_kalman_regression.py` | `results/phase2/shared_kalman/{prefix,fixed}/` (gitignored) | tóm tắt trong `diff_summary.csv` (trong ZIP) | `python src/verify_shared_kalman_regression.py` |
| BDD100K-MOT | **chưa có trên máy** | — | `src/check_bdd100k.py` khi có; fixture nhỏ `data/fixtures/bdd100k_mini` có trong ZIP |
| Jetson Nano | chưa kết nối | — | `scripts/nano/` |

Môi trường: `handoff/versions.json` (Python, torch/cuda, ultralytics 8.4.141, trackeval 1.3.0, pip freeze, GPU GTX 1650, Windows 11).
Bước cần GPU: tạo cache detection, `test_run_tracking_replay.py`, `profile_pipeline.py`. Mọi bước tracking/đánh giá còn lại chạy CPU từ cache.
