# Vòng 44 — khai báo phụ thuộc để chạy test fixture từ gói delta (R43-B1)

## Mã dự án gửi kèm (`support/`, không đổi so với base `4d54744`)

Đây là bao đóng import tĩnh (gồm cả import trong hàm) của:
- entry point `src/danh_gia_detector_bdd_val.py`;
- `src/test_danh_gia_bdd_val.py`;
- `src/test_detector_5lop.py` (test evaluator / exporter cũ, gửi kèm tùy chọn).

Hai file đầu nằm trong patch, vì chúng mới từ vòng 43.

| File | Ai cần | Nạp khi chạy test? |
|:--|:--|:--|
| `src/config.py` | mọi file | có |
| `src/bdd_yolo_export.py` | entry point, test BDD, evaluator | có |
| `src/detector_classes.py` | entry point, evaluator | có |
| `src/eval_detector_5lop.py` | entry point, test BDD | có |
| `src/mot_datasets/__init__.py`, `base.py`, `bdd100k_coco.py`, `bdd100k_mot.py` | `NguonCocoVid` (đọc nhãn COCO-video) | có |
| `src/mot_datasets/detrac.py` | `mot_datasets.get_dataset("detrac")`, `prepare_mixed_yolo_dataset` | không (chỉ tham chiếu tĩnh) |
| `src/test_detector_5lop.py` | test cũ | có |
| `src/train_detector.py`, `src/tiep_tuc_huan_luyen.py` | test cũ (nhóm F/G/J, runner) | có |
| `src/pilot_chan_doan.py` | `train_detector` chế độ pilot | không (chỉ tham chiếu tĩnh) |
| `src/mot_datasets/make_bdd5_train_fixture.py` | test cũ (fixture) | có |
| `src/prepare_mixed_yolo_dataset.py` | test cũ (nhóm D, chạy qua subprocess) | có |
| `src/export_yolo_dataset.py` | `prepare_mixed_yolo_dataset` (nhánh UA-DETRAC) | không (chỉ tham chiếu tĩnh) |
| `configs/phase2/detector/train_bdd5.yaml` | test cũ (nhóm F/G/J, đọc khi có weights) | file dữ liệu |

Cột cuối lấy từ lần thử cây sạch (`05_do_dependency_cay_sach.log`). `config.py` tạo thư mục `models/` cạnh `src/` khi
import; đây là hành vi sẵn có, không ghi file nào.

## Gói Python ngoài (không đóng gói)

Đã kiểm trên CPython 3.11.9, Windows 11, env `khoaluan-mot`.

| Test | Gói cần | Phiên bản đã dùng |
|:--|:--|:--|
| `test_danh_gia_bdd_val.py` (22 phép) | numpy; pandas (import ở mức module của `mot_datasets.bdd100k_*`); Pillow (ảnh fixture); opencv (`cv2`, B2: exporter tô xám vùng bỏ qua) | numpy 2.4.6, pandas 2.3.3 (+ python-dateutil, pytz, six), Pillow 12.3.0, opencv-python 5.0.0.93 |
| `test_detector_5lop.py` (92 phép) | như trên + PyYAML, ultralytics, torch / torchvision, pycocotools | PyYAML 6.0.3, ultralytics 8.4.141, torch 2.6.0+cu124, torchvision 0.21.0+cu124, pycocotools 2.0.11 |

- **pyarrow không cần.** Trên máy tác giả, Smart App Control chặn DLL `pyarrow._compute` (`02a_codeintegrity_pyarrow.log`),
  làm `import pandas` lỗi. Vì vậy các lần kiểm dùng `kiem_cay_sach.py --chan-pyarrow`: `import pyarrow` thất bại có chủ
  ý, nên pandas chạy như khi không cài pyarrow. Trên máy không bị chặn thì bỏ cờ này.
- **Tài nguyên ngoài gói** cho `test_detector_5lop.py`:
  - **weights**, đặt bằng `DET5_MODELS_DIR=<thư mục>` với `yolov8n.pt` (sha256 `f59b3d83…`) và
    `candidates/yolo11n.pt` (`0ebbc80d…`). Nhóm cần weights: A7, E6, F1–F5, G, J3/J4/J6. Thiếu weights thì các nhóm
    này in `[SKIP] … CHUA KIEM`, **không** tính PASS. Kết quả của tác giả: không có weights 76/76 PASS + 6 SKIP; có
    weights 92/92;
  - **pycocotools**: nhóm E7–E9 so AP với pycocotools thật. Thiếu thư viện thì nhóm này SKIP "CHUA KIEM".
- Không cần dataset, ảnh thật, NPZ hay GPU.

## Cách chạy (từ một thư mục bất kỳ ngoài repo)

```text
# dung cay tu goi, kiem sha256 moi file voi MANIFEST, chay test, kiem module du an nap tu cay
python <goi>/logs/handoff/logs/review44/kiem_cay_sach.py --zip <goi.zip> --thu-muc <thu muc MOI> \
    --chay "src/danh_gia_detector_bdd_val.py --help" --chay src/test_danh_gia_bdd_val.py \
    --chay src/test_detector_5lop.py [--models-dir <weights>] [--chan-pyarrow] --out <ket qua.json>

# hoac thu cong: cay = files/** + support/** cua goi, roi
python <cay>/src/test_danh_gia_bdd_val.py
```
