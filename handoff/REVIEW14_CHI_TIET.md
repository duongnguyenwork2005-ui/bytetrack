# Phụ lục vòng 14 — nghiệm thu launcher PC (`a5eec39`) và gói chuyển Jetson Nano độc lập

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Review: `handoff/CODEX_REVIEW_a5eec39.md`. Hướng dẫn trong gói: `handoff/nano_transfer/NANO_START_HERE.md`
(nguồn: `scripts/nano/NANO_START_HERE.md`). Trạng thái Nano: `handoff/NANO_STATUS.md`.

## 0. Trạng thái trước khi làm (`handoff/logs/review14/trang_thai_truoc.log`)

HEAD lúc bắt đầu `1de4fc5`, nhánh `phase2-mixed-traffic`; **2 commit sau `a5eec39` chưa được review**: `c6c60f0` (builder gói
Nano v1 + tài liệu), `1de4fc5` (log v1). Working tree chỉ có file riêng. Gói `review_delta_latest.zip` khớp **chính xác**
1.918.799 byte / sha256 `1c6b1336…` → lưu `handoff/archive/review_delta_a5eec39.zip`; base review → **`a5eec39`** (vòng 14,
`3c20fbe`). PASS WITH LIMITATIONS chỉ cho launcher PC; Nano F1/F3 và chạy trên thiết bị **chưa** nghiệm thu.

## 1. Thiết bị Nano — thông tin đã có và kết quả thử kết nối

| Nguồn | Nội dung |
|:--|:--|
| Người dùng cung cấp | Jetson Nano Dev Kit 4 GB, aarch64; Ubuntu 18.04.6, L4T R32.7.5; Python hệ thống **3.6.9**; numpy 1.19.5, scipy 0.19.1, pandas 0.22.0; **chưa có `lap`**; torch 1.10.0 `cuda.is_available()=True`; ~26 GB trống; SSH `jetson@10.11.53.239` đã đăng nhập được (từ Windows, bằng mật khẩu) |
| Thử từ phiên này (`logs/review14/thu_ket_noi_nano.log`) | PC ở `10.11.53.64` cùng /24; `known_hosts` có host; **không** có khóa SSH cục bộ; `ssh -o BatchMode=yes` → **timeout**; `ping` → "Destination host unreachable" ⇒ thiết bị **không tới được** lúc này (tắt/khác mạng). Không kiểm được CUDA/TensorRT/OpenCV/pip trên máy |

⇒ Mọi việc chuyển/chạy trên thiết bị (mục 5 yêu cầu) **chưa thực hiện**; gói và khối lệnh đã sẵn sàng để người dùng chạy.

## 2. Yêu cầu thực tế của mã cho đường tracker-từ-cache

- **Chuỗi import thật** (`logs/review14/import_chain.json`, tiến trình chặn torch/ultralytics): 23 module trong gói
  (`run_tracking` → `det_cache`, `ul_backend`, `bytetrack_np`, `tracker_ctrv`, `ekf/ukf/pf_ctrv`, `lost_association`,
  `time_units`, `run_state`, `run_fingerprint`, `mot_datasets` (registry), vendor `byte_tracker/basetrack/kalman_filter/
  matching/stracks/np_compat`); site-packages: **numpy, lap, pandas, yaml** (+ phụ thuộc của pandas). **scipy không được
  nạp** (chỉ dùng khi `use_lap=False`); torch/ultralytics không.
- **Python tối thiểu** (`logs/review14/vermin_min_python.log`, vermin 1.8.0 tĩnh): **3.8** — 21 file `from __future__
  import annotations` (3.7), `dataclasses` (3.7), `importlib.metadata` trong `ul_backend.fingerprint()` (3.8).
  ⇒ Python 3.6.9 hệ thống **không chạy được** mã hiện tại; sửa để chạy 3.6 là đổi 21+ file → **không** làm (tối thiểu
  thay đổi). Hướng đi: **môi trường riêng ≥ 3.8** (python3.8 từ apt universe + venv, hoặc miniforge aarch64), giữ nguyên
  python3 hệ thống và torch CUDA.
- **Kiểm stack cũ trên PC** (`logs/nano_transfer/kiem_goi_sach_oldstack.*`): venv scratch numpy **1.24.4**, pandas 1.5.3,
  pyyaml 6.0.1, lap 0.5.12, **không scipy, không cv2** → toàn bộ chuỗi replay + parity ĐẠT, `data/*.txt` **bit-identical**
  với tham chiếu tạo bằng numpy 2.4.6/lap 0.5.13 ⇒ mã không phụ thuộc numpy 2.x; `requirements_nano.txt` không ghim phiên
  bản (chỉ `lap>=0.4`).

## 3. Gói chuyển `handoff/nano_transfer_latest.zip` (dựng bằng `src/make_nano_transfer.py build`, `10d26fa`)

**63.764.728 byte**, sha256 `025187770c0876ad849ac32c27ba94b6aff934742f3592e80cf79ed8a99a8fd5`, 537 thành viên; thư mục
`handoff/nano_transfer/` 68,5 MB (untracked — không commit gói/zip, `.gitignore` là file riêng).

| Thành phần | Chi tiết |
|:--|:--|
| Mã/cấu hình/vendor | `src/` (không `src/demo/**`), `configs/`, `third_party/ultralytics_trackers/` (+`LICENSE` AGPL-3.0, `PROVENANCE.json`), `scripts/nano/` |
| Công cụ | `src/nano_env_check.py` (cú pháp 3.6, báo rõ thiếu gì), `src/run_tracking.py` (replay), `src/nano_parity.py` + `test_nano_parity.py`, `requirements_nano.txt` |
| Đoạn dữ liệu | `MVI_39031` (UA-DETRAC test) **frame 1–300**, 25 fps, 960×540, 12,0 s: 300 JPG **y nguyên byte** + mp4 nén lại (lossy, chỉ để `--source`) + `MVI_39031_smoke.json` + `seqinfo.ini`/`ignored_regions.txt` |
| Detection cache | `nano_smoke_MVI_39031_f0001-0300/` = **cắt tiền tố** frame 1–300 từ cache PC (`meta.json` copy nguyên văn = cùng DetectorSpec; sidecar ghi nguồn cắt; 845 box; đoạn này **không có frame rỗng** — cấu trúc vẫn giữ `n_per_frame=0`); `yolov8n-coco_conf0.05_all/` = cache gốc **1470 frame** (10.145 box, **34 frame rỗng giữ nguyên**, `time_s` tăng) |
| Tham chiếu PC | `data/processed/trackers/NANO-smoke/pc-ref-{cv,ekf}` (545/545 hàng) và `NANO-full/pc-ref-{cv,ekf}` (5842/5845 hàng): backend **np**, torch/ultralytics **bị chặn** khi tạo; đối chiếu backend ultralytics cùng đầu vào: parity ĐẠT **và bit-identical** (`reference/parity_pc_np_vs_ul_*.json`); cấu hình hiệu lực trong `run_meta.json` (c25: conf 0,25, lớp 1 2 3 5 7, max_det 300, fps 25, buffer 30 bước, elapsed steps) |
| Detector | `models/yolov8n.pt` (sha256 `f59b3d83…` = `weights_sha256` của cache), `models/export/yolov8n_640_op12.onnx` (sha256 `a66a5e03…`; `.export.json`: lệnh/tham số export, đối chiếu 127/127 initializer + 262/262 node với bản re-export, `logs/nano_transfer/kiem_onnx.*`) |
| Metadata | `PACKAGE_INFO.json`: HEAD/commit đóng gói `10d26fa`, commit code cuối không tính builder `56ce2c1`, môi trường PC, python tối thiểu, phụ thuộc, hash cache/ONNX/checkpoint; `MANIFEST_SHA256.txt` 536 file |

## 4. Kiểm gói trên PC (`src/make_nano_transfer.py verify`) — giải nén vào thư mục tạm ngoài repo, cwd = gốc gói, đường dẫn tương đối

| Bước | PC env (Py 3.11.9, numpy 2.4.6, lap 0.5.13, pandas 2.3.3) | Stack cũ (numpy 1.24.4, pandas 1.5.3, lap 0.5.12, **không scipy/cv2**) |
|:--|:--|:--|
| 0 manifest | 536/536 khớp | 536/536 |
| 1 `nano_env_check` + import np (chặn torch/ul) | ĐẠT (1–4); `torch=false, ul=false` | ĐẠT; `torch=false, ul=false` |
| 2 `test_nano_parity` | 20/20 | 20/20 |
| 3–4 replay CV/EKF + parity vs `pc-ref` | NANO-smoke 545/545 ×2, NANO-full 5842 / 5845: **ĐẠT, bit-identical** | **ĐẠT, bit-identical** (4/4) |
| 5 ONNX đọc/kiểm | ir 7, opset 12, 262 node, `1×3×640×640` | (dùng onnx của env PC) |
| 6 đoạn dữ liệu | 300 JPG 540×960×3, mp4 đọc lại 300 frame | (dùng cv2 của env PC) |

Kết luận: **ĐẠT — gói đủ file/phụ thuộc**; **không** chứng minh tương thích Nano. Log: `handoff/logs/nano_transfer/kiem_goi_sach{,_oldstack}.{log,json}`.
Tiêu chí parity giữ nguyên: bbox ≤ 1e-3 px, conf ≤ 1e-6, ID tương đương bằng song ánh cố định toàn sequence, mơ hồ không PASS.

## 5. Khối lệnh cho người dùng chạy trên Nano (mốc 1)

```bash
# --- tren PC (PowerShell/Git Bash), chep goi (khong ghi de thu muc du an dang co) ---
scp D:/LABOTARY/bytetrack/handoff/nano_transfer_latest.zip jetson@10.11.53.239:~/
# --- tren Nano ---
sha256sum ~/nano_transfer_latest.zip     # phai = 025187770c0876ad849ac32c27ba94b6aff934742f3592e80cf79ed8a99a8fd5
mkdir -p ~/nano_run && cd ~/nano_run && unzip -q ~/nano_transfer_latest.zip && cd nano_transfer
python3 src/nano_env_check.py --json reference/env_check_py36.json      # se bao Python 3.6.9 KHONG DAT -> can moi truong rieng
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev build-essential   # hoac miniforge aarch64
python3.8 -m venv ~/nano_env && source ~/nano_env/bin/activate && pip install --upgrade pip && pip install -r requirements_nano.txt
python src/nano_env_check.py --json reference/env_check_nano.json && bash scripts/nano/00_check_stack.sh
python src/test_nano_parity.py
bash reference/lenh_tai_hien.sh   # HOAC chay tay 4 lenh run_tracking trong NANO_START_HERE.md muc 3 voi --tracker-name nano-cv / nano-ekf
python src/nano_parity.py --ref data/processed/trackers/NANO-smoke/pc-ref-cv  --test data/processed/trackers/NANO-smoke/nano-cv  --json reference/parity_nano_cv.json
python src/nano_parity.py --ref data/processed/trackers/NANO-smoke/pc-ref-ekf --test data/processed/trackers/NANO-smoke/nano-ekf --json reference/parity_nano_ekf.json
# gui lai: reference/env_check_*.json, reference/parity_nano_*.json, data/processed/trackers/NANO-*/nano-*/run_meta.json, nguyen van loi (neu co)
```

`reference/lenh_tai_hien.sh` chạy với `--tracker-name pc-ref-*` (tạo lại tham chiếu tại chỗ); trên Nano dùng
`nano-cv`/`nano-ekf` như mục 3 của `NANO_START_HERE.md` để giữ nguyên tham chiếu PC. Không ghi mật khẩu vào đâu.

## 6. Đã làm / chưa làm / còn thiếu

| Hạng mục | Trạng thái |
|:--|:--|
| Launcher PC | ✅ nghiệm thu `a5eec39` (PASS WITH LIMITATIONS), base → `a5eec39` |
| Gói chuyển độc lập + kiểm sạch trên PC (2 stack) | ✅ |
| Yêu cầu Python/dependency của mã | ✅ đo tĩnh + kiểm stack cũ; **không sửa mã** cho 3.6 |
| Chuyển gói / chạy trên Nano (import, parity) | ❌ **chưa** — thiết bị không tới được lúc này; khối lệnh ở mục 5 |
| Detector TensorRT + adapter + video trên Nano; số đo tốc độ | ❌ chưa (mốc 2, sau khi parity đạt) |
| Còn thiếu từ thiết bị | kết quả `nano_env_check`, `00_check_stack.sh` (CUDA/TensorRT/`trtexec` ở đâu), `parity_nano_*.json`, cách tạo môi trường đã dùng |

Không đổi `run_tracking.py`, `det_cache.py`, tracker, cấu hình khóa, GT, kết quả UA-DETRAC/BDD, 4 run `v2`; không đụng file riêng.
Không chạy full suite (chỉ thêm công cụ đóng gói/kiểm môi trường + tài liệu; mã lõi không đổi).
