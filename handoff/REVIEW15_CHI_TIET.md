# Phụ lục vòng 15 — triển khai Nano: hai finding công cụ bàn giao, script thiết bị, detector TensorRT (phần kiểm được trên PC)

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Hướng dẫn trong gói: `NANO_START_HERE.md` (nguồn `scripts/nano/NANO_START_HERE.md`).
Trạng thái: `handoff/NANO_STATUS.md`. Review `CODEX_REVIEW_7a2feb6.md` **không có trong repo/Downloads** — hai finding xử lý theo
mô tả của người dùng; gửi file để lưu vào `handoff/`.

## 0. Trạng thái trước / thiết bị

`logs/review15/trang_thai_truoc.log`: HEAD `7a2feb6`, tree chỉ file riêng; base review **`a5eec39`** (giữ). Thử kết nối
(`logs/review15/thu_ket_noi_nano.log`, `quet_mang.log`): ping `10.11.53.239` timeout; SSH BatchMode timeout; quét /24 — ARP có
mục `.239` (MAC `de-f9-1b-…`, không phải OUI NVIDIA, có thể cũ); host duy nhất mở SSH là `.179` nhưng chạy OpenSSH 9.6 và
host key ed25519 **khác** key đã lưu của Nano ⇒ **Nano không có trên mạng lúc này**; không có khóa SSH cục bộ (chỉ mật khẩu).
⇒ Mọi bước trên thiết bị **chưa chạy**; khối lệnh ở mục 5.

## 1. Hai finding của review `7a2feb6` (công cụ bàn giao) — đã xử lý

| Finding | Xử lý | Kiểm |
|:--|:--|:--|
| Script chạy Nano riêng, chỉ ghi `nano-*`; không chạy script tái tạo `pc-ref-*` trên Nano | `src/nano_run_cache.py`: chỉ ghi `<tag>-{cv,ekf}` (từ chối `--tag pc-ref*`); `reference/lenh_tai_hien.sh` ghi rõ **chỉ để tái tạo tham chiếu trên PC**; `NANO_START_HERE.md` bước 3–4 dùng script này | verify gói (2 stack): `nano_run_cache --tag kiem` → smoke + full CV/EKF parity ĐẠT, bit-identical ×4, cây `pc-ref-*` **không đổi** sau chạy |
| Kiểm tồn tại/đọc được cache, metadata, đủ **bốn** tham chiếu PC; thiếu → báo lỗi | bước 0 của script: 2 cache (meta.json đọc được, npz đủ 7 mảng, `Σn_per_frame == len(data)`, sidecar), 4 file `pc-ref-*/data/MVI_39031.txt` đọc được, YAML, mã, backend np import với torch/ultralytics **chặn** → thiếu bất kỳ → **exit 2, không chạy** | verify: đổi tên tạm `NANO-full/pc-ref-ekf/data/MVI_39031.txt` → exit 2, không tạo output `kiem2-*` |
| Hash tham chiếu trước/sau | build ghi `reference/pc_ref_sha256.txt` (SHA256 mọi file của 4 thư mục `pc-ref-*`); script đối chiếu trước khi chạy (khác → exit 2) và hash lại sau khi chạy (khác → KHÔNG ĐẠT) | verify: `pc_ref_khop_ban_dong_goi=true`, `pc_ref_khong_doi_sau_chay=true` |

## 2. Detector cho Nano — kiến trúc và phần đã kiểm trên PC

Binding Python của TensorRT trên JetPack 4.6 chỉ có cho **Python 3.6**; mã tracker cần **≥ 3.8** (vermin, vòng 14).
⇒ `src/nano_detector.py` viết bằng **cú pháp 3.6**, chạy ở `python3` hệ thống (tensorrt + numpy + cv2; bộ đệm GPU bằng torch
1.10 có sẵn hoặc pycuda), ghi **det cache đúng định dạng PC**; tracker chạy ở venv ≥ 3.8 replay cache — **cùng cơ chế PC,
không có logic tracker riêng cho Nano**. Quy ước giữ nguyên: letterbox 640 (auto=False, static ONNX), BGR→RGB /255, decode
`84×8400` → lớp max, conf > 0,05, NMS theo lớp (offset 7680, IoU > 0,7 loại), max_det 3000, `scale_boxes` + clip đúng công thức
ultralytics; đầu ra `(n,6) [x1,y1,x2,y2,conf,cls]`. Ghi rõ khác biệt **có chủ ý**: cache PC dùng `.pt` với `rect=True` → input
640×384; đường ONNX/TRT dùng 640×640.

`src/test_nano_detector.py` **6/6** (`logs/review15/test_nano_detector.log`; suy luận thô bằng torch FP32 trên PC thay TRT — ghi rõ):

| # | Phép kiểm | Kết quả |
|:--|:--|:--|
| 1 | `letterbox(640)` vs ultralytics `LetterBox(auto=False)` (3 frame 960×540) | **pixel-identical**, r 0,6667, pad (0,140) |
| 2 | `preprocess` vs permute/flip/÷255 của `BasePredictor` | bằng nhau chính xác |
| 3 | `decode_nms + scale_boxes` vs `nms.non_max_suppression + ops.scale_boxes` trên **cùng** đầu ra thô (5 frame) | cùng số box/thứ tự; bbox, conf **lệch 0**; cls bằng |
| 4 | NMS tổng hợp: khác lớp giữ cả hai; cùng lớp IoU>0,7 giữ 1; IoU==0,7 **giữ**; max_det; classes; rỗng→(0,6) | đúng |
| 5 | pipeline 40 frame (che vùng bỏ qua) → cache → `open_cache` strict → `run_tracking` exit 0 → so cache PC | ghép **100 %**, IoU 0,9975, \|Δconf\| tb 0,0033 (PC cuda FP32: decode 3,7 / pre 4,8 / infer 14,4 / post 0,5 ms — **không phải số Nano**) |
| 6 | ghi thêm vào cache có meta khác (conf) | `RuntimeError`, không trộn |

**Baseline e2e trên PC** (`logs/review15/e2e_pc_baseline/`, chạy trong bản giải nén sạch của gói; detector = torch FP32 GPU +
letterbox 640×640, **không phải TRT**): 300 frame → cache → CV/EKF (np, chặn torch) → so với `pc-ref`: **không parity** như dự
kiến (khác detector: 549 vs 545 hàng CV, 546 vs 545 EKF); `det_compare` mức vào tracker (c25): 579/581 ghép (99,7 %), IoU tb
0,996, \|Δconf\| tb 0,0078, max 0,122, trùng lớp 99,7 %, 291/300 frame cùng số box; mức thô (conf 0,05): 836/845 ghép, IoU
0,994. ⇒ Đây là **hiệu ứng letterbox 640×640 riêng**; kết quả TRT FP16 trên Nano sẽ được so với cả hai mốc này (cache PC và
baseline này) để tách phần do FP16.

Công cụ kèm theo: `src/nano_det_compare.py` (so detection hai cache: ghép IoU, \|Δconf\|, trùng lớp, chênh số box/frame — mô tả,
không đạt/không đạt), `src/render_mot_video.py` (MOT + ảnh → mp4; kiểm trong verify: 300 frame, 545 box), `scripts/nano/03_e2e_nano.sh`
(engine **trên Nano** → detector TRT → cache → `det_compare` → tracker CV/EKF → `nano_parity` mô tả → video).

## 3. Gói chuyển v3 — `handoff/nano_transfer_latest.zip`

**63.792.494 byte**, sha256 `4dc0530aec7946e1437bbb767d755057d691e634c159c9cca53161feb0164396`, 544 thành viên; dựng tại
`0b190b3` (tree sạch ngoài log). Thêm so với v2: `nano_run_cache.py`, `nano_detector.py`, `nano_det_compare.py`,
`render_mot_video.py`, `test_nano_detector.py`, `03_e2e_nano.sh`, `reference/pc_ref_sha256.txt`, `NANO_START_HERE.md` mới.

Kiểm gói (giải nén thư mục tạm ngoài repo, cwd = gốc gói; `logs/nano_transfer/kiem_goi_sach{,_oldstack}.*`), **2 stack**
(PC: Py 3.11.9 / numpy 2.4.6 / lap 0.5.13; cũ: numpy 1.24.4 / pandas 1.5.3 / lap 0.5.12, **không scipy/cv2**): manifest 543/543;
`nano_env_check` ĐẠT; import np `torch=false`; `test_nano_parity` 20/20; **`nano_run_cache --tag kiem`**: smoke 545/545 ×2 và
full 5842 / 5845 parity ĐẠT + bit-identical ×4, `pc-ref` không đổi; thiếu đầu vào → exit 2; `det_compare` cache cắt vs gốc =
845/845 ghép, IoU 1, Δconf 0; ONNX đọc được; 300 JPG + mp4; render 300 frame. **ĐẠT** — chỉ chứng minh gói đủ, không chứng
minh tương thích Nano.

## 4. Đã chạy thật trên Nano / chưa

| Mốc | Trạng thái |
|:--|:--|
| Tracker-from-cache (import + parity CV/EKF) trên Nano | ❌ **chưa chạy** — thiết bị không tới được (mục 0) |
| End-to-end (engine → detector TRT → tracker) trên Nano | ❌ chưa; mã + script sẵn, hợp đồng tiền/hậu xử lý đã kiểm trên PC |
| Hiệu năng Nano | ❌ chưa có số nào; mọi số ms trong vòng này là PC |

## 5. Khối lệnh chạy trên Nano (khi thiết bị bật, cùng LAN; kiểm lại IP nếu đổi)

```bash
# --- PC ---
scp D:/LABOTARY/bytetrack/handoff/nano_transfer_latest.zip jetson@10.11.53.239:~/
# --- Nano ---
sha256sum ~/nano_transfer_latest.zip    # 4dc0530aec7946e1437bbb767d755057d691e634c159c9cca53161feb0164396
mkdir -p ~/nano_run && cd ~/nano_run && unzip -q ~/nano_transfer_latest.zip && cd nano_transfer
sha256sum -c MANIFEST_SHA256.txt --quiet && echo MANIFEST_OK
bash scripts/nano/00_check_stack.sh reference/nano_stack.txt      # CUDA/TensorRT/trtexec o dau, nvpmodel, RAM
python3 src/nano_env_check.py --json reference/env_check_py36.json  # bao ro: Py 3.6.9 KHONG DAT -> can moi truong rieng
apt-cache policy python3.8 | head -3                                # kiem nguon package TRUOC khi cai (khong mac dinh co)
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev build-essential   # hoac miniforge aarch64 neu khong co
python3.8 -m venv ~/nano_env && source ~/nano_env/bin/activate && pip install --upgrade pip && pip install -r requirements_nano.txt
python src/nano_env_check.py --json reference/env_check_nano.json   # phai DAT (1-4)
python src/test_nano_parity.py                                      # 20/20
python src/nano_run_cache.py                                        # MOC 1: chi ghi nano-*; exit 0 = parity DAT (smoke + full)
# --- MOC 2 (chi khi moc 1 DAT) ---
bash scripts/nano/03_e2e_nano.sh ~/nano_env/bin/python yolov8n_fp16.engine nano-trt
# gui lai: reference/{nano_stack.txt, env_check_*.json, nano_run_*.{json,log}, parity_nano_*.json}, results_nano/nano-trt/, nguyen van loi neu co
```

Không ghi mật khẩu vào đâu; không đổi `python3` hệ thống/torch.

## 6. Toàn vẹn, commit

Không đổi `run_tracking.py`, `det_cache.py`, tracker, cấu hình khóa, GT, kết quả UA-DETRAC/BDD, 4 run `v2`; không đụng file
riêng. Không chạy full suite (mã lõi không đổi); chạy `test_nano_detector` 6/6 (đăng ký vào `run_all_tests.py`), verify gói 2 stack.
Commit: `0b190b3` (mã + script + START_HERE), sau đó log/tài liệu. Base review giữ `a5eec39`.
