# REVIEW_BRIEF — vòng 15: triển khai Nano — script thiết bị, detector TensorRT (phần kiểm được trên PC), gói v3

<!-- BASE_TARGET -->

Base review **giữ `a5eec39`**. `CODEX_REVIEW_7a2feb6.md` **không có trong repo** — hai finding xử lý theo mô tả của người
dùng (gửi file để lưu). Không tuning/train; tracker, cấu hình khóa, GT, kết quả UA-DETRAC/BDD, 4 run `v2` **không đổi**.
Chi tiết: **`handoff/REVIEW15_CHI_TIET.md`**; hướng dẫn: `NANO_START_HERE.md` trong gói.

## Thiết bị: chưa chạy được gì

`10.11.53.239` ping/SSH timeout; quét /24: host duy nhất mở SSH (`.179`) có host key **khác** Nano đã lưu ⇒ Nano **không có
trên mạng**; không có khóa SSH. **Chưa có bước nào chạy thật trên Nano.** Khối lệnh đầy đủ ở phụ lục mục 5.

## Hai finding `7a2feb6` — đã xử lý (`src/nano_run_cache.py`)

Script **riêng cho thiết bị**: kiểm **đầu vào** trước (2 cache meta/npz/sidecar; **đủ 4** `pc-ref-*` đọc được; YAML; mã;
backend np với torch/ultralytics **chặn**) — thiếu → **exit 2, không chạy**; hash cây `pc-ref-*` đối chiếu
`reference/pc_ref_sha256.txt` **trước** và kiểm **không đổi sau**; smoke CV/EKF → parity + bit-identical, đạt mới chạy full;
**chỉ ghi `<tag>-*`**; `lenh_tai_hien.sh` chỉ dùng trên PC. Verify gói chạy chính script này: ĐẠT ×4 bit-identical, `pc-ref`
không đổi; thiếu đầu vào → exit 2.

## Detector cho Nano — phần kiểm được trên PC

Binding TensorRT trên JetPack 4.6 chỉ cho **Python 3.6**, tracker cần **≥ 3.8** ⇒ `src/nano_detector.py` (cú pháp 3.6,
`python3` hệ thống: tensorrt + numpy + cv2, bộ đệm GPU torch/pycuda) ghi **det cache đúng định dạng PC**; tracker ở venv ≥ 3.8
replay — **cùng cơ chế PC, không có tracker riêng cho Nano**. Quy ước ultralytics giữ nguyên; ghi rõ cache PC dùng `.pt` rect 640×384.

`test_nano_detector` **6/6** (torch FP32 thay TRT — ghi rõ): letterbox **pixel-identical**; decode+NMS+scale **lệch 0** so với
`nms.non_max_suppression`/`ops.scale_boxes` trên cùng đầu ra thô; NMS tổng hợp; cache → `open_cache` → `run_tracking`;
so cache PC 40 frame: ghép 100 %, IoU 0,9975. **Baseline e2e PC** (bản giải nén sạch, 300 frame): so cache PC ở mức vào tracker
579/581 ghép, IoU 0,996, |Δconf| 0,0078; tracking **không parity** với `pc-ref` (549 vs 545 hàng CV) — đúng dự kiến, là mốc
"hiệu ứng letterbox" để tách phần FP16 khi có số Nano. Kèm `nano_det_compare.py`, `render_mot_video.py`,
`scripts/nano/03_e2e_nano.sh` (engine **trên Nano** → detector → cache → so → tracker → video).

## Gói v3 `handoff/nano_transfer_latest.zip`

**63.792.494 byte**, sha256 `4dc0530aec7946e1437bbb767d755057d691e634c159c9cca53161feb0164396`, 544 thành viên, dựng tại
`0b190b3`. Verify (thư mục tạm, 2 stack: PC và numpy 1.24/lap 0.5.12/không scipy): manifest 543/543, env_check, import np,
`test_nano_parity` 20/20, **`nano_run_cache`** smoke + full parity ĐẠT bit-identical ×4, thiếu đầu vào → exit 2,
`det_compare` cắt-vs-gốc 845/845, ONNX, JPG/mp4, render → **ĐẠT** (gói đủ; không chứng minh tương thích Nano).

## Trạng thái, commit, bước tiếp theo

Nano: **chưa chạy** (mốc 1, 2 đều chưa). Commit: `0b190b3` mã + script + START_HERE; sau đó log/tài liệu. Không chạy full
suite (mã lõi không đổi); `test_nano_detector` đăng ký vào runner. Bước tiếp theo: khi Nano bật, chạy khối lệnh phụ lục mục 5
(scp → manifest → `00_check_stack` → venv 3.8 → `nano_run_cache` → nếu ĐẠT: `03_e2e_nano.sh`), gửi lại `reference/` +
`results_nano/`.
