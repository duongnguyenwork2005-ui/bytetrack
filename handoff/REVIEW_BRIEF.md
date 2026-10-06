# REVIEW_BRIEF — vòng 44: review `a03f6e1` — bổ sung mã phụ thuộc (R43-B1), sửa metadata ngưỡng vận hành (R43-N1)

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW44_CHI_TIET.md`**. Phạm vi: chỉ xử lý review `a03f6e1`. Không train, không chạy lại inference,
không UA-DETRAC (D1–D5 chưa duyệt). Base **giữ `4d54744`**, vì review chưa PASS đầy đủ.

## R43-B1 — bổ sung dependency (không phải lỗi evaluator)

- Gói kèm **17 file không đổi** trong `support/` (cơ chế `--support`, blob target nguyên byte). Đây là bao đóng import của
  entry point BDD, `test_danh_gia_bdd_val.py` và test evaluator cũ `test_detector_5lop.py`. Danh sách, vai trò và gói
  Python ngoài: `logs/review44/phu_thuoc_python.md`.
- **Hash**: `eval_detector_5lop` và `detector_classes` khớp bảng review. `bdd_yolo_export` (`1ca964b0`) và `bdd100k_coco`
  (`5028ffb9`) trong bảng là hash **file trên đĩa (CRLF)**, đo lúc chạy; gói gửi blob LF (`2fbc2314`, `c6c3ad28`).
  Đổi LF của blob thành CRLF thì trùng từng byte. Đây chỉ là EOL lúc checkout, không phải snapshot khác (`06_*`).
- **Cây sạch ngoài repo** (`kiem_cay_sach.py`: mọi tiến trình được ghi nguồn mô-đun, 0 mô-đun từ repo gốc):
  - test BDD **22/22**;
  - test cũ **76/76 + 6 SKIP** khi không có weights, **92/92** khi có weights ngoài gói.

## R43-N1 — metadata `--conf-van-hanh`

- Tái hiện bằng script reviewer: `bug_count 1`.
- Sửa: `quy_tac_hieu_luc()` tạo bản sao sâu theo từng lượt; metadata, scorer và chẩn đoán dùng cùng ngưỡng; `QUY_TAC`
  không bị sửa. Không đổi AP / ghép / ignore / NMS / ánh xạ lớp.
- Regression B5 qua entry point:
  - mã cũ **20/22** (bắt lỗi);
  - biến thể sửa trực tiếp hằng **21/22** (bắt rò cấu hình);
  - mã mới **22/22**.

  Script reviewer sau sửa: `bug_count 0`.
- Ba lượt BDD vòng 43 dùng 0,25, nên lỗi không kích hoạt. Kết quả, checkpoint và file riêng không đổi (`07_*`).

## Giới hạn

- Kiểm trên Windows, Python 3.11. Smart App Control chặn DLL của pyarrow, nên các lần kiểm dùng `--chan-pyarrow` (pandas
  chạy không có pyarrow); lần chạy trong repo FAIL vì lý do này, log được giữ. Linux / 3.12 **chưa chạy**.
- B5 và script reviewer chỉ kiểm tính nhất quán của ngưỡng, không chứng minh AP đúng.

## Kiểm gói cuối

Chạy `kiem_cay_sach.py --zip <gói>` (lệnh ở `phu_thuoc_python.md`). Kết quả của tác giả nằm trong tin nhắn bàn giao.

**Nên xem**: `logs/review44/03_*`, `04_*`, `05_*`, `06_*`, `phu_thuoc_python.md`, `DEPENDENCIES.md` (mục support).
