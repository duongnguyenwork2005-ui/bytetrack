# REVIEW_BRIEF — vòng 21: sửa R20-P2/P3; luồng trực tiếp video → TensorRT → tracker trên Nano

<!-- BASE_TARGET -->

Base **`45e086a`** (`handoff/CODEX_REVIEW_45e086a.md`). Benchmark vòng 20, UA-DETRAC/BDD, `pc-ref-*` không đổi. Chi tiết:
**`handoff/REVIEW21_CHI_TIET.md`**.

## Ba điểm review

- **R20-P2** (`281117f`): patch là **bytes** từ stdout Git đến ZIP; kiểm đường dẫn trên bản xem riêng (`split("\n")`). Sau khi dựng gói
  tự kiểm **roundtrip** (clone tạm + index tạm + `apply --cached`, so blob từng đường dẫn). `--giu-khai-bao` giữ khai báo NPZ vòng 16.
  Tái hiện: 15 B → 13 B, apply exit 0. `test_review_delta` 19/19; phép 14–18 (thêm/sửa/xóa CRLF, trộn LF/CRLF + CR lẻ, binary;
  không `.gitattributes`) **fail trên mã `45e086a`**.
- **R20-P3** (`1efae6f`): `wall_s` = subprocess `run_tracking.py`, không gồm `conda run` của cha (warm-up có gồm); câu thống kê thay bằng
  "chưa thực hiện kiểm định hoặc xây dựng khoảng tin cậy; IQR chỉ mô tả độ phân tán của 6 hiệu số". Số +0,752 ms/frame không đổi.

## Luồng trực tiếp (`f63903c`, `47ef97e`)

Hai tiến trình vì TensorRT chỉ có binding Python 3.6: con `nano_detect_stream.py` (3.6, tái dùng `nano_detector`) gửi **từng frame** qua
pipe có cửa sổ credit K = 4; cha `nano_live.py` (3.8, backend np) lọc bằng `det_cache.filter_dets` và chạy **`run_tracking.track_video`**
dùng chung với replay. Mã thoát 0 hoàn tất / 2 lỗi trước vòng lặp / 3 lỗi giữa luồng (ghi phần đã xử lý, không "hoàn tất") / 4 dừng
(`touch <out>/STOP` hoặc Ctrl+C). Khởi động: `scripts/nano/04_live_nano.sh cv|ekf <out>`.

## Trên Nano thật

- Gói v6: `test_nano_stream` 12/14 → phát hiện **OpenCV 3.2 abort cả tiến trình** với BMP hỏng, và test giết con trước HELLO. Sửa rồi
  dựng **v7 `6dc4100d…`**: test thiết bị 20/20, 9/9, **14/14** (kể cả SIGINT).
- Engine FP16 build trên máy (483 s, `c32690cd…`).
- Máy **mất nguồn** khoảng 11:52 (ext4 phục hồi, 0 lỗi); mọi kết quả dưới đây tạo sau đó.
- **Parity 10/10**: giả lập trên 300 JPG thật == `pc-ref` smoke. **TensorRT thật CV và EKF: live == replay từ cache, byte giống hệt.**
  Biên: ảnh đen, mp4 300/300, stop-file, SIGINT, `kill -9` con, engine sai → đúng mã thoát.

## FPS toàn luồng (điều kiện công bố trước `42dff0c`; 6 lượt, không loại lượt)

300 JPG 960×540, c25, không `--regions`, không render, MAXN, `schedutil`.

| | FPS vòng lặp (3 lượt) | trễ/frame trung vị | `tracker.update()` |
|:--|:--|:--|:--|
| CV | **9,29** (9,11–9,36) | 114,6–118,6 ms | 5,65–5,76 ms |
| EKF | **9,37** (9,29–9,44) | 114,2–115,8 ms | 6,39–6,56 ms |

Tiến trình detector quyết định thông lượng: decode 11,5 + pre 13,3 + infer ~74 (execute ~68,6) + post 7,3 ≈ 106 ms/frame. Tracker chạy
song song ở cha nên chênh lệch CV/EKF không thấy trong FPS. Kết quả mô tả, không kiểm định.

## Giới hạn

300 frame smoke, một sequence, không camera/render; không cộng số vòng 16/20; nguồn điện Nano chưa ổn định.
Full suite PC: lần 1 33/34 (Smart App Control chặn DLL fastparquet; chạy riêng 14/14), lần 2 **34/34**. Delta từ `45e086a` đã kiểm roundtrip theo byte/hash; không có gói Nano, engine, weights hay video.
