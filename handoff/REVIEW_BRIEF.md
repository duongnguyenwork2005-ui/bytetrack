# REVIEW_BRIEF — vòng 39: resume cho runner huấn luyện và bộ giám sát RAM bám PID

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW39_CHI_TIET.md`**.
Kế hoạch kiểm đăng ký trước, kèm phụ lục sửa đổi: `handoff/logs/review39/01_ke_hoach_kiem_resume.md`.

**Phạm vi** (người dùng duyệt): thêm resume cho runner và sửa bộ đo RAM **trước** lượt training dài; kiểm kỹ thuật ngắn
(fixture CPU, dataset smoke trên GPU); bàn giao; dừng. Không training dài, không dataset đầy đủ, không Nano.

## Ghi nhận nghiệm thu — ĐÃ CHẠY

- Codex `b83dc12`: **PASS WITH LIMITATIONS**. Bằng chứng khớp 137/137. Gói 816.591 B (`f4efa478…`) khớp, lưu archive.
  **Base → `b83dc12`**; 13 khai báo NPZ giữ nguyên.
- Đã gộp hai đính chính vào tài liệu; số liệu không đổi:
  - **R38-N1**: 789 bước/epoch là ước tính với accumulate cố định;
  - **R38-N2**: phép kiểm frame dùng lại `chon_frame()`, nên là "kiểm chéo", không độc lập.

## Resume — ĐÃ CÀI, ĐÃ KIỂM

- **Cú pháp**: `--che-do day-du --xac-nhan-train-dai --tiep-tuc`. `--dry-run --tiep-tuc` chỉ kiểm, không ghi gì.
- **Cơ chế**: resume của ultralytics với **đường dẫn** checkpoint (không `resume=True`). Tiếp tục **sau epoch đã hoàn
  tất**; không giữa epoch, không giữa batch.
- **Từ chối trước khi train** (exit 2, không gọi `train()`):
  - không có nhật ký lượt (lượt smoke/pilot cũ);
  - checkpoint hỏng, thiếu trạng thái, hoặc không thuộc lượt;
  - lượt đã xong, hoặc epoch mang quyết định dừng;
  - cấu hình / dataset / môi trường không tương thích; không cộng thêm epoch.
- **Khôi phục và kiểm lại sau khi nạp** (13 mục):
  - epoch, tổng epoch; optimizer; EMA + updates;
  - model đang học (= EMA của checkpoint);
  - LR / scheduler, best_fitness, EarlyStopping (runner khôi phục), tham số, batch.

  Mosaic thực tế được ghi mỗi epoch.
- **Không khôi phục**:
  - RNG, thứ tự dữ liệu, augmentation ngẫu nhiên;
  - gradient tích lũy dở: bước tối ưu đầu sau resume chỉ mang 1 batch.

  Vì vậy tiếp tục **không trùng từng bit** với lượt liên tục.
- **Bảo toàn**:
  - bản sao checkpoint mỗi epoch;
  - lưu file cũ trước khi ghi tiếp;
  - `best.pt` theo quỹ đạo;
  - `results.csv` cắt về epoch đã lưu.

## Bộ giám sát — ĐÃ CÀI, ĐÃ KIỂM

- `src/chay_giam_sat.py` bám PID của tiến trình **do chính công cụ tạo** (kèm `create_time`).
- Mỗi mẫu ghi RSS tiến trình chính, RSS từng tiến trình con và RAM trống toàn máy. Không ghi 0. Không gộp VRAM.
- Exit code được giữ, kể cả NTSTATUS. Có `trang-thai` (thay cho việc đoán từ monitor hết hạn) và `tinh-lai`.

## Kiểm thử — ĐÃ CHẠY

| Test | Kết quả |
|:--|:--|
| `test_chay_giam_sat` (tiến trình giả, 11 ca) | 23/23 |
| `test_tiep_tuc_huan_luyen` (CPU, fixture, 2 model; chạy thật + spy) | lần 5: **68/68**. Lần 1–4 giữ log: lỗi test; đo mosaic sai; 60/60; 64/64 |
| Kiểm GPU, dataset smoke, 2 model, mọi lượt qua bộ giám sát | **ĐẠT** 21/21 tiêu chí mỗi model |
| Tính lại RAM độc lập từ CSV (6 lượt GPU) | 60/60 điều kiện |
| `test_detector_5lop` / `test_pilot_chan_doan` | 92/92 / 34/34 |

**Trên CPU** (fixture):
- lượt liên tục trùng từng bit với chính nó;
- ngắt rồi tiếp tục trùng lượt liên tục ở epoch, ni, LR, momentum, batch, mosaic (cả khi tiếp tục sau epoch tắt mosaic);
- khác ở loss, thứ tự dữ liệu, EMA (\|lệch\| max 0,37 / 0,40), đúng như dự đoán.

**Trên GPU**:
- 6/6 lượt bám đúng PID runner; exit 0 / 15 / 0 được giữ;
- hai lượt cùng cấu hình có thể đã khác nhau (yolov8n), nên không tách được phần khác do resume;
- RAM trống toàn máy xuống **17 MiB** khi runner tạo 6 worker.

**Phát hiện**:
- `pilot_chan_doan.py` vòng 37 ghi trường `mosaic_thuc_te` bằng thuộc tính không tồn tại, nên trường này luôn `False`.
  Kết luận vòng 37 vẫn có bằng chứng từ log.
- **Sự cố**: 4 file tài liệu bị ghi trong lúc kiểm GPU đang chạy, do lỗi công cụ. Bằng chứng không bị ảnh hưởng; chi tiết ở
  phụ lục mục 10.

## Trạng thái

- **Kỹ thuật: sẵn sàng** cho lượt dài. Lệnh chạy trong terminal riêng, cách lưu log và lấy exit code: phụ lục mục 8.
- **Quyết định: chưa sẵn sàng**. Chưa được duyệt; vòng này cũng chưa được nghiệm thu.

**Chờ duyệt**:

- số epoch (đề xuất 15), ngân sách (kịch bản 56,64 h / 73,95 h), thời điểm bắt đầu;
- đóng bớt ứng dụng (RAM);
- có sửa trường mosaic của pilot vòng 37 hay không.

**Nên xem**: `logs/review39/01_*`, `03_*_lan5.log`, `06_kiem_resume_gpu.json`, `05_gpu/`, `07_*`.
