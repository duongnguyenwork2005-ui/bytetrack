# REVIEW_BRIEF — vòng 37: pilot chẩn đoán 2 epoch trên dataset smoke

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW37_CHI_TIET.md`**. Base review vẫn là `2351c67`; vòng 36 chưa nghiệm thu nên nằm chung trong delta.

**Phạm vi** (người dùng duyệt): YOLOv8n và YOLO11n, mỗi model một lượt **2 epoch**, chạy lần lượt trên dataset smoke đã khóa
(334/84). Không training dài, không xuất dataset, không Nano.

## Chuẩn bị — ĐÃ CHẠY

- Runner có chế độ `pilot` / `kiem-pilot`. Cấu hình pilot kế thừa nguyên cấu hình gốc theo sha256 và **chỉ đổi `epochs`**;
  tham số hiệu lực khác smoke ở bất kỳ khóa nào khác thì runner từ chối.
- **Sửa lỗi đo**: ultralytics `final_eval` gọi lại `on_fit_epoch_end` và từng ghi đè `val_s`/`metrics` epoch cuối. `val_s`
  smoke vòng 35 vì vậy gồm cả `final_eval`.
- **Test**:
  - lần 1: 24/30 — lỗi của chính phép đo (`args.half` đã đổi thành `quantize`); đã sửa, log giữ lại;
  - lần 2: **34/34 PASS**, gồm **A/B CPU**: lượt có đo trùng lượt không đo **từng bit**;
  - hồi quy detector: **92/92**.
- Khóa cấu hình commit **trước** khi train. Ngay trước mỗi lượt, cấu hình được so lại: khớp **46/46**.

## Pilot — ĐÃ CHẠY, ĐẠT cả hai

Exit 0. Đủ 2 epoch, 17 lần cập nhật/model, loss hữu hạn, checkpoint nạp lại đúng 5 lớp, đủ 7 phép đo; mọi phép đo trong lúc
train không đổi trạng thái (RNG CUDA bị tiêu thụ, đã khôi phục).

Đo bằng nguyên evaluator, **conf 0,001**, trên 84 ảnh dev:

| | Trước học | Epoch 1 (model / EMA) | Epoch 2 (model / EMA) | Checkpoint |
|:--|:--|:--|:--|:--|
| v8n: detection | **0** | 631 / 554 | 7.496 / 6.031 | 6.037 |
| v8n: car AP50 | 0 | 0,375 / 0,338 | 0,510 / 0,498 | 0,498 |
| v8n: điểm max/ảnh (trung vị, trước ngưỡng) | **0,00019** | 0,012 / 0,008 | 0,109 / 0,070 | 0,071 |
| 11n: detection | **0** | 18 / 3 | 3.798 / 3.606 | 3.603 |
| 11n: car AP50 | 0 | 0,005 / 0,005 | 0,413 / 0,392 | 0,393 |
| 11n: điểm max/ảnh (trung vị, trước ngưỡng) | **0,00028** | 0,0005 / 0,0004 | 0,055 / 0,041 | 0,041 |

- **Bằng chứng nạp**, khớp log ultralytics: 33 (v8n) / 48 (11n) tensor nhánh phân lớp `cv3` **khởi tạo mới**; chỉ 3 bias lớp
  được remap theo tên.
- **Warmup thực chạy**:
  - nw = 42, trọn epoch 1; LR/momentum/accumulate từng batch khớp công thức (0 lệch);
  - epoch 2 LR 0,00505;
  - cập nhật 12 + 5 lần.
- **Mosaic** tắt cả hai epoch, như mã dự đoán. Smoke thì có mosaic.
- **Thời gian**:
  - train epoch: v8n 50 / 39 s, 11n 58 / 46 s;
  - mỗi phép đo 8–15 s;
  - VRAM cả GPU tối đa 2.655 / 2.892 MiB;
  - RAM trống thấp nhất **7–59 MiB**.

## Kết luận

**Có bằng chứng**:

- Model 5 lớp **trước bước học** đã có điểm lớp gần 0, trước mọi bước tối ưu.
- Trong pilot, các bước học làm điểm lớp và AP **tăng**.

**Chưa xác định**:

- trạng thái trước học của chính lượt smoke vòng 35;
- vai trò của việc không warmup (smoke) — không có đối chứng;
- ảnh hưởng của mosaic;
- vì sao 11n tăng chậm hơn ở epoch 1;
- AP motorcycle/bicycle (không có GT).

Không xếp hạng model.

## Chờ người dùng duyệt

Xuất dataset đầy đủ · bước frame · số epoch · ngân sách training local. Đối chứng warmup chỉ khi được duyệt.

**Nên xem**: `src/pilot_chan_doan.py`, `src/test_pilot_chan_doan.py`, `logs/review37/01_*` (khóa), `06_*`, `07_*`.
