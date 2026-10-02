# REVIEW_BRIEF — vòng 35: khóa split, smoke 2 model trên dữ liệu thật, dự toán training local

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW35_CHI_TIET.md`**.

**Phạm vi**: nghiệm thu `ccd801f` → khóa split → tập con smoke → smoke → dự toán, rồi **dừng**.

- Training dài, epoch và ngân sách **chưa duyệt, chưa chạy**.
- `src/` không đổi.
- Không Nano, không tracking.

## Đã chạy

| Bước | Kết quả |
|:--|:--|
| Nghiệm thu `ccd801f` | PASS WITH LIMITATIONS; bằng chứng 15/15 khớp; **base → `ccd801f`** |
| Khóa split (`4237149`, trước mọi train/AP) | Theo video, dev 10%, seed 0, phân tầng motorcycle/bicycle. **train-fit 1.261 video / dev 139**, 0 video chung. Trùng tên với BDD val / UA-DETRAC test: 0 |
| Giữ byte split (`aeed110`) | `text=auto` từng đổi blob CRLF→LF, không kiểm được sha256 `b8a2848e…`; thêm `-text`, nội dung không đổi |
| Tập con smoke | **334 / 84 ảnh**; hợp đồng v2 0 lỗi; trùng nội dung train/dev và BDD val: 0 |
| Smoke lần 1 | **FAIL trước vòng train**: bộ gắn mốc giờ **của tôi** chết vì ký tự "━", pipe đứt, runner báo `OSError [Errno 22]`. Đã lưu log |
| Sửa công cụ đo | Regression: mã cũ FAIL (tái hiện errno 22), mã mới 12/12 |
| Smoke lần 2 | **PASS cả hai** (exit 0). Nạp 334/84 ảnh khớp dataset đã xác minh; checkpoint nạp lại đúng 5 lớp |

| Smoke lần 2 | YOLOv8n | YOLO11n |
|:--|--:|--:|
| Khởi tạo / train / val | 40 / 51 / 34 s | 36 / 63 / 9 s |
| s/batch ổn định | 0,991 | 1,131 |
| torch reserved / cả GPU | 2.134 / 2.624 MiB | 2.344 / 2.840 MiB |
| dev smoke mAP50 | 0,098 | **0** |

AP chỉ để kiểm pipeline, không so sánh model.

**YOLO11n ra 0 detection**:

- Điểm lớp lớn nhất giảm 0,80 → 0,0003, dưới ngưỡng 0,001.
- Ở conf 1e-6, car AP50 = 0,233, nên **không phải lỗi đường ống**.
- Smoke 1 epoch không có warmup (quy tắc ultralytics), nên không phản ánh training 15 epoch.

## Dự toán (ước tính)

Dataset đầy đủ bước 5 (đếm, chưa xuất): **50.487 / 5.559 ảnh**, ~8,5 GB.

| Phương án | Tổng GPU, hai model tuần tự |
|:--|:--|
| **Bước 5, 15 epoch (đề xuất)** | **~57 h** (cao ~74 h) |
| Bước 10, 15 epoch | ~29 h (cao ~38 h) |

- VRAM ≤ ~3,1 / 4 GB.
- **RAM là rủi ro chính**: còn trống 16–146 MiB khi các ứng dụng khác đang mở.

## Cần quyết định

1. Epoch, bước frame và ngân sách.
2. Xuất dataset đầy đủ.
3. Có chạy pilot ngắn có warmup trước không (lượt thêm, cần duyệt).
4. Đóng bớt ứng dụng khi train dài.

## Giới hạn

- Test suite NOT_RUN vì `src/` không đổi.
- Dự toán ngoại suy từ một epoch 42 batch.
- Nhãn là snapshot bên thứ ba (phương án C).

**Nên xem**: `logs/review35/09_*`, `08_*` → `10b/10c_*`, `12_ket_qua_runner_*_lan2.json`, `13*_*`, `15_du_toan_training.py`.
