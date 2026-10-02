# REVIEW_BRIEF — vòng 33: sửa R32-F1 và R32-F2 của review `c4d64a1`; nhãn **vẫn cách ly**

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW33_CHI_TIET.md`**.

**Phạm vi**: sửa bộ kiểm, đối chiếu lại nhãn cách ly, đính chính báo cáo. **Không** nhập nhãn, khóa split, xuất, smoke hay
training. `src/` và `configs/` không đổi. Base giữ **`bcb92f8`**. Bằng chứng reviewer lưu ở
`review_evidence/CODEX_c4d64a1/`, `SHA256SUMS` khớp 42/42.

## R32-F1 — diễn giải lớp/ignore: ĐÃ SỬA

- Chạy lại **nguyên** `audit_real_labels.py` trên PC. Kết quả trùng bản reviewer; `class_mismatches.csv` và
  `ignore_exceptions.csv` **trùng từng byte**.
- 465.512 cặp: **0 lệch box, 0 xung đột ID hai chiều**.
- **183 lệch lớp** ngoài ba ánh xạ (46 track, 40 video). Hàng nhiều nhất: truck→car 73, pedestrian→car 62.
- **50 ngoại lệ ignore**: 24 other person ignore=0, 21 pedestrian ignore=1, 5 other vehicle ignore=0. Đây là hai phép kiểm,
  không cộng.
- Nguyên nhân **UNKNOWN**: repo không có mã chuyển đổi hay ghi chép nguồn của COCO val.
- **Quan sát**: COCO của chính người đóng gói HF trùng native ở cả 183 vị trí và dùng ignore=0 cho mọi annotation. Điều này
  không cho biết bên nào đúng.
- **Đính chính** ở `logs/review32/14_nguon_nhan_vong32.md` (khối “Đính chính vòng 33”) và `BDD100K_INVENTORY.md`:
  - chỉ kết luận **box và ID khớp**;
  - lớp/ignore còn khác;
  - thống kê so với bài báo chỉ **tương đồng tổng hợp, có sai lệch**.

## R32-F2 — bộ kiểm báo đạt khi thiếu bằng chứng: ĐÃ SỬA

Sửa `logs/review32/kiem_nhan_ung_vien.py`:

- **Thiếu đầu vào → exit 2**: root, split bắt buộc, bộ nhãn rỗng, thư mục ảnh, file COCO được yêu cầu. Kiểm một phần phải
  chọn rõ bằng `--split` hoặc `--khong-doi-chieu-coco`, phần còn lại ghi `NOT_RUN`.
- **Schema → exit 1**:
  - `labels` sai kiểu (dict/chuỗi/số) là lỗi; `[]` hợp lệ; `null` hoặc thiếu khóa đếm riêng;
  - tọa độ phải hữu hạn, không NaN/Inf, không boolean;
  - ID trùng trong frame, frame trùng.
- **COCO**:
  - khóa trùng bị bắt trước khi đưa vào dictionary;
  - song ánh ID hai chiều theo từng video;
  - lớp/ignore theo định nghĩa reviewer, thêm iscrowd;
  - CSV từng hàng.
- **Trạng thái tách riêng**: schema `DAT`/`LOI`/`NOT_RUN`; COCO `MATCH`/`DIFFERENCES`/`NOT_RUN`/`LOI_DAU_VAO`. **Khác biệt
  COCO → exit 3**, nên exit 0 không bao giờ có nghĩa “có khác biệt”.

**Kiểm chứng**:

| Phép chạy | Kết quả |
|:--|:--|
| Script Codex (giữ nguyên) | mã cũ exit 1, `bug_count=5` → mã sửa **exit 0, `bug_count=0`**. Đối chứng sạch đạt; quan sát COCO exit 2/3 |
| Regression mới `logs/review33/test_kiem_nhan_ung_vien.py` | bản sửa **30/30**; bản `c4d64a1` 3/30 |
| Test cũ vòng 32 | 3/3 |
| Nhãn thật (1.400 train + 200 val) | schema/độ đầy đủ **DAT**, COCO **DIFFERENCES** (exit 3). Dữ liệu thật: 0 NaN/Inf, 0 ID trùng, 0 sai kiểu, 0 track đổi lớp |
| Hash nhãn cách ly trước/sau | 1.602/1.602 khớp manifest `a2e416f2…`; COCO `074ff795…` không đổi |
| `kiem-nguon` | bản cách ly exit 0; đường chính **exit 2** vì nhãn chưa nhập |

## Trạng thái nguồn và quyết định

- **Nguồn**: HF `jaffe03195/bdd100k_sot` @ `694197099692…`, hash từng file đã ghim.
- **Chưa xác thực bằng archive gốc** (md5 `6be40e0c…`). Nhãn **vẫn cách ly**.
- **Phương án C** đang **chờ người dùng chốt**: nhập nguyên byte 1.400 JSON train, giữ COCO val cho đánh giá.
- Test suite tracking/Nano: **NOT_RUN**, vì đường chạy đó không đổi.

## Đề nghị reviewer xem

1. Chính sách exit 0/1/2/3 và phần trạng thái có đủ để không đọc nhầm exit 0 thành “khớp hết” không.
2. Định nghĩa `labels` `null`/thiếu khóa: hợp lệ nhưng đếm riêng.
3. Cách đính chính trong `14_nguon_nhan_vong32.md`: sửa tại chỗ, bản gốc vẫn nằm ở `c4d64a1`.
