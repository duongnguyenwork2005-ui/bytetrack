# REVIEW_BRIEF — vòng 38: xuất dataset YOLO đầy đủ (bước 5) và kiểm chứng

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW38_CHI_TIET.md`**.

**Phạm vi** (người dùng duyệt): xuất dataset đầy đủ từ BDD100K MOT train theo split đã khóa, 1 frame mỗi 5 frame; kiểm chứng;
dừng. Không training, không inference, không Nano.

## Ghi nhận nghiệm thu — ĐÃ CHẠY

- Codex `0c0a07a`: **PASS WITH LIMITATIONS**, đóng R35-F1/F2 về báo cáo. Bằng chứng khớp 91/91.
- Gói 967.546 B (`a2985214…`) khớp, đã lưu archive. **Base → `0c0a07a`**; 13 khai báo NPZ giữ nguyên.

## Xuất — ĐÃ CHẠY, exit 0

- `kiem-nguon` exit 0; split `b8a2848e…`, danh sách `a59a8d06…`, vân tay nhãn `d2024493…` đều khớp khóa.
- **Bước 5** theo `chon_frame`: chỉ số 0-based chia hết cho 5 trong khoảng frame có nhãn.
- Wall 777 s. **50.487 ảnh train-fit / 5.559 ảnh dev**, khớp số đếm vòng 35.
  - 278.079 frame nguồn = 56.046 chọn + 221.729 lệch bước + 304 ngoài khoảng có nhãn.
- **Kích thước**: 8,03 GB, 130.699 file. D: còn 54,83 GB.

## Kiểm chứng — ĐẠT

- **Hợp đồng v2**, chạy cả lúc xuất và chạy lại: **0 lỗi**.
  - Băm 56.046 ảnh, 56.046 nhãn, 18.600 file ignore; 0 `.npy`; đường dẫn đúng `images/train`, `images/dev`.
- **Kiểm độc lập**:
  - tập (split, video, frame) trong manifest **khớp từng phần tử** với tập tính lại từ nguồn nhãn;
  - 475.234 box hợp lệ; 0 video chung.
- **Kiểm trùng**:
  - nội dung train↔dev: 0 cặp;
  - BDD val: 0 trùng tên + 0 trùng nội dung;
  - UA-DETRAC test: **chỉ so tên video**.

| | train-fit | dev |
|:--|--:|--:|
| Video / ảnh / âm tính / có ignore | 1.261 / 50.487 / 686 / 17.040 | 139 / 5.559 / 95 / 1.560 |
| Box car / truck / bus | 383.387 / 27.358 / 10.708 | 42.211 / 2.971 / 984 |
| **Box motorcycle / bicycle** | 2.235 / 4.747 | **226 / 407** |
| **Ảnh có motorcycle / bicycle** | 1.822 / 3.598 | **205 / 334** |
| **Video có motorcycle / bicycle** | 230 / 397 | **24 / 44** |

- **Lỗi đo**: bộ đo RAM ngoài bám **nhầm** shell của lệnh Monitor chạy song song, nên RSS tiến trình của lần xuất không hợp
  lệ. RAM hệ thống còn trống vẫn hợp lệ (thấp nhất 449 MiB).

## Sẵn sàng training — chưa chạy

- Dry-run hai model (không train) exit 0: runner nhận đúng dataset; 6.311 batch / 789 bước tối ưu mỗi epoch.
- Kịch bản 56,64 h / 73,95 h không đổi (số ảnh trùng số đã dùng).
- RAM còn trống 1,26 GB.
- Lệnh training dự kiến ở phụ lục mục 5.

**Chờ duyệt**:

- số epoch (đề xuất 15), ngân sách, thời điểm;
- khởi chạy trong terminal riêng, vì mỗi lượt khoảng 26–30 h, dài hơn giới hạn tác vụ nền;
- có thêm resume không;
- đóng bớt ứng dụng;
- sửa bộ đo RAM.

**Nên xem**: `logs/review38/02_*`, `03_*`, `04_*`, `05_*`, `06_*`.
