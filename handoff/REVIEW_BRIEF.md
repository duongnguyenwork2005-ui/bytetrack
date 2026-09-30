# REVIEW_BRIEF — vòng 30: ghi nhận nghiệm thu `bcb92f8`, kiểm kê BDD100K MOT train

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW30_CHI_TIET.md`**. Phạm vi: PC; không SSH hay đổi Nano v16, không training dài, không cloud, không
thêm dataset hay model. **Vòng này không sửa mã.**

## Ghi nhận nghiệm thu — ĐÃ CHẠY

- **Kết luận**: Codex `bcb92f8` **PASS WITH LIMITATIONS**, đóng R28-F1. Báo cáo lưu ở `handoff/CODEX_REVIEW_bcb92f8.md`.
- **Bằng chứng**: `review_evidence/CODEX_bcb92f8/`. `SHA256SUMS` khớp 11/11, cả với blob đã commit. ZIP gốc ở `archive/`.
- **Script Codex**: Codex **tự chạy nguyên** `REPRO_REVIEW_07615ba.py` trên target: **exit 0, `bug_count=0`**. Lượt NOT_RUN
  trên máy tác giả (vòng 29) **giữ nguyên làm lịch sử**; chỉ thêm ghi chú bổ sung vào `REVIEW29_CHI_TIET.md`. Không chạy lại
  smoke tổng hợp chỉ để đổi trạng thái log.
- **Base**: delta đã nghiệm thu lưu thành `archive/review_delta_bcb92f8.zip` (717.372 B, sha256 `5a931976…`, khớp). Base
  review chuyển `239b9c6` → **`bcb92f8`**.

## Kiểm kê dữ liệu thật — ĐÃ CHẠY, kết quả thiếu

Các nơi đã kiểm (không quét toàn ổ): `data/raw/bdd100k`, `data/raw`, `D:\LABOTARY` (nơi chứa gói val), Downloads (0 file
tải dở).

- **Có**: ảnh MOT **val** (200 video, 39.973 ảnh) và nhãn val COCO-video — nguồn đánh giá, không dùng làm train/dev.
- **Không có**: ảnh `images/track/train`, nhãn `labels/box_track_20/train`, và bản COCO-video train.
- **`--buoc kiem-nguon` → exit 2**, liệt kê đúng hai đường dẫn thiếu.
- **Cần đặt**:
  - ảnh: `data/raw/bdd100k/images/track/train/<video>/<video>-<7 số>.jpg` — giải nén gói *MOT 2020 Images — train* vào
    `data/raw/`;
  - nhãn: `data/raw/bdd100k/labels/box_track_20/train/<video>.json`.

  Ổ D: còn 126 GB; ước tính cần ~76 GB (zip + ảnh + dataset YOLO).
- **Không làm**: không dùng val hay fixture thay thế, không tự tải gói lớn.

## Bị chặn

Các bước sau phụ thuộc MOT train nên **chưa chạy**:

- kiểm ảnh–nhãn, bbox, 5 lớp, thống kê lớp/video;
- khóa split train-fit/dev theo video;
- kiểm trùng: BDD val theo tên + sha256, UA-DETRAC test chỉ tên video, train/dev theo sha256;
- xuất manifest v2;
- smoke thật YOLOv8n + YOLO11n;
- cập nhật ước lượng thời gian.

**Tập con smoke đã định**: 16 video train-fit + 4 video dev từ split đã khóa, bước 10 frame, seed 0.
- Theo cấu tạo tối đa khoảng 336 / 84 ảnh, trong giới hạn **≤ 1.000 / ≤ 200**.
- Chọn trước khi xem AP; hai model dùng cùng tập con.
- Vượt giới hạn thì dừng và hỏi.

## Ước lượng và quyết định

- Ngân sách 15 epoch (~27 h YOLOv8n + ~32 h YOLO11n, ±30%) **vẫn là ước tính từ fixture**, chưa cập nhật và **chưa duyệt**.
- Chưa chọn nơi chạy training dài.
- Test: **NOT_RUN** vì không có thay đổi mã.
