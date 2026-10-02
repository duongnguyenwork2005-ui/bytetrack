# REVIEW_BRIEF — vòng 34: ghi nhận nghiệm thu `4e71d32`; nhập nhãn train theo phương án C

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW34_CHI_TIET.md`**.

**Phạm vi**: ghi nhận nghiệm thu, nhập nguyên byte 1.400 JSON native train, chạy `kiem-nguon`. **Không** khóa split, xuất,
smoke hay training. `src/` và `configs/` không đổi. Không SSH Nano.

## Ghi nhận nghiệm thu — ĐÃ CHẠY

- Codex `4e71d32`: **PASS WITH LIMITATIONS**, đóng R32-F1 và R32-F2. Báo cáo ở `handoff/CODEX_REVIEW_4e71d32.md`.
- Bằng chứng ở `review_evidence/CODEX_4e71d32/`: `SHA256SUMS` khớp 56/56, cả trên blob đã commit.
- Gói đã nghiệm thu: 1.156.972 B, sha256 `a34037a1…`, khớp báo cáo; lưu ở `archive/review_delta_4e71d32.zip`.
- **Base review: `bcb92f8` → `4e71d32`**; 13 khai báo NPZ được mang sang.

## Nhập nhãn train — ĐÃ CHẠY

Người dùng **chọn phương án C**: dùng có điều kiện snapshot bên thứ ba **đã ghim**, gồm HF `jaffe03195/bdd100k_sot`, revision
`694197099692fe1254e93215e73e68ece8b1f41f` và hash từng file trong manifest vòng 32 `a2e416f2…`.

| Bước | Kết quả |
|:--|:--|
| Kiểm trước (`nhap_nhan_train.py --chi-kiem`, fixture 9/9) | Bản cách ly có **1.400** JSON train: 0 thiếu, 0 thừa, 0 lệch byte/sha256. COCO val `074ff795…` khớp. Thư mục đích chưa có |
| Nhập nguyên byte vào `data/raw/bdd100k/labels/box_track_20/train/` | Ghi mới 1.400 file, qua file tạm rồi đổi tên |
| Kiểm sau | Đích **1.400/1.400** khớp sha256, 0 file lạ. Bản cách ly 1.602/1.602 vẫn khớp manifest. COCO val không đổi |
| Provenance | Để ở handoff, không đặt trong thư mục nhãn. Manifest nhập: `logs/review34/03_manifest_nhap_nhan_train.tsv` (`c5a548bc…`) |

Chỉ nhập native train. Native val, hai COCO phụ của mirror và COCO val của dự án giữ nguyên.

## `kiem-nguon` đường chính — ĐÃ CHẠY

`python src/prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml --buoc kiem-nguon` → **exit 0**.

- `scalabel`; **1.400 video nhãn = 1.400 video ảnh**; **278.079 frame**.
- 0 ảnh thiếu, 0 video lệch; **1.400 video dùng được**.

**Phân biệt**:

- Đạt `kiem-nguon` nghĩa là **đủ đầu vào để đi tiếp**.
- Native val so với COCO val **vẫn lệch 183 lớp và 50 ngoại lệ ignore**; nguyên nhân UNKNOWN.
- Không điều nào ở trên xác thực nhãn là bản phát hành chính thức.

## Giới hạn còn tồn tại

- **Chưa xác thực bằng archive gốc** (md5 `6be40e0c…`); train chưa có tham chiếu độc lập theo từng annotation.
- Lớp/ignore khác giữa native và COCO, chưa biết bên nào đúng hơn. **Giữ COCO val** cho đánh giá đã bàn giao và các kết quả
  lịch sử.
- Nếu sau này có archive gốc: đối chiếu đường dẫn và sha256 từng file, báo khác biệt **trước** khi thay snapshot.
- Test suite: **NOT_RUN** vì không đổi mã dự án. Script nhập nằm trong `logs/review34/`, kiểm thử fixture 9/9.

## Bước tiếp theo — CHƯA CHẠY

Khóa split → xuất dataset → smoke YOLOv8n và YOLO11n trên dữ liệu thật theo `DETECTOR_TRAIN_PLAN.md`; huấn luyện local trên
PC. **Training dài, epoch và ngân sách chưa được duyệt.**

## Đề nghị reviewer xem

1. `logs/review34/nhap_nhan_train.py`: kiểm trước khi ghi, xử lý xung đột (không ghi đè, không xóa), kiểm sau khi nhập.
2. Các log `02`–`05` và manifest nhập; tính nhất quán giữa `PLAN_DA_CHOT`, `BDD100K_INVENTORY` và `CODEX_HANDOFF`.
