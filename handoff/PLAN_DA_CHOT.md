# Kế hoạch đã chốt — từ vòng 27 (2026-09-29)

Nguồn: yêu cầu của người dùng sau khi Codex nghiệm thu `239b9c6` (**PASS WITH LIMITATIONS**, `handoff/CODEX_REVIEW_239b9c6.md`).
File này là phạm vi làm việc cho các vòng sau. Thay đổi phạm vi phải do người dùng quyết định.

## Thứ tự các mốc

| # | Mốc | Trạng thái (vòng 27) |
|:--|:--|:--|
| 1 | **Giao diện Nano** (HDMI + chuột + bàn phím, launcher, cửa sổ xem trực tiếp) | ✅ **Đã nghiệm thu với giới hạn** (review `239b9c6`): thao tác HDMI/zenity được tự động hóa, reviewer không bấm tay; nhánh SIGKILL chưa kiểm với zenity treo thật trên Nano; không camera; không vận hành nhiều giờ. Gói đang dùng: `~/nano_run/v16` |
| 2 | **Chuẩn bị BDD100K train** và **fine-tune YOLOv8n, YOLO11n** (5 lớp). **YOLOv8n COCO hiện tại** (`models/yolov8n.pt`, sha256 `f59b3d83…`) giữ làm **baseline** | 🔄 Vòng 27: mã chuẩn bị dữ liệu / huấn luyện / đánh giá + smoke đường chạy trên fixture. Vòng 28: sửa R27-F1/F2/F3 của review `4937c85` (hợp đồng dataset dùng chung, khóa nội dung, ghép hòa IoU như COCO); Codex `07615ba` xác nhận 3 finding đã hết nhưng mở R28-F1 (cache ảnh `.npy`). Vòng 29: sửa R28-F1 — Codex `bcb92f8` **PASS WITH LIMITATIONS**, đóng R28-F1; base review → `bcb92f8`. Vòng 30: kiểm lại dữ liệu — BDD100K MOT train **vẫn chưa có**, nên khóa split / xuất / smoke thật **bị chặn**. Vòng 31: **ảnh** MOT train tải + xác minh md5 + giải nén đủ 7/7 phần (1.400 video, 278.079 ảnh); **nhãn** MOT 2020 `box_track_20/train` **không lấy được** từ nguồn chính thức nào (ETH NXDOMAIN, nút Berkeley bị gỡ / 404); `kiem-nguon` exit 2, chỉ còn thiếu nhãn. Vòng 32–33: ứng viên nhãn Scalabel từ zip đóng gói lại trên HF (`jaffe03195/bdd100k_sot` @ `694197099692…`) được cách ly và kiểm chứng. Box/ID của native val khớp COCO val của dự án, nhưng lớp/ignore khác (183/50, UNKNOWN). Codex `4e71d32` **PASS WITH LIMITATIONS**; base review → `4e71d32`. Vòng 34: người dùng **chọn phương án C** — dùng có điều kiện snapshot bên thứ ba đã ghim revision và hash. Đã nhập nguyên byte 1.400 JSON train; hash 1.400/1.400 khớp; `kiem-nguon` **exit 0** (1.400 video, 278.079 frame). Chưa xác thực bằng archive gốc. **Tiếp theo** (chưa chạy): khóa split → xuất dataset → smoke 2 model trên dữ liệu thật; huấn luyện local trên PC; epoch / ngân sách training dài **chưa duyệt** |
| 3 | **Đánh giá detector riêng** (AP theo lớp, mAP50, mAP50–95, precision/recall tại ngưỡng vận hành), tạo **detection cache tái lập** | Evaluator 5 lớp đã có (vòng 27), chưa chạy trên dữ liệu thật |
| 4 | So sánh bộ lọc **CV / EKF / UKF / PF**, đối chứng **SORT / ByteTrack** bằng các phép so sánh có kiểm soát | Chưa bắt đầu trong plan mới |
| 5 | **Đánh giá cuối** trên **BDD100K val** và **UA-DETRAC test** | Chưa |
| 6 | Cuối cùng mới **tối ưu tốc độ hệ thống Nano**, so với cấu hình baseline đã ghi nhận | Chưa |

Lớp đã khóa: `0 car, 1 truck, 2 bus, 3 motorcycle, 4 bicycle` (`src/detector_classes.py`).

## Quy tắc phạm vi

- **Hỏi trước**: mọi đề xuất thêm model, dataset, thuật toán, ablation hoặc tính năng ngoài danh sách trên phải hỏi người
  dùng trước khi thực hiện. Ví dụ đang **chờ quyết định**: dùng gói BDD100K *detection* 100K (ảnh tĩnh, nhãn `det_20`) thay
  hoặc cùng gói MOT train. Gói này khác nguồn với MOT, nên chưa dùng.
- **Chọn bằng dev nội bộ**: checkpoint, số epoch và mọi tham số chỉ chọn bằng **dev nội bộ**. Dev này tách **theo video** từ
  split `train` chính thức của BDD100K MOT.
- **Hai tập đánh giá cuối không tham gia lựa chọn**: BDD100K val (bản COCO-video đã định danh) và UA-DETRAC test.
  - Không dùng hai tập này làm train hay dev.
  - Cả hai đã được xem trong các vòng nghiên cứu trước (thí nghiệm transfer BDD, mốc 2 UA-DETRAC), nên **không gọi là tập
    test hoàn toàn chưa từng quan sát**.
- **Phạm vi vòng 27**: chỉ chuẩn bị và smoke. Chưa chạy training dài, benchmark tracking đầy đủ hay tối ưu Nano.
- **Giữ nguyên**: hệ thống Nano v16, kết quả và lock cũ, file riêng của người dùng. Dataset, weights, video và cache lớn nằm
  ngoài gói delta.
