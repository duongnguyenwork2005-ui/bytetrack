# REVIEW_BRIEF — vòng 27: chuẩn bị dữ liệu và đường fine-tune detector 5 lớp

<!-- BASE_TARGET -->

Review `handoff/CODEX_REVIEW_239b9c6.md` kết luận PASS WITH LIMITATIONS; N25-F1/F2 đã đóng.
- Gói đã review được lưu thành `handoff/archive/review_delta_239b9c6.zip` (11.188.466 B, sha256 `854b09a3…`, khớp).
- Base review chuyển sang **`239b9c6`**.
- Phạm vi mới ghi ở `handoff/PLAN_DA_CHOT.md`. Chi tiết vòng này: **`handoff/REVIEW27_CHI_TIET.md`**.
- Kế hoạch training để review: **`handoff/DETECTOR_TRAIN_PLAN.md`**.

## Kiểm kê dữ liệu (chạy thật, `src/kiem_ke_bdd100k.py`)

- **Máy chỉ có BDD100K MOT val**: 200 video, 39.973 ảnh 1280×720, nhãn COCO-video 465.512 annotation.
  - motorcycle: 902 box / 29 video.
  - Có 2 frame val trùng byte.
- **Không có** ảnh hoặc nhãn MOT **train**, cũng không có gói detection 100K.
- Vì vậy **khóa split thật, xuất dataset thật, smoke trên dữ liệu thật và thống kê train đều bị chặn**. Gói cần tải và
  đường dẫn đặt: `BDD100K_INVENTORY.md`, mục “Vòng 27”. Không thay bằng UA-DETRAC hay BDD val.

## Mã mới / sửa

- **Dataset** (`bdd_yolo_export.py`; `prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml`):
  - chỉ lấy BDD MOT train;
  - chế độ cũ không còn mặc định trộn UA-DETRAC; thiếu BDD thì exit 2 và không tạo thư mục.

  Quy tắc:
  - chia theo video: 10% dev, seed 0, phân tầng theo motorcycle/bicycle; split được khóa kèm hash;
  - lấy 1 frame mỗi 5 frame;
  - ảnh âm tính: giữ nếu nằm trong khoảng có nhãn, loại đuôi video không nhãn;
  - crowd/ignore tô xám nhưng chừa pixel của box có nhãn;
  - tọa độ theo kích thước ảnh thật;
  - kiểm trùng BDD val / UA-DETRAC test theo tên video, tên ảnh và sha256, **trước** khi ghi.

  `val` trong `data.yaml` là **dev nội bộ**. Nhãn Scalabel và COCO-video cho kết quả xuất giống nhau từng byte.
- **Evaluator 5 lớp** (`eval_detector_5lop.py`), dùng cùng quy tắc cho YOLOv8n COCO / YOLOv8n FT / YOLO11n FT:
  - ánh xạ lớp theo **tên** đọc từ checkpoint;
  - AP COCO theo lớp;
  - P/R tại conf 0,25 và IoU 0,5, theo lớp.

  `eval_detector.py` cũ (1 lớp, ảnh 960×540 cố định) chỉ giữ để tái lập Giai đoạn C.
- **Runner** (`train_detector.py`, `configs/phase2/detector/train_bdd5.yaml`):
  - `--dry-run` không còn train;
  - không ghi đè lượt cũ;
  - cấm giá trị `auto`;
  - ghi giá trị ultralytics resolve, VRAM/RAM/GPU, sha256 checkpoint, kết quả nạp lại, đánh giá dev;
  - training dài bắt buộc `--xac-nhan-train-dai`.

## Kiểm chứng

- **`test_detector_5lop` 52/52.** Lần đầu 47/52 (log giữ lại) — 5 FAIL là lỗi thật của mã mới:
  - `round()` kiểu banker;
  - `data.yaml` trỏ vào thư mục tạm.

  Test có đối chiếu AP với pycocotools (chênh 0), kiểm dữ liệu thiếu, trùng tập đánh giá, split sửa tay, ghi đè.
- Chạy cùng hai suite liên quan: `test_bdd100k_adapter` 42/42, `test_bdd100k_coco` 11/11.
- **Smoke GPU trên fixture** (dữ liệu tổng hợp 1280×720, chạy đúng luồng chia → xuất → train → nạp lại → đánh giá):
  - cả hai model ĐẠT 1 epoch với **đúng cấu hình đề xuất**;
  - batch 16 tràn VRAM, chậm 2–4 lần → **batch 8**;
  - YOLOv8n 8,27 ảnh/s, YOLO11n 6,91 ảnh/s; GPU bận 94–95%.

  Số AP trên fixture không mang ý nghĩa chất lượng.

## Cần review / quyết định

- **Ngân sách**: 15 epoch × bước 5 frame, ước ~27 h (YOLOv8n) + ~32 h (YOLO11n), ±30%. Đây là ước từ fixture và số video
  BDD công bố; phải đo lại bằng smoke trên dữ liệu thật trước khi train dài.
- Người dùng có muốn dùng gói detection 100K không — chưa dùng, chờ hỏi.
- Dev để tinh chỉnh tracker ở mốc 4 phải nằm trong dev nội bộ này.

## Giới hạn

- Chưa có training dài và chưa có số liệu detector thật.
- RAM 7,8 GB bị các ứng dụng khác chiếm; RAM trống tối thiểu khi train chỉ vài chục MiB.
- Không đụng Nano v16.
- Full suite không chạy lại: đường chạy đã nghiệm thu không bị sửa.
