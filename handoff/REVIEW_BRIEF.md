# REVIEW_BRIEF — vòng 28: sửa R27-F1/F2/F3 (review `4937c85`)

<!-- BASE_TARGET -->

Review `handoff/CODEX_REVIEW_4937c85.md`: REQUEST CHANGES. **Base giữ `239b9c6`**. **Trạng thái: đã sửa, chờ Codex nghiệm
thu.** Chi tiết (trước sửa → thay đổi → kiểm → giới hạn): **`handoff/REVIEW28_CHI_TIET.md`**. Không training dài, không
duyệt epoch/ngân sách, không đụng Nano.

## Tái hiện trước sửa

Script Codex chạy trên bản sao `src/` + `configs/` của `4937c85` (hash khớp git): **exit 1**, 6/6 cờ `true`, số F3 trùng
JSON gốc.

## Sửa

- **R27-F1** — `bdd_yolo_export.kiem_hop_dong_dataset()` là **hợp đồng dùng chung** cho exporter, runner và evaluator.
  - Kiểm đúng đường dẫn ultralytics resolve từ `data.yaml`: `train`/`val` = `images/train`/`images/dev`, `path` tuyệt
    đối; liệt kê ảnh và suy file nhãn như ultralytics.
  - Split đã khóa; train/dev không trùng video, tên hay nội dung.
  - Nhãn **rỗng** hợp lệ, nhãn **thiếu** là lỗi.
  - Runner gọi hợp đồng **trước** khi nạp model hay tạo thư mục.
- **R27-F2** — **manifest phiên bản 2** khóa sha256 từng ảnh, nhãn và ignore.
  - Manifest v1 bị từ chối; không bổ sung hash.
  - Evaluator xác minh trước khi dự đoán và ghi danh tính dataset.
  - Cache nhãn ultralytics chỉ kiểm kích thước + đường dẫn, nên runner đối chiếu thêm **dữ liệu thực nạp** trước vòng train.
- **R27-F3** — khi hòa IoU, chọn GT **sau** như `COCOeval.evaluateImg`; IoU tính như `bbIou`. **Giữ nguyên** ngưỡng và
  công thức AP.
- **Theo probe phụ của Codex** — exporter dừng khi ảnh dev trùng nội dung ảnh train-fit; không tự loại ảnh.

## Kiểm chứng (tự chạy)

- Script Codex trên mã sửa: **exit 0, 6/6 `false`**. AP50 1,0 = pycocotools; AP50–95 lệch 1 ulp (thứ tự cộng số thực).
- `test_detector_5lop` **83/83**, cả tại HEAD `a86ed5c`. Nhóm mới:
  - G: entry point runner, **spy** tại ranh giới model/training; ca train=dev và thiếu nhãn bị exit 2, không nạp model;
  - H: toàn vẹn nội dung;
  - I: dữ liệu đã nạp, gồm ca **ultralytics thật** dùng cache cũ;
  - E7–E9: **pycocotools thật**; thiếu thư viện thì SKIP, không tính PASS.
- Trên mã cũ: 58/83. 25 FAIL được phân loại trong phụ lục.
- Suite liên quan: 83/83, 42/42, 11/11. Không chạy full suite vì module sửa không nằm trên đường đã nghiệm thu.
- **Smoke GPU fixture**: dataset v1 bị từ chối; v2 + YOLOv8n/YOLO11n 1 epoch **ĐẠT**, dữ liệu nạp 719/90 khớp. Số
  batch 16/12 là bằng chứng GPU cũ của vòng 27.

## Đính chính diễn giải

- Motorcycle val: 915 = 902 không crowd/ignore + 13 crowd.
- Kiểm trùng nội dung: BDD val có; UA-DETRAC test **chỉ tên video**; train/dev có từ vòng 28.
- Batch 16 hoàn tất nhưng chậm, không có OOM; paging chưa đo.

## Giới hạn

- Chưa có BDD MOT train.
- Hash dựa vào `tom_tat.json`, không có ký số.
- Dataset gắn với vị trí thư mục.
- Ngưỡng 0,9 lệch 1 ulp so với pycocotools.
- Chưa đối chiếu COCO trên dự đoán thật.
