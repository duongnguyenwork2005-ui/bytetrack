# REVIEW_BRIEF — vòng 28: sửa R27-F1/F2/F3 (review `4937c85`)

<!-- BASE_TARGET -->

Review `handoff/CODEX_REVIEW_4937c85.md`: REQUEST CHANGES. **Base giữ `239b9c6`**. **Trạng thái: đã sửa, chờ Codex nghiệm
thu.** Chi tiết theo cấu trúc trước sửa → thay đổi → kiểm → giới hạn: **`handoff/REVIEW28_CHI_TIET.md`**.
Không training dài, không duyệt epoch/ngân sách, không thêm dataset hay model, không đụng Nano.

## Tái hiện trước sửa

Script Codex chạy trên bản sao `src/` + `configs/` của `4937c85` (hash khớp git): **exit 1**, 6/6 cờ `true`, số F3 trùng
JSON gốc. Ba file review được gửi sau khi bắt đầu vòng; trước đó Claude viết script tái dựng 7 ca và cũng được exit 1.

## Sửa

- **R27-F1**: `bdd_yolo_export.kiem_hop_dong_dataset()` là **hợp đồng dùng chung** cho exporter, runner và evaluator.
  - Kiểm đúng đường dẫn ultralytics resolve từ `data.yaml`: `train`/`val` = `images/train`/`images/dev`, `path` tuyệt
    đối; liệt kê ảnh và suy file nhãn như ultralytics.
  - Train/dev thuộc split đã khóa và không trùng video, tên hay nội dung.
  - Ảnh, nhãn và ignore khớp manifest; nhãn **rỗng** hợp lệ, nhãn **thiếu** là lỗi.
  - Runner gọi hợp đồng **trước** khi nạp model hay tạo thư mục; số ảnh báo cáo là số ảnh trainer sẽ dùng.
- **R27-F2**: **manifest phiên bản 2**.
  - Khóa sha256 từng ảnh, nhãn và ignore; phân biệt “không có vùng ignore” với “file ignore đã khai báo mà thiếu”.
  - Manifest v1 bị từ chối và phải xuất lại; không bổ sung hash.
  - Evaluator xác minh trước khi dự đoán và ghi danh tính dataset.
  - Cache nhãn ultralytics chỉ kiểm kích thước + đường dẫn, nên runner đối chiếu thêm **dữ liệu ultralytics thực nạp**
    trước vòng train.
- **R27-F3**: khi hòa IoU, chọn GT **sau** như `COCOeval.evaluateImg`; IoU tính như `bbIou`. **Giữ nguyên** ngưỡng và
  công thức AP. Ngưỡng 0,9 lệch 1 ulp so với pycocotools; điều này được ghi thành giới hạn.
- **Bổ sung theo ghi chú review**: exporter dừng khi ảnh dev trùng nội dung ảnh train-fit (probe phụ của Codex), không tự
  loại ảnh.

## Kiểm chứng (tự chạy)

- Script Codex trên mã sửa: **exit 0, 6/6 `false`**. F3: AP50 1,0 = pycocotools; AP50–95 lệch 1 ulp do thứ tự cộng số
  thực.
- `test_detector_5lop` **83/83**, cả tại HEAD `a86ed5c`. Nhóm mới:
  - G: entry point runner với **spy** tại ranh giới model/training — ca train=dev và ca thiếu nhãn bị exit 2, không nạp
    model, không train; ca hợp lệ tới đúng ranh giới;
  - H: toàn vẹn nội dung;
  - I: dữ liệu đã nạp, gồm một ca **ultralytics thật** dùng cache nhãn cũ cùng kích thước bị phát hiện;
  - E7–E9: **pycocotools thật**; thiếu thư viện thì SKIP, không tính PASS.
- Cùng test trên mã cũ: 58/83. 25 FAIL được phân loại trong phụ lục: hành vi sai thật, tính năng mới chưa có, và H9 không
  tính là bằng chứng.
- Suite liên quan: 83/83, 42/42, 11/11. Không chạy full suite vì module sửa không nằm trên đường chạy đã nghiệm thu.
- **Smoke GPU fixture** với runner mới:
  - dataset v1 của vòng 27 bị từ chối;
  - dataset v2 + YOLOv8n/YOLO11n 1 epoch **ĐẠT**; dữ liệu nạp 719/90 khớp, 0 lệch.

  Số batch 16/12 của vòng 27 là bằng chứng GPU cũ.

## Đính chính diễn giải

- Motorcycle val: 915 = 902 không crowd/ignore + 13 crowd.
- Kiểm trùng theo nội dung: BDD val có; UA-DETRAC test **chỉ tên video**; train/dev có từ vòng 28.
- Batch 16 hoàn tất nhưng chậm, không có OOM; paging là giả thuyết chưa đo.

## Giới hạn

- Chưa có BDD MOT train.
- Hash dựa vào `tom_tat.json`, không có ký số.
- Dataset gắn với vị trí thư mục.
- Chưa đối chiếu COCO trên dự đoán thật.
