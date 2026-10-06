# REVIEW_BRIEF — vòng 43: đánh giá detector trên BDD100K MOT val (3 checkpoint); UA-DETRAC chờ quyết định

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW43_CHI_TIET.md`**. Kế hoạch khóa trước khi chạy: `handoff/logs/review43/04_ke_hoach_danh_gia.md`.

**Phạm vi** (người dùng duyệt): đánh giá trên PC local cho baseline YOLOv8n COCO và `best.pt` của YOLOv8n / YOLO11n
(vòng 42), trên BDD val và UA-DETRAC test. Không train, không tuning theo test. Báo cáo rồi dừng.

## Review `4d54744` — đã ghi nhận

PASS WITH LIMITATIONS cho hồ sơ training. Bằng chứng 10/10. Base → **`4d54744`**, giữ 13 khai báo. R42-N1/N2/N3 chỉ
sửa câu chữ / bảng.

## Chuẩn bị (trước khi xem metric)

- Ba checkpoint kiểm trực tiếp từ file: sha256 khớp, nạp được, tensor hữu hạn, tên lớp đúng.
- Evaluator đã nghiệm thu không đọc được BDD val (đã tái hiện: exit 2). Thêm điểm vào **chỉ đọc GT** theo quy tắc
  exporter; dự đoán, ghép, AP dùng nguyên hàm evaluator. Test mới 19/19, gồm phép chứng minh GT trùng đường đã nghiệm
  thu; suite liên quan 92/92.
- Kế hoạch được khóa và commit trước lượt đầu; không sửa sau đó.

## Kết quả BDD val — ĐÃ CHẠY

3 lượt tuần tự, khởi chạy độc lập với phiên Claude. Mỗi lượt:
- exit 0; 39.973/39.973 ảnh;
- nhãn, tập video, tập ảnh, w/h và nội dung ảnh 39.973/39.973 khớp; sha256 checkpoint khớp.

Kiểm hợp lệ 39/39. Tái tính metric từ dự đoán đã lưu trùng tuyệt đối `ket_qua.json`.

| Model | mAP50 | mAP50–95 | Δ mAP50–95 (điểm) | P / R macro @ conf 0,25, IoU 0,5 |
|:--|--:|--:|--:|:--|
| `yolov8n_coco` (baseline) | 0,244 | 0,157 | — | 0,572 / 0,219 |
| `yolov8n_ft` | 0,384 | 0,251 | +9,4 | 0,613 / 0,347 |
| `yolo11n_ft` | 0,376 | 0,247 | +9,0 | 0,613 / 0,338 |

AP50–95 theo lớp (số GT):

| Lớp | baseline | `yolov8n_ft` | `yolo11n_ft` |
|:--|--:|--:|--:|
| car (341.436) | 0,337 | 0,518 | 0,514 |
| truck (27.343) | 0,135 | 0,268 | 0,263 |
| bus (9.087) | 0,201 | 0,298 | 0,280 |
| motorcycle (902) | 0,058 | 0,080 | 0,080 |
| bicycle (4.135) | 0,055 | 0,090 | 0,096 |

mAP là trung bình 5 lớp. Ba model dùng cùng ảnh, cùng GT, cùng thiết lập.

## UA-DETRAC test — CHƯA CHẠY

Ánh xạ lớp chưa chốt. Cần người dùng quyết định:
- `van` (38.519 box): vùng bỏ qua, hay GT car?
- `others` (16.915 box): vùng bỏ qua, hay GT truck?
- lớp báo AP (motorcycle / bicycle là N/A);
- 173 ảnh không có `<frame>`: ảnh âm tính, hay loại?
- `ignored_region`: dùng làm vùng bỏ qua theo IoA (đề xuất).

## Giới hạn

- Mỗi model một lượt, một seed; chênh lệch là mô tả, không kiểm định. Chênh 0,251 / 0,247 không đủ để chọn model.
- Baseline là model COCO ánh xạ theo tên lớp. Chênh lệch với baseline gộp cả khác biệt miền dữ liệu lẫn định nghĩa lớp.
- BDD val đã được xem ở các vòng trước, nhưng không dùng để chọn checkpoint. Kết quả dev nội bộ (vòng 42) để riêng.
- P/R tại cùng conf 0,25 phụ thuộc độ hiệu chỉnh confidence của từng model.
- Log trạng thái trước lượt `yolov8n_ft` được bù sau khi khởi chạy. Không đóng gói NPZ, checkpoint, dữ liệu; chỉ có hash.

## Trạng thái

BDD val xong. **Chưa nghiệm thu.** UA-DETRAC chờ quyết định ánh xạ. So sánh bộ lọc, tracking và Nano chưa làm.

**Nên xem**: `logs/review43/04_*`, `ket_qua/bang_ket_qua.md`, `ket_qua/kiem_hop_le.json`, `11_*`.
