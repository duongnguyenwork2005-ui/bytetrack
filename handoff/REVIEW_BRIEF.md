# REVIEW_BRIEF — vòng 45: ghi nhận `3e94b21`; chuẩn bị quyết định D1–D5 cho UA-DETRAC (chưa chạy GPU)

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW45_CHI_TIET.md`**. Dự thảo giao thức (**chưa khóa, chưa chạy**):
`handoff/logs/review45/06_du_thao_giao_thuc_detrac.md`.

## Review `3e94b21` — đã ghi nhận

- PASS WITH LIMITATIONS, đóng R43-B1 / R43-N1, trong phạm vi mã / fixture / hồ sơ. Reviewer không chạy lại inference
  dữ liệu thật.
- Bằng chứng 43/43. Gói `9bb1bbab…` được lưu archive **trước**, rồi base mới chuyển → **`3e94b21`** (giữ 13 khai báo).
- Không chạy lại ba lượt BDD.

## Dữ kiện UA-DETRAC test (chỉ đọc; khớp số vòng 43)

- 40 video, 56.167 `<frame>`, 56.340 ảnh (960×540), 173 ảnh không có `<frame>`.
- Box: car 548.555, bus 71.785, van 38.519 (126 track), others 16.915 (chỉ 31 track). Không có truck / xe máy / xe đạp.
- **XML không có frame rỗng nào**, nên thiếu `<frame>` nghĩa là "không có annotation", không phải "không có xe".
  MVI_39271 f78 có xe ngoài vùng bỏ qua mà không có nhãn.
- **Xem bằng mắt** (chủ quan, theo track):
  - others: 21 xe tải chở hàng, 7 xe cứu hộ (54 % box), 1 giống xe buýt, 2 không rõ;
  - van: phần lớn kiểu xe con, ~2/48 là minibus.
- GT car / bus có IoA ≥ 0,5 với `ignored_region`: 35.248 / 5.587.
- Toolkit chính thức (đọc mã): hợp vùng, ngưỡng cố định 0,5, loại detection trước khi chấm.

## Đề xuất chờ người dùng duyệt

| # | Đề xuất |
|:--|:--|
| D1 van | vùng bỏ qua cho mọi lớp |
| D2 others | vùng bỏ qua (không AP truck) |
| D3 | AP car, bus; mAP (car, bus); truck / xe máy / xe đạp N/A, không chấm |
| D4 | loại 173 ảnh không có `<frame>`, ghi danh sách |
| D5 | giữ nguyên quy tắc ignore BDD (IoA max từng vùng ≥ ngưỡng IoU, chỉ cho detection chưa ghép; GT trong vùng vẫn là GT; không tô xám ảnh) |

## Giới hạn

Phân loại bằng mắt là chủ quan. Chưa có dự đoán, nên ảnh hưởng chỉ ở phía GT / quy tắc. Không sửa mã, không chạy test.

**Nên xem**: `REVIEW45_CHI_TIET.md` mục 3, `logs/review45/06_*`, `01_*`, `03_*`, `phan_tich/`.
