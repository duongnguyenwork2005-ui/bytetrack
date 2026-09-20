# Tốc độ — tách **tracker từ cache** khỏi **toàn pipeline**

**Chỉ là số đo mô tả của chính các lượt đã chạy** (máy còn chạy việc khác, không lặp có kiểm soát) ⇒ **không** đủ để xếp hạng hiệu năng.

Ba cột **không** phải ba pha độc lập: `update` = **tổng** lời gọi `tracker.update()`; `predict` = chỉ `multi_predict`; phần còn lại = `update − predict`, **bao gồm** association, bước cập nhật của bộ lọc và quản lý vòng đời.

## 1. Tracker chạy TỪ CACHE (không gồm detector)

Đo từ cache **loại** chi phí **detector**. Mục 2 ghép thêm chi phí detector đã ghi lúc dựng cache để có con số **toàn pipeline**.

### `c25`

| tracker | update ms/frame | trong đó predict ms | còn lại ms |
|:--|--:|--:|--:|
| cv | 0.993 | 0.109 | 0.884 |
| ekf | 1.228 | 0.238 | 0.989 |
| ukf | 2.924 | 1.587 | 1.337 |
| pf500 | 5.150 | 2.265 | 2.885 |

### `c10`

| tracker | update ms/frame | trong đó predict ms | còn lại ms |
|:--|--:|--:|--:|
| cv | 1.138 | 0.108 | 1.030 |
| ekf | 1.322 | 0.229 | 1.093 |
| ukf | 3.021 | 1.556 | 1.465 |
| pf500 | 5.178 | 2.209 | 2.969 |


## 2. Toàn pipeline (detector + tracker)

⚠️ Đây là **tổng của hai pha đo RIÊNG** (detector đo lúc dựng cache, tracker đo lúc chạy ma trận), **không phải** một phép đo end-to-end. Chạy thật một lượt liền mạch có thể khác (chồng lấn I/O, giữ GPU nóng, không ghi/đọc cache).

- Detector (`yolov8n`, conf 0,05, **80 lớp COCO**, `max_det` 3000): **22.9 ms/frame** — 15.3 phút cho 39,973 frame.
- Số này **gom cả đọc + giải mã JPEG**, không phải riêng suy luận.
- Detector ở đây **dư thừa có chủ ý** cho mục đích cache (giữ 80 lớp để mọi điều kiện dùng chung một cache); một triển khai thật chỉ cần 5 lớp và `max_det` 300 sẽ **nhanh hơn**. Vì vậy con số này là **cận trên**, không dùng để suy ra FPS triển khai.

### `c25`

| tracker | detector ms | tracker ms | tổng ms/frame | FPS suy ra | tracker chiếm |
|:--|--:|--:|--:|--:|--:|
| cv | 22.9 | 0.993 | 23.9 | 41.8 | 4.1 % |
| ekf | 22.9 | 1.228 | 24.2 | 41.4 | 5.1 % |
| ukf | 22.9 | 2.924 | 25.9 | 38.7 | 11.3 % |
| pf500 | 22.9 | 5.150 | 28.1 | 35.6 | 18.3 % |

### `c10`

| tracker | detector ms | tracker ms | tổng ms/frame | FPS suy ra | tracker chiếm |
|:--|--:|--:|--:|--:|--:|
| cv | 22.9 | 1.138 | 24.1 | 41.5 | 4.7 % |
| ekf | 22.9 | 1.322 | 24.3 | 41.2 | 5.4 % |
| ukf | 22.9 | 3.021 | 26.0 | 38.5 | 11.6 % |
| pf500 | 22.9 | 5.178 | 28.1 | 35.6 | 18.4 % |

> **Đọc đúng mức**: với cấu hình cache này, chi phí **detector áp đảo**; chênh lệch giữa các bộ lọc gần như không đổi FPS toàn pipeline trên máy này. Điều này **không** suy ra được cho Jetson Nano: tỷ lệ CPU/GPU khác hẳn, chưa đo.

