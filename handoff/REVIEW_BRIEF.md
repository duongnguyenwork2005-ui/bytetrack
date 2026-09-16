# REVIEW_BRIEF — chốt mốc 2, chuẩn bị mốc 3

<!-- BASE_TARGET -->

Base là snapshot bạn đã review (**PASS WITH LIMITATIONS**, không còn lỗi chặn —
`handoff/CODEX_REVIEW_a8ecaf6.md`). Đợt này **không có thay đổi logic nào** ở pipeline: chỉ chốt báo cáo,
kiểm kê dữ liệu và bổ sung giao thức đo.

## Đã làm

| # | Việc | Kết quả | File |
|:--|:--|:--|:--|
| 1 | **Chốt báo cáo thực nghiệm mốc 2**, số lấy trực tiếp từ file kết quả gốc | Nhóm A tách **theo từng điều kiện confidence**; Nhóm B dùng `_fix1`, ghi rõ **thành phần thay đổi** của từng cặp; Nhóm C báo cả `preserved` tăng lẫn HOTA giảm; tách dev/test; tách **tốc độ tracker-từ-cache** với **tốc độ toàn pipeline** | `handoff/MOC2_FINAL_SUMMARY.md` |
| 2 | **Kiểm kê BDD100K** | **Chưa có dữ liệu trên máy.** `check_bdd100k.py --part val` → exit 1, fail-loud đúng đường dẫn cần. Tìm toàn ổ `C:`/`D:`: thư mục duy nhất khớp là `D:\LABOTARY\ByteTrack + EKF + Kitty\datasets\BDD100K` — **rỗng**. Ghi 2 gói cần tải, vị trí đặt, 4 lệnh kiểm tra theo thứ tự | `handoff/BDD100K_INVENTORY.md` |
| 3 | **Smoke BDD100K** | **KHÔNG chạy** — thiếu dữ liệu. Fixture `bdd100k_mini` **không** dùng thay thế | — |
| 4 | **Kế hoạch đánh giá BDD** | Hai thí nghiệm tách biệt; **không trùng sequence theo cấu tạo** (tuning từ `train`, đánh giá trên `val`), vẫn kiểm giao hai manifest trước khi chạy | `BDD100K_INVENTORY.md` §6 |
| 5 | **Nano** | Thêm **giao thức đo CV ↔ EKF cùng điều kiện**: đo tracker từ **cùng detection cache**, xen kẽ A/B/A/B ≥ 5 cặp, trung vị + IQR của hiệu số **theo cặp**, ghi `nvpmodel`/clocks/nhiệt độ mỗi lượt, loại lượt throttling. Lý do: trên PC chênh lệch CV↔EKF là **0,24 ms** còn nhiễu detector **2,2 ms** (~9×), đã làm đảo thứ tự fps end-to-end | `scripts/nano/README.md` §4c |

## Kết quả thực nghiệm

**Không số liệu nào thay đổi.** Không chạy lại thí nghiệm, không tuning, không đụng cấu hình khóa. Hai lệnh
read-only chạy lại để chốt số hiện tại: cache **3 `ok` + 53 `legacy` + 0 `mismatch`**; run **21 `OK` +
113 `LEGACY` + 0 nghi vấn** (134 thư mục).

## Hai chỗ tự sửa vì lệch file nguồn

1. `scripts/nano/README.md` ghi `yolo11n` là ứng viên **duy nhất** qua kiểm tra op TRT 8.2. Theo
   `results/phase2/export/*.json`: **cả `yolov8n` và `yolo11n`** đều qua; chỉ `yolo26n` bị loại (op `Mod`).
2. Bỏ con số chưa kiểm chứng được ("IoU trung bình 0,06 sau trung vị 25 bước") khỏi báo cáo chốt; thay bằng
   thứ có trong file nguồn: `prediction_drift` là nguyên nhân số 1 ở tầng `full`, c25 — **cv 11/22,
   ekf 13/23, ukf 14/23** — tiêu chí gán nhãn **IoU(dự đoán, det) < 0,2**.

## Còn thiếu (chặn mốc 3)

- **Dữ liệu BDD100K-MOT**: *MOT 2020 Images* + *MOT 2020 Labels* (bdd-data.berkeley.edu) → `data/raw/bdd100k/`.
- **Jetson Nano**: chưa kết nối ⇒ chưa có số nào trên thiết bị.
- **Vòng lặp ByteTrack không phụ thuộc torch**: chưa làm; thiếu nó thì trên Nano chỉ đo được latency detector.
- Giới hạn provenance legacy giữ nguyên, không tạo bằng chứng hồi tố.

## Cần bạn kiểm kỹ

1. `MOC2_FINAL_SUMMARY.md` §2 và §4: mức bằng chứng có bị phát biểu quá không — câu "lợi ích EKF nhỏ và phụ
   thuộc điều kiện", và câu về C2 (`preserved` tăng ≠ chất lượng tăng).
2. §6: tách 6a/6b đã đủ rõ để không ai so chéo hai bảng chưa (profile PF dùng **N=300**, bảng kết quả **N=500**).
3. `BDD100K_INVENTORY.md` §5: quy ước `occlusion_ratio = 0,5` cho cờ `occluded`, và việc **không** suy diễn
   gap thành che hoàn toàn.

## Thứ tự đọc

`REVIEW_BRIEF.md` → `patch/` → `files/handoff/MOC2_FINAL_SUMMARY.md` + `files/handoff/BDD100K_INVENTORY.md`
→ `logs/` → `DEPENDENCIES.md`. Thiếu file nào, báo lại — sẽ bổ sung ở gói kế tiếp.
