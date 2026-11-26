# R66 Sherbrooke — gói smoke 128 frame để review (02/11/2026)

**Đính chính sau review độc lập.** Đây là hồ sơ của smoke ban đầu. Review yêu cầu sửa B1–B7 trước khi mở full. CV có **258–330 FP**. HOTA của CV cao hơn trung bình PF500 **9,14–12,02 điểm**. Kiểm tái tổ hợp ban đầu chỉ so HOTA, IDF1 và MOTA. Khi TP ở một ngưỡng IoU bằng 0, helper cũ làm sai AssA (sai số lớn nhất 0,2631579 trên thang 0–1). Điểm TrackEval gốc được giữ. Các dòng ngân sách của smoke ban đầu thiếu deadline. Hồ sơ v2 và lượt trước review tách riêng ở `../v2/`.

**Giao thức:** GPT Q1/Q5 (`D:/LABOTARY/automation/runs_local/hoi_codex_R66_dataset_moi_pf_20261102/answer.json`).

**Mã:** `src/r66_ngoai/danh_gia_r66.py` và `chay.sh`.
- Wrapper riêng, gọi helper R64 đã khóa: `kiem_gt`, `detector_r54`, ánh xạ COCO, `hau_kiem_tracking`, `cham`,
  `max_time_lost_thuc`.
- **Không sửa `src/r64_ngoai`.**

**Log:**
- `../01_smoke.log`: lượt đầu dừng ở bước chấm.
- `../02_smoke_lai.log`: chạy lại.
- `03_lap_pf.log`: lặp PF cùng seed.
- `track_logs/`, `ngan_sach.jsonl`, `combined_tung_run.csv`, `tong_hop_trang_thai.json`.

## 1. Nguồn và bộ chuyển đổi

**Nguồn:**
- Ảnh: `sherbrooke_frames.zip` (sha256 trong `../../sha256_zip.txt`), giải nén ra F:.
- GT: `sherbrooke_gt_cars.sqlite`, 15 ID, gồm cả ID loại `unknown`.
- Mask: `sherbrooke_mask.png`, tải từ trang tác giả về `data/processed/_r66_nguon/sherbrooke/` (sha256
  `674f360b6341a3e3…`). Kèm bản lưu trang `dataset_page.html` (`204901de…`) và `index_page.html` (`978b4328…`).
- Trang tác giả ghi "800x600 … 1001 frames (30 fps)".
- Thông báo bản quyền trên trang là của **IEEE cho bài báo** ("Personal use of this material is permitted …"); không
  thấy giấy phép mở cho dataset.

**Quy đổi khung và ID:**
- Frame nguồn 2754–3754 thành frame MOT 1–1001; giữ cả frame không có GT (box thực tế nằm trong frame 2755–3690).
- ID nguồn 0–14 được ánh xạ cố định sang 1–15 (`gt_log.json`).

**Quy ước cạnh box:** `w = x2 − x1`, không cộng 1. Bằng chứng: x2 lớn nhất bằng 799 = W − 1, nên x2 là chỉ số pixel;
không tự cộng 1 theo đúng chỉ dẫn. Không nội suy box thiếu.

**Mask (800×600, giá trị {0, 1}):**
- Cực tính: 2.012/2.015 tâm box GT nằm trong vùng 1, nên **vùng 1 là annotated zone**.
- Quy tắc ROI: tâm box (lấy floor, kẹp trong ảnh) rơi vào mask = 1. Áp **cùng quy tắc** cho GT (lúc chuyển đổi) và
  output tracker (trước khi chấm). Output thô giữ ở `trackers/<split>/<run>/data_tho/`.
- GT bị loại ngoài ROI: **3 dòng** (ID 6 frame 436–437, ID 12 frame 912), đều là box cao 3 px ở mép mask. Smoke 128
  frame không loại dòng nào.

**Kiểm bộ chuyển đổi:** ảnh `../kiem_converter/frame_{0002,0300,0436,0600,1001}.jpg` (viền đỏ là mask, xanh là GT
chấm), số GT mỗi frame ở `kiem_converter.json`. Box ôm đúng xe; frame 1001 không có GT, khớp với nguồn.

**Phát hiện cần review:** một xe đỏ đỗ hoặc chờ ở góc dưới trái, nằm trong ROI, có mặt ít nhất từ frame 2 đến 436
nhưng **không có GT**. Nếu detector bắt được, xe này bị tính là FP. Các xe đỗ khác gần viền mask cũng có thể như vậy.
Đây là giới hạn của nhãn nguồn, không sửa GT. FP ở smoke (khoảng 260–300 trên 235 dòng GT) phù hợp với hiện tượng này,
nhưng chưa tách định lượng.

## 2. Cache (3 detector R54)

**Thông số sinh cache:** imgsz 640 rect, conf 0,05, iou 0,7, max_det 3000, FP32. mixed5 được ánh xạ sang ID COCO,
`cls_goc` được lưu lại.

**Cổng nội dung** (meta và fingerprint, provenance bắt buộc, ánh xạ tính lại, timeline 1–128, `time_s = (i−1)/30`,
hữu hạn, box, cls) đạt cho cả 3 detector, sau đó mới ghi marker.

| Detector | Số box | Tốc độ |
|:--|--:|--:|
| yolov8n_coco | 3.698 | 9,7 fps |
| yolov8n_mixed5 | 3.914 | 12,4 fps |
| yolo11n_mixed5 | 4.178 | 10,8 fps |

Tên cache chứa hash giao thức (`00c7363e`), gồm split, frame, FPS, lớp, sha256 GT SQLite và sha256 mask.

## 3. Tracking smoke: đủ 36 run, XONG

**Cấu hình:**
- Lớp tracker: COCO 2 (car).
- Lọc detection: conf > 0,25 / > 0,10, max 300.
- 30 fps, buffer 1,2 s = 36 bước (`max_frames_lost` thực = 36), elapsed steps.
- PF500 chạy seed 0/1/2, K1/K2 tắt.

**Hậu kiểm R64:** đạt trên output thô.

**Tái tổ hợp `raw_fields`:** khớp COMBINED, |Δ| ≤ 2,2e-16.

**Lặp PF cùng seed:** yolov8n_mixed5 c25 PF500 chạy lại dưới tên `r66smoke-lap-*`. MOT thô **trùng byte** với bản gốc
(s0 `9f620f35…`, s1 `4b41c6b7…`), và s0 khác s1.

**Sự cố đã sửa:** lượt đầu, hàm lọc ROI ghi số bằng `repr` của numpy 2 (`np.float64(...)`), nên TrackEval không đọc
được. Đã sửa để giữ nguyên byte từng dòng gốc và chỉ lọc dòng. Run hỏng được cách ly ở `trackers/_r66_hong/…`
(`cach_ly.log`), không xóa.

**HOTA (thang 100; PF500 là trung bình 3 seed):**

| Detector | Ngưỡng | CV | EKF | UKF | PF500 | FP (CV) | FN (CV) |
|:--|:--|--:|--:|--:|--:|--:|--:|
| yolov8n_coco | c25 | 41,36 | 41,21 | 41,21 | 31,32 | 299 | 75 |
| yolov8n_coco | c10 | 41,36 | 41,21 | 41,21 | 30,90 | 299 | 75 |
| yolov8n_mixed5 | c25 | 40,89 | 40,84 | 40,72 | 28,87 | 258 | 92 |
| yolov8n_mixed5 | c10 | 38,03 | 37,91 | 37,87 | 28,89 | 330 | 92 |
| yolo11n_mixed5 | c25 | 41,40 | 41,35 | 41,32 | 29,88 | 281 | 89 |
| yolo11n_mixed5 | c10 | 41,35 | 41,31 | 41,02 | 30,32 | 282 | 89 |

Smoke chỉ có 128 frame và 4 ID GT, nên số liệu **chỉ để kiểm kỹ thuật, không chọn hay chỉnh gì theo đây**. Quan sát:
PF500 thấp hơn CV khoảng 10–12 điểm HOTA, FN và FP cao hơn, cùng chiều với RoundaboutHD.

## 4. Dự toán full (1.001 frame)

| Bước | Ước tính |
|:--|:--|
| Cache | khoảng 100 s mỗi detector |
| Tracking CV/EKF/UKF | vài giây mỗi run |
| Tracking PF500 | dưới 1 phút mỗi seed |
| Tổng | dưới 20 phút |

Dung lượng không đáng kể.

## 5. Chờ review

Xin duyệt chạy full 36 run trên `Sherbrooke-cars-f2754-3754-roi-v1`. Các điểm cần GPT xác nhận:
1. Quy ước cạnh box (không +1).
2. Quy tắc ROI theo tâm box.
3. Cách xử lý xe đỗ không có GT.
4. Wrapper và cổng R66.
