# R64 — SVMOT test (9 chuỗi): đánh giá hệ thống 3 detector × 4 bộ lọc (02/11/2026) — XONG 36/36

**Giao thức** (GPT 01/11, `D:/LABOTARY/automation/runs_local/hoi_codex_R64_dataset_moi_20261101/answer.json` Q1; sửa
theo review smoke `.../hoi_codex_R64_review_smoke_20261102/answer.json`):
- Dữ liệu: 9 chuỗi test chính thức scene_05 (seq_22–25, sáng) và scene_06 (seq_26–30, tối), 8.250 frame 1920×1080, 30 fps.
  GT gộp 7 lớp xe thành class 1, giữ visibility, không loại xe đứng yên hay visibility thấp. **Ảnh toàn khung, không tô
  `roi_mask.png`** — đây là giao thức full-image của dự án, không phải giao thức chính thức của SVMOT. Xe ở xa không có
  nhãn nên mọi dự đoán không ghép được GT đều chịu FP.
- Detector: 3 checkpoint R54 (sha256 kiểm), imgsz 640 rect, conf sinh 0,05, FP32; mixed5 → ID COCO. Cache
  `data/interim/det_cache/r64_svmot_<det>_c0.05`: 27/27 đơn vị qua cổng hoàn tất + nội dung. Bốn đơn vị yolov8n_coco
  seq_22–25 được sinh bằng mã trước review và có biên bản xác minh bổ sung.
- Tracker: ByteTrack YAML hiện có, lớp 1 2 3 5 7, max 300, c25 / c10, 30 fps, buffer 1,2 s = 36 bước (`max_frames_lost`
  thực tế 36), elapsed steps (n_steps = 1 mọi frame). Evaluator `mot`. PF500 chạy seed 0/1/2.
- Mã: `src/r64_ngoai/danh_gia_ngoai.py`, `chay.sh`. Log (cùng thư mục): `04_svmot_cache_v2.log`, `05_svmot_track_*.log`,
  `06_svmot_pf_*.log`, `svmot/track_logs/`. Kết quả chấm: `results/trackeval/SVMOTSYS-test/r64-*`, mỗi run kèm
  `cong_r64.json` (cổng trước, hậu kiểm, hash GT/mã). Tracking 3 detector chạy **song song trên CPU** ⇒ thời gian trong
  log không phải benchmark tốc độ.
- **Thiếu bằng chứng ngân sách (ghi nhận, không hồi tố):** lượt đêm 01–02/11 có kiểm thời gian/dung lượng trước mỗi job
  nhưng mã khi đó **không ghi nhật ký** và chưa giữ dự phòng 30 phút. Từ bản sửa 02/11 04:3x (review cuối GPT), mỗi job ghi
  `<ds>/ngan_sach.jsonl` (deadline, còn lại, dự toán, dự phòng, ổ đích, dung lượng trống, dự toán ghi). Kiểm lại cổng mới
  trên toàn bộ 39 đơn vị cache và 48 cấu hình run: `10_kiem_lai_cong_moi.log` (đều ĐẠT/xong).
- Thống kê: chỉ **mô tả**. Số COMBINED tái tổ hợp từ raw_fields, khớp TrackEval ≤ 1e-9. PF500 lấy trung bình metric
  COMBINED của 3 seed. **Không tính CI**: 9 chuỗi chỉ thuộc 2 scene nên không phải 9 quan sát độc lập.
  Bảng: `svmot/combined_tung_run.csv`, `tung_chuoi.csv`, `theo_scene.csv`.

## COMBINED (điểm thang 100; PF500 = trung bình 3 seed)

| Detector | Ngưỡng | HOTA CV / EKF / UKF / PF500 | IDF1 CV / EKF / UKF / PF500 | IDSW CV / EKF / UKF / PF500 | DetA (CV) | MOTA (CV) |
|:--|:--|:--|:--|:--|--:|--:|
| yolov8n_coco | c25 | 48,66 / 49,65 / 49,21 / 50,70 | 50,30 / 51,91 / 51,18 / 53,36 | 435 / 407 / 418 / 427,3 | 52,30 | 47,22 |
| yolov8n_coco | c10 | 47,53 / 48,51 / 48,05 / 48,51 | 49,37 / 50,34 / 49,63 / 50,62 | 311 / 308 / 306 / 342,3 | 49,58 | 38,66 |
| yolov8n_mixed5 | c25 | 44,34 / 45,71 / 45,02 / 45,41 | 45,86 / 47,69 / 46,82 / 47,39 | 265 / 261 / 249 / 296,7 | 42,05 | 7,77 |
| yolov8n_mixed5 | c10 | 42,08 / 43,32 / 43,07 / 43,23 | 42,56 / 44,41 / 44,14 / 43,99 | 264 / 244 / 237 / 265,7 | 38,77 | −7,60 |
| yolo11n_mixed5 | c25 | 44,08 / 44,36 / 45,04 / 44,95 | 43,63 / 44,26 / 44,98 / 45,04 | 259 / 257 / 250 / 272,7 | 41,83 | 2,44 |
| yolo11n_mixed5 | c10 | 42,34 / 42,49 / 43,17 / 43,08 | 41,19 / 41,22 / 42,31 / 42,77 | 234 / 216 / 217 / 241,7 | 38,20 | −16,21 |

PF500 từng seed (c25, yolov8n_coco): HOTA 51,30 / 51,31 / 49,50, tức dao động giữa các seed khoảng 1,8 điểm.

## HOTA theo scene (c25, tái tổ hợp raw_fields)

| Detector | Scene | CV | EKF | UKF | PF500 |
|:--|:--|--:|--:|--:|--:|
| yolov8n_coco | scene_05 (sáng) | 46,78 | 47,10 | 47,09 | 46,53 |
| yolov8n_coco | scene_06 (tối) | 50,39 | 51,98 | 51,16 | 54,38 |
| yolov8n_mixed5 | scene_05 | 41,88 | 42,11 | 42,17 | 41,66 |
| yolov8n_mixed5 | scene_06 | 46,33 | 48,58 | 47,32 | 48,37 |
| yolo11n_mixed5 | scene_05 | 42,31 | 42,37 | 42,45 | 41,95 |
| yolo11n_mixed5 | scene_06 | 45,60 | 46,07 | 47,25 | 47,48 |

## Quan sát mô tả (không suy nguyên nhân chắc chắn, không chọn gì theo đây)

- **Detector:** yolov8n_coco cao nhất ở mọi ô (HOTA cao hơn khoảng 4–6 điểm). Hai detector mixed5 có DetA thấp hơn khoảng 10 điểm
  và MOTA rất thấp, thậm chí âm ở c10, tức FP nhiều so với GT. Cần xem precision/recall trước khi giải thích; một khả năng
  là xe ở xa không có nhãn.
- **Bộ lọc:** ở c25, EKF/UKF/PF500 có HOTA cao hơn CV 0,3–2,0 điểm; ở c10 chênh lệch tương tự. Phần lớn chênh lệch đến từ
  AssA, còn DetA gần như bằng nhau. PF500 có IDSW trung bình cao hơn EKF/UKF trong cả sáu nhóm (detector × ngưỡng);
  thứ hạng HOTA giữa EKF/UKF/PF500 phụ thuộc detector và ngưỡng (vd PF500 thấp hơn EKF với yolov8n_mixed5, thấp hơn UKF
  với yolo11n_mixed5, ở cả hai ngưỡng). Phần HOTA PF500 cao hơn CV tập trung ở **scene_06**; ở scene_05 PF500 ngang hoặc
  thấp hơn CV. Chỉ có 2 scene, chưa tính CI ⇒ **chưa kết luận** bộ lọc nào tốt
  hơn trên SVMOT.
- c10 làm HOTA giảm khoảng 1,1–2,4 điểm so với c25 ở cả 3 detector.
