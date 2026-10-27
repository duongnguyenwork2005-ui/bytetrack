# R59 — Quyết định và giao thức đối chứng SORT–ByteTrack trên test40 (chỉ c25)

Ngày: 2026-10-23. Người soạn: Claude (`claude-opus-5-5`, effort high). Reviewer: Codex (một review cuối cổng A, một
review nghiệm thu cuối B).

## 1. Nguồn ủy quyền

- Plan R59 do Codex (GPT) đề xuất sau lượt `bytetrack_r58_sort_design_20261023_053436` (lời gọi `thu_cong_20261023_r58`,
  23/10/2026 10:53); **có hiệu lực vì người dùng đã duyệt** (ghi chú ngày 23/10/2026).
- Sáu lựa chọn Q-A..Q-F do Codex trả lời theo ủy quyền người dùng:
  `D:/LABOTARY/automation/runs_local/hoi_codex_R58_need_decision_20261023/answer.json` (đọc, không sửa).
- R58A (`handoff/logs/review58/01_thiet_ke_sort_test40.md`) **chỉ là hồ sơ thiết kế**. Approve của nó
  (`.../bytetrack_r58_sort_design_20261023_053436/reports/review_002.json`) chỉ nghiệm thu hồ sơ, không cấp quyền chạy.
  Hồ sơ R58 giữ nguyên làm lịch sử; các quyết định ghi ở đây.
- B được cấp quyền trước **có điều kiện**: chỉ khi cổng A có đủ bằng chứng đạt, khóa R59 hợp lệ và Codex review A
  approve, không còn vấn đề chặn. Thiếu quota review thì dừng ở cổng; chờ không có nghĩa là approve.

## 2. Quyết định đã chốt

| # | Quyết định | Hệ quả |
|---|---|---|
| Q-A | Chỉ c25: 6 run SORT mới = 3 detector × `sort-cv30` / `sort-ekf30` | 6 × 40 = 240 đơn vị run–video |
| Q-B | Không áp dụng (không chạy c10); `det_thresh = 0,25` theo hai YAML có sẵn | Không YAML mới, không fixture lọc kép (P3b9dd663 đã bị từ chối) |
| Q-C | `assoc_log = false` cho SORT (như R51) | Khác ByteTrack R56 (`assoc_log = true`); không sửa mixin (P74d51e88 đã bị từ chối) |
| Q-D | Tái dùng 6 run ByteTrack c25 cv/ekf của R56, bắt buộc đạt mục 4.3 hồ sơ R58 | Không đạt → dừng, báo need_decision; không chạy lại ByteTrack |
| Q-E | Chỉ điểm metric và hiệu số ByteTrack − SORT; không CI | Kết quả mô tả; không suy luận thống kê, không áp CI R57 sang SORT |
| Q-F | Kiểm hash `config.py` (byte thô + chuẩn hóa LF) và tham số bộ lọc hiệu lực từ instance ở bước chuẩn bị | Hash LF chỉ giải thích khác quy ước; toàn vẹn theo khóa gốc là byte thô |

## 3. Giao thức (`configs/phase2/r59_sort_test40/giao_thuc.json`)

Sáu run theo đúng thứ tự, mỗi run ánh xạ một nguồn ByteTrack R56:

| Run SORT (R59) | YAML | Đối chứng ByteTrack R56 |
|---|---|---|
| `r59-yolov8n_coco-c25-sort-cv30` | `configs/phase2/trackers/sort_cv_age30.yaml` | `r54-yolov8n_coco-c25-cv` |
| `r59-yolov8n_coco-c25-sort-ekf30` | `configs/phase2/trackers/sort_ekf_ctrv_age30.yaml` | `r54-yolov8n_coco-c25-ekf` |
| `r59-yolov8n_mixed5-c25-sort-cv30` | `sort_cv_age30.yaml` | `r54-yolov8n_mixed5-c25-cv` |
| `r59-yolov8n_mixed5-c25-sort-ekf30` | `sort_ekf_ctrv_age30.yaml` | `r54-yolov8n_mixed5-c25-ekf` |
| `r59-yolo11n_mixed5-c25-sort-cv30` | `sort_cv_age30.yaml` | `r54-yolo11n_mixed5-c25-cv` |
| `r59-yolo11n_mixed5-c25-sort-ekf30` | `sort_ekf_ctrv_age30.yaml` | `r54-yolo11n_mixed5-c25-ekf` |

- Dùng nguyên hai YAML khóa (không sửa): `det_thresh 0,25`, `sort_max_age 30`, `sort_iou_threshold 0,3`,
  `sort_min_hits 3`. Override runner `{"track_buffer": 30}` như R55 (SORT không đọc khóa này; giá trị trùng).
- Giữ như R56: lọc lúc load `conf > 0,25` (ngặt) → lớp `[2,3,5,7]` → cắt `[:300]`; fps 25; `elapsed_mode skip`; GT
  DETRAC-test hiện hành; evaluator `W.cham` (TrackEval MotChallenge2DBox, HOTA/CLEAR/Identity). `pf_diag = true` giữ
  như R56; với SORT không có hiệu lực (SORT không có `kalman_filter`), ghi trong cấu hình hiệu lực.
- Nguồn duy nhất: phiên `results/phase2/r55_tracking/thuc_thi/phien_20261021_161832_735800_pid11640`; khóa
  `results/phase2/r55_tracking/chuan_bi/khoa_20261021_155510_440500_pid8768/khoa.json` SHA256
  `ce85628e34bc8507f9577c51a06d381a2d8e5d0f35cdf099c18330429d35247b`; lần chấm theo
  `results/phase2/r56_tong_hop/phien_20261021_161832_735800_pid11640/nguon.json` (SHA256 `96b2c543…d817d5d4`, giá trị R57
  đã ghi). Thư mục lần chấm = cha thứ ba của `pedestrian_summary.txt` mà `nguon.json` trỏ tới; **không** chọn mới nhất,
  không chọn theo điểm.
- D4 = ảnh không có `<frame>` GT, không phải ảnh không detection; giữ trong timeline, không ép D4 không xuất box.
- Smoke K7: `MVI_39031`, `MVI_39051`, frame 1–128, split `DETRAC-smoke128`, cache R56 `yolov8n_coco`, GT smoke tạo trong
  namespace R59 bằng `W.tao_gt_smoke` (quy tắc R54). Chấm nguồn: `r54-yolov8n_coco-c25-cv` (sao byte 40 MOT, không chạy
  tracker). Tái lập B: `MVI_39031` (video đầu manifest) × hai run SORT `yolov8n_coco`.

### 3.1 Quy tắc metric (khóa trước khi có số SORT)

- Nguồn: `COMBINED_SEQ` của TrackEval (`combined_metrics.csv` của từng lần chấm); không trung bình điểm video.
- Chính HOTA; bổ sung AssA, DetA, IDF1, IDSW; tham chiếu MOTA, FP.
- Tỷ lệ báo theo phần trăm (giá trị evaluator × 100), làm tròn 3 số lẻ; hiệu số ByteTrack − SORT tính từ giá trị
  chưa làm tròn, đơn vị điểm phần trăm (pp), làm tròn 3 số lẻ. IDSW/FP là số đếm nguyên, hiệu số nguyên.
- Đối chiếu nguồn R56: (a) `combined_metrics.csv` ByteTrack == `combined` trong `HOAN_TAT.json` của lần chấm đó (so
  float, `==`); (b) số đã làm tròn so với `tong_hop_36_run.csv` R56 với dung sai HOTA/AssA/DetA/IDF1/MOTA 0,0005 pp
  (theo làm tròn 3 số lẻ của R56), IDSW/FP **0** (đếm chính xác). Dung sai này **không** dùng cho các kiểm trùng byte
  (chấm nguồn, chấm lại, lặp tracker).

## 4. Cổng A → B

A đạt khi: nguồn/khóa/hash R56 và backend hợp lệ (mục 4.3 điều 1–3); fixture CPU đạt; cấu hình hiệu lực từ instance
và quy ước băm `config.py` có bằng chứng, không có khác biệt nội dung/tham số chưa giải thích (K1); parity detection
nội dung + thứ tự 120/120 tổ hợp, đủ 56.340 ảnh / 173 D4 (K2); K3–K6; smoke K7 2 SORT × 2 video × 128 frame chấm
được, không gọi detector; chấm lại `r54-yolov8n_coco-c25-cv` trùng byte cả ba file metric (mục 4.3 điều 4); khóa R59
mới hợp lệ; gói bằng chứng đủ hash/log/exit code; không full run. Sau đó Codex review A; chỉ approve đúng gói + khóa
mới mở B (`--buoc mo-phien --khoa <ID_A> --review-a <review>`; wrapper yêu cầu verdict `approve`, nội dung review nhắc
ID khóa hoặc ID gói A, không còn vấn đề P0/P1 mở).

## 5. Giới hạn diễn giải (ghi lại ở mọi báo cáo kết quả)

- Chỉ c25: ở c25 `conf_min = det_thresh = 0,25`, vòng ghép điểm thấp của ByteTrack rỗng; đối chứng **không** đo lợi
  ích detection điểm thấp.
- Gọi đúng: "association và vòng đời kiểu SORT, dùng bộ lọc CV/EKF của ByteTrack, `max_age 30`"; không phải SORT
  nguyên bản, không phải "chỉ đổi lifecycle".
- Không CI, không suy luận thống kê, không kết luận tốc độ (logging khác nhau: SORT `assoc_log=false`, ByteTrack R56
  `assoc_log=true`).
- Không so detector, không chọn conf/detector bằng test; không chuyển kết quả R51 dev12 sang test40.
