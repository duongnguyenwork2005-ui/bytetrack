# R65 — chẩn đoán PF500 thấp trên RoundaboutHD prefix3000 (02/11/2026)

**Quyết định và phạm vi** (khóa trước khi chạy, theo GPT 02/11 Q2:
`D:/LABOTARY/automation/runs_local/hoi_codex_R64_viec_tiep_20261102/answer.json`; `00_giao_thuc.json`):
- Dữ liệu: imagesc001–004, frame 1–3000.
- Detector và ngưỡng: chỉ yolov8n_mixed5, c25.
- Bộ lọc: CV, EKF, UKF và PF500 seed 0/1/2, tổng 24 đơn vị camera × bộ lọc/seed.
- Đầu vào giữ nguyên R64: cache `r64_roundabout_yolov8n_mixed5_c0.05`, GT SCT đã sửa, cấu hình R64 (15 fps, buffer 18
  bước, elapsed steps). PF giữ N = 500, ESS 0,5N, ước lượng mean, không roughening.
- **Không tune, không thay benchmark R64, không ablation.** Đây chỉ là lượt tìm bằng chứng giải thích, không phải điều
  kiện nghiệm thu R64.

Mã: `src/r65_pf_roundabout/` (`quan_sat.py`, `chan_doan.py`, `phan_tich.py`, `test_quan_sat.py`). Kết quả:
`results/phase2/r65_pf_roundabout_diag/` (`phan_tich/thong_ke.json`, `thong_ke.csv`, `phan_tich/mau_*.csv.gz`).
Manifest hash đầu vào/mã/thư viện: `manifest_sha256.json`. Ngân sách: `ngan_sach.jsonl`. Log chạy:
`01_chay_cham_parity.log`, `track_logs/`.

## 1. Cổng "log không đổi hành vi" — ĐẠT 24/24 (`parity.json`)

Mỗi đơn vị chạy hai nhánh, cả hai cùng bộ quan sát chỉ đọc `quan_sat.py`:
- `off`: không log.
- `on`: bật `--assoc-log --pf-diag`.

Bộ quan sát băm nối chuỗi trạng thái tại các mốc sau:
- **Mỗi frame:** mọi track Tracked/Lost gồm mean, covariance; riêng PF thêm particles, logw và
  `rng.bit_generator.state`.
- **Từng bước PF:** propagate, update_particles, reseed_motion, sample_initial.

Kết quả so khớp:
- MOT `data/<v>.txt` và `data_cls/<v>.csv`: **trùng byte off = on = R64 gốc**.
- `per_video_metrics` và `combined_metrics`: off = on = R64.
- Chuỗi hash frame, chuỗi hash PF, bộ đếm sự kiện và `pf_update.csv`: off = on.

Phạm vi sự kiện mà dữ liệu thật đã đi qua (PF, tổng 3 seed):

| Sự kiện | Số lần |
|:--|--:|
| resample | 152.485 |
| reseed chuyển động | 5.048 |
| update từ trạng thái Lost (refind) | 6.148 |
| suy biến (reset) | **0** |

Vì dữ liệu thật không kích hoạt suy biến, phần này được kiểm bằng fixture `test_quan_sat.py` (2/2 đạt). Fixture có suy
biến, resample, Lost/refind; với diag bật/tắt và observer bật/tắt, output và trạng thái RNG đều trùng.

## 2. Chuyển động GT (SCT, frame liên tiếp)

| | n | trung vị | p90 | p95 |
|:--|--:|--:|--:|--:|
| dịch tâm / chiều cao box | 75.316 | 0,097 | 0,269 | 0,328 |
| \|h_t / h_{t−1} − 1\| | 75.316 | 0,013 | 0,037 | 0,050 |
| chiều cao box (px) | 75.316 | 72,7 | 239,9 | 345,3 |

Trung vị dịch tâm/h theo camera: cam1 0,077; cam2 0,095; cam3 0,138; cam4 0,091.

## 3. Sai số dự đoán trước association (box dự đoán trong `assoc_log`)

**Cách nối track với GT:**
1. Ghép một-một IoU ≥ 0,5 (Hungarian) giữa output MOT và GT ở **quan sát trước** (frame output gần nhất < t).
2. Theo cùng GT đó sang frame t.
3. Tính e = ‖c_pred − c_GT‖ / h_GT.

`gap` ở đây là t − frame output trước.

**Độ phủ mẫu:**

| Bộ lọc | hợp lệ | không nối được GT | thiếu GT ở t |
|:--|--:|--:|--:|
| CV | 54.161 | 22.015 | 1.367 |
| EKF | 54.137 | 22.143 | 1.351 |
| UKF | 54.141 | 22.132 | 1.354 |
| PF s0–s2 | 53.691–53.937 | 30.271–31.075 | 1.577–1.644 |

PF có nhiều mẫu không nối được hơn, vì track PF ở quan sát trước ít khi trùng GT ở IoU ≥ 0,5. Như vậy các thống kê bên
dưới có thể **chệch theo hướng có lợi cho PF**, vì chỉ đo trên những track còn bám được GT.

**Sai số e (trung vị / p90):**

| Bộ lọc | gap = 1 | gap 2–3 | gap 4–9 | gap ≥ 10 | IoU(pred, GT) trung vị | tỉ lệ IoU < 0,5 |
|:--|:--|:--|:--|:--|--:|--:|
| CV | 0,063 / 0,151 | 0,122 / 0,362 | 0,274 / 1,30 | 0,80 / 3,24 | 0,783 | 11,5 % |
| EKF | 0,060 / 0,144 | 0,113 / 0,317 | 0,267 / 1,01 | 0,86 / 2,57 | 0,787 | 11,5 % |
| UKF | 0,060 / 0,145 | 0,114 / 0,321 | 0,270 / 1,05 | 0,86 / 2,70 | 0,786 | 11,7 % |
| PF s0 | 0,098 / 0,257 | 0,174 / 0,567 | 0,533 / 1,74 | 1,65 / 4,46 | 0,686 | 23,0 % |
| PF s1 | 0,098 / 0,256 | 0,175 / 0,564 | 0,528 / 1,76 | 1,58 / 4,52 | 0,686 | 23,8 % |
| PF s2 | 0,097 / 0,258 | 0,179 / 0,569 | 0,525 / 1,73 | 1,58 / 4,26 | 0,686 | 23,4 % |

Sai số kích thước tương đối (trung vị / p90):
- |h/h_GT − 1|: CV 0,066 / 0,265; PF 0,079–0,080 / 0,277–0,280.
- |w/w_GT − 1|: CV 0,103 / 0,335; PF 0,131–0,134 / 0,388–0,391.

**Tập chung** gồm 35.466 bộ (video, GT, frame) có gap = 1 và hợp lệ ở **cả 6** run:

| Bộ lọc | e trung vị | e p90 | IoU trung vị | ghép đúng | chi phí không đạt |
|:--|--:|--:|--:|--:|--:|
| CV | 0,063 | 0,148 | 0,810 | 95,3 % | 0,34 % |
| EKF / UKF | 0,060 | 0,141–0,142 | 0,813–0,814 | 95,3 % | 0,34 % |
| PF s0–s2 | 0,095–0,096 | 0,247–0,251 | 0,739–0,742 | 94,8 % | 0,85–0,87 % |

## 4. Kết quả association của các mẫu hợp lệ (tỉ lệ)

| Bộ lọc | ghép đúng GT | không có det phù hợp (IoU det–GT < 0,5) | tranh chấp (det bị track khác lấy) | det còn nhưng chi phí không đạt | ghép det vật khác |
|:--|--:|--:|--:|--:|--:|
| CV | 77,2 % | 17,6 % | 2,8 % | 1,8 % | 0,7 % |
| EKF | 77,2 % | 17,7 % | 2,7 % | 1,8 % | 0,7 % |
| UKF | 77,1 % | 17,6 % | 2,8 % | 1,8 % | 0,7 % |
| PF s0–s2 | 71,1–71,4 % | 18,5–18,7 % | 6,2–6,4 % | 3,2 % | 0,7 % |

**Khác biệt theo gap:**
- **gap = 1:** chênh nhỏ (ghép đúng: CV 40.383 so với PF ~37.170 trên số mẫu khác nhau).
- **gap ≥ 4:** tranh chấp của PF gấp khoảng 2,1–3 lần CV. Ví dụ ở gap ≥ 10: tranh chấp CV 926, PF 1.961–2.070.
- **Ca "chi phí không đạt":** IoU giữa box dự đoán và det phù hợp có trung vị CV 0,434 so với PF 0,347–0,367; chi phí
  fused trung vị 0,855 so với 0,875–0,883 (ngưỡng tầng 1 là 0,8).

## 5. Phân mảnh và sự kiện

| Bộ lọc | track / GT (TB) | p90 | đổi track liên tiếp trên GT | new | lost | refind | removed | IDSW (TrackEval) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| CV | 2,34 | 4 | 581 | 6.244 | 3.405 | 2.333 | 1.039 | 471 |
| EKF | 2,33 | 4 | 574 | 6.283 | 3.430 | 2.345 | 1.053 | 477 |
| UKF | 2,37 | 4 | 592 | 6.279 | 3.426 | 2.348 | 1.047 | 494 |
| PF s0–s2 | 3,91–4,05 | 7 | 1.077–1.126 | 7.391–7.496 | 3.705–3.733 | 2.030–2.063 | 1.615–1.676 | 1.016–1.073 |

Số frame mất trước khi refind có phân bố gần nhau: trung vị 3, p90 7–8 ở mọi bộ lọc.

## 6. Thống kê nội bộ PF (3 seed gần trùng nhau)

- **Mẫu số:** chỉ dòng `update` không suy biến (50.784–50.889 dòng); 1.654–1.714 dòng `init_motion` đếm riêng; suy biến
  = 0.
- **ESS/N trước resample:** trung vị 0,188, p90 0,394, p95 0,424.
  - ESS/N < 0,2: 52,6–52,8 % số update.
  - ESS/N < 0,5: 99,98–99,99 %.
  - ⇒ resample ở **gần như mọi update**.
- **Số tổ tiên khác nhau / N sau resample** (lấy từ `last_resample_idx`, không phải `n_unique` theo cx): trung vị 0,356,
  p90 0,52.
- **Độ trải của đám hạt** (lấy sau resample, chia cho chiều cao box dự đoán cùng frame):

| | trung vị | p90 | p95 |
|:--|--:|--:|--:|
| spread_xy / h | 0,057 | 0,063 | 0,066 |
| spread_v / h | 0,016 | 0,033–0,035 | 0,065 |
| spread_θ | 0,105–0,113 | 0,61–0,65 | — |

- **maha2_best:** trung vị 1,82–1,90, p90 35–39, p95 82–87.
- **Nghĩa thời gian (theo lưu ý của GPT):** `gap` trong `pf_update` = frame hiện tại − `t.frame_id` lúc gọi bộ lọc.
  - Update thường (state Tracked) có gap = 0, vì `frame_id` đã được gán trước khi gọi bộ lọc. Vì vậy `n_lost` trong
    `pf_diag` **không** phải số frame thiếu quan sát.
  - Update từ Lost có gap 2–19.

## 7. Finding phụ (không đặc thù PF, không sửa)

Track bị đánh dấu Removed ở frame t (`frame_id − end_frame > 18`) vẫn nằm trong pool Lost thêm **một** frame:
- **Nguyên nhân:** `merge_track_pools` trừ `removed_stracks` **trước** khi nối danh sách removed của frame hiện tại.
  Thứ tự này giống bản ultralytics gốc.
- **Hệ quả:** track đó được dự đoán ở frame t+1 (`assoc_log` state 3, n_lost = 20; khoảng 1.039–1.676 dòng/run) và
  đôi khi được ghép lại.
- **Số lần ghép lại:** CV 8, EKF 6, UKF 5, PF 4–10 mỗi run.

Hiện tượng có ở mọi bộ lọc với quy mô tương đương, nên không giải thích được khoảng thua của PF. Ghi lại để mô tả đúng
"buffer 18 bước" trên thực tế (track tồn tại tối đa thêm 1 frame). Theo giao thức, không sửa và không chạy thay.

## 8. Kết luận (chỉ nêu chuỗi bằng chứng quan sát được; chưa xác lập nguyên nhân)

1. **Box dự đoán của PF500 lệch GT nhiều hơn CV/EKF/UKF ngay ở frame liền kề.** Trên tập chung, sai số tâm trung vị là
   0,095 h so với 0,060–0,063 h, và IoU dự đoán–GT trung vị 0,74 so với 0,81. Khoảng cách này lớn dần khi track bị mất
   (Lost): ở gap ≥ 4, sai số của PF khoảng gấp đôi CV. Kích thước box (w, h) cũng lệch hơn.
2. **Sai số lớn hơn đi kèm association kém hơn.** PF có tỉ lệ det còn nhưng chi phí không đạt khoảng 1,8 lần CV, và tỉ
   lệ tranh chấp khoảng 2,3 lần CV. Kéo theo đó, track/GT là 3,9–4,05 so với 2,34, đổi track liên tiếp gần gấp đôi, và
   IDSW 1.016–1.073 so với 471. Tỉ lệ "không có det phù hợp" gần như bằng nhau giữa các bộ lọc, nên phần thua **không**
   đến từ detection đầu vào (cache dùng chung).
3. **Đặc điểm nội bộ đi cùng sai số dự đoán của PF:**
   - ESS/N trước resample thấp (trung vị 0,19), resample gần như mọi update, chỉ còn khoảng 36 % tổ tiên khác nhau.
   - Đám hạt hẹp theo vị trí (spread_xy khoảng 0,06 h) so với dịch chuyển GT giữa hai frame (trung vị 0,10 h, p90
     0,27 h).
   - Độ trải vận tốc rất nhỏ (spread_v trung vị 0,016 h/bước).
4. **Chưa chứng minh nguyên nhân.** ESS thấp hay đám hạt hẹp, xét riêng từng thứ, chỉ là **tương quan đi kèm**. Lượt
   này không tách được vai trò của nhiễu quá trình, nhiễu đo, số hạt, ước lượng mean sau resample hay FPS/độ phân giải.
   Muốn tách phải có ablation, và ablation cần kế hoạch duyệt riêng (theo Q2: không tự mở). Kết quả R64 giữ nguyên.

**Giới hạn:**
- Một detector, một ngưỡng, 4 camera của cùng một nút giao.
- Thống kê mẫu chỉ có trên các track nối được GT (chệch có lợi cho PF, xem mục 3).
- `chi phí fused` được tính lại offline theo công thức `1 − IoU·conf`.
- Thời gian chạy trong log không phải benchmark tốc độ (CPU, có observer và hash).
