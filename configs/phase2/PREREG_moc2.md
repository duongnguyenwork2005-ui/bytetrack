# Đăng ký trước (pre-registration) — Nhóm C, mốc 2 (dev UA-DETRAC, 12 video)

Viết **trước** khi chạy các thí nghiệm tương ứng (xem `created` trong `run_meta.json` của từng run và
lịch sử git). Mọi thay đổi sau khi xem kết quả phải ghi thành mục "Sửa đổi sau đăng ký" ở cuối file.

Cố định cho toàn bộ Nhóm C: detector + cache `yolov8n-coco_conf0.05_all` (conf > 0,25 = điều kiện `c25`),
lớp [2,3,5,7], `max_det` 300, association ByteTrack mặc định (`match_thresh` 0,8, `fuse_score` true,
`track_high/low/new_track_thresh` 0,25/0,10/0,25), `elapsed_mode: skip` (DETRAC không bỏ frame), 12 video dev
(`results/phase2/manifests/detrac_dev.json`). Chỉ số: HOTA/DetA/AssA/IDF1/MOTA/IDSW/FP/FN (COMBINED_SEQ),
giữ ID qua che khuất (`preserved` = id_match, `id_continuous`, tầng full/partial), **ghép nhầm**
(`wrong_refind_rate` từ assoc_log: refind mà det ghép thuộc GT khác GT trước khi mất). CI 95 % bootstrap
5000 lần theo video, seed 0, tái tổ hợp COMBINED (`boot_stats.py`).

Tiêu chí chọn giá trị cho mỗi trục (áp dụng giống nhau cho CV và EKF): giá trị có **HOTA c25 cao nhất**;
chỉ coi là "cải thiện" khi CI 95 % của ΔHOTA so với cấu hình hiện tại **không chứa 0**; nếu chứa 0 →
giữ cấu hình hiện tại (báo cáo là "không cải thiện rõ"). Hòa HOTA (chênh < 0,001) → IDF1 cao hơn.
Không dùng kết quả test để chọn.

## C1 — Thời gian giữ track mất dấu (`track_buffer`)

- Giá trị: `track_buffer_sec` ∈ {0,4; 0,8; **1,2** (hiện tại, GĐ1); 2,0; 3,0; 4,0} s → 10/20/30/50/75/100
  bước @ 25 fps (`time_units.track_buffer_frames` = round(s·fps)). Quy tắc xóa trong implementation
  (`BYTETracker._remove_stale_lost`): track LOST bị xóa khi `frame_id − end_frame > track_buffer`
  (tức giữ tối đa `track_buffer` bước không quan sát); ultralytics còn giữ track vừa xóa trong pool thêm
  đúng 1 frame (`merge_track_pools`), ghi nhận như một đặc điểm của thư viện.
- Bộ lọc: {cv (`bytetrack.yaml`), ekf (`configs/bytetrack_ekf_ctrv.yaml`)} — cùng buffer khi so CV–EKF.
- Thành phần thay đổi: **chỉ** `track_buffer`. Kỳ vọng ghi trước: buffer dài → preserved tăng, FP/ghép
  nhầm tăng; tồn tại điểm bão hòa; hiệu ứng buffer lớn hơn hiệu ứng bộ lọc (GĐ1 Stage E, README).
- File: `configs/phase2/experiments/expC1_buffer_dev.yaml`.

## C3 — Tham số nhiễu (tuning tham số, KHÔNG đổi mô hình nhiễu)

- Ngân sách: EKF 12 cấu hình = `ctrv_std_theta` ∈ {0,005; **0,01**; 0,02; 0,05} × `ctrv_std_omega` ∈
  {0,0025; **0,005**; 0,01} (đơn vị rad/bước, rad/bước²; hiện tại in đậm). CV 9 cấu hình =
  `cv_std_weight_position` ∈ {1/40; **1/20**; 1/10} × `cv_std_weight_velocity` ∈ {1/320; **1/160**; 1/80}
  (`bytetrack_cv` = CV bộ lọc theo instance, mặc định == `bytetrack` bit-identical). Tổng 21 run, deterministic.
- Cố định: detector, association, `track_buffer` 1,2 s. Chỉ số chọn: HOTA c25; tiêu chí như trên.
- Mô hình nhiễu (Q tỉ lệ theo h cho CV; random-walk θ/ω cho CTRV) **không đổi**; thay đổi mô hình nhiễu
  (vd roughening PF, nhiễu tham số) không thuộc C3.
- GĐ1 đã quét ω offline (20 tổ hợp) + 6 video pipeline thật: không cải thiện (README "Thử hiệu chỉnh
  nhiễu quá trình cho omega"). C3 là lần quét trên **cùng cache**, 12 video, có CI.
- File: `configs/phase2/experiments/expC3_noise_dev.yaml`.

## C2 — Nối lại track mất dấu (chọn SAU phân tích nguyên nhân, TRƯỚC khi chạy)

Quy tắc chọn (ghi trước): lấy `cause_summary.csv` (results/phase2/failure_analysis/expA_detrac_dev) của
**ekf và cv, c25, cả hai tầng**; nguyên nhân nổi bật nhất trong số các nguyên nhân **thuộc cơ chế ghép**
(không tính `detector_miss`, `buffer_expired` — hai loại này không sửa được bằng cơ chế ghép; `buffer_expired`
thuộc C1). Ứng viên tương ứng:

| Nguyên nhân nổi bật | Thay đổi nhỏ (chỉ áp dụng cho track LOST, giống nhau cho CV và EKF) | Giả thuyết |
|---|---|---|
| `score_fusion_reject` | `lost_no_fuse`: ở vòng 1, hàng chi phí của track LOST dùng 1 − IoU (không nhân conf) | det tái xuất hiện có conf thấp; fuse_score đẩy cost > 0,8 dù IoU đủ |
| `competition` / `new_track_from_det` | `lost_recover_stage`: vòng ghép thứ 3 giữa track LOST chưa ghép và det high còn dư, **trước** khi sinh track mới; cost 1 − IoU, ngưỡng riêng `lost_recover_iou` | det bị track khác lấy hoặc thành ID mới trong khi track cũ vẫn sống |
| `prediction_drift` | `lost_recover_stage` với gate Mahalanobis theo P của chính bộ lọc (vị trí) thay IoU | dự đoán lệch nhưng còn trong vùng bất định của bộ lọc; khác GĐ1 (gate theo số frame + IoA bão hòa → hại) |
| `hijacked_during` | không có ứng viên "nhỏ" — báo cáo | track bám vật che khi đang Tracked; không phải bài toán nối lại |

Sau khi chọn: điền mục "C2 — đã chọn" dưới đây (giả thuyết, tham số, điều kiện áp dụng) rồi mới chạy
`expC2_recover_dev.yaml`: {cv, ekf} × {gốc, +C2}; đo phục hồi đúng (preserved, tầng) **và** ghép nhầm.

### C2 — đã chọn (điền 2026-09-16, sau phân tích nguyên nhân, TRƯỚC khi chạy `expC2_recover_dev.yaml`)

Bằng chứng chọn (`results/phase2/failure_analysis/expA_detrac_dev/cause_summary.csv`, c25, tính trên các
thất bại `switched` + `lost_after`, chỉ xét nguyên nhân **thuộc cơ chế ghép** — không tính `detector_miss`
và `buffer_expired` (C1 đã phủ)):

| nguyên nhân | cv (full+partial) | ekf (full+partial) |
|:--|--:|--:|
| **prediction_drift** | 11 + 9 = **20** | 13 + 10 = **23** |
| score_fusion_reject | 5 + 7 = 12 | 2 + 6 = 8 |
| competition | 1 + 2 = 3 | 2 + 2 = 4 |
| hijacked_during | 0 + 1 = 1 | 0 + 0 = 0 |

→ nổi bật nhất: **`prediction_drift`** (IoU giữa box dự đoán và detection tái xuất hiện < 0,2 nên cặp đó
không bao giờ qua được `match_thresh` 0,8, dù detection có conf = 1).

**Thay đổi được chọn**: `lost_recover: maha` (trong `src/lost_recovery.py`) — vòng ghép **thứ 3** chỉ giữa
track **LOST** chưa ghép và detection **high** chưa ghép, đặt **sau** vòng 1 và **trước** vòng 2 + trước khi
sinh track mới; điều kiện nhận: khoảng cách Mahalanobis **vị trí** theo đúng hiệp phương sai của bộ lọc
(`kalman_filter.gating_distance(..., only_position=True)`) ≤ `lost_recover_maha2 = 5.991`
(χ² 2 bậc tự do, mức 95 %); xếp hạng trong gate theo khoảng cách tâm chuẩn hoá theo kích thước detection.
`lost_recover_max_lost = 0` (không giới hạn thêm — `track_buffer` đã chặn ở 30 bước).

**Giả thuyết (ghi trước)**: các thất bại `prediction_drift` là trường hợp dự đoán lệch **nhưng vẫn nằm
trong vùng bất định mà chính bộ lọc khai báo**; dùng vùng đó làm cổng (thay vì IoU cố định) sẽ nối lại được
một phần các track này, và vì gate co/giãn theo P nên số **ghép nhầm** (`wrong_refind_rate`) không tăng
đáng kể. Khác với Giai đoạn B của GĐ1 (gate nới theo *số frame* + IoA bão hoà ở 1.0 → mọi detection trong
gate đều cost 0 → IDSW tăng 5–6 lần, `results/phaseB/`): ở đây gate **chỉ lọc**, xếp hạng vẫn theo khoảng
cách, và độ rộng gate do bộ lọc quyết định.

**Đối chứng (control, không phải "thay đổi được chọn")**: `lost_recover: iou` với
`lost_recover_iou = 0.05` — cùng vòng ghép thứ 3, nhưng cổng là IoU nới lỏng thay vì hiệp phương sai. Dùng
để tách phần lợi ích đến từ *việc có thêm một vòng ghép cho track LOST* khỏi phần đến từ *thông tin bất định
của bộ lọc*. Chạy cho cả CV và EKF.

**Ma trận** (c25, mọi thứ khác cố định, `track_buffer` 1,2 s): {cv, ekf} × {gốc, +maha, +iou(control)} = 6 lần chạy.
**Chỉ số quyết định**: HOTA COMBINED + CI; kèm bắt buộc: `preserved` theo tầng (full/partial) **và**
`wrong_refind_rate` (ghép nhầm) — một thay đổi làm tăng phục hồi nhưng tăng ghép nhầm mạnh sẽ **không** được chọn.

## Kết hợp

Sau khi chốt từng trục: `expC_combo_dev.yaml` {cv, ekf} × {gốc, C1*, C2*, C3*, C1*+C2*+C3*}; chỉ chạy
tổ hợp của các giá trị đã chọn theo tiêu chí trên (nếu một trục "không cải thiện rõ" thì trục đó giữ
giá trị hiện tại và ghi rõ).

## Sửa đổi sau đăng ký

_(không có)_
