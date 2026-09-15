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

### C2 — đã chọn

_(điền sau phân tích nguyên nhân, trước khi chạy)_

## Kết hợp

Sau khi chốt từng trục: `expC_combo_dev.yaml` {cv, ekf} × {gốc, C1*, C2*, C3*, C1*+C2*+C3*}; chỉ chạy
tổ hợp của các giá trị đã chọn theo tiêu chí trên (nếu một trục "không cải thiện rõ" thì trục đó giữ
giá trị hiện tại và ghi rõ).

## Sửa đổi sau đăng ký

_(không có)_
