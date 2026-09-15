# KHÓA cấu hình trước khi chạy tập TEST UA-DETRAC (mốc 2)

Bản khóa máy đọc được: [results/phase2/lock/lock_moc2.json](../../results/phase2/lock/lock_moc2.json)
(commit, thời điểm, sha256 từng file cấu hình + từng YAML tracker, 40 sequence, cache meta + sha256 weights,
phiên bản thư viện). Sinh bởi `python src/make_lock.py --exps configs/phase2/experiments/expA_detrac_test.yaml
configs/phase2/experiments/expB_detrac_test.yaml configs/phase2/experiments/expC_detrac_test.yaml`.

**Quy tắc**: sau thời điểm ghi bản khóa, không thay đổi cấu hình cho tập test và không tuning trên test.
Nếu phát hiện lỗi cài đặt phải sửa, ghi vào mục "Sửa sau khi khóa" ở cuối file (lỗi gì, cấu hình nào bị ảnh
hưởng, kết quả nào phải chạy lại) — không sửa lặng lẽ.

## 1. Tiêu chí lựa chọn (đăng ký trước, `configs/phase2/PREREG_moc2.md`)

HOTA ở điều kiện `c25` trên **dev** (12 video), chỉ coi là cải thiện khi CI 95 % bootstrap (COMBINED, lấy
mẫu lại theo video, 5000 lần, seed 0) **không chứa 0**. Kết quả dev:

| Trục | Ứng viên tốt nhất trên dev | ΔHOTA vs cấu hình hiện tại (CI 95 %) | Kết luận theo tiêu chí |
|:--|:--|:--|:--|
| C1 `track_buffer` | 3,0 s (EKF) | +0,0006 [−0,0004, +0,0016] so với 1,2 s (EKF); CV: mọi mức ≈ 0 | **không cải thiện rõ** → giữ **1,2 s** |
| C3 tham số nhiễu | EKF `std_theta` 0,01 / `std_omega` 0,01 | +0,0006 [−0,0006, +0,0018] so với 0,01/0,005 | **không cải thiện rõ** → giữ mặc định |
| C2 vòng ghép 3 cho track LOST | `lost_recover: maha` | CV +0,0019 [−0,0015, +0,0044]; EKF −0,0024 [−0,0055, +0,0011] | **không đạt tiêu chí HOTA** |

→ Cấu hình **được chọn theo tiêu chí** cho tập test = cấu hình hiện tại (bộ đệm 1,2 s, tham số nhiễu mặc
định, association ByteTrack gốc). Đó là các nhánh `cv`, `ekf`, `ukf`, `pf500` của `expA_detrac_test.yaml`
và toàn bộ `expB_detrac_test.yaml`.

## 2. Nhánh thăm dò (exploratory) — khai báo trước, KHÔNG phải "cấu hình được chọn"

C2 (`lost_recover: maha`, gate Mahalanobis theo hiệp phương sai của chính bộ lọc, `lost_recover_maha2`
= 5,991) **không** đạt tiêu chí HOTA nhưng cải thiện **có ý nghĩa** chỉ số sự kiện trên dev:
giữ ID qua che khuất `preserved` +0,049 [+0,011, +0,080] (CV) — bootstrap theo video, cùng 184 sự kiện;
tầng che hoàn toàn 0,091 → 0,152 (CV), 0,061 → 0,121 (EKF).

Vì hai chỉ số không thống nhất, `expC_detrac_test.yaml` chạy `cv-maha`/`ekf-maha` trên test như **nhánh
kiểm chứng giả thuyết**, ghi rõ trong báo cáo là kết quả thăm dò; không được trình bày như "cấu hình tốt
nhất đã chọn". Đối chứng `lost_recover: iou` chỉ chạy trên dev (đã đủ để tách vai trò của cổng bất định).

## 3. Cấu hình khóa

| Thí nghiệm | Điều kiện | Tracker | Ghi chú |
|:--|:--|:--|:--|
| `expA_detrac_test.yaml` | c25, c10 | cv, ekf, ukf, pf500 × seed {0,1,2} | so sánh **bộ lọc**; PF chỉ N = 500 (N lớn nhất đã khảo sát trên dev, cải thiện đơn điệu theo N) |
| `expB_detrac_test.yaml` | c25 | sort-orig1 (baseline), sort-orig30, sort-cv30, sort-ekf30, sort-ekf1, bt-ekf | so sánh **thuật toán**, mỗi cặp đổi một yếu tố |
| `expC_detrac_test.yaml` | c25 | cv, cv-maha, ekf, ekf-maha | nhánh thăm dò C2 |

Cố định cho cả ba: cache `yolov8n-coco_conf0.05_all` (conf 0,05, 80 lớp, max_det 3000, `yolov8n.pt` sha256
`f59b3d83…`), lọc lúc load: conf > 0,25 (c25) / > 0,10 (c10), lớp [2,3,5,7], max_det 300; `track_buffer`
1,2 s = 30 bước @ 25 fps; `elapsed_mode: skip`; 40 video `results/phase2/manifests/detrac_test.json`;
occlusion_eval `--part test --context-s 1.2`, bootstrap 5000 seed 0.

## 4. Sửa sau khi khóa

_(chưa có)_
