Cấu hình được **khóa trước** khi chạy (`configs/phase2/LOCK_moc2.md`, `results/phase2/lock/lock_moc2.json`,
commit `a64cb754`, ghi lúc **02:33:51**); lần chạy test sớm nhất lúc **02:44:45** → **22/22 lần chạy sau khi
khóa**. Không có tuning hay ablation mới sau khi xem kết quả test. Chi tiết: `PHASE2_REPORT.md` mục 16.

**A — bộ lọc** (40 video, 1324 sự kiện che khuất; COMBINED + CI 95 % bootstrap ghép cặp theo video):

| cặp | c25 (conf > 0,25) | c10 (conf > 0,10) |
|:--|:--|:--|
| EKF − CV (ΔHOTA) | +0.0012 [−0.0003, +0.0027] | **+0.0033 [+0.0011, +0.0057]** |
| UKF − CV | +0.0001 [−0.0012, +0.0014] | +0.0019 [−0.0002, +0.0041] |
| UKF − EKF | **−0.0011 [−0.0019, −0.0002]** | −0.0015 [−0.0040, +0.0007] |
| PF500 − EKF | **−0.0316 [−0.0389, −0.0255]** | **−0.0325 [−0.0394, −0.0264]** |

HOTA tuyệt đối (c25): cv 0.5824, ekf 0.5835, ukf 0.5824, pf500 0.5519. Giữ ID qua che khuất: cv 0.480,
ekf 0.487, ukf 0.485, pf500 0.435. **Tầng che nặng** (`has_full` = sự kiện **có đoạn che ≥ 0,90**, không
phải che 100 % ở mọi frame; 182 sự kiện): **tại c25** cả ba bộ lọc đều 0,060 (11/182); **tại c10** UKF 12/182
(0,066), CV/EKF vẫn 11/182 — chênh **một** sự kiện, không có ý nghĩa thống kê. Phát biểu "cả ba như nhau" chỉ
áp dụng cho **c25**.

**B — thuật toán** (c25, mỗi cặp chỉ đổi một yếu tố) — **số của bản `_fix1`**, tức sau khi sửa lỗi vòng đời
SORT (`results/phase2/expB_detrac_test_fix1/c25/paired_pairs.csv`); số trong ngoặc là bản trước khi sửa, chỉ
còn giá trị lịch sử:

| cặp (yếu tố được kiểm soát) | ΔHOTA `_fix1` | (trước khi sửa) | ΔIDSW `_fix1` |
|:--|:--|:--|:--|
| bộ đệm `max_age` 1 → 30 | **+0.0274 [+0.0201, +0.0345]** | (+0.0280) | −1055 [−1390, −759] |
| **gói** biểu diễn trạng thái **+ thiết lập bộ lọc** (KF 7-D + Q/R/P₀ của SORT → XYAH + Q/R/P₀ của ByteTrack) | **−0.0132 [−0.0185, −0.0077]** | (−0.0133) | +4 [−140, +156] (CI chứa 0) |
| motion model CV → EKF–CTRV (cùng XYAH, cùng vòng đời SORT) | +0.0020 [−0.0001, +0.0042] (CI chạm 0) | (+0.0020) | **−108 [−186, −28]** |
| chỉ đổi bộ lọc, `max_age` 1 | **−0.0140 [−0.0167, −0.0116]** | (−0.0140) | +186 [+62, +308] |
| vòng đời/association SORT → ByteTrack (cùng EKF) | **+0.0363 [+0.0296, +0.0428]** | (+0.0364) | −1685 [−2090, −1292] |

Mỗi hiệu số chỉ có nghĩa **trong đúng cặp được kiểm soát** — không phải một phép phân rã cộng, cũng không phải
thứ hạng áp dụng cho mọi dataset. `bt-ekf` **bit-identical** giữa hai bản (40/40 video test).

**C — nhánh thăm dò** (vòng ghép thứ 3 cho track LOST, gate Mahalanobis; **không** phải cấu hình được chọn):
giữ ID qua che khuất **tăng có ý nghĩa** (+0.036 [+0.019, +0.051] với CV; tầng che hoàn toàn 0,060 → **0,121**)
nhưng HOTA **giảm có ý nghĩa** (cv −0.0046 [−0.0072, −0.0021]; ekf −0.0034 [−0.0058, −0.0012]) vì FP +8 000 và
IDSW +261…+319 → **kết quả âm, không bật cơ chế này nếu mục tiêu là HOTA**.

**Tốc độ tracker từ cache** (56 340 frame, CPU, không có detector, đo khi máy còn chạy việc khác → tham khảo):
sort-orig1 774 fps, bt-ekf 498, cv 420–570, ekf 333–478, ukf 175–187, pf500 106–114. Toàn pipeline có detector
(GTX 1650): 45–51 fps (mục 7.3). **Chưa đo trên Nano.**

Kết quả đầy đủ: `results/phase2/{expA,expB,expC}_detrac_test/<cond>/{summary.csv, paired_bootstrap_combined.csv,
paired_pairs.csv, occlusion_strata.csv, bootstrap_DETRAC-test.csv}`, log chạy trong `handoff/logs/exp*_detrac_test.log`.
