# Mốc 2 — báo cáo thực nghiệm chốt (UA-DETRAC)

Nhánh `phase2-mixed-traffic`, snapshot review độc lập `a8ecaf6` (**PASS WITH LIMITATIONS**, không còn lỗi
chặn). Mọi số trong tài liệu này lấy **trực tiếp từ file kết quả gốc** đã dẫn ở từng bảng; không gõ lại từ
báo cáo. Cấu hình khóa: [LOCK_moc2.md](../configs/phase2/LOCK_moc2.md) +
[lock_moc2.json](../results/phase2/lock/lock_moc2.json) (`created` 2026-09-16T02:33:51, **trước** mọi
`run_meta.created` của các run test). Khoảng tin cậy: **bootstrap ghép cặp theo video, 5000 lần, seed 0**,
tái tổ hợp COMBINED theo công thức TrackEval ([boot_stats.py](../src/boot_stats.py), kiểm chứng tới 1e-9).

**Giao thức chung**: cùng một detection cache (`yolov8n-coco_conf0.05_all`, conf 0,05, 80 lớp COCO,
`max_det` 3000, lọc lúc load) cho **mọi** tracker ⇒ khác biệt giữa các dòng **chỉ** do motion model /
vòng đời / association. Bộ lọc chạy ở **đơn vị bước** (dt = 1); `track_buffer` quy đổi theo giây → bước.

---

## 1. Phân biệt dev và test

| | dev | test |
|:--|:--|:--|
| Dữ liệu | **12 video** train UA-DETRAC (phân tầng che khuất × độ cong, seed 0) — [detrac_dev.json](../results/phase2/manifests/detrac_dev.json) | **40 video** test UA-DETRAC — [detrac_test.json](../results/phase2/manifests/detrac_test.json) |
| Dùng để | thăm dò, đăng ký trước, chọn/loại cấu hình (Nhóm C) | **chỉ chạy sau khi khóa**; không tuning, không ablation mới |
| Sự kiện che khuất | 184 | 1324 (182 ở tầng `full`, 1142 `partial`) |

Nhóm C **chỉ** chạy trên dev, trừ nhánh thăm dò C2 được đưa lên test để báo cáo kết quả âm (mục 4).

---

## 2. Nhóm A — so sánh bộ lọc, **theo từng điều kiện confidence**

Nguồn: [expA_detrac_test/{c25,c10}/summary.csv](../results/phase2/expA_detrac_test/),
`paired_pairs.csv` cùng thư mục.

**HOTA tuyệt đối (test, 40 video)**

| tracker | c25 | c10 |
|:--|--:|--:|
| cv (KF–CV, baseline) | 0.5824 | 0.5797 |
| ekf (EKF–CTRV) | **0.5835** | **0.5830** |
| ukf (UKF–CTRV) | 0.5824 | 0.5815 |
| pf500 (PF–CTRV, N=500, 3 seed) | 0.5519 | 0.5505 |

**Hiệu số từng cặp (COMBINED + CI 95 %)** — điều kiện là **yếu tố quyết định**:

| cặp | c25 | c10 |
|:--|:--|:--|
| ekf − cv, ΔHOTA | +0.0012 [−0.0003, +0.0027] — **CI chứa 0** | **+0.0033 [+0.0011, +0.0057]** |
| ekf − cv, ΔIDF1 | +0.0024 [−0.0004, +0.0051] | **+0.0053 [+0.0019, +0.0089]** |
| ekf − cv, ΔAssA | +0.0025 [−0.0004, +0.0056] | **+0.0072 [+0.0024, +0.0123]** |
| ekf − cv, ΔIDSW | +27 [−25, +77] | −18 [−74, +35] |
| ukf − cv, ΔHOTA | +0.0001 [−0.0012, +0.0014] | +0.0019 [−0.0002, +0.0041] |
| ukf − ekf, ΔHOTA | **−0.0011 [−0.0019, −0.0002]** | −0.0015 [−0.0040, +0.0007] |
| pf500 − ekf, ΔHOTA | **−0.0316 [−0.0389, −0.0255]** | **−0.0325 [−0.0394, −0.0264]** |

**dev (12 video, c25)** — [expA_detrac_dev/c25](../results/phase2/expA_detrac_dev/c25/): cv 0.5900,
ekf 0.5925, ukf 0.5879; ekf − cv **+0.0025 [−0.0018, +0.0060] (CI chứa 0)**. PF kém có ý nghĩa ở cả ba mức
N: pf100 −0.0967, pf300 −0.0718, pf500 −0.0631 (mọi CI không chứa 0). Khoảng cách **thu hẹp khi N tăng** ⇒
số hạt là **một phần** nguyên nhân; nhưng ở N = 500 PF vẫn kém rõ rệt, nên số hạt **chưa giải thích được toàn
bộ** — các yếu tố khác (thiết kế nhiễu, proposal, thiếu roughening) chưa được tách riêng.

**Đọc đúng mức bằng chứng**
- **CI chứa 0 ⇒ "không đủ bằng chứng về khác biệt", KHÔNG phải "tương đương".** Với cỡ mẫu và độ biến thiên
  ở đây, CI của `ekf − cv` tại c25 là [−0.0003, +0.0027] — khoảng này vẫn cho phép một khác biệt thật tới
  +0.0027 HOTA. Không có phép kiểm tương đương (equivalence test) nào được đăng ký trước, nên không kết luận
  được hai bộ lọc như nhau.
- Lợi ích của EKF so với CV là **nhỏ và phụ thuộc điều kiện**: có ý nghĩa thống kê ở **c10** (+0.0033 HOTA
  ≈ **+0,57 %** tương đối), **không** có ý nghĩa ở c25 và **không** có ý nghĩa trên dev. Không đủ cơ sở để
  phát biểu "EKF tốt hơn CV" mà không kèm điều kiện confidence và tập dữ liệu.
- UKF **không cho thấy** cải thiện so với CV, và **kém EKF một chút có ý nghĩa** ở c25.
- Ở **tầng che hoàn toàn** (`has_full` = sự kiện có đoạn che ≥ 0,90; 182 sự kiện, test c25):
  cv / ekf / ukf **đều 0,0604 (11/182)** — cả ba như nhau. Ở c10, ukf 12/182 (0,0659), cv/ekf vẫn 11/182:
  chênh **một** sự kiện, không có ý nghĩa. Phát biểu "cả ba như nhau" chỉ đúng ở **c25**.
  Nguồn: [occlusion_strata.csv](../results/phase2/expA_detrac_test/c25/occlusion_strata.csv).

---

## 3. Nhóm B — thuật toán, mỗi cặp đổi **đúng** thành phần được ghi

Dùng **`_fix1`** (sau khi sửa lỗi vòng đời SORT so với upstream `bce9f0d`; xem
[test_sort_upstream_parity.py](../src/test_sort_upstream_parity.py), lệch ≤ 5,8·10⁻⁵ px). Số của bản trước
khi sửa chỉ còn giá trị lịch sử. Nguồn:
[expB_detrac_test_fix1/c25/paired_pairs.csv](../results/phase2/expB_detrac_test_fix1/c25/paired_pairs.csv).

**HOTA tuyệt đối (test, c25)**: sort-orig1 0.5311 · sort-orig30 0.5585 · sort-cv30 0.5452 · sort-ekf30 0.5472
· sort-ekf1 0.5171 · bt-ekf 0.5835.

| cặp — **thành phần thay đổi** | ΔHOTA | ΔIDSW |
|:--|:--|:--|
| `sort-orig30 − sort-orig1`: **chỉ** bộ đệm `max_age` 1 → 30 (SORT nguyên bản) | **+0.0274 [+0.0201, +0.0345]** | **−1055 [−1390, −759]** |
| `sort-cv30 − sort-orig30`: **gói** biểu diễn trạng thái **+ thiết lập bộ lọc** — KF 7-D + Q/R/P₀ của SORT → XYAH + Q/R/P₀ của ByteTrack (cùng vòng đời SORT, cùng `max_age` 30) | **−0.0132 [−0.0185, −0.0077]** (thô −0.0132500) | +4 [−140, +156] |
| `sort-ekf30 − sort-cv30`: **chỉ** motion model CV → EKF–CTRV (cùng XYAH, cùng vòng đời SORT) | +0.0020 [−0.0001, +0.0042] (**CI chạm 0**) | **−108 [−186, −28]** |
| `sort-ekf1 − sort-orig1`: **chỉ** bộ lọc, ở `max_age` 1 | **−0.0140 [−0.0167, −0.0116]** | +186 [+62, +308] |
| `bt-ekf − sort-ekf30`: **chỉ** vòng đời/association SORT → ByteTrack (cùng EKF–CTRV) | **+0.0363 [+0.0296, +0.0428]** | **−1685 [−2090, −1292]** |

**dev `_fix1` (12 video)**: sort-orig1 0.5715 · sort-orig30 0.5854 · sort-cv30 0.5677 · sort-ekf30 0.5652 ·
sort-ekf1 0.5500 · bt-ekf 0.5925 —
[expB_detrac_dev_fix1/A/summary.csv](../results/phase2/expB_detrac_dev_fix1/A/summary.csv).

**Đọc đúng mức bằng chứng**: mỗi hiệu số chỉ có nghĩa **trong đúng cặp được kiểm soát**. Chúng **không** tạo
thành một phép phân rã cộng, cũng **không** phải thứ hạng áp dụng cho dataset khác. Cặp `sort-cv30 −
sort-orig30` đổi **cả** biểu diễn trạng thái **và** bộ tham số nhiễu ⇒ **không** được đọc là "đổi biểu diễn
trạng thái làm giảm HOTA".

---

## 4. Nhóm C — ba trục, mỗi trục đổi **một** thành phần

Đăng ký trước: [PREREG_moc2.md](../configs/phase2/PREREG_moc2.md) (giá trị, ngân sách, tiêu chí, giả thuyết).

### C1 — thời gian giữ track ([expC1_buffer_dev](../results/phase2/expC1_buffer_dev/c25/summary.csv))
6 mức `track_buffer` (0,4 → 4,0 s) × 2 bộ lọc. HOTA **bão hòa** từ 0,8 s trở lên (CV 0.5900 → 0.5898;
EKF 0.5896 → 0.5930, mọi CI chứa 0) và `preserved` **không tăng thêm một sự kiện nào** (CV giữ nguyên 0,5054
từ 0,8 s đến 4,0 s). Ngược lại **ghép nhầm tăng đơn điệu** (CV 2,5 % → 5,3 %; EKF 1,7 % → 3,0 %), FP và IDSW
tăng nhẹ. Buffer 0,4 s kém rõ ở cả hai bộ lọc → **giữ 1,2 s**. Nới thời gian giữ track **không** phải cách
cải thiện phục hồi sau che khuất trên UA-DETRAC.

### C2 — cơ chế nối lại track (vòng ghép thứ 3 cho track LOST, gate Mahalanobis)
Đây là **nhánh thăm dò**, **không** phải cấu hình được chọn. Nguồn:
[expC2_recover_dev](../results/phase2/expC2_recover_dev/c25/summary.csv),
[expC_detrac_test](../results/phase2/expC_detrac_test/c25/).

| | dev (12 video) | test (40 video) |
|:--|:--|:--|
| `preserved` chung | cv 0.505 → cv-maha **0.554**; cv-iou 0.565 | cv 0.480 → cv-maha **0.516**; ekf 0.487 → ekf-maha **0.526** |
| Tầng **che hoàn toàn** (182 sự kiện) | — | cv/ekf **0.0604** → cv-maha/ekf-maha **0.1209** — **gấp đôi** |
| ΔHOTA, cặp **đúng đối chứng** | cv-maha − cv +0.0019; ekf-maha − ekf −0.0025 | **cv-maha − cv −0.0046 [−0.0072, −0.0021]**; **ekf-maha − ekf −0.0034 [−0.0058, −0.0012]** |
| ΔIDSW | cv-maha −24; ekf-maha −2 | **cv-maha +261 [+103, +441]**; **ekf-maha +319 [+155, +495]** |
| FP | cv 12 119 → cv-maha 13 157 | cv 73 478 → cv-maha 81 428 (**+7 950**) |

**Kết luận C2**: khả năng **giữ ID qua che khuất tăng có ý nghĩa** — kể cả ở tầng che hoàn toàn, nơi mọi
motion model đều bế tắc. Nhưng trên test, **HOTA giảm có ý nghĩa** ở cả hai cặp đúng đối chứng, vì FP tăng
~8 000 và IDSW tăng 261…319. Đây là ví dụ trực tiếp cho nguyên tắc: **khả năng phục hồi ID tăng KHÔNG đồng
nghĩa chất lượng tracking tổng thể tăng**. Hai chỉ số đo hai thứ khác nhau (`preserved` là chỉ số **sự kiện**,
HOTA là chỉ số **toàn cục**) và ở đây chúng đi **ngược chiều** ⇒ **không bật** cơ chế này nếu mục tiêu là HOTA.

### C3 — tham số nhiễu ([expC3_noise_dev](../results/phase2/expC3_noise_dev/c25/summary.csv))
21 cấu hình công bố trước (EKF 4×3 `std_theta` × `std_omega`; CV 3×3 `std_weight_position` × `..._velocity`),
detector/association/buffer cố định. **Không cấu hình nào cải thiện HOTA có ý nghĩa**: EKF tốt nhất 0.5931 so
với 0.5925 hiện tại (ΔHOTA +0.0006 [−0.0007, +0.0013]). **Một số cấu hình CV kém hơn có ý nghĩa** — ví dụ
`std_weight_position` 1/10 với `std_weight_velocity` 1/320: **−0.0094 [−0.0197, −0.0019]**. Tham số **thực sự
có hiệu lực** (đã kiểm chứng riêng: covariance đổi rõ, `P[θ,θ]` trung bình 2,46 / 2,83 / 3,88 theo
`std_omega`; 0/12 video bit-identical) ⇒ kết quả âm **không** phải do override bị bỏ qua.
Tỉ lệ "~99 % số dòng lệch dưới 1 px" **chỉ áp dụng cho nhánh EKF**; nhánh CV chỉ 61–83 %.

**Ma trận ablation** ([ablation_matrix.csv](../results/phase2/ablation_matrix.csv)): không trục nào đạt tiêu
chí đã đăng ký ⇒ **không có tổ hợp nào để chạy**. Đây là kết quả âm và được báo cáo như vậy.

---

## 5. PF–CTRV — giới hạn kết luận

Đo được ([pf_diag](../results/phase2/pf_diag/expA_detrac_dev/), 3 seed × 3 mức N): PF kém rõ rệt ở mọi chỉ số
trên cả dev và test; biến thiên giữa 3 seed rất nhỏ (trên test: SD HOTA **0,00054** ở c25 và 0,00043 ở
c10; SD lớn nhất trong các chỉ số là AssA **0,00090**) ⇒ chênh lệch **không** do seed. Ở các lần **cập nhật sau một khoảng mất dấu dài**: ESS/N ~1 % (đo **trước** resample) và chỉ còn ~3 %
particle phân biệt (đo **sau** resample).

**Không** suy ra được từ log này rằng mất đa dạng xảy ra *trước khi có quan sát mới*, cũng **không** phải bằng
chứng đó là *nguyên nhân* của các lần nối lại thất bại (những lần đó không có bước cập nhật nên không sinh
dòng log). Kết luận về PF **giới hạn trong**: implementation bootstrap PF của repo này, các mức N đã thử
(100/300/500), bộ tham số nhiễu đã thử, và dữ liệu UA-DETRAC. Thiết kế khác (roughening, proposal khác, Q
riêng cho PF) **chưa chạy** ⇒ không kết luận "PF không phù hợp cho bài toán này".

---

## 6. Tốc độ — **hai đại lượng khác nhau, không được trộn**

### 6a. Tracker chạy **từ cache** (CPU, **không** có detector)

**Ba cột trong `summary.csv` KHÔNG phải ba pha độc lập** ([run_tracking.py](../src/run_tracking.py):177–183):

| cột | đo đúng cái gì |
|:--|:--|
| `update_ms_per_frame` | **tổng** một lời gọi `tracker.update()` — đây là chi phí tracker mỗi frame |
| `predict_ms_per_frame` | **chỉ** phần `multi_predict` (bước dự đoán của bộ lọc) |
| `assoc_ms_per_frame` | **phần còn lại** = `update − predict`: association **cộng với** bước cập nhật (correction) của bộ lọc, quản lý vòng đời track, loại track trùng, khởi tạo track mới |

Vì `assoc` chứa cả bước cập nhật của bộ lọc, **CV và EKF không có lý do gì phải bằng nhau ở cột này**.

| tracker (test c25) | `update` ms (= chi phí tracker) | trong đó `predict` ms | phần còn lại ms | fps *chỉ tracker* |
|:--|--:|--:|--:|--:|
| cv | 2,041 | 0,285 | 1,755 | ~490 |
| ekf | 2,661 | 0,664 | 1,997 | ~376 |
| ukf | 5,457 | 3,218 | 2,240 | ~183 |
| pf500 | 8,792 | 3,460 | 5,332 | ~114 |

Nhóm B (test c25), theo `update_ms`: sort-orig30 ~882 · sort-orig1 ~842 · sort-ekf1 ~708 · sort-cv30 ~650 ·
sort-ekf30 ~619 · bt-ekf ~502 fps.

> Các số này đo **trong lúc chạy đánh giá**, khi máy còn chạy việc khác ⇒ chỉ dùng để **so tương đối giữa các
> bộ lọc**, không phải benchmark.

### 6b. **Toàn pipeline** có detector (PC, GTX 1650)
[results/phase2/profile/*.json](../results/phase2/profile/) — MVI_39031, 450 frame sau 50 warm-up, batch 1,
960×540 → 640:

| cấu hình | fps end-to-end | detector_wall ms | tracker ms |
|:--|--:|--:|--:|
| ByteTrack + KF–CV | **51,4** | 14,9 | 0,78 |
| ByteTrack + UKF–CTRV | 48,1 | 15,4 | 1,90 |
| ByteTrack + PF–CTRV (**N=300**) | 46,2 | 15,9 | 2,22 |
| ByteTrack + EKF–CTRV | 45,4 | 17,1 | 1,02 |

**Cảnh báo khi đọc bảng 6b**: thứ tự fps end-to-end **không** khớp chi phí tracker (UKF nhanh hơn EKF
end-to-end dù tracker UKF đắt gấp ~1,9 lần). Nguyên nhân là `detector_wall` dao động giữa các lần chạy
(14,9–17,1 ms, tức 2,2 ms — lớn hơn toàn bộ chênh lệch tracker 0,24 ms giữa CV và EKF). Vì vậy **chỉ bảng 6a
mới so được chi phí của motion model**; bảng 6b cho biết **ngân sách tổng** và cho thấy trên phần cứng này
**detector chiếm 73,6–77,9 % thời gian**, tracker 4,0–10,2 % (tính từ `stages_ms` trong
[profile/*.json](../results/phase2/profile/): `detector_wall_ms.mean / total_ms.mean`). Bản profile PF dùng
**N=300**, không phải N=500 của bảng kết quả — không so chéo hai con số này.

**Chưa đo trên Jetson Nano.** Mọi số ở trên là PC; xem [scripts/nano/README.md](../scripts/nano/README.md).

---

## 7. Giới hạn provenance và phạm vi kết luận

| Hạng mục | Trạng thái hiện tại | Ý nghĩa |
|:--|:--|:--|
| Cache detection | [cache_consistency.csv](../results/phase2/cache_consistency.csv): **56 sequence = 3 `ok` + 53 `legacy` + 0 `mismatch`** | Chỉ **3** sequence (fixture, sinh sau khi có fingerprint) xác minh được bằng `spec_sha256`. **53 sequence `legacy`** — gồm **toàn bộ 52 sequence UA-DETRAC dùng cho luận văn** — sinh trước đó, **không có fingerprint nên không xác minh được bằng hash**. Việc mỗi `cache-id` chỉ có **một** giá trị `conf`/`max_det`/`weights_sha256`/`versions` là thuộc tính của **một file metadata được chép vào mọi dòng**, **không** phải bằng chứng độc lập. Không phát hiện mismatch trong phần kiểm tra được; **không** bổ sung fingerprint hồi tố |
| Thư mục run | [run_provenance_audit.csv](../results/phase2/run_provenance_audit.csv): **134 thư mục = 21 `OK` + 113 `LEGACY` + 0 nghi vấn** | 21 `OK` gồm 12 run Nhóm B `_fix1` có **sha256 từng file tracking khớp `eval_meta.json`**. 113 `LEGACY` chạy bởi bản runner cũ (chưa có `eval_meta`/`video_meta`) ⇒ **không xác minh được bằng fingerprint**; 0 dấu hiệu bất thường nhưng cũng **không chứng minh được** |
| `video_meta/` | 0/129 thư mục run bàn giao | Chỉ có ở run sinh sau bản sửa vòng 2 |

Không sửa lại số liệu lịch sử, không mở thêm tuning/ablation trên test.

---

## 8. Kết luận chốt của mốc 2

1. **Trên UA-DETRAC, khác biệt đo được giữa KF–CV, EKF–CTRV và UKF–CTRV là rất nhỏ.** EKF hơn CV **có ý
   nghĩa chỉ ở c10** (+0.0033 HOTA, ≈ +0,57 % tương đối); ở c25 và trên dev **CI chứa 0 ⇒ không đủ bằng
   chứng về khác biệt**, không phải bằng chứng về sự tương đương. UKF không cho thấy cải thiện so với CV và
   kém EKF một chút ở c25.
2. **Yếu tố ảnh hưởng lớn hơn motion model một bậc**: vòng đời/association (SORT → ByteTrack, +0.0363) và bộ
   đệm (`max_age` 1 → 30, +0.0274) — lớn hơn ~15–18 lần so với đổi motion model trong cùng khung (+0.0020).
3. **Trong khoảng đã thử, motion model không phải yếu tố quyết định ở tầng che hoàn toàn**: cả ba bộ lọc
   đều chỉ giữ được ~6 % ID. Điều này **không loại trừ** motion model là một phần nguyên nhân — ba bộ lọc
   này chia sẻ cùng một họ giả định (chuyển động trơn, nhiễu Gauss) và cùng một khung association, nên chúng
   có thể cùng thất bại vì cùng một lý do. Một họ dự đoán khác hẳn (ví dụ mô hình học được, dùng ngoại cảnh)
   chưa được thử. Phân tích
   nguyên nhân (dev, 184 sự kiện đủ điều kiện — [cause_summary.csv](../results/phase2/failure_analysis/expA_detrac_dev/cause_summary.csv))
   cho thấy **`prediction_drift` là nguyên nhân số 1** ở tầng `full`, c25: **cv 11/22 ca (50 %), ekf 13/23
   (56 %), ukf 14/23 (61 %)**. Nhãn này được gán khi *track còn sống tại frame detector thấy lại GT, nhưng
   **IoU(box dự đoán, det) < 0,2*** ([cause_evidence_c25.csv](../results/phase2/failure_analysis/expA_detrac_dev/cause_evidence_c25.csv)) —
   tức đã thấp hơn rào `match_thresh` (0,8 trên cost 1 − IoU·conf) rất xa. Nhãn nguyên nhân là **suy luận có
   quy tắc** từ số liệu, không phải quan sát trực tiếp.
4. **Tăng khả năng phục hồi ID không kéo theo chất lượng tổng thể**: C2 làm `preserved` ở tầng che hoàn toàn
   **gấp đôi** nhưng HOTA **giảm có ý nghĩa** trên test ở cả hai cặp đúng đối chứng.
5. **Ba trục Nhóm C đều cho kết quả âm** theo tiêu chí đã đăng ký trước ⇒ giữ cấu hình đã khóa.
6. **PF–CTRV kém rõ rệt** trong implementation và các cấu hình đã thử; không mở rộng kết luận ra ngoài phạm
   vi đó.
7. **Chi phí tính toán**: trên PC, detector chiếm **73,6–77,9 %** thời gian; chênh lệch CV ↔ EKF ở phần
   tracker là ~0,24 ms/frame (`tracker_ms` toàn pipeline) / ~0,62 ms/frame (`update_ms` chạy từ cache).
   **Chưa có số trên Jetson Nano.**

**Việc còn lại (mốc 3)**: BDD100K (chưa có dữ liệu trên máy — xem
[BDD100K_INVENTORY.md](BDD100K_INVENTORY.md)), đo trên Jetson Nano thật, và huấn luyện lại detector cho 5 lớp
(thí nghiệm riêng, sau khi pipeline tracking và quy trình đánh giá đã ổn định).
