# BẢN DEMO NHANH — kết quả để xem trước

> **Đây là bản demo, KHÔNG phải kết quả chính thức.** Chạy trên 8/60 video train,
> nhiều chỗ làm tắt (liệt kê ở mục 3). Mọi thứ nằm trong `results/demo/`,
> **không ghi đè bất cứ kết quả cũ nào**.
>
> Tái lập: `python src/demo/gt_omega.py` → `case_ab.py` → `render_videos.py`
>
> **Bổ sung mục 5 (Giai đoạn E):** khi xem video minh hoạ đã phát hiện một khuyết điểm
> thật trong cài đặt EKF — `ω` không bị chặn. Đã đo, đã sửa, đã chạy lại 60 video:
> **không đảo ngược kết quả**. Số liệu mục 1–4 giữ nguyên.

---

## 1. Tỉ lệ Case A / Case B ở 3 mức ngưỡng

**236 đoạn che khuất, 8 video train.**

| Ngưỡng \|ω\| | n | %CASE_A | %CASE_B | %STRAIGHT | %CASE_C |
|---|---|---|---|---|---|
| 0.10 °/frame | 236 | 17.4% | 14.8% | 41.1% | 26.7% |
| 0.25 °/frame | 236 | 5.1% | 14.8% | 69.9% | 10.2% |
| 0.50 °/frame | 236 | **1.3%** | **13.6%** | 83.1% | 2.1% |

Số tuyệt đối:

| Ngưỡng | n_CASE_A | n_CASE_B | n_STRAIGHT | n_CASE_C |
|---|---|---|---|---|
| 0.10 | 41 | 35 | 97 | 63 |
| 0.25 | 12 | 35 | 165 | 24 |
| 0.50 | 3 | 32 | 196 | 5 |

**Định nghĩa:** CASE_A = khúc cua đã hình thành trước khi che (CTRV có cơ sở ngoại suy) ·
CASE_B = khúc cua phát sinh trong lúc mất quan sát (không model nào đoán được) ·
CASE_C = đổi chiều quay giữa đoạn che · STRAIGHT = đi thẳng.

### Đọc kết quả

**CASE_A không phải đa số — nó là thiểu số nhỏ và teo rất nhanh khi siết ngưỡng**
(17.4% → 5.1% → 1.3%). Phần lớn "khúc cua hình thành trước" chỉ là dao động quanh
ngưỡng, không phải khúc cua thật.

**CASE_B ổn định ~14–15% ở mọi ngưỡng** — đây là con số vững nhất trong bảng.

**CASE_A < CASE_B ở MỌI ngưỡng.** Gộp B+C (đều là khúc cua không đoán được):

| Ngưỡng | Không đoán được (B+C) | Đoán được (A) | Tỉ lệ |
|---|---|---|---|
| 0.10 | 41.5% | 17.4% | 2.4× |
| 0.25 | 25.0% | 5.1% | 4.9× |
| 0.50 | 15.7% | 1.3% | **12×** |

→ **Ủng hộ mạnh kết luận "trần thông tin"** ở Giai đoạn A (ω GT trước che chỉ dự đoán
đúng dấu ω GT trong che 58.2% số lần, Pearson +0.0093). Không mâu thuẫn.

### Tỉ lệ giữ ID theo nhóm (ghép được, nhưng mẫu quá nhỏ)

Ngưỡng 0.25 °/frame, detector COCO, `buffer=30` — 198/236 đoạn ghép được:

| Nhóm | n | KF + CV | EKF + CTRV | UKF + CTRV |
|---|---|---|---|---|
| CASE_A | **10** | 20.0% | 20.0% | 20.0% |
| CASE_B | 30 | 30.0% | 33.3% | 33.3% |
| CASE_C | 21 | 23.8% | 23.8% | 23.8% |
| STRAIGHT | 137 | 28.5% | 27.7% | 27.7% |

Ở CASE_A — nơi CTRV *lẽ ra* phải thắng — cả 3 model **giống hệt nhau**. Nhưng n=10,
không kết luận được gì.

---

## 2. Mỗi video được chọn từ bao nhiêu đoạn ứng viên

Chọn bằng **tiêu chí tự động**, không chọn tay (ngưỡng phân loại 0.25 °/frame):

| Video | Tiêu chí chọn | Chọn từ | Đoạn được chọn | FDE KF | FDE EKF |
|---|---|---|---|---|---|
| `1_ctrv_thang.mp4` | CASE_A, `FDE_CV − FDE_CTRV` lớn nhất | **12 đoạn** | MVI_40992 t44, f2143–2160 | 53.1px | **31.6px** |
| `2_cv_thang.mp4` | STRAIGHT, `FDE_CTRV − FDE_CV` lớn nhất | **165 đoạn** | MVI_63553 t134, f1001–1066 | **247.6px** | 845.3px |
| `3_tran_thong_tin.mp4` | CASE_B, `T_occ` dài nhất | **35 đoạn** | MVI_40992 t21, f1084–1126 | 63.4px | 67.4px |
| `4_tran_detector.mp4` | Cả 3 model `lost`, `T_occ` gần trung vị | **88 đoạn** | MVI_40241 t133, f967–991 | 141.2px | 134.4px |

Video: 960×540, 10 fps, 3.0–7.6 giây. Xem trước 2 frame tĩnh ở
`preview_1_ctrv_thang.png` và `preview_2_cv_thang.png`.

> **Lưu ý về cỡ nhóm:** pool của Video 1 chỉ có **12 đoạn** — đó chính là số khúc cua
> "hình thành trước" thật sự trong 8 video. Nếu dùng ngưỡng lỏng 0.10 °/frame thì pool
> là 41. Con số nhỏ này **tự nó là một kết quả**, không phải hạn chế kỹ thuật.

---

## 3. Những chỗ đã làm tắt — cần sửa nếu làm thật

### 3.1 Ảnh hưởng đến CON SỐ (phải sửa)

| Chỗ làm tắt | Ảnh hưởng | Nếu làm thật |
|---|---|---|
| Chỉ 8/60 video train, chọn theo "nhiều đoạn che nhất" | **Thiên lệch mẫu** — video nhiều che khuất chưa chắc đại diện | Chạy cả 100 video (60 train + 40 test) |
| **622/862 đoạn (72%) bị loại** vì track có <10 frame trước lúc che | Mẫu thiên về track đã ổn định | Báo cáo riêng nhóm này; nó cũng là một phát hiện: phần lớn đoạn che **không có đủ lịch sử** cho bất kỳ motion model nào |
| Ghép tỉ lệ giữ ID dùng kết quả **detector COCO + `buffer=30`** (cấu hình cũ nhất) | Giai đoạn C cho thấy detector tốt hơn **xoá** ưu thế biểu kiến của EKF | Ghép với kết quả detector fine-tune |
| Video 4 xác định `lost` cũng từ cấu hình cũ đó | Như trên | Như trên |
| Chưa kiểm định thống kê cho bảng Case A/B | Chưa biết tỉ lệ có ổn định không | Bootstrap CI cho từng tỉ lệ |

### 3.2 Quyết định kỹ thuật đã thêm (không có trong yêu cầu nhưng bắt buộc)

**Lọc frame có tốc độ < 1.5 px/frame** khi tính ω. Không có trong đặc tả, nhưng bắt
buộc: khi xe gần đứng yên, hướng di chuyển không xác định và nhiễu annotation vài pixel
bị đọc thành "quay rất nhanh". Đã đo ở giai đoạn trước: trên đoạn xe đứng yên, **mọi**
cách đo độ cong đều bị thổi phồng **5.9–7.2 lần**. Bỏ bước này thì bảng Case A/B sai hoàn toàn.

### 3.3 Ảnh hưởng đến VIDEO (chất lượng render)

| Vấn đề | Biểu hiện | Cách sửa |
|---|---|---|
| **Dự đoán bay ra ngoài khung hình thì không nhìn thấy** | Video 2: EKF sai 845px → box biến mất, người xem tưởng lỗi render | Vẽ mũi tên ở mép khung chỉ hướng + khoảng cách |
| **Kích thước box trôi khi ngoại suy dài** | Box phình to bất thường (thấy rõ ở Video 2) | Khoá `w/h` về giá trị lúc bắt đầu che, hoặc ghi chú rõ |
| FDE tính từ **quỹ đạo GT**, không phải trạng thái thật của tracker | Đây là **cận trên lạc quan** — tracker thật có thể đã mất track từ trước | ✅ **ĐÃ LÀM** — `src/demo/side_by_side.py` vẽ thẳng box+ID mà ByteTrack thực sự xuất ra |
| Chỉ vẽ KF và EKF, **không có UKF** | Đúng đặc tả nhưng thiếu 1/3 bức tranh | Chưa làm. Mức ưu tiên thấp: mục 5.2 cho thấy UKF thua cả hai ở mọi chỉ số |
| Chưa có thanh tiến trình / đánh dấu thời điểm phân kỳ | Khó thấy chính xác lúc nào 2 model tách nhau | ✅ **ĐÃ LÀM** — `side_by_side.py --diff-segments` viền đỏ + nhãn ở đúng đoạn 2 model khác kết cục; `long_compare.py` có timeline dưới đáy |
| Hai model vẽ **chồng lên nhau** trên cùng khung → box đè nhau, khó đọc | Thấy rõ khi 2 dự đoán gần nhau | ✅ **ĐÃ LÀM** — `long_compare.py` tách 2 panel cạnh nhau, mỗi panel 1 model + GT |
| Video chỉ ~3 giây, chỉ 1 đoạn che | Không thấy được hành vi lặp lại | ✅ **ĐÃ LÀM** — `long_compare.py` chạy suốt đời track qua nhiều lần che (16–24 giây) |

---

## 4. Ước lượng làm bản đầy đủ

**Thời gian máy** (100 video, mỗi loại 3–5 video):

| Bước | Ước lượng |
|---|---|
| `gt_omega.py` trên 100 video | ~5 phút |
| Tính FDE (~3.000–4.000 đoạn) | ~10 phút |
| Render 12–20 video | ~10 phút |
| **Tổng thời gian máy** | **~25 phút** |

**Thời gian sửa các chỗ làm tắt** (mục 3) — đây mới là phần chính:

| Việc | Ước lượng |
|---|---|
| Ghép lại với kết quả detector fine-tune (3.1) | ~1 giờ |
| Bootstrap CI cho bảng Case A/B | ~1 giờ |
| Sửa render: mũi tên ngoài khung, khoá kích thước box, thêm UKF, timeline | ~2 giờ |
| Lấy trạng thái thật của tracker thay vì quỹ đạo GT | ~2–3 giờ *(việc nặng nhất — phải sửa pipeline để log trạng thái mỗi frame)* |
| **Tổng** | **~6–7 giờ** |

**Đề xuất:** nếu chỉ cần hình minh hoạ cho luận văn thì **bỏ mục cuối** (dùng quỹ đạo GT
là chấp nhận được nếu ghi chú rõ) → còn **~4 giờ + 25 phút máy**.

---

## 5. CẬP NHẬT — Giai đoạn E: khuyết điểm `ω` không bị chặn

> Phần này thêm sau, khi xem video minh hoạ phát hiện box dự đoán của EKF **quay
> vòng tròn tại chỗ**. Toàn bộ số liệu mục 1–4 ở trên **không đổi**.

### 5.1 Chuyện gì xảy ra

CTRV ngoại suy trên đường tròn bán kính `R = v/|ω|`, sau `T` frame quét một cung
`arc = |ω|·T`. Quay tròn **tự nó là hành vi đúng** khi `ω` lớn — nên câu hỏi đúng
không phải "code có sai không" mà "`ω` đó có hợp lý không".

- **Toán không sai:** 26/26 unit test PASS, gồm so Jacobian giải tích với sai phân số
  và kiểm tra đi hết một vòng tròn về đúng chỗ cũ (sai lệch 3,6·10⁻¹³ px).
- **Nhưng không có gì chặn `ω`.** Ngưỡng vật lý: xe rẽ ngã tư quét ~90° trong 2–4 giây
  = 0,013–0,031 rad/frame. Lấy rộng rãi `|ω| > 0,05 rad/frame` (72 °/giây) là bất khả thi.

| Chỉ tiêu (50 đoạn che, 48 track dài) | Bản gốc | Chặn ω |
|---|---|---|
| Trung vị \|ω\| | 0,366 °/frame *(GT: 0,441)* | 0,258 °/frame |
| Phân vị 90 | **19,9 °/frame** | 2,45 °/frame |
| Lớn nhất | **111,4 °/frame** | 2,87 °/frame |
| Vượt ngưỡng bất khả thi | **28 %** | **0 %** |
| `R` < 100 px *(nhỏ hơn chiếc xe)* | 26 % | 8 % |
| Quét **> 360°** *(trọn một vòng)* | **20 %** | **0 %** |

### 5.2 Sửa rồi thì sao — chạy lại đủ 60 video

| Model | HOTA | AssA | IDF1 | IDSW |
|---|---|---|---|---|
| KF + CV (baseline) | **0,6104** | **0,6541** | 0,7783 | **2251** |
| EKF + CTRV nguyên bản | 0,6101 | 0,6536 | 0,7782 | 2290 |
| **EKF + CTRV chặn ω** | 0,6102 | 0,6537 | **0,7783** | 2280 |

Mức từng đoạn (1.544 đoạn): chặn `ω` chỉ đổi kết cục **3 đoạn (0,19 %)**.
McNemar KF vs EKF chặn ω: 19 so 17, **p = 0,8679**. **Không đảo ngược kết quả.**

> **Phát hiện ngược chiều:** chặn `ω` có thể làm sai số **TĂNG**. MVI_40992 t12 đoạn 2:
> FDE **556,5 → 1208,4 px**. Vì khi `ω` lớn, box quay tít trong vòng tròn nhỏ nên **vô
> tình nằm gần chỗ cũ**; chặn `ω` lại cho nó bay gần như thẳng ra xa. Đây là lý do FDE
> **trung vị** giảm (114,5 → 105,1 px) nhưng FDE **trung bình** lại tăng (236,7 → 244,5 px).

### 5.3 Ảnh hưởng tới chính bản demo này — gần như không

Render lại toàn bộ bằng bản chặn `ω`. Trên **240 đoạn** của 8 video demo, chỉ **5 đoạn
(2,1 %)** đổi FDE (3 tốt hơn, 2 tệ hơn); FDE trung bình 165,1 → 165,0 px.
**4 video ở mục 2 không đổi một pixel nào:**

| Video | FDE KF | EKF gốc | EKF chặn ω |
|---|---|---|---|
| `1_ctrv_thang` | 53,1 px | 31,6 px | 31,6 px |
| `2_cv_thang` | 247,6 px | 845,3 px | 845,3 px |
| `3_tran_thong_tin` | 63,4 px | 67,4 px | 67,4 px |
| `4_tran_detector` | 141,2 px | 134,4 px | 134,4 px |

Đáng chú ý: `2_cv_thang` (EKF sai 845 px) có `|ω| = 0,0317 rad/frame` — **vốn đã dưới
ngưỡng chặn**. Sai số đó **không đến từ khuyết điểm này** mà từ trần thông tin: xe cua
trước khi bị che rồi đi thẳng suốt 66 frame.

### 5.4 Quét 48 track dài — chống cherry-picking

Quét **mọi** track ≥150 frame, ≥2 đoạn che, có di chuyển thật:

| | Số track | Tỉ lệ |
|---|---|---|
| CTRV tốt hơn | 5 | 10 % |
| **CV tốt hơn** | **32** | **67 %** |
| Hoà | 11 | 23 % |

FDE trung bình KF **266,0 px** vs EKF **297,4 px**. Tương quan giữa **góc cua của xe**
và lợi thế CTRV: **−0,021** — bằng không. Ba track cua gắt nhất (132°, 148°, 151°) thì
CTRV thua cả ba.

### 5.5 File sinh ra

| Script | Sản phẩm |
|---|---|
| `src/diagnose_ekf_circle.py` *(`--clamped`)* | `ekf_circle_diagnosis[_clamped].csv` |
| `src/omega_clamp_experiment.py` | `clamp_test.csv` |
| `src/demo/long_compare.py` *(`--clamped`)* | `videos/long_<video>_t<id>[_clamped].mp4` |
| `src/demo/render_videos.py` *(`--clamped`)* | `videos/<n>_*[_clamped].mp4`, `selected_segments[_clamped].csv` |
| `src/demo/side_by_side.py` *(`--ekf-tracker`)* | `sbs_<video>_<a>_<b>[_clamped].mp4` |

Video `long_*` và `sbs_*` đã gitignore vì nặng 10–121 MB; tái tạo bằng script ở trên.
