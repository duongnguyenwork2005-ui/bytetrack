# BẢN DEMO NHANH — kết quả để xem trước

> **Đây là bản demo, KHÔNG phải kết quả chính thức.** Chạy trên 8/60 video train,
> nhiều chỗ làm tắt (liệt kê ở mục 3). Mọi thứ nằm trong `results/demo/`,
> **không ghi đè bất cứ kết quả cũ nào**.
>
> Tái lập: `python src/demo/gt_omega.py` → `case_ab.py` → `render_videos.py`

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
| FDE tính từ **quỹ đạo GT**, không phải trạng thái thật của tracker | Đây là **cận trên lạc quan** — tracker thật có thể đã mất track từ trước | Lấy trạng thái thật từ tracker (cần sửa `baseline_track.py` để log) |
| Chỉ vẽ KF và EKF, **không có UKF** | Đúng đặc tả nhưng thiếu 1/3 bức tranh | Thêm màu thứ 4 |
| Chưa có thanh tiến trình / đánh dấu thời điểm phân kỳ | Khó thấy chính xác lúc nào 2 model tách nhau | Thêm timeline dưới đáy |

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
