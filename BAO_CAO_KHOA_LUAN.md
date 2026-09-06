# Báo cáo tổng hợp khoá luận: So sánh KF / EKF / UKF cho Multi-Object Tracking trên UA-DETRAC

> Tài liệu này tự chứa — đọc xong là nắm được toàn bộ tình hình, không cần bối cảnh khác.
> Mục đích: nhờ hỗ trợ viết luận văn / phản biện phương pháp.

---

## 1. Đề tài và thiết lập

**Câu hỏi nghiên cứu ban đầu:** Mô hình chuyển động phi tuyến CTRV (Constant Turn Rate and
Velocity) có giúp giữ định danh (ID) của xe tốt hơn mô hình tuyến tính CV (Constant
Velocity) khi xe **vừa rẽ vừa bị che khuất** không?

**Pipeline:** YOLOv8n (pretrained COCO, `conf=0.25`) → ByteTrack (ultralytics) → TrackEval
(HOTA / MOTA / IDF1). Chỉ **motion model** thay đổi giữa 3 phương pháp; detector và
data association giữ nguyên tuyệt đối để so sánh công bằng.

**Ba phương pháp so sánh:**

| | Vector trạng thái | Ghi chú |
|---|---|---|
| KF + CV (baseline) | `[x, y, a, h, vx, vy, va, vh]` (8 chiều) | `KalmanFilterXYAH` gốc của ultralytics |
| EKF + CTRV | `[cx, cy, v, θ, ω, a, h, va, vh]` (9 chiều) | Tuyến tính hoá bậc 1 (Jacobian) |
| UKF + CTRV | như trên | Unscented Transform, 19 sigma point |

Khối `(a, h, va, vh)` giữ nguyên Constant Velocity y hệt baseline để **cô lập đúng một
biến nghiên cứu**. Mô hình đo `z = [cx, cy, a, h]` tuyến tính, giống hệt baseline → bước
`update` trùng khít KF chuẩn, **toàn bộ tính phi tuyến nằm trong bước `predict`**.

**Dữ liệu:** UA-DETRAC tập train — 60 video, 83.791 ảnh, 598.281 bounding box, 5.952 track.
Đã tự tính `occlusion_ratio` từ `region_overlap` (dataset không có sẵn trường này).

---

## 2. Kết quả cốt lõi

### 2.1 Mô phỏng: cơ chế CÓ THẬT và rất mạnh

Cho xe chạy theo quỹ đạo CTRV đã biết, quan sát 25 frame rồi cắt detection (mô phỏng bị
che hoàn toàn). Sai số ngoại suy vị trí (px):

| Đoạn che | Turn rate | KF+CV | EKF+CTRV | Chênh |
|---|---|---|---|---|
| 0.8s | 3°/f | 60.97 | **5.24** | CTRV tốt hơn 12× |
| 2.0s | 3°/f | 217.41 | **14.51** | CTRV tốt hơn **15×** |
| 2.0s | 0°/f (thẳng) | **5.18** | 15.71 | CV tốt hơn 3× |

→ Giả thuyết của khoá luận là **đúng về mặt cơ chế**.

### 2.2 Video thật, chỉ số tổng hợp (60 video, `track_buffer=30` mặc định)

| Motion model | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| KF + CV (baseline) | **0.6104** | **0.5735** | **0.6541** | **0.6615** | **0.7783** | **2.251** |
| EKF + CTRV | 0.6101 (−0.054%) | 0.5733 | 0.6536 | 0.6613 | 0.7782 | 2.290 |
| UKF + CTRV | 0.6095 (−0.151%) | 0.5732 | 0.6524 | 0.6611 | 0.7770 | 2.302 |

→ Ba model **gần như không phân biệt được**, và CTRV còn thấp hơn baseline một chút.
**Nghịch lý so với mô phỏng.**

### 2.3 Phân tích phân tầng — nơi hiệu ứng thật sự lộ ra

Xây phép đo bám sát cơ chế: với mỗi đoạn che khuất, xác định tracker có giữ được **cùng
một ID** trước và sau đoạn đó không (`preserved` / `switched` / `lost`). Ghép GT ↔ tracker
theo từng frame bằng **Hungarian trên ma trận IoU** (ngưỡng 0.5), vì file tracker dùng ID
riêng của ByteTrack.

**Chỉ so sánh trên tập đoạn chung của cả 3 model** (1.544/3.349 đoạn) để mẫu số giống hệt nhau.

Ô trọng tâm — che ≥0.90, độ dài 0.5–1.5s (n = 78):

| KF + CV | EKF + CTRV | UKF + CTRV |
|---|---|---|
| 6.41% | **11.54%** | 7.69% |

→ EKF giữ ID **gần gấp đôi** baseline, **đúng tầng mà mô phỏng dự đoán**.
Nhưng McNemar ghép cặp cho **p = 0.2188** — chưa đủ ý nghĩa thống kê.

---

## 3. Bốn nguyên nhân khiến CTRV không thắng (đã truy được, có số liệu)

### 3.1 Motion model chỉ có "tiếng nói" ở 2.7% trường hợp

Trên 1.544 đoạn, **cả 3 model cho kết cục giống hệt nhau ở 97.3%** (1.502 đoạn). Chỉ 42
đoạn có khác biệt — và trong đó KF thắng 24, EKF 21, UKF 19.

Phân bố kết cục gần như trùng khít: `preserved` 530/527/525, `switched` 306/308/310,
`lost` 708/709/709.

### 3.2 `track_buffer` = trần cứng (floor effect)

`bytetrack.yaml` đặt `track_buffer: 30` frame = **1.2 giây** ở 25 fps. Quá ngưỡng này
ByteTrack **xoá hẳn track**, không còn gì để ngoại suy.

| Độ dài đoạn (che ≥0.90) | n | Giữ được ID |
|---|---|---|
| ≤ 30 frame (trong buffer) | 257 | 52 (**20.2%**) |
| > 30 frame (vượt buffer) | 21 | 0 (**0.0%**) |

**100% đoạn `long` (>1.5s) đều dài hơn 30 frame** (min 38, median 48) → cả 3 model đều 0%.
Đây chính là lời giải cho nghịch lý mô phỏng: mô phỏng cho CTRV ngoại suy 50 frame và thắng
15×, nhưng trong ByteTrack thật track đã bị xoá từ frame 30.

### 3.3 `ω` ước lượng được KHÔNG đáng tin (nguyên nhân sâu nhất)

Đo trên 507 đoạn che khuất thật:

| | EKF | UKF |
|---|---|---|
| `ω` đúng dấu với khúc cua thật | **55.6%** | 54.4% |
| Độ lớn `\|ω\|` trung vị | 0.213 °/f | 0.212 °/f |
| `\|ω\|` GT thực tế | **0.071 °/f** | — |

→ Gần như **tung đồng xu** về hướng, và **phóng đại gấp 3 lần** về độ lớn.

CTRV giả định turn rate không đổi, nhưng trên UA-DETRAC xe thường đổi hướng **đúng lúc**
vào/ra vùng bị che (rẽ ở giao lộ, bị xe khác chắn ngay khi bắt đầu che). Ngoại suy một
đường cong với `ω` sai dấu gần một nửa số lần thì **tệ hơn là cứ đi thẳng**.

### 3.4 Gần một nửa số đoạn thất bại do detector, không phải motion model

**45.9% số đoạn là `lost`** — tracker không bao giờ tìm lại được xe sau khi hết che, với
**cả 3 model như nhau**. Phụ thuộc detector và ghép cặp IoU, nằm ngoài phạm vi motion model.

### 3.5 Riêng UKF: hiệu ứng dây cung (bất đẳng thức Jensen)

Với `α=1, κ=0, n=9` thì `λ=0` → **`Wm[0] = 0` chính xác**: kỳ vọng dự đoán **hoàn toàn bỏ
qua** `f(x̂)`, chỉ là trung bình 18 điểm sigma lệch tâm. Đo trực tiếp (quỹ đạo thật
`R = v/ω = 152.79 px`):

| Hiệp phương sai `P` | R dự đoán EKF | R dự đoán UKF |
|---|---|---|
| ×0.01 | **152.79** | 141.17 |
| ×1 | **152.79** | 136.75 |
| ×100 | **152.79** | 110.70 |

EKF đúng tuyệt đối mọi mức `P` (truyền tất định `x' = f(x̂)`); UKF **luôn hụt bán kính**,
hụt nặng hơn khi `P` phình to — tức đúng lúc bị che lâu, khi CTRV cần phát huy nhất.

**Đã kiểm chứng UKF KHÔNG có bug** ở cả 3 chỗ xử lý biến góc: (a) trung bình vòng cho `θ`
đúng, (b) residual `θ` có wrap về `[-π, π)` đúng — nếu quên bước này phương sai bị thổi
phồng 408.000 lần, (c) measurement `[cx, cy, a, h]` không chứa góc nên không cần wrap.

---

## 4. Hai hướng cải thiện đã thử

### 4.1 Nới `track_buffer` — CÓ tác dụng, nhưng chưa đủ

Chạy lại toàn bộ 60 video với `track_buffer = 90`:

| Motion model | HOTA | AssA | IDF1 | IDSW |
|---|---|---|---|---|
| KF + CV | 0.6101 | 0.6533 | 0.7771 | 2.360 |
| **EKF + CTRV** | **0.6107** | **0.6552** | **0.7794** | **2.307** |
| UKF + CTRV | 0.6096 | 0.6530 | 0.7777 | 2.353 |

**Lần đầu tiên trong toàn bộ đề tài, EKF vượt baseline** — và vượt trên đúng nhóm chỉ số đo
chất lượng **liên kết**: HOTA +0.10%, AssA +0.29%, IDF1 +0.30%, ít hơn 53 ID switch.

Ô trọng tâm (che ≥0.90 × medium, n=78) — **chỉ EKF hưởng lợi**:

| buffer | KF + CV | EKF + CTRV | UKF + CTRV |
|---|---|---|---|
| 30 | 6.41% | 11.54% | 7.69% |
| 90 | 6.41% | **12.82%** | 7.69% |

McNemar: Δ tăng **+4 → +5**, p giảm **0.2188 → 0.1250**. Xu hướng đơn điệu theo buffer
(30/60/90 → Δ 0/+2/+4, p 1.000/0.727/0.344).

**Nhưng trên toàn bộ 1.544 đoạn thì hoà:** KF 530→536, EKF 527→536, UKF 525→533.

### 4.2 Hiệu chỉnh nhiễu quá trình cho `ω` — KHÔNG tác dụng, đã đóng

Quét 20 tổ hợp `CTRV_STD_OMEGA × CTRV_INIT_STD_OMEGA`:

- Tỉ lệ đúng dấu **phẳng** trên toàn vùng quét (53.3–55.8%; hiện tại 55.6%, tốt nhất 55.8%)
- Độ lớn `|ω|` thì sửa được: từ 2.98× GT xuống **0.84× GT**

Kiểm chứng trên pipeline thật (6 video, GPU): `init=0.01` và `init=0.10` cho kết quả
**giống hệt nhau từng ô một**, cả EKF lẫn UKF.

→ Kết cục giữ ID phụ thuộc vào **dấu** của `ω`, không phải độ lớn. Mà dấu thì không đoán
trước được — **giới hạn cấu trúc**, không phải lỗi hiệu chỉnh tham số.

---

## 5. Một số điểm về tính chặt chẽ đã xử lý

- **Sửa lỗi thống kê nghiêm trọng:** code gọi `binomtest(...).pvalue * 2`, nhưng
  `binomtest` mặc định `alternative="two-sided"` nên giá trị trả về **đã là hai phía**.
  Mọi p-value báo cáo ban đầu **bị gấp đôi** (0.4375 → đúng là 0.2188). Đã sửa và tính lại.
- **Chỉ so sánh trên tập đoạn chung** của cả 3 model, để mẫu số giống hệt nhau — nếu không,
  mỗi model có mẫu số khác nhau (22 vs 24 đoạn) và tỉ lệ % không so sánh trực tiếp được.
- **Bootstrap giữ tính ghép cặp:** mỗi lần resample dùng **cùng bộ chỉ số** cho cả 2 model
  rồi mới lấy hiệu; resample độc lập sẽ phá tương quan và làm KTC rộng giả. Báo cáo cả
  cluster bootstrap **theo track** (nhiều đoạn cùng một xe không độc lập).
- **Cảnh báo so sánh bội:** kiểm định 15 cặp ở mức 5% → kỳ vọng 0.75 phát hiện giả. Quan
  sát đúng 1 (`partial × long`, KF > EKF, +1.30pp, KTC [0.21, 2.58]). Khi chuyển sang
  buffer=90 thì p tăng 0.0703 → 0.1796 — **xác nhận đó là dương tính giả**.
- **Độ cong đo theo TỪNG ĐOẠN CHE**, không phải cả track: `track_curvature.csv` gán nhãn cho
  cả quỹ đạo, nên xe rẽ gắt đầu video rồi đi thẳng suốt đoạn che vẫn bị gán `curved`. So
  sánh 3 cách đo, chọn "góc giữa hướng trung bình 1/3 đầu và 1/3 cuối" (tương quan 0.899 với
  phép khớp đường tròn hình học, trong khi cách tích luỹ `Σ|Δθ|` chỉ 0.572 vì bị nhiễu
  annotation lấn át).
- **Sửa một nhận định sai về FPS:** README cũ ghi KF 54.6 / EKF 80.8 / UKF 46.3 FPS — EKF
  "nhanh hơn" KF 48% là vô lý vì EKF phải tính thêm Jacobian. Đo lại 5 lần lặp: FPS
  end-to-end **không phân biệt được** 3 model (p = 0.68–0.92), hệ số biến thiên tới 26.8%.
  Chi phí thật của motion model: KF 1.14% / EKF 1.82% (1.60×) / UKF 8.07% (6.74×) runtime.

---

## 6. Kết luận trung thực hiện tại

**Chưa chứng minh được CTRV cải thiện việc giữ ID, nhưng cũng KHÔNG bác bỏ được.**

- Hướng chênh lệch **nhất quán ủng hộ EKF+CTRV** ở tầng che nặng, và càng rõ khi nới
  `track_buffer` (xu hướng đơn điệu qua 3 mức buffer)
- Nhưng **không tầng nào đạt p < 0.05** (tốt nhất p = 0.125)
- Nguyên nhân đã định vị chính xác: cỡ mẫu tự nhiên của UA-DETRAC ở đúng vùng CTRV phát huy
  quá nhỏ (chỉ 18 đoạn `che ≥0.90 × long` trên cả 60 video), cộng với `track_buffer` chặn trần

**Cách phát biểu đúng:** *"Dữ liệu hiện có không đủ để phân biệt 3 motion model"* —
**không phải** *"3 motion model như nhau"*. Đây là kết quả **underpowered**, khác hẳn với
kết quả âm tính.

**Cấu hình tốt nhất đo được:** EKF + CTRV với `track_buffer = 90`.
**UKF không có lý do để dùng:** thấp hơn EKF ở mọi chỉ số, chậm hơn 6.74×, và hiệu ứng dây
cung là hạn chế mang tính cấu trúc (đã thử giảm `α`, không cải thiện).

---

## 7. Khung luận văn đề xuất

**Tên:** *"Vì sao chỉ số tổng hợp không phát hiện được lợi thế của mô hình chuyển động phi
tuyến: phân tầng, hiệu ứng trần `track_buffer` và giới hạn dữ liệu trong theo dõi đa đối
tượng có che khuất"*

**Mạch:** (1) giả thuyết → (2) mô phỏng xác nhận cơ chế (15×) → (3) video thật có vẻ mâu
thuẫn → (4) **đóng góp phương pháp luận**: xây phép đo phân tầng đúng cơ chế → (5) tín hiệu
đúng hướng nhưng chưa đủ ý nghĩa → (6) **chẩn đoán nguyên nhân**: floor effect của
`track_buffer`, chứng minh bằng thực nghiệm → (7) **phát hiện phụ**: hiệu ứng dây cung của
UKF → (8) kết luận về phương pháp đánh giá.

**Điểm mạnh của khung này:** không mệnh đề nào bị thổi phồng; câu hỏi khó nhất của hội đồng
("vậy CTRV có tốt hơn không?") có câu trả lời sẵn, rõ ràng; và hai phát hiện (floor effect,
hiệu ứng dây cung) **có giá trị độc lập** với việc giả thuyết ban đầu đúng hay sai.

---

## 8. Công cụ đã xây (đều có trong repo, tiếng Việt không dấu, giải thích rõ công thức)

| File | Nội dung |
|---|---|
| `src/ekf_ctrv.py` | EKF + CTRV, 14 unit test (Jacobian giải tích vs sai phân số, giới hạn `ω→0`, khởi tạo 2 khung hình) |
| `src/ukf_ctrv.py` | UKF + CTRV, 12 unit test (trung bình vòng, sigma point, update trùng khít EKF) |
| `src/tracker_ctrv.py` | Ghép filter CTRV vào ByteTrack (chỉ 3 điểm chạm vào bố cục state) |
| `src/stratified_analysis.py` | Ghép GT↔tracker (Hungarian/IoU), phân tầng, McNemar, bootstrap |
| `src/diagnose_ukf.py` | Chẩn đoán UKF: kiểm tra xử lý góc, hiệu ứng dây cung, độ bền vững của `ω` |
| `src/benchmark_fps.py` | Đo FPS 2 tầng (micro + end-to-end có lặp) |
| `src/buffer_sweep_analysis.py` | Phân tích quét `track_buffer` |
| `src/tune_omega_noise.py` | Quét tham số nhiễu quá trình cho `ω` |

---

## 9. Câu hỏi muốn được hỗ trợ

1. Khung luận văn ở mục 7 có hợp lý không? Có nên điều chỉnh trọng tâm gì không?
2. Cách phát biểu kết luận "underpowered chứ không phải âm tính" đã đủ chặt để bảo vệ chưa?
3. Có phương pháp thống kê nào phù hợp hơn McNemar + bootstrap cho tình huống này không
   (ví dụ hồi quy logistic hỗn hợp với covariate liên tục thay vì chia ô, để giữ nhiều
   "power" hơn)?
4. Còn hướng cải thiện nào đáng thử mà chưa nghĩ tới không? (đã thử: nới `track_buffer` —
   có tác dụng; hiệu chỉnh nhiễu `ω` — không tác dụng)
5. Cách trình bày phần "hiệu ứng dây cung của UKF" sao cho hội đồng dễ hiểu nhất?
