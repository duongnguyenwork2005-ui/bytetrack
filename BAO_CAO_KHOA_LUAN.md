# Báo cáo tổng hợp khoá luận: So sánh KF / EKF / UKF cho Multi-Object Tracking trên UA-DETRAC

> Tài liệu tự chứa — đọc xong là nắm được toàn bộ tình hình, không cần bối cảnh khác.
> **Cập nhật lần 5:** bổ sung đánh giá lại test 40 video đã kiểm chứng ở mục 2.3 và
> thay kết luận hiện hành ở mục 6 bằng kết quả từ các CSV đã lưu. Các kết luận định lượng
> về test trong mục 2.2 và 4.6 là tư liệu lịch sử, không dùng thay cho mục 2.3.
> **Cập nhật lần 4:** bổ sung mục 4.9 (Giai đoạn F — triệt tiêu `ω`, phân tích lịch sử
> về vai trò của `ω` và lỗi `P` mất tính xác định dương) và các mục
> 16–19 phần tính chặt chẽ. Trước đó,
> **cập nhật lần 3:** bổ sung mục 2.3 (kết cục ghép cặp trên 1.544 đoạn),
> mục 4.7 (Giai đoạn E — khuyết điểm `ω` không bị chặn, phát hiện khi xem video
> minh hoạ) và mục 4.8 (quét 48 track dài).

---

## 1. Đề tài và thiết lập

**Câu hỏi nghiên cứu:** Mô hình chuyển động phi tuyến CTRV (Constant Turn Rate and
Velocity) có giúp giữ định danh (ID) của xe tốt hơn mô hình tuyến tính CV (Constant
Velocity) khi xe **vừa rẽ vừa bị che khuất** không?

**Pipeline:** YOLOv8n → ByteTrack → TrackEval (HOTA / MOTA / IDF1). Đánh giá test
cuối có hai điều kiện đã khóa trước: A dùng `conf=0.25`, B dùng `conf=0.10`. Trong mỗi
điều kiện, chỉ **motion model** thay đổi; detector, data association và các tham số
tracker còn lại giữ nguyên. Nguồn cấu hình: `results/test_eval/manifest.json`.

| | Vector trạng thái | Ghi chú |
|---|---|---|
| KF + CV (baseline) | `[x, y, a, h, vx, vy, va, vh]` (8 chiều) | `KalmanFilterXYAH` gốc của ultralytics |
| EKF + CTRV | `[cx, cy, v, θ, ω, a, h, va, vh]` (9 chiều) | Tuyến tính hoá bậc 1 (Jacobian) |
| UKF + CTRV | như trên | Unscented Transform, 19 sigma point |

Khối `(a, h, va, vh)` giữ nguyên CV y hệt baseline để **cô lập đúng một biến**. Mô hình
đo `z = [cx, cy, a, h]` tuyến tính → bước `update` trùng khít KF chuẩn, **toàn bộ tính
phi tuyến nằm trong `predict`**.

**Dữ liệu:** UA-DETRAC.
- Train: 60 video, 83.791 ảnh, 598.281 bbox, 5.952 track
- Test: 40 video, 675.774 bbox, 2.337 track — **có annotation đầy đủ** (xem mục 5.6)

---

## 2. Kết quả cốt lõi

### 2.1 Mô phỏng: cơ chế CÓ THẬT và rất mạnh

Sai số ngoại suy vị trí (px) khi cắt detection:

| Đoạn che | Turn rate | KF+CV | EKF+CTRV | Chênh |
|---|---|---|---|---|
| 0.8s | 3°/f | 60.97 | **5.24** | CTRV tốt hơn 12× |
| 2.0s | 3°/f | 217.41 | **14.51** | CTRV tốt hơn **15×** |
| 2.0s | 0°/f (thẳng) | **5.18** | 15.71 | CV tốt hơn 3× |

### 2.2 Video thật — bảng tiến hoá qua các giai đoạn

| Cấu hình | KF + CV | EKF + CTRV | UKF + CTRV |
|---|---|---|---|
| 60 video, detector COCO, `buffer=30` | **0.6104** | 0.6101 | 0.6095 |
| 60 video, detector COCO, `buffer=90` | 0.6101 | **0.6107** | 0.6096 |
| 60 video, **detector fine-tune** | 0.6783 | 0.6780 | **0.6791** |
| **40 video TEST, detector fine-tune (lịch sử; thay bằng mục 2.3)** | 0.6325 | **0.6344** | 0.6334 |

*(HOTA. Thứ tự đổi ở mỗi cấu hình — dấu hiệu chênh lệch nằm trong vùng nhiễu.)*

### 2.3 Đánh giá test 40 video đã kiểm chứng — kết quả hiện hành

Mục này thay thế các nhận định định lượng về test trước đây trong mục 2.2 và 4.6.
Kết quả nằm trong commit đánh giá `b5bd086f8e4a625850ab35c977d1ed40776e810d`; tập test
đã từng được sử dụng trong nghiên cứu trước, nên không mô tả là tập held-out hoàn toàn.

| Lượt / model | HOTA | IDF1 | AssA | IDSW | FP | FN | `id_match` | `id_continuous` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A, CV (`conf=0.25`) | 58.484% | 70.225% | 64.174% | 2.500 | 72.646 | 198.712 | 48.187% | 29.683% |
| A, EKF | 58.600% | 70.469% | 64.427% | 2.532 | 72.314 | 198.835 | 48.943% | 29.683% |
| A, UKF | 58.490% | 70.292% | 64.206% | 2.549 | 72.421 | 198.844 | 48.716% | 29.758% |
| B, CV (`conf=0.10`) | 58.217% | 69.407% | 63.581% | 2.082 | 99.903 | 180.571 | 48.867% | 36.631% |
| B, EKF | 58.548% | 69.937% | 64.298% | 2.063 | 99.579 | 180.532 | 49.245% | 37.009% |
| B, UKF | 58.399% | 69.716% | 63.971% | 2.110 | 99.731 | 180.446 | 49.773% | 37.009% |

Nguồn số liệu: `results/test_eval/summary/configuration_metrics.csv`. Các tỷ lệ trong
bảng được đổi từ tỷ lệ CSV sang phần trăm và làm tròn ba chữ số thập phân; IDSW, FP và
FN là số đếm nguyên trong CSV.

`id_match` là tỷ lệ ID ngay trước và sau sự kiện trùng nhau. `id_continuous` là tỷ lệ
ID được duy trì liên tục theo định nghĩa của script; hai chỉ số không đồng nghĩa. Từ GT,
2.174 sự kiện sau gộp được lập trước khi so sánh tracker, trong đó 1.324 sự kiện đủ điều
kiện và 850 bị loại (559 thiếu GT trước, 291 thiếu GT sau). Cả sáu tracker dùng cùng
40 video và GT; 39 video có sự kiện đủ điều kiện cho phép so sánh che khuất. Có 484 đoạn
full nằm trong partial và đã được gộp, không đếm lặp. Nguồn:
`results/test_eval/summary/configuration_metrics.csv`,
`results/test_eval/summary/validation.csv` và `results/test_eval/manifest.json`.

Các KTC bootstrap 95% ghép cặp theo 39 video có sự kiện đủ điều kiện (5.000 lần, seed 0)
của `id_match` cho
EKF−CV / UKF−CV lần lượt là A: +0,755 [−0,584; +2,035] / +0,529 [−0,519; +1,309]
điểm phần trăm; B: +0,378 [−0,498; +1,316] / +0,906 [−0,548; +2,246]. Tất cả chứa 0,
vì vậy chưa đủ bằng chứng về khác biệt về `id_match`; điều này không xác lập các mô hình
tương đương và không hỗ trợ tuyên bố EKF hoặc UKF xử lý che khuất tốt hơn CV. Nguồn:
`results/test_eval/summary/paired_contrasts_bootstrap.csv`.

Ở lượt B, EKF−CV có KTC danh nghĩa dương cho HOTA (+0,331 [+0,112; +0,569]), IDF1
(+0,531 [+0,189; +0,891]) và AssA (+0,716 [+0,244; +1,238]) điểm phần trăm. Các KTC
này chưa hiệu chỉnh nhiều so sánh, nên chỉ là tín hiệu chỉ số riêng lẻ và không chứng minh
EKF thắng toàn diện. Nguồn: `results/test_eval/summary/paired_contrasts_bootstrap.csv`.

So với A, B giảm IDSW (CV: 2.500→2.082; EKF: 2.532→2.063; UKF: 2.549→2.110) và FN,
đồng thời tăng FP (lần lượt +27.257, +27.265, +27.310) và `id_continuous` (+6,949,
+7,326, +7,251 điểm phần trăm). HOTA/IDF1/AssA B−A đều giảm theo điểm ước lượng trên
test; hướng này ngược với xu hướng tăng theo điểm ước lượng trên train. Các tổng ID,
FP, FN và association không xác định nguyên nhân cơ chế nếu không có log trực tiếp đầy
đủ. Nguồn test: `results/test_eval/summary/configuration_metrics.csv` và
`results/test_eval/summary/paired_contrasts_bootstrap.csv`; số train đối chiếu:
`TEST_EVALUATION_REPORT.md`, mục 6.

Thời gian của sáu lần chạy chỉ mang tính mô tả vì tải máy không được kiểm soát; không
dùng chúng làm benchmark. Nguồn: `results/test_eval/summary/metadata.json` và
`TEST_EVALUATION_REPORT.md`, mục 7.

### 2.4 Kết cục ghép cặp trên 1.544 đoạn che — phân tích lịch sử

Các số trong mục này thuộc phân tích train lịch sử, không là kết quả test cuối và không
được dùng để suy ra quan hệ nhân quả cho các tổng chỉ số ở mục 2.3.

Thay vì so tỉ lệ trung bình, ghép cặp **từng đoạn che một** giữa KF+CV và EKF+CTRV
(60 video train, detector COCO, `buffer=30`):

| Kết cục | Số đoạn | Tỉ lệ |
|---|---|---|
| Cả hai **mất** ID | 998 | **64,6 %** |
| Cả hai **giữ** được ID | 511 | 33,1 % |
| Chỉ KF + CV giữ được | 19 | 1,2 % |
| Chỉ EKF + CTRV giữ được | 16 | 1,0 % |

Hai model chỉ khác nhau ở **2,3 %** số đoạn, và trong đó KF thắng 19 lần, EKF thắng 16
lần — không nghiêng về bên nào.

> **Con số thuyết phục nhất nằm trong nhóm "cả hai cùng hỏng":**
>
> | | EKF: `lost` | EKF: `switched` |
> |---|---|---|
> | **KF: `lost`** | **708** | 0 |
> | **KF: `switched`** | 0 | **290** |
>
> Đường chéo tuyệt đối. Hai model không chỉ cùng thất bại — chúng thất bại **y hệt cách
> nhau, 998/998 lần, không một ngoại lệ**. Đây là bằng chứng trực tiếp rằng ở chế độ
> thất bại, motion model **hoàn toàn không có tiếng nói**; thứ quyết định là detector và
> `track_buffer`.

**Yếu tố thật sự quyết định là thời gian bị che, không phải motion model:**

| T_occ (frame) | n | Cả hai mất ID |
|---|---|---|
| 1–10 | 427 | 58,8 % |
| 11–20 | 274 | 54,0 % |
| 21–30 | 251 | 54,6 % |
| 31–45 | 239 | 69,0 % |
| 46–60 | 149 | 83,2 % |
| 61–90 | 132 | 87,1 % |

Vượt `track_buffer` = 30 frame, tỉ lệ hỏng nhảy từ **56,3 %** lên **78,0 %**.

**Loại vật cản cũng có ảnh hưởng** (dùng `occ_ratio_by_background` vs
`occ_ratio_by_vehicle` của annotation):

| Nguồn che | n | Cả hai mất ID | 2 model khác nhau |
|---|---|---|---|
| Vật cản **nền** (cột, cây) | 678 | **69,6 %** | 1,8 % |
| **Xe khác** | 866 | 60,7 % | 2,7 % |

Không phải do xe nhỏ hay đông xe: Pearson giữa tỉ lệ hỏng và diện tích xe là **+0,060**,
với mật độ xe là **−0,035** — cả hai ≈ 0.

---

## 3. Các giả thuyết và quan sát lịch sử về CTRV

Các phân tích trong mục 2.4–4.9 là chẩn đoán lịch sử trên các tập và thiết lập trước
đánh giá test cuối. Chúng không thay thế kết luận định lượng ở mục 2.3 và không thiết
lập quan hệ nhân quả từ các chỉ số tổng hợp.

### 3.1 Motion model chỉ có "tiếng nói" ở ~3–4% trường hợp
3 model cho kết cục **giống hệt nhau** ở 96–97% số đoạn, ở mọi cấu hình đã thử.
Chi tiết theo cặp và theo kiểu thất bại: xem mục **2.4**.

### 3.2 `track_buffer` là giới hạn cấu hình
`track_buffer=30` frame = 1.2s. Quá ngưỡng, ByteTrack **xoá hẳn track**.

| Độ dài đoạn (che ≥0.90) | n | Giữ được ID |
|---|---|---|
| ≤ 30 frame | 257 | 52 (**20.2%**) |
| > 30 frame | 21 | 0 (**0.0%**) |

### 3.3 `ω` ước lượng được không ổn định trong phân tích lịch sử
Trên 507 đoạn: `ω` đúng dấu chỉ **55.6%** (EKF), độ lớn **gấp 3 lần** GT.
Giai đoạn A ghi nhận giới hạn này trong thiết lập khi đó — xem mục 4.3.
Giai đoạn E bổ sung: `ω` GT trước che so với trong che **ngược dấu** (Spearman −0,448,
p = 0,0023) — xem mục 4.7.

### 3.4 Gần một nửa số đoạn thất bại do detector
`lost` = 45.9% với detector COCO. **Giai đoạn C giảm còn 35.1%** — xem mục 4.5.

### 3.5 Riêng UKF: hiệu ứng dây cung
`α=1, κ=0, n=9` → `λ=0` → **`Wm[0]=0`**: kỳ vọng dự đoán bỏ qua `f(x̂)`, chỉ là trung
bình 18 điểm sigma lệch tâm. Đo được: quỹ đạo thật `R=152.79 px`, EKF dự đoán **chính
xác 152.79** ở mọi mức `P`; UKF hụt còn 141 → 137 → **111** khi `P` phình.
**Đã kiểm chứng UKF không có bug** ở cả 3 chỗ xử lý góc.

---

## 4. Bảy hướng cải thiện đã thử, và một phép quét đối chứng

### 4.1 Nới `track_buffer` (30 → 90) — CÓ tác dụng, chưa đủ
EKF lần đầu vượt baseline (AssA +0.29%). Ở ô trọng tâm, **chỉ EKF hưởng lợi**
(11.54% → 12.82%); KF và UKF đứng yên. McNemar p: 0.2188 → 0.1250. Xu hướng **đơn điệu**
theo buffer (Δ 0 → +2 → +4).

### 4.2 Hiệu chỉnh nhiễu quá trình cho `ω` — KHÔNG tác dụng
Quét 20 tổ hợp: tỉ lệ đúng dấu **phẳng** (53.3–55.8%). Độ lớn `|ω|` sửa được
(2.98× → 0.84× GT) nhưng **không đổi kết cục** — kiểm chứng trên pipeline thật cho kết
quả **giống hệt từng ô một**.

### 4.3 GIAI ĐOẠN A — Ước lượng `ω` hồi cố (RTS + circle fit) → **ĐÓNG**

| Phương pháp | % đúng dấu | \|ω\| so với GT |
|---|---|---|
| ω state (hiện tại) | **55.6%** | 2.98× |
| RTS smoother | 53.1% | 3.51× |
| Circle fit | 51.3% | 8.94× |

Cả hai cách hồi cố đều **tệ hơn**. Đã tự kiểm chứng code trước khi kết luận: circle fit
khớp cung tròn hoàn hảo sai số 0.01% (cài đúng), nhưng đường thẳng + nhiễu 1px cho
`ω` gấp **162 lần** GT — điểm yếu bản chất của phương pháp, không phải bug.

> **Quan sát lịch sử cần kiểm tra lại trên dữ liệu độc lập.** Đo bằng **GT hoàn hảo ở cả
> hai phía** (không filter, không nhiễu, không sai số ước lượng), trên 361 đoạn:
> **ω GT trước che → ω GT trong che: đúng dấu 58,2%, Pearson +0,0093**. Bộ lọc hiện tại
> đạt 59,3% trên cùng tập. Quan sát này mô tả thiết lập lịch sử, không xác lập giới hạn
> phổ quát hay cơ chế cho kết quả test ở mục 2.3.

### 4.4 GIAI ĐOẠN B — Nới cửa liên kết cho track LOST → **ĐÓNG**

**Chẩn đoán xác nhận vách đứng của IoU:** ~70–74% lần liên kết lại trượt ở cổng IoU,
một nửa ở **đúng IoU = 0**. Nhưng CTRV **không** dự đoán tốt hơn CV tại thời điểm đó
(EKF có *nhiều* IoU=0 hơn KF: 54.3% vs 48.7%).

Đã phát hiện **2 cạm bẫy toán học** làm cả hai cách ngây thơ hỏng hoàn toàn:
- Thay thẳng DIoU: `DIoU ≤ 0` khi không giao → `cost ≥ 1 > 0.8` → **không cứu được ca nào**
- Giãn box rồi tính IoU: giãn 3× thì IoU tối đa chỉ 0.111 < 0.2 → **tự chết**

Sau khi sửa (chuẩn hoá DIoU + đổi mẫu số sang IoA), rồi sửa tiếp (gate lọc + DIoU xếp
hạng), kết quả cuối:

| Mức đo | Kết quả |
|---|---|
| 250 đoạn che khuất | Lợi ròng **+6/+7/+3** — *có lợi* |
| Toàn video (TrackEval) | IDSW **+234…+352**, HOTA **giảm** 0.45–0.72% — *có hại* |

> **Mẫu hình "lợi cục bộ, hại toàn cục".** Tỉ lệ ~1 đoạn che cứu được : ~50 ID switch
> phát sinh. Trên dữ liệu đông xe, nới cửa liên kết là **lợi bất cập hại**.
> **Bài học phương pháp:** chỉ số mức đoạn là *cần* nhưng *không đủ*. Tôi đã kết luận
> sai một lần vì chỉ nhìn mức đoạn; chỉ IDSW mới lộ ra tác hại.

### 4.5 GIAI ĐOẠN C — Fine-tune detector → **CẢI THIỆN LỚN NHẤT**

3-fold cross-validation (mỗi video suy diễn bằng detector **chưa từng thấy nó**):

| | mAP50 | R@0.25 |
|---|---|---|
| Trước (COCO) | 0.8523 | 0.7978 |
| **Sau (fine-tune)** | **0.9311** | **0.9042** |

Precision gần như không đổi → recall tăng **không đánh đổi bằng false positive**.

| | `lost` | `preserved` | HOTA (KF) | IDSW |
|---|---|---|---|---|
| COCO | 45.9% | 34.2% | 0.6104 | 2.251 |
| **Fine-tune** | **35.1%** | **55.0%** | **0.6783** | **1.359** |

**HOTA +6.8 điểm — lớn hơn toàn bộ can thiệp motion model cộng lại.**

> **NHƯNG: ưu thế của EKF BIẾN MẤT.** Ô `che ≥0.90 × medium`:
> COCO cho KF 6.41% / **EKF 11.54%** / UKF 7.69% (gần gấp đôi);
> fine-tune cho **64.91% / 65.79% / 67.54%** (bằng nhau).
> Gợi ý ưu thế đó **không phải lợi ích thật của CTRV** mà là hiện tượng của chế độ
> detector yếu — khi rất ít detection thì kết cục gần như ngẫu nhiên.
>
> Số đoạn 3 model khác nhau chỉ tăng **42 → 53** (2.7% → 3.1%) — mục tiêu "mở rộng mẫu
> số" **đạt rất hạn chế**.

### 4.6 GIAI ĐOẠN D — Mở rộng sang tập TEST 40 video (lịch sử; thay bằng mục 2.3)

Các số và kiểm định trong mục này là bản đánh giá test lịch sử. Kết quả hiện hành dùng
chung manifest, CSV và bootstrap ghép cặp theo video nằm ở mục 2.3; không dùng mục này
để kết luận về test cuối.

**Cỡ mẫu ở ô then chốt: 17 → 115 (6.8×)** — giải quyết đúng điểm nghẽn.

Tập test **thuận lợi hơn hẳn**: đoạn che ≥0.90 dài trung vị **28 frame** (train chỉ 7),
xe cong 41.5% (train 37.6%), bbox bị che 32.3% (train 23.7%).

**Chỉ số tổng hợp tập test — EKF dẫn đầu cả 3 chỉ số liên kết:**

| Model | HOTA | AssA | IDF1 |
|---|---|---|---|
| KF + CV | 0.6325 | 0.6763 | 0.7502 |
| **EKF + CTRV** | **0.6344** | **0.6802** | **0.7542** |
| UKF + CTRV | 0.6334 | 0.6781 | 0.7519 |

**McNemar trên toàn bộ tập (phép so sánh cơ bản nhất):**

| Tập | Cặp | KF thắng | CTRV thắng | Δ | **p** |
|---|---|---|---|---|---|
| TRAIN | KF vs EKF | 23 | 21 | −2 | 0.8804 |
| TRAIN | KF vs UKF | 20 | 26 | +6 | 0.4614 |
| TEST | KF vs EKF | 19 | 31 | +12 | 0.1189 |
| **TEST** | **KF vs UKF** | 21 | 36 | **+15** | **0.0627** |

Trong 73 đoạn 3 model khác nhau trên test: **KF 28 | EKF 40 | UKF 43**.

> **Nhưng hai tập độc lập KHÔNG đồng thuận.** Tập train không tái lập (p = 0.88 và 0.46).
> Và **0/12 ô đã đăng ký trước** đạt p<0.05 — ưu thế trên test là **khuếch tán** trên
> nhiều tầng chứ không tập trung ở tầng đã dự báo. Nếu cơ chế đúng như giả thuyết thì
> nó **phải tập trung**.

### 4.7 GIAI ĐOẠN E — Khuyết điểm `ω` không bị chặn → **có thật, nhưng không đảo ngược kết quả**

**Cách phát hiện:** khi xem video minh hoạ side-by-side, box dự đoán của EKF đôi lúc
**quay tròn tại chỗ** thay vì đi tiếp. Đây là khuyết điểm được tìm ra bằng *quan sát trực
quan*, không phải bằng chỉ số — một lập luận đáng nêu trong luận văn.

**Phân định bug hay hành vi đúng.** CTRV ngoại suy trên đường tròn bán kính `R = v/|ω|`,
sau `T` frame quét một cung `arc = |ω|·T`. Quay tròn **tự nó là hành vi đúng** khi `ω`
lớn. Nên câu hỏi đúng là: `ω` đó có hợp lý không?

- **Phần toán không sai:** 26/26 unit test PASS, gồm so Jacobian giải tích với sai phân
  số (cả nhánh cong và nhánh `ω→0`), kiểm tra liên tục tại `ω→0`, và kiểm tra đi hết một
  vòng tròn về đúng chỗ cũ (sai lệch 3,6·10⁻¹³ px).
- **Nhưng không có gì chặn `ω`.** Ngưỡng vật lý: xe rẽ ngã tư quét ~90° trong 2–4 giây =
  18–45 °/giây = 0,013–0,031 rad/frame. Lấy rộng rãi `|ω| > 0,05 rad/frame`
  (= 2,9 °/frame = 72 °/giây) là bất khả thi với xe hơi.

Đo trên 50 đoạn che của 48 track dài, tại frame cuối trước khi mất quan sát:

| Chỉ tiêu | Bản gốc | Chặn ω |
|---|---|---|
| Trung vị \|ω\| | 0,257 °/frame *(GT: 0,441)* | 0,258 °/frame |
| Phân vị 90 | **193,5 °/frame** | 2,45 °/frame |
| Lớn nhất | **1077,1 °/frame** | 2,87 °/frame |
| Vượt ngưỡng bất khả thi | **20 %** | **0 %** |
| Bán kính `R` < 100 px *(nhỏ hơn chiếc xe)* | 22 % | 8 % |
| Quét **> 360°** *(trọn một vòng)* | **16 %** | **0 %** |
| Quét 180–360° | 4 % | 4 % |
| Quét 45–180° | 4 % | 14 % |

Ngưỡng 0,05 rad/frame cần **126 frame** mới quét trọn vòng, mà đoạn che dài nhất trong
dữ liệu chỉ 73 frame — đó là lý do cột chặn không còn đoạn nào vượt 360°.

> **Lỗi thứ hai tự phát hiện trong lúc đo Giai đoạn E — đã sửa, số liệu ở đây là SAU khi
> sửa.** Khi xem lại video minh hoạ, một đoạn cho sai số 1208 px trông bất thường so với
> 2 đoạn còn lại của *cùng* track. Điều tra theo quy trình loại trừ (ghép cặp GT↔tracker
> → khoảng trống frame trong GT → dữ liệu GT bất thường) xác định: 4 script chạy bộ lọc
> **xuyên qua nhiều đoạn che liên tiếp** (`long_compare.py`, `diagnose_ekf_circle.py`,
> `omega_clamp_experiment.py`, script quét 48 track, và script chẩn đoán
> `diagnose_seg2.py`) đều gọi `initiate_from_motion(..., n_frames=1.0)` — hằng số cứng —
> trong khi khoảng cách thật giữa hai quan sát có thể là hàng chục frame khi track vừa
> sinh đã bị che ngay. Đo được trên MVI_40992 track 12: khoảng cách thật 27 frame, vận
> tốc bị gán **294 px/frame** thay vì đúng **10,9 px/frame** — thổi phồng 27 lần.
> **Pipeline tracking thật (`tracker_ctrv.py`) không dính lỗi này** — nó đã tính đúng
> `n_frames = max(1, frame_id - self.frame_id)` từ đầu; các script phân tích khác dùng
> cửa sổ liền mạch trước đoạn che cũng không dính. Nên **mọi số HOTA/IDSW và bảng 1.544
> đoạn không đổi**, nhưng số định lượng của Giai đoạn E (bảng trên, các bảng dưới, và
> mục 4.8) đều đã tính lại sau khi sửa. Trên đúng đoạn phát hiện ra lỗi, FDE giảm từ
> 556,5 xuống 267,1 px (bản gốc) và từ 1208,4 xuống 349,8 px (bản chặn ω). Kết luận định
> tính không đổi: `ω` vẫn sai ở đuôi phân bố, EKF vẫn thua KF trên phần lớn số cặp.

**Chặn `ω` rồi chạy lại đầy đủ 60 video** (tracker riêng `yolov8n-ekf-ctrv-clamped`,
mọi tham số ghép cặp giữ y hệt — chỉ khác đúng một biến). Lần chạy 60 video này không đi
qua 4 script có lỗi n_frames nên số dưới đây không cần tính lại:

| Model | HOTA | AssA | IDF1 | IDSW |
|---|---|---|---|---|
| KF + CV (baseline) | **0,6104** | **0,6541** | 0,7783 | **2251** |
| EKF + CTRV nguyên bản | 0,6101 | 0,6536 | 0,7782 | 2290 |
| **EKF + CTRV chặn ω** | 0,6102 | 0,6537 | **0,7783** | 2280 |
| UKF + CTRV | 0,6095 | 0,6524 | 0,7770 | 2302 |

Mức từng đoạn (1.544 đoạn): tỉ lệ giữ ID KF **34,33 %** | EKF gốc 34,13 % | EKF chặn
**34,20 %** | UKF 34,00 %. Chặn `ω` chỉ đổi kết cục **3/1544 đoạn (0,19 %)**.
McNemar KF vs EKF chặn ω: 19 so 17, **p = 0,8679**. Không phân tầng nào đạt p < 0,05.

Trên 48 track dài (83 đoạn, sau khi sửa lỗi n_frames):

| Cấu hình | FDE trung bình | FDE trung vị |
|---|---|---|
| KF + CV | 202,3 px | 106,6 px |
| EKF + CTRV nguyên bản | 226,8 px | 115,2 px |
| EKF + CTRV chặn ω | 226,8 px | **105,1 px** |

So từng cặp: chặn `ω` **thắng KF** 10 đoạn, **thua KF** 34 đoạn, hoà 39. So với bản
nguyên bản: chặn `ω` tốt hơn 7 đoạn, tệ hơn 8 đoạn, không đổi 68.

> **Phát hiện ngược chiều, vẫn giữ lại sau khi sửa lỗi n_frames:** chặn `ω` có thể làm
> **sai số TĂNG**, dù biên độ nhỏ hơn số đã báo cáo trước đây nhiều. MVI_40992 track 12
> đoạn 2: FDE **267,1 → 349,8 px** (trước khi sửa lỗi n_frames, cùng hiện tượng này đo
> được 556,5 → 1208,4 px — phóng đại do lỗi, nhưng hướng thì đúng cả hai lần đo). Cơ chế
> không đổi: khi `ω` lớn, box quay tít trong vòng tròn bán kính nhỏ nên **vô tình nằm
> gần chỗ cũ**; chặn `ω` lại cho nó bay thẳng ra xa hơn.

**Kết luận lịch sử Giai đoạn E:** chặn `ω` sửa được **triệu chứng nhìn thấy** trong các
ca minh hoạ. Đo trên 44 đoạn của các track dài, `ω` GT *trước* che so với `ω` GT
*trong* che có **Spearman −0,448
(p = 0,0023)**, chỉ **25 %** số lần cùng dấu — không những không dự báo được mà còn có
xu hướng **ngược dấu**. Đây là quan sát mô tả, không suy ra nguyên nhân cho các chỉ số
test. *(Pearson ra −0,755 nhưng bị một điểm ngoại lai kéo; bỏ điểm đó còn −0,385, nên
báo cáo Spearman. Hai con số này không bị ảnh hưởng bởi lỗi n_frames vì chỉ dùng `ω` đo
trực tiếp từ GT, không qua bộ lọc.)*

> **Vì sao khuyết điểm nghiêm trọng lúc đo FDE nhưng vô hại lúc tracking thật:**
> `track_buffer` = 30 frame **xoá track trước khi** CTRV kịp ngoại suy đủ lâu để `ω` phi
> lý gây hại. Lỗi chỉ lộ ra khi ép bộ lọc ngoại suy mù suốt 40–70 frame. Đây là diễn
> giải cho ca minh hoạ, không phải suy luận nhân quả từ tổng số chỉ số test.

Render lại **toàn bộ 10 video minh hoạ** bằng bản chặn `ω` (và render lại lần nữa sau khi
sửa lỗi n_frames): trên 240 đoạn của 8 video demo gốc (không đi qua 4 script có lỗi
n_frames) chỉ **5 đoạn (2,1 %)** đổi FDE khi chặn `ω`, FDE trung bình 165,1 → 165,0 px.
Đáng chú ý, video `2_cv_thang` (EKF sai 845 px) có `|ω| = 0,0317 rad/frame` — **vốn đã
dưới ngưỡng chặn**, nên ca đó không được giải thích bởi riêng ngưỡng chặn trong phân tích
lịch sử này.
Trong 5 video `long_*` minh hoạ theo track, chỉ **MVI_40992 t12** dính lỗi n_frames (vì
track này bị che ngay từ lúc sinh); 4 track còn lại không đổi một pixel nào trước/sau
khi sửa.

### 4.8 Quét 48 track dài — CTRV thua trên hầu hết

Để tránh cherry-picking khi chọn video minh hoạ, quét **mọi** track ≥150 frame, ≥2 đoạn
che, có di chuyển thật (48 track), so FDE cuối mỗi đoạn che *(số liệu sau khi sửa lỗi
n_frames ở mục 4.7 — script quét này dùng lại cùng hàm chạy bộ lọc nên cũng dính lỗi)*:

| | Số track | Tỉ lệ |
|---|---|---|
| CTRV tốt hơn | 7 | 15 % |
| **CV tốt hơn** | **30** | **62 %** |
| Hoà | 11 | 23 % |

FDE trung bình: KF **266,0 px** vs EKF **292,1 px**. Tương quan giữa **góc cua của xe**
và lợi thế của CTRV: **+0,080** — về bản chất vẫn là không (trước khi sửa lỗi đo được
−0,021; cả hai đều xấp xỉ 0, không đáng kể với n = 48, việc đổi dấu chỉ phản ánh nhiễu
thống kê). Ba track cua gắt nhất (150,7° / 148,4° / 140,8°) thì CTRV thua cả ba.

> **Đây là phản chứng trực tiếp cho giả thuyết ban đầu.** Nếu CTRV có lợi thế do mô hình
> hoá khúc cua, lợi thế đó phải **tăng theo góc cua**. Nó không tăng.

### 4.9 GIAI ĐOẠN F — Triệt tiêu `ω` hoàn toàn: phân tích lịch sử

Giai đoạn E cho thấy **kẹp** `ω` không phải fix sạch (đoạn 2 của MVI_40992 t12 tệ đi:
267 → 350 px). Nghi vấn: kẹp tạo bước nhảy rời rạc trong state, tương tác xấu với `v/θ`.
Giai đoạn F thử cách khác — đặt `ω = 0` **hoàn toàn** trong suốt quá trình ngoại suy mù,
khiến CTRV suy biến về chuyển động thẳng.

**Phép sanity check bắt buộc.** Về toán học, `ω = 0` làm phương trình CTRV suy biến:
`cx' = cx + v·cos(θ)·dt`, `cy' = cy + v·sin(θ)·dt` — tức đúng dạng CV. Vậy bản triệt `ω`
**phải** cho sai số gần bằng KF + CV. Kết quả trên MVI_40992 track 12:

| Đoạn | Che | KF + CV | Triệt `ω` | Chênh |
|---|---|---|---|---|
| 1 | 492–517 | 283,1 px | 283,1 px | 0,0 % |
| **2** | **524–563** | **161,4 px** | **35,0 px** | **78 %** |
| 3 | 575–609 | 63,7 px | 68,1 px | 6,9 % |

**Sanity check THẤT BẠI** — nhưng theo hướng bất ngờ: triệt `ω` cho sai số *nhỏ hơn* KF
tới 4,6 lần chứ không phải lớn hơn. Nếu `ω` là nguyên nhân duy nhất thì con số phải
xấp xỉ 161 px.

#### Phân tích 1 — ước lượng vận tốc trong ca minh hoạ

Tại frame neo 523: KF + CV có `|v| = 11,045 px/f` (khớp GT 11,05); CTRV chỉ có
**6,300 px/f** — thấp hơn **43 %**. Phép **đối chứng tách biến** ép `ω = 0` *và*
`v` = vận tốc của KF:

| Đoạn | KF + CV | `ω`=0, `v` ước lượng | `ω`=0, `v` đúng | Lệch với KF |
|---|---|---|---|---|
| 1 | 283,1 px | 283,1 px | 283,1 px | 0,0 px |
| 2 | 161,4 px | 35,0 px | **160,0 px** | **1,4 px** |
| 3 | 63,7 px | 63,4 px | **65,7 px** | **2,0 px** |

**CTRV suy biến đúng về CV khi `ω` = 0 và vận tốc khớp** — lệch tối đa 2,0 px trên cả ba
đoạn. Phần toán của mô hình chuyển động **không sai**; vấn đề nằm ở **ước lượng trạng thái**.

> **Con số 35,0 px là ăn may, không phải ưu thế mô hình.** Xe **giảm tốc** còn
> **6,91 px/f** trong lúc bị che (từ 11,02 px/f trước đó). Ước lượng thiếu của CTRV
> (6,300 px/f) trùng hợp khớp với tốc độ thật trong lúc che, còn KF chốt đúng vận tốc
> tại frame neo (11,045) nên vượt quá. Không mô hình nào trong hai cái "biết" xe sẽ chậm
> lại — cả CV lẫn CTRV đều giả định **tốc độ không đổi**. Đây là điểm mù chung, và là
> lý do sâu hơn cả `ω`: **thiếu biến gia tốc trong vector trạng thái**.

Vì sao `v` tụt còn 6,3? Truy vết: `initiate_from_motion` gán đúng 10,89 px/f tại frame
518, nhưng `v` tụt xuống 2,92 ở frame 520 rồi mới bò lại 6,30 ở frame 523. Nguyên nhân:
vị trí dự đoán tại frame 518 lệch xa quan sát (bộ lọc vừa ngoại suy mù 26 frame), bước
`update` kéo mạnh vào **vị trí** và qua Jacobian kéo luôn `v/θ` đi theo. Chỉ có **6 frame
nhìn thấy** giữa hai đoạn che — không đủ để hội tụ.

#### Phân tích 2 — ma trận `P` mất tính xác định dương *(ghi nhận lịch sử)*

| Frame | Trị riêng nhỏ nhất của `P` | `Var(ω)` |
|---|---|---|
| 517 | +2,7·10⁻⁹ | +1,065·10⁻² |
| **518** | **−9,7·10⁻²** | **−9,667·10⁻³** |
| 530 | **−703** | −3,848·10⁻² |

Phương sai âm là vô nghĩa vật lý. Chuyển sang âm **đúng tại frame gọi
`initiate_from_motion`**, rồi phân kỳ trong lúc ngoại suy mù.

**Nguyên nhân:** hàm đó ghi đè hai phần tử **đường chéo** `P[V,V]` và `P[THETA,THETA]`
bằng hằng số nhưng **giữ nguyên các phần tử ngoài đường chéo** (đo được `P[V,CX] = 498,76`).
Thu nhỏ phương sai riêng mà không thu nhỏ tương quan tương ứng phá vỡ tính nửa xác định
dương của ma trận hiệp phương sai.

> **Lưu ý phạm vi:** `P` **không** ảnh hưởng tới giá trị trung bình khi ngoại suy mù
> (EKF truyền `x' = f(x)`, không dùng `P`), nên lỗi này **không** giải thích các con số
> FDE ở trên. Nhưng nó **có** ảnh hưởng tới `gating_distance` và Kalman gain ở **mọi**
> bước `update` trong pipeline thật.
>
> **Trạng thái lịch sử:** tại thời điểm ghi phần này, cách sửa là thu nhỏ cả **hàng và
> cột** tương ứng theo tỉ lệ `sqrt(phương_sai_mới / phương_sai_cũ)` thay vì chỉ ghi đè
> đường chéo — khi đó trị riêng nhỏ nhất trở lại **+2,7·10⁻⁹**. Sửa lỗi và đánh giá lại
> được ghi trong `FIX_REPORT.md` và `TEST_EVALUATION_REPORT.md`; không dùng đoạn minh
> hoạ này để suy luận về kết quả test hiện hành.

#### Ghi chú: giả thuyết "detection bị cắt xén" không đúng ở đoạn này

Giả thuyết ban đầu là detection cuối bị che một phần làm `ω̂` nhiễu. Đo được:
`occlusion_ratio` tại frame neo của đoạn 2 chỉ **0,077** — xe còn nhìn thấy **92 %**.
Nên biến thể "triệt `ω` có điều kiện" (ngưỡng 0,5) **không kích hoạt**. Đã cài cả hai
biến thể (có điều kiện / vô điều kiện) để sanity check không bị vô hiệu hoá.

Bán kính quỹ đạo tại neo đoạn 2: `R = v/|ω| = 30 px` — **nhỏ hơn cả chiếc xe**
(185×91 px); cung quét dự kiến **487,5°**, tức box đi hết hơn một vòng tròn.

---

## 5. Tính chặt chẽ đã xử lý

1. **Sửa lỗi thống kê nghiêm trọng:** `binomtest(...).pvalue * 2` — `binomtest` mặc định
   đã là two-sided, nhân đôi làm **mọi p-value gấp đôi** (0.4375 → đúng là 0.2188).
2. **Chỉ so sánh trên tập đoạn chung** của cả 3 model, để mẫu số giống hệt nhau.
3. **Bootstrap giữ tính ghép cặp**, báo cáo cả cluster bootstrap **theo track**.
4. **Cảnh báo so sánh bội:** phát hiện "có ý nghĩa" duy nhất ở Giai đoạn D cũ
   (`partial × long`) đã **tan biến** khi đổi cấu hình (p 0.0703 → 0.1796) — xác nhận
   là dương tính giả.
5. **Độ cong đo theo TỪNG ĐOẠN CHE**, không phải cả track. So sánh 3 cách đo, chọn cách
   tương quan 0.899 với phép khớp hình học (cách tích luỹ chỉ 0.572).
6. **Sửa một ghi chú SAI từ Giai đoạn 1:** README ghi *"tập test không có ground truth
   công khai nên vô dụng"*. **Sai** — XML tập test có đầy đủ `occlusion` và
   `region_overlap`. Ghi chú này đã làm mất 40 video trong suốt các giai đoạn trước.
7. **Chống rò rỉ ở Giai đoạn C bằng 3-fold**, ở Giai đoạn D bằng train/test tách biệt.
8. **Ba lớp bảo vệ chống optional stopping** ở Giai đoạn D: ô đăng ký trước, không tìm
   ô mới, báo cáo train/test riêng biệt.
9. **Sửa lỗi sẽ làm hỏng âm thầm:** `baseline_track.py` lọc cứng `classes=[2,3,5,7]`;
   model fine-tune chỉ có 1 lớp → sẽ ra **rỗng mọi frame** mà vẫn "chạy thành công".
10. **Sửa FPS sai:** con số "EKF 80.8 FPS nhanh hơn KF 54.6" là **nhiễu đo**. Đo lại 5
    lần lặp: FPS end-to-end **không phân biệt được** (p = 0.68–0.92, CV tới 26.8%).
    Chi phí thật: EKF **1.60×**, UKF **6.74×** so với KF.
11. **Gộp đoạn che chồng nhau.** Bảng che-một-phần (≥0.10) và che-hoàn-toàn (≥0.90) mô
    tả *cùng một lần bị che* ở hai mức, nên đoạn "hoàn toàn" luôn nằm **lồng trong** đoạn
    "một phần". Không gộp thì một lần bị che bị đếm thành 2–3 đoạn và **FDE bị tính ở
    giữa đoạn** — sai hoàn toàn ý nghĩa.
12. **Bịt bẫy ghi đè âm thầm.** `stratified_analysis.py` và `compare_models.py` lấy danh
    sách model mặc định từ `config.MOTION_MODELS`. Thêm biến thể kiểm chứng
    (`ekf_ctrv_clamped`) vào config sẽ khiến **lệnh cũ không đổi tham số** lặng lẽ phân
    tích 4 model và **ghi đè** `segment_outcomes_*.csv` / `comparison_*.csv` đã báo cáo.
    Đã chốt cứng `MAIN_MODELS` và thêm `--tag` cho cả hai script.
13. **Bản gốc không bị sửa khi thử biến thể.** `ekf_ctrv.py` giữ nguyên; bản chặn `ω`
    nằm ở file riêng kế thừa lại. Kết quả mới ghi vào thư mục/hậu tố riêng, đã kiểm chứng
    bằng `git show --stat` rằng không commit nào chạm vào file kết quả cũ.
14. **Chọn video minh hoạ bằng quét toàn bộ, không chọn tay.** Sau khi bị chất vấn "vài
    ca lẻ tẻ thì thiếu thuyết phục", đã quét cả 48 track ứng viên (mục 4.8) và báo cáo
    phân bố thắng/thua thay vì chỉ trưng ca thuận lợi.
15. **Báo Spearman kèm Pearson khi có ngoại lai.** Tương quan `ω` trước/trong đoạn che:
    Pearson −0,755 nhưng bỏ **một** điểm còn −0,385; Spearman −0,448 (p = 0,0023) mới là
    con số bền vững.
16. **Sửa lỗi `n_frames=1.0` cứng** trong 4 script chạy bộ lọc xuyên qua nhiều đoạn che
    (mục 4.7). Vận tốc bị thổi phồng đúng bằng số frame bị che (đo được 27 lần).
    Pipeline thật không dính; mọi số HOTA/IDSW không đổi, nhưng toàn bộ số của Giai đoạn
    E đã tính lại.
17. **Đối chứng TÁCH BIẾN thay vì suy luận.** Khi bản triệt `ω` cho kết quả bất thường,
    không dừng ở "chắc do `ω`" mà ép **cả** `ω = 0` **và** `v` = vận tốc của KF để tách
    hai biến. Kết quả (lệch tối đa 2,0 px so với KF trên 3 đoạn) hỗ trợ sanity check rằng
    CTRV suy biến gần CV trong ca minh hoạ; không loại trừ hay xác lập cơ chế cho test.
18. **Bắt được lỗi trong chính script kiểm chứng.** Bản đối chứng đầu tiên can thiệp ở
    **mọi** lần vào đoạn che (kể cả đoạn 1) làm hỏng trạng thái trước đoạn 2, cho kết quả
    202,1 px và kết luận sai là "còn nguyên nhân thứ ba". Sau khi giới hạn can thiệp đúng
    đoạn đang xét: 160,0 px — khớp KF. Ghi lại đây vì đó là lỗi suýt dẫn tới kết luận sai.
19. **Kiểm tra bất biến toán học của bộ lọc, không chỉ kiểm tra đầu ra.** Theo dõi trị
    riêng nhỏ nhất của `P` qua từng frame mới phát hiện `initiate_from_motion` phá vỡ
    tính nửa xác định dương (mục 4.9) — thứ mà mọi chỉ số HOTA/FDE đều không lộ ra.

---

## 6. Kết luận trung thực

Kết luận hiện hành dựa trên đánh giá lại 40 video ở mục 2.3, không dựa trên bảng test
lịch sử ở mục 4.6. Ảnh hưởng của motion model nhỏ và không ổn định giữa train/test.
Với `id_match`, bốn KTC 95% EKF−CV và UKF−CV đều chứa 0. Dữ liệu hiện có chưa đủ bằng
chứng về khác biệt trong phục hồi ID sau che khuất; kết quả này không cho phép gọi các
motion model tương đương, cũng không hỗ trợ tuyên bố EKF hoặc UKF thắng CV.

Ở điều kiện B (`conf=0.10`), EKF−CV có KTC danh nghĩa dương cho HOTA, IDF1 và AssA,
nhưng các so sánh chưa được hiệu chỉnh nhiều lần và các chỉ số khác không cùng xác lập
một kết luận tổng thể. Vì vậy chỉ nên trình bày đó là tín hiệu cần kiểm tra thêm, không
phải bằng chứng EKF thắng toàn diện.

Đổi A sang B đi kèm đánh đổi có thể quan sát: IDSW và FN giảm, `id_continuous` tăng,
nhưng FP tăng ở cả ba motion model. HOTA, IDF1 và AssA B−A đều giảm theo điểm ước lượng
trên test, trái hướng tăng theo điểm ước lượng đã thấy trên train. `id_match` chỉ sự
trùng ID trước/sau sự kiện; `id_continuous` yêu cầu duy trì liên tục, nên không được
diễn giải thay thế nhau. Các tổng ID, FP, FN và association không đủ để kết luận cơ chế
nhân quả khi không có log trực tiếp đầy đủ.

Phân tích che khuất có 2.174 sự kiện sau gộp, 1.324 sự kiện đủ điều kiện; 484 đoạn full
nằm trong partial đã được gộp để tránh đếm lặp. Tập test từng được sử dụng trước đây nên
đây là đánh giá xác nhận lại trên tập đã biết, không phải đánh giá held-out hoàn toàn.
Thời gian chạy chỉ mô tả một lần chạy trên tải máy không kiểm soát, không phải benchmark.
Nguồn cho toàn bộ số hiện hành: `results/test_eval/summary/configuration_metrics.csv`,
`results/test_eval/summary/paired_contrasts_bootstrap.csv`,
`results/test_eval/summary/validation.csv` và `TEST_EVALUATION_REPORT.md`.

---

## 7. Khung luận văn đề xuất

**Tên:** *"Vì sao chỉ số tổng hợp không phát hiện được lợi thế của mô hình chuyển động
phi tuyến: phân tầng, hiệu ứng trần và giới hạn dữ liệu trong theo dõi đa đối tượng
có che khuất"*

**Mạch:** (1) giả thuyết → (2) mô phỏng cho thấy cơ chế trong điều kiện kiểm soát →
(3) phân tích lịch sử trên video thật → (4) đánh giá lại 40 video theo manifest đã khóa,
TrackEval và bootstrap ghép cặp theo video → (5) kết luận có điều kiện theo từng chỉ số.

**Điểm mạnh của phần thực nghiệm cuối:**
- Sáu cấu hình A/B × CV/EKF/UKF dùng cùng 40 video và cùng GT; 240/240 file tracking
  hợp lệ, với kiểm tra gộp sự kiện che khuất tránh đếm lặp.
- Báo cáo tách `id_match` khỏi `id_continuous`, đồng thời đưa KTC bootstrap theo video
  thay vì diễn giải chỉ từ dấu của chênh lệch.
- Kết quả train và test được đặt cạnh nhau: ảnh hưởng của motion model nhỏ, không ổn
  định giữa hai tập; đổi ngưỡng detector tạo đánh đổi IDSW/FN, FP và tính liên tục.
- Các phân tích Giai đoạn A–F là tư liệu phương pháp và chẩn đoán lịch sử; chúng cần
  được trình bày với phạm vi dữ liệu tương ứng, không dùng làm kết luận tổng quát.

**Một mạch phụ đáng kể cho phần phương pháp:** khuyết điểm `ω` không bị chặn được phát
hiện bằng **quan sát video**, sau khi mọi chỉ số tổng hợp đã "sạch". Điều này minh hoạ
đúng luận điểm trung tâm của luận văn — chỉ số tổng hợp che giấu cơ chế — và cho thấy
kiểm tra trực quan là một phần bắt buộc của quy trình đánh giá, không phải trang trí.

---

## 8. Công cụ đã xây

| File | Nội dung |
|---|---|
| `src/ekf_ctrv.py` / `src/ukf_ctrv.py` | EKF/UKF + CTRV, 14 và 12 unit test |
| `src/tracker_ctrv.py` | Ghép filter CTRV vào ByteTrack (3 điểm chạm) |
| `src/stratified_analysis.py` | Ghép GT↔tracker (Hungarian/IoU), phân tầng, McNemar, bootstrap |
| `src/diagnose_ukf.py` | Chẩn đoán UKF: xử lý góc, dây cung, độ bền vững `ω` |
| `src/benchmark_fps.py` | Đo FPS 2 tầng (micro + end-to-end có lặp) |
| `src/buffer_sweep_analysis.py` | Quét `track_buffer` |
| `src/tune_omega_noise.py` | Quét nhiễu quá trình cho `ω` |
| `src/omega_smoother.py` | **GĐ A** — RTS smoother + circle fit + phân tích `ω` lịch sử |
| `src/lost_association.py` | **GĐ B** — DIoU chuẩn hoá + expanded gate cho track LOST |
| `src/phaseB_iou_diagnosis.py` / `src/phaseB_analysis.py` | **GĐ B** — chẩn đoán và đánh giá |
| `src/export_yolo_dataset.py` | **GĐ C** — xuất YOLO, 3-fold chia theo video |
| `src/eval_detector.py` | **GĐ C** — chấm AP trước/sau, tự viết vì lệch số lớp |
| `src/phaseD_confirm.py` | **GĐ D** — kiểm định xác nhận, chống optional stopping |
| `src/ekf_ctrv_clamped.py` | **GĐ E** — EKF+CTRV chặn `\|ω\| ≤ 0,05` rad/frame (kế thừa, không sửa bản gốc) |
| `src/diagnose_ekf_circle.py` | **GĐ E** — chẩn đoán hiện tượng box quay vòng: `ω`, `R = v/ω`, cung quét |
| `src/omega_clamp_experiment.py` | **GĐ E** — so KF / EKF gốc / EKF chặn `ω` trên cùng tập đoạn |
| `src/demo/diagnose_seg2.py` | **GĐ 0** — loại trừ lỗi ghép cặp / khoảng trống frame trên 1 đoạn |
| `src/demo/omega_suppress.py` | **GĐ F** — triệt tiêu `ω`, đối chứng tách biến `ω` vs vận tốc, video 3 panel |
| `src/demo/gt_omega.py` / `case_ab.py` | Demo — tính `ω` từ GT, phân loại Case A/B |
| `src/demo/render_videos.py` | Demo — 4 video minh hoạ, chọn đoạn bằng tiêu chí tự động |
| `src/demo/side_by_side.py` | Demo — 2 panel trên **kết quả tracker thật**, đánh dấu đoạn 2 model khác nhau |
| `src/demo/long_compare.py` | Demo — video dài, theo 1 xe qua nhiều lần che, 2 panel |

---

## 9. Câu hỏi muốn được hỗ trợ

1. Khung luận văn ở mục 7 có hợp lý không?
2. Với p = 0.0627 trên tập test nhưng không tái lập trên train, cách trình bày nào là
   trung thực nhất mà vẫn không tự phủ nhận công sức?
3. Có nên gộp train+test để tăng power, hay giữ riêng biệt vì tính xác nhận độc lập
   quan trọng hơn?
4. Ba "trần" (thông tin / `track_buffer` / detector) nên trình bày như phát hiện chính
   hay như giải thích cho kết quả âm tính?
5. Cách trình bày "hiệu ứng dây cung của UKF" sao cho hội đồng dễ hiểu nhất?
6. Mục 2.3 (thất bại đồng nhất 998/998) và mục 4.8 (CV thắng 32/48 track) là bằng chứng
   **bất lợi** cho giả thuyết ban đầu. Nên đặt chúng ở **đầu** phần kết quả như một phát
   hiện, hay ở **cuối** như phần thảo luận giới hạn?
7. Giai đoạn E cho thấy tôi tự tìm ra khuyết điểm trong code của chính mình, đo tác động,
   rồi kết luận nó không đảo ngược kết quả. Có nên đưa hẳn vào luận văn không, hay hội
   đồng sẽ đọc thành "code có lỗi"?
