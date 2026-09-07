# Báo cáo tổng hợp khoá luận: So sánh KF / EKF / UKF cho Multi-Object Tracking trên UA-DETRAC

> Tài liệu tự chứa — đọc xong là nắm được toàn bộ tình hình, không cần bối cảnh khác.
> **Cập nhật lần 2:** bổ sung toàn bộ Giai đoạn A–D (4 hướng cải thiện).

---

## 1. Đề tài và thiết lập

**Câu hỏi nghiên cứu:** Mô hình chuyển động phi tuyến CTRV (Constant Turn Rate and
Velocity) có giúp giữ định danh (ID) của xe tốt hơn mô hình tuyến tính CV (Constant
Velocity) khi xe **vừa rẽ vừa bị che khuất** không?

**Pipeline:** YOLOv8n (`conf=0.25`) → ByteTrack → TrackEval (HOTA / MOTA / IDF1).
Chỉ **motion model** thay đổi; detector và data association giữ nguyên tuyệt đối.

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
| **40 video TEST, detector fine-tune** | 0.6325 | **0.6344** | 0.6334 |

*(HOTA. Thứ tự đổi ở mỗi cấu hình — dấu hiệu chênh lệch nằm trong vùng nhiễu.)*

---

## 3. Bốn nguyên nhân khiến CTRV không thắng

### 3.1 Motion model chỉ có "tiếng nói" ở ~3–4% trường hợp
3 model cho kết cục **giống hệt nhau** ở 96–97% số đoạn, ở mọi cấu hình đã thử.

### 3.2 `track_buffer` = trần cứng
`track_buffer=30` frame = 1.2s. Quá ngưỡng, ByteTrack **xoá hẳn track**.

| Độ dài đoạn (che ≥0.90) | n | Giữ được ID |
|---|---|---|
| ≤ 30 frame | 257 | 52 (**20.2%**) |
| > 30 frame | 21 | 0 (**0.0%**) |

### 3.3 `ω` ước lượng được KHÔNG đáng tin — và đây là **giới hạn thông tin**
Trên 507 đoạn: `ω` đúng dấu chỉ **55.6%** (EKF), độ lớn **gấp 3 lần** GT.
**Giai đoạn A đã chứng minh đây là trần, không phải lỗi ước lượng** — xem mục 4.3.

### 3.4 Gần một nửa số đoạn thất bại do detector
`lost` = 45.9% với detector COCO. **Giai đoạn C giảm còn 35.1%** — xem mục 4.5.

### 3.5 Riêng UKF: hiệu ứng dây cung
`α=1, κ=0, n=9` → `λ=0` → **`Wm[0]=0`**: kỳ vọng dự đoán bỏ qua `f(x̂)`, chỉ là trung
bình 18 điểm sigma lệch tâm. Đo được: quỹ đạo thật `R=152.79 px`, EKF dự đoán **chính
xác 152.79** ở mọi mức `P`; UKF hụt còn 141 → 137 → **111** khi `P` phình.
**Đã kiểm chứng UKF không có bug** ở cả 3 chỗ xử lý góc.

---

## 4. Sáu hướng cải thiện đã thử

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

> **PHÁT HIỆN QUAN TRỌNG NHẤT — TRẦN THÔNG TIN.** Đo bằng **GT hoàn hảo ở cả hai phía**
> (không filter, không nhiễu, không sai số ước lượng), trên 361 đoạn:
> **ω GT trước che → ω GT trong che: đúng dấu 58.2%, Pearson +0.0093** (≈ 0).
> Bộ lọc hiện tại đạt 59.3% trên cùng tập — **đã chạm trần**. Không bộ ước lượng nào,
> dù hoàn hảo, có thể vượt qua: thông tin về khúc cua sắp xảy ra **không tồn tại**
> trong dữ liệu trước đoạn che.

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

### 4.6 GIAI ĐOẠN D — Mở rộng sang tập TEST 40 video → **GẦN NGƯỠNG NHẤT**

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

---

## 6. Kết luận trung thực

**Chưa chứng minh được CTRV cải thiện việc giữ ID, nhưng cũng KHÔNG bác bỏ được.**

Sau 4 giai đoạn cải thiện có hệ thống:
- **A đóng** — giới hạn là **thông tin**, không phải ước lượng (trần 58.2%)
- **B đóng** — nới cửa liên kết lợi bất cập hại trên dữ liệu đông xe
- **C thành công lớn về tracking** nhưng **xoá luôn ưu thế biểu kiến của EKF**
- **D đưa p về 0.0627** nhưng **hai tập độc lập không đồng thuận**

**Cách phát biểu đúng:** *"Dữ liệu hiện có không đủ để phân biệt 3 motion model"* —
**không phải** *"3 motion model như nhau"*. Đây là kết quả **underpowered**, khác hẳn
kết quả âm tính.

**Cấu hình tốt nhất đo được:** EKF hoặc UKF + CTRV với detector fine-tune, `track_buffer` lớn.
**UKF vẫn không có lý do để dùng** trong ứng dụng thực: chậm hơn EKF **6.74×** mà không
tốt hơn đáng kể.

---

## 7. Khung luận văn đề xuất

**Tên:** *"Vì sao chỉ số tổng hợp không phát hiện được lợi thế của mô hình chuyển động
phi tuyến: phân tầng, hiệu ứng trần và giới hạn dữ liệu trong theo dõi đa đối tượng
có che khuất"*

**Mạch:** (1) giả thuyết → (2) mô phỏng xác nhận cơ chế (15×) → (3) video thật có vẻ
mâu thuẫn → (4) **đóng góp phương pháp luận**: phép đo phân tầng đúng cơ chế →
(5) tín hiệu đúng hướng nhưng chưa đủ ý nghĩa → (6) **bốn hướng cải thiện có hệ thống**,
mỗi hướng đều truy được cơ chế → (7) **ba trần chồng lên nhau**: `track_buffer`,
detector, và **giới hạn thông tin** → (8) kết luận về phương pháp đánh giá.

**Điểm mạnh:** ba phát hiện **có giá trị độc lập** với việc giả thuyết ban đầu đúng hay sai:
- **Trần thông tin** (Giai đoạn A): chứng minh định lượng rằng `ω` không dự báo được
- **Trần detector** (Giai đoạn C): detector là nút thắt chính, và cải thiện nó *xoá* ưu
  thế biểu kiến của CTRV
- **Hiệu ứng dây cung** (UKF): lý thuyết cao cấp hơn không đảm bảo thực nghiệm tốt hơn

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
| `src/omega_smoother.py` | **GĐ A** — RTS smoother + circle fit + đo trần thông tin |
| `src/lost_association.py` | **GĐ B** — DIoU chuẩn hoá + expanded gate cho track LOST |
| `src/phaseB_iou_diagnosis.py` / `src/phaseB_analysis.py` | **GĐ B** — chẩn đoán và đánh giá |
| `src/export_yolo_dataset.py` | **GĐ C** — xuất YOLO, 3-fold chia theo video |
| `src/eval_detector.py` | **GĐ C** — chấm AP trước/sau, tự viết vì lệch số lớp |
| `src/phaseD_confirm.py` | **GĐ D** — kiểm định xác nhận, chống optional stopping |

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
