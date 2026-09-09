# FIX_REPORT — Sửa lỗi khởi tạo bộ lọc và phép đánh giá che khuất

**Nhánh:** `fix/tracker-evaluation-correctness` (tách từ `main` tại `568ba2a`)
**Chưa push, chưa merge.** Không ghi đè bất kỳ kết quả thí nghiệm cũ nào —
kết quả mới nằm trong `results/eval_fixed/`.

**Môi trường:** Python 3.11.9 · numpy 2.4.6 · scipy 1.17.1 · pandas 2.3.3 ·
ultralytics 8.4.141 · torch 2.6.0+cu124 · GPU GTX 1650 · conda env `khoaluan-mot`

**Cấu hình giữ nguyên:** `yolov8n.pt`, `conf=0.25`, mọi ngưỡng tracker như baseline.
Không đổi dataset, detector, thuật toán, không tinh chỉnh tham số để tăng điểm.

---

## 1. Tóm tắt

| # | Lỗi | Trạng thái | Ảnh hưởng kết quả cũ |
|---|---|---|---|
| 1 | `initiate_from_motion` làm `P` mất tính bán xác định dương | ✅ Đã sửa | Có — cần chạy lại tracking |
| 2 | Đánh dấu khởi tạo chuyển động xong dù thất bại | ✅ Đã sửa | Có — cần chạy lại tracking |
| 3 | Hungarian chạy trên toàn bộ IoU rồi mới lọc ngưỡng | ✅ Đã sửa | Có — chỉ cần chấm lại |
| 4 | Đoạn che `full` lồng trong `partial` bị đếm hai lần | ✅ Đã sửa | Có — chỉ cần chấm lại |
| 5 | Trường hợp không đánh giá được bị tính là tracker thất bại | ✅ Đã sửa | Có — chỉ cần chấm lại |
| 6 | Sự kiện tracker chưa từng ghép được bị bỏ qua âm thầm | ✅ Đã sửa | Có — chỉ cần chấm lại |
| 7 | Vòng association thứ hai của ByteTrack bị vô hiệu hoàn toàn | 🔍 Đã đo, **chưa sửa** | Không — là đặc tính cấu hình |

Bốn lỗi đầu đều **tái hiện được bằng ví dụ cụ thể** trước khi sửa.

> **Chưa có bằng chứng nào cho thấy EKF/UKF vượt KF**, và pipeline **chưa** được
> chứng minh là đã đúng hoàn toàn. Báo cáo này chỉ nói về các lỗi đã xác nhận.

---

## 2. Lỗi 1 — `P` mất tính bán xác định dương

### Ví dụ tái hiện

`src/repro_init_covariance.py`, đúng kịch bản được yêu cầu: khởi tạo
`[100, 200, 0.5, 40]`, predict N bước, gọi `initiate_from_motion` với
`[100+3N, 200+4N, 0.5, 40]` và `n_frames=N`.

| N | Trị riêng nhỏ nhất **trước** | **sau** (bản cũ) | Kết quả |
|---|---|---|---|
| 1 | +2,000·10⁻¹⁰ | +2,000·10⁻¹⁰ | đạt |
| 5 | +6,000·10⁻¹⁰ | **−1,413·10⁻²** | **hỏng** |
| 10 | +1,100·10⁻⁹ | **−5,858·10⁻²** | **hỏng** |
| 26 | +2,700·10⁻⁹ | **−2,207·10⁻¹** | **hỏng** |

6/8 trường hợp hỏng (cả EKF và UKF — UKF kế thừa hàm này). Trên dữ liệu thật,
`MVI_40992` track 12 phân kỳ tới **−703** trong lúc ngoại suy mù.

### Nguyên nhân

```python
covariance[self.THETA, self.THETA] = config.CTRV_HEADING_STD_AFTER_INIT ** 2
covariance[self.V, self.V] = (...) ** 2
```

Ghi đè **phần tử đường chéo** nhưng giữ nguyên **phần tử ngoài đường chéo**.
Ma trận hiệp phương sai phân rã được thành `P = D C D` với `C` là ma trận tương
quan. Thu nhỏ `σ_i` mà giữ `P_ij` kéo hệ số tương quan ngầm
`ρ_ij = P_ij / (σ_i σ_j)` vượt quá 1 → `P` không còn bán xác định dương.
Đo được: `P[θ,θ]` giảm từ 9,15 xuống 0,09 (100 lần) trong khi `P[v,cx] = 498,76`
giữ nguyên.

### Cách sửa và lý do

Thêm `set_marginal_variance()` trong [src/ekf_ctrv.py](src/ekf_ctrv.py) — đặt lại
phương sai biên duyên bằng **phép biến đổi đồng dạng**:

```
P' = S P Sᵀ ,  S = diag(1, …, s, …, 1) ,  s = σ'/σ
```

tức nhân **cả hàng và cột** với cùng hệ số. **Định lý quán tính Sylvester** bảo
đảm phép đồng dạng với `S` khả nghịch giữ nguyên **dấu** của mọi trị riêng, nên
`P'` bán xác định dương khi và chỉ khi `P` bán xác định dương. Mọi hệ số tương
quan được giữ nguyên.

Đây là **sự thật toán học**, không phải cắt trị riêng âm hay thêm jitter để che
lỗi gốc. Test `1b` kiểm tra trực tiếp tính chất này trên 20 ma trận ngẫu nhiên.

**Sau khi sửa:** 0/8 trường hợp hỏng. Trị riêng sau khởi tạo **bằng đúng** trị
riêng trước khởi tạo (2,700·10⁻⁹ → 2,700·10⁻⁹), đúng như định lý dự báo.

---

## 3. Lỗi 2 — Đánh dấu khởi tạo xong dù thất bại

### Ví dụ tái hiện

`_init_motion_if_needed` luôn đặt `_motion_initialized = True`, kể cả khi bộ lọc
trả về nguyên trạng vì dịch chuyển dưới `CTRV_MIN_SPEED_FOR_HEADING = 0.5`.
Kịch bản: 1 frame đứng yên (dịch 0,1 px) rồi chạy thật 8 px/frame, 6 frame.

| Trục | Logic | v cuối | θ cuối | Kết quả |
|---|---|---|---|---|
| **y** | **cũ** | EKF 2,66 · **UKF 0,14** | 49,3° · 4,3° | **hỏng** |
| **y** | mới | EKF 8,00 · UKF 8,30 | 90,0° · 90,0° | đạt |
| x | cũ | EKF 6,86 · UKF 6,68 | 0,0° | đạt |
| x | mới | EKF 8,00 · UKF 8,30 | 0,0° | đạt |

UKF **kẹt hoàn toàn** (v = 0,14 thay vì 8). Đáng chú ý: **trục x vẫn chạy được
với logic cũ** vì `θ` khởi tạo = 0 tình cờ khớp — lỗi bị che giấu nếu chỉ test
theo phương ngang. Đó là lý do regression test kiểm tra **cả hai trục**.

### Cách sửa và lý do

Trong [src/ekf_ctrv.py](src/ekf_ctrv.py):
- Thêm `try_initiate_from_motion()` trả về `(mean, cov, ok)`.
- `initiate_from_motion()` giữ nguyên chữ ký cũ (2 giá trị) để các script phân
  tích hiện có không vỡ.
- Thêm tham số `ref_pos`: mốc tính dịch chuyển phải là vị trí **quan sát** trước
  đó, không phải vị trí trong `mean` (đã qua predict). Khi thử lại ở các frame
  sau, `v` có thể đã khác 0 nên predict làm dịch vị trí, và `dx, dy` sẽ thành
  **phần dư sau predict** chứ không phải độ dịch chuyển thật.

Trong [src/tracker_ctrv.py](src/tracker_ctrv.py):
- `_motion_initialized = bool(ok)` — **thử lại** ở quan sát sau nếu chưa được.
- Lưu `_last_obs_xy`, cập nhật trong `update()` và `re_activate()`.
- `activate()` đặt mốc ban đầu từ quan sát đầu tiên.

Không đổi giao diện tracker: `predict`, `multi_predict`, `tlwh`, `activate`,
`update`, `re_activate` giữ nguyên chữ ký.

---

## 4. Lỗi 3 — Ghép IoU: Hungarian trước, lọc ngưỡng sau

### Ví dụ tái hiện

Đúng ma trận được nêu, ngưỡng 0,5:

```
[[0.1281, 0.3739],
 [0.3963, 0.6384]]
```

Hungarian tối ưu **tổng** IoU nên chọn đường chéo phụ (0,7702) thay vì đường
chéo chính (0,7665). Cả hai cặp đều dưới ngưỡng → loại hết.
**Cặp (1,1) = 0,6384 hợp lệ bị mất.** Kết quả đo: cách cũ giữ được **0** cặp,
cách mới giữ **1** cặp.

### Cách sửa và mục tiêu tối ưu

Trong [src/occlusion_eval.py](src/occlusion_eval.py), hàm `match_frame()`.

**Mục tiêu tối ưu (phát biểu rõ):**
1. Tối đa hoá **số** cặp hợp lệ (IoU ≥ ngưỡng).
2. Trong các lời giải cùng số cặp hợp lệ, tối đa hoá **tổng** IoU.

Cách cài: chi phí của cặp dưới ngưỡng = `BIG_M = 1e6`; cặp hợp lệ = `−IoU ∈ [−1, 0]`.
Vì `BIG_M` lớn hơn mọi tổng IoU có thể có, hàm mục tiêu trở thành **từ điển**.
Các cặp không hợp lệ còn sót (do Hungarian buộc phải ghép đủ `min(n,m)` cặp) bị
loại sau khi giải — và việc loại chúng **không thể** đẩy mất cặp hợp lệ nào, vì
đổi một cặp hợp lệ lấy một cặp không hợp lệ làm chi phí tăng ít nhất `BIG_M − 1`.

Cho phép đối tượng **không được ghép** (không ép ghép). So sánh ngưỡng dùng
`>=` — **đúng ngưỡng**, không phải `>`.

**Không sửa TrackEval** — lỗi chỉ nằm trong script phân tích riêng.

---

## 5. Lỗi 4–6 — Định nghĩa sự kiện che khuất

### 5.1 Đếm lặp (lỗi 4)

Đoạn che `full` (≥ 0,90) **luôn nằm lồng** trong đoạn `partial` (≥ 0,10) của cùng
một lần bị che — bbox bị che 90 % thì đương nhiên cũng bị che 10 %. Gộp hai bảng
rồi coi mỗi dòng là một mẫu độc lập là **đếm cùng một lần bị che hai lần**.

Đo trên dữ liệu thật (60 video train):

| | Số |
|---|---|
| Đoạn thô | 2.765 partial + 584 full = **3.349** |
| Sau khi gộp thành sự kiện | **2.766** |
| **Giảm do đếm lặp** | **583** |

583/584 đoạn `full` (**99,8 %**) nằm lồng trong `partial`. Đơn vị phân tích mới:
một **sự kiện che khuất** = hợp của các đoạn chồng lấn/kề nhau của cùng
`(video, track_id)`. Mức độ ghi lại làm thuộc tính `has_full`, **không tạo thêm mẫu**.

### 5.2 Không đánh giá được ≠ tracker thất bại (lỗi 5)

Bản cũ đặt `status = "lost"` khi không tìm thấy kết nối sau đoạn che, không phân
biệt GT còn hay hết. Bản mới tách rõ:

**Điều kiện đủ xác định từ GT** (không dùng kết quả tracker nào), nhờ vậy mọi
tracker được chấm trên **đúng cùng một tập sự kiện**:

| Phân loại | Ý nghĩa | Số sự kiện |
|---|---|---|
| `eligible` | GT có mặt cả trước và sau sự kiện | **1.003** |
| `gt_missing_before` | Track sinh ra khi đã bị che | 1.186 (loại) |
| `gt_missing_after` | Hết video / xe rời cảnh | 577 (loại) |

Với sự kiện đủ điều kiện, kết quả tracker:

| Kết quả | Ý nghĩa |
|---|---|
| `preserved` | ID trước == ID sau |
| `switched` | ID trước != ID sau |
| `lost_after` | Ghép được trước, không ghép được sau — **GT vẫn còn** → thất bại thật |
| `no_match_before` | Chưa từng ghép được trước sự kiện |

### 5.3 Không bỏ qua âm thầm (lỗi 6)

Bản cũ đặt `no_before` rồi lặng lẽ loại — việc này **thiên vị tracker hỏng sớm**.
Bản mới tính là `no_match_before` và **báo cáo**. Đo được: **19,9–20,0 %** số sự
kiện rơi vào nhóm này. File tracking rỗng cũng không bị bỏ qua: sự kiện vẫn được
chấm (test `B5`).

### 5.4 Định nghĩa "giữ ID"

Hai khái niệm **khác nhau**, đo riêng, không dùng lẫn:

| Khái niệm | Định nghĩa |
|---|---|
| `id_match` | ID ngay trước == ID ngay sau. **Không** đòi hỏi liên tục. |
| `id_continuous` | ID đó được duy trì ở **mọi** frame có GT trong sự kiện. |

Bản cũ chỉ đo `id_match` nhưng gọi là "giữ được ID". Đo được chênh lệch lớn:
48,3 % so với 30,2 %.

---

## 6. Chấm lại kết quả tracking cũ bằng phép đánh giá đã sửa

> **Đây là tracking CŨ được chấm lại — chưa phản ánh bản sửa bộ lọc (lỗi 1, 2).**
> Muốn có số phản ánh cả bản sửa bộ lọc thì phải chạy lại tracking (lượt A).

Trên **1.003 sự kiện đủ điều kiện**, **55 video**:

| Tracker | `id_match` | `id_continuous` |
|---|---|---|
| `yolov8n-bytetrack` (KF + CV) | **48,26 %** | **30,21 %** |
| `yolov8n-ekf-ctrv` | 47,46 % | 30,01 % |
| `yolov8n-ukf-ctrv` | 47,56 % | 30,01 % |

Phân bố kết quả (KF + CV): `preserved` 48,3 % · `switched` 20,2 % ·
`lost_after` 11,6 % · `no_match_before` 19,9 %.

**Bootstrap theo VIDEO** (5.000 lần, giữ tính ghép cặp), chênh lệch so với KF + CV:

| Tracker | Chênh `id_match` | KTC 95 % |
|---|---|---|
| `yolov8n-ekf-ctrv` | −0,80 % | [−1,83 %, +0,19 %] |
| `yolov8n-ukf-ctrv` | −0,70 % | [−1,56 %, +0,08 %] |

**Cách đọc đúng:** khoảng tin cậy chứa 0 **không** có nghĩa hai mô hình tương
đương — chỉ có nghĩa dữ liệu hiện có chưa đủ để phân biệt. Lấy mẫu ở cấp video
vì các sự kiện trong cùng video **không độc lập** (chung cảnh, chung điều kiện
ánh sáng, chung detector). Nếu dùng McNemar theo từng sự kiện, p-value sẽ **hẹp
hơn thực tế** vì coi các sự kiện là độc lập.

---

## 7. Lỗi 7 — Vòng association thứ hai bị vô hiệu (đã đo, chưa sửa)

`ultralytics 8.4.141`, `byte_tracker.py` dòng 316–317:

```python
remain_inds = valid & (scores >= self.args.track_high_thresh)   # vòng 1
inds_low    = valid & (scores >  self.args.track_low_thresh)
                    & (scores <  self.args.track_high_thresh)   # vòng 2
```

Baseline dùng YOLO `conf=0.25` và `track_high_thresh=0.25`. YOLO đã **lọc bỏ**
mọi detection dưới 0,25 trước khi vào tracker → dải `(0.10, 0.25)` **rỗng**.

**Đo trực tiếp** (`src/probe_conf_bands.py`, 3 video × 50 frame, ngưỡng dò 0,01):

| Dải điểm | Số detection | Tỉ lệ |
|---|---|---|
| ≥ 0,25 (vào vòng 1) | 1.254 | 13,6 % |
| **(0,10 – 0,25) (vào vòng 2)** | **1.123** | **12,2 %** |
| ≤ 0,10 (bỏ hoàn toàn) | 6.816 | 74,1 % |

Dải vòng 2 bằng **89,6 %** so với vòng 1 — không hề nhỏ.

**Xác nhận trên tracking thật** (`src/smoke_three_models.py`, 2 video × 100 frame),
ba con số phân biệt rõ như yêu cầu:

| | conf = 0,25 | conf = 0,10 |
|---|---|---|
| Frame **không có** detection điểm thấp | **200/200 (100 %)** | 0/200 (0 %) |
| Detection điểm thấp **đưa vào** vòng 2 | **0** | 2.078 |
| Track **đủ điều kiện** tham gia | 105–107 | 270–283 |
| Cặp **ghép thành công** | **0** | **230–243** |

Với `conf=0.25`, cơ chế đặc trưng của ByteTrack **không bao giờ chạy**, dù có
105–107 track đủ điều kiện đang chờ.

> **Chưa khẳng định `conf=0.1` tốt hơn.** Detection điểm thấp cũng mang theo
> false positive; tỉ lệ ghép được chỉ 11,1–11,7 % số detection đưa vào. Phải
> chạy lượt B rồi đo mới biết.

**Chưa sửa** vì việc này đổi cấu hình baseline đang khoá — thuộc phạm vi lượt B.

---

## 8. File đã thay đổi

| File | Thay đổi |
|---|---|
| `src/ekf_ctrv.py` | Thêm `set_marginal_variance()`, `try_initiate_from_motion()`; `initiate_from_motion()` thành wrapper giữ chữ ký cũ |
| `src/tracker_ctrv.py` | `_motion_initialized` chỉ đặt khi thành công; thêm `_last_obs_xy` và `_remember_obs()` |
| `src/repro_init_covariance.py` | **mới** — tái hiện lỗi 1 và 2 |
| `src/test_init_regression.py` | **mới** — 16 regression test |
| `src/occlusion_eval.py` | **mới** — phép đánh giá đã sửa (lỗi 3–6) |
| `src/test_occlusion_eval.py` | **mới** — 14 test cho ghép IoU và định nghĩa sự kiện |
| `src/probe_conf_bands.py` | **mới** — đo phân bố điểm tin cậy |
| `src/assoc_stats.py` | **mới** — đếm hoạt động vòng association thứ hai |
| `src/smoke_three_models.py` | **mới** — smoke test 3 model, xác minh lớp bộ lọc runtime |

`src/stratified_analysis.py` **không sửa** — giữ nguyên để kết quả cũ tái lập được.

---

## 9. Lệnh đã chạy và kết quả thực tế

```bash
conda activate khoaluan-mot

# Tái hiện lỗi 1 và 2
python src/repro_init_covariance.py

# Test
python src/test_ekf_ctrv.py            # 14/14 PASS  (không đổi)
python src/test_ukf_ctrv.py            # 12/12 PASS  (không đổi)
python src/test_init_regression.py     # 16/16 PASS  (mới)
python src/test_occlusion_eval.py      # 14/14 PASS  (mới)

# Chấm lại tracking cũ bằng phép đánh giá đã sửa
python src/occlusion_eval.py --split-name DETRAC-all \
  --trackers yolov8n-bytetrack yolov8n-ekf-ctrv yolov8n-ukf-ctrv \
  --out-dir results/eval_fixed

# Đo dải điểm tin cậy
python src/probe_conf_bands.py --videos MVI_20011 MVI_40171 MVI_40992 --n-frames 50

# Smoke test 3 model + vòng association thứ hai
python src/smoke_three_models.py --videos MVI_20011 MVI_40171 --n-frames 100 --conf 0.25
python src/smoke_three_models.py --videos MVI_20011 MVI_40171 --n-frames 100 --conf 0.10
```

**Tổng: 56/56 test PASS.** Smoke test xác minh runtime dùng đúng
`KalmanFilterXYAH` / `EKFTrackerCTRV` / `UKFTrackerCTRV`, 0 track có
`mean`/`covariance` hỏng.

**Kết quả ghi vào:** `results/eval_fixed/` (thư mục mới, không đè lên gì).

---

## 10. Phần CHƯA kiểm chứng

| Việc | Thiếu gì |
|---|---|
| Ảnh hưởng của bản sửa bộ lọc lên HOTA/IDF1/IDSW | Phải chạy lại tracking 60 video (~1 giờ GPU/model) |
| Ảnh hưởng lên tập test 40 video | Như trên |
| `conf=0.1` có tốt hơn không | Phải chạy lượt B rồi đo |
| Lỗi 1–2 ảnh hưởng bao nhiêu tới các kết luận đã báo cáo | Cần lượt A xong mới so được |
| Detector fine-tune (Giai đoạn C) | Chưa chạy lại với bản sửa |

Smoke test chỉ 2 video × 100 frame — **quá nhỏ để so sánh hiệu năng**, mục đích
chỉ là xác minh runtime và tính đúng đắn.

---

## 11. Kết quả cũ nào cần tính lại

**Phải chạy lại tracking** (bản sửa bộ lọc đổi hành vi runtime):
- `results/comparison_DETRAC-all.csv` và mọi bảng HOTA/MOTA/IDF1/IDSW
- `results/trackeval/DETRAC-all/*`
- `data/processed/trackers/DETRAC-all/*` (kết quả tracking thô)
- Mọi kết luận trong `BAO_CAO_KHOA_LUAN.md` mục 2.2, 4.1–4.9

**Chỉ cần chấm lại** (không cần chạy tracking):
- `results/stratified/*` — thay bằng `results/eval_fixed/`
- Bảng 1.544 đoạn ở mục 2.3 báo cáo → nay là **1.003 sự kiện** đủ điều kiện
- Mọi con số tỉ lệ giữ ID và McNemar theo đoạn

**Không bị ảnh hưởng:**
- Giai đoạn A (trần thông tin) — đo trực tiếp từ GT, không qua bộ lọc
- Phân bố `ω` của GT, tương quan Spearman −0,448

---

## 12. Lệnh để tiếp tục lượt A và lượt B

Cả hai lượt ghi vào **thư mục riêng**, không đè baseline.

### Lượt A — code đã sửa, `conf=0.25`

```bash
conda activate khoaluan-mot
cd /d/LABOTARY/bytetrack
git checkout fix/tracker-evaluation-correctness

for MM in cv ekf_ctrv ukf_ctrv; do
  python src/baseline_track.py --all-train --split-name DETRAC-all \
    --motion-model $MM --conf 0.25 \
    --tracker-name "runA-$MM" --skip-existing
done

for MM in cv ekf_ctrv ukf_ctrv; do
  python src/run_trackeval.py --split-name DETRAC-all --tracker "runA-$MM"
done

python src/occlusion_eval.py --split-name DETRAC-all \
  --trackers runA-cv runA-ekf_ctrv runA-ukf_ctrv \
  --out-dir results/runA
```

### Lượt B — cùng code, `conf=0.1`

Chỉ đổi **một** biến: ngưỡng detector. Mọi ngưỡng tracker giữ y hệt lượt A.

```bash
for MM in cv ekf_ctrv ukf_ctrv; do
  python src/baseline_track.py --all-train --split-name DETRAC-all \
    --motion-model $MM --conf 0.10 \
    --tracker-name "runB-$MM" --skip-existing
done

for MM in cv ekf_ctrv ukf_ctrv; do
  python src/run_trackeval.py --split-name DETRAC-all --tracker "runB-$MM"
done

python src/occlusion_eval.py --split-name DETRAC-all \
  --trackers runB-cv runB-ekf_ctrv runB-ukf_ctrv \
  --out-dir results/runB
```

### So sánh A với B

```bash
python src/compare_models.py --split-name DETRAC-all \
  --models cv ekf_ctrv ukf_ctrv --tag runA
python src/compare_models.py --split-name DETRAC-all \
  --models cv ekf_ctrv ukf_ctrv --tag runB
```

**Thời gian ước tính:** ~1 giờ GPU cho mỗi model × 60 video, tức ~3 giờ mỗi lượt,
~6 giờ cho cả hai. Lượt B sẽ chậm hơn vì nhiều detection hơn.

> `compare_models.py` hiện suy tên thư mục tracker từ `config.MOTION_MODELS`
> (`yolov8n-<suffix>`), nên với tên `runA-*` cần thêm mục vào `MOTION_MODELS`
> hoặc đổi tên thư mục cho khớp. Chưa làm vì việc đó đụng cấu hình baseline.
