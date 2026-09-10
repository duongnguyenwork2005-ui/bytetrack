> **Cập nhật bàn giao test 10/09/2026:** Đã xong 40 video × 3 model × A/B; xem **mục 16** và [TEST_EVALUATION_REPORT.md](TEST_EVALUATION_REPORT.md). Commit kết quả đánh giá là **`b5bd086f8e4a625850ab35c977d1ed40776e810d`**, đã push lên `origin/fix/tracker-evaluation-correctness`; parent của nó là `aa45f6c`. Nhánh riêng này chưa merge vào `main`, nơi vẫn ở `568ba2a`. Bảng Git và các danh sách “còn thiếu” bên dưới là bản ghi lịch sử trước commit kết quả; toàn bộ nội dung báo cáo có sẵn lúc tiếp quản được giữ nguyên.

# FIX_REPORT — Sửa lỗi khởi tạo bộ lọc và phép đánh giá che khuất

**Nhánh:** `fix/tracker-evaluation-correctness` (tách từ `main` tại `568ba2a`)

**Trạng thái Git tại thời điểm chạy đánh giá:**

| | |
|---|---|
| Commit code chạy tracking/chấm điểm | **`aa45f6c`** |
| Nhánh trên remote (`origin`) tại thời điểm đó | **đã đồng bộ với `aa45f6c`** |
| Merge vào `main` | **CHƯA** — `main` vẫn ở `568ba2a` |

> **Ghi chép lịch sử (không phải trạng thái hiện tại).** Mục 15 được viết khi
> commit sau nghiệm thu còn nằm ở local. Commit đó sau này được amend hai lần
> (sửa thuật ngữ tương hợp/đồng dạng trong báo cáo rồi trong code) nên **các SHA
> nêu trong mục 15 là SHA lịch sử, đã bị thay thế** — chỉ dùng để đọc hiểu tiến
> trình, **không** dùng làm trạng thái hiện tại. SHA hiện hành luôn là bảng trên.

Không ghi đè bất kỳ kết quả thí nghiệm cũ nào — kết quả mới nằm trong
`results/eval_fixed/`, `results/runA/`, `results/runB/`.

**Môi trường:** Python 3.11.9 · numpy 2.4.6 · scipy 1.17.1 · pandas 2.3.3 ·
ultralytics 8.4.141 · torch 2.6.0+cu124 · GPU GTX 1650 · conda env `khoaluan-mot`

**Cấu hình:** `yolov8n.pt`, mọi ngưỡng tracker giữ như baseline. Không đổi
dataset, detector, thuật toán, không tinh chỉnh tham số để tăng điểm.

| Điều kiện | Ngưỡng detector |
|---|---|
| Baseline (đã khoá) và **Lượt A** | **`conf = 0.25`** |
| **Lượt B** | **`conf = 0.10`** |

Lượt B đổi **đúng một biến** so với lượt A: ngưỡng `conf` của detector. Mọi thứ
khác — weights, `track_high_thresh`, `track_low_thresh`, `new_track_thresh`,
`track_buffer`, `match_thresh`, `fuse_score` — giữ y hệt.

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
>
> **Cập nhật:** Lượt A và Lượt B (mục 13, 14) đã chạy xong đầy đủ trên 60 video
> train. Bản sửa lỗi 1–2 không đổi kết luận so sánh 3 model. Lượt B (conf=0,10)
> cho HOTA/AssA cao hơn và IDSW thấp hơn ~12% ở cả 3 model so với lượt A — bằng
> chứng ban đầu ủng hộ nới ngưỡng detector, nhưng CHƯA đủ để đổi baseline
> (xem 14.4 về giới hạn).
>
> **Sau nghiệm thu độc lập (PASS WITH LIMITATIONS):** đã sửa diễn giải định luật
> Sylvester, sửa số liệu thời gian sai ở 14.5, thêm test vòng đời tracker, và hạ
> mức các khẳng định nhân quả — chi tiết ở **mục 15**. Tổng test: 56 → **77**.

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
phương sai biên duyên bằng **phép biến đổi tương hợp (congruence)**:

```
P' = S P Sᵀ ,  S = diag(1, …, s, …, 1) ,  s = σ'/σ
```

tức nhân **cả hàng và cột** với cùng hệ số. **Định luật quán tính Sylvester** bảo
đảm phép biến đổi tương hợp với `S` khả nghịch (ở đây `s > 0`) **bảo toàn quán
tính** của ma trận — tức giữ nguyên **số lượng** trị riêng dương, âm và bằng 0.
Do đó `P'` bán xác định dương khi và chỉ khi `P` bán xác định dương. Mọi hệ số
tương quan cũng được giữ nguyên.

> **Tương hợp (congruence) KHÁC đồng dạng (similarity).** Đồng dạng là
> `P' = S P S⁻¹` — bảo toàn *toàn bộ phổ* trị riêng. Tương hợp là `P' = S P Sᵀ`
> — nói chung **không** bảo toàn trị riêng, chỉ bảo toàn quán tính. Định luật
> quán tính Sylvester phát biểu cho **tương hợp**. Ở đây `S` là ma trận đường
> chéo dương nên `Sᵀ = S ≠ S⁻¹` (trừ khi `s = 1`), tức đây đúng là tương hợp
> chứ không phải đồng dạng.

> **Phạm vi của định luật — nói cho đúng.** Định luật quán tính Sylvester **chỉ**
> bảo toàn *số lượng* trị riêng theo dấu. Nó **không** bảo toàn *giá trị* của
> từng trị riêng, và cũng không thiết lập một phép ghép cặp nào giữa trị riêng
> trước và sau phép biến đổi. Phép tương hợp nói chung **làm thay đổi** giá trị
> các trị riêng.

Đây là **sự thật toán học**, không phải cắt trị riêng âm hay thêm jitter để che
lỗi gốc. Test `1b` kiểm tra trực tiếp tính chất bảo toàn quán tính này trên 20 ma
trận ngẫu nhiên.

**Sau khi sửa:** 0/8 trường hợp hỏng — không còn trị riêng âm nào, đúng như định
luật bảo đảm. Trong bộ ca thử này, trị riêng nhỏ nhất quan sát được gần như không
đổi (2,700·10⁻⁹ → 2,700·10⁻⁹); đó là **kết quả thực nghiệm của riêng các ca thử
cụ thể** (ma trận hiệp phương sai gần suy biến, hệ số `s` áp lên đúng hai chiều
`V`/`THETA`), **không phải hệ quả tổng quát** của định luật.

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

> **Mục này viết khi lượt A và B chưa chạy. Giữ lại để thấy tiến trình; trạng
> thái cập nhật ở cột bên phải.** Danh sách còn thiếu *thật sự* ở mục **14.6**.

| Việc | Trạng thái hiện tại |
|---|---|
| Ảnh hưởng của bản sửa bộ lọc lên HOTA/IDF1/IDSW | ✅ **Đã đo — mục 13.2.** ΔHOTA ≤ 5·10⁻⁴ |
| `conf=0.1` có tốt hơn không | ✅ **Đã chạy lượt B — mục 14.** Có bằng chứng ban đầu, chưa dứt khoát |
| Lỗi 1–2 ảnh hưởng bao nhiêu tới các kết luận đã báo cáo | ✅ **Đã so — mục 13.2.** Không đảo ngược kết luận nào |
| Ảnh hưởng lên tập test 40 video | ⬜ Chưa chạy |
| Detector fine-tune (Giai đoạn C) | ⬜ Chưa chạy lại với bản sửa |

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

---

## 13. CẬP NHẬT — Lượt A đã chạy xong (60/60 video × 3 model)

**Đã chạy đầy đủ**, không phải smoke test. Toàn bộ 3 model, 60 video train,
`conf=0.25`, code đã sửa (lỗi 1–2), ghi vào `runA-{cv,ekf_ctrv,ukf_ctrv}`.

### 13.1 Kiểm tra không có file rỗng/thiếu

```
runA-cv:       0 file rỗng, 60/60 video, 49 MB
runA-ekf_ctrv: 0 file rỗng, 60/60 video, 49 MB
runA-ukf_ctrv: 0 file rỗng, 60/60 video, 49 MB
```

### 13.2 HOTA/AssA/IDSW: trước và sau khi sửa lỗi bộ lọc

| Model | HOTA cũ | HOTA mới (runA) | ΔHOTA | ΔAssA | ΔIDSW |
|---|---|---|---|---|---|
| KF + CV | 0,610429 | 0,610429 | **+0,0000** | +0,0000 | **+0** |
| EKF + CTRV | 0,610102 | 0,610045 | −0,0001 | −0,0002 | −1 |
| UKF + CTRV | 0,609505 | 0,608956 | −0,0005 | −0,0012 | +2 |

**KF + CV giống hệt tuyệt đối** — đúng như kỳ vọng, vì bản sửa lỗi 1–2 chỉ chạm
`ekf_ctrv.py`/`tracker_ctrv.py`, không đụng `KalmanFilterXYAH` gốc của
ultralytics. EKF/UKF lệch ở bậc 10⁻⁴, nằm trong nhiễu đo (so với chênh lệch giữa
các model vốn đã ~5·10⁻⁴). **Bản sửa lỗi 1–2 không đảo ngược bất kỳ kết luận
tổng thể nào** đã có trong `main` — không có bằng chứng EKF/UKF vượt KF trước
và sau khi sửa.

Lý do biên độ nhỏ: lỗi 1–2 chỉ lộ rõ ở các ca hiếm (track sinh ra ngay lúc bị
che nên `initiate_from_motion` phải nhảy nhiều frame, hoặc xe đứng yên rồi mới
chạy) — không phải hành vi phổ biến trên 598.281 bbox của 60 video.

### 13.3 Chấm bằng phép đánh giá đã sửa (occlusion_eval.py) trên runA

| Tracker | `id_match` | `id_continuous` |
|---|---|---|
| `runA-cv` | 48,26 % | 30,21 % |
| `runA-ekf_ctrv` | 47,46 % | 30,01 % |
| `runA-ukf_ctrv` | 47,56 % | 30,01 % |

Bootstrap theo video, chênh lệch so với `runA-cv`:

| Tracker | Chênh `id_match` | KTC 95 % |
|---|---|---|
| `runA-ekf_ctrv` | −0,80 % | [−1,83 %, +0,19 %] |
| `runA-ukf_ctrv` | −0,70 % | [−1,62 %, +0,10 %] |

**Trùng khớp gần như tuyệt đối** với bảng chấm lại tracking cũ ở mục 6
(48,26 % / 47,46 % / 47,56 % — giống hệt). Củng cố thêm kết luận: bản sửa lỗi 1–2
không đổi bức tranh tổng thể.

### 13.4 Lệnh đã chạy

```bash
for MM in cv ekf_ctrv ukf_ctrv; do
  python src/baseline_track.py --all-train --split-name DETRAC-all \
    --motion-model $MM --conf 0.25 --tracker-name "runA-$MM" --skip-existing
done
for MM in cv ekf_ctrv ukf_ctrv; do
  python src/run_trackeval.py --split-name DETRAC-all --tracker "runA-$MM"
done
python src/occlusion_eval.py --split-name DETRAC-all \
  --trackers runA-cv runA-ekf_ctrv runA-ukf_ctrv --out-dir results/runA
```

Kết quả: `results/trackeval/DETRAC-all/runA-*/`, `results/runA/`.
Chưa ghi đè `results/comparison_DETRAC-all.csv` hay bất kỳ file cũ nào.

### 13.5 Còn thiếu *(tại thời điểm viết mục 13 — nay đã cập nhật)*

- ~~**Lượt B (`conf=0.1`) chưa chạy**~~ → ✅ **đã chạy xong, xem mục 14.**
- Chưa chạy lại trên 40 video test.
- Chưa chạy lại detector fine-tune (Giai đoạn C).

---

## 14. CẬP NHẬT — Lượt B đã chạy xong (60/60 video × 3 model, `conf=0.10`)

**Đã chạy đầy đủ**, cùng code đã sửa, chỉ đổi `conf: 0.25 → 0.10`. Mọi ngưỡng
tracker khác giữ y hệt lượt A. Kiểm tra: 0 file rỗng, 60/60 video mỗi model.

### 14.1 HOTA/AssA/IDSW: Lượt A (conf=0,25) vs Lượt B (conf=0,10)

| Model | Lượt | HOTA | AssA | IDSW | FP | FN |
|---|---|---|---|---|---|---|
| CV | A | 0,610429 | 0,654065 | 2.251 | 53.332 | 146.919 |
| CV | **B** | **0,614274** | **0,659982** | **1.982** | 70.469 | 133.530 |
| EKF+CTRV | A | 0,610045 | 0,653398 | 2.289 | 53.246 | 147.086 |
| EKF+CTRV | **B** | **0,613842** | **0,659537** | **1.988** | 70.739 | 133.684 |
| UKF+CTRV | A | 0,608956 | 0,651257 | 2.304 | 53.338 | 147.122 |
| UKF+CTRV | **B** | **0,613521** | **0,658919** | **2.007** | 70.808 | 133.624 |

**Ở cả 3 model, lượt B cho HOTA cao hơn lượt A** (+0,0038 đến +0,0046), AssA cao
hơn, và **IDSW thấp hơn rõ** (giảm ~270–320 lần chuyển ID, tức khoảng −12 %).
Đổi lại: FP tăng mạnh (+17.000), FN giảm mạnh (−13.400) — đúng dự đoán khi hạ
ngưỡng detector: nhiều detection hơn thì bắt được nhiều xe hơn (giảm FN) nhưng
cũng lẫn nhiều nhiễu hơn (tăng FP). MOTA vì vậy **giảm nhẹ** (0,6615→0,6557 ở
CV) dù HOTA tăng — hai chỉ số này cân đối FP/FN khác nhau.

### 14.2 Bằng chứng cơ chế: số ID track duy nhất giảm

Không cần chạy lại đo vòng association — số liệu này đã có sẵn trong file tóm
tắt (`baseline_track_summary_*`, cột `n_tracks` = số ID track **duy nhất**
xuất hiện trên 60 video, độc lập với `occlusion_eval.py`):

| Model | Số ID (A, conf=0,25) | Số ID (B, conf=0,10) | Chênh | Tổng bbox (A) | Tổng bbox (B) |
|---|---|---|---|---|---|
| CV | 9.819 | **9.549** | **−270** | 504.694 | 535.220 |
| EKF+CTRV | 9.873 | **9.597** | **−276** | 504.441 | 535.336 |
| UKF+CTRV | 9.870 | **9.585** | **−285** | 504.497 | 535.465 |

**Nhất quán ở cả 3 model:** conf=0,10 tạo **ít ID track hơn** dù xử lý **nhiều
bbox hơn ~30.500**. Kết quả này **phù hợp với giả thuyết** rằng vòng association
thứ hai (mục 7) giúp ByteTrack tái sử dụng track cũ thay vì sinh ID mới: có
detection điểm thấp để ghép lại với track đang chờ, thay vì để track đó chết rồi
một detection khác buộc phải sinh ID mới.

> **Không coi đây là bằng chứng nhân quả trực tiếp.** Tổng số ID là một chỉ số
> **gộp** ở đầu ra; nó tương thích với giả thuyết trên nhưng cũng tương thích với
> các cơ chế khác (ví dụ detection dày hơn làm track ít bị đứt ngay từ vòng một,
> hoặc `new_track_thresh` lọc khác đi). Phép đo **trực tiếp** vòng association
> hiện mới chỉ là **smoke test 2 video × 100 frame** (mục 7) — quá nhỏ để khái
> quát. Muốn khẳng định trên cả 60 video thì phải log `n_low_dets`,
> `n_cand_tracks` và `n_matched` **xuyên suốt toàn bộ lượt chạy**;
> `src/assoc_stats.py` đã có sẵn cơ chế đếm ba con số đó, chỉ cần bật khi chạy.

> **Lưu ý về con số ở mục 7.** Phép đo bằng `probe_conf_bands.py` (mẫu 3 video ×
> 50 frame, đo trực tiếp phân bố điểm của detector) cho thấy dải `(0,10; 0,25)`
> bằng 89,6 % so với dải `≥0,25`. Con số đó là phân bố điểm tin cậy **thô** của
> detector trên một mẫu nhỏ, khác với tổng bbox tracker xuất ra trên cả 60 video
> (tăng khiêm tốn hơn, ~6 %) — vì tracker output đã qua NMS, lọc lớp xe, và phụ
> thuộc động lực theo dõi theo thời gian chứ không phải one-shot theo frame. Hai
> con số đo hai thứ khác nhau, không mâu thuẫn nhau, nhưng không nên gộp lẫn.

### 14.3 Chấm bằng phép đánh giá đã sửa (occlusion_eval.py) trên runB

| Tracker | `id_match` | `id_continuous` |
|---|---|---|
| `runB-cv` | 51,15 % | **37,79 %** |
| `runB-ekf_ctrv` | 50,75 % | 37,79 % |
| `runB-ukf_ctrv` | 50,55 % | 37,79 % |

So với lượt A: `id_match` 48,26 % → **51,15 %** (CV), `id_continuous`
30,21 % → **37,79 %** — tăng rõ rệt và nhất quán ở cả 3 model, đồng hướng với
giảm ID phân mảnh ở mục 14.2.

Bootstrap theo video, chênh lệch so với `runB-cv` (chưa đổi kết luận so sánh
giữa 3 motion model — vẫn không phân biệt được):

| Tracker | Chênh `id_match` | KTC 95 % |
|---|---|---|
| `runB-ekf_ctrv` | −0,40 % | [−1,49 %, +0,71 %] |
| `runB-ukf_ctrv` | −0,60 % | [−1,76 %, +0,52 %] |

### 14.4 Kết luận về `conf=0,1` — có bằng chứng nhưng chưa dứt khoát

**Ủng hộ `conf=0,1`:** HOTA/AssA cao hơn, IDSW thấp hơn ~12 %, `id_continuous`
tăng từ 30,0 % lên 37,8 %, ít ID phân mảnh hơn — nhất quán ở cả 3 model.

**Chưa đủ để khẳng định chắc chắn:**
- FP tăng ~33 % — chưa đo ảnh hưởng tới các ứng dụng hạ nguồn nhạy với false
  positive (đếm xe, ước lượng mật độ).
- Chỉ chạy trên 60 video train, `yolov8n` gốc (chưa fine-tune) — chưa biết có
  giữ nguyên xu hướng trên tập test hay với detector khác không.
- MOTA giảm nhẹ dù HOTA tăng — hai độ đo phản ánh trọng số FP/FN khác nhau,
  cần biết ứng dụng ưu tiên độ đo nào.
- **Đây vẫn không phải quyết định thay đổi baseline** — chỉ là bằng chứng ban
  đầu để cân nhắc, đúng phạm vi lượt B đã đề ra.

### 14.5 Lệnh đã chạy

```bash
for MM in cv ekf_ctrv ukf_ctrv; do
  python src/baseline_track.py --all-train --split-name DETRAC-all \
    --motion-model $MM --conf 0.10 --tracker-name "runB-$MM" --skip-existing
done
for MM in cv ekf_ctrv ukf_ctrv; do
  python src/run_trackeval.py --split-name DETRAC-all --tracker "runB-$MM"
done
python src/occlusion_eval.py --split-name DETRAC-all \
  --trackers runB-cv runB-ekf_ctrv runB-ukf_ctrv --out-dir results/runB
```

Kết quả: `results/trackeval/DETRAC-all/runB-*/`, `results/runB/`.

**Thời gian chạy — tính lại bằng tổng cột `seconds` của 6 file
`data/interim/baseline_track_summary_DETRAC-all_run{A,B}-*.csv`:**

| Model | Lượt A (`conf=0,25`) | Lượt B (`conf=0,10`) | Chênh |
|---|---|---|---|
| CV | 2014,5 s | 1780,5 s | B ít hơn 234,0 s |
| EKF + CTRV | 2124,6 s | 1811,4 s | B ít hơn 313,2 s |
| UKF + CTRV | 2296,7 s | 2211,7 s | B ít hơn 85,0 s |

> **Không kết luận `conf=0,10` làm chương trình chạy nhanh hơn.** Hai lượt chạy
> ở hai thời điểm khác nhau và **không kiểm soát điều kiện chạy** (tiến trình
> nền, nhiệt độ GPU, trạng thái bộ nhớ đệm ổ đĩa đều không được ghi nhận). Về
> mặt tính toán, `conf` thấp hơn tạo **nhiều** detection hơn nên đáng lẽ phải
> **tốn** thời gian hơn; kết quả đo lại ngược chiều. **Chưa xác định được nguyên
> nhân của chênh lệch này** — có thể do điều kiện chạy khác nhau, có thể do yếu
> tố khác chưa được đo. Vì vậy các số này chỉ có **giá trị mô tả**, ghi lại cho
> đầy đủ, **chưa phải benchmark có kiểm soát**. Muốn so tốc độ thật thì phải
> chạy xen kẽ nhiều lần, cố định tần số GPU và đo trên máy nhàn rỗi.
>
> *(Bản báo cáo trước ghi lượt B "gấp ~2,5× lượt A" — sai cả về hướng lẫn độ
> lớn. Đã sửa theo đúng số trong CSV; **không** sửa CSV cho khớp báo cáo.)*

### 14.6 Còn thiếu — danh sách CUỐI CÙNG

Đây là danh sách đầy đủ những việc **thực sự chưa làm** (các mục 10 và 13.5 ở
trên là ảnh chụp trạng thái cũ, giữ lại cho có lịch sử):

| # | Việc chưa làm | Cần gì |
|---|---|---|
| 1 | Chạy trên **tập test 40 video** | ~3 giờ GPU |
| 2 | Chạy lại với **detector fine-tune** (Giai đoạn C) | ~3 giờ GPU |
| 3 | Thử các mức **`conf` trung gian** (0,15; 0,20) để biết xu hướng có đơn điệu không | ~3 giờ GPU mỗi mức |
| 4 | **Benchmark thời gian trong điều kiện kiểm soát** — chạy xen kẽ nhiều lần, cố định tần số GPU, máy nhàn rỗi (số ở 14.5 chỉ mô tả) | Máy nhàn rỗi |
| 5 | Đo **ảnh hưởng FP tăng** tới ứng dụng hạ nguồn (đếm xe, ước lượng mật độ) | Định nghĩa bài toán hạ nguồn |
| 6 | Log `n_low_dets` / `n_cand_tracks` / `n_matched` **trên cả 60 video** để khẳng định cơ chế vòng association (nay mới có smoke test 2 video) | ~3 giờ GPU |

---

## 15. Sửa sau nghiệm thu

Nghiệm thu độc lập cho kết quả **PASS WITH LIMITATIONS**. Mục này ghi lại các
điểm đã sửa sau đó. **Không** chạy lại tracking, **không** sửa file kết quả A/B.

### 15.1 Diễn giải định luật quán tính Sylvester — đã sửa

**Sai ở đâu.** Báo cáo cũ viết phép biến đổi `P' = S P Sᵀ` làm trị riêng sau
khởi tạo *"bằng đúng"* trị riêng trước khởi tạo, và gọi đó là *"đúng như định lý
dự báo"*. Sai: định luật quán tính Sylvester **chỉ** bảo toàn **số lượng** trị
riêng dương, âm và bằng 0 — **không** bảo toàn *giá trị* từng trị riêng, và cũng
không thiết lập phép ghép cặp nào giữa trị riêng trước/sau.

**Đã sửa.**

| Chỗ | Sửa thành |
|---|---|
| `FIX_REPORT.md` mục 2 | Phát biểu đúng về **bảo toàn quán tính**; thêm hộp "Phạm vi của định luật"; trị riêng nhỏ nhất gần như không đổi nay ghi là **quan sát thực nghiệm của riêng ca thử**, không phải hệ quả tổng quát |
| `src/ekf_ctrv.py` — docstring `set_marginal_variance()` | Như trên, thêm đoạn "PHAM VI CUA DINH LUAT - noi cho dung" |
| `src/test_init_regression.py` | Đổi tên `test_congruence_preserves_eigen_sign` → `test_congruence_preserves_inertia`; thêm hàm `_inertia()` đếm tường minh `(n₊, n₋, n₀)` |

**Cách sửa covariance giữ nguyên** — chỉ lời giải thích sai, bản thân phép biến
đổi tương hợp vẫn đúng và vẫn là cách xử lý có cơ sở.

Thêm **test 1c** khẳng định chiều ngược lại — rằng phép tương hợp **có** làm
thay đổi *giá trị* trị riêng — để chặn việc hiểu nhầm trở lại. Số phép kiểm tra
trong file này: 16 → **17**.

### 15.2 Số liệu thời gian lượt A/B — đã sửa

Tính lại tổng cột `seconds` từ 6 file
`data/interim/baseline_track_summary_DETRAC-all_run{A,B}-*.csv`:

| Model | Lượt A (`conf=0,25`) | Lượt B (`conf=0,10`) |
|---|---|---|
| CV | 2014,5 s | 1780,5 s |
| EKF + CTRV | 2124,6 s | 1811,4 s |
| UKF + CTRV | 2296,7 s | 2211,7 s |

Câu cũ ở mục 14.5 — *"gấp ~2,5× lượt A do nhiều detection hơn"* — **sai cả về
hướng lẫn độ lớn**: lượt B thực tế tốn **ít** thời gian hơn ở cả 3 model. Đã
thay bằng bảng trên, kèm cảnh báo rằng hai lượt chạy **không kiểm soát tải máy**
nên số liệu chỉ có **giá trị mô tả**, chưa phải benchmark. **Không sửa CSV** —
báo cáo phải khớp CSV, không phải ngược lại.

### 15.3 Test vòng đời tracker — file mới

`src/test_tracker_lifecycle.py`, **20 phép kiểm tra**, chạy cho **cả**
`CTRVSTrack` (EKF) và `UKFSTrack` (UKF). *(Ghi chú: lớp EKF tên là `CTRVSTrack`,
không phải `EKFSTrack`.)*

Đi **thật** qua đường chạy của tracker, không gọi tắt bộ lọc:

```
activate(frame 1) → update(frame 2, dịch 0,1 px) → mark_lost()
    → predict() × gap → re_activate(frame 2+gap, dịch 3·gap, 4·gap)
```

| # | Kiểm tra |
|---|---|
| 1 | Sau `activate`: `_motion_initialized is False` |
| 2 | Dịch chuyển quá nhỏ (0,1 px < ngưỡng 0,5) → **vẫn** `False` |
| 3 | `_last_obs_xy` bám theo vị trí **quan sát**, không phải vị trí đã predict |
| 4 | Sau `re_activate`: `_motion_initialized is True`, `state = Tracked`, `frame_id` đúng |
| 5 | **Seed chính xác**: `n_frames == gap`, `ref_pos ==` vị trí quan sát, `v == 5,0` |
| 6 | Vận tốc sau update **bất biến theo `gap`** |
| 7 | Học đúng hướng có thành phần **dọc** (θ ≈ 53,13°) |
| 8 | `mean`/`P` hữu hạn, `P` đối xứng, không trị riêng âm — ở **mọi** mốc |
| 9 | Đứng yên suốt → **không bịa ra hướng** |

**Điểm 5 dùng spy.** Bọc `try_initiate_from_motion` của *chính instance* bộ lọc
bằng wrapper ghi lại rồi uỷ quyền cho hàm thật. Tracker vẫn chạy y nguyên qua
`re_activate()`, nhưng quan sát được giá trị **ngay tại thời điểm seed** — trước
khi bước KF update của `super().re_activate()` làm nhiễu. Nhờ vậy khẳng định
được đẳng thức chính xác thay vì phải nới lỏng ngưỡng.

**Một quan sát ghi lại, không phải lỗi.** Sau `re_activate`, `mean[V]` ≈ 10,8
chứ không phải 5,0 (giá trị seed). Nguyên nhân: `super().re_activate()` chạy một
bước KF update với innovation rất lớn (vị trí dự đoán sau các bước predict mù
cách quan sát ~50 px), kéo mạnh vị trí và qua hiệp phương sai chéo đẩy `v` lên
khoảng 2×. Đây là **hành vi đúng của KF khi innovation lớn**. Vì vậy điểm 6
**không** ràng buộc giá trị tuyệt đối, chỉ kiểm **bất biến theo `gap`** (chênh
0,11 px/frame giữa `gap=10` và `gap=20`) và **cách xa giá trị sai**.

**Đã kiểm test không rỗng (mutation test).** Tạm tái lập lỗi cũ
(`n_frames=1.0` thay vì khoảng cách thật) → test **FAIL** đúng chỗ với
`v seed = 50,000` thay vì 5,0. Khôi phục → 20/20 PASS trở lại.

**Dung sai theo float32.** `STrack` của ultralytics lưu `_tlwh` ở float32 nên
`100.1` được lưu thành `100.0999984741211`. Ngưỡng dùng `atol=1e-4` cho vị trí
và `1e-5` cho vận tốc — đúng độ chính xác của kiểu dữ liệu, và vẫn chặt hơn
nhiều bậc so với lỗi cần bắt (sai lệch `50` vs `5`, không phải ở chữ số thứ 7).

### 15.4 Hạ mức khẳng định nhân quả — mục 14.2

Câu *"Ít ID hơn với nhiều detection hơn **nghĩa là** ByteTrack đang tái sử dụng
track cũ…"* → *"Kết quả này **phù hợp với giả thuyết**…"*, kèm hộp ghi rõ:

- Tổng số ID là **chỉ số gộp ở đầu ra**, tương thích với giả thuyết trên nhưng
  cũng tương thích với cơ chế khác — **không phải bằng chứng nhân quả trực tiếp**.
- Phép đo **trực tiếp** vòng association hiện mới là **smoke test 2 video ×
  100 frame**, quá nhỏ để khái quát.
- Muốn khẳng định trên cả 60 video phải log `n_low_dets`, `n_cand_tracks`,
  `n_matched` xuyên suốt lượt chạy — `src/assoc_stats.py` đã có sẵn cơ chế.

### 15.5 Lệnh và kết quả test thực tế

`pytest` **chưa cài** trong env `khoaluan-mot`, nên chạy trực tiếp bằng `python`.
Các file test vẫn viết tương thích pytest (hàm `test_*`, dùng `assert`).

```bash
conda activate khoaluan-mot && cd /d/LABOTARY/bytetrack
python src/test_ekf_ctrv.py           # 14/14 PASS
python src/test_ukf_ctrv.py           # 12/12 PASS
python src/test_init_regression.py    # 17/17 PASS  (16 -> 17, thêm test 1c)
python src/test_occlusion_eval.py     # 14/14 PASS
python src/test_tracker_lifecycle.py  # 20/20 PASS  (MỚI)
```

**Tổng: 77/77 phép kiểm tra PASS** (trước khi sửa: 56/56).

Xác nhận không làm hỏng kết quả cũ. `git status --short` **ngay trước khi
commit `b20ca98`** chỉ có 4 file:

```
 M FIX_REPORT.md
 M src/ekf_ctrv.py
 M src/test_init_regression.py
?? src/test_tracker_lifecycle.py
```

**Sau khi commit `b20ca98`, working tree sạch** (`git status --short` không ra
dòng nào).

**Không** file nào trong `results/runA/`, `results/runB/`,
`results/trackeval/DETRAC-all/run*/` hay `data/interim/baseline_track_summary_*`
bị thay đổi. Mọi số HOTA / AssA / IDSW / FP / FN / `id_match` / `id_continuous`
giữ nguyên — mục 15 chỉ sửa lời văn, số liệu thời gian, và thêm test.


---

## 16. Bàn giao đánh giá lại test 40 video — 10/09/2026

Đã hoàn tất 240/240 file tracking (6 cấu hình × 40 video), không thiếu/rỗng. Lượt B kết thúc trước khi tiếp quản lúc 04:32:37 +07; không chạy lại tracking. Chấm B bằng TrackEval và phép đánh giá che khuất đã sửa, giữ nguyên code chạy `aa45f6c`, weights và cấu hình A=0,25/B=0,10. Báo cáo đầy đủ, 40 tên video, môi trường, đường dẫn/lệnh và giới hạn: [TEST_EVALUATION_REPORT.md](TEST_EVALUATION_REPORT.md).

### 16.1 Kết quả từ CSV

HOTA/IDF1/AssA/DetA/LocA/MOTA thang 0–100, gộp COMBINED_SEQ:

| Cấu hình | HOTA | IDF1 | AssA | DetA | LocA | MOTA | IDSW | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| testA-cv | 58.484 | 70.225 | 64.174 | 53.801 | 84.700 | 59.475 | 2,500 | 72,646 | 198,712 |
| testA-ekf_ctrv | 58.600 | 70.469 | 64.427 | 53.795 | 84.700 | 59.501 | 2,532 | 72,314 | 198,835 |
| testA-ukf_ctrv | 58.490 | 70.292 | 64.206 | 53.781 | 84.691 | 59.481 | 2,549 | 72,421 | 198,844 |
| testB-cv | 58.217 | 69.407 | 63.581 | 53.817 | 84.327 | 58.188 | 2,082 | 99,903 | 180,571 |
| testB-ekf_ctrv | 58.548 | 69.937 | 64.298 | 53.830 | 84.327 | 58.244 | 2,063 | 99,579 | 180,532 |
| testB-ukf_ctrv | 58.399 | 69.716 | 63.971 | 53.817 | 84.325 | 58.228 | 2,110 | 99,731 | 180,446 |

2.174 partial + 484 full; **484/484 full lồng trong partial**, gộp thành 2.174 sự kiện. Tất cả cấu hình dùng cùng **1.324 sự kiện đủ điều kiện trên 39 video**; loại 559 `gt_missing_before` + 291 `gt_missing_after` (850). Không đếm lặp full.

| Cấu hình | id_match = preserved | id_continuous | switched | lost_after | no_match_before |
| --- | --- | --- | --- | --- | --- |
| testA-cv | 638 (48.187%) | 393 (29.683%) | 351 (26.511%) | 78 (5.891%) | 257 (19.411%) |
| testA-ekf_ctrv | 648 (48.943%) | 393 (29.683%) | 343 (25.906%) | 77 (5.816%) | 256 (19.335%) |
| testA-ukf_ctrv | 645 (48.716%) | 394 (29.758%) | 346 (26.133%) | 78 (5.891%) | 255 (19.260%) |
| testB-cv | 647 (48.867%) | 485 (36.631%) | 350 (26.435%) | 80 (6.042%) | 247 (18.656%) |
| testB-ekf_ctrv | 652 (49.245%) | 490 (37.009%) | 345 (26.057%) | 80 (6.042%) | 247 (18.656%) |
| testB-ukf_ctrv | 659 (49.773%) | 490 (37.009%) | 338 (25.529%) | 80 (6.042%) | 247 (18.656%) |

### 16.2 So sánh có KTC

Hiệu và KTC 95% theo điểm phần trăm; 5.000 bootstrap, seed=0, cụm video, giữ ghép cặp:

| So sánh | id_match | id_continuous |
| --- | --- | --- |
| testA-ekf_ctrv minus testA-cv | +0.755 [-0.584; +2.035] | +0.000 [-0.431; +0.421] |
| testA-ukf_ctrv minus testA-cv | +0.529 [-0.519; +1.309] | +0.076 [-0.309; +0.480] |
| testB-ekf_ctrv minus testB-cv | +0.378 [-0.498; +1.316] | +0.378 [-0.079; +0.812] |
| testB-ukf_ctrv minus testB-cv | +0.906 [-0.548; +2.246] | +0.378 [-0.080; +0.872] |
| testB-cv minus testA-cv | +0.680 [-0.331; +1.897] | +6.949 [+5.202; +8.703] |
| testB-ekf_ctrv minus testA-ekf_ctrv | +0.302 [-1.447; +2.394] | +7.326 [+5.477; +9.285] |
| testB-ukf_ctrv minus testA-ukf_ctrv | +1.057 [-0.463; +2.901] | +7.251 [+5.473; +9.150] |

KTC `id_match` giữa motion model đều chứa 0: **chưa đủ bằng chứng về khác biệt**, không chứng minh tương đương hay EKF/UKF thắng về `id_match`. Riêng lượt B, hiệu EKF−CV của HOTA/IDF1/AssA có KTC danh nghĩa dương; kết quả chưa hiệu chỉnh cho nhiều so sánh và không hỗ trợ tuyên bố EKF thắng toàn diện. Train A cho EKF−CV/UKF−CV −0,798/−0,698 điểm phần trăm, test A đổi dấu +0,755/+0,529; KTC `id_match` đều chứa 0. HOTA/AssA/IDF1 B−A tăng trên train nhưng giảm theo điểm ước lượng trên test; IDSW/FN giảm, FP tăng và `id_continuous` tăng. KTC pooled HOTA/IDF1/AssA/DetA/LocA/MOTA và số đếm nằm trong mục 5 của báo cáo test và [paired_contrasts_bootstrap.csv](results/test_eval/summary/paired_contrasts_bootstrap.csv).

Đây là xác nhận lại trên **test đã từng được sử dụng**, không gọi là held-out hoàn toàn chưa xem. Tổng ID không chứng minh cơ chế nhân quả; thời gian chỉ mô tả tải máy không kiểm soát. Không đổi baseline hoặc tinh chỉnh dựa trên kết quả test.

### 16.3 Kiểm tra, bảo toàn và trạng thái bàn giao

- 5 file test: **77/77 PASS** (14 + 12 + 17 + 14 + 20); log [tests.log](results/test_eval/provenance/tests.log). `git diff --check` và diff staged được kiểm tra trước commit.
- SHA-256 2.258 file hiện hữu được đối chiếu trước/sau: [preservation_check.json](results/test_eval/provenance/preservation_check.json). Train A/B, ba test cũ, test A/B tracking và manifest không bị ghi đè; nguyên văn chỉnh sửa `FIX_REPORT.md` lúc tiếp quản được giữ lại.
- Lệnh tracking cũ lọc stdout/stderr qua grep, không có pipefail; không thể chứng nhận tuyệt đối không có cảnh báo lịch sử bị lọc. Audit MOT/ảnh hiện tại không phát hiện lỗi. Log chấm B mới đầy đủ và exit code được kiểm tra.
- Commit kết quả đánh giá: **`b5bd086f8e4a625850ab35c977d1ed40776e810d`** (`Chay danh gia tap test 40 video`), parent `aa45f6c`, đã push lên `origin/fix/tracker-evaluation-correctness`; chưa merge hoặc force-push vào `main` (vẫn `568ba2a`).
- Mục test “chưa chạy” ở 10/13.5/14.6 đã hoàn thành. Các nhận xét chưa biết xu hướng test ở mục 14 là lịch sử, được thay bằng kết quả mục 16. Ngưỡng trung gian trong mục 14.6 không thuộc phạm vi được phép thử trên test hiện tại.
- Các việc ngoài phạm vi còn lại: detector fine-tune, dữ liệu độc lập, benchmark tốc độ có kiểm soát, ảnh hưởng FP lên ứng dụng, log association xuyên suốt; việc merge vào `main` chỉ thực hiện khi có yêu cầu riêng.
