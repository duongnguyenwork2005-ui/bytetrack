# R48C (M3) — Giao thức kiểm chứng PF, chốt trước khi xem kết quả

Viết ngày 2026-10-11, HEAD `eaf2e87`, nhánh `phase2-mixed-traffic`. File này được viết **trước** mọi lần chạy so sánh
bộ lọc của R48C (smoke `kc48_pf_smoke`, mô phỏng, pipeline dev). sha256 của file được chụp bằng
`src/kc48/ghi_sha256.py` trước khi chạy smoke và ghi trong `05_dung_dan_pf_va_smoke.md`. Sau khi chụp, nội dung mục 1–12
**không sửa**; lệch giao thức chỉ ghi ở **phụ lục** cuối file, kèm lý do và thời điểm.

Nhãn: [LT] lý thuyết, [QS] quan sát đã có (kèm nguồn), [GT] giả thuyết chưa kiểm, [QĐ] quyết định của giao thức.

## 1. Câu hỏi và phạm vi

- Q1 (cơ chế, mô phỏng): ở cấp bộ lọc, khi biết chắc liên kết đo–đối tượng, PF–CTRV hiện tại có sai số dự đoán / tỉ lệ
  thất bại khác EKF–CTRV không, ở nhóm chuyển động nào, mức mất đo và nhiễu nào, kể cả khi mô hình CTRV **sai**?
- Q2 (cơ chế, mô phỏng): hai kỹ thuật ở mục 6 có giảm các triệu chứng đã quan sát (ESS sụp sau mất đo — H4; nghèo hạt
  và trôi θ / ω khi đang bám — H5; `02_doi_chieu_code_pf.md` mục 4) mà không làm tệ nhóm khác không?
- Q3 (pipeline thật, dev): các khác biệt trên có còn thấy được trong ByteTrack trên dev UA-DETRAC (HOTA / IDF1 / IDSW,
  giữ ID theo lớp, kích thước, nhóm cua, nhóm xe nhỏ cạnh xe lớn) không?

Mô phỏng chỉ kiểm **cơ chế**, không thay benchmark thật. Kết quả dev12 không suy rộng ra tập khác (dev12 chỉ 12 video,
`03_audit_du_lieu.md` mục 10 ý 2). Không chạy pipeline trên UA-DETRAC test hay BDD val trong lượt này.

## 2. Đầu vào cố định [QĐ]

| Mục | Giá trị | Nguồn |
|:--|:--|:--|
| Cache detection | `yolov8n-coco_conf0.05_all` | như mốc 2 (`configs/phase2/experiments/expB_detrac_dev_fix1.yaml`) |
| Điều kiện | A: `conf_min 0.25`; `classes [2, 3, 5, 7]`; `max_det 300` | như mốc 2 |
| Thời gian | `fps 25`, dt = 1 bước (`src/time_units.py`), `elapsed_mode skip`, `track_buffer 30` (1,2 s) | yaml tracker |
| Tập dev | 12 video `results/phase2/manifests/detrac_dev.json` (split `DETRAC-all`, part `train`) | manifest |
| Evaluator | `src/run_trackeval.py` (TrackEval MotChallenge2DBox, HOTA + CLEAR + Identity, ngưỡng 0,5), chỉ dùng, không sửa | mã khóa |

**Xác thực cache** [QĐ]: `src/kc48/kiem_cache_dev.py` tính `run_fingerprint.cache_fingerprint` cho 12 video dev và so
với dấu vân tay đã commit trong `results/trackeval/DETRAC-all/p2-expB_detrac_dev_fix1-A-bt-ekf/eval_meta.json`
(`meta_sha256 870cb45f…`, `npz_sha256 9d596eb4…`, `weights_sha256 f59b3d83…`). Lệch ⇒ dừng, không chạy so sánh. Mỗi run
pipeline còn tự ghi dấu vân tay cache vào `eval_meta.json` (cơ chế sẵn có).

**Tập đánh giá cuối** (UA-DETRAC test, BDD val): không dùng để chọn tham số hay ngưỡng, không chạy pipeline.

**Hạn chế đã biết của GT dev**: 572 ảnh dev12 không có `<frame>` trong XML (`03` mục 6.2). Evaluator xử lý chúng thế nào
chưa kiểm (đề xuất riêng đang chờ người dùng). Mọi biến thể chấm trên **cùng** GT nên so sánh ghép cặp dùng chung sai
lệch này; mức ảnh hưởng lên hiệu số chưa biết [GT].

## 3. Bộ lọc và biến thể

| Mã | Mô tả | Cấu hình pipeline |
|:--|:--|:--|
| `ekf` | EKF–CTRV mặc định (tham chiếu chính) | `configs/bytetrack_ekf_ctrv.yaml` |
| `pf{N}` | PF–CTRV hiện tại: proposal = prior, resample hệ thống khi ESS < 0,5 N, roughening 0, ước lượng `mean` | `configs/phase2/trackers/bytetrack_pf_ctrv_n{100,300,500}.yaml` |
| `pf{N}-k1` | `pf{N}` + K1 (mục 6.1) | `configs/phase2/trackers/kc48/` (R48D) |
| `pf{N}-k2` | `pf{N}` + K2 (mục 6.2) | `configs/phase2/trackers/kc48/` (R48D) |
| `pf{N}-k1k2` | cả hai (chỉ sau khi đã có hai ablation một yếu tố) | `configs/phase2/trackers/kc48/` (R48D) |

Tham số CTRV dùng chung (σ_θ, σ_ω, prior, ε_ω, Q/R theo h) giữ mặc định `src/config.py` cho mọi biến thể. Mọi so sánh
biến thể PF chỉ đổi **một** yếu tố so với `pf{N}` cùng N; `k1k2` chỉ diễn giải sau hai ablation một yếu tố.

## 4. Cấp (a) — mô phỏng biết liên kết

### 4.1 Bộ sinh (`src/kc48/mo_phong_quy_dao.py`)

- Trạng thái trên mặt phẳng ảnh, 1 bước = 1 frame, T = 90 bước. Box GT (cx, cy, a = w/h, h), a = 1,6.
- Mặc định: cx0 = 480, cy0 = 270, v0 = 4 px/bước (**cùng tốc độ tuyệt đối** cho hai cỡ box, để cỡ box là một yếu tố
  thật; nếu v ∝ h thì bài toán gần bất biến tỉ lệ vì Q, R ∝ h), θ0 ~ U[−π, π), dấu quay s = ±1 ngẫu nhiên.
- Kịch bản (ω tính bằng rad/bước):

| Mã | Nhóm | Chuyển động GT | Đúng CTRV? |
|:--|:--|:--|:--|
| `thang` | thẳng | ω = 0 | đúng |
| `cua_deu` | cua đều | ω = 0,03·s suốt quỹ đạo | đúng |
| `doi_omega` | đổi tốc độ góc | ω = 0 (t < 30) → tăng tuyến tính tới 0,06·s (t = 40) → giữ tới t = 60 → giảm về 0 (t = 70) | từng đoạn; ω đổi |
| `cua_gat` | cua gắt | ω = 0,10·s với 30 ≤ t < 50, ngoài ra 0 (≈ 115° trong 0,8 s) | từng đoạn; ω nhảy bậc |
| `phoi_canh` | lệch mô hình | đi thẳng đều trên mặt đất, chiếu pinhole (f = 800 px, camera cao 6 m, xe cao 1,5 m), độ sâu Z(t) = Z0 (1 − 0,5 t / T): trên ảnh tốc độ và h tăng theo 1/Z | sai (gia tốc ảnh, h phi tuyến) |
| `gia_toc` | lệch mô hình | v(t) = 4 + 0,05 t, ω = 0,03·s | sai (v không hằng) |
| `doi_lan` | lệch mô hình | θ(t) = θ0 + 0,35 sin(2π t / 40) (ω biến thiên liên tục, \|ω\| ≤ 0,055) | sai (ω không hằng từng đoạn) |
| `nhieu_duoi_day` | lệch mô hình đo | chuyển động như `cua_deu`; nhiễu đo Student-t ν = 3 cùng độ lệch chuẩn + 5 % outlier dời ±0,5 h ở cx, cy | sai (likelihood Gauss) |

- Phép đo: z_t = (cx, cy, a, h)_GT + κ·σ ⊙ ε_t, σ = (h/20, h/20, 0,1, h/20) theo h GT tại t — đúng dạng R của mã
  (`src/ekf_ctrv.py:EKFTrackerCTRV._measurement_noise`, conf = None) khi κ = 1. ε_t ~ N(0, I) (trừ `nhieu_duoi_day`).
- Mất đo: L bước liên tiếp bắt đầu từ t_gap = 30 (`doi_omega`, `cua_gat`: mất đo đúng lúc bắt đầu cua) hoặc 40 (còn lại).
  Bước ghép lại t_r = t_gap + L.
- Yếu tố: L ∈ {0, 5, 10, 20} bước (< `track_buffer` 30); κ ∈ {0,5, 1, 2} (κ = 1 khớp R của mô hình, κ = 2 là R bị đánh
  giá thấp); h0 ∈ {24, 72} px (xe nhỏ / lớn). Lưới đầy đủ: 8 × 4 × 3 × 2 = 192 ô.
- Seed và ghép cặp: seed nhiễu d sinh θ0, s, ε, outlier bằng `SeedSequence([d, mã kịch bản])` — **không** phụ thuộc L, κ,
  h0 hay bộ lọc ⇒ mọi bộ lọc, mọi L / κ / h0 cùng d nhận **cùng** quỹ đạo và cùng nhiễu chuẩn hóa (số ngẫu nhiên chung).
  Seed PF p ∈ {0, 1, 2}: `pf.set_seed(p, video = "<kịch bản>|<d>")`, track_id = 1. d đánh giá: 0–29; d tinh chỉnh:
  1000–1009 (tách rời).

### 4.2 Khai thác bộ lọc (harness, R48D: `src/kc48/thi_nghiem_mo_phong.py`)

Gọi đúng API mã thật, cùng thứ tự như track trong ByteTrack:
1. t = 0: `initiate(z_0)`; PF: `sample_initial`, logw đều.
2. Mỗi bước t ≥ 1: nếu bước trước không có đo thì đặt v_h = 0 trước khi dự đoán (luật LOST của ByteTrack; PF dùng
   `PFTrackView.predict(..., lost=True)`). Dự đoán → ghi box dự đoán. Nếu có đo: lần đầu có hai quan sát thì
   `try_initiate_from_motion` (PF qua `PFTrackView`, reseed v, θ), rồi `update`.
3. Liên kết luôn đúng (không có association) ⇒ cô lập bộ lọc.

### 4.3 Chỉ số [QĐ]

- **Chính 1 — sai số dự đoán khi ghép lại**: e_r = ‖ĉ(t_r) − c_GT(t_r)‖ / h_GT(t_r), ĉ là tâm box dự đoán (sau predict,
  trước update) tại t_r. Với L = 0: trung bình sai số dự đoán 1 bước chuẩn hóa trên t ∈ [t_gap, t_gap + 20).
- **Chính 2 — thất bại cấp bộ lọc**: F = 1[IoU(box dự đoán, box GT) tại t_r < 0,2 / 0,9] (vòng ghép đầu của ByteTrack:
  1 − IoU·score ≤ 0,8 với score 0,9); trạng thái không hữu hạn tính là thất bại. Độ nhạy: ngưỡng IoU 0,5 (vòng ghép thứ hai).
- Phụ: RMSE vị trí lọc trên các bước có đo (t ≥ 10); sai số hướng |wrap(θ̂ − θ_GT)| và ω̂ − ω_GT tại t_gap − 1 (chỉ khi tốc
  độ GT ≥ 1 px/bước; với `phoi_canh` θ_GT, ω_GT là hướng / tốc độ đổi hướng của tâm box trên ảnh); NIS tại t_r
  (ν^T S^{-1} ν, S từ `project`) so với kỳ vọng 4 của χ²₄.
- Chẩn đoán PF: ESS/N **trước** resampling ở mọi update; tỉ lệ update có resample; số tổ tiên khác nhau sau resample
  (roughening = 0: `n_unique` theo cx của `enable_diag`; khi roughening > 0 cần chỉ số resample — R48D thêm hook chẩn đoán
  chỉ đọc và test chứng minh không đổi kết quả); độ phân tán vị trí / θ / v; innovation; thời điểm mất đo. ESS ngay sau
  reset trọng số **không** dùng làm bằng chứng đa dạng.
- Chẩn đoán H6 (không phải biến thể): tại t_r so độ dời của box `mean` với f(x̂) tính từ ước lượng trước khoảng mất đo.
- Tài nguyên: thời gian tường mỗi bước (predict + update) mỗi track theo biến thể và N, `time.perf_counter`, một tiến
  trình, cùng máy.

### 4.4 Phân tích định trước [QĐ]

- Đơn vị thống kê = seed nhiễu d (quỹ đạo độc lập). PF: lấy trung bình 3 seed PF trong mỗi d rồi ghép cặp với EKF theo d.
  Báo mean, SD theo d, và SD giữa seed PF trong cùng d (biến thiên Monte Carlo).
- CI 95 %: bootstrap phân vị theo d, B = 5000, seed bootstrap 0; hiệu tỉ lệ thất bại cũng bootstrap ghép cặp theo d.
- Đối chiếu chính (báo **mọi** ô, không chọn ô đẹp):
  - C1: `pf100` − `ekf`, e_r và F tại L ∈ {10, 20}, κ = 1, cả hai h0, từng kịch bản (nhóm đúng CTRV và nhóm lệch mô hình báo
    tách riêng).
  - C2: `pf100-k1` − `pf100` cùng các ô, thêm L = 0 (sai số 1 bước), ESS/N tại t_r, tỉ lệ resample khi đang bám, sai số
    hướng trước mất đo.
  - C3: `pf100-k2` − `pf100` như C2.
  - C4: đường tài nguyên e_r, F, thời gian theo N ∈ {100, 300, 500} cho `pf` và các biến thể K được giữ.
- Quy tắc diễn giải: gọi là **khác biệt nhất quán** khi CI loại trừ 0 ở ≥ 2/3 số ô của đối chiếu, cùng dấu, và không ô
  nào có CI loại trừ 0 ngược dấu; ngược lại ghi **chưa phân biệt / phụ thuộc kịch bản**. κ = 0,5 và κ = 2 là phân tích độ
  nhạy. Không kết luận "PF tốt hơn khi cua" từ mô phỏng đúng CTRV; nhóm lệch mô hình luôn báo cạnh nhóm đúng mô hình.

## 5. Cấp (b) — pipeline ByteTrack trên dev UA-DETRAC

### 5.1 Chạy

- Runner: `src/run_experiment.py` với cấu hình `configs/phase2/experiments/kc48_*.yaml`, khai báo như nhóm cũ. Tên run
  `p2-kc48_*`. Thư mục tracker của runner (`data/processed/trackers/`) nằm ngoài `write_paths` của R48C / R48D, nên chạy
  qua `src/kc48/chay_thi_nghiem_kc48.py`: gọi đúng `run_experiment.main` cho bước `track` và `eval`, chỉ chuyển gốc
  `config.PROCESSED_DIR` sang `results/phase2/<tên>/_processed/` (sao GT `gt.txt` + `seqinfo.ini`, kiểm sha256 trùng
  bản gốc, `.gitignore` `*`). Kết quả TrackEval vẫn ghi `results/trackeval/DETRAC-all/p2-kc48_*`. Bước `occlusion` /
  `aggregate` của runner chạy tiến trình con đọc đường cố định, nên R48D gọi các hàm tương ứng trong cùng tiến trình qua
  wrapper (hoặc ghi rõ phần chưa chạy).
- PF seed {0, 1, 2} cho mọi biến thể PF; EKF tất định.

### 5.2 Chỉ số và suy luận [QĐ]

- Chính: HOTA, IDF1, IDSW (COMBINED, TrackEval); kèm AssA, DetA, MOTA.
- Suy luận: bootstrap ghép cặp **theo video** (12 video, B = 5000), tính lại chỉ số COMBINED từ trường thô từng video
  (`src/boot_stats.py`, cách của mốc 2); với PF, mỗi mẫu bootstrap lấy trung bình 3 seed. Frame không phải mẫu độc lập.
  Báo thêm khoảng giữa các seed.
- Đối chiếu: P1 `pf100` vs `ekf`; P2 `pf100-k1` vs `pf100`, `pf100-k2` vs `pf100`; P3 biến thể K tốt hơn vs `ekf`.

### 5.3 Phân tầng định trước [QĐ]

Ghép GT–tracker từng frame: Hungarian trên IoU ≥ 0,5 (cùng ngưỡng CLEAR). Với mỗi track GT: ID chính = ID tracker được ghép
nhiều nhất; `ty_le_giu_id` = số frame ghép với ID chính / số frame của track GT; `n_doi_id` = số lần ID ghép đổi giữa hai
frame ghép liên tiếp.

| Tầng | Định nghĩa (cố định từ GT, trước khi xem đầu ra tracker) | Cỡ mẫu đã biết (dev12) |
|:--|:--|:--|
| Che khuất hoàn toàn | sự kiện `src/occlusion_eval.py`, ngữ cảnh 1,2 s, như mốc 2 | theo đầu ra occlusion_eval |
| Lớp | `vehicle_type` GT DETRAC: car / van / bus / others | bus: 14 track / 8 video (`03` mục 5.2) |
| Kích thước | tam phân vị của trung vị h GT theo track, ngưỡng tính trên toàn bộ track GT dev12 | — |
| Nhóm cua | `results/phase2/kc48_audit/_dem/quy_dao/detrac_train.csv`, video dev12: `loai_max` ∈ {cua, cua_gat} = cua; `thang` = thẳng; `cong_nhe` riêng; loại `loai_ngan`, `loai_cham`, `nghi_nhieu`; độ nhạy theo `loai_p90` | 32 track cua (8 theo p90) |
| Xe nhỏ cạnh / bị che bởi xe lớn | tiêu chí khóa `canh_xe_lon` trong `results/phase2/kc48_audit/tieu_chi_cong_bo.json` (sha256 `d7543fff…`), dựng lại danh sách sự kiện bằng hàm của `src/kc48/audit_du_lieu.py`; phải khớp 88 sự kiện / 65 track nhỏ / 9 video, lệch ⇒ dừng và báo | 88 / 65 / 9 video; 3 video không có sự kiện: `MVI_40992`, `MVI_63552`, `MVI_63554` |

Kết cục sự kiện xe nhỏ: **giữ ID** khi track GT nhỏ được ghép trong 1,2 s trước sự kiện và trong 1,2 s sau sự kiện với
cùng ID tracker. **Loại** (đếm theo lý do): không ghép được trước sự kiện; sự kiện chạm đầu / cuối video; track GT kết thúc
trong sự kiện. Mã track ở `quy_dao` phải khớp ID target GT (R48D kiểm số frame trùng; lệch ⇒ báo).

Mọi tầng: báo số video / track / sự kiện / trường hợp loại; CI bootstrap theo video; tầng có < 5 video chỉ **mô tả**,
không suy luận. Không chọn tầng theo kết quả.

## 6. Kỹ thuật PF (tối đa hai) [QĐ]

Căn cứ: [QS] ESS/N 0,007–0,017 sau 11–30 bước mất đo; ESS/N ≈ 0,25 và `n_unique`/N ≈ 0,38 khi đang bám (log chẩn đoán dev,
`02` mục 4 H4, H5); [QS] `src/test_pf_ctrv.py` test 9: đường thẳng, N = 100 cho θ = −49,3°, v = 3,58 (thật 30°, 5);
[TN] ca xe D: θ̂ trôi 0,02 → 0,99 rad khi đang bám rồi mất ID sau 8 frame mất đo (`02` mục 4, một ca, một seed).

### 6.1 K1 — proposal tối ưu cho bước có đo (mới, tùy chọn, mặc định tắt)

[LT] Mô hình của mã: x_k = f(x_{k−1}) + w, w ~ N(0, Q(h_{k−1})) với Q **chéo** (`process_noise_std_batch`); đo tuyến tính
z = H x + v, v ~ N(0, R), H chọn (cx, cy, a, h), R chéo tính tại h quan sát. Khi đó mật độ đề xuất tối ưu
p(x_k | x_{k−1}^i, z_k) là Gauss đóng dạng và tách theo tọa độ:
- tọa độ đo j: s_j² = (1/q_{ij}² + 1/r_j²)^{−1}, m_{ij} = s_j² (f_j(x^i)/q_{ij}² + z_j/r_j²);
- tọa độ không đo (v, θ, ω, v_a, v_h): giữ phân bố prior N(f_j(x^i), q_{ij}²) (H không chạm, Q chéo).

Trọng số theo công thức tổng quát w_k ∝ w_{k−1} p(z|x) p(x|x_{k−1}) / q(x|x_{k−1}, z) cho đúng
w_k^i ∝ w_{k−1}^i · p(z_k | x_{k−1}^i) = w_{k−1}^i · Π_j N(z_j; f_j(x^i), q_{ij}² + r_j²) (không phụ thuộc mẫu mới).
Trong pipeline, predict (trước association) đã cộng nhiễu prior; với K1, lúc update chỉ **rút lại** các tọa độ đo từ
N(m_{ij}, s_j²) dựa trên f(x^i) và Q_i lưu từ bước predict; tọa độ không đo giữ mẫu prior đã rút (đúng phân bố đề xuất).
Bước không có đo: proposal = prior, trọng số không đổi (như hiện tại). Giữ nguyên ngưỡng ESS, ước lượng, luật suy biến;
phép chặn h ≥ 10⁻³ giữ như mã hiện tại và ghi là heuristic. [GT] K1 làm giảm phương sai trọng số mỗi update (Q và R vị trí
đều có σ = h/20), nên tăng ESS, giảm số lần resample khi đang bám (nhắm H5) và giảm một phần ESS sụp khi ghép lại (H4);
không sửa được độ phân tán do nhiều bước mất đo tích lũy. Đây là kỹ thuật đã biết trong lý thuyết PF (proposal tối ưu;
nguồn chưa đối chiếu toàn văn trên máy), **không** ghi là đóng góp lý thuyết mới; cái mới chỉ là áp vào mã này.
Test bắt buộc (R48D): mặc định tắt cho kết quả bit-identical với mã cũ; với tùy chọn bật, trọng số khớp công thức trên
ca nhỏ tính tay độc lập và trung bình có trọng số hội tụ về hậu nghiệm Kalman trên mô hình tuyến tính một bước.

### 6.2 K2 — roughening sau resample (đã có trong mã, không phải đóng góp mới)

`pf_roughening` (`src/pf_ctrv.py:PFCTRV.resample_if_needed`, mặc định 0 = tắt): cộng nhiễu roughening·σ_Q sau resample.
Nhắm mất đa dạng sau resample (H5). Giá trị ứng viên {0,25; 0,5; 1,0}.

### 6.3 Không chọn trong lượt này

Đổi ngưỡng ESS, bước MCMC (resample-move), hiệu chỉnh tiến dần (progressive correction), Rao-Blackwell hóa: hoặc là
tham số đã có không nhắm thẳng triệu chứng, hoặc thay đổi lớn hơn ngân sách. Không đổi luật suy biến, ước lượng
`mean`, prior khởi tạo.

## 7. Association / gating [QĐ]

Không đổi association hay gating ở mọi biến thể: ByteTrack hai vòng IoU, `match_thresh 0.8`, `fuse_score True`,
`LostAwareAssociation` tắt (mặc định yaml), association không theo lớp (không khóa lớp theo detection). Vì không có thay
đổi association, lượt này **không có** ablation association; mọi khác biệt pipeline là do bộ lọc (cộng heuristic khác
nhau giữa PF và EKF đã liệt kê ở `02` mục 5).

## 8. Ngân sách tinh chỉnh cân bằng [QĐ]

- Tham số CTRV dùng chung: không tinh chỉnh cho bất kỳ bộ lọc nào.
- K1: không có siêu tham số.
- K2: tối đa 3 giá trị (mục 6.2). EKF được cùng ngân sách: 3 cấu hình nhân (σ_θ, σ_ω) với {0,5; 1; 2} qua khóa
  `ctrv_std_theta`, `ctrv_std_omega`.
- Chọn chỉ trên seed tinh chỉnh d = 1000–1009 của mô phỏng, quy tắc: F (ngưỡng chính) trung bình nhỏ nhất trên mọi kịch
  bản (gồm nhóm lệch mô hình), L ∈ {10, 20}, κ = 1, cả hai h0; chênh < 0,005 thì so e_r; vẫn hòa thì chọn giá trị nhỏ hơn.
- Dev: không tinh chỉnh thêm; chỉ đánh giá các cấu hình đã chọn. EKF mặc định luôn là tham chiếu chính; EKF đã chỉnh
  (nếu khác mặc định) thêm ở ưu tiên P2.
- N ∈ {100, 300, 500} là đường tài nguyên, không phải tham số được chọn.

## 9. Ngân sách tính toán [QĐ]

- Mô phỏng: tối đa khoảng 90 phút CPU, chia đoạn < 10 phút / lệnh, tiếp tục được. Vượt ⇒ giảm theo thứ tự: số seed nhiễu
  30 → 20; bỏ κ = 0,5; bỏ N = 500 trong mô phỏng. Không bao giờ bỏ: nhóm lệch mô hình, L ∈ {10, 20}, nhánh EKF.
- Pipeline dev: ước lượng từ smoke R48C (giây mỗi video × số video × số seed, theo từng biến thể). Tối đa khoảng 120 phút
  tracking + đánh giá. Ưu tiên P1: `ekf`, `pf100`, `pf100-k1` (3 seed); P2: `pf100-k2`, `pf100-k1k2`, EKF đã chỉnh; P3:
  N = 300 / 500 cho `pf` và biến thể K tốt hơn. Phần vượt ngân sách ghi là **bị chặn**, không chạy full matrix.

## 10. Cổng tính đúng trước thí nghiệm lớn [QĐ]

Trước R48D: `src/kc48/test_pf_dung_dan.py` exit 0; test cũ `src/test_pf_ctrv.py` (trừ test 11 nếu người dùng chưa quyết
định, xem `05`), `src/test_ekf_ctrv.py`, `src/test_ukf_ctrv.py`, `src/test_tracker_lifecycle.py`,
`src/test_init_regression.py` đạt; smoke `kc48_pf_smoke` chạy hết, exit 0. Smoke chỉ dùng kiểm pipeline và đo thời gian;
số HOTA / IDF1 của smoke **không** dùng cho kết luận nào. Lỗi chưa giải quyết ở cổng này ⇒ không chạy R48D.

## 11. Báo cáo

Báo mọi biến thể đã chạy, kể cả kết quả bất lợi và ô không phân biệt; báo cỡ mẫu từng tầng và trường hợp loại; mỗi số truy
về run / cấu hình / commit / file kết quả. Phần chưa chạy ghi là chưa chạy. Đầu ra R48D: `results/phase2/kc48_mo_phong/`,
`results/phase2/kc48_pf_dev/`, `results/trackeval/DETRAC-all/p2-kc48_pf_dev-*`.

## 12. Giới hạn biết trước

- Dev12 nhỏ (12 video; 32 track cua; 88 sự kiện xe nhỏ trong 9 video): CI rộng; dev12 từng dùng trong thiết kế mốc 2.
- Nguồn gốc cấu hình của phần lớn sequence trong cache `legacy` không xác minh được bằng hash (mục 0 của
  `handoff/PLAN_DA_CHOT.md`); xác thực ở mục 2 chỉ chứng minh cache **trùng byte** với cache mốc 2 dev đã dùng.
- Mô phỏng trên mặt phẳng ảnh; `phoi_canh` chỉ là một dạng lệch mô hình, không đại diện mọi cảnh thật.
- Khác biệt pipeline gộp cả heuristic triển khai khác nhau giữa PF và EKF (`02` mục 5), không chỉ cách biểu diễn hậu nghiệm.

## Phụ lục — lệch giao thức

(Chưa có. Mỗi mục thêm sau khi chụp sha256 ghi: thời điểm, mục bị lệch, lý do, ảnh hưởng.)

### P1 — Rút ngoại lệ "trừ test 11" ở mục 10 (2026-10-11, sau lần chụp 3)

- **Thời điểm**: thêm ngay sau lần chụp 3 trong `results/phase2/kc48_pf_code/giao_thuc_sha256.json`. Lần chụp 3 trùng
  sha256 `7063c532…` của lần chụp 1, nên mục 1–12 chưa sửa lần nào. Lần chụp 4 ghi sha256 của file có phụ lục này,
  kèm kiểm 22.510 byte đầu vẫn có sha256 `7063c532…`.
- **Mục bị lệch**: mục 10 có cụm "(trừ test 11 nếu người dùng chưa quyết định, xem `05`)". Agent tự thêm cụm này khi chưa
  có quyết định của người dùng, nên nó **không** thay được tiêu chí R48C đã duyệt ("chạy cùng `src/test_pf_ctrv.py`").
- **Lý do rút**: ngày 11/10/2026, người dùng quyết định cho `src/test_pf_ctrv.py` ghi fixture tổng hợp ở
  `data/interim/det_cache/fixture_pf_seed_test/` và `data/processed/trackers/DETRAC-sample/p2-pftest-s{0,1}/`.
- **Hiệu lực từ nay**: cổng mục 10 đòi `python src/test_pf_ctrv.py` chạy **đủ 11 test, gồm `test_11_runner_seeds`**,
  exit 0. Trước và sau lần chạy phải chụp `results/phase2/kc48_pf_code/fixture_{truoc,sau}.json`; phép so phải cho thấy
  ngoài 3 thư mục fixture không file nào trong `data/interim`, `data/processed` bị đổi, thêm hay mất. Ngoại lệ cũ bị bỏ,
  không thay bằng ngoại lệ khác.
- **Ảnh hưởng**: chỉ siết cổng tính đúng; không đổi thiết kế thí nghiệm, chỉ số hay ngân sách. Kết quả chạy:
  `05_dung_dan_pf_va_smoke.md` mục 4.

### P2 — Ghi nhận cách thực thi R48D và các lệch nhỏ (2026-10-11, sau khi chạy R48D)

- **Thời điểm khóa**: trước mọi so sánh R48D, `src/kc48/thi_nghiem_mo_phong.py --giai-doan chup-giao-thuc` ghi
  `results/phase2/kc48_mo_phong/giao_thuc_sha256_r48d.jsonl` lúc 15:31:12 +07: sha256 `479497fc…` (file có P1), 22.510
  byte đầu vẫn `7063c532…`. Đơn vị mô phỏng đầu tiên xong lúc 15:31:24 sau 6,31 s, tức bắt đầu khoảng 15:31:18 (`nhat_ky.jsonl` dòng 1); run dev đầu tiên xong lúc
  15:58:50 (`results/phase2/kc48_pf_dev/nhat_ky_song_song.jsonl`). P2 này viết **sau** khi xem kết quả, chỉ ghi nhận, không
  đổi mục 1–12.
- **Mục 5.3, tầng xe nhỏ cạnh xe lớn — luật loại**: cài bằng `src/kc48/su_kien_dev.py`. Luật loại theo GT dùng
  `occlusion_eval.add_gt_eligibility` (ngữ cảnh 30 frame): `gt_missing_before` gộp "sự kiện chạm đầu video" và "track bắt
  đầu trong / sát sự kiện"; `gt_missing_after` gộp "chạm cuối video" và "track GT kết thúc trong sự kiện". Luật "không
  ghép được trước sự kiện" phụ thuộc tracker, nên không loại khỏi tập chung mà báo riêng là kết cục `no_match_before`.
  Tỉ lệ giữ ID báo hai cách: (i) trên mọi sự kiện đủ điều kiện GT (`no_match_before` tính là không giữ, như mốc 2);
  (ii) trên sự kiện ghép được trước, đúng chữ của mục 5.3. Hiệu ghép cặp ở (ii) chỉ tính trên sự kiện mà **cả hai** biến
  thể ghép được trước ở mọi seed. Lý do: giữ phép so ghép cặp trên cùng tập sự kiện.
- **Mục 5.3, che khuất**: gọi các hàm `build_events`, `add_gt_eligibility`, `match_video`, `evaluate_events` của
  `src/occlusion_eval.py` trong cùng tiến trình, với gốc `config.PROCESSED_DIR` đã chuyển (đúng cách mục 5.1 cho phép).
  Không dùng bước `occlusion` của runner. Báo cả mọi sự kiện lẫn tầng con `has_full`.
- **Mục 10 trong R48D**: `test_11_runner_seeds` ghi vào `data/interim/det_cache/fixture_pf_seed_test/` và
  `data/processed/trackers/DETRAC-sample/p2-pftest-s{0,1}/`. Hai thư mục này **không** nằm trong `write_paths` của R48D,
  nên sau khi thêm K1, test 1–10 chạy qua `src/kc48/chay_test_pf_cu.py`; test 11 không chạy lại ở R48D. Test 11 đã đạt ở
  R48C, trên mã trước khi thêm K1. Bằng chứng thay thế cho tính tái lập của đường mặc định sau khi sửa mã:
  - `src/kc48/test_k1_k2.py` test 1: mặc định bit-identical với mã `eaf2e87`;
  - `results/phase2/kc48_pf_dev/phan_tich/mac_dinh_vs_smoke.json`: file tracking `ekf`, `pf100-s0`, `pf100-s1` trên
    2 video trùng sha256 với smoke R48C.
- **Mục 5.2, P3 "biến thể K tốt hơn"**: chọn K1. Cơ sở: C2 / C3 mô phỏng (K1 giảm F nhất quán; K2 chỉ giảm e_r) và P2 dev
  (hiệu HOTA của K1 so với `pf100` lớn hơn hiệu của K2). Dev là tập phát triển, không phải tập đánh giá cuối. Đây là
  lựa chọn biến thể để chạy tiếp, không phải tinh chỉnh tham số.
- **Mục 4.3, tài nguyên**: `us_buoc` đo trong harness có bật `enable_diag()` cho PF, nên gồm cả phần ghi chẩn đoán.
  `results/phase2/kc48_mo_phong/toc_do.csv` đo một tiến trình, tuần tự; cột `us_buoc_song_song_tb` trong `c4_tai_nguyen.csv`
  đo khi 6 tiến trình chạy song song, chỉ để tham khảo.
- **Ảnh hưởng**: không đổi chỉ số, đối chiếu, quy tắc diễn giải hay ngân sách. Toàn bộ P1–P3 của mục 9 đều chạy
  (26 run dev), không phần nào bị chặn vì ngân sách.

### P3 — Cài đặt R48D lệch chữ của mục 4.3, 5.2, 5.3; đã sửa theo Codex review_003 (2026-10-11, sau khi xem kết quả)

Phụ lục này không đổi mục 1–12. Văn bản giao thức đã đúng; phần cài đặt của R48D lệch khỏi văn bản ở ba chỗ, nay sửa cho
khớp. Chi tiết và số trước / sau: `07` mục 11.
- **Mục 5.3 (ghép GT–tracker)**: giao thức ghi "Hungarian trên IoU ≥ 0,5". Bản cài cũ (`phan_tich_dev.giu_id`) chạy
  Hungarian trên mọi IoU rồi mới lọc ngưỡng. Nay dùng `occlusion_eval.match_frame`, tức ràng buộc ngưỡng trước Hungarian.
- **Mục 5.2 (seed PF)**: giao thức ghi "mỗi mẫu bootstrap lấy trung bình 3 seed" của chỉ số COMBINED. Bản cài cũ lấy trung
  bình trường thô trước. Nay tính COMBINED từng seed rồi trung bình, trong từng mẫu (`src/kc48/boot_seed.py`).
- **Mục 4.3 (chẩn đoán PF)**: "ESS/N trước resampling ở mọi update" và "tỉ lệ update có resample" chỉ có nghĩa với update
  qua nhánh trọng số. Nhánh tái khởi tạo khi suy biến (`src/pf_ctrv.py:308–322`) không có ESS thật và không resample hệ
  thống, nên nay bị loại khỏi hai chỉ số đó và đếm riêng:
  - `reinit_tr`, `n_reinit_bam` / `n_upd_bam`, `n_deg` / `n_update` theo từng quỹ đạo;
  - đối chiếu chẩn đoán C2 / C3 chỉ dùng các d hợp lệ ở cả hai biến thể, ghi `n_d`.
  Đây là làm rõ định nghĩa, không thêm chỉ số chính mới.
- **Chạy bổ sung**: 14.073 quỹ đạo mô phỏng (12.235 cần thiết + 1.838 đối chứng), cùng cấu hình `chon.json` và seed của
  mục 4.1, 169 s. Không chạy lại tracking dev.
- **Ảnh hưởng**: e_r, F và mọi đối chiếu chính C1–C3, P1–P3 không đổi trạng thái. Thay đổi đáng kể duy nhất: C3 (K2)
  ESS/N tại t_r giảm từ 10 ô CI > 0 xuống 0 ô; kết luận "chưa phân biệt" giữ nguyên.
