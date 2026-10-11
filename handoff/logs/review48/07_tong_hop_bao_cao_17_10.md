# R48D / M6 — Tổng hợp cho báo cáo 17/10/2026

Viết ngày 2026-10-11. Nhánh `phase2-mixed-traffic`, HEAD `eaf2e87`. Thay đổi chưa commit: `src/pf_ctrv.py`,
`src/tracker_ctrv.py` (K1 + hook chẩn đoán, `git diff` ở mục 6.1), `src/kc48/*`, `configs/phase2/trackers/kc48/*`,
`configs/phase2/experiments/kc48_*.yaml`. Mọi run R48D chạy trên cây làm việc này; `git_head` ghi trong nhật ký là
`eaf2e87`. **Chưa commit**: người dùng quyết định riêng.

Nhãn: [LT] lý thuyết có nguồn, [QS] quan sát đo được (kèm file), [GT] giả thuyết chưa kiểm. Cỡ mẫu ghi cạnh từng số.
Phần chưa chạy được ghi là chưa chạy.

*Đồng bộ R49A (2026-10-12, chỉ văn bản, không chạy lại gì, không đổi số liệu):* sửa đường dẫn sai / mơ hồ và ghi rõ mẫu
số (đơn vị đếm, tập dữ liệu, điều kiện chọn / loại, phạm vi) cạnh các số liệu. Đường dẫn tương đối `phan_tich/…`,
`chon.json`, `toc_do.csv`, `nhat_ky.jsonl`, `hinh_hat_so_huu.*` ở mục 5 và mục 11 (phần F2) là dưới
`results/phase2/kc48_mo_phong/`; ở mục 6 và mục 11 (phần F1, F3) là dưới `results/phase2/kc48_pf_dev/` (tên CSV không
có thư mục ở mục 6.6, như `co_mau_va_loai.csv`, `bang_su_kien.csv`: dưới `results/phase2/kc48_pf_dev/su_kien/`). Tên `01`–`08`,
`05b`, `05c` là file cùng thư mục `handoff/logs/review48/`.

Thuật ngữ mẫu số dùng chung:
- **Mô phỏng (mục 5, 11)**: *ô* = một tổ hợp (kịch bản, L, κ, h0) của lưới 8 × 4 × 3 × 2 (`04` mục 4.1). *Quỹ đạo* = một
  lần chạy bộ lọc trên một d (seed nhiễu); mỗi ô có 30 d đánh giá (0–29); PF thêm 3 seed PF nên mỗi ô PF có 90 quỹ đạo,
  EKF 30. Đơn vị thống kê và đơn vị bootstrap là d (PF: trung bình 3 seed PF trong mỗi d trước), B = 5000 (`04` mục 4.4).
- **Pipeline (mục 6)**: dev12 = 12 video UA-DETRAC Train trong `results/phase2/manifests/detrac_dev.json`, điều kiện A
  (`04` mục 2), không dùng tập test / BDD val. Đơn vị bootstrap là video (12), B = 5000, seed 0.

## 1. Lý thuyết (chi tiết: `01_ly_thuyet_kf_ekf_pf.md`)

- KF–CV (trạng thái XYAH 8-D) của ByteTrack. Nguồn: ByteTrack arXiv 2110.06864v3, đã đối chiếu toàn văn tr. 2–7 trong
  `tai_lieu_nguon/`.
- EKF–CTRV và PF–CTRV có phương trình. SIS / SIR, ESS = 1 / Σw², resample hệ thống, log-sum-exp; đơn vị bước, chuẩn hóa
  góc, giới hạn ω ≈ 0. Các ý lấy từ Arulampalam et al. 2002 (DOI 10.1109/78.978374) **chưa đối chiếu toàn văn**: trên máy
  chỉ có metadata.
- Proposal tối ưu (K1): với Q, R chéo và H tuyến tính chọn tọa độ, p(x_k | x_{k−1}, z_k) là Gauss đóng dạng; trọng số
  w_k ∝ w_{k−1} · N(z; H f(x_{k−1}), H Q Hᵀ + R) (`04` mục 6.1). Đây là kỹ thuật đã biết, **không** phải đóng góp lý thuyết
  mới; nguồn chưa đối chiếu toàn văn. Roughening (K2) có sẵn trong mã từ trước, cũng không phải đóng góp mới.

## 2. Audit dữ liệu (chi tiết: `03_audit_du_lieu.md`; số liệu `results/phase2/kc48_audit/`)

| Mục | Kết quả | Nguồn |
|:--|:--|:--|
| Mốc slide lần 2 | khớp khi đếm lại: BDD train-fit 1.261 video / 50.487 ảnh; dev 139 / 5.559; BDD val 200 / 39.973. *Mẫu số*: "dev" là dev nội bộ BDD của detector (không phải dev12 UA-DETRAC); ảnh train-fit / dev là frame lấy mẫu `buoc_frame` 5 trong khoảng có nhãn; ảnh val là mọi ảnh của COCO-video val | `results/phase2/kc48_audit/moc_doi_chieu.csv`; `03` mục 2 |
| Toàn vẹn ảnh | 458.183 ảnh: sha256, header, SOI / EOI 0 lỗi; giải mã 6.783 ảnh mẫu 0 lỗi. *Mẫu số*: đơn vị là file ảnh trên đĩa; 458.183 = mọi ảnh của BDD train 1.400 video (train-fit + dev, mọi frame có nhãn), BDD val và UA-DETRAC 100 video (dev12, train khác, test40); 6.783 = mẫu 4 frame / video của các tập này | `03` mục 6.1 |
| Nhãn BDD train | 1.400/1.400 file khớp manifest nhập vòng 34; **chưa** xác thực với archive gốc. *Mẫu số*: file nhãn JSON Scalabel của 1.400 video BDD train (train-fit + dev), so byte và sha256 với `handoff/logs/review34/03_manifest_nhap_nhan_train.tsv` | `03` mục 4 |
| Bản xuất YOLO | 320/320 mẫu khớp nhãn gốc; ảnh tô xám: kiểm pixel trên 160 ảnh mẫu (R48E) cho 131 đạt / 26 chưa kết luận (vùng mỏng) / 3 không đạt (MAE box chồng vùng > 2,0; hậu kiểm: phù hợp mất mát do nén lại JPEG); chỉ mẫu, không suy ra toàn tập. *Mẫu số*: đơn vị ảnh của `data/yolo_bdd5_v1` (split train / dev); 320 = 8 tầng (split × xử lý × có lớp hiếm) × 40 ảnh, seed 48; 160 = 4 tầng `to_xam` × 40 ảnh trong 320 ảnh đó, trên tổng 18.600 ảnh tô xám | `03` mục 6.3, 14 |
| Chồng lấn | 70/139 video dev BDD chung tiền tố tên với train-fit; 1 cặp gần trùng xuyên train-fit / dev. Giữa train / val (BDD) và train / test (DETRAC) chưa phát hiện chồng lấn trong phạm vi đã kiểm. *Mẫu số*: 70/139 tính theo video dev nội bộ BDD; "cặp gần trùng" là cặp frame thuộc hai video khác nhau, dHash ≤ 20 bit, chỉ trên mẫu 4 frame / video (6.383 frame BDD, 400 frame DETRAC); có 3 cặp như vậy, 1 cặp xuyên train-fit / dev | `03` mục 6.4 |
| UA-DETRAC dev12 | 12 video; 32 track cua (8 theo p90); bus 14 track / 8 video; 88 sự kiện xe nhỏ cạnh xe lớn / 65 track / 9 video. *Mẫu số*: 12 video / 16.347 frame có `<frame>` trong XML; track cua = track có `loai_max` ∈ {cua, cua gắt} trong 715 track hợp lệ (1.185 track, bỏ `loai_ngan`, `loai_cham`, `nghi_nhieu`), p90 trên cùng tập hợp lệ; bus theo `vehicle_type` gốc; sự kiện = chuỗi frame liên tiếp của cùng cặp (car nhỏ, bus / others lớn) thỏa tiêu chí khóa `canh_xe_lon`; 65 là số track car nhỏ liên quan | `03` mục 5.2, 8, 9 |
| Lệch miền | bus DETRAC kt_rel p50 0,18–0,20 so với BDD 0,05–0,07; bus DETRAC cắt khung 25,9–48,6 %. *Mẫu số*: đơn vị bbox bus; kt_rel = √(diện tích box / diện tích ảnh), p50 trên bbox; DETRAC: train60, dev12, test40 (mọi frame có `<frame>`); BDD: frame `_b5` của train-fit, dev và mọi frame val; "cắt khung" = % bbox bus có `truncation_ratio` > 0 (dev12 25,9 %, test40 31,2 %, train60 48,6 %) | `03` mục 5.1, 5.2 |

## 3. Đối chiếu code (chi tiết: `02_doi_chieu_code_pf.md`)

- Mỗi track PF có mảng particle, log-trọng số và rng riêng. Chỉ đối tượng `PFCTRV` (cấu hình + bộ đếm) dùng chung.
  Giải thích "cạnh tranh hạt toàn cục / xe lớn hút hạt" (H1) **không áp dụng** cho mã này (`results/phase2/kc48_pf_code/so_huu_hat.json`).
  Test `src/kc48/test_pf_dung_dan.py` test 10 kiểm lại: đổi detection của track B thì particle của track A vẫn
  bit-identical.
- Giả thuyết H2–H7 ở `02` mục 4. R48D kiểm được một phần H4, H5, H6, H7 trên mô phỏng (mục 5). H2 (association) và
  H3 (likelihood trên dữ liệu thật): **chưa kiểm**.
- R48C không sửa lỗi mã tracker (`05` mục 3). R48D thêm K1 dưới dạng tùy chọn (mục 6.1).

## 4. Giao thức (`04_giao_thuc_kiem_chung_pf.md`)

Giao thức viết trước mọi so sánh. sha256 mục 1–12 là `7063c532…`, chụp trước smoke R48C và chụp lại trước R48D lúc
15:31:12 (`results/phase2/kc48_mo_phong/giao_thuc_sha256_r48d.jsonl`). Phụ lục P1 do R48C thêm; P2 do R48D thêm sau khi
chạy (cách cài luật loại sự kiện, test 11, chọn K1 cho P3). Sau P2, file có sha256 `2bd7c1dd…` (chụp 16:42:17). P2 được
sửa câu chữ một lần nữa, cùng số byte, nên sha256 đổi thành `b3e56bb3…` (chụp 16:46:05, chụp lại 20:07:56 khi chốt
R48D). Ở cả ba lần, 22.510 byte đầu vẫn là `7063c532…`: chỉ phụ lục đổi, mục 1–12 không đổi
(`giao_thuc_sha256_r48d.jsonl` dòng 2–4).

Tóm tắt: mô phỏng biết liên kết gồm 8 kịch bản, trong đó 4 kịch bản lệch mô hình. L ∈ {0, 5, 10, 20}, κ ∈ {0,5; 1; 2},
h0 ∈ {24, 72}. 30 seed nhiễu d ghép cặp, 3 seed PF. Pipeline dev12 điều kiện A với cache `yolov8n-coco_conf0.05_all`,
bootstrap theo video. Quy tắc "khác biệt nhất quán": CI loại 0 ở ≥ 2/3 số ô, cùng dấu, và không ô nào ngược dấu
(ô và CI theo d: xem thuật ngữ ở đầu file; số ô của từng đối chiếu là cột `n_o` của
`results/phase2/kc48_mo_phong/phan_tich/ket_luan_doi_chieu.csv`).

## 5. Kết quả cấp bộ lọc — mô phỏng (`results/phase2/kc48_mo_phong/`)

Lệnh: `python src/kc48/thi_nghiem_mo_phong.py --giai-doan {chup-giao-thuc, tinh-chinh, danh-gia, phan-tich, hinh-hat,
do-toc-do}`. Mọi đơn vị trong `results/phase2/kc48_mo_phong/nhat_ky.jsonl`. Chỉ số: e_r là sai số tâm dự đoán khi ghép
lại, đơn vị h. F là thất bại: IoU dự đoán tại t_r < 0,2/0,9. Cả hai tính trên từng quỹ đạo tại bước ghép lại t_r (L = 0:
sai số dự đoán 1 bước trung bình trên t ∈ [t_gap, t_gap + 20)); F trung bình là tỉ lệ quỹ đạo thất bại (`04` mục 4.3).

### 5.1 Tinh chỉnh cân bằng (`chon.json`; seed d = 1000–1009, tách khỏi seed đánh giá)

*Mẫu số*: F trung bình trên mọi kịch bản (gồm nhóm lệch mô hình), L ∈ {10, 20}, κ = 1, cả hai h0, d 1000–1009; K2: 960
quỹ đạo mỗi giá trị c (có 3 seed PF), EKF: 320 quỹ đạo mỗi hệ số nhân (cột `n_dong` của `chon.json`; quy tắc `04` mục 8).

- K2: roughening c ∈ {0,25; 0,5; 1,0} cho F trung bình 0,792 / 0,788 / 0,767 ⇒ chọn **c = 1,0**.
- EKF: nhân (σ_θ, σ_ω) với {0,5; 1; 2} cho F 0,300 / 0,294 / 0,266 ⇒ chọn **×2** (`ekf-chinh`).

### 5.2 Đối chiếu định trước (`phan_tich/ket_luan_doi_chieu.csv`, ô: `phan_tich/doi_chieu.csv`)

*Mẫu số*: cột "Số ô" đếm ô (đơn vị là ô, không phải quỹ đạo); CI 95 % của mỗi ô bootstrap ghép cặp theo 30 d đánh giá.
"Hiệu TB các ô" là trung bình trên các ô của hiệu trung bình theo d (cột `hieu_tb_cac_o`). Phạm vi ô theo `04` mục 4.4:
C1, P3, `ekf-chinh` − `ekf` và mọi dòng ESS / resample: L ∈ {10, 20}, κ = 1, 8 kịch bản, 2 h0 (32 ô); C2, C3 với F, e_r
thêm L = 0 (48 ô); "nhóm lệch mô hình" = 4 kịch bản `phoi_canh`, `gia_toc`, `doi_lan`, `nhieu_duoi_day` (16 ô); "nhóm đúng
CTRV" = `thang`, `cua_deu` (8 ô). ESS/N và tỉ lệ resample khi đang bám chỉ tính trên update có đo không rơi vào nhánh tái
khởi tạo (mẫu số `n_upd_bam` − `n_reinit_bam`, xem mục 11).

| Đối chiếu | Chỉ số | Số ô CI > 0 / CI < 0 / tổng | Hiệu TB các ô | Kết luận theo quy tắc |
|:--|:--|:--|--:|:--|
| C1 `pf100` − `ekf` (L 10, 20; κ 1) | F | 25 / 0 / 32 | +0,525 | nhất quán: PF thất bại **nhiều hơn** |
| | e_r | 32 / 0 / 32 | +2,54 h | nhất quán: PF sai **nhiều hơn** |
| C1, nhóm lệch mô hình | F | 12 / 0 / 16 | +0,494 | nhất quán (PF kém hơn) |
| C2 `pf100-k1` − `pf100` (thêm L 0) | F | 0 / 39 / 48 | −0,243 | nhất quán: K1 **giảm** thất bại |
| | e_r | 0 / 48 / 48 | −1,36 h | nhất quán |
| | ESS/N khi đang bám | 32 / 0 / 32 | +0,290 | nhất quán |
| | tỉ lệ resample khi đang bám | 0 / 32 / 32 | −0,318 | nhất quán |
| | ESS/N tại t_r (chỉ d có ESS hợp lệ ở cả hai biến thể, mục 11) | 27 / 0 / 32 | +0,005 | nhất quán, nhưng độ lớn rất nhỏ |
| C3 `pf100-k2` − `pf100` | F | 0 / 13 / 48 | −0,053 | chưa phân biệt |
| | e_r | 0 / 36 / 48 | −0,455 h | nhất quán (giảm) |
| | ESS/N tại t_r (như trên) | 0 / 0 / 32 | −0,000 | chưa phân biệt (trước vòng sửa: 10 ô CI > 0 — do trộn tái khởi tạo, mục 11) |
| P3 `pf-k1` (N 100) − `ekf` | F | 23 / 2 / 32 | +0,344 | chưa phân biệt (có 2 ô ngược dấu); nhóm đúng CTRV: 8/8 PF kém hơn |
| | e_r | 30 / 0 / 32 | +1,00 h | nhất quán: PF + K1 vẫn kém EKF |
| `ekf-chinh` − `ekf` | F | 0 / 2 / 32 | −0,016 | chưa phân biệt |
| | e_r | 11 / 10 / 32 | −0,015 h | chưa phân biệt |

Độ nhạy κ ∈ {0,5; 2} (`C1_nhay`, `C2_nhay`, `C3_nhay`, 64 ô mỗi chỉ số theo `n_o`) cùng chiều với phân tích chính. Một
số nhóm rơi về "chưa phân biệt" (xem CSV).

### 5.3 Chẩn đoán cơ chế — ô `cua_deu`, L = 20, κ = 1 (`phan_tich/tom_tat_o.csv`, 30 seed nhiễu d × 3 seed PF)

| Bộ lọc | h0 | e_r (h) | F | RMSE lọc khi có đo (h) | sai số θ trước mất đo (rad) | NIS tại t_r (kỳ vọng 4) | ESS/N tại t_r [quỹ đạo hợp lệ / 90] | tái khởi tạo tại t_r / 90 | ESS/N khi bám | tỉ lệ tổ tiên khi bám | độ dời mean − f(x̂) (h, H6) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `ekf` | 24 | 0,23 | 0,00 | 0,051 | 0,23 | 0,83 | — | — | — | — | — |
| `pf100` | 24 | 5,49 | 0,96 | 1,62 | 1,45 | 1,3·10⁹ | 0,011 [47] | 43 | 0,066 | 0,11 | 0,075 |
| `pf100-k1` | 24 | 2,64 | 0,78 | 0,13 | 0,92 | 63 | 0,013 [87] | 3 | 0,345 | 0,43 | 0,105 |
| `pf100-k2` | 24 | 4,04 | 0,91 | 1,22 | 1,33 | 1,3·10⁹ | 0,011 [71] | 19 | 0,065 | 0,13 | 0,091 |
| `ekf` | 72 | 0,40 | 0,10 | 0,051 | 0,81 | 1,27 | — | — | — | — | — |
| `pf100` | 72 | 1,43 | 0,87 | 0,31 | 1,64 | 7,0·10⁹ | 0,012 [89] | 1 | 0,168 | 0,29 | 0,052 |
| `pf100-k1` | 72 | 0,96 | 0,72 | 0,061 | 1,39 | 8,9 | 0,015 [90] | 0 | 0,459 | 0,51 | 0,096 |

Vòng sửa F2 (mục 11): ESS/N và tỉ lệ resample chỉ tính trên update **không** rơi vào nhánh tái khởi tạo. Tái khởi tạo đếm
riêng; mẫu số ở `phan_tich/mau_so_chan_doan.csv`. ESS/N tại t_r = trung bình theo d của trung bình các seed PF hợp lệ
(`tom_tat_o.csv`). Trung bình gộp trên 47 quỹ đạo hợp lệ của ô `pf100` h0 24 là 0,01075
(`results/phase2/kc48_mo_phong/phan_tich/so_sanh_truoc_sau_vong_sua.json`, khóa `o_codex_cua_deu_h24`). Trước vòng sửa,
ô này ghi 0,006 vì 43 lần tái khởi tạo bị tính là ESS = 0.

*Mẫu số của bảng*: mỗi dòng là một ô (`cua_deu`, L 20, κ 1, h0 ghi trong bảng); PF 90 quỹ đạo (30 d × 3 seed PF), EKF 30
quỹ đạo (30 d). e_r, F tại t_r; RMSE lọc trên các bước có đo với t ≥ 10; sai số θ tại t_gap − 1 (chỉ khi tốc độ GT ≥ 1
px/bước); NIS tại t_r (`04` mục 4.3). "[quỹ đạo hợp lệ / 90]" và "tái khởi tạo tại t_r / 90" đếm quỹ đạo. ESS/N khi bám:
mẫu số là update có đo không tái khởi tạo; tỉ lệ tổ tiên khi bám chỉ dùng các lần systematic resampling (mục 11; số đếm
từng ô ở `phan_tich/mau_so_chan_doan.csv`). Gộp: theo `04` mục 4.4, PF lấy trung bình 3 seed PF trong mỗi d rồi trung
bình theo d; `phan_tich/tom_tat_o.csv` ghi số d dùng ở cột `n_d` (ESS/N tại t_r: `n_d_ess_tr`).

Quan sát [QS], chỉ ở cấp mô phỏng, trong hai ô `cua_deu`, L 20, κ 1 của bảng trên (trừ khi ghi khác):
- PF bootstrap N = 100 nghèo hạt **ngay khi đang bám**: ESS/N trung bình 0,07–0,17, gần như update nào cũng resample,
  sau resample chỉ còn 11–29 % tổ tiên khác nhau. RMSE lọc lớn hơn EKF khoảng 6 lần (h0 = 72) đến 32 lần (h0 = 24) (H5).
- Sau mất đo dài, ESS/N hợp lệ tại lần ghép lại còn khoảng 1 % ở mọi biến thể (H4). Ở h0 = 24, ngoài ra nhiều quỹ đạo
  không còn particle nào gần quan sát (Mahalanobis² nhỏ nhất > 10⁴) nên rơi vào nhánh tái khởi tạo tại t_r: `pf100` 43/90,
  K2 19/90, K1 3/90. Đây là một dạng thất bại khác của đám mây, không phải ESS. NIS cỡ 10⁹ cho thấy đám mây quá tự tin so
  với sai số thật. K1 hạ NIS xuống 9–63 nhưng vẫn xa giá trị kỳ vọng 4.
- Co trung bình (H6) chỉ cỡ 0,05–0,1 h, nhỏ so với e_r cỡ 1–5 h ⇒ không phải nguyên nhân chính trong các ô này.
- Lệch mô hình (H7): PF không có lợi thế ở nhóm lệch mô hình; C1 nhóm `lech` là 12/16 ô PF kém hơn (chỉ số F, ô có CI
  > 0 / số ô của nhóm, mục 5.2; không thuộc ô `cua_deu` của bảng trên).

### 5.4 Đường tài nguyên (`phan_tich/c4_tai_nguyen.csv`; thời gian: `toc_do.csv`, một tiến trình, có bật chẩn đoán)

| Biến thể | N | e_r TB (h) | F TB | µs / bước / track |
|:--|--:|--:|--:|--:|
| `ekf` | — | 0,62 | 0,30 | 57 |
| `pf` | 100 / 300 / 500 | 3,16 / 2,54 / 2,17 | 0,83 / 0,72 / 0,66 | 328 / 433 / 522 |
| `pf-k1` | 100 / 300 / 500 | 1,62 / 1,18 / 1,09 | 0,65 / 0,50 / 0,45 | 366 / 488 / 595 |
| `pf-k2` | 100 / 300 / 500 | 2,67 / 1,85 / 1,63 | 0,80 / 0,68 / 0,61 | 349 / 478 / 581 |
| `pf-k1k2` | 100 | 1,53 | 0,64 | 392 |

Trung bình trên 8 kịch bản × L {10, 20} × 2 h0 × 30 d, κ = 1 (960 đơn vị ô × d mỗi biến thể, cột `n_o_d` của
`c4_tai_nguyen.csv`; PF trung bình 3 seed PF trong mỗi d). Cột µs / bước / track: thời gian tường predict + update mỗi
bước của một track, trung bình trên 160 quỹ đạo mỗi biến thể (cột `n_quy_dao` của `results/phase2/kc48_mo_phong/toc_do.csv`),
một tiến trình, cùng máy. Mẫu đo thời gian **khác** mẫu e_r / F ở trên: 8 kịch bản × d = 0–4 × 2 h0 × L {10, 20}, κ = 1,
seed PF = 0 (một seed, không phải 30 d × 3 seed PF); một lượt làm nóng (kịch bản `thang`, d = 99) chạy trước và không
nằm trong 160 quỹ đạo (`src/kc48/thi_nghiem_mo_phong.py:684–704`, `giai_doan_do_toc_do`).
`pf-k1` N = 100 tốt hơn `pf` N = 500 ở cả e_r lẫn F, và nhanh hơn. Không biến thể PF nào đạt mức EKF. Hình: `phan_tich/hinh_tai_nguyen_va_ess.png`, `phan_tich/hinh_c1_e_r.png`.

### 5.5 Minh họa hạt theo track sở hữu (`results/phase2/kc48_mo_phong/hinh_hat_so_huu.png`, `…/hinh_hat_so_huu.json`; mô phỏng, không phải ảnh dataset)

Hai track PF bootstrap N = 100: xe lớn h = 72, xe nhỏ h = 24 đi lớp cạnh, xe nhỏ mất đo t = 30–41 và đổi lần khi bị
che. Đám mây của mỗi track tô màu riêng và không trộn lẫn. Sai tâm của xe nhỏ: 3,6 px ở t = 29, 18,3 px ở t = 35,
36,3 px ở t = 41; sau update t = 42 còn 25,0 px, độ phân tán đám mây 2·10⁻¹³ px (sụp về một particle). Đây là **một** ca
minh họa, không phải số thống kê.

## 6. Kết quả cấp pipeline — dev12 UA-DETRAC (`results/phase2/kc48_pf_dev/`, `results/trackeval/DETRAC-all/p2-kc48_pf_dev-*`)

### 6.1 Tùy chọn PF mới và tính đúng

- K1 `pf_optimal_proposal` (mặc định tắt): `src/pf_ctrv.py` (`propagate(..., return_prior)`, `optimal_proposal_step`,
  `update_particles(..., prior)`, `PFTrackView`) và `src/tracker_ctrv.py` (`PFSTrack._k1_prior`, khóa yaml). Hook chỉ đọc
  `last_resample_idx` và hai bộ đếm K1.
- K2 dùng khóa `pf_roughening` có sẵn. Đây là tùy chọn mới thứ hai được phép, nhưng R48D không thêm mã cho nó.
- `src/kc48/test_k1_k2.py`: 16/16 PASS, exit 0. Gồm: mặc định bit-identical với mã `eaf2e87` (bộ lọc và pipeline 30
  frame, roughening 0 và 0,5); trọng số K1 khớp công thức đa biến (sai < 1e-10); K1 với N = 40.000 khớp hậu nghiệm
  Kalman; prior dùng đúng một lần; K1 trong pipeline tái lập theo seed; hook không đổi kết quả.
- Trên dữ liệu thật: file tracking `ekf`, `pf100-s0`, `pf100-s1` của dev (mã có K1, K1 tắt) trùng sha256 với smoke R48C
  (mã `eaf2e87`) trên `MVI_20052`, `MVI_20062` — `results/phase2/kc48_pf_dev/phan_tich/mac_dinh_vs_smoke.json`, 6/6
  trùng (đơn vị: file tracking, 3 run × 2 video).
- Test cũ chạy lại sau khi thêm K1, đều exit 0 (số dạng a/b ở đây và ở mục 11 là số phép kiểm đạt / tổng phép kiểm của
  bộ test, theo stdout; không phải số run hay video): `src/kc48/test_pf_dung_dan.py` 29/29; `src/kc48/chay_test_pf_cu.py`
  (test 1–10 của `src/test_pf_ctrv.py`) 21/21; EKF 14/14; UKF 12/12; lifecycle 20/20; init 17/17. **Test 11 không chạy lại ở R48D**: nó
  ghi ngoài `write_paths` của R48D (`04` P2). Test 11 đã đạt ở R48C, trên mã trước khi thêm K1.

### 6.2 Chạy

26 run (`results/phase2/kc48_pf_dev/runs.csv`; run = một biến thể × một seed PF; EKF tất định nên một run mỗi biến thể; mỗi run trên 12 video dev12),
điều kiện A, 12 video:
- `ekf`, `ekf-chinh`: tất định.
- `pf100`, `pf100-k1`, `pf100-k2`, `pf100-k1k2`, `pf300`, `pf500`, `pf300-k1`, `pf500-k1`: mỗi biến thể 3 seed PF.

Lệnh:
- `python src/kc48/chay_song_song_dev.py --track` (5 tiến trình) rồi bước eval qua `src/kc48/chay_thi_nghiem_kc48.py`.
  Cả hai đều gọi `run_experiment.main`, chuyển gốc `PROCESSED_DIR`.
- `results/phase2/kc48_pf_dev/nhat_ky_wrapper.jsonl`: 26 dòng track (mỗi dòng một run) và 1 dòng eval, tất cả exit 0. Track 15:58:15 →
  16:08:10, eval 16:08:20 → 16:12:57. Không thư mục `p2-kc48_*` nào được tạo trong `data/processed`.
- Ngân sách 120 phút: dùng khoảng 15 phút. Mọi ưu tiên P1–P3 đều chạy, không phần nào bị chặn.
- Không ghi đè kết quả cũ: `python src/kc48/kiem_khong_ghi_de.py` exit 0
  (`results/phase2/kc48_pf_dev/kiem_khong_ghi_de.json`). Từ mốc khóa giao thức R48D (15:31:12) không file nào có mtime
  mới trong `results/trackeval/DETRAC-all` (365 file, trừ `p2-kc48_pf_dev-*`), `results/phase2` (2.388 file, trừ
  `kc48_mo_phong`, `kc48_pf_dev`, `kc48_audit/overlay`), `data/processed` (23.570), `data/interim` (609),
  `handoff/logs/review40`–`review47`. Đây là bằng chứng mtime, không phải hash trước / sau. Các số 365 / 2.388 / 23.570 /
  609 đếm file trong từng phạm vi đã nêu.

### 6.3 Bảng chính COMBINED (`phan_tich/bang_chinh.csv`; PF: COMBINED từng seed rồi trung bình 3 seed, [min–max giữa seed])

*Mẫu số*: COMBINED của TrackEval gộp trường thô của 12 video dev12 (điều kiện A) rồi tính chỉ số; IDSW là tổng
số lần đổi ID trên 12 video. Ngưỡng: HOTA, DetA, AssA là trung bình trên 19 ngưỡng alpha (mảng 19 phần tử, gộp theo
từng ngưỡng rồi lấy trung bình — `src/boot_stats.py:66–78`, `src/run_trackeval.py:191–205`); IDF1 (Identity), MOTA và
IDSW (CLEAR) dùng một ngưỡng IoU 0,5 (`"THRESHOLD": 0.5`, `src/run_trackeval.py:126`). `ekf`, `ekf-chinh`: 1 run. Biến thể PF: 3 seed.
[min–max giữa seed] chỉ in ở vài ô; mọi ô khác có ở các cột `*_min_seed`, `*_max_seed` của `bang_chinh.csv`.

Vòng sửa F3 (mục 11): trước đây PF lấy trung bình trường thô giữa seed rồi mới tính COMBINED. Lệch tối đa giữa hai cách:
HOTA 2,8·10⁻⁵, AssA 8,7·10⁻⁵. Ở 4 chữ số, chỉ ba ô AssA đổi: `pf100-k2`, `pf100-k1k2`, `pf300`.

| Biến thể | HOTA | AssA | IDF1 | IDSW | DetA | MOTA |
|:--|--:|--:|--:|--:|--:|--:|
| `ekf` | 0,5925 | 0,6838 | 0,7245 | 472 | 0,5207 | 0,5719 |
| `ekf-chinh` | 0,5931 | 0,6850 | 0,7258 | 476 | 0,5207 | 0,5719 |
| `pf100` | 0,4933 [0,4885–0,4969] | 0,5383 | 0,5509 | 2.069 [2.060–2.086] | 0,4692 | 0,4566 |
| `pf100-k1` | 0,5377 [0,5364–0,5388] | 0,6203 | 0,6334 | 1.094 | 0,4802 | 0,4978 |
| `pf100-k2` | 0,5148 [0,5130–0,5170] | 0,5670 | 0,5866 | 1.646 | 0,4810 | 0,4830 |
| `pf100-k1k2` | 0,5496 [0,5489–0,5501] | 0,6272 | 0,6505 | 980 | 0,4934 | 0,5258 |
| `pf300` | 0,5182 | 0,5763 | 0,5909 | 1.549 | 0,4809 | 0,4762 |
| `pf500` | 0,5269 | 0,5896 | 0,6056 | 1.352 | 0,4847 | 0,4838 |
| `pf300-k1` | 0,5505 | 0,6408 | 0,6517 | 914 | 0,4862 | 0,5047 |
| `pf500-k1` | 0,5548 [0,5528–0,5561] | 0,6469 | 0,6583 | 836 | 0,4883 | 0,5074 |

### 6.4 Bootstrap ghép cặp theo video (`phan_tich/doi_chieu_bootstrap.csv`; 12 video, B = 5000, seed 0)

Trong mỗi mẫu bootstrap: COMBINED riêng từng seed rồi trung bình (`src/kc48/boot_seed.py`, `04` mục 5.2; vòng sửa F3).
So với cách cũ, hiệu và CI lệch tối đa 1,4·10⁻⁴. Không đối chiếu nào đổi trạng thái "CI loại 0". Mọi số dưới đây giữ
nguyên ở 3 chữ số. *Mẫu số*: hiệu = biến thể − tham chiếu trên chỉ số COMBINED của 12 video dev12; mẫu bootstrap lấy lại
video (12) có hoàn lại, B = 5000, seed 0; hiệu IDSW tính bằng số lần đổi ID tổng trên các video của mẫu.

| Đối chiếu | Hiệu HOTA [CI 95 %] | Hiệu IDF1 [CI] | Hiệu IDSW [CI] |
|:--|:--|:--|:--|
| P1 `pf100` − `ekf` | −0,099 [−0,178; −0,050] | −0,174 [−0,292; −0,085] | +1.597 [+719; +2.555] |
| P2 `pf100-k1` − `pf100` | +0,044 [+0,018; +0,087] | +0,083 [+0,034; +0,146] | −975 [−1.567; −434] |
| P2 `pf100-k2` − `pf100` | +0,021 [+0,008; +0,045] | +0,036 [+0,014; +0,063] | −423 [−657; −202] |
| P2 `pf100-k1k2` − `pf100` | +0,056 [+0,023; +0,109] | +0,100 [+0,039; +0,179] | −1.089 [−1.747; −489] |
| P2 `ekf-chinh` − `ekf` | +0,001 [−0,001; +0,001] | +0,001 [−0,001; +0,003] | +4 [−7; +16] |
| P3 `pf100-k1` − `ekf` | −0,055 [−0,093; −0,031] | −0,091 [−0,150; −0,051] | +622 [+277; +995] |
| P3 `pf500-k1` − `ekf` | −0,038 [−0,065; −0,021] | −0,066 [−0,109; −0,037] | +364 [+148; +604] |
| P3 `pf300-k1` − `pf300` | +0,032 [+0,014; +0,063] | +0,061 [+0,025; +0,114] | −635 [−1.054; −259] |
| P3 `pf500-k1` − `pf500` | +0,028 [+0,011; +0,055] | +0,053 [+0,018; +0,106] | −516 [−859; −204] |

Kết quả bất lợi, nêu rõ: **mọi** biến thể PF, kể cả `pf500-k1`, vẫn kém EKF, CI loại 0. K1 và K2 cải thiện PF có ý nghĩa
trên dev12 nhưng không lấp được khoảng cách. EKF đã chỉnh không khác EKF mặc định. Phần lớn khoảng cách nằm ở AssA / IDSW;
DetA cũng thấp hơn (0,469–0,493 so với 0,521; khoảng trên 8 biến thể PF của bảng 6.3 so với `ekf`, COMBINED dev12).

### 6.5 Phân tầng định trước (`phan_tich/phan_tang.csv`, `phan_tich/giu_id_theo_track.csv`; 1.185 track GT, số frame khớp XML 1.185/1.185)

Tỉ lệ giữ ID: số frame ghép với ID chính / số frame của track GT. Ghép GT–tracker từng frame bằng
`occlusion_eval.match_frame`: cặp IoU < 0,5 bị loại **trước** Hungarian (vòng sửa F1, mục 11). Hiệu so với `ekf`, CI
bootstrap theo video:

*Mẫu số* (`src/kc48/phan_tich_dev.py`, hàm `phan_tang`): đơn vị là track GT của dev12 (1.185 track trong `gt.txt`); cột
`ekf` = trung bình không trọng số của tỉ lệ giữ ID trên các track của tầng; PF lấy trung bình 3 seed trong từng track
trước. Hiệu = trung bình trên track của (biến thể − `ekf`); CI: lấy lại video có hoàn lại, B = 5000, rng seed 0, chỉ tầng
có ≥ 5 video. Tầng kích thước: tam phân vị của trung vị h theo track trên mọi track dev12; tầng giữa (`vua`) có trong
`phan_tang.csv`, không đưa vào bảng. Tầng cua / thẳng theo `loai_max` (`03` mục 8); track `cong_nhe` và track bị loại
(`loai_ngan`, `loai_cham`) không có dòng trong bảng. Cột "n track / video" là số track GT / số video có track của tầng.

| Tầng | n track / video | `ekf` | `pf100` hiệu [CI] | `pf100-k1` hiệu [CI] | `pf500-k1` hiệu [CI] |
|:--|:--|--:|:--|:--|:--|
| car | 1.021 / 12 | 0,585 | −0,238 [−0,284; −0,172] | −0,123 [−0,150; −0,084] | −0,090 [−0,110; −0,060] |
| van | 138 / 11 | 0,613 | −0,252 [−0,297; −0,169] | −0,148 [−0,182; −0,093] | −0,101 [−0,140; −0,038] |
| bus | 14 / 8 | 0,619 | −0,096 [−0,153; +0,001] | −0,063 [−0,101; −0,016] | −0,068 [−0,113; −0,012] |
| others | 12 / 6 | 0,466 | −0,142 [−0,241; −0,047] | −0,014 [−0,046; +0,015] | −0,002 [−0,030; +0,014] |
| nhỏ (tam phân vị h) | 396 / 10 | 0,548 | −0,238 [−0,346; −0,139] | −0,105 [−0,161; −0,058] | −0,059 [−0,104; −0,024] |
| lớn | 395 / 12 | 0,640 | −0,243 [−0,282; −0,174] | −0,151 [−0,176; −0,116] | −0,124 [−0,145; −0,096] |
| cua (`loai_max`) | 32 / 5 | 0,651 | −0,057 [−0,072; −0,037] | −0,040 [−0,050; −0,019] | −0,023 [−0,036; −0,007] |
| thẳng | 617 / 12 | 0,617 | −0,285 [−0,353; −0,207] | −0,128 [−0,168; −0,089] | −0,085 [−0,115; −0,057] |
| cua (`loai_p90`, độ nhạy) | 8 / 4 | 0,569 | < 5 video: chỉ mô tả (`pf100` 0,508) | (0,548) | (0,562) |

Ở nhóm cua, PF kém EKF ít hơn so với nhóm thẳng. Tuy vậy, nhóm cua chỉ có 32 track / 5 video, và PF vẫn kém EKF ở
mọi tầng có suy luận. [GT] Khoảng cách nhỏ hơn có thể do EKF cũng khó ở đoạn cua, hoặc do cỡ mẫu; chưa kiểm. **Không**
kết luận "PF tốt khi cua".

### 6.6 Tầng sự kiện (`results/phase2/kc48_pf_dev/su_kien/`; lệnh `python src/kc48/su_kien_dev.py --phan cham`, rồi `--phan tong_hop`, exit 0)

Cỡ mẫu và trường hợp loại (`co_mau_va_loai.csv`):
- Che khuất (occlusion_eval, ngữ cảnh 30 frame): 717 sự kiện; loại 369 `gt_missing_before`, 164 `gt_missing_after`.
  Còn **184 đủ điều kiện / 12 video / 137 track**, trong đó 33 sự kiện có đoạn che ≥ 0,9 (`has_full`).
  *Mẫu số*: đơn vị sự kiện che khuất do `src/occlusion_eval.py` dựng từ GT dev12; 717 là mọi sự kiện trước khi loại.
- Xe nhỏ cạnh xe lớn: dựng lại khớp **88 / 65 / 9** (sự kiện / track nhỏ / video;
  `results/phase2/kc48_pf_dev/su_kien/kiem_su_kien_xe_nho.json`; 0 frame sự kiện thiếu trong `gt.txt`).
  Loại 32 + 14 sự kiện (32 `gt_missing_before`, 14 `gt_missing_after`, cột `loai_*` của `co_mau_va_loai.csv`) ⇒ còn
  **42 đủ điều kiện / 7 video / 29 track**.

Tỉ lệ giữ ID (ID trước = ID sau) trên mọi sự kiện đủ điều kiện; PF trung bình 3 seed (`bang_su_kien.csv`). *Mẫu số*
(`src/kc48/su_kien_dev.py`, hàm `tong_hop`): đơn vị sự kiện đủ điều kiện; mỗi sự kiện có kết cục giữ / không giữ, PF lấy
trung bình 3 seed trong từng sự kiện; tỉ lệ = trung bình không trọng số trên sự kiện của tầng. Hiệu = trung bình trên sự
kiện của (biến thể − `ekf`); CI: lấy lại video có hoàn lại, B = 5000, rng seed 0, chỉ khi tầng có ≥ 5 video:

| Tầng | `ekf` | `pf100` hiệu [CI] | `pf100-k1` hiệu [CI] | `pf500-k1` hiệu [CI] |
|:--|--:|:--|:--|:--|
| Che khuất (184) | 0,489 | −0,196 [−0,338; −0,090] | −0,098 [−0,169; −0,050] | −0,078 [−0,141; −0,038] |
| Che ≥ 0,9 (33 / 6 video) | 0,061 | −0,040 [−0,444; 0,000] | −0,030 [−0,375; 0,000] | −0,030 [−0,375; 0,000] |
| Xe nhỏ cạnh xe lớn (42 / 7 video) | 0,548 | −0,135 [−0,243; −0,032] | −0,024 [−0,037; 0,000] | −0,016 [−0,028; 0,000] |

Theo đúng chữ của `04` mục 5.3 (chỉ sự kiện ghép được trước, ở cả hai biến thể): xe nhỏ `ekf` 0,852 (27 sự kiện).
Hiệu `pf100` −0,218 [−0,374; −0,059] trên 26 sự kiện ghép cặp; `pf100-k1` −0,037 [−0,065; 0,000] trên 27 sự kiện
(cột `n_su_kien_co_truoc_ca_hai`). *Mẫu số*: trong 42 sự kiện xe nhỏ đủ điều kiện, chỉ giữ sự kiện mà track GT nhỏ được
ghép trước sự kiện ở `ekf` và ở **mọi** seed của biến thể PF.

Diễn giải:
- Ở tầng che ≥ 0,9, mọi bộ lọc gần như đều mất ID: EKF giữ được 2/33 (sự kiện `has_full` đủ điều kiện). Khớp với kết luận cũ rằng `track_buffer` 30 frame
  là trần cứng. CI ở tầng này rộng và chạm 0.
- Với xe nhỏ cạnh xe lớn, K1 thu hẹp khoảng cách với EKF về mức không phân biệt được trên 42 sự kiện (CI chạm 0). Cỡ mẫu
  rất nhỏ; đây **không** phải bằng chứng PF + K1 ngang EKF.

## 7. Trả lời Q1–Q3 của giao thức (trong phạm vi đã đo)

- **Q1**: Khi biết liên kết, PF–CTRV hiện tại (N = 100) có sai số dự đoán và tỉ lệ thất bại cao hơn EKF–CTRV một cách nhất
  quán, ở cả nhóm đúng mô hình lẫn lệch mô hình. Không có nhóm chuyển động nào trong 8 kịch bản mà PF tốt hơn.
- **Q2**: K1 giảm nhất quán triệu chứng nghèo hạt khi đang bám: ESS/N +0,29 (C2, hiệu TB trên 32 ô, mục 5.2), ít resample hơn, nhiều tổ tiên hơn. Nó cũng
  giảm F và e_r. K1 **không** sửa được sụp ESS sau mất đo dài (ESS/N hợp lệ tại t_r vẫn khoảng 1 %), nhưng ít quỹ đạo rơi
  vào nhánh tái khởi tạo tại t_r hơn (ô `cua_deu` L 20 h0 24: 3/90 so với 43/90). K2 (c = 1) giảm e_r nhưng không giảm F
  một cách nhất quán, và không phân biệt được với PF gốc về ESS/N tại t_r.
- **Q3**: Trên dev12, chiều của K1 / K2 khớp với mô phỏng: K1 > K2 > PF gốc; HOTA, IDF1, IDSW cải thiện có CI loại 0.
  Khoảng cách với EKF vẫn còn ở mọi tầng có suy luận.
- Giả thuyết chưa kiểm còn lại: H2 (association có bị ảnh hưởng thêm bởi rung Monte Carlo của box), H3 (likelihood trên
  sai số detector thật), và vì sao DetA của PF thấp hơn.

## 8. Giới hạn

- Dev12 chỉ có 12 video và từng dùng trong thiết kế mốc 2. Các tầng cua, bus, xe nhỏ có cỡ mẫu vài chục track / sự kiện.
  CI theo video với 5–12 cụm (video có mặt trong tầng) là xấp xỉ. Không suy rộng sang UA-DETRAC test hay BDD (không chạy ở lượt này).
- Mô phỏng nằm trên mặt phẳng ảnh; 4 kịch bản lệch mô hình không đại diện mọi cảnh thật. Kịch bản `nhieu_duoi_day` chỉ
  là một dạng likelihood sai.
- Khác biệt pipeline gộp cả heuristic triển khai khác nhau giữa PF và EKF (`02` mục 5), không riêng cách biểu diễn hậu
  nghiệm.
- K1 dùng Q, R chéo và phép chặn h ≥ 10⁻³ như mã; công thức đúng cho mô hình đó, không cho mô hình có tương quan.
- Thời gian đo trong harness Python có bật chẩn đoán, không phải FPS end-to-end; chưa đo trên Jetson Nano (đã hoãn).
- Hướng xử lý 572 ảnh dev12 không có `<frame>` của evaluator chưa kiểm; mọi biến thể chấm trên cùng GT. (Đơn vị: file
  ảnh của 12 video dev12 có trên đĩa nhưng XML không có `<frame>`; 572 = 16.919 ảnh − 16.347 frame, `03` mục 6.2.)
- Test 11 (runner seed) chưa chạy lại sau khi thêm K1 (mục 6.1).
- Tái tạo mốc 2 tại HEAD mới kiểm EKF điều kiện A trên 2/12 video (`05b`).
- M5 chỉ có thiết kế (`06_thiet_ke_augmentation.md`), không có số huấn luyện nào.

## 9. Hình điển hình

| Hình | Nội dung | Nguồn |
|:--|:--|:--|
| `results/phase2/kc48_mo_phong/hinh_hat_so_huu.png` | hạt tô màu theo track sở hữu; xe nhỏ bị xe lớn che 12 bước | mô phỏng (mục 5.5) |
| `results/phase2/kc48_mo_phong/phan_tich/hinh_c1_e_r.png` | e_r PF − EKF theo kịch bản | mục 5.2 |
| `results/phase2/kc48_mo_phong/phan_tich/hinh_tai_nguyen_va_ess.png` | e_r / F theo N và ESS | mục 5.4 |
| `results/phase2/kc48_audit/bieu_do_phan_bo_lop_kich_thuoc.png`, `bieu_do_quy_dao_cua.png` | phân bố lớp / cỡ; quỹ đạo cua | audit R48B |

Overlay trên ảnh dataset chỉ nằm ở `results/phase2/kc48_audit/overlay/`, không commit vì repo public. R48D không tạo
overlay mới.

## 10. Truy nguồn

| Số liệu | File | Lệnh / run | Cấu hình |
|:--|:--|:--|:--|
| Mục 5 | `results/phase2/kc48_mo_phong/{chon.json, phan_tich/*, toc_do.csv, hinh_hat_so_huu.*}` | `src/kc48/thi_nghiem_mo_phong.py` (nhật ký `nhat_ky.jsonl`) | `src/kc48/mo_phong_quy_dao.py` (`--tu-kiem` 10/10, R48C) |
| Mục 6.1 | `src/kc48/test_k1_k2.py` (stdout), `results/phase2/kc48_pf_dev/phan_tich/mac_dinh_vs_smoke.json` | lệnh test ở mục 6.1 và mục 11 của file này; `04` phụ lục P2 (R49A: bản trước trỏ `05` mục 6.1, nhưng `05` là báo cáo R48C, mục 6 của nó là quan sát EKF, không có lệnh test K1) | — |
| Mục 6.3–6.5 | `results/phase2/kc48_pf_dev/phan_tich/*.csv` | `src/kc48/phan_tich_dev.py` (exit 0) | `configs/phase2/experiments/kc48_pf_dev.yaml`, `configs/phase2/trackers/kc48/*.yaml`, `configs/bytetrack_ekf_ctrv.yaml`, `configs/phase2/trackers/bytetrack_pf_ctrv_n{100,300,500}.yaml` |
| Mục 6.6 | `results/phase2/kc48_pf_dev/su_kien/*.csv`, `results/phase2/kc48_pf_dev/su_kien/kiem_su_kien_xe_nho.json` | `src/kc48/su_kien_dev.py` | như trên |
| TrackEval | `results/trackeval/DETRAC-all/p2-kc48_pf_dev-A-*` | wrapper eval, `results/phase2/kc48_pf_dev/nhat_ky_wrapper.jsonl` dòng 27 | cache `yolov8n-coco_conf0.05_all`, xác thực ở R48C (`results/phase2/kc48_pf_code/cache_dev.json`) |
| Không ghi đè (mục 6.2) | `results/phase2/kc48_pf_dev/kiem_khong_ghi_de.json` | `src/kc48/kiem_khong_ghi_de.py` (exit 0) | — |
| Slide | `handoff/logs/review48/08_de_cuong_slide.md` (chỉ dẫn lại số của file này) | — | — |
| Dataset | UA-DETRAC Train XML (dev12 theo `results/phase2/manifests/detrac_dev.json`) | — | — |
| Vòng sửa F2 (mục 5.2–5.3) | `results/phase2/kc48_mo_phong/chan_doan_v2/{kiem_ban_cu.json, so_sanh_ban_cu.json, danh_gia/*.csv}`, `phan_tich/mau_so_chan_doan.csv` | `src/kc48/sua_chan_doan_mo_phong.py --buoc {kiem, chay-lai, so-sanh}`, rồi `thi_nghiem_mo_phong.py --giai-doan phan-tich` | cùng `chon.json`, seed như mục 5 |
| Vòng sửa F1, F3 (mục 6.3–6.5) | `results/phase2/kc48_pf_dev/phan_tich/*.csv` | `src/kc48/phan_tich_dev.py` (exit 0) | như mục 6.3–6.5 |
| So trước / sau vòng sửa | `results/phase2/kc48_pf_dev/phan_tich/so_sanh_truoc_sau_vong_sua.json`, `results/phase2/kc48_mo_phong/phan_tich/so_sanh_truoc_sau_vong_sua.json`; bản cũ ở `results/phase2/kc48_pf_dev/phan_tich/_ban_cu_truoc_vong_sua/` và `results/phase2/kc48_mo_phong/_ban_cu_truoc_vong_sua/` (R49A: sửa đường dẫn dev thiếu `phan_tich/`, theo Codex `nghiem_thu_r48cde/0009_review_answer.json`) | `src/kc48/so_sanh_truoc_sau_sua.py` | — |

## 11. Vòng sửa sau cổng D vòng 1 (Codex revise F1–F3, 2026-10-11)

Nguồn: `D:\LABOTARY\automation\runs\bytetrack_r48_tiep_effort_high_20261011_1507\reports\review_003.json`. Không chạy lại
tracking, không đổi mã tracker (`src/pf_ctrv.py`, `src/tracker_ctrv.py`), không sửa `src/boot_stats.py` hay evaluator.
Trước khi ghi đè, đầu ra cũ được sao nguyên byte (`src/kc48/luu_ban_cu.py`). Bản sao nằm ở
`results/phase2/kc48_pf_dev/phan_tich/_ban_cu_truoc_vong_sua/` và `results/phase2/kc48_mo_phong/_ban_cu_truoc_vong_sua/`;
`manifest.json` ghi sha256 nguồn = bản sao.

**F1 — ghép phân tầng.** Lỗi: `phan_tich_dev.giu_id` giải Hungarian trên mọi IoU rồi mới loại cặp < 0,5. Cách này có thể
mất cặp hợp lệ hoặc đổi cặp. Sửa: `ghep_frame` gọi `occlusion_eval.match_frame` (cặp dưới ngưỡng bị phạt BIG_M trước
Hungarian), không sửa evaluator. Bằng chứng (`src/kc48/test_sua_r48d.py`):
- Phản ví dụ tổng hợp IoU [[0,60; 0,45]; [0,45; 0,10]]: cách cũ cho 0 cặp, cách mới giữ (0, 0). `giu_id` trên bảng 3
  frame dựng từ phản ví dụ này cho GT 1 giữ ID 3/3 frame.
- Ca thật MVI_20052, run `ekf`, frame 445: cách cũ ghép GT 38 với tracker 350 (IoU 0,72466); cách mới ghép với 290
  (IoU 0,74063). Mọi cặp mới có IoU ≥ 0,5, và số cặp không giảm (10 / 10; đơn vị cặp GT–tracker của frame này, cách mới /
  cách cũ).
- Tính lại từ file tracking đã có: 167 / 30.810 dòng (track GT × run) đổi `ty_le_giu_id` hoặc `n_doi_id`. Tầng lệch tối
  đa 0,0005 (tỉ lệ giữ ID), 0,0018 (CI). Không tầng nào đổi trạng thái CI. Số đổi trong bảng 6.5: bus (`pf100` CI trên
  0,000 → +0,001; `pf100-k1` CI dưới −0,017 → −0,016), nhỏ (`pf100-k1` −0,104 → −0,105), lớn (`pf100` −0,242 → −0,243).
  Tầng sự kiện (mục 6.6) vốn dùng `occlusion_eval.match_video`, nên không bị lỗi này và không tính lại.

**F2 — tái khởi tạo trộn vào ESS.** Lỗi: nhánh suy biến của `PFCTRV.update_particles` (`src/pf_ctrv.py:308–322`) ghi
`ess_ratio = 0`, `resampled = True` theo quy ước, dù đây không phải ESS thật hay systematic resampling. Harness cũ đưa
thẳng hai trường này vào ESS/N và tỉ lệ resample. Ngoài ra `n_deg` là bộ đếm tích lũy của cả một đơn vị, không phải số
của từng quỹ đạo. Sửa harness `thi_nghiem_mo_phong.chay_mot`; không đổi `src/pf_ctrv.py`:
- Cờ `reinit_tr`; ESS và resample tại t_r = NaN khi tái khởi tạo.
- Khi bám: `n_upd_bam` (số update có đo khi bám), `n_reinit_bam` (đếm riêng). Mẫu số của `ess_bam` và `rs_bam` là
  `n_upd_bam` − `n_reinit_bam` (update hợp lệ); `tt_bam` (tỉ lệ tổ tiên) chỉ dùng các update có systematic resampling.
  (R49A: làm rõ theo giới hạn của Codex `nghiem_thu_r48cde/0009_review_answer.json`; bản trước ghi `n_upd_bam` là mẫu số.)
- `n_deg`, `n_update` tính theo từng quỹ đạo.
- Đối chiếu C2 / C3 cho chẩn đoán chỉ dùng các d có giá trị hợp lệ ở cả hai biến thể; `n_d` ghi số d dùng.
- Chuỗi minh họa có cột `tai_khoi_tao`; trên hình ESS, điểm tái khởi tạo vẽ dấu ×.

Dữ liệu cũ đủ thông tin cho 91.445 / 103.680 quỹ đạo PF: chúng không có tái khởi tạo ngoài t_r, tại t_r tách được bằng
`deg_tr`, còn `n_deg` theo quỹ đạo lấy bằng hiệu liên tiếp của bộ đếm tích lũy (`chan_doan_v2/kiem_ban_cu.json`; hiệu âm:
0). Với 12.235 quỹ đạo có tái khởi tạo ngoài t_r, `ess_bam` / `rs_bam` cũ có thể đã trộn hai cơ chế. Các quỹ đạo này được
chạy lại bằng harness đã sửa, cùng cấu hình / seed. Chạy thêm 1.838 quỹ đạo đối chứng (mỗi dòng thứ 50 trong số không cần
chạy lại). Tổng 14.073 quỹ đạo, khoảng 14 % số quỹ đạo PF của giai đoạn đánh giá. Chạy hết 169 s với 6 tiến trình
(`results/phase2/kc48_mo_phong/nhat_ky.jsonl`, giai đoạn `chan-doan-v2`), trong ngân sách mục 9 của `04`. *Mẫu số*:
103.680 là mọi quỹ đạo PF của giai đoạn đánh giá (d 0–29, 3 seed PF) theo `chan_doan_v2/kiem_ban_cu.json`; cách tách
theo biến thể / N / ô chưa ghi trong file này.
`results/phase2/kc48_mo_phong/chan_doan_v2/so_sanh_ban_cu.json`: 80/80 đơn vị đạt (đơn vị = một tổ hợp biến thể PF × N ×
kịch bản, ứng với một file `chan_doan_v2/danh_gia/<biến thể>_N<N>_<kịch bản>.csv`):
- 12 cột đầu ra bộ lọc của mọi dòng chạy lại trùng khít bản cũ (e_r, IoU, F, RMSE, NIS, độ phân tán…). Nghĩa là chẩn đoán
  không đổi đầu ra bộ lọc.
- `n_deg` mới bằng hiệu bộ đếm cũ.
- Đối chứng: `ess_bam` / `rs_bam` / `tt_bam` trùng; mẫu số tính từ mặt nạ đo bằng mẫu số của harness.
- Mọi ESS/N hợp lệ ≥ 1/N.

Test: ép nhánh tái khởi tạo ở mọi update (ESS / resample = NaN, `n_reinit_bam = n_upd_bam`); `n_deg` của quỹ đạo thứ hai
không cộng dồn quỹ đạo thứ nhất. Bật / tắt `enable_diag` trên cùng chuỗi có tái khởi tạo ép cho particle, logw, mean, cov
trùng khít ở mọi bước (4 tổ hợp K1 / K2). Trên quỹ đạo có cả hai cơ chế (26 / 79 update tái khởi tạo; đơn vị: update có
đo của một quỹ đạo dùng trong test; file ghi số này: chưa xác định trong nguồn, đã tra `src/kc48/test_sua_r48d.py`), ESS/N hợp lệ nhỏ nhất 0,0111 ≥ 1/N.

Ảnh hưởng (`results/phase2/kc48_mo_phong/phan_tich/so_sanh_truoc_sau_vong_sua.json`):
- e_r, F, RMSE, NIS, độ phân tán, H6 và C4 (e_r, F, thời gian): không đổi (lệch ≤ 2·10⁻⁶, sai số làm tròn).
- C2 ESS/N tại t_r: 31 → 27 ô CI > 0 (trên 32 ô), vẫn "nhất quán", hiệu TB +0,007 → +0,005.
- C3 ESS/N tại t_r: 10 → 0 ô CI > 0 (trên 32 ô). Mức tăng ESS mà K2 có vẻ tạo ra trước đây là do trộn tái khởi tạo.
- ESS/N khi bám, tỉ lệ resample khi bám: lệch ≤ 0,0009, không đổi kết luận.
- Tỉ lệ tái khởi tạo trên mọi update có đo (mọi ô): `pf` N 100 / 300 / 500 là 0,99 % / 0,58 % / 0,44 %; `pf-k1` là
  0,03 % / 0,01 % / 0,01 %; `pf-k2` N 100 là 0,56 %. *Mẫu số*: tổng số update có đo của mọi quỹ đạo đã chạy của biến thể
  và N đó, gộp mọi ô (khóa `tong_tai_khoi_tao_theo_bien_the`, cột `n_update`, `n_tai_khoi_tao`, `ty_le`; ví dụ `pf` N 100:
  13.689 / 1.386.720 update trên 17.280 quỹ đạo).

**F3 — thứ tự trung bình seed.** Lỗi: `phan_tich_dev` lấy trung bình trường thô giữa seed (`boot_stats.average_seeds`)
rồi mới tính COMBINED. `04` mục 5.2 yêu cầu COMBINED rồi trung bình, trong từng mẫu bootstrap. HOTA, IDF1, MOTA phi tuyến
theo trường thô, nên hai cách cho số khác nhau. Sửa: thêm `src/kc48/boot_seed.py`, không sửa `src/boot_stats.py`. Dùng
cùng dãy rng với `paired_bootstrap`. Test:
- Hai seed tổng hợp có thống kê khác nhau: HOTA 0,384773 (mới) so với 0,358059 (cũ), IDF1 0,512724 so với 0,524407. IDSW
  tuyến tính nên bằng nhau.
- Từng mẫu bootstrap khớp phép tính tay (sai 0).
- Với một seed mỗi bên, kết quả trùng khít `boot_stats.paired_bootstrap`.
- Số thật `pf500-k1`: HOTA 0,5547572114 (mới) so với 0,5547515202 (cũ), trùng số Codex nêu.

Bảng chính và CI tính lại từ `raw_fields.json` có sẵn: lệch tối đa HOTA 2,8·10⁻⁵, AssA 8,7·10⁻⁵, CI 1,4·10⁻⁴. Không đối
chiếu nào đổi trạng thái "CI loại 0".

**Lệnh và kết quả (2026-10-11, đều exit 0):**
- `python src/kc48/test_sua_r48d.py`: 29/29.
- `python src/kc48/sua_chan_doan_mo_phong.py --buoc kiem`, rồi `--buoc chay-lai --toi-da-giay 420` (80 đơn vị, 169 s),
  rồi `--buoc so-sanh` (`tat_ca_dat: true`).
- `python src/kc48/thi_nghiem_mo_phong.py --giai-doan phan-tich`: chuỗi minh họa có `e_pred_h` trùng khít bản cũ.
- `python src/kc48/phan_tich_dev.py`: mặc định trùng smoke `eaf2e87`; 1.185 / 1.185 track GT khớp XML.
- `python src/kc48/so_sanh_truoc_sau_sua.py`.
- Hồi quy: `src/kc48/test_k1_k2.py` 16/16, `src/kc48/test_pf_dung_dan.py` 29/29.
- `python src/kc48/kiem_khong_ghi_de.py`: 0 file cũ có mtime sau mốc khóa. Đây là bằng chứng mtime, không phải hash.
- Sau khi thêm phụ lục P3 vào `04`: `python src/kc48/thi_nghiem_mo_phong.py --giai-doan chup-giao-thuc` ghi lúc
  20:43:58: sha256 file `b34ee6d9…`, 22.510 byte đầu vẫn `7063c532…`, tức mục 1–12 không đổi
  (`results/phase2/kc48_mo_phong/giao_thuc_sha256_r48d.jsonl`).

## 12. Phụ lục: chẩn đoán PF dev mốc 2 sau F2 (bước P9a8ee8bf, 2026-10-11; chi tiết `05c_kiem_pf_diag_moc2.md`)

Câu hỏi: `src/pf_diag_report.py` của mốc 2 có trộn dòng tái khởi tạo vào ESS/N và tỉ lệ resample như lỗi F2 không.

- Về công thức: có. Script lọc `kind == "update"` nhưng không loại `degenerate`.
- Về số liệu mốc 2: không, trong phạm vi 24 run có `pf_diag`. Con số 24 run / 456 CSV là kết quả kiểm kê mọi thư mục
  `pf_diag` dưới `data/processed` và `results`.
  - *Mẫu số* (`05c` mục 1–2): run = một tổ hợp điều kiện (`c10` / `c25`, nhãn run mốc 2; ý nghĩa nhãn chưa xác định
    trong nguồn, đã tra `05c`) × N × seed PF; CSV = file `pf_diag` của một run trên một video; update = dòng
    `kind == "update"` trong `pf_diag`, tổng qua 3 seed.
  - Dev: 18 run `expA_detrac_dev` (12 video dev12; `c10` / `c25` × N 100 / 300 / 500 × seed 0 / 1 / 2), 216 CSV, 0 / 1.664.036 update rơi vào nhánh tái khởi tạo. Thống kê tách tái khởi tạo
    trùng số cũ, lệch ≤ 1,1·10⁻¹⁶. Bảng dev không đổi sau khi sửa F1 (sha256 bảng trước / sau giống nhau).
  - Test UA-DETRAC: 6 run `expA_detrac_test` (N = 500; 40 video test; `c10` / `c25` × seed 0 / 1 / 2), 240 CSV, 0 / 3.400.029 update. `pf_diag_report` chưa từng chạy
    trên test nên không có số cũ để đối chiếu. ESS/N hợp lệ 0,317 (c10) / 0,318 (c25) chỉ để báo cáo, không dùng chọn
    tham số. (ESS/N đo trước resample trên update hợp lệ; trung bình giữa 3 seed của trung bình từng run, `05c` mục 2b,
    `theo_tracker_test.csv`.)
  - maha2_best hợp lệ lớn nhất theo run (giá trị lớn nhất trong các update hợp lệ của mỗi run; khoảng là min–max giữa
    các run của dev, của test): dev 3.382–8.440, test 3.724–4.561, đều dưới 10⁴. Kết quả nằm trong
    `results/phase2/kc48_pf_code/ess_moc2_tach_tai_khoi_tao/`; bảng test có hậu tố `_test`.
- Đính chính (F1): bản đầu của phụ lục này và của `05c` chỉ quét dev. Bản đó viết sai rằng không có `pf_diag` ở split
  khác và các run PF khác không bật `pf_diag`. Thực tế 6 run test có `pf_diag`; nay đã kiểm, cũng 0 tái khởi tạo.
- Các số ESS mốc 2 mà `01`, `02` và `04` trích dẫn (đều là dev) vẫn giữ nguyên. Không file kết quả cũ nào đổi: sha256
  của 456 CSV, 24 `run_meta.json` và các báo cáo cũ giống nhau trước và sau. Kết luận không áp dụng cho run PF không có
  log `pf_diag`.
