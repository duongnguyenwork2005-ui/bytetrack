# R48D / M6 — Đề cương slide báo cáo 17/10/2026

Viết ngày 2026-10-11. Mọi số trên slide lấy từ `07_tong_hop_bao_cao_17_10.md`; mỗi slide ghi mục của 07 và file nguồn.
Không thêm số ngoài 07. Nhãn giữ như 07: [LT] lý thuyết có nguồn, [QS] quan sát đo được, [GT] giả thuyết chưa kiểm.
Phần chưa chạy ghi là chưa chạy, kể cả trên slide.

Thời lượng gợi ý: 15–20 phút, 14 slide chính và 3 slide dự phòng.

## Phần A — Bối cảnh và lý thuyết

**Slide 1. Câu hỏi của đợt này**
- Vì sao PF–CTRV kém EKF–CTRV ở mốc 2, và hai kỹ thuật PF có cơ sở có thu hẹp được khoảng cách không?
- Thứ tự làm: lý thuyết → audit dữ liệu và mã → khóa giao thức → thí nghiệm.
- Nguồn: `handoff/KE_HOACH_BAO_CAO_20261017.md`; kết quả mốc 2 kế thừa: `00_tiep_nhan_va_ke_thua.md`.

**Slide 2. KF–CV, EKF–CTRV, PF–CTRV trong ByteTrack** [LT]
- Trạng thái XYAH 8-D của ByteTrack; CTRV có θ, ω; đơn vị theo bước.
- Nguồn: ByteTrack arXiv 2110.06864v3 (đã đối chiếu toàn văn); `01` mục 2, 3, 5.

**Slide 3. SIS / SIR, ESS, resample hệ thống** [LT]
- w ∝ w · p(z | x); ESS = 1 / Σw²; resample khi ESS/N < 0,5; log-sum-exp.
- Ghi trên slide: các ý lấy từ Arulampalam et al. 2002 **chưa đối chiếu toàn văn**.
- Nguồn: `01` mục 5, 7; 07 mục 1.

**Slide 4. Hai kỹ thuật thử: K1 và K2** [LT]
- K1: proposal tối ưu, đóng dạng khi Q, R chéo và H tuyến tính; w_k ∝ w_{k−1} · N(z; H f(x), H Q Hᵀ + R).
- K2: roughening, **có sẵn trong mã từ trước**.
- Không cái nào là đóng góp lý thuyết mới.
- Nguồn: `04` mục 6.1; 07 mục 1, 6.1.

## Phần B — Audit dữ liệu và mã

**Slide 5. Audit dữ liệu: các mốc đã xác minh** [QS]
- Bảng 07 mục 2: mốc slide lần 2 khớp khi đếm lại; 458.183 ảnh không lỗi toàn vẹn; nhãn BDD train khớp manifest nhập
  (chưa xác thực với archive gốc).
- Rủi ro: 70/139 video dev BDD chung tiền tố tên với train-fit; lệch miền kích thước bus.
- Pixel ảnh tô xám (R48E, chỉ 160 ảnh mẫu): 131 đạt / 26 chưa kết luận / 3 không đạt ngưỡng MAE ở box chồng vùng
  (hậu kiểm: phù hợp với mất mát do nén lại JPEG; chưa chứng minh nguyên nhân); box có nhãn không bị tô (`03` mục 14).
- Nguồn: `03`; `results/phase2/kc48_audit/moc_doi_chieu.csv`.

**Slide 6. Đối chiếu mã PF** [QS]
- Mỗi track có particle, log-trọng số, rng riêng ⇒ "xe lớn hút hạt của xe nhỏ" (H1) không áp dụng cho mã này.
- Hình: `results/phase2/kc48_mo_phong/hinh_hat_so_huu.png` (mô phỏng, một ca minh họa).
- Nguồn: `02`; `results/phase2/kc48_pf_code/so_huu_hat.json`; 07 mục 3, 5.5.

## Phần C — Giao thức

**Slide 7. Giao thức khóa trước khi so sánh**
- Hai cấp: mô phỏng biết liên kết (8 kịch bản, 4 kịch bản lệch mô hình) và pipeline dev12 UA-DETRAC.
- 30 seed nhiễu ghép cặp, 3 seed PF; bootstrap theo video; quy tắc "khác biệt nhất quán".
- Khóa bằng sha256 `7063c532…` (mục 1–12); phụ lục P1, P2 chỉ thêm ở cuối.
- Nguồn: `04`; `results/phase2/kc48_mo_phong/giao_thuc_sha256_r48d.jsonl`; 07 mục 4.

## Phần D — Kết quả

**Slide 8. Mô phỏng: PF hiện tại kém EKF một cách nhất quán** [QS]
- C1 `pf100` − `ekf`: F 25/0/32 ô, e_r 32/0/32 ô; nhóm lệch mô hình cũng vậy.
- Hình: `results/phase2/kc48_mo_phong/phan_tich/hinh_c1_e_r.png`.
- Nguồn: 07 mục 5.2; `phan_tich/ket_luan_doi_chieu.csv`.

**Slide 9. Cơ chế: nghèo hạt ngay khi đang bám** [QS]
- Ô `cua_deu`, L = 20: ESS/N khi bám 0,07–0,17; 11–29 % tổ tiên sau resample; NIS cỡ 10⁹.
- K1 nâng ESS/N, giảm resample, nhưng không sửa sụp ESS sau mất đo dài (≈ 1 %).
- ESS chỉ tính trên update không tái khởi tạo; tái khởi tạo đếm riêng. h0 = 24, tại t_r: `pf100` 43/90 quỹ đạo tái khởi
  tạo, K1 3/90 (07 mục 5.3, mục 11).
- Co trung bình (H6) nhỏ so với e_r ⇒ không phải nguyên nhân chính trong các ô này.
- Nguồn: 07 mục 5.3; `phan_tich/tom_tat_o.csv`.

**Slide 10. Đánh đổi tài nguyên** [QS]
- Bảng 07 mục 5.4: `pf-k1` N = 100 tốt hơn `pf` N = 500 ở cả e_r lẫn F, và nhanh hơn; không biến thể PF nào đạt EKF.
- Ghi trên slide: thời gian đo trong harness Python có bật chẩn đoán, **không phải FPS end-to-end**; chưa đo trên Nano.
- Hình: `phan_tich/hinh_tai_nguyen_va_ess.png`. Nguồn: `phan_tich/c4_tai_nguyen.csv`, `toc_do.csv`.

**Slide 11. Pipeline dev12: bảng chính** [QS]
- Bảng 07 mục 6.3 (HOTA, AssA, IDF1, IDSW; PF: COMBINED từng seed rồi trung bình 3 seed, kèm min–max).
- Nguồn: `results/phase2/kc48_pf_dev/phan_tich/bang_chinh.csv`.

**Slide 12. Bootstrap theo video: K1, K2 giúp PF nhưng PF vẫn kém EKF** [QS]
- Bảng 07 mục 6.4: P2 K1 / K2 / K1+K2 so với `pf100` có CI loại 0; P3 `pf100-k1`, `pf500-k1` so với `ekf` vẫn âm, CI
  loại 0. EKF đã chỉnh không khác EKF mặc định.
- Nói rõ: kết quả bất lợi cho PF.
- Nguồn: `phan_tich/doi_chieu_bootstrap.csv`.

**Slide 13. Phân tầng và sự kiện** [QS] + [GT]
- Giữ ID theo lớp / cỡ / cua (07 mục 6.5): PF kém EKF ở mọi tầng có suy luận; nhóm cua chỉ 32 track / 5 video.
- Che ≥ 0,9: EKF giữ 2/33 sự kiện; mọi bộ lọc gần như đều mất ID — khớp kết luận cũ về trần `track_buffer`.
- Xe nhỏ cạnh xe lớn: 42 sự kiện / 7 video; K1 về mức không phân biệt được với EKF (CI chạm 0) — **không** phải bằng
  chứng ngang EKF.
- [GT] vì sao khoảng cách ở nhóm cua nhỏ hơn: chưa kiểm. **Không** kết luận "PF tốt khi cua".
- Nguồn: `phan_tich/phan_tang.csv`, `su_kien/bang_su_kien.csv`, `su_kien/co_mau_va_loai.csv`.

## Phần E — Kết luận, giới hạn, bước tiếp

**Slide 14. Trả lời Q1–Q3 và giới hạn**
- Q1–Q3 theo 07 mục 7, nguyên văn ý.
- Giới hạn chính (07 mục 8): dev12 nhỏ và từng dùng ở mốc 2; mô phỏng trên mặt phẳng ảnh; khác biệt pipeline gộp cả
  heuristic triển khai; test 11 chưa chạy lại sau K1; tái tạo mốc 2 mới kiểm 2/12 video.
- Chưa chạy: pipeline trên UA-DETRAC test / BDD val với K1; H2 (association), H3 (likelihood thật); M5 chỉ có thiết kế
  (`06`), không có số huấn luyện; đo trên Jetson Nano (hoãn).
- Bước tiếp (cần người dùng quyết định, không tự làm): có chạy K1 trên tập đánh giá cuối hay không; có thực thi thiết kế
  `06` hay không.

## Slide dự phòng

- **D1. Bảng tinh chỉnh cân bằng** (07 mục 5.1; `results/phase2/kc48_mo_phong/chon.json`): K2 c = 1,0; EKF ×2.
- **D2. Thiết kế augmentation M5** (`06` mục 3–6): một thay đổi đề xuất, tập con lồng nhau 25/50/100 %, cách tính
  compute. Ghi rõ: **chỉ thiết kế, chưa huấn luyện**.
- **D3. Truy nguyên "EKF lệch mốc 2"** (`05b`): file tracking trùng byte trên 2/12 video; nhận định lệch trước đây do quy
  ước hash khác đời (commit `935a445`).

## Quy ước hình

Chỉ dùng hình mô phỏng và biểu đồ số liệu (07 mục 9). Overlay trên ảnh dataset nằm ở
`results/phase2/kc48_audit/overlay/`, không đưa vào repo public; nếu cần cho slide, người dùng tự chèn từ máy.
