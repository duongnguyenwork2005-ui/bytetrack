# R51C — Tổng hợp, khoảng tin cậy và bảng slide (UA-DETRAC dev12)

Ngày: 2026-10-18. Claude (`claude-opus-5-5`, effort high) triển khai; Codex review ở phiên riêng (chưa review). Đây
là Claude làm và Codex soát lại quy trình, **không** phải kiểm chứng chéo kết quả khoa học giữa hai mô hình.

Trạng thái: R51A **approve** (`review_thu_cong_R51A_vong2/answer.json`), R51B **approve**
(`review_thu_cong_R51B_vong2/answer.json`), R51C **approve** ở review Codex vòng 2 ngày 19/10
(`review_thu_cong_R51C_vong2/answer.json`; vòng 1 revise R51C-F1, xem mục 6.1). Sau approve chỉ sửa một lỗi làm tròn
reviewer nêu (cận trên CI ΔIDF1 bt-ekf − sort-ekf30: +0,0677 → +0,0676, khớp `slide_hieu_so_ci.csv`). Cột
`trang_thai` "C cho review" trong CSV slide là trạng thái lúc đóng băng; nay C đã approve. Người dùng duyệt làm R51C ngày 18/10 ("Tiếp tục đi"
sau câu hỏi có làm R51C không). Mọi số dưới đây chưa công bố cho tới khi reviewer approve.

Phạm vi: chỉ tổng hợp đầu ra đã approve của B. Không chạy tracking, không chấm lại TrackEval, không thêm thí nghiệm,
không mở H2/H3. Giao thức cố định: `01_giao_thuc.md` mục 1, 5–7.

- Script: `src/r51_controlled/summarize.py` (chỉ đọc `eval/`, `runs/_processed/`, kết quả R48D/R48B; chỉ ghi
  `results/phase2/r51_controlled/summary/`). Test: `src/r51_controlled/test_summary.py`.
- Nguồn từng số: `summary/source_map.json` (sha256 của 120 file `raw_fields.json` / `eval_meta.json` /
  `combined_metrics.csv`, mã và đầu vào phân tầng, hash đầu ra).

## 1. Cách tính (đúng giao thức mục 7)

| Mục | Cách làm |
|---|---|
| Tập | UA-DETRAC dev12 (12 video thuộc phần train; `results/phase2/manifests/detrac_dev.json`). **Không có kết quả test** trong R51. |
| Detector | cache `yolov8n-coco_conf0.05_all` (YOLOv8n COCO gốc), điều kiện A `conf > 0,25`. Không phải detector mixed5. |
| Quy tắc chấm | R0 = TrackEval MotChallenge2DBox hiện tại (**chính**). R1 = bỏ box tracker ở 572 ảnh không có `<frame>` (độ nhạy, mục 3.4). |
| Metric | HOTA, AssA, DetA, IDF1, IDSW; MOTA, FP chỉ tham chiếu. COMBINED trên 12 video từ `raw_fields.json` theo công thức TrackEval (`src/boot_stats.combine_stacked`). Tái tổ hợp khớp `combined_metrics.csv` của TrackEval, lệch lớn nhất 5,6e-17 (40/40 run, `kiem_bang_chinh.json`). |
| PF nhiều seed | COMBINED **riêng từng seed** 0, 1, 2 rồi lấy trung bình. Khoảng min–max giữa seed chỉ là mô tả (cột `*_seed_*_mo_ta`), **không** dùng làm CI. |
| Bootstrap | Ghép cặp theo **video** (n = 12), B = 5.000, `numpy.default_rng(0)`, mỗi mẫu `integers(0, 12, size=12)` (cùng dãy với `boot_stats.paired_bootstrap`). **Một** dãy chỉ số dùng chung cho mọi đối chứng, mọi biến thể và mọi seed PF (`summary/chi_so_bootstrap_video.npy`). Trong mỗi mẫu, PF tính COMBINED theo seed 0, 1, 2 rồi trung bình. CI 95 % phân vị cho hiệu số a − b. |
| Đọc CI | Đơn vị độc lập là video; frame, box, track không độc lập. "CI chứa 0" = không phân biệt được trên 12 video, **không** có nghĩa là bằng nhau. |
| Nguồn run | `moi` = chạy mới ở B (bt-cv, bt-ukf, sort-cv30, sort-ekf30). `tai_dung` = run kc48 đạt cổng provenance, kiểm bằng run neo 12/12 (bt-ekf, mọi PF). |

## 2. Q1 — bộ lọc trong cùng ByteTrack (R0)

Điểm COMBINED (`summary/bang_chinh_R0.csv`, `slide_q1_bo_loc_bytetrack.csv`):

| Biến thể | Nguồn | Seed | HOTA | AssA | DetA | IDF1 | IDSW |
|---|---|---|---|---|---|---|---|
| ByteTrack + KF-CV | mới | — | 0,5900 | 0,6789 | 0,5200 | 0,7177 | 489 |
| ByteTrack + EKF-CTRV | tái dùng | — | 0,5925 | 0,6838 | 0,5207 | 0,7245 | 472 |
| ByteTrack + UKF-CTRV | mới | — | 0,5879 | 0,6734 | 0,5206 | 0,7151 | 474 |
| ByteTrack + PF-CTRV N=100 | tái dùng | 3 | 0,4933 | 0,5383 | 0,4692 | 0,5509 | 2.069,3 |
| ByteTrack + PF-CTRV N=100 + K1 | tái dùng | 3 | 0,5377 | 0,6203 | 0,4802 | 0,6334 | 1.094,3 |

Hiệu số và CI 95 % (`summary/doi_chieu_R0.csv`, `slide_hieu_so_ci.csv`); **in đậm** = CI không chứa 0:

| Đối chứng | ΔHOTA | ΔAssA | ΔDetA | ΔIDF1 | ΔIDSW |
|---|---|---|---|---|---|
| ekf − cv | +0,0025 [−0,0018; +0,0060] | +0,0049 [−0,0039; +0,0127] | +0,0006 [−0,0003; +0,0015] | +0,0068 [−0,0031; +0,0155] | −17 [−53; +23] |
| ukf − cv | −0,0021 [−0,0085; +0,0010] | −0,0055 [−0,0190; +0,0014] | +0,0005 [−0,0004; +0,0012] | −0,0025 [−0,0130; +0,0028] | −15 [−51; +20] |
| ukf − ekf | **−0,0047 [−0,0085; −0,0008]** | **−0,0103 [−0,0187; −0,0015]** | −0,0001 [−0,0003; +0,0001] | **−0,0093 [−0,0160; −0,0011]** | +2 [−13; +16] |
| pf100 − ekf | **−0,0992 [−0,1781; −0,0498]** | **−0,1454 [−0,2740; −0,0715]** | **−0,0515 [−0,0787; −0,0278]** | **−0,1736 [−0,2919; −0,0854]** | **+1.597 [+719; +2.555]** |
| pf100k1 − pf100 | **+0,0444 [+0,0184; +0,0867]** | **+0,0820 [+0,0303; +0,1596]** | **+0,0110 [+0,0065; +0,0177]** | **+0,0825 [+0,0344; +0,1459]** | **−975 [−1.567; −434]** |
| pf100k1 − ekf | **−0,0548 [−0,0925; −0,0310]** | **−0,0635 [−0,1162; −0,0401]** | **−0,0405 [−0,0644; −0,0201]** | **−0,0911 [−0,1497; −0,0510]** | **+622 [+277; +995]** |

Đọc (chỉ trong phạm vi dev12, detector YOLOv8n COCO, điều kiện A):

- KF-CV, EKF-CTRV và UKF-CTRV **không phân biệt được** khi so với CV: hai CI đều chứa 0, chênh HOTA dưới 0,003.
  UKF thấp hơn EKF một chút và CI không chứa 0 (HOTA −0,005, chủ yếu từ AssA). Độ lớn này rất nhỏ so với khoảng cách
  với PF; không suy ra UKF "kém" một cách thực tế.
- PF-CTRV N=100 thấp hơn EKF rõ ràng ở mọi metric. K1 cải thiện PF rõ ràng nhưng PF+K1 **vẫn thấp hơn EKF** ở mọi
  metric. Không có kết quả nào ở đây cho thấy PF thắng.
- PF có DetA thấp hơn (−0,05) chứ không chỉ AssA. Nguyên nhân DetA nằm ngoài phạm vi R51 (mục 8 giao thức).

Phụ lục (chỉ tái dùng, không thuộc đối chứng chính, không bootstrap ở R51; CI cho các biến thể này đã có ở R48D
`results/phase2/kc48_pf_dev/phan_tich/doi_chieu_bootstrap.csv`): HOTA pf100-k2 0,5148; pf500 0,5269;
pf500-k1 0,5548 (vẫn dưới EKF 0,5925).

## 3. Q2 — SORT-kiểu so với ByteTrack, cùng bộ lọc (R0)

"SORT-kiểu" = association và vòng đời kiểu SORT, dùng **đúng** bộ lọc của ByteTrack, `max_age 30` (giao thức mục 6).
Không phải SORT nguyên bản. Vòng ghép điểm thấp của ByteTrack rỗng ở điều kiện A, nên Q2 không đo lợi ích BYTE.

| Biến thể | Nguồn | HOTA | AssA | DetA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| SORT-kiểu + KF-CV | mới | 0,5677 | 0,6427 | 0,5091 | 0,6821 | 680 |
| ByteTrack + KF-CV | mới | 0,5900 | 0,6789 | 0,5200 | 0,7177 | 489 |
| SORT-kiểu + EKF-CTRV | mới | 0,5652 | 0,6376 | 0,5091 | 0,6767 | 679 |
| ByteTrack + EKF-CTRV | tái dùng | 0,5925 | 0,6838 | 0,5207 | 0,7245 | 472 |

| Đối chứng | ΔHOTA | ΔAssA | ΔDetA | ΔIDF1 | ΔIDSW |
|---|---|---|---|---|---|
| bt-cv − sort-cv30 | **+0,0223 [+0,0192; +0,0274]** | **+0,0362 [+0,0267; +0,0477]** | **+0,0109 [+0,0063; +0,0171]** | **+0,0356 [+0,0257; +0,0436]** | **−191 [−371; −38]** |
| bt-ekf − sort-ekf30 | **+0,0273 [+0,0199; +0,0327]** | **+0,0462 [+0,0267; +0,0601]** | **+0,0116 [+0,0066; +0,0181]** | **+0,0478 [+0,0269; +0,0676]** | **−207 [−359; −79]** |

Với cùng bộ lọc, association và vòng đời của ByteTrack cho HOTA cao hơn SORT-kiểu khoảng 0,02–0,03, IDSW ít hơn
khoảng 200; CI không chứa 0 ở cả hai bộ lọc. Không ghép Q1 và Q2 thành một bảng xếp hạng (giao thức mục 1).

## 4. Độ nhạy R1

R1 đổi hiệu số điểm tối đa 0,0002 (HOTA), 0,00003 (AssA), 0,0002 (DetA), 0,00034 (IDF1), 0 (IDSW) và **không đổi**
kết luận "CI chứa 0 / không chứa 0" ở ô nào trong 8 đối chứng × 5 metric (0/40) (`summary/doi_chieu_R1.csv`; cột `R1_cung_ket_luan_HOTA` trong `slide_hieu_so_ci.csv`).
IDSW giống hệt R0, vì R1 chỉ bỏ box ở ảnh không có GT.

## 5. Phân tầng (R0; R1 không đổi kết quả tầng)

Định nghĩa dùng lại, không định nghĩa mới (giao thức mục 7). Ghép GT–tracker theo từng frame **có GT**
(`occlusion_eval.match_video`, Hungarian có ràng buộc IoU ≥ 0,5). Vì R1 chỉ bỏ box ở ảnh không có GT, đầu vào phân
tầng của R0 và R1 trùng nhau (test `test_r1_khong_doi_dau_vao_phan_tang`, 20 run × 12 video). PF lấy trung bình 3 seed
trong từng đơn vị (cách R48D). Bootstrap theo video của tầng, B = 5.000, `default_rng(0)` riêng mỗi tầng, dùng chung cho
mọi đối chứng của tầng; hiệu = trung bình hiệu theo đơn vị. Chỉ suy luận khi ≥ 5 video đóng góp và ≥ 20 đơn vị.

| Tầng | Đơn vị | Định nghĩa / nguồn | Mẫu số | Suy luận? |
|---|---|---|---|---|
| Che khuất | sự kiện | `occlusion_eval` (gaps=None, ngữ cảnh 30 frame = 1,2 s), qua `su_kien_dev.su_kien_che` | 717 sự kiện → **184 đủ điều kiện / 12 video / 137 track** (loại 369 thiếu GT trước, 164 thiếu GT sau) | có |
| Che hoàn toàn (`has_full`) | sự kiện | tầng con của che khuất, có đoạn che ≥ 0,90 | **33 / 6 video** | có (sát ngưỡng) |
| Car nhỏ cạnh xe lớn | sự kiện | tiêu chí khóa `canh_xe_lon` (R48D), `su_kien_dev.su_kien_xe_nho` | 88 sự kiện / 65 track nhỏ / 9 video → **42 đủ điều kiện / 7 video / 29 track** | có |
| Cua | track GT | `loai_max` ∈ {cua, cua gắt} (audit R48B `quy_dao/detrac_train.csv`) | **32 / 5 video** trên 1.185 track GT dev12 | có (sát ngưỡng) |
| Cua theo p90 (độ nhạy) | track GT | `loai_p90` | 8 / 4 video | **không**, chỉ số đếm |

Chỉ số: sự kiện — tỷ lệ giữ ID (`preserved`: ID ngay trước = ID ngay sau) trên **mọi** sự kiện đủ điều kiện (không ghép
được trước tính là không giữ); kèm tỷ lệ trên sự kiện ghép được trước ở cả hai bên (`phan_tang_doi_chieu.csv`). Track
cua — tỷ lệ frame GT ghép với ID chính (`phan_tich_dev.giu_id`).

Tỷ lệ theo biến thể (`summary/phan_tang_bien_the.csv`):

| Biến thể | Che khuất (184) | Che hoàn toàn (33) | Car nhỏ (42) | Cua: giữ ID / track (32) |
|---|---|---|---|---|
| ByteTrack + KF-CV | 0,505 | 0,091 | 0,548 | 0,646 |
| ByteTrack + EKF | 0,489 | 0,061 | 0,548 | 0,651 |
| ByteTrack + UKF | 0,495 | 0,061 | 0,524 | 0,646 |
| ByteTrack + PF100 (TB 3 seed) | 0,293 | 0,020 | 0,413 | 0,594 |
| ByteTrack + PF100 + K1 | 0,391 | 0,030 | 0,524 | 0,611 |
| SORT-kiểu + KF-CV | 0,446 | 0,091 | 0,524 | 0,583 |
| SORT-kiểu + EKF | 0,440 | 0,091 | 0,524 | 0,584 |

Hiệu số và CI 95 % (`summary/phan_tang_doi_chieu.csv`, `slide_phan_tang.csv`); **in đậm** = CI không chứa 0:

| Đối chứng | Che khuất | Che hoàn toàn | Car nhỏ | Cua |
|---|---|---|---|---|
| ekf − cv | −0,016 [−0,049; +0,005] | −0,030 [−0,250; 0,000] | 0,000 [0,000; 0,000] | +0,005 [−0,000; +0,010] |
| ukf − cv | −0,011 [−0,044; +0,008] | −0,030 [−0,250; 0,000] | −0,024 [−0,077; 0,000] | +0,001 [−0,016; +0,010] |
| ukf − ekf | +0,005 [0,000; +0,016] | 0,000 [0,000; 0,000] | −0,024 [−0,077; 0,000] | −0,004 [−0,016; +0,001] |
| pf100 − ekf | **−0,196 [−0,329; −0,084]** | −0,040 [−0,444; 0,000] | **−0,135 [−0,242; −0,032]** | **−0,057 [−0,072; −0,037]** |
| pf100k1 − pf100 | **+0,098 [+0,022; +0,185]** | +0,010 [0,000; +0,125] | **+0,111 [+0,022; +0,230]** | **+0,017 [+0,002; +0,036]** |
| pf100k1 − ekf | **−0,098 [−0,165; −0,048]** | −0,030 [−0,375; 0,000] | −0,024 [−0,037; 0,000] | **−0,040 [−0,050; −0,018]** |
| bt-cv − sort-cv30 | **+0,060 [+0,018; +0,117]** | 0,000 [0,000; 0,000] | +0,024 [0,000; +0,107] | **+0,062 [+0,042; +0,089]** |
| bt-ekf − sort-ekf30 | +0,049 [0,000; +0,105] | −0,030 [−0,250; 0,000] | +0,024 [0,000; +0,107] | **+0,067 [+0,047; +0,094]** |

Đọc:

- Hướng ở các tầng giống bảng chính. PF100 thấp hơn EKF ở che khuất, car nhỏ và cua. K1 cải thiện PF nhưng vẫn thấp
  hơn EKF ở che khuất và cua; ở car nhỏ, CI của pf100k1 − ekf chạm 0 (không phân biệt được trên 42 sự kiện / 7 video).
- CV / EKF / UKF không phân biệt được ở mọi tầng.
- Ở tầng cua, PF vẫn thấp hơn EKF. **Không** kết luận "PF tốt khi cua" (giống R48D).
- Tầng che hoàn toàn: tỷ lệ giữ ID của mọi biến thể đều rất thấp (0,02–0,09) và mọi CI đều chứa 0. Với 33 sự kiện
  rời rạc trên 6 video, bootstrap percentile cho nhiều CI có một đầu đúng bằng 0; tầng này chỉ nên đọc như mô tả.
- Trên tầng cua, ByteTrack giữ ID tốt hơn SORT-kiểu khoảng 0,06–0,07, cùng hướng với bảng chính.
- Các số của run tái dùng (ekf, mọi PF) tái tạo đúng R48D (`bang_su_kien.csv`, `phan_tang.csv`; lệch < 1e-12).
  CI có thể khác R48D ở chữ số thứ ba, vì R48D dùng một dãy RNG tuần tự cho mọi tầng, còn R51C dùng `default_rng(0)`
  riêng mỗi tầng.

## 6. Kiểm và lệnh

Python `khoaluan-mot`, cwd repo, chạy CPU song song với train YOLO11n trên GPU (thời gian là đồng hồ tường).

| Lệnh | Exit | Thời gian | Ghi chú |
|---|---|---|---|
| `python -B -X utf8 src/r51_controlled/summarize.py chinh` | 0 | 33 s | tái tổ hợp 40/40 khớp TrackEval (lệch ≤ 5,6e-17) |
| `python -B -X utf8 src/r51_controlled/summarize.py phan-tang` | 0 | 582 s | 20 run × 12 video; kiểm xe nhỏ 88 / 65 / 9 khớp |
| `python -B -X utf8 src/r51_controlled/summarize.py slide` | 0 | — | `kiem_nguon` đạt: 40 `eval_meta` cùng evaluator (TrackEval 1.3.0, `mot`), cùng GT `07de8402…`, cùng npz cache; `quy_tac` khớp thư mục |
| `python -B -X utf8 src/r51_controlled/test_summary.py` | 0 | 40 s | 15/15 PASS (lần đầu, trước vòng sửa F1) |
| vòng sửa F1: `summarize.py tong-hop-phan-tang`, rồi `slide` | 0, 0 | < 10 s | chỉ tổng hợp lại từ `don_vi_*.csv` có sẵn; không chấm lại đơn vị |
| `python -B -X utf8 src/r51_controlled/test_summary.py` | 1, 1, rồi 0 | ~40 s | lần 1–2: lỗi trong test mới (đếm 10 dòng thay vì 8 ở bảng đối chiếu; cờ `ci_loai_0` = False ở dòng không có CI → sửa mã để trống); lần cuối **16/16 PASS** (`summary/test_summary.log`) |

### 6.1 Vòng sửa R51C-F1 (review Codex vòng 1, `review_thu_cong_R51C/answer.json`, revise)

- F1: bảng `phan_tang_bien_the.csv` vẫn xuất `ty_le_giu_id_tb`, `n_doi_id_tb` cho tầng `cua_loai_p90_do_nhay`
  (8 track / 4 video), trái mục 7 giao thức ("không đạt thì chỉ báo số đếm"). Cổng mẫu khi đó chỉ chặn hiệu số/CI.
- Sửa: cổng `du_mau` áp cho cả bảng theo biến thể; tầng thiếu mẫu chỉ giữ số đếm / metadata (`n_video`, `n_don_vi`,
  `n_track`) và `ghi_chu`; mọi chỉ số tổng hợp để trống. Cờ `ci_loai_0` ở bảng đối chiếu để trống khi không có CI
  (trước đó là False, dễ đọc nhầm thành "CI chứa 0"). Dữ liệu từng đơn vị (`don_vi_*.csv`) giữ nguyên.
- Test mới `test_nguong_mau_moi_bang_xuat`: biên ngưỡng (4, 20), (5, 19) thiếu, (5, 20) đủ; mọi bảng phân tầng xuất ra
  (theo biến thể, đối chiếu, slide) không có chỉ số tổng hợp / hiệu / CI ở dòng thiếu mẫu, gồm đúng ca p90 8 / 4.
- Số ở mục 2–5 không đổi (tầng p90 vốn không đưa vào bảng báo cáo).
- Ghi nhận giới hạn reviewer nêu: `handoff/logs/review51/02_thuc_thi.md` (file B) khác bản B đã approve ở một chỗ,
  "10 ca lỗi" → "9 ca lỗi". Đây là sửa lỗi đếm mà reviewer B vòng 2 nêu trong `limitations`; Claude sửa lúc 18:04
  18/10, sau khi B approve và trước khi bắt đầu C. Không đổi số liệu hay mã của B. R51C không sửa thêm file B.

Các test tự tính lại từ file gốc:

- Đối chứng khớp hàng "Đối chứng chính" của giao thức.
- Điểm bằng trung bình theo seed của `combined_metrics.csv` từ TrackEval (≤ 1e-9).
- Thứ tự PF "COMBINED trước, trung bình sau" khác thứ tự "trung bình trường thô trước", nên test phân biệt được hai cách.
- CI của 8 đối chứng × {HOTA, IDF1, IDSW} × {R0, R1} trùng `boot_seed.paired_bootstrap_seed`, tính độc lập (≤ 1e-12).
- Dãy chỉ số video trùng `default_rng(0)`.
- Không có cột seed min–max trong bảng CI.
- Bảng sự kiện trùng R48D, các số đếm đúng.
- Run tái dùng tái tạo R48D.
- R1 không đổi đầu vào phân tầng.
- Bootstrap phân tầng tính lại bằng vòng lặp thường cho kết quả trùng.
- Ngưỡng mẫu được áp đúng.
- Hash trong source map khớp file.
- Bảng slide khớp bảng đối chiếu.
- Mã chỉ ghi vào `summary/`.

Đầu ra (`results/phase2/r51_controlled/summary/`):

- Bảng chính: `bang_chinh_{R0,R1}.csv`, `bang_tung_seed.csv`.
- Bảng đối chiếu: `doi_chieu_{R0,R1}.csv`, `chi_so_bootstrap_video.npy`.
- Phân tầng: `phan_tang_bien_the.csv`, `phan_tang_doi_chieu.csv`, `co_mau_phan_tang.csv`; dữ liệu theo đơn vị: `don_vi_su_kien.csv`, `don_vi_track.csv`, `su_kien_*.csv`.
- Bảng slide: `slide_q1_bo_loc_bytetrack.csv`, `slide_q2_sort_vs_bytetrack.csv`, `slide_hieu_so_ci.csv`, `slide_phan_tang.csv`.
- Nguồn và kiểm: `source_map.json`, `kiem_*.json`, các log.

## 7. Giới hạn

- Chỉ dev12: 12 video phần train UA-DETRAC, từng dùng khi thiết kế mốc 2; CI theo video với n = 12 khá rộng và
  bootstrap percentile với n nhỏ có thể hẹp hơn thực tế. Không có kết quả tập test.
- Detector là YOLOv8n COCO gốc; kết quả có thể khác với detector mixed5 đang train.
- Điều kiện A làm vòng ghép điểm thấp rỗng; Q2 không đo phần BYTE.
- Phân tầng có cỡ mẫu vài chục sự kiện/track; GT UA-DETRAC không gán nhãn xe hai bánh nên không có tầng
  motorcycle/bicycle và không suy rộng sang chúng.
