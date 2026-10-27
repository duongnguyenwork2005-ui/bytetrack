# R51B — Thực thi giao thức trên UA-DETRAC dev12

Ngày chạy: 2026-10-18. Người triển khai: Claude (`claude-opus-5-5`, effort high). Reviewer: Codex ở phiên riêng.
Bước này thực thi mục 8 của `01_giao_thuc.md`, phiên bản Codex approve ngày 18/10 ở
`review_thu_cong_R51A_vong2/answer.json`. Không đổi bộ lọc, tracker, evaluator, YAML hay tham số nào. Lần chấm đầu
**sai trình tự** cổng R0 trước R1 (review vòng 1, R51B-F1); đã sửa và chạy lại theo đúng trình tự, xem mục 2.1.

Lý do chạy thủ công: orchestrator R51 dừng ở `quota_wait` của R51A từ 16/10, và repo đã khác mốc init do lượt R52.
Người dùng duyệt chạy R51B ngày 18/10 ("chạy R51B đi"). Tôi không sửa state hay plan của lượt R51.

## 1. Đã chạy gì

| Loại | Run | Nguồn |
|---|---|---|
| Chạy mới để so sánh (4, tất định) | bt-cv, bt-ukf, sort-cv30, sort-ekf30 | `results/phase2/r51_controlled/runs/_processed/trackers/DETRAC-all/p2-r51_controlled-A-*` |
| Run neo kiểm tái dùng (2) | bt-ekf; bt-pf100-k1 seed 0 | như trên |
| Tái dùng (16) | `p2-kc48_pf_dev-A-{ekf, pf100, pf100-k1, pf100-k2, pf500, pf500-k1}` × seed 0/1/2 | bản sao byte-trùng từ `results/phase2/kc48_pf_dev/_processed/…` |

- Runner: `src/r51_controlled/chay_buoc_b.py`. Cấu hình: `configs/phase2/r51_controlled/r51_b.yaml`, chỉ diễn đạt
  lại mục 2 và trỏ đúng các YAML ở mục 4.3.
- Bước track gọi **đúng** `run_experiment.stage_track`, cùng đường code với các run kc48. Bước chấm gọi đúng
  `run_trackeval.run`, `flatten_results`, `boot_stats.save_raw_fields` và `run_fingerprint`, giống
  `stage_eval`.
- Wrapper chỉ đổi chỗ ghi: `config.PROCESSED_DIR` → `runs/_processed/`; metric → `eval/{R0,R1}/<dir>/`.
- Chỉ dùng CPU, đọc từ cache detection. HEAD `117d62d`. Cùng lúc GPU đang train YOLO11n (lượt R52), không liên quan.

## 2. Lệnh, exit, thời gian

Python `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -X utf8`, cwd repo. Nhật ký từng bước nằm ở
`results/phase2/r51_controlled/runs/nhat_ky_b.jsonl`.

| Lệnh `chay_buoc_b.py …` | Exit | Thời gian | Kết quả |
|---|---|---|---|
| `chuan-bi` | 0 | 13 s | YAML 6/6 trùng sha12 đã khóa; cache npz/meta/weights trùng giao thức; backend trùng; 24 file GT/seqinfo byte-trùng bản gốc, dấu vân tay GT ở gốc mới trùng `07de8402…`/`024df16d…`; 192 file tái dùng byte-trùng |
| `track` | 0 | 252 s | bt-cv 24,7 s; bt-ukf 75,0 s; sort-cv30 25,6 s; sort-ekf30 25,4 s; bt-ekf 30,4 s; bt-pf100-k1-s0 66,0 s (12 video mỗi run, chạy tuần tự) |
| `neo` | 0 | 0,4 s | **bt-ekf 12/12, bt-pf100-k1-s0 12/12** trùng sha256 (quy ước LF) với kc48 → tiêu chí 7 đạt |
| `r1` | 0 | 8 s | 240 file R1; số box bị bỏ mỗi run 323–373 |
| `cham` | 0 | 557 s | 40 lần TrackEval (20 run × R0/R1), mỗi lần 13–14 s (lần đầu, **trình tự sai**, xem mục 2.1) |
| `cham` (sau sửa F1) | 0 | 21 s | 20 R0 tái dùng → **cổng R0 16/16 đạt** → 20 R1 tái dùng (`tests/cong_r0.json`) |
| `kiem-cham` | 0 | 1 s | xem mục 3 |
| `bang`, `doi-chieu` | 0 | 1 s | bảng mục 4; đối chiếu mục 3 |
| `src/r51_controlled/test_buoc_b.py` | 0 | 14 s | **16/16 PASS** (`tests/test_buoc_b.log`) |
| `src/r51_controlled/test_protocol.py` | 0 | — | 19/19 PASS (`tests/test_protocol.log`) |

Chi tiết lần chạy: lần đầu `test_buoc_b.py` được 12/13. Lỗi nằm ở phép so của chính test (chuỗi rỗng ghi ra CSV đọc lại
thành NaN), giá trị không lệch. Đã sửa phép so. Sau khi tách hàm `loc_r1` (tách code, logic không đổi), tôi chạy lại
`chuan-bi, neo, r1, cham, kiem-cham, bang, doi-chieu`. Lần đó `cham` tái dùng đủ 40/40, vì dấu vân tay không đổi.
Vậy mọi đầu ra hiện có đều do bản code cuối cùng sinh ra.

### 2.1 Sửa R51B-F1 (review Codex vòng 1: revise)

- **Sai lệch:** mục 8 yêu cầu R0 của các run tái dùng tái tạo số cũ (≤ 1e-12) **trước khi** chấm R1. Lần chạy đầu
  chấm R0 rồi R1 liền nhau. Phép đối chiếu chỉ nằm ở `kiem-cham`, chạy sau cả hai, và một lần chấm R0 thất bại chỉ
  được đếm lỗi rồi chạy tiếp. Số liệu không sai: đối chiếu sau đó cho chênh lệch 0. Nhưng trình tự không đúng giao
  thức.
- **Sửa:** `trinh_tu_cham` chấm R0 cho mọi run. Nếu có lần R0 thất bại thì dừng, không chạy cổng và không chạy R1
  (exit 1). Tiếp đó là `cong_r0`, dựa trên hàm thuần `so_r0_voi_cu`: thiếu file, thiếu cột, số dòng COMBINED khác 1,
  tập video khác hoặc trùng, giá trị không hữu hạn, sai số > 1e-12 đều là lỗi. Cổng không đạt thì dừng, không chạy R1
  (exit 2). Chỉ khi cổng đạt mới chấm R1. `kiem-cham` gọi lại đúng cổng này.
- **Test (không chạy thí nghiệm thật):** `test_so_r0_voi_cu_fixture` có ca đạt, ca trong ngưỡng và 9 ca lỗi.
  `test_trinh_tu_cham_cong_chan_r1` kiểm thứ tự gọi: cổng không đạt hoặc R0 thất bại thì không có lần R1 nào.
  `test_nhat_ky_cong_r0_truoc_r1` kiểm trên nhật ký của lần `cham` cuối cùng rằng thứ tự đúng là 20 R0, rồi cổng đạt
  16/16, rồi 20 R1.
- **Chạy lại:** `cham` dùng cơ chế tái dùng theo dấu vân tay. Không chạy lại tracking hay TrackEval, không xóa bằng
  chứng cũ (các dòng nhật ký lần đầu vẫn giữ). Đầu ra metric không đổi.

## 3. Kiểm (tự chạy, đếm trên nguồn đầy đủ)

- **Tái dùng (tiêu chí 7):** hai run neo chạy lại tại HEAD trùng 12/12 (`tests/neo_kiem.json`;
  `test_neo_trung_12_12`).
- **Chấm R0 tái tạo số cũ (cổng trước R1):** chấm lại R0 cho 16 run tái dùng cho lại `combined_metrics.csv` và
  `per_video_metrics.csv` cũ ở `results/trackeval/DETRAC-all/…`. Chênh lệch lớn nhất bằng 0, dưới ngưỡng 1e-12, đạt
  16/16 (`tests/cong_r0.json`, `tests/kiem_cham.json`; `test_r0_tai_tao_combined_cu`).
- **R1 khớp lý thuyết CLEAR:** ở cả 20 run, FP giảm đúng bằng số box rơi vào 572 frame (đếm lại độc lập từ file R0).
  TP, FN và IDSW giữ nguyên. Với 16 run tái dùng, FP của R1 trùng con số dự đoán ở bước A (`r1_CLR_FP_moi_chinh_xac`),
  đạt 16/16.
- **Run mới đạt cổng provenance của bước A:** áp chính hàm `audit_protocol.kiem_provenance` (tiêu chí 1–6) cho 4 run
  mới cùng `eval_meta` R0 của chúng: đạt 4/4 (`test_run_moi_dat_cong_provenance`).
- **Cấu hình:** `tracker_cfg` của 6 run mới bằng đúng `tracker_cfg` của run cũ cùng tên. Thông số bộ lọc hiệu lực
  (`bo_loc_hieu_luc`) trùng các cột `bl_*` của `kiem_ke_run.csv`. Seed PF của run neo bằng 0, là seed của runner
  (`test_run_moi_dung_cau_hinh_khoa`).
- **Không ghi vào vùng cấm:** trước và sau `track` và `cham`, danh sách `data/processed/trackers/DETRAC-all`,
  `results/trackeval/DETRAC-all` và `kc48_pf_dev/_processed/trackers` không đổi. mtime lớn nhất của các file kc48 và
  các file `results/trackeval` tái dùng cũng không đổi. Không có thư mục `p2-r51_controlled-*` nào ở đó
  (`test_khong_ghi_thu_muc_cam`).
- **Đối chiếu thông tin (không dùng để quyết định):** đầu ra mới của bt-cv, bt-ukf, sort-cv30 và sort-ekf30 trùng
  sha256 12/12 với các run cũ không được tái dùng (`expA c25-cv/c25-ukf`, `expB_fix1 sort-cv30/sort-ekf30`). HOTA,
  AssA, IDF1 và IDSW cũng trùng (`tests/doi_chieu_run_cu.json`). Các run cũ này bị loại vì thiếu provenance, không phải
  vì kết quả sai. Số dùng cho báo cáo là số của run mới.

## 4. Kết quả COMBINED trên 12 video (điểm, **chưa có CI**)

Nguồn: `results/phase2/r51_controlled/eval/combined_theo_bien_the.csv` (từng run ở `combined_tung_run.csv`).
PF: COMBINED tính riêng từng seed rồi lấy trung bình 3 seed. Quy tắc chính là R0.

| Biến thể | Nguồn | HOTA | AssA | DetA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| bt-cv | mới | 0,5900 | 0,6789 | 0,5200 | 0,7177 | 489 |
| bt-ekf | tái dùng | 0,5925 | 0,6838 | 0,5207 | 0,7245 | 472 |
| bt-ukf | mới | 0,5879 | 0,6734 | 0,5206 | 0,7151 | 474 |
| bt-pf100 (TB 3 seed) | tái dùng | 0,4933 | 0,5383 | 0,4692 | 0,5509 | 2.069,3 |
| bt-pf100-k1 (TB 3 seed) | tái dùng | 0,5377 | 0,6203 | 0,4802 | 0,6334 | 1.094,3 |
| sort-cv30 | mới | 0,5677 | 0,6427 | 0,5091 | 0,6821 | 680 |
| sort-ekf30 | mới | 0,5652 | 0,6376 | 0,5091 | 0,6767 | 679 |
| *Phụ lục:* bt-pf100-k2 / bt-pf500 / bt-pf500-k1 | tái dùng | 0,5148 / 0,5269 / 0,5548 | 0,5670 / 0,5896 / 0,6469 | 0,4810 / 0,4847 / 0,4883 | 0,5866 / 0,6056 / 0,6583 | 1.646 / 1.352 / 835,7 |

Hiệu số điểm của các đối chứng chính (`eval/hieu_so_diem.csv`, R0):

| Câu hỏi | Đối chứng | ΔHOTA | ΔAssA | ΔDetA | ΔIDF1 | ΔIDSW |
|---|---|---|---|---|---|---|
| Q1 | ekf − cv | +0,0025 | +0,0049 | +0,0006 | +0,0068 | −17 |
| Q1 | ukf − cv | −0,0021 | −0,0055 | +0,0005 | −0,0025 | −15 |
| Q1 | ukf − ekf | −0,0047 | −0,0103 | −0,0001 | −0,0093 | +2 |
| Q1 | pf100 − ekf | −0,0992 | −0,1454 | −0,0515 | −0,1736 | +1.597 |
| Q1 | pf100k1 − pf100 | +0,0444 | +0,0820 | +0,0110 | +0,0825 | −975 |
| Q1 | pf100k1 − ekf | −0,0548 | −0,0635 | −0,0405 | −0,0911 | +622 |
| Q2 | bt-cv − sort-cv30 | +0,0223 | +0,0362 | +0,0109 | +0,0356 | −191 |
| Q2 | bt-ekf − sort-ekf30 | +0,0273 | +0,0462 | +0,0116 | +0,0478 | −207 |

- R1 so với R0: mọi hiệu số đổi nhiều nhất 0,00034 (IDF1), IDSW không đổi, không đổi dấu ở đối chứng nào (`hieu_so_diem.csv`, các dòng R1).
  Theo quy tắc báo cáo ở mục 3.4, trạng thái CI của R0 và R1 phải được so ở bước C.
- Đây là số điểm trên 12 video, chưa có khoảng tin cậy. Bootstrap ghép cặp theo video (5.000 mẫu) và mọi kết luận
  "phân biệt được / không phân biệt được" thuộc bước C. Ở bước này chưa kết luận gì về ý nghĩa. Ví dụ, chênh lệch
  ekf − cv (+0,0025 HOTA) có thể nằm trong nhiễu giữa các video.
- Q1 và Q2 là hai câu hỏi riêng, không gộp thành một bảng xếp hạng. Q2 so "association và vòng đời kiểu SORT, dùng bộ
  lọc của ByteTrack, `max_age 30`", **không** phải SORT nguyên bản. Ở điều kiện A, vòng ghép điểm thấp của ByteTrack
  rỗng (mục 6 của giao thức).

## 5. Đầu ra

- `results/phase2/r51_controlled/runs/`: 380 MB, git bỏ qua (`.gitignore: runs/`). Gồm đầu ra tracker mới, bản sao
  tái dùng, bản R1 và bản sao GT (có `.gitignore '*'`, vì GT là nhãn dataset còn repo là public).
- `results/phase2/r51_controlled/eval/`: 1,4 MB. Gồm `R0/` và `R1/` (mỗi run có `per_video_metrics.csv`,
  `combined_metrics.csv`, `raw_fields.json`, `eval_meta.json` với trường `quy_tac`) và ba bảng CSV tổng hợp.
- `results/phase2/r51_controlled/tests/`: `chuan_bi.json`, `neo_kiem.json`, `r1_tao.json`, `kiem_cham.json`,
  `doi_chieu_run_cu.json` và log hai bộ test.

## 6. Giới hạn

- Không chạy lại 16 run tái dùng. Bằng chứng cho chúng gồm cổng provenance (bước A), hai run neo 12/12, và việc chấm
  R0 tái tạo đúng số cũ.
- Run neo chỉ có seed 0 của pf100-k1, đúng theo giao thức. Các seed 1/2 và các họ phụ lục dựa vào cổng provenance.
- Thời gian ghi ở đây là đồng hồ tường trên một máy, chạy cùng lúc với tiến trình train GPU. Chỉ dùng để dự toán,
  không phải benchmark.
- Chạy script sinh `__pycache__` trong `src/` (git bỏ qua).
- Không làm: bootstrap/CI, phân tầng, tập test, BDD, c10, tuning, H2/H3, GPU. Những phần này thuộc bước C hoặc nằm
  ngoài phạm vi.
