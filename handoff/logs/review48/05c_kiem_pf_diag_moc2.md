# 05c — Kiểm chẩn đoán PF mốc 2: tái khởi tạo có trộn vào ESS / tỉ lệ resample không (bước P9a8ee8bf, 2026-10-11)

Bước người dùng duyệt qua email: kiểm chỉ đọc `src/pf_diag_report.py` và log mốc 2; nếu có trộn thì tính lại thống kê
ESS mốc 2, tách tái khởi tạo, ghi file mới `kc48_*`, không sửa kết quả cũ. Bối cảnh: F2 ở `07` mục 11 đã phát hiện và sửa
lỗi trộn này trong harness mô phỏng R48D. Câu hỏi còn lại là số chẩn đoán của mốc 2. Phần dev được `01`, `02` và `04`
trích dẫn.

**Đính chính F1 (review Codex, cùng ngày).** Bản đầu của file này chỉ quét `data/processed/trackers/DETRAC-all`. Bản đó
khẳng định sai hai điều: (a) "Glob ở các split khác cũng không thấy `pf_diag`" (mục 3), và (b) "Các run PF khác của mốc 2
không bật `pf_diag`" (mục 4). Thực tế `configs/phase2/experiments/expA_detrac_test.yaml:16` có `pf_diag: true`, và
`data/processed/trackers/DETRAC-test/p2-expA_detrac_test-c{10,25}-pf500-s{0,1,2}/pf_diag/` có 6 run × 40 video = 240 CSV
mà bản đầu bỏ sót. Bản này đã sửa script để kiểm kê mọi thư mục `pf_diag`, tính riêng nhóm test và sửa các câu sai.
Bảng dev không đổi (mục 5).

## 1. Kết luận

- **Về công thức: CÓ trộn.** Nhánh suy biến của `PFCTRV.update_particles` (`src/pf_ctrv.py:308–322`) ghi một dòng
  `kind = "update"` với `ess_ratio = 0.0`, `resampled = True`, `n_unique = N`, `degenerate = True`. `pf_diag_report.py`
  lọc `df.kind == "update"` (dòng 57) rồi lấy trung bình hoặc phân vị trực tiếp: `ess_ratio_mean`, `ess_ratio_p10`,
  `frac_resampled`, `frac_ess_lt_0.2`, `n_unique_ratio_*` (dòng 64–67), bảng theo `n_lost` (dòng 75–78). Script không
  loại dòng `degenerate`. Vì vậy, nếu có tái khởi tạo, ESS/N bị kéo xuống, còn tỉ lệ resample và `n_unique` bị đẩy lên.
  Quy ước ghi này có từ commit thêm chẩn đoán `6e28ffa` và không đổi qua `a64cb75`, `76748c8` đến mã hiện tại (đã đối
  chiếu bằng `git show`). Dòng `init_motion` (khởi tạo hai khung hình) đã bị lọc khỏi ESS, chỉ được đếm riêng, nên không
  trộn.
- **Về số liệu mốc 2 có `pf_diag`: KHÔNG bị trộn trên thực tế, ở cả dev và test.** Kiểm kê đủ 24 run có `pf_diag`
  (456 CSV, 24 `run_meta.json`) dưới `data/processed` và `results`. Kết quả:
  - **dev**: 18 run (`c10`/`c25` × N 100/300/500 × seed 0/1/2), 216 CSV, 1.664.036 update, **0** tái khởi tạo. Tính
    lại chỉ trên update hợp lệ cho kết quả trùng số cũ, lệch tối đa 1,1·10⁻¹⁶, là sai số làm tròn. Không số nào trong
    `handoff/logs/pf_diag_report.log` hay `results/phase2/pf_diag/expA_detrac_dev/*.csv` phải sửa. Các trích dẫn ở `01`,
    `02` và `04` (ESS/N ≈ 0,25 khi bám, ESS/N 0,007–0,017 sau 11–30 bước mất đo, resample ở 98,4–99,98 % update) vẫn
    đúng.
  - **test UA-DETRAC**: 6 run (`c10`/`c25` × N 500 × seed 0/1/2), 240 CSV, 3.400.029 update, **0** tái khởi tạo.
    `pf_diag_report.py` chưa từng chạy trên test: không có `results/phase2/pf_diag/expA_detrac_test/`, và
    `pf_diag_report.log` chỉ có dòng `-> …expA_detrac_dev`. Vì vậy không có số cũ để đối chiếu. Grep `ESS` trong
    `CODEX_HANDOFF.md`, `PHASE2_REPORT.md` và `handoff/*.md` chỉ thấy bảng dev (ví dụ `PHASE2_REPORT.md:563–565`, ESS/N
    0,248 với N = 100, trùng c25 dev), không thấy ESS test. Bảng test ở mục 2b là số mới. Đây là tập đánh giá cuối nên số chỉ để báo cáo, không dùng chọn
    ngưỡng hay tham số.
- maha2_best lớn nhất trong các update hợp lệ, theo từng run: dev 3.382–8.440, test 3.724–4.561. Cả 24 run đều dưới
  ngưỡng `pf_degenerate_maha2 = 10⁴`; ngưỡng này lấy từ `run_meta.json` của từng run. Giá trị gần ngưỡng nhất là 8.440
  (`c10-pf300-s1`, dev). Đây là quan sát, không phải giải thích cơ chế. Một giả thuyết chưa kiểm là association IoU của
  ByteTrack đã chặn các quan sát ở xa trước khi chúng tới bộ lọc, như docstring `pf_ctrv.py:39` nói. Ngược lại, mô phỏng
  R48D không có association nên có tái khởi tạo 0,44–0,99 % (`07` mục 11).

## 2. Bảng ESS mốc 2 tính lại (mẫu số và tái khởi tạo tách riêng)

Các tổng là tổng qua 3 seed. Tỉ lệ là trung bình giữa seed của thống kê từng run, giống `summary_by_tracker.csv` cũ; SD
giữa seed nằm trong CSV. ESS/N đo trước resample, chỉ trên update hợp lệ.

### 2a. Dev (12 video) — `results/phase2/kc48_pf_code/ess_moc2_tach_tai_khoi_tao/theo_tracker.csv`

| cond | tracker | update (mọi dòng) | tái khởi tạo | update hợp lệ (mẫu số) | ESS/N TB | ESS/N p10 | tỉ lệ resample | tỉ lệ ESS/N < 0,2 | n_unique/N TB |
|---|---|---|---|---|---|---|---|---|---|
| c10 | pf100 | 284.102 | 0 | 284.102 | 0,2500 | 0,0231 | 0,9843 | 0,4154 | 0,3769 |
| c10 | pf300 | 287.940 | 0 | 287.940 | 0,2578 | 0,0164 | 0,9989 | 0,3856 | 0,3840 |
| c10 | pf500 | 288.463 | 0 | 288.463 | 0,2608 | 0,0152 | 0,9998 | 0,3751 | 0,3883 |
| c25 | pf100 | 265.700 | 0 | 265.700 | 0,2476 | 0,0221 | 0,9837 | 0,4241 | 0,3735 |
| c25 | pf300 | 268.485 | 0 | 268.485 | 0,2564 | 0,0156 | 0,9989 | 0,3913 | 0,3818 |
| c25 | pf500 | 269.346 | 0 | 269.346 | 0,2598 | 0,0147 | 0,9998 | 0,3801 | 0,3868 |

Theo run: `theo_run.csv` có 18 dòng. Mỗi dòng gồm `n_update_tat_ca`, `n_tai_khoi_tao`, `n_update_hop_le`,
`n_init_motion`, thống kê hợp lệ, thống kê theo công thức cũ và `lech_max_vs_summary_cu`. Theo bin `n_lost`:
`theo_n_lost.csv` dùng cùng bin với `pf_diag_report` (khoảng đóng bên phải); `n_tai_khoi_tao` = 0 ở mọi bin.

### 2b. Test UA-DETRAC (40 video, chỉ báo cáo) — `…/theo_tracker_test.csv`

| cond | tracker | update (mọi dòng) | tái khởi tạo | update hợp lệ (mẫu số) | ESS/N TB | ESS/N p10 | tỉ lệ resample | tỉ lệ ESS/N < 0,2 | n_unique/N TB |
|---|---|---|---|---|---|---|---|---|---|
| c10 | pf500 | 1.764.485 | 0 | 1.764.485 | 0,3170 | 0,0862 | 0,9998 | 0,2190 | 0,4525 |
| c25 | pf500 | 1.635.544 | 0 | 1.635.544 | 0,3185 | 0,0888 | 0,9998 | 0,2180 | 0,4537 |

SD giữa seed của ESS/N TB: 0,0004 (c10) và 0,0006 (c25). Theo run: `theo_run_test.csv` có 6 dòng, `n_video` = 40 ở mọi
dòng, `n_init_motion` 8.205–8.663. Các cột `*_cong_thuc_cu` trùng cột `*_hop_le` vì không có tái khởi tạo. Không có cột
lệch so với số cũ vì test không có báo cáo cũ. Theo bin `n_lost`: `theo_n_lost_test.csv`, `n_tai_khoi_tao` = 0 ở mọi
bin. Ở bin 11–30, ESS/N hợp lệ là 0,0124–0,0130 (n = 494–772 update mỗi run).

Bảng test không so sánh được trực tiếp với bảng dev (khác tập video, chỉ N = 500), nên không rút kết luận về khác biệt
dev/test ở đây.

## 3. Cách kiểm và bằng chứng

Script chỉ đọc là `src/kc48/kiem_pf_diag_moc2.py`. Script không import pandas, dùng csv + numpy, với phân vị `linear`
giống mặc định của pandas. Các bước:
1. **Kiểm kê** (sửa F1): `os.walk` toàn bộ `data/processed` và `results`, lấy mọi thư mục tên `pf_diag`. Tìm được 25
   thư mục:
   - 24 thư mục là log của run: 18 `DETRAC-all/p2-expA_detrac_dev-*-pf*-s*` (12 CSV mỗi run) và 6
     `DETRAC-test/p2-expA_detrac_test-*-pf500-s*` (40 CSV mỗi run). Tổng 456 CSV, 24 `run_meta.json`.
   - 1 thư mục `results/phase2/pf_diag` là đầu ra của `pf_diag_report` (báo cáo cũ, không phải run). Thư mục này được
     tách ra và ghi riêng trong `ket_qua.json`.

   Run nào không thuộc nhóm dev/test thì bị báo lỗi; lần chạy cuối có `khong_thuoc_nhom: []`. Danh sách từng run và số
   CSV nằm trong `ket_qua.json` → `kiem_ke.run`.
2. Kiểm bất biến trên từng run, cả 24 run. Không bất biến nào bị vi phạm (`loi_bat_bien: []`):
   - `kind` chỉ nhận `update` hoặc `init_motion`;
   - `degenerate` ⇔ `ess_ratio == 0` ⇔ `maha2_best > pf_degenerate_maha2`, với ngưỡng lấy từ `run_meta.json`;
   - mọi ESS/N hợp lệ ≥ 1/N;
   - `n_unique ≤ N`;
   - `pf_num_particles` trong `run_meta.json` khớp tên run;
   - số CSV mỗi run đúng 12 (dev) hoặc 40 (test).
3. Tái tạo số cũ (chỉ dev, vì test không có số cũ) bằng đúng công thức của `pf_diag_report`. Kết quả:
   - `n_update` khớp `summary.csv` ở 18/18 run;
   - `n` theo bin khớp `by_n_lost.csv`;
   - lệch tối đa 1,1·10⁻¹⁶.
   Như vậy dữ liệu đọc là đúng dữ liệu báo cáo cũ đã dùng.
4. Chụp sha256 trước và sau khi tính cho các file sau:
   - 456 CSV `pf_diag` và 24 `run_meta.json`;
   - `results/phase2/pf_diag/expA_detrac_dev/{summary,by_n_lost,summary_by_tracker}.csv`;
   - `handoff/logs/pf_diag_report.log`;
   - `src/pf_diag_report.py`.

   Kết quả: `file_doi_sau_khi_chay: []`. Hash nằm trong `ket_qua.json` (`hash_truoc`).

Test `src/kc48/test_kiem_pf_diag_moc2.py` chạy trên API thật của `PFCTRV` (bật `enable_diag`, ép nhánh suy biến bằng
quan sát nhảy 10⁴ px):
- (1) dòng suy biến đúng quy ước mà script dựa vào; dòng hợp lệ có ESS/N ≥ 1/N và maha2 ≤ 10⁴;
- (2) với `resample_ess_ratio = 0`, công thức tách cho tỉ lệ resample 0, còn công thức cũ cho 3/30. ESS/N theo công thức
  cũ thấp hơn, tức công thức cũ trộn khi có tái khởi tạo;
- (3) không có tái khởi tạo thì hai công thức trùng khít;
- (4) bin `n_lost` trùng `pd.cut` ở các biên;
- (5) (thêm khi sửa F1) regex và thư mục split của nhóm dev/test tách đúng; run test không lọt vào dev và ngược lại.

**Lệnh, bản đầu (2026-10-11, trước F1):** `test_kiem_pf_diag_moc2.py` exit 0, 8/8 PASS (lần đầu 1 FAIL do kiểm định
viết quá chặt, đã sửa test bằng ca `ratio = 0`); `kiem_pf_diag_moc2.py` exit 0, `run 18, file 216, update 1664036, tai
khoi tao 0; lech max 1.110e-16; loi bat bien 0; file doi 0`.

## 4. Giới hạn

- Kết luận "không bị trộn trên thực tế" chỉ áp dụng cho 24 run `expA_detrac_{dev,test}` có `pf_diag`. Các run này dùng
  `bytetrack_pf_ctrv_n{100,300,500}.yaml` (test chỉ N = 500) và `pf_degenerate_maha2 = 10⁴`. Mọi run PF khác của mốc 2
  không có log `pf_diag` dưới `data/processed` hay `results`, nên không kiểm được. Phạm vi kiểm kê là hai gốc đó; các vùng lưu trữ ngoài repo không được quét.
- Số ESS test chỉ để báo cáo, không dùng chọn tham số. Bước này không chạy lại pipeline trên test.
- `pf_diag_report.py` vẫn trộn về công thức. Nếu sau này chạy trên dữ liệu có tái khởi tạo, ví dụ khi đổi ngưỡng hoặc
  association, số sẽ bị lệch. Bước này chỉ đọc nên không sửa `pf_diag_report.py`. Cách tính tách nằm trong
  `src/kc48/kiem_pf_diag_moc2.py`.

## 5. Sửa F1 — lệnh, exit và đối chiếu bảng dev (2026-10-11)

Thay đổi mã:
- `src/kc48/kiem_pf_diag_moc2.py`:
  - kiểm kê `os.walk` mọi `pf_diag`;
  - thêm nhóm test (`RE_RUN_TEST`, `DETRAC-test`, 40 video/run) với bảng riêng hậu tố `_test`;
  - hash trước/sau cho đủ 456 CSV + 24 `run_meta.json`;
  - hash bảng kết quả trước/sau khi ghi lại;
  - exit 1 nếu kiểm kê khác 24 run / 456 CSV, có thư mục không thuộc nhóm nào, hoặc bảng dev đổi.
- `src/kc48/test_kiem_pf_diag_moc2.py`: thêm test 5.

Lệnh (Python `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe`):
1. `src/kc48/kiem_pf_diag_moc2.py` bằng **script chưa sửa**: exit 0, đầu ra như bản đầu (`run 18, file 216, …, tai khoi
   tao 0`). Lần chạy này vô tình: tôi gọi `--help`, nhưng script không có argparse nên chạy luôn. Nó ghi lại 3 bảng dev
   từ cùng dữ liệu đầu vào.
2. `src/kc48/test_kiem_pf_diag_moc2.py` sau sửa: **exit 0, 10/10 PASS**. Sau khi sửa kiểm kê ở lệnh 3, test được chạy
   lại lần cuối: vẫn exit 0, 10/10 PASS.
3. `src/kc48/kiem_pf_diag_moc2.py` sau sửa, lần 1: **exit 1**. Kiểm kê thấy 25 thư mục; `results/phase2/pf_diag` (thư
   mục báo cáo cũ) bị tính như một run 0 CSV. Đã sửa: tách thư mục đó ra và ghi riêng. Lần chạy này vẫn ghi bảng và
   báo `bang dev doi so voi lan ghi truoc: 0`.
4. `src/kc48/kiem_pf_diag_moc2.py` sau sửa, lần 2: **exit 0**. Đầu ra:
   - `kiem ke: 24 run co pf_diag, 456 CSV, 24 run_meta; khong thuoc nhom: 0`;
   - `[dev] run 18, file 216, update 1664036, tai khoi tao 0`;
   - `[test] run 6, file 240, update 3400029, tai khoi tao 0`;
   - `lech max … (dev): 1.110e-16; loi bat bien: 0; file dau vao/cu doi: 0; bang dev doi so voi lan ghi truoc: 0 (co
     hash truoc: 3/3)`.

**Đối chiếu bảng dev.** sha256 của `theo_run.csv` `3344fc4e…`, `theo_tracker.csv` `c76c1ca9…` và `theo_n_lost.csv`
`a8c58a24…` giống nhau trước và sau khi script đã sửa ghi lại (`ket_qua.json` → `hash_bang_kq_truoc_khi_ghi` /
`hash_bang_kq_sau_khi_ghi`). Bảng 2a giữ nguyên từng số so với bản đầu.

Giới hạn của đối chiếu này: hash "trước" là của bảng do script **chưa sửa** ghi ở lệnh 1, chạy trên cùng đầu vào (hash
đầu vào không đổi). Bảng của bản đầu chưa được chụp hash riêng, và thư mục `kc48_pf_code/` chưa được git theo dõi. Vì
vậy bằng chứng "không đổi" là: cùng mã cũ + cùng đầu vào → cùng bảng → mã mới ghi lại giống từng byte, cộng với các số
in ra trùng bảng 2a của bản đầu.

`git status --porcelain -- results/phase2/kc48_pf_code results/phase2/pf_diag handoff/logs/pf_diag_report.log
src/pf_diag_report.py data/processed/trackers/DETRAC-all data/processed/trackers/DETRAC-test` chỉ cho
`?? results/phase2/kc48_pf_code/`, là thư mục kết quả mới của lượt R48. Việc không file cũ nào đổi dựa trên sha256 ở
mục 3 bước 4, vì `data/processed` không được git theo dõi.
