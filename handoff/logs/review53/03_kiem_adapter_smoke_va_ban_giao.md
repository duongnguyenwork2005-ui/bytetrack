# R53B — Kiểm adapter, khóa, smoke và bàn giao lượt đánh giá đầy đủ

Ngày 2026-10-20, Claude (người triển khai). Bước `R53B_LOCK_SMOKE`, lượt orchestrator mới chỉ gồm R53B. Kết thúc tại
cổng review Codex. **Đánh giá đầy đủ chưa chạy; tracking chưa chạy.** Chưa commit hay stage.

## 0c. Vòng 5 — sửa F2 [P2] sau review vòng 4: cổng nguồn so đường dẫn adapter (Phụ lục 3), 2026-10-20 10:40–10:48

Bối cảnh: review vòng 4 (`reports/review_004.json` của lượt `bytetrack_r53b_lock_smoke_20261020_0356`) = **blocked**.
Reviewer đã tự đối chiếu: hash khóa gốc/PL1/PL2, luồng chấm lại (cổng GT, NPZ, thứ tự ảnh, scorer cũ, không suy luận),
log fixture phục hồi, ba lệnh bàn giao, PLAN. Còn **F2 [P2]**: `chuan_bi_danh_gia.py:905` chỉ so `Path(x).name` của
adapter, nên `D:/other_checkout/src/danh_gia_detector_bdd_val.py` vẫn qua. Đây là vòng sửa-review 5/5 của ngân sách bổ
sung (vòng 3 blocked khi hết 3/3; người dùng duyệt Pdaa0fd9d kèm ngân sách bổ sung; vòng 4 blocked). PLAN mục 0 đã cập
nhật trước báo cáo này. Phạm vi giữ đúng Pdaa0fd9d: không bảng so sánh, không kiểm DETRAC, không GPU/WMI/full/tracking.

### Sửa

- `cwd_giam_sat(tom, tt)` (mới): cwd của lượt BDD lấy từ `full/_giam_sat/bdd_val__<tên>/trang_thai.json`. Đây là file
  `src/chay_giam_sat.py:208,266` ghi `"cwd": os.getcwd()`, còn `tom_tat.json` (`:260`) không có cwd. `lenh` của hai file
  phải trùng, `cwd` phải là đường dẫn tuyệt đối; thiếu file → `FileNotFoundError` → exit 2. Bản cũ dùng `tom.get("cwd") or
  repo`, tức với dữ liệu thật luôn ngầm lấy cwd = repo. Cách đó cũng sai khi lượt chạy thật ở thư mục khác.
- `kiem_nguon_bdd(..., cwd)`: phần tử lệnh trùng tên adapter được resolve theo cwd đó và phải bằng
  `(repo / "src/danh_gia_detector_bdd_val.py").resolve()`; khác → `LoiChuanBi` → exit 2. Nội dung file adapter trong repo
  do cổng khóa (SHA256SUMS gốc) kiểm. `--model`, `--out` và đối số dữ liệu cũng resolve theo cùng cwd.
- `ket_qua.json` chấm lại ghi thêm `cham_lai.adapter_goc`, `cham_lai.cwd_giam_sat`.
- Khóa: `buoc_phu_luc_2` được gộp thành `buoc_phu_luc_noi(ch, k)`, thêm `--buoc phu-luc-3` (mức 3 nối tiếp mức 2, bản
  trước của mã lấy từ snapshot `review_points/0009_review/files`, kiểm hash). Hành vi `phu-luc-2` không đổi (nhóm P vẫn đạt).
- Test: fixture `tom_tat.json`/`trang_thai.json` theo đúng dạng `chay_giam_sat`. Thêm X12 (6 ca) và Q1–Q8.
- Không sửa adapter, scorer, `detector_classes.py`, `chay_giam_sat.py`; hash bảy file mã cũ không đổi (`ket_qua_test.json`
  `ma_sha256` + `kiem-khoa` đạt).

### Lệnh đã chạy (mỗi lệnh đứng riêng, foreground)

| # | Lệnh | Exit | Bằng chứng |
|---|---|---|---|
| 1 | `python src/r53_detector/test_chuan_bi_danh_gia.py` | 1 | 119/120, `tests_eval/lan_20261020_104531_789341_pid16144/`: X12 "tương đối + cwd khác" dùng `--lan 9` trùng thư mục của X0 → wrapper từ chối ghi đè (exit 1 thay vì 2). Lỗi ở test |
| 2 | như trên, sau khi đổi số lần chạy của X12 sang 21–26 (wrapper không đổi) | 0 | **120/120**, `tests_eval/lan_20261020_104602_061598_pid15436/ket_qua_test.json` (sha256 `caf2903a…3d61`) |
| 3 | `python src/r53_detector/chuan_bi_danh_gia.py --buoc kiem-khoa` (trước phụ lục 3) | 2 | `[LOI KHOA] src/r53_detector/chuan_bi_danh_gia.py: sha256 khac phu_luc_2` — đúng kỳ vọng |
| 4 | `python src/r53_detector/chuan_bi_danh_gia.py --buoc phu-luc-3` | 0 | `[DAT] phu luc 3: 3 file doi sau phu luc 2 (...02..., chuan_bi_danh_gia.py, test_chuan_bi_danh_gia.py); SHA256SUMS phu luc 3 ab1a397c18a9f1e4` |
| 5 | `python src/r53_detector/chuan_bi_danh_gia.py --buoc kiem-khoa` | 0 | `25 file khop SHA256SUMS (e1442ea6…250d), khoa luc 2026-10-20T04:14:32; chuoi phu luc dat (phu_luc_1, phu_luc_2, phu_luc_3)` |

(`python` = `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe`.)

### Fixture mới (qua CLI `main()` của wrapper; log từng ca ở `.../fixture_phu_luc/log_cli/X12_*.txt`, `Q*.txt`)

- X12_21: bản sao **nguyên byte** của adapter BDD ở `fixture_phu_luc/checkout_khac/src/danh_gia_detector_bdd_val.py`, gọi
  bằng đường dẫn tuyệt đối → exit 2 `khong phai adapter BDD trong repo`, không `ket_qua.json`.
- X12_22: `src\danh_gia_detector_bdd_val.py` tương đối, `trang_thai.json` cwd = `checkout_khac` → exit 2.
- X12_23 thiếu `cwd`; X12_24 `lenh` trong `trang_thai.json` khác `tom_tat.json`; X12_26 thiếu `trang_thai.json` → exit 2.
- X12_25: đường dẫn tuyệt đối đúng adapter của repo → exit 0, `adapter_goc` = `src/danh_gia_detector_bdd_val.py`,
  `cwd_giam_sat` = repo. Không ca nào gọi dự đoán giả lập.
- Q1–Q8: phụ lục 3 không đổi gì, thiếu bản PL2 trong snapshot, sửa phần trước mốc, thiếu chuỗi → exit 2; đạt → exit 0
  (bản trước = hash PL2, trỏ PL2); tạo lại PL3/PL2 → exit 1; sửa mã, 02, bản sao PL3, SUMS PL2 → `kiem-khoa` exit 2;
  `cham-lai-bdd` sau PL3 → exit 0, chuỗi khóa bốn mức, `theo_lop` trùng lần chấm đầu.
- Mọi ca cũ (A, M, T, K, S, H, E, C, R, F, L, P, X0–X11) vẫn đạt trên mã mới.

### Khóa phụ lục 3 (`results/phase2/r53_detector/chuan_bi/phu_luc_3/`)

- `thoi_diem_phu_luc_3.json`: 2026-10-20T10:47:08, `SHA256SUMS` 16 file `ab1a397c18a9f1e496136833b28db049e71535738d2f3f2bedd27b3cc797bfc8`,
  `khoa_phu_luc_3.json` `c8ae82216ebcca2275c2b478332a455caf85b7105cb3c17a7c98a5bdc1782640`, trỏ PL2 `195c919a…95ab`.
- Ba file đổi so với PL2: tài liệu 02 (`b18a7636…3276` → `2492ee0a…620b`, chỉ nối sau `## Phụ lục 3`), wrapper
  (`535d53cc…56cf` → `dfa2665dddb9363df05266542ec7321a4f588011fe26d3a0c3143394eb6163dc`), test (`8d705a88…3022` →
  `2c59a6e49b471545a303588a1b92fea6daff7832812dd1c1058b84da0acc0a12`). Bản PL2 của hai file mã ở `phu_luc_3/ban_khoa_truoc/`.
- Khóa gốc, `phu_luc_1/`, `phu_luc_2/`, `smoke/` không bị ghi.

### Giới hạn

- Cổng nguồn chỉ chứng minh lệnh giám sát **ghi** đường dẫn adapter của repo và adapter đó khớp khóa **lúc chấm lại**.
  Nó không chứng minh adapter không bị sửa rồi khôi phục trong lúc lượt dài chạy. Điều đó dựa vào quy trình mục 8
  (kiểm hash trước chạy) và `kiem-khoa` trước/sau lượt dài.
- Không kiểm `lenh[0]` (python) của lệnh giám sát; ngoài phạm vi F2 [P2].
- Ba lệnh chấm lại thật **chưa chạy** (chưa có thư mục full); chỉ kiểm trên fixture tổng hợp.
- Fixture `trang_thai.json`/`tom_tat.json` dựng theo dạng mã `chay_giam_sat.py`, không phải đầu ra của một lượt WMI thật.

## 0b. Vòng 4 — bước hẹp Pdaa0fd9d: Phụ lục 2, chấm lại BDD từ NPZ (đóng F2), 2026-10-20 06:43–06:47

Bối cảnh: vòng 3 kết thúc **blocked** (hết ngân sách 3/3 vòng sửa-review, F2 còn mở). Người dùng (Codex chọn thay theo
ủy quyền khi ngủ) duyệt bước hẹp Pdaa0fd9d với ngân sách bổ sung; vòng này là 4/4. P1517bbbd (bảng so sánh, kiểm DETRAC)
và P56df4991 (+2 vòng) vẫn bị từ chối. PLAN mục 0 đã cập nhật trước báo cáo này. Lời gọi Claude trước đó (0007_coder)
dừng vì hết hạn mức tài khoản sau khi thêm hằng số / lệnh bàn giao và tổng quát hóa `kiem_chuoi_phu_luc`; lời gọi này
làm tiếp phần còn lại.

**Thay đổi mã** (chỉ hai file trong write_paths; adapter, scorer, `detector_classes.py` không đổi):
- `src/r53_detector/chuan_bi_danh_gia.py` `6fc0c72a…94e4` → `535d53cca5cebac7c8d47179f715f77b284efca05b1316876a938d2878f756cf`:
  - `--buoc cham-lai-bdd --model <tên> [--lan N]` (`buoc_cham_lai_bdd`, `_cham_lai_bdd`, `bang_chung_npz`,
    `kiem_nguon_bdd`, `doi_chieu_goc`). Thứ tự kiểm và đầu ra: tài liệu 02 PL2.2. Đích `full/bdd_val/<tên>__cham_lai/` mới,
    đã có → exit 1; lỗi khóa / nguồn / hash / dữ liệu / danh sách–thứ tự ảnh → exit 2, không ghi `ket_qua.json`; đầu vào
    đổi hoặc khác `ket_qua.json` gốc → exit 3. `du_doan_ultralytics` của hai adapter bị thay bằng hàm chặn trong lúc chấm.
    Grep: trong adapter, scorer và các module mã khóa, `ultralytics` chỉ được import bên trong hàm dự đoán
    (`eval_detector_5lop.py:269`), không ở cấp module; `ket_qua.json` ghi `module_da_nap` (ultralytics, torch) để đối
    chiếu khi chạy thật — chưa có lần chạy thật nào của bước này.
  - `--buoc phu-luc-2` (`buoc_phu_luc_2`): khóa nối tiếp phụ lục 1 (tài liệu 02 PL2.5).
  - `lenh_ghi_bang_chung_npz`: thêm `-Width 500` để dòng `Path` không bị ngắt (chưa nằm trong khóa nào trước đây).
  - `kiem-khoa` in tên các mức phụ lục.
- `src/r53_detector/test_chuan_bi_danh_gia.py` `4e1f2235…2fbf` → `8d705a88804e25fa339e0754c1f63ecfee72a344b3bd7f49ca51f9bcee2f3022`:
  nhóm P (P1–P8, phụ lục 2) và X (X0–X11, chấm lại), trong `nhom_l` sau L11; danh sách ca: tài liệu 02 PL2.6.

**Lệnh đã chạy** (mỗi lệnh đứng riêng; không GPU, không WMI, không full, không tracking):

| Lệnh | Exit | Bằng chứng |
|:--|:--|:--|
| `test_chuan_bi_danh_gia.py` lần 1 (06:43:48) | 1 (101/102: điều kiện P3 sai, sửa phần khóa gốc nên lỗi ở cổng phụ lục 1 trước) | `tests_eval/lan_20261020_064348_930427_pid944/` |
| `test_chuan_bi_danh_gia.py` lần 2 (06:44:20), tách P3a / P3 | 0 (103/103); dự đoán giả lập BDD cho AP = 0 | `tests_eval/lan_20261020_064420_833829_pid23204/` |
| `test_chuan_bi_danh_gia.py` lần 3 (06:45:06), mã cuối, dự đoán giả lập bám GT (mAP50 1,0; mAP50-95 0,48) | 0 (103/103) | `tests_eval/lan_20261020_064506_842269_pid20924/` (`ket_qua_test.json` `a2ea5544…63a6`) |
| `chuan_bi_danh_gia.py --buoc phu-luc-2` (06:46:55) | 0: 3 file đổi sau phụ lục 1 | `chuan_bi/phu_luc_2/` |
| `chuan_bi_danh_gia.py --buoc kiem-khoa` | 0: 25 file khóa gốc khớp; chuỗi `phu_luc_1, phu_luc_2` đạt | stdout |

Log và exit của từng lệnh CLI trong fixture: `<lần test>/fixture_phu_luc/log_cli/` (P*, X*, adapter) và `log.txt` +
`exit.json` trong mỗi thư mục `full/bdd_val/<tên>__cham_lai*/` của fixture.

**Khóa phụ lục 2** (`results/phase2/r53_detector/chuan_bi/phu_luc_2/`): `SHA256SUMS` 13 file
`195c919a5984dab75da1a34a90b75cdf9d8dff0583ea4a79e82158873e8795ab`; `khoa_phu_luc_2.json`
`2936c8972c3389c40e35123716a704c62b698df5140c1ea1c8473cab5a2c2277`; `thoi_diem_phu_luc_2.json` (06:46:55, trỏ SHA256SUMS
gốc `e1442ea6…250d` và phụ lục 1 `2acb9db4…78cd`). Tài liệu 02 lúc khóa: `b18a7636e030a907ae34ead16a354dfcba3c2d89f2763bb3f73ee091953c3276`,
tiền tố trước `## Phụ lục 2` = bản phụ lục 1 `5e27b4b5…ddb5`. Bản phụ lục 1 của hai file mã lấy từ snapshot review
`0004_review` / `0006_review` của orchestrator (đã kiểm hash), chép vào `phu_luc_2/ban_khoa_truoc/`. Khóa gốc, phụ lục 1 và
`smoke/` không bị ghi.

**Tiêu chí bước — đối chiếu:**
- Chấm lại từ NPZ trùng từng trường với lượt thường: X1 (trừ `anh_xa_lop`, `thoi_gian_s` không có trong NPZ; `theo_lop.csv`
  trùng từng byte), X4 (đối chiếu `ket_qua.json` gốc: không trường nào khác).
- NPZ bị sửa → 2 (X5); sai danh sách / thứ tự ảnh → 2 (X6, hai ca, hash bằng chứng vẫn khớp); đích đã có → 1 (X3).
- Phục hồi sau lỗi chấm: X1 (exit gốc 1, không `ket_qua.json`) và X7 (khôi phục NPZ → thư mục mới `__lan4`).
- Không suy luận, đầu vào không đổi: X2 (dự đoán giả lập không được gọi, `goi_suy_luan` 0, cây đầu vào + dữ liệu fixture
  băm trước/sau trùng).
- Test chạy qua `main(argv, ch)` (parser CLI của wrapper) và `main()` của adapter BDD gốc; log/exit trong `tests_eval/`.
- Tài liệu 02 PL2.3 có ba lệnh `cham-lai-bdd` cụ thể — chỉ bàn giao, **chưa chạy**. `kiem-khoa` đạt cả ba mức.

**Giới hạn:**
- Chưa chạy `cham-lai-bdd` trên dữ liệu thật (không có thư mục full; tạo thư mục đích dưới `full/` nằm ngoài write_paths).
  Thời gian chấm 39.973 ảnh chưa đo.
- `tom_tat.json` trong fixture là bản tổng hợp theo dạng `chay_giam_sat` (lệnh trỏ adapter BDD gốc); không có lần chạy
  `chay_giam_sat` thật nào bọc adapter thật ở bước này (F4 của phụ lục 1 bọc tiến trình `--con-adapter`).
- Bằng chứng hash NPZ ghi sau khi adapter kết thúc (PL2.4), không trong tiến trình adapter; khoảng giữa đối chiếu bằng
  `LastWriteTime` và log giám sát.
- Không commit; người dùng quyết.

## 0a. Vòng 3, sau review vòng 2 (`review_002.json`, blocked), 2026-10-20

Review vòng 2: F1, F3 đóng; F2 còn mở, vì tài liệu 02 PL1.4 chưa có lệnh chấm lại BDD từ NPZ trên dữ liệu thật.
Reviewer cũng nêu trạng thái `Ultralytics/settings.json` đã cũ.

**Vòng này không đổi mã, khóa, tài liệu 02 hay `smoke/`.** Chỉ sửa PLAN mục 0 và mục 0a này.

**F2 vẫn mở. Lý do: muốn đóng, phải có mã mới mà người dùng chưa duyệt.**
- Không có CLI cũ nào chấm lại BDD từ NPZ:
  - `src/danh_gia_detector_bdd_val.py` không có `--cham-lai` (parser dòng 175–189);
  - `src/eval_detector_5lop.py main()` (dòng 315–324) chỉ chấm sau khi suy luận.
- Lệnh đó chỉ có được nếu viết mã mới gọi `bv.kiem_va_doc_gt` + `eval_detector_5lop.danh_gia`. Mã này đúng là phần lõi
  của P1517bbbd, đề xuất người dùng đã từ chối. Wrapper `chuan_bi_danh_gia.py` cũng không được thêm thao tác ngoài
  kế hoạch (ràng buộc "chỉ các thao tác bàn giao, tiền kiểm, khóa và smoke").
- Tôi không tự thêm công cụ dưới một tên khác, và không tự bỏ tiêu chí 7. Người dùng chọn phương án (PLAN mục 0, "Cần
  người dùng quyết" (1)):
  - (a) duyệt riêng một bước chấm lại BDD từ NPZ, chỉ đọc NPZ, ghi ra thư mục mới;
  - (b) bỏ tiêu chí 7 cho BDD; nếu chấm lỗi thì chạy lại vào `__lan2`;
  - (c) hướng khác.

**`Ultralytics/settings.json`.** Kiểm lúc 20/10: Glob `Ultralytics/**` trong repo không ra file nào, và
`git status --short -- Ultralytics results/phase2/r53_detector/full` không in gì. Nghĩa là file đã không còn, và cũng
chưa có `full/`. Agent không xóa file này; chưa rõ ai đã xóa. Các dòng "chờ xóa" ở mục 0 bên dưới và tài liệu 02 PL1.5
giờ đã cũ. Tài liệu 02 được giữ nguyên để khóa phụ lục 1 không đổi.

**Lệnh đã chạy vòng 3 (mỗi lệnh đứng riêng):**

| # | Lệnh | Exit | Bằng chứng |
|:--|:--|:--|:--|
| 1 | `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc kiem-khoa` | 0 | stdout `[DAT] kiem-khoa: 25 file khop SHA256SUMS (e1442ea6…250d), khoa luc 2026-10-20T04:14:32; phu luc 1 dat` |
| 2 | `git status --short --untracked-files=normal -- Ultralytics results/phase2/r53_detector/full` | 0 | không in gì |

Không chạy test, GPU, full, smoke hay tiền kiểm ở vòng này. Mã không đổi nên test 74/74 lần 4
(`tests_eval/lan_20261020_051609_986864_pid5104/`) vẫn ứng với mã hiện tại; reviewer vòng 2 đã đối chiếu hash.

**Trạng thái:** R53B chưa nghiệm thu, chờ người dùng quyết F2. Đánh giá đầy đủ chưa chạy; tracking chưa chạy.

## 0. Vòng sửa sau review vòng 1 (F1–F3), 2026-10-20 05:13–05:18

Review vòng 1 (`review_001.json`) trả về revise với F1–F3. Mục 1–6 bên dưới là báo cáo vòng 1; mục này ghi phần sửa,
không viết lại các mục đó. Không chạy GPU, không chạy full, không chạy lại smoke hay tiền kiểm. Chi tiết và 6 dòng lệnh
nằm ở tài liệu 02, **Phụ lục 1**.

**Mã.** Chỉ hai file trong `write_paths` thay đổi. Adapter và scorer không đổi.
- `chuan_bi_danh_gia.py` → `6fc0c72a4d0bc34fc8c616f77dc44b47f949575f93152f4bc7812540249194e4`. Thêm:
  - `moi_truong_doc_lap`, `doi_so_day_du`, `dong_lenh_doc_lap`, `lenh_doc_lap`;
  - bước `phu-luc`;
  - `kiem_khoa` kiểm theo chuỗi (gốc + phụ lục);
  - `lenh-day-du` in thêm 6 dòng lệnh khởi chạy độc lập.
- `test_chuan_bi_danh_gia.py` → `4e1f2235803151527d8985bb136874e6abc7225e1b2836ab2cdb392ca5622fbf`. Thêm các nhóm
  E, C, R, F, L và chế độ con `--con-adapter`. Chế độ con chỉ nhận cấu hình trong `tests_eval/`.

**F1 — CLI hai adapter gốc.** Nhóm C gọi `main()` và parser của chính adapter, với đối số `doi_so_day_du`; E1 kiểm
đối số này trùng 6 lệnh mục 7. Kết quả: C1/C2 exit 0, đủ ảnh và đầu ra, batch 16, device "0", conf 0,25, giới hạn 0.
Sai sha256 → 2; `--out` đã có → 1; ảnh đổi → 2. Mỗi kiểm chạy cho cả hai adapter; khi bị chặn, không dự đoán và không
tạo thư mục. Log và exit: `fixture_cli/log_cli/`.

**F2 — NPZ đã lưu nhưng chấm lỗi.**
- R1–R3 (UA-DETRAC): còn NPZ và nguồn, sha256 khớp; `--cham-lai` exit 0, D3 trùng lượt thường; NPZ bị sửa → 2.
- R4–R6 (BDD): còn NPZ, không có `ket_qua.json`/CSV; log có `[du lieu]`. Chấm lại bằng hàm cũ trên NPZ trùng mọi trường
  với lượt thường; NPZ không đổi. Adapter BDD không có `--cham-lai`.
- F4 lặp lại trường hợp BDD qua bộ giám sát: exit 1 thật, log giữ traceback.
- Quy trình cụ thể ở Phụ lục 1, PL1.4: lệnh PowerShell ghi bằng chứng NPZ và bằng chứng nguồn.
- **Giới hạn do quyết định người dùng:** P1517bbbd (công cụ hậu kiểm chấm lại BDD từ NPZ) đã bị từ chối, nên không tạo
  lệnh chấm lại BDD trên dữ liệu thật. Khi BDD lỗi ở bước chấm, quy trình dừng và hỏi người dùng. Tiêu chí 7 mục 10
  với BDD cũng chờ quyết định này.

**F3 — môi trường khởi chạy độc lập.** WMI không kế thừa biến của phiên đang mở. Vì vậy 6 dòng `cmd.exe /d /c` đặt 7
biến: `PYTHONDONTWRITEBYTECODE`, `PYTHONUTF8`, `PYTHONIOENCODING`, `PYTHONUNBUFFERED`, `YOLO_CONFIG_DIR` và
`MPLCONFIGDIR` (cùng trỏ `full\_cau_hinh_thu_vien`), `YOLO_OFFLINE`. Sau đó dòng lệnh tạo thư mục cấu hình rồi chạy
`chay_giam_sat` và adapter, cả hai bằng `python -B`.

Test F1–F3 chạy đúng chuỗi lệnh trên, với môi trường đã bỏ 7 biến này. Kết quả:
- bytecode tắt, cả cờ và biến;
- UTF-8 bật, 7 biến đúng giá trị;
- cấu hình ultralytics nằm trong vùng đầu ra;
- không có `.pyc` mới trong `src/`.

**Sự cố trong vòng sửa (đã ghi, chưa dọn):** lần test 1 (05:13) dùng `YOLO_CONFIG_DIR` chưa tồn tại. Ultralytics lặng
lẽ dùng `<cwd>\Ultralytics\`, nên đã tạo `D:\LABOTARY\bytetrack\Ultralytics\settings.json` ở gốc repo, ngoài vùng
ghi; `git status` hiện thấy `?? Ultralytics/`. Tôi không xóa vì không có quyền và đây là quyết định của người dùng.
Lỗi đã sửa bằng bước `mkdir` trong dòng lệnh. Lỗi này cũng cho thấy F3 có thật: lượt dài chạy theo lệnh cũ cũng sẽ
ghi cấu hình ra ngoài vùng ghi.

**Lệnh đã chạy (mỗi lệnh đứng riêng):**

| # | Lệnh (`C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe …`) | Exit | Bằng chứng |
|:--|:--|:--|:--|
| 1 | `src/r53_detector/test_chuan_bi_danh_gia.py` | 1 (73/74: F2 lộ lỗi thư mục cấu hình) | `tests_eval/lan_20261020_051353_550120_pid20388/` |
| 2 | `src/r53_detector/test_chuan_bi_danh_gia.py` | 1 (73/74: điều kiện F2 chưa tính thư mục con `Ultralytics`) | `tests_eval/lan_20261020_051447_323753_pid13972/` |
| 3 | `src/r53_detector/test_chuan_bi_danh_gia.py` | 0 (74/74), mã trước khi thêm in lệnh | `tests_eval/lan_20261020_051506_648198_pid2788/` |
| 4 | `src/r53_detector/chuan_bi_danh_gia.py --buoc kiem-khoa` | 2 (`chuan_bi_danh_gia.py: sha256 khac khoa`); cổng gốc vẫn chặn | stdout |
| 5 | `src/r53_detector/chuan_bi_danh_gia.py --buoc lenh-day-du` | 0; 6 dòng chép vào PL1.5 | stdout |
| 6 | `src/r53_detector/test_chuan_bi_danh_gia.py` | **0 (74/74), mã cuối** | `tests_eval/lan_20261020_051609_986864_pid5104/ket_qua_test.json` (`ad5f8785…d3fa`) |
| 7 | `src/r53_detector/chuan_bi_danh_gia.py --buoc phu-luc` | 0; 3 file đổi; `SHA256SUMS` phụ lục `2acb9db4…78cd` | `chuan_bi/phu_luc_1/` |
| 8 | `src/r53_detector/chuan_bi_danh_gia.py --buoc kiem-khoa` | **0**: 25 file gốc + chuỗi phụ lục đạt | stdout |

**Reviewer nên đối chiếu:**
- tiền tố byte của tài liệu 02 trước `\n## Phụ lục 1` có sha256 `7d1f42e3…e2f5`;
- `chuan_bi/phu_luc_1/ban_khoa_goc/` có hash `ffef2f84…`/`806980c6…` = snapshot `review_points/0002_review/files`;
- `SHA256SUMS`, `khoa_danh_gia.json`, `thoi_diem_khoa.json` gốc và `smoke/` không đổi;
- không có `results/phase2/r53_detector/full/`.

## 1. Mã mới (chỉ trong `src/r53_detector/`)

**`chuan_bi_danh_gia.py`** (sha256 `ffef2f8474248dea340b733ef7e46e4cbad5014a656d843f12a802f322b7f794`). Có năm bước
cố định; không nhận lệnh hay đường dẫn tùy ý:

| Bước | Việc làm |
|:--|:--|
| `tien-kiem` | gọi `kiem_va_doc_gt` của adapter (băm lại mọi ảnh), so số lượng/danh tính với hồ sơ, chọn mẫu và ghi |
| `lenh-day-du` | chỉ in |
| `khoa` | tạo khóa |
| `kiem-khoa` | chỉ đọc |
| `smoke` | chạy smoke một model trên một tập |

- Smoke gọi nguyên các hàm cũ: `du_doan_ultralytics` (qua module adapter), `luu_du_doan`, `danh_gia`,
  `quy_tac_hieu_luc`, `chuan_hoa_d3`, `doc_du_doan`, `ghi_nguon`, `ghi_ket_qua`, `chan_doan`.
- Không file cũ nào bị sửa. Hash adapter, evaluator, `detector_classes`, `fingerprint_detrac_test` và
  `bdd_yolo_export` trùng đúng giá trị ghi ở R47 (xem `khoa_danh_gia.json` → `ma`).
- Script đặt `PYTHONDONTWRITEBYTECODE` và UTF-8 trước khi import. Cấu hình ultralytics nằm ở
  `results/phase2/r53_detector/smoke/_cau_hinh_thu_vien/` (`YOLO_CONFIG_DIR`). Log smoke xác nhận settings được tạo ở
  đó, không ở hồ sơ người dùng.

**`test_chuan_bi_danh_gia.py`** (`806980c6c3401618cda90e2be9aec0169ede0fa24896cc10cd7bd14b101ec2f3`).
- Chạy trên fixture tổng hợp: UA-DETRAC 2 video, 5 ảnh, 1 ảnh D4; BDD 2 video, 6 ảnh, đủ năm lớp.
- Dự đoán giả lập; không dùng dữ liệu thật, không GPU.
- Mỗi lần chạy ghi vào thư mục mới `results/phase2/r53_detector/tests_eval/lan_<thời điểm>_pid<pid>/`.
- Fingerprint của fixture UA-DETRAC được dựng bằng hàm của mốc A (`fpt.bam_anh`, `csv_bytes`, `sha256_thu_tu`,
  `sha256_ds`). Lý do: script mốc A từ chối `--out` trong `results/` (cây chỉ đọc), mà ngoài repo thì không được ghi.

Nhóm kiểm:
- **A**: ánh xạ COCO-80 / 5 lớp theo tên; mixed5 trả về checkpoint COCO → exit 3.
- **M**: chọn mẫu. Ảnh đầu tiên có GT từng lớp, loại trùng, bổ sung tới 64, giữ thứ tự. Ít ảnh thì không vượt số có.
  Thiếu lớp → lỗi.
- **T**: tiền kiểm. Lệch hồ sơ → exit 2, không ghi. Chạy lại → exit 1, file không đổi.
- **K**: khóa. Thiếu tài liệu, thiếu một lệnh, hoặc còn `TODO` → exit 2, không tạo khóa. Khóa lại → exit 1.
- **S**: smoke. Trước khóa → exit 2, không dự đoán. Sáu smoke giả lập exit 0.
  - Tham số dự đoán đúng khóa; cùng mẫu; NPZ/hash/nguồn đầy đủ.
  - D3: N/A trong JSON và CSV; `mAP (car, bus)` bằng scorer gọi trực tiếp. BDD đủ năm lớp.
  - Chạy lại → exit 1.
- **H**: cổng hash. Sửa mẫu, checkpoint, `SHA256SUMS`, tài liệu 02, hoặc nội dung một ảnh (UA-DETRAC và BDD) → exit 2,
  **không gọi dự đoán**, và `kiem-khoa` cũng exit 2. Khôi phục đúng byte → exit 0.

## 2. Lệnh đã chạy (theo thứ tự, mỗi lệnh đứng riêng)

| # | Lệnh (`C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe ...`) | Exit | Bằng chứng |
|--:|:--|--:|:--|
| 1–4 | `src/r53_detector/test_chuan_bi_danh_gia.py` (trước khóa, 4 lần trong lúc sửa test) | 1, 0, 0, 0 | lần 1 lỗi fixture (fingerprint mốc A từ chối `--out` trong `results/`) → đổi cách dựng fixture; 36/36 → 38/38 → 38/38; `tests_eval/lan_20261020_0407*`, `_040809_*`, `_040824_*`, `_041134_*` |
| 5 | `... --buoc tien-kiem --tap detrac_test` | 0 | 40 video, 56.167 ảnh đánh giá, kiểm 64 s; `chuan_bi/tien_kiem_detrac_test.json`, `mau_smoke_detrac_test.json` |
| 6 | `... --buoc tien-kiem --tap bdd_val` | 0 | 200 video, 39.973 ảnh, kiểm 82 s; `chuan_bi/tien_kiem_bdd_val.json`, `mau_smoke_bdd_val.json` |
| 7, 9 | `... --buoc lenh-day-du` (chỉ in) | 0 | 3 checkpoint khớp; hash mã; căn cứ dự toán; ổ D còn 42,6 GB |
| 8, 10 | `test_chuan_bi_danh_gia.py` (sau khi thêm `kiem-khoa` và làm luồng Tee chắc hơn) | 0 | 38/38, `tests_eval/lan_20261020_041134_*`, `_041211_*` |
| 11 | `... --buoc khoa` | 0 | khóa lúc **2026-10-20T04:14:32**; 25 file trong `SHA256SUMS` |
| 12 | `test_chuan_bi_danh_gia.py` (sau khóa; test nằm trong khóa, không đổi) | 0 | 38/38, `tests_eval/lan_20261020_041435_*` |
| 13 | `... --buoc kiem-khoa` | 0 | 25/25 file khớp |
| 14–19 | sáu lệnh `--buoc smoke` (mục 4) | 0 ×6 | `smoke/<tập>/<model>/` |

Chưa chạy:
- sáu lệnh đánh giá đầy đủ;
- chấm lại đầy đủ;
- tracking;
- review độc lập.

## 3. Tiền kiểm và mẫu

Cả hai tập khớp hồ sơ: R47 cho UA-DETRAC, R43 cho BDD. Danh tính đầy đủ nằm ở tài liệu 02 mục 2 và
`khoa_danh_gia.json` → `du_lieu`.
- Không fingerprint, bảng hash hay manifest nào bị cập nhật.
- Mẫu UA-DETRAC (64 ảnh) gồm ảnh car đầu tiên `MVI_39031/img00001.jpg` và ảnh bus đầu tiên `MVI_39051/img00616.jpg`
  (chỉ số 2085), cộng 62 ảnh đầu chưa chọn. GT trong mẫu: car 104, bus 1.
- Mẫu BDD (64 ảnh) gồm car/truck/motorcycle tại ảnh 0, bus tại ảnh 55, bicycle tại ảnh 606. GT trong mẫu: car 1.349,
  truck 48, bus 8, motorcycle 63, bicycle 1.
- Danh sách và sha256 từng ảnh được ghi lúc tiền kiểm, trước suy luận, rồi đưa vào khóa.

## 4. Khóa và smoke

Khóa nằm ở `results/phase2/r53_detector/chuan_bi/`:
- `khoa_danh_gia.json` (`75578d2b2d4145139967e21da8f83c43b5cd261fb87831e938eef01b15b29041`): ba model, dữ liệu, mã,
  môi trường, quy tắc, mẫu, đầu ra, tiêu chí, lệnh;
- `SHA256SUMS` (`e1442ea6b02f74440cb5a3b1903ac897cde7dcdccc536bdaab8a979ac8a4250d`), 25 file;
- `thoi_diem_khoa.json`.

Tài liệu khóa `02_ke_hoach_khoa_danh_gia.md` có sha256 `7d1f42e37c9cf4dae48d0a2ba0eddd9e46350e1312add80125ace5e1df24e2f5`.

Sáu smoke chạy tuần tự lúc 04:14:39–04:23:19, mỗi lệnh một model, GPU NVIDIA GeForce GTX 1650. Mọi lệnh exit 0 và
`kiem` đều đạt:
- khóa đạt;
- checkpoint đúng;
- danh tính dữ liệu bằng khóa;
- mẫu bằng khóa;
- 64/64 ảnh được dự đoán;
- tên ảnh trong NPZ đúng;
- chấm lại từ NPZ trùng;
- loại checkpoint đúng;
- D3 đạt (UA-DETRAC) hoặc đủ năm lớp có GT (BDD).

| Tập / model | Kiểm dữ liệu (s) | Dự đoán 64 ảnh (s) | Loại checkpoint | NPZ sha256 |
|:--|--:|--:|:--|:--|
| detrac_test / yolov8n_coco | 58 | 52,6 (gồm khởi tạo CUDA lần đầu) | coco80 | `32613541e3075978272a7f5ea4954018bb106b662533bcc727a6270c49cd5a57` |
| detrac_test / yolov8n_mixed5 | 56 | 8,5 | lop5 | `f3eeeb07d68c8c38e5a3355fae72d582c35259affdc3239db731b4a0a5654c48` |
| detrac_test / yolo11n_mixed5 | 57 | 8,4 | lop5 | `24c53fdb5abebd4ef5ce36adddca5b016a7fdd0abd2d419a9a11ce860a7132f9` |
| bdd_val / yolov8n_coco | 76 | 9,0 | coco80 | `f1d71827f6acb48ac8d40b36ceb01a1feb198e150908154ade74bee4681d3bc7` |
| bdd_val / yolov8n_mixed5 | 77 | 8,5 | lop5 | `2f265ff1f8014d8470b2568c714b25bff412620e5d284601bc062411da6abe3f` |
| bdd_val / yolo11n_mixed5 | 77 | 8,5 | lop5 | `63998b510de25f8c56ab22c47b24edec77233b5d0b08c1ed5567d0bf6cd34938` |

Mỗi thư mục có:
- `log.txt` và `exit.json`;
- `danh_sach_anh.json`, ghi trước suy luận;
- `du_doan_<tên>.npz` và `nguon_du_doan_<tên>.json`;
- `ket_qua.json`, có trường `canh_bao` "smoke kỹ thuật";
- CSV: UA-DETRAC có `theo_lop.csv`, `tong_hop.csv`, `so_lieu_tho_khong_phai_metric.csv`; BDD có `theo_lop.csv`.

`smoke/.gitignore` loại `*.npz`. Gốc `.gitignore` vốn đã loại `results/phase2/**/*.npz` và không bị sửa.

**AP smoke không được trích dẫn hay so sánh ở đây.** Đây là kiểm kỹ thuật trên tối đa 64 ảnh, không dùng để xếp hạng
hay chỉnh tham số. Không tham số nào đổi sau smoke.

## 5. Bàn giao lượt đầy đủ

Tài liệu 02 mục 7–10 có:
- sáu lệnh đầy đủ, dùng hai adapter gốc với `--model`, `--sha256` cụ thể, `--batch 16 --device 0 --conf-van-hanh 0.25
  --gioi-han-anh 0`, và thư mục mới `results/phase2/r53_detector/full/<tập>/<model>`;
- quy trình chạy riêng: `kiem-khoa`, rồi bộ giám sát `chay_giam_sat.py` qua WMI như R43/R47, tuần tự, thư mục giám
  sát riêng; khi lỗi thì giữ log và chạy lại vào thư mục mới;
- lệnh chấm lại đối chiếu cho UA-DETRAC;
- dự toán;
- tiêu chí hậu kiểm.

Dự toán là **kịch bản theo giả định**:
- căn cứ là ba lượt R47 và ba lượt R43 cùng kiến trúc, cùng máy;
- kết quả ≈ 54 phút cho UA-DETRAC và ≈ 59 phút cho BDD, tổng ≈ 1,9 giờ; kịch bản chậm ≈ 2,8 giờ; NPZ ≈ 1,1 GB.

Các lệnh này chỉ là bàn giao, **chưa thực thi**. Lượt dài cần R53B được approve và người dùng cho phép.

## 6. Giới hạn và điểm reviewer nên đối chiếu

1. Trường `lenh` trong `log.txt`, `exit.json` và `ket_qua.json` của smoke/tiền kiểm ghi `sys.executable` cùng các tham
   số, nhưng **thiếu đường dẫn script** (`sys.argv[1:]` bị ghép thẳng sau trình thông dịch). Lệnh thật là các dòng ở
   tài liệu 02 mục 6. Chỉ là thiếu sót ghi chép; không ảnh hưởng kiểm hay kết quả. Mã đã khóa nên không sửa trong lượt
   này.
2. BDD adapter không có `--cham-lai`. Phần chấm lại đối chiếu cho BDD trong lượt dài cần công cụ hậu kiểm của phiên
   sau (tài liệu 02 mục 8); công cụ này chưa được viết.
3. Dự toán giả định model mixed5 chạy nhanh như model BDD-only cùng kiến trúc. Số detection khác thì thời gian chấm và
   NPZ có thể khác.
4. Mẫu UA-DETRAC chỉ có 1 GT bus, vì bus đầu tiên xuất hiện ở ảnh 2085. Đủ cho kiểm kỹ thuật D3; không có ý nghĩa
   về chất lượng.
5. Các file khóa (tài liệu 02, `khoa_danh_gia.json`, `SHA256SUMS`, `thoi_diem_khoa.json`) chưa commit. Dấu thời gian
   chỉ là nội dung file, không có xác nhận bên ngoài như email ở R47.
6. Thư mục `tests_eval/lan_20261020_0407*` là lần test đầu bị lỗi fixture. Thư mục này được giữ làm bằng chứng.
7. `PLAN_DA_CHOT.md` mục 0 đã được cập nhật trước báo cáo này. Mục 0 ghi R53A approve theo context của plan
   (`reports/review_002.json`); tôi chưa tự mở file review đó.
