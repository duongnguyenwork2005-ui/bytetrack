# Vòng 46 — xử lý review `d4715aa` (R45-F1…F4); chưa triển khai đánh giá UA-DETRAC (chi tiết)

**Phạm vi** (người dùng duyệt 2026-10-07):
- Chỉ xử lý bốn finding và ba điểm cần làm rõ của review `d4715aa`.
- Không train, không GPU, không SSH Nano, không khóa giao thức, không viết adapter UA-DETRAC.
- Không đụng dataset, checkpoint, kết quả cũ, `.gitignore`, `src/demo/**`.
- **Base review giữ `3e94b21`.** `d4715aa` chưa PASS. **D1–D5 vẫn chưa được người dùng duyệt.** Vòng 46 **chưa** được
  nghiệm thu.

> **Cập nhật 2026-10-09** (kế hoạch R46, mốc A): Codex review `e4a0311` — **PASS WITH LIMITATIONS**, đóng R45-F1…F4
> trong phạm vi công cụ phân tích và hồ sơ dự thảo; hai ghi chú không chặn R46-N1 / R46-N2. Ghi nhận ở **mục 9**. Câu
> "chưa được nghiệm thu" ở trên là trạng thái lúc bàn giao, giữ nguyên. **D1–D5 vẫn chưa được người dùng duyệt.**
>
> **Cập nhật mốc B** (2026-10-09): đã đính chính câu chữ R46-N1 ở ba vị trí review nêu và R46-N2 ở câu duy nhất, nằm ở
> **mục 2** của `REVIEW45_CHI_TIET.md` (review ghi mục 3). Xem **mục 10**. D1–D5 vẫn chưa duyệt.

**Phân loại kết quả**
- **Sửa bằng mã** (công cụ audit, không phải mã lõi): R45-F3, R45-F4.
- **Sửa diễn giải**: R45-F1, R45-F2, và làm rõ 1–3.
- **Còn phụ thuộc D1–D5**:
  - chọn D3 / D4 / D5;
  - nếu chọn D3 như đề xuất, bước chuẩn hoá báo cáo phải triển khai sau đó;
  - adapter, danh sách ảnh có thứ tự, gate w/h mọi ảnh.

## 0. Trạng thái đầu vòng

Log `logs/review46/00_trang_thai_dau.log`:
- HEAD `d4715aa` = target được review; **0 commit** sau đó.
- Working tree chỉ có file riêng của người dùng.
- Đang cắm sạc, gói High performance.

## 1. Ghi nhận review `d4715aa` (commit `e3c1cd1`)

- **Báo cáo** `handoff/CODEX_REVIEW_d4715aa.md`: trùng byte bản trong `D:\LABOTARY\review_codex` (15.138 B, sha256
  `9c80ea0a…`).
- **Kết luận**: **REQUEST CHANGES**. Kiểm toàn vẹn gói đạt: 74 payload + 1 alias, 75/75 đường dẫn áp ngược / xuôi.
- **Bốn finding**: R45-F1 (P2), R45-F2 (P2), R45-F3 (P3), R45-F4 (P3). Ba điểm cần làm rõ: trình tự khóa, tên ảnh trùng
  trong NPZ, kiểm w/h mọi ảnh.
- **Bằng chứng**: `CODEX_REVIEW_d4715aa_EVIDENCE.zip` 501.772 B, sha256 `b984b3a4…`.
  - Giải nén vào thư mục mới, ZIP gốc giữ nguyên. `SHA256SUMS` khớp **9/9** trên đĩa.
  - Lưu ở `handoff/review_evidence/CODEX_d4715aa/`: blob đã stage khớp 8/8.
  - `input/review_delta_latest.zip` **không** commit, vì `handoff/.gitignore` có `*.zip`. File này trùng byte
    `handoff/archive/review_delta_d4715aa.zip`, tức gói đã review (`b1c207df…`).
- Script reviewer `audit_review45.py` **có** trong bằng chứng và được chạy nguyên byte (mục 2).
- `review_base.json` **không đổi**.

## 2. Tái hiện trước khi sửa (`logs/review46/01_tai_hien_truoc_sua.log`)

Chạy nguyên script reviewer (sha256 `7f427b92…`) trên gói `d4715aa` trong bản sao tạm, chỉ CPU và fixture:
- **Lần 1 FAIL do môi trường**: script gọi `read_text()` không chỉ định encoding, nên Windows dùng cp1252 (Linux của
  reviewer mặc định UTF-8). Lỗi `UnicodeDecodeError`, exit 1. Log được giữ.
- **Lần 2** dùng `python -X utf8`, **không sửa script**: exit 0. `findings_reproduced` gồm đủ 4 mục: D3 vẫn đếm FP, D4
  thiếu vùng ignore tĩnh, D5 AP không chắc tăng, XML mất metadata.
- `probes` / `tables` / `summary` / `integrity` **trùng** kết quả của reviewer (bỏ trường `environment`). Đầu ra lưu ở
  `01_recheck_truoc_sua/`.

## 3. Sửa

### R45-F4 — `doc_xml()` mất `sequence_attribute` (mã công cụ; commit `85b8eda`)

- **Nguyên nhân**: biểu thức `(root.find("sequence_attribute") or ET.Element("x"))`. Element không có phần tử con có
  truthiness False dù **có** thuộc tính, nên phần tử tự đóng bị thay bằng Element rỗng.
- **Sửa**: bản `logs/review46/phan_tich_detrac_test.py` kiểm `node is not None`, thêm `co_sequence_attribute`.
  - Cột `camera_state` / `weather` ghi đúng giá trị XML.
  - Nếu thiếu thật thì ghi rõ "XML không có phần tử sequence_attribute" hoặc "phần tử không có thuộc tính này".
  - Không để trống, không điền giá trị suy đoán.
  - Bản vòng 45 giữ nguyên ở `logs/review45/` làm bằng chứng lịch sử và làm "mã cũ" cho regression.
- **Trên XML thật** (`03_chay_lai_cong_cu_tren_xml_that.log`):
  - cả **40/40** video đều có đủ hai thuộc tính: camera 31 `stable` / 9 `unstable`; thời tiết 8 `sunny` / 11 `cloudy` /
    12 `night` / 9 `rainy`;
  - 40 ô trống ở bảng vòng 45 là do parser, không phải XML thiếu;
  - mọi đầu ra khác **trùng từng byte** bản vòng 45: `anh_khong_frame.csv`, `mau_box_xem_tay.csv`, `tong_hop.json`.

### R45-F3 — bảng phương án giữ 173 ảnh thiếu 265 lượt vùng ignore (mã công cụ; commit `85b8eda`)

- **Sửa**: bản `logs/review46/dem_theo_phuong_an.py` cộng `ignored_region` cấp sequence cho mọi ảnh thiếu `<frame>` được
  giữ.
  - `--out-csv` ghi bằng chứng theo video / frame.
  - Đơn vị ghi rõ: **số lượt vùng áp dụng trên ảnh** (đếm lặp qua ảnh), **không** phải số vùng hình học duy nhất. Có 163
    vùng hình học.
- **Bằng chứng** (`phan_tich/vung_tren_anh_khong_frame.csv`), khớp bảng của reviewer:

| Video | Ảnh thiếu `<frame>` | Vùng tĩnh mỗi ảnh | Lượt vùng |
|:--|--:|--:|--:|
| MVI_39051 | 69 (992–1060) | 2 | 138 |
| MVI_39211 | 94 (477–570) | 1 | 94 |
| MVI_39271 | 4 (77–80) | 4 | 16 |
| MVI_40743 | 1 (1630) | 6 | 6 |
| MVI_40763 | 4 (1742–1745) | 2 | 8 |
| MVI_40793 | 1 (1960) | 3 | 3 |
| **Tổng** | **173** | — | **265** |

- **Tính lại `vung_bo_qua_cong_moi_anh`** khi giữ 173 ảnh. Bốn phương án loại 173 ảnh, tổng GT và số ảnh **không đổi**:

| D1 | D2 | Vòng 45 | Vòng 46 |
|:--|:--|--:|--:|
| van bỏ qua | others bỏ qua | 275.724 | **275.989** |
| van bỏ qua | others → truck | 258.809 | **259.074** |
| van → car | others bỏ qua | 237.205 | **237.470** |
| van → car | others → truck | 220.290 | **220.555** |

- D4 **vẫn chưa chọn**: giữ hay loại 173 ảnh là quyết định của người dùng.

### R45-F1 — D3: lớp N/A vẫn bị scorer đếm FP (diễn giải)

**Đối chiếu đường chạy thật**:
- `eval_detector_5lop.danh_gia` duyệt cả 5 lớp.
- Với lớp không có GT: AP / AP75 / AP50–95, recall, F1 = `null` và lớp bị loại khỏi mAP / macro. Nhưng TP / FP **thô** tại
  điểm vận hành **vẫn được đếm**, precision = TP / (TP + FP) vẫn tính, và `n_det` vẫn đếm.
  Với lớp không GT, P=0 khi còn FP>0 sau xử lý ignore; P=null khi TP+FP=0.
  *(Đính chính R46-N1, kế hoạch R46 mốc B — mục 10: bản trước ghi "(= 0 nếu có detection lớp đó với conf ≥ 0,25)",
  rộng hơn hành vi thật — detection không ghép GT mà có IoA ≥ 0,5 với một vùng bỏ qua thì không bị đếm FP.)*
- Entry point kiểu vòng 43 ghi các số này vào `ket_qua.json` và `theo_lop.csv` cho cả 5 lớp.
- Fixture E1 (`test_cong_cu_detrac.py`, trùng fixture reviewer) cho truck: FP = 1, precision = 0, AP = null; macro 2 lớp,
  mAP50 = 1.
- BDD vòng 43 không bị ảnh hưởng: cả 5 lớp đều có GT.

**Sửa tài liệu**:
- `REVIEW45_CHI_TIET.md` (hàng D3 + ghi chú đính chính);
- dự thảo `logs/review45/06_*` (mục 4, 5, 6, 8, 9);
- `REVIEW43_CHI_TIET.md` (ghi chú + mục 8 D3);
- kế hoạch khóa vòng 43: thêm vào **phụ lục mục 10**, không sửa mục 8, đúng quy tắc của chính kế hoạch.

Nội dung đính chính:
- Bộ đếm thô của scorer **không** phải metric và không được trình bày như chất lượng detector.
- Để báo N/A trong JSON / CSV cần **bước chuẩn hoá báo cáo theo D3**. Đây là **thay đổi phải triển khai sau khi D3 được
  duyệt**, kèm fixture đi qua entry point thật kiểm cả JSON lẫn CSV; không phải chức năng đã có.
- **Không** đổi evaluator chung, không sửa kết quả BDD. D3 **chưa chọn**.

### R45-F2 — D5: kết luận một chiều và phát biểu toolkit vượt bằng chứng (diễn giải)

- **Phản ví dụ** (fixture E2, trùng fixture reviewer, scorer thật): hai GT car, một GT trong vùng ignore được detect đúng,
  một GT ngoài vùng bị sót.
  - Giữ GT trong vùng: AP50 0,5050, recall 0,5.
  - Chuyển GT đó thành vùng bỏ qua: AP50 0, recall 0.

  Vậy "bỏ GT trong vùng ignore làm recall / AP tăng" **không** đúng tổng quát.
- **Đã sửa**:
  - (b)/(c) **thay đổi tập GT và cách tính TP / FP / FN**; recall / AP **có thể tăng hoặc giảm**; chiều và mức trên dữ
    liệu thật **chưa biết** khi chưa có prediction;
  - gỡ luôn câu "D5(a) cho AP thấp hơn cách chấm kiểu toolkit";
  - không chọn D5 theo metric test.
- **Toolkit chỉ theo phần mã đã đọc**. Bản trích nguyên byte kèm sha256 ở `logs/review46/toolkit_detrac_trich/`:
  - `preProcessDetectionResults.m` → `dropDetections.m`: loại detection có tỉ lệ pixel trong **hợp** vùng ≥ 0,5,
    **trước** khi gọi `AP_DET_EVAL.exe`;
  - `parseGT.m`: **chỉ đường MOT**, cắt GT ở hai đầu quỹ đạo;
  - `checkResultFormat.m`: detection 7 cột, không có trường lớp;
  - `README_curve.txt`: mô tả PR theo ngưỡng score;
  - **Chưa xác minh**: cách `AP_DET_EVAL.exe` (nhị phân, sha256 `40d310fb…`) xử lý GT trong vùng bỏ qua và dùng
    `vehicle_type`. Câu cũ "toolkit chấm một lớp vehicle" như đã biết đã được rút.
  - Phương án (c) được ghi rõ là **tự định nghĩa**.
- **40.835** (= 35.248 car + 5.587 bus) chỉ là **GT XML gốc đạt ngưỡng trong thống kê này** (max từng vùng, box chưa cắt
  khung, chưa loại box nhỏ), **không** phải số GT toolkit chắc chắn loại.

### Làm rõ 1–3 (dự thảo)

1. **Trình tự**: sau khi D1–D5 được duyệt → fingerprint dữ liệu (hash và w/h **mọi** ảnh, danh sách ảnh có thứ tự) →
   adapter + fixture → **khóa** kế hoạch mới tham chiếu đúng các hash, commit trước suy luận. Không sửa đè kế hoạch đã
   khóa sau khi xem kết quả.
2. **NPZ**: `luu_du_doan` lưu `Path.name`. Ảnh UA-DETRAC của các video khác nhau trùng tên (`img00001.jpg`); `idx` giữ
   đúng vị trí nên chưa phải lỗi AP. Adapter phải lưu **danh sách `video/frame` có thứ tự** và đối chiếu khi tái tính.
   Đây là lưu ý thiết kế, không phải regression BDD (tên ảnh BDD gồm cả tên video và đã được kiểm không trùng ở vòng 43).
3. **w/h**: phân tích vòng 45 mới đọc header ảnh đầu / cuối mỗi video. Gate kiểm w/h **mọi** ảnh trước prediction được
   giữ trong dự thảo.

## 4. Kiểm sau khi sửa

| Phép kiểm | Loại | Kết quả | Log |
|:--|:--|:--|:--|
| Script reviewer trên gói `d4715aa` (trước sửa) | script reviewer nguyên byte | lần 1 FAIL (môi trường cp1252); lần 2 (`-X utf8`) exit 0, tái hiện 4/4, trùng kết quả reviewer | `01` |
| `test_cong_cu_detrac.py` trên công cụ **cũ** (`logs/review45`) | tích hợp nhỏ (X, D) + đối chiếu hành vi (E) | **2/8**, FAIL đúng X1–X4, D1, D2 | `02` |
| `test_cong_cu_detrac.py` trên công cụ **sửa** (`logs/review46`) | như trên | **8/8 PASS** | `02` |
| Công cụ sửa trên XML / ảnh thật (chỉ đọc) | chạy công cụ trên dữ liệu thật | exit 0; chỉ đổi đúng các cột / trường dự kiến | `03` |
| Đối chiếu tài liệu | grep + đọc lại | các phát biểu sai đã rút; chỗ còn khớp từ khoá đều đúng ngữ cảnh | `05` |
| Script reviewer trên **gói mới** | script reviewer nguyên byte | script gắn cứng đặc điểm gói `d4715aa` (`len(inventory) == 80`, tên patch, 75 đường dẫn); kết quả nằm trong tin nhắn bàn giao, vì log này không thể nằm trong chính gói | — |

- Không chạy full suite: thay đổi chỉ ở công cụ audit trong `handoff/logs/` và tài liệu; `src/` không đổi.
- Fixture E dùng **evaluator thật** (`src/eval_detector_5lop.py`, không đổi); X / D dùng **hàm và `main()` thật** của
  công cụ trên XML / ảnh tổng hợp.

## 5. Bằng chứng và support trong gói

- `logs/review46/`: `00`–`03` (log), `05` (đối chiếu tài liệu), `01_recheck_truoc_sua/`, hai công cụ sửa,
  `test_cong_cu_detrac.py`, `phan_tich/` (đầu ra mới), `toolkit_detrac_trich/` (8 file text + `NGUON.txt`).
- **Support** (không đổi so với base `3e94b21`):
  - `src/eval_detector_5lop.py`, `src/detector_classes.py`: để chạy nhóm E;
  - `results/phase2/manifests/detrac_test.json`: manifest video, theo đề nghị của review.
- **Không gửi**: XML, ảnh, crop, checkpoint, NPZ, file `.exe` của toolkit (chỉ sha256).

## 6. Giới hạn

- Tác giả chạy trên Windows / Python 3.11.9 (script reviewer cần `-X utf8`); reviewer dùng Linux / 3.12.
- Nhóm E chỉ **đối chiếu hành vi** để tài liệu mô tả đúng. Nó không đổi hay chứng minh lại evaluator.
- Hành vi bên trong `AP_DET_EVAL.exe` chưa xác minh.
- Phân loại van / others bằng mắt vẫn là quan sát chủ quan của vòng 45. Ảnh minh hoạ không được gửi.
- UA-DETRAC chưa chạy; mọi ảnh hưởng tới metric vẫn chưa biết.

## 7. Lệnh (từ gốc repo) và mã thoát

```text
python -B audit_review45.py --package package --out recheck            # (ban sao tam cua bang chung) exit 1: UnicodeDecodeError cp1252
python -X utf8 -B audit_review45.py --package package --out recheck2   # exit 0, tai hien 4/4
python handoff/logs/review46/test_cong_cu_detrac.py --cong-cu-dir handoff/logs/review45   # exit 1, 2/8 (du kien)
python handoff/logs/review46/test_cong_cu_detrac.py                                       # exit 0, 8/8
python handoff/logs/review46/phan_tich_detrac_test.py --out handoff/logs/review46/phan_tich                          # exit 0
python handoff/logs/review46/dem_theo_phuong_an.py --out handoff/logs/review46/phan_tich/dem_theo_phuong_an.json \
    --out-csv handoff/logs/review46/phan_tich/vung_tren_anh_khong_frame.csv                                          # exit 0
python src/make_review_delta.py --support src/eval_detector_5lop.py src/detector_classes.py results/phase2/manifests/detrac_test.json
```

## 8. Commit vòng 46

| Commit | Nội dung |
|:--|:--|
| `e3c1cd1` | Ghi nhận review `d4715aa` + bằng chứng; tái hiện bằng script reviewer; base giữ `3e94b21` |
| `85b8eda` | Sửa R45-F4 + R45-F3 (bản công cụ vòng 46) + test |
| `c5b6074` | Bằng chứng: test cũ 2/8 / sửa 8/8, chạy lại trên XML thật, trích toolkit |
| commit chứa phụ lục này (target delta) | Đính chính tài liệu (REVIEW45, dự thảo, REVIEW43, phụ lục kế hoạch vòng 43), brief, phụ lục, handoff, log `05` |

## 9. Ghi nhận nghiệm thu `e4a0311` (kế hoạch R46, mốc A — 2026-10-09)

Người dùng duyệt kế hoạch R46 gồm ba mốc A → B → C. Mốc A chỉ đối chiếu và ghi nhận: **chưa** đính chính R46-N1 /
R46-N2 (mốc B), **chưa** hoàn thiện bảng D1–D5 và brief (mốc C). Không chạy script Python, không GPU, không train,
không SSH; chỉ đọc file và lệnh Git chỉ đọc.

**Nguồn đã đọc** (ngoài repo, chỉ đọc):
- Báo cáo `D:\LABOTARY\review_codex\CODEX_REVIEW_e4a0311.md` (review ngày 2026-10-07). Không sửa.
- Bằng chứng `CODEX_REVIEW_e4a0311_EVIDENCE.zip` (1.182.614 B, sha256 `ca6e0e7a…`) giữ nguyên. Bản giải nén
  `D:\LABOTARY\review_codex\e4a0311_evidence_extracted_20261009\`: theo báo cáo giải nén, `SHA256SUMS` khớp **29/29**,
  không thiếu / thừa.
  - Bản review trong gói bằng chứng (9.255 B, `83188a2c…` theo báo cáo giải nén) có nội dung đọc được trùng bản ở trên
    (77 dòng). Chưa băm bản ngoài để so từng byte.
  - Đã đọc: `evidence/README.md`, `package_check.json`, `test_summary.json`, `author_test_current.log`,
    `author_test_old.log`, `original_script_new_package.log`, `audit_review46.json`, `independent_tables.json`,
    `reproduction_check/reproduction_summary.json`; `check_package.py` chỉ đọc, không chạy. Không giải nén ZIP lồng
    `input/*.zip`.

**Đối chiếu với repo**

| Mục | Review / bằng chứng | Repo | Khớp |
|:--|:--|:--|:--|
| Target | `e4a0311954278c85c6732f0a53fd2d3cee42ed1b` | `git rev-parse HEAD` cùng hash, 0 commit sau target; chính là "commit chứa phụ lục này" ở mục 8 | ✅ |
| Base | `3e94b21683b7e63c5852a7ddc04baf513952ddc4` | `review_base.json` `base_commit` cùng hash | ✅ |
| Base / target trong manifest gói | `check_package.py` chốt `3e94b21` / `e4a0311`; `package_check.json` pass, 118 đường dẫn | — | ✅ theo bằng chứng |
| Gói đã review | 678.886 B, `128ed119…` = `input/review_delta_e4a0311.zip` trong `SHA256SUMS` | `handoff/review_delta_latest.zip` trên đĩa **chưa băm lại** | chưa kiểm |
| Support | 3 file không đổi | mục 5 | ✅ |
| Regression | test mới 8/8 exit 0; công cụ vòng 45 2/8 exit 1 (trượt X1–X4, D1, D2) | `logs/review46/02`: cùng tên phép, cùng kết quả | ✅ |
| R45-F3 | 173 ảnh, 265 lượt vùng; 275.989 / 259.074 / 237.470 / 220.555 | mục 3 | ✅ |
| R45-F4 | camera 31 stable / 9 unstable; thời tiết 8 / 11 / 12 / 9 | mục 3 | ✅ |
| Bằng chứng cũ | 43/43 (`CODEX_3e94b21`), 9/9 (`CODEX_d4715aa`); 13 khai báo NPZ | `REVIEW45_CHI_TIET.md` mục 1, mục 1 ở trên; `review_base.json` 13 khai báo | ✅ |
| Script reviewer cũ trên gói mới | dừng ở `len(inventory) == 80` (gói có 121) | dòng cuối bảng mục 4 | ✅ |
| Vị trí N1 | dự thảo mục 6; phần F1 của file này; bảng D3 của `REVIEW45_CHI_TIET.md` | câu "= 0 nếu có detection…" có đúng ba chỗ này (dự thảo mục 6, mục 3 phần R45-F1 ở trên, hàng D3 trong mục 3 của REVIEW45) | ✅ |
| Vị trí N2 | `REVIEW45_CHI_TIET.md` mục 3 | câu "Mọi ảnh 960×540" ở **mục 2** của REVIEW45, một lần duy nhất | lệch số mục |

**Kết luận ghi nhận** (nguyên ý review): **PASS WITH LIMITATIONS**.
- **R45-F1**: đóng **ở mức đặc tả / hồ sơ**. Bước chuẩn hoá báo cáo theo D3 chưa viết nên **chưa** được nghiệm thu.
- **R45-F2**: đóng. Không suy ra AP DET chính thức từ `parseGT.m`; không kết luận chiều chênh AP trên dữ liệu thật.
- **R45-F3**: đóng. Bốn phương án loại 173 ảnh không đổi; bốn phương án giữ tăng đúng 265.
- **R45-F4**: đóng. Giá trị trên dữ liệu thật chỉ được đối chiếu qua bảng / log bàn giao, không đọc lại XML gốc.

**Hai ghi chú không chặn** (chưa sửa ở mốc A; đính chính ở mốc B):
- **R46-N1**: câu "precision = 0 nếu có detection lớp đó với conf ≥ 0,25" rộng hơn hành vi thật. Probe của reviewer:
  detection truck conf 0,9 nằm hoàn toàn trong vùng ignore → FP 0, precision null; bỏ vùng → FP 1, precision 0.
  Vị trí: dự thảo `logs/review45/06_*` mục 6, phần R45-F1 ở mục 3 file này, bảng D3 của `REVIEW45_CHI_TIET.md`.
- **R46-N2**: `REVIEW45_CHI_TIET.md` ghi "Mọi ảnh 960×540", trong khi công cụ chỉ đọc header ảnh đầu / cuối mỗi video.
  Review ghi vị trí là mục 3; trong file, câu này nằm ở **mục 2** (Dữ kiện UA-DETRAC test) và chỉ xuất hiện một lần.

**Lưu ý provenance** (không đổi danh sách video): manifest `detrac_test.json` gửi trong gói là blob LF 882 B,
`021b56e1…`; đổi xuống dòng sang CRLF được 931 B, `ff2b4b9c…`, là hash đang ghi trong tài liệu. JSON như nhau, 40 video
khớp toolkit. Kế hoạch khóa sau này phải ghi rõ quy ước hash hoặc ghi cả hai, không gọi là trùng byte.

**Giới hạn của nghiệm thu** (theo review):
- reviewer không có ảnh, XML gốc, crop, checkpoint; chỉ đọc tám trích đoạn toolkit, không chạy toolkit, không có
  executable DET;
- chỉ kiểm đường dẫn đổi và 3 file support, không lặp lại phép kiểm toàn cây của tác giả; không chạy full suite;
- dự thảo giao thức **chưa khóa**; adapter UA-DETRAC **chưa** viết; chưa có kết quả detector UA-DETRAC.

**Chưa làm ở mốc A** (đường dẫn bảo vệ hoặc ngoài phạm vi ghi của kế hoạch R46; chờ người dùng quyết định):
- chưa sao báo cáo vào `handoff/CODEX_REVIEW_e4a0311.md`, chưa lưu bằng chứng vào `handoff/review_evidence/`;
- chưa lưu gói đã review vào `handoff/archive/`;
- `review_base.json` **không đổi**: base vẫn `3e94b21`; chưa chạy `make_review_delta.py --set-base`.

**Trạng thái sau mốc A**: vòng sửa `e4a0311` đã được nghiệm thu **với giới hạn**. **D1–D5 vẫn CHƯA được người dùng
duyệt.** Theo review, bước tiếp theo là người dùng chốt D1–D5; kế hoạch R46 chỉ còn mốc B và C, đều là tài liệu.

## 10. Đính chính R46-N1 / R46-N2 (kế hoạch R46, mốc B — 2026-10-09)

Chỉ sửa câu chữ theo hai ghi chú không chặn của review `e4a0311`. Không sửa scorer, công cụ, log / CSV / JSON, dataset,
checkpoint; không chạy lại; không chạy Python, không GPU, không SSH. Câu cũ được trích lại trong ghi chú đính chính để
truy vết. Báo cáo reviewer gốc không sửa.

**Đối chiếu với mã trước khi sửa** (chỉ đọc, không chạy):
- **N1**: `src/eval_detector_5lop.py` dòng 242–245: `vh = (conf >= conf_van_hanh) & ~bq[ti_vh]`, `FP = vh.sum() - TP`,
  `P = TP / (TP + FP) if TP + FP else nan`. `bq` đánh dấu detection không ghép được GT mà IoA max theo từng vùng ≥ ngưỡng
  (`ghep_anh_lop`, dòng 192–193); `_json_safe` (dòng 305–307) ghi NaN thành `null`. Lớp không GT có TP = 0, nên P = 0
  khi còn FP > 0 sau xử lý ignore và P = `null` khi TP + FP = 0. Logic này khớp kết quả probe của reviewer (truck conf
  0,9 nằm trọn trong vùng ignore → FP 0, precision null); probe **không** được chạy lại ở mốc B.
- **N2**: `logs/review45/phan_tich_detrac_test.py` dòng 123–124 (bản vòng 46 dòng 140–141) chỉ đọc kích thước `anh[0]` và
  `anh[-1]` mỗi video. `theo_video.csv` của vòng 45 và vòng 46 đều có 40/40 dòng với `kich_thuoc` =
  `kich_thuoc_anh_cuoi` = 960x540.

**Vị trí trước / sau** (trước = blob trong commit `e4a0311`; sau = bản làm việc, chưa commit)

| Ghi chú | File, vị trí | Trước (`e4a0311`) | Sau |
|:--|:--|:--|:--|
| N1 | `logs/review45/06_du_thao_giao_thuc_detrac.md` mục 6 | dòng 103–104: "precision = TP / (TP + FP) vẫn được tính (= 0 nếu có detection lớp đó với conf ≥ 0,25)" | dòng 107–110: bỏ phần ngoặc; câu đính chính ở dòng 108; ghi chú trích câu cũ; khối "Đính chính R46-N1" đầu file dòng 11–13 |
| N1 | file này, mục 3, phần R45-F1 | dòng 102–103: "precision = TP / (TP + FP) vẫn tính (= 0 nếu có detection lớp đó với conf ≥ 0,25)" | dòng 109–112: câu đính chính ở dòng 110; ghi chú trích câu cũ |
| N1 | `REVIEW45_CHI_TIET.md` mục 3, hàng D3 | dòng 102: "precision (= 0 nếu có detection lớp đó với conf ≥ 0,25)" | dòng 118: câu đính chính + ghi chú trong cùng ô |
| N2 | `REVIEW45_CHI_TIET.md` mục 2 (review ghi mục 3) | dòng 60: "… 0 `<frame>` không có ảnh. Mọi ảnh 960×540." | dòng 72–76: câu đính chính ở dòng 73; ghi chú nguồn |

Câu đính chính (nguyên văn theo review):
- N1: "Với lớp không GT, P=0 khi còn FP>0 sau xử lý ignore; P=null khi TP+FP=0."
- N2: "Ảnh đầu và cuối của 40 video đều 960×540; chưa kiểm từng ảnh."

**Lệch số mục N2**: review ghi `REVIEW45_CHI_TIET.md` **mục 3**; câu "Mọi ảnh 960×540" chỉ có một lần, ở **mục 2** (Dữ kiện
UA-DETRAC test). Mục 3 (bảng D1–D5) không có câu nào về kích thước ảnh. Nhiệm vụ mốc B trích đúng câu này, nên đính chính
áp vào vị trí thật của nó ở mục 2.

**Đính chính bản trước của mục này** (sửa theo review mốc B, kết luận revise): bản trước ghi thêm hai việc "theo quyết
định người dùng". Cả hai thuộc quyết định đã duyệt cho **mốc sau** (chưa tới lượt ở mốc B), không thuộc nhiệm vụ mốc B:
- ghi chú N2 ở hàng 41 `CODEX_HANDOFF.md`: **đã gỡ**. Hàng 41 trở lại đúng nội dung `e4a0311` ("… 56.340 anh 960x540 /
  173 anh khong <frame> …"); `git diff` không còn thay đổi ở dòng này.
- "trạng thái vòng 44–46 nối vào cột trạng thái, mốc 3 của `handoff/PLAN_DA_CHOT.md`": **không còn đúng**. Theo bản ghi khôi
  phục của orchestrator, thay đổi đó đã được hoàn tác theo lựa chọn của người dùng (2026-10-09, khoảng 11:16, trước lượt
  sửa này). Hiện `git status` / `git diff` không thấy `PLAN_DA_CHOT.md` khác `e4a0311`; mốc B không sửa file này.

**Ghi chú trạng thái kèm theo**: khối đầu `REVIEW45_CHI_TIET.md` (dòng 26–31), khối đầu dự thảo (dòng 11–13), khối đầu
file này (dòng 13–15), brief, hàng 42 `CODEX_HANDOFF.md`.

**Không đổi**:
- câu đã đúng: dự thảo mục 3 ("Cả 40 video đều 960×540 (đọc header ảnh đầu và cuối)") và mục 7 (w/h **mọi** ảnh phải
  kiểm trước dự đoán); fixture E1 (truck FP = 1, precision = 0: detection không bị bỏ qua theo vùng ignore);
- ghi nhận mốc A (mục 9, khối đầu REVIEW45, brief) giữ nguyên làm lịch sử;
- `review_base.json`, log, CSV / JSON, kết quả, dataset, checkpoint, mã.

**Kiểm sau khi sửa** (Grep và `git diff`, chỉ đọc):
- câu N1 nguyên văn có ở đúng ba vị trí review nêu (dự thảo dòng 108, file này dòng 110, REVIEW45 dòng 118); ngoài ra
  chỉ được trích ở mục 10 này;
- câu N2 nguyên văn: REVIEW45 dòng 73; ngoài ra chỉ được trích ở mục 10 này;
- cụm cũ "0 nếu có detection lớp đó" chỉ còn trong ba ghi chú đính chính (trích câu cũ), mục 9 và mục 10 này;
- `CODEX_HANDOFF.md` so với `e4a0311`: chỉ hàng 42 đổi, hàng 41 như cũ; `PLAN_DA_CHOT.md` không đổi; diff staged rỗng.

**Chưa làm**:
- các quyết định người dùng đã duyệt cho mốc sau, chưa tới lượt ở mốc B; không thực hiện và không dùng làm căn cứ:
  - lưu review `e4a0311` + bằng chứng, archive gói, chuyển base → `e4a0311` (đích ghi thuộc đường dẫn bảo vệ của kế hoạch
    R46). `review_base.json` vẫn base `3e94b21`;
  - ghi chú N2 ở hàng 41 `CODEX_HANDOFF.md`;
  - cập nhật trạng thái vòng 44–46 trong `handoff/PLAN_DA_CHOT.md`;
- bảng D1–D5 và brief vòng kế tiếp: mốc C.

**Trạng thái sau mốc B**: R46-N1 / R46-N2 đã đính chính câu chữ. **D1–D5 vẫn CHƯA được người dùng duyệt.** Còn mốc C
(tài liệu).

## 11. Bảng quyết định D1–D5 cho người dùng (kế hoạch R46, mốc C — 2026-10-09) — **CHƯA DUYỆT**

Mốc C chỉ tổng hợp hồ sơ để người dùng chọn. **Không phương án nào được chọn**; cột "Lựa chọn của người dùng" ở 11.4 để
trống. Không có số mới: mọi số lấy từ hồ sơ vòng 45–46 đã được nghiệm thu ở `e4a0311` (`REVIEW45_CHI_TIET.md` mục 2–4
kèm đính chính; mục 3 và 10 file này; `logs/review46/phan_tich/dem_theo_phuong_an.json`). Không chạy Python, không đọc lại
XML / ảnh, không GPU, không SSH. Chưa có dự đoán nào trên UA-DETRAC: mọi hệ quả dưới đây là phía GT / quy tắc chấm, không
phải metric.

### 11.1 Dữ kiện chung

- 40 video test; 56.340 ảnh trên đĩa, 56.167 ảnh có `<frame>`, 173 ảnh không có `<frame>`. Kích thước: ảnh đầu / cuối mỗi
  video là 960×540, chưa kiểm từng ảnh; dự thảo (mục 7) yêu cầu kiểm w/h mọi ảnh trước dự đoán.
- 675.774 box XML: car 548.555, bus 71.785, van 38.519, others 16.915. **Không có** nhãn truck, xe máy, xe đạp. 163
  `ignored_region`, 4 cặp chồng nhau.
- Lớp đã khóa của model: car, truck, bus, motorcycle, bicycle. Ba checkpoint vòng 43 (baseline COCO, hai bản fine-tune chỉ
  học BDD train); không checkpoint nào học UA-DETRAC.
- Phân loại `van` / `others` bằng mắt là **chủ quan** của tác giả, trên mẫu, mỗi track một crop, chưa kiểm chéo. Reviewer
  `e4a0311` không có ảnh nên chưa xác minh.
- Số GT là **box XML gốc**: chưa cắt khung, chưa loại box < 1 px. Số GT cuối cùng do bộ đọc GT ghi lúc chạy.
- Áp dụng cho mọi phương án:
  - không tô xám vùng bỏ qua trên ảnh (khác pipeline tracking; dự thảo mục 5);
  - mAP trên UA-DETRAC không so với mAP 5 lớp của BDD val, cũng không so với kết quả DETRAC chính thức;
  - tập test không tham gia lựa chọn (`PLAN_DA_CHOT.md`); hiện cũng chưa có metric test nào.

### 11.2 Phương án và hệ quả

| Quyết định | Phương án | Hệ quả (box XML gốc) | Ràng buộc / việc kéo theo |
|:--|:--|:--|:--|
| **D1** `van`: 38.519 box, 126 track, 35 video. Mẫu 48/126 track: 45 van / MPV kiểu xe con (91,7 % box mẫu), 2 minibus kiểu xe buýt (7,4 %), 1 van chở hàng | (a) vùng bỏ qua cho mọi lớp | car GT 548.555 (55.764 ảnh). Detection không ghép GT mà IoA với box van ≥ ngưỡng → không FP; bỏ sót van → không FN. 72 ảnh có `<frame>` không còn GT car / bus | box van vào mảng vùng bỏ qua dùng chung mọi lớp (`REVIEW45_CHI_TIET.md` mục 4) |
| | (b) GT car | car GT 587.074 (+38.519, +7,0 %), 55.994 ảnh. Detect van là car → TP, bỏ sót → FN car; detection bus trên minibus → FP bus + FN car | giả định taxonomy chưa kiểm: repo không có bằng chứng BDD gán van vào lớp nào |
| **D2** `others`: 16.915 box, 31 track / 21 video. Đủ 31 track: 21 xe tải chở hàng (34,9 % box), 7 xe cứu hộ / kéo xe (54,2 %), 1 giống xe buýt (2,3 %), 2 không rõ (8,6 %) | (a) vùng bỏ qua cho mọi lớp | không có GT truck. Detection không ghép GT mà IoA với box others ≥ ngưỡng → không FP | box others vào mảng vùng bỏ qua dùng chung mọi lớp |
| | (b) GT truck | truck GT 16.915 box / 15.957 ảnh. Detection bus đúng trên track giống xe buýt → FP bus + FN truck. AP truck bị vài track dài chi phối (7 xe cứu hộ chiếm 54,2 % box) | không đủ bằng chứng mọi `others` là truck; là điều kiện của D3 (b) |
| **D3** lớp báo AP. GT tương thích: car (xe con / taxi), bus (xe buýt thành phố) | (a) car, bus; "mAP (car, bus)" = trung bình 2 lớp; truck / motorcycle / bicycle = N/A | Scorer giữ nguyên vẫn đếm TP / FP thô và precision cho lớp không GT (precision phụ thuộc xử lý ignore, đính chính R46-N1 ở mục 10); AP / recall / F1 = null; lớp không vào mAP / macro. Số thô không phải metric. Nếu D2 = (b) mà chọn (a): truck có GT nên scorer đưa truck vào mAP, bước chuẩn hoá phải loại ra | cần **bước chuẩn hoá báo cáo theo D3** (JSON và CSV hiển thị N/A): chưa có, triển khai sau khi duyệt, kèm fixture qua entry point thật |
| | (b) car, bus, truck; "mAP (car, bus, truck)" | motorcycle / bicycle = N/A; AP truck mang các hệ quả của D2 (b) | chỉ có nghĩa khi D2 = (b); vẫn cần bước chuẩn hoá như (a) |
| **D4** 173 ảnh không có `<frame>`, ở 6 video (khe giữa MVI_39051, MVI_39211, MVI_39271; đuôi MVI_40743, MVI_40763, MVI_40793). XML không liệt kê các ảnh này và không có frame rỗng nào: đây là "không có annotation", không phải "đã xác nhận không có xe" | (a) loại khỏi đánh giá, ghi danh sách | 56.167 ảnh chấm; GT không đổi | danh sách ảnh loại (`logs/review45/phan_tich/anh_khong_frame.csv`) ghi vào fingerprint và mỗi lượt (dự thảo đầu file, mục 7) |
| | (b) giữ làm ảnh âm tính | 56.340 ảnh (+173, 0,31 %), không GT, vẫn chịu `ignored_region` cấp sequence (+265 lượt vùng). Mọi detection ngoài vùng bỏ qua là FP, kể cả trên xe có thật không gán nhãn (ít nhất MVI_39271 f77–80) → chỉ thêm FP; P và AP không thể tăng; FN không đổi | — |
| **D5** `ignored_region`: 163 vùng. GT có IoA ≥ 0,5 với vùng (max từng vùng / hợp pixel): car 35.248 / 35.757, bus 5.587 / 6.454; 7.779 car nằm trọn trong vùng | (a) quy tắc BDD hiện dùng, không đổi | chỉ detection **không** ghép GT bị xét: IoA max theo từng vùng ≥ ngưỡng IoU đang xét → bỏ qua (không TP, không FP). GT trong vùng vẫn là GT (detect → TP, sót → FN) | cùng quy tắc với BDD val vòng 43; evaluator `eval_detector_5lop.py` (sha256 `30a1fd5a…`) không đổi |
| | (b) theo phần mã toolkit đã đọc: hợp vùng, ngưỡng cố định 0,5, loại detection trước khi chấm | đổi tập detection được chấm và cách tính TP / FP / FN; recall / AP có thể tăng hoặc giảm, chiều và mức trên dữ liệu thật chưa biết | cần sửa evaluator hoặc thêm bộ chấm: **ngoài kế hoạch**, phải duyệt riêng. Xử lý GT của đường DET (`AP_DET_EVAL.exe`) chưa xác minh, nên vẫn không phải điểm DET chính thức. Cách áp cho box van / others khi D1 / D2 = (a) chưa định nghĩa |
| | (c) tự định nghĩa: giữ evaluator, chuyển GT có IoA ≥ 0,5 thành vùng bỏ qua | đổi tập GT: car / bus đạt ngưỡng như cột đầu; thêm van 2.844 / 3.035 nếu D1 = (b), others 401 / 667 nếu D2 = (b). Recall / AP có thể tăng hoặc giảm (phản ví dụ fixture E2: AP50 0,5050 → 0) | quy tắc GT mới của dự án, khác BDD và khác toolkit; cần chốt cách tính IoA (max từng vùng hay hợp pixel) |

### 11.3 Số đếm theo tổ hợp D1 × D2 × D4

Nguồn `logs/review46/phan_tich/dem_theo_phuong_an.json` (bản sửa R45-F3; reviewer `e4a0311` tính lại khớp). Số trên box
XML gốc.

| D1 `van` | D2 `others` | D4 173 ảnh | Ảnh chấm | GT car | GT bus | GT truck | Ảnh không có GT | Lượt vùng bỏ qua | Lớp có GT |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| (a) bỏ qua | (a) bỏ qua | (a) loại | 56.167 | 548.555 | 71.785 | — | 72 | 275.724 | car, bus |
| (a) bỏ qua | (a) bỏ qua | (b) giữ | 56.340 | 548.555 | 71.785 | — | 245 | 275.989 | car, bus |
| (a) bỏ qua | (b) truck | (a) loại | 56.167 | 548.555 | 71.785 | 16.915 | 72 | 258.809 | car, bus, truck |
| (a) bỏ qua | (b) truck | (b) giữ | 56.340 | 548.555 | 71.785 | 16.915 | 245 | 259.074 | car, bus, truck |
| (b) car | (a) bỏ qua | (a) loại | 56.167 | 587.074 | 71.785 | — | 0 | 237.205 | car, bus |
| (b) car | (a) bỏ qua | (b) giữ | 56.340 | 587.074 | 71.785 | — | 173 | 237.470 | car, bus |
| (b) car | (b) truck | (a) loại | 56.167 | 587.074 | 71.785 | 16.915 | 0 | 220.290 | car, bus, truck |
| (b) car | (b) truck | (b) giữ | 56.340 | 587.074 | 71.785 | 16.915 | 173 | 220.555 | car, bus, truck |

- Ảnh có từng lớp: car 55.764 (D1 = (a)) hoặc 55.994 (D1 = (b)); bus 30.922; truck 15.957 (D2 = (b)).
- "Ảnh không có GT": ảnh chấm không còn box GT nào; với D4 = (b) gồm cả 173 ảnh giữ thêm.
- "Lượt vùng bỏ qua": số lượt vùng áp dụng trên ảnh, đếm lặp qua ảnh (`ignored_region` cấp sequence × ảnh chấm, cộng box
  van / others chuyển thành vùng); **không** phải số vùng hình học (163).
- D3 không đổi các số này. D5 = (a) không đổi; (c) đổi tập GT (11.2); (b) chưa xác định.

### 11.4 Phiếu chọn — **NGƯỜI DÙNG ĐÃ CHỐT** (2026-10-09; bản mốc C ghi "CHƯA DUYỆT")

| Quyết định | Phương án | Lựa chọn của người dùng | Ngày quyết định | Nguồn |
|:--|:--|:--|:--|:--|
| D1 `van` | (a) vùng bỏ qua / (b) GT car | Chuyển các box van thành vùng bỏ qua cho mọi lớp; không gộp van vào GT car. | 2026-10-09 (Asia/Ho_Chi_Minh) | chuỗi email QK9N (nguyên văn dưới bảng) |
| D2 `others` | (a) vùng bỏ qua / (b) GT truck | Chuyển các box others thành vùng bỏ qua cho mọi lớp; không gán toàn bộ others thành GT truck. | 2026-10-09 (Asia/Ho_Chi_Minh) | chuỗi email QK9N (nguyên văn dưới bảng) |
| D3 lớp báo AP | (a) car, bus / (b) car, bus, truck (chỉ khi D2 = (b)) | Chỉ báo AP cho car và bus. mAP là trung bình hai lớp, ghi rõ “mAP (car, bus)”. Truck, motorcycle và bicycle ghi N/A, không tính vào mAP. | 2026-10-09 (Asia/Ho_Chi_Minh) | chuỗi email QK9N (nguyên văn dưới bảng) |
| D4 173 ảnh không có `<frame>` | (a) loại, ghi danh sách / (b) giữ làm ảnh âm tính | Loại 173 ảnh không được liệt kê bằng phần tử frame trong XML khỏi tập đánh giá và lưu danh sách. Không coi các ảnh này là ảnh âm tính đã xác nhận. | 2026-10-09 (Asia/Ho_Chi_Minh) | chuỗi email QK9N (nguyên văn dưới bảng) |
| D5 `ignored_region` | (a) quy tắc BDD / (b) theo phần mã toolkit đã đọc (cần kế hoạch sửa evaluator) / (c) tự định nghĩa (cần chốt cách tính IoA) | Giữ quy tắc dự án đang dùng trên BDD: ghép detection với GT hợp lệ trước; chỉ xét bỏ qua đối với detection chưa ghép. IoA tính bằng diện tích giao chia diện tích detection, lấy max theo từng vùng; ngưỡng IoA bằng ngưỡng IoU đang xét. GT hợp lệ nằm trong vùng bỏ qua vẫn được chấm. Không tô xám vùng bỏ qua trên ảnh. | 2026-10-09 (Asia/Ho_Chi_Minh) | chuỗi email QK9N (nguyên văn dưới bảng) |

**Ngày và nguồn** (nguyên văn):
- Ngày quyết định: 2026-10-09, múi giờ Asia/Ho_Chi_Minh.
- Nguồn quyết định: phản hồi trực tiếp của chủ tài khoản trong chuỗi email QK9N này, xác nhận lại năm lựa chọn D1–D5 đã nêu trong cuộc trao đổi.

> **Cập nhật 2026-10-09** (sau mốc C; ghi theo quyết định người dùng `P712983ae`): cột "Lựa chọn của người dùng" chép
> nguyên văn từ văn bản quyết định đó. Người ghi không trực tiếp đọc chuỗi email, không diễn giải, không quy đổi sang chữ
> (a) / (b) / (c). Bản mốc C ghi "**CHƯA DUYỆT** (để trống)" ở cả năm ô và "**CHƯA DUYỆT**" ở tiêu đề 11.4; hai cột ngày /
> nguồn mới thêm. Cột "Phương án" và mục 11.1–11.3 không đổi. Đầu mục 11, mục 11.5, 11.6 (kể cả "Trạng thái sau mốc C")
> và hai gạch đầu dòng dưới đây (mẫu trả lời, đề xuất vòng 45) giữ nguyên, là trạng thái lúc mốc C; căn cứ là nguyên văn ở
> bảng. Ghi lựa chọn **chưa** cho phép chạy: chưa dựng fingerprint, chưa viết adapter hay bước chuẩn hoá báo cáo theo D3,
> chưa khóa kế hoạch, chưa chạy đánh giá; mỗi bước cần người dùng duyệt riêng (11.5).

- Mẫu trả lời: `D1=a|b; D2=a|b; D3=a|b; D4=a|b; D5=a|b|c`. Người dùng cũng có thể chọn khác hoặc yêu cầu thêm phân tích
  trước khi chọn.
- Đề xuất của tác giả ở vòng 45 (`REVIEW45_CHI_TIET.md` mục 3, dự thảo mục 4) là phương án (a) cho cả năm quyết định. Đó
  **chỉ là đề xuất**: chưa được duyệt, không có hiệu lực, không dùng làm mặc định. Review `e4a0311` cũng ghi: nhận xét về
  quy tắc chấm không thay cho quyết định D1–D5.

### 11.5 Sau khi người dùng chọn

Chốt D1–D5 **chưa** cho phép chạy đánh giá. Các bước tiếp theo theo trình tự ở đầu dự thảo (fingerprint dữ liệu → adapter
+ fixture → khóa kế hoạch mới, commit trước suy luận → ba lượt tuần tự trên PC); mỗi bước cần người dùng duyệt và không
thuộc kế hoạch R46. Nếu chọn D5 = (b), cần thêm kế hoạch riêng để sửa evaluator.

### 11.6 File sửa ở mốc C

- `REVIEW46_CHI_TIET.md`: thêm mục 11 ở cuối file. Khối đầu và mục 1–10 không sửa, để số dòng trích ở mục 10 vẫn đúng.
- `REVIEW_BRIEF.md`: phần đầu viết theo trạng thái sau `e4a0311` và bước tiếp theo. Brief vòng 46 cùng hai khối cập nhật
  mốc A / B giữ nguyên văn ở phụ lục cuối file (chỉ hạ cấp tiêu đề; dấu điền base / target chuyển lên đầu file, còn đúng
  một chỗ).
- `REVIEW45_CHI_TIET.md`: thêm một ghi chú ngay sau bảng mục 3, trỏ tới mục 11. Bảng và các dòng phía trên không đổi.
- `CODEX_HANDOFF.md`: nối trạng thái mốc C vào cuối cột trạng thái hàng 42, cùng dòng. Hàng 41 không đổi.
- `logs/review45/06_du_thao_giao_thuc_detrac.md`: **không sửa**. Dự thảo đã ghi D1–D5 chưa duyệt, dự thảo chưa khóa và chữ
  in đậm ở mục 4 là đề xuất chờ duyệt; giữ nguyên để số dòng trích ở mục 10 không lệch.

**Kiểm sau khi sửa** (Grep và `git diff`, chỉ đọc; không chạy script, không chạy test vì mốc C chỉ sửa tài liệu):
- câu đính chính N1 nguyên văn: dự thảo dòng 108, file này dòng 110, REVIEW45 dòng 118 (đúng ba vị trí và số dòng mục 10
  trích), cộng một lần trích ở mục 10; mục 11 không lặp lại câu này;
- câu đính chính N2 nguyên văn: REVIEW45 dòng 73, cộng một lần trích ở mục 10;
- cụm cũ "0 nếu có detection lớp đó": chỉ trong ba ghi chú đính chính, mục 9 và mục 10, như sau mốc B;
- brief: dấu điền base / target còn đúng **một** chỗ (dòng 3). So với bản sau mốc B (`git diff` trước và sau khi sửa),
  phụ lục chỉ khác ở bốn tiêu đề hạ cấp; tiêu đề gốc được trích trong câu dẫn của phụ lục, dấu điền chuyển lên đầu file;
  hai khối cập nhật mốc A / B giữ nguyên văn;
- `git diff --numstat`: `CODEX_HANDOFF.md` 1 dòng (hàng 42); dự thảo `06_*` vẫn +8 / −2 như sau mốc B; `PLAN_DA_CHOT.md` và
  `review_base.json` không có trong diff; `git diff --cached` rỗng;
- `.gitignore` và `src/demo/rerun_with_state_log.py` đã ở trạng thái sửa trong `git status` đầu phiên, trước mốc C; mốc C
  không đụng tới;
- số ở 11.3: đối chiếu bằng cách đọc `dem_theo_phuong_an.json`, khớp 8/8 tổ hợp (ảnh chấm, GT từng lớp, ảnh không có GT,
  lượt vùng, lớp có GT). Số ở 11.1–11.2: đối chiếu với `REVIEW45_CHI_TIET.md` mục 2–4 và mục 3 file này.

**Trạng thái sau mốc C**: tài liệu của ba mốc A, B, C đã làm, chưa commit; mốc C chưa được review. **D1–D5 CHƯA DUYỆT.**
Bước tiếp theo: người dùng chốt D1–D5 theo phiếu ở 11.4.
