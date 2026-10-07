# Vòng 46 — xử lý review `d4715aa` (R45-F1…F4); chưa triển khai đánh giá UA-DETRAC (chi tiết)

**Phạm vi** (người dùng duyệt 2026-10-07):
- Chỉ xử lý bốn finding và ba điểm cần làm rõ của review `d4715aa`.
- Không train, không GPU, không SSH Nano, không khóa giao thức, không viết adapter UA-DETRAC.
- Không đụng dataset, checkpoint, kết quả cũ, `.gitignore`, `src/demo/**`.
- **Base review giữ `3e94b21`.** `d4715aa` chưa PASS. **D1–D5 vẫn chưa được người dùng duyệt.** Vòng 46 **chưa** được
  nghiệm thu.

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
  điểm vận hành **vẫn được đếm**, precision = TP / (TP + FP) vẫn tính (= 0 nếu có detection lớp đó với conf ≥ 0,25), và
  `n_det` vẫn đếm.
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
