# Vòng 44 — review `a03f6e1`: bổ sung mã phụ thuộc (R43-B1) và sửa metadata ngưỡng vận hành (R43-N1) (chi tiết)

**Phạm vi** (người dùng duyệt 2026-10-07):
- Chỉ xử lý review `a03f6e1`: sửa nhỏ, kiểm bằng fixture, đóng gói cho Codex review tiếp.
- Không train, không chạy lại inference BDD, không SSH Nano, không triển khai UA-DETRAC. D1–D5 vẫn chưa được duyệt.
- Review chưa PASS đầy đủ, nên **base giữ `4d54744`**. Vòng 44 **chưa** được nghiệm thu.

**Tóm tắt**
- **R43-B1** là **thiếu dependency** trong môi trường reviewer, không phải lỗi evaluator. Gói này gửi kèm 17 file mã /
  cấu hình **không đổi** trong `support/`. Test BDD chạy được trên cây sạch ngoài repo bằng mã thật.
- **R43-N1** là **lỗi metadata**, chỉ xảy ra khi `--conf-van-hanh` khác mặc định. Đã sửa, có regression qua entry point;
  regression bắt được mã cũ.

## 0. Trạng thái đầu vòng

Log: `logs/review44/00_trang_thai_dau.log`.
- HEAD `a03f6e1` = target được review; không có commit mới. Working tree chỉ có file riêng của người dùng.
- Môi trường như vòng 43 (Python 3.11.9, ultralytics 8.4.141, torch 2.6.0+cu124). Đang cắm sạc, gói High performance.

## 1. Ghi nhận review `a03f6e1` (commit `c9e673f`)

- **Báo cáo** `handoff/CODEX_REVIEW_a03f6e1.md`: trùng byte bản trong `D:\LABOTARY\review_codex` (8.707 B, sha256
  `9c0e87ae…`).
- **Kết luận**: **CHƯA NGHIỆM THU ĐẦY ĐỦ** — cần bổ sung mã phụ thuộc. Phần reviewer đã kiểm độc lập:
  - gói 319.428 B `f93a35e6…`: 66 thành viên; 61/61 đường dẫn khớp target; 6 preimage khớp base;
  - số học 180/180. Đây là đối chiếu hồ sơ; reviewer không tính lại AP từ dự đoán.
- **Bằng chứng**: `CODEX_REVIEW_a03f6e1_EVIDENCE.zip` 52.594 B, sha256 `cc205d04…`. Giải nén vào thư mục mới, ZIP gốc
  giữ nguyên. `SHA256SUMS` khớp **28/28** trên đĩa và trên blob đã stage. Lưu ở `handoff/review_evidence/CODEX_a03f6e1/`.
- **Gói đã review** lưu ở `handoff/archive/review_delta_a03f6e1.zip` (chỉ cục bộ). `review_base.json` **không đổi**.
- **Giới hạn ghi đúng như review**:
  - reviewer không có checkout Git đầy đủ, nên chưa tự xác nhận "4.642 file toàn cây";
  - AP theo lớp vẫn là đầu vào lấy từ kết quả tác giả;
  - nhánh GT mới đưa mọi vùng bỏ qua vào một mảng `ign` dùng chung cho mọi lớp. Đây là **quy tắc vùng bỏ qua của dự
    án**, không mặc nhiên đồng nhất với benchmark COCO / BDD chính thức;
  - chênh 0,397 điểm mAP50–95 giữa hai model fine-tune chưa phải căn cứ chọn model;
  - D1–D5 chưa được chốt.

## 2. R43-N1 — metadata ngưỡng vận hành (commit `5cbd434`)

**Tái hiện trước khi sửa** (`01_tai_hien_r43n1_truoc_sua.log`): chạy script reviewer `REPRO_REVIEW_a03f6e1.py` nguyên
byte (sha256 `ef71cb76…`). Script dùng **dependency doubles**: nó chỉ kiểm đường truyền tham số và metadata ghi ra,
**không** chứng minh AP đúng. Kết quả: với `--conf-van-hanh 0.5`, scorer nhận 0,5 nhưng `quy_tac.van_hanh.conf` = 0,25;
`bug_count 1`, exit 1.

**Sửa** (`src/danh_gia_detector_bdd_val.py`):
- `quy_tac_hieu_luc(conf)` tạo **bản sao sâu** của `QUY_TAC`, mang ngưỡng thực dùng. Metadata `quy_tac`, tham số dự đoán,
  `danh_gia` và `chan_doan` đều lấy ngưỡng từ bản sao này.
- Nếu `van_hanh` mà scorer trả về khác cấu hình hiệu lực thì exit 3.
- Hằng dùng chung `QUY_TAC` **không** bị sửa.
- Không đổi thuật toán AP, ghép GT, vùng bỏ qua, NMS hay ánh xạ lớp. Lời gọi `danh_gia(gt, dd, conf)` giữ 3 tham số vị trí
  như cũ.

**Regression qua entry point** (`src/test_danh_gia_bdd_val.py`, nhóm B5). Ngưỡng scorer thực dùng được bắt bằng hàm bọc
quanh `danh_gia` thật.
- **B5a** mặc định: metadata = ngưỡng scorer = 0,25.
- **B5b** `--conf-van-hanh 0.5`: cả hai bằng 0,5. P/R và chẩn đoán trùng lời gọi trực tiếp với 0,5, và khác kết quả 0,25.
- **B5c** hai lần gọi liên tiếp trong cùng tiến trình (0,4 rồi mặc định): mỗi lần dùng đúng ngưỡng của nó; `QUY_TAC` và
  mặc định CLI vẫn 0,25.

**Chứng minh regression** (`03_regression_b5_ma_cu_bien_the_moi.log`): cùng test mới, cây sạch, ba bản entry point.

| Entry point | Kết quả | Phép kiểm FAIL |
|:--|:--|:--|
| cũ `a03f6e1` (`d1e2fd22…`) | 20/22, exit 1 | B5b (metadata 0,25 ≠ scorer 0,5), B5c (0,25 ≠ 0,4) |
| biến thể sai: sửa trực tiếp `QUY_TAC` (`tao_bien_the_sai.py`) | 21/22, exit 1 | B5c (lần gọi mặc định rò thành 0,4) |
| mới `5cbd434` (`de2c06f4…`) | **22/22**, exit 0 | — |

**Script reviewer sau khi sửa** (`04_repro_reviewer_sau_sua.log`): `bug_count 0`, exit 0. Ở cả 0,25 và 0,5, metadata
khớp ngưỡng scorer.

Ba lượt BDD vòng 43 đều dùng 0,25, nên lỗi không kích hoạt. Không sửa JSON hay kết quả lịch sử, không chạy lại GPU.
`07_kiem_khong_doi.log` xác nhận:
- 9 file lớn của vòng 43 khớp bytes và sha256;
- 12 file kết quả nhỏ trùng bản đã commit;
- 3 checkpoint đúng hash;
- 170 file riêng không file nào bị sửa trong vòng 44, không file nào bị stage.

## 3. R43-B1 — mã phụ thuộc gửi kèm (`support/` trong gói này)

Gói dùng **cơ chế `--support` sẵn có** của packager:
- file đặt ở `support/<đường dẫn>`, không vào patch, không tính là thay đổi;
- packager **từ chối** file khác nhau giữa base và target;
- packager gửi **blob Git của target, nguyên byte**.

Cách xác định danh sách:
1. Phân tích import tĩnh (AST, gồm cả import trong hàm) từ entry point, test BDD và test cũ `test_detector_5lop.py`.
2. Thử trên cây sạch ngoài repo (`05_do_dependency_cay_sach.log`). Bộ 9 file lõi đủ cho test BDD: 22/22, 10 mô-đun dự
   án đều nạp từ cây. 17 file đủ cho test cũ: 76/76 + 6 SKIP khi không có weights; 92/92 khi có weights ngoài gói.
3. Test cũ còn chạy `prepare_mixed_yolo_dataset.py` và `train_detector.py` qua subprocess, nên hai file này cùng bao đóng
   của chúng nằm trong danh sách.

Danh sách đầy đủ và vai trò từng file: `logs/review44/phu_thuoc_python.md`. Tóm tắt:
- **lõi**: `config`, `bdd_yolo_export`, `detector_classes`, `eval_detector_5lop`, `mot_datasets/{__init__, base,
  bdd100k_coco, bdd100k_mot, detrac}`;
- **cho test cũ**: `test_detector_5lop`, `train_detector`, `tiep_tuc_huan_luyen`, `pilot_chan_doan`,
  `mot_datasets/make_bdd5_train_fixture`, `prepare_mixed_yolo_dataset`, `export_yolo_dataset`,
  `configs/phase2/detector/train_bdd5.yaml`.

`detrac`, `pilot_chan_doan` và `export_yolo_dataset` chỉ được tham chiếu tĩnh, không nạp khi chạy test. Chúng được gửi
để bao đóng import tự đủ. Không gửi dataset, ảnh, weights, NPZ hay toàn repo.

**Đối chiếu hash với bảng của review** (`06_doi_chieu_hash_support.log`, 17/17 đạt). Mọi file support có blob tại base
**trùng** target; commit cuối chạm các file này đều trước base.

| File | Bảng review | Blob gửi trong `support/` | Giải thích |
|:--|:--|:--|:--|
| `src/eval_detector_5lop.py` | `30a1fd5a…` | `30a1fd5a…` | trùng |
| `src/detector_classes.py` | `7bba3439…` | `7bba3439…` | trùng |
| `src/bdd_yolo_export.py` | `1ca964b0…` | `2fbc2314…` | chỉ khác EOL, xem dưới |
| `src/mot_datasets/bdd100k_coco.py` | `5028ffb9…` | `c6c3ad28…` | chỉ khác EOL, xem dưới |
| `src/config.py` | (ghi khi đóng gói) | `0e3f8b75…` | file trên đĩa là `37e4f562…` (CRLF) |

Repo dùng `* text=auto`, máy dùng `core.autocrlf=true`, nên checkout ghi CRLF ra đĩa trong khi blob lưu LF. Hash trong hồ
sơ chạy vòng 43 băm **file trên đĩa**, tức đúng bytes đã chạy.
- Đổi CRLF→LF bản trên đĩa thì trùng blob.
- Đổi mọi LF của blob thành CRLF thì trùng **tung byte** bản trên đĩa (6/6 file CRLF).

Vậy đây là chuyển đổi EOL khi checkout, **không phải** snapshot khác hay thay đổi có chủ ý. Python đọc mã nguồn theo
universal newlines, nên hai dạng EOL chạy như nhau. Reviewer tái tạo hash của bảng bằng
`sha256(blob.replace(b"\n", b"\r\n"))`.

**Gói Python ngoài**:
- test BDD cần numpy, pandas, Pillow, opencv;
- test cũ cần thêm PyYAML, ultralytics, torch, pycocotools, và weights qua `DET5_MODELS_DIR`;
- pyarrow **không** cần.

Phiên bản đã dùng và nhóm cần tài nguyên ngoài gói: `phu_thuoc_python.md`.

## 4. Kiểm trên cây sạch ngoài repo — `logs/review44/kiem_cay_sach.py`

- **Dựng cây** trong thư mục **mới** ngoài repo, theo một trong hai cách:
  - từ gói delta (`--zip`): kiểm sha256 mọi file với MANIFEST;
  - từ blob Git (`--tu-git`, có `--thay` để cài entry point cũ hay biến thể).
- **Chạy** `python <cây>/src/<test>.py` với cwd ngoài repo. Môi trường bỏ PYTHONPATH / PYTHONHOME / user site;
  PYTHONPATH **chỉ** là một shim `sitecustomize.py`.
- **Ghi nguồn mô-đun**: shim ghi file của mọi mô-đun đã nạp, cho **mọi tiến trình Python** kể cả tiến trình con.
  Mô-đun dự án phải nằm trong cây; một mô-đun nạp từ repo gốc là KHÔNG ĐẠT.
- **Sau khi chạy**: kiểm lại sha256 của cây; test không được sửa mã.
- **`--chan-pyarrow`**: trên máy này, Smart App Control chặn DLL `pyarrow._compute`, nên `import pandas` lỗi
  (`02_test_bdd_repo_lan1.log`; CodeIntegrity 3033/3077 trong `02a_*`). Lần chạy test BDD trong repo vì thế **FAIL trước
  khi chạy phép kiểm nào**; log thất bại được giữ. Cờ này làm `import pyarrow` thất bại có chủ ý trong tiến trình kiểm,
  nên pandas chạy như khi không cài pyarrow. Mã dự án không dùng pyarrow, và cờ này không thay mô-đun dự án nào.
- **Kiểm gói cuối**: chạy cùng công cụ ở chế độ `--zip` trên chính gói bàn giao. Kết quả nằm trong tin nhắn bàn giao, vì
  log của gói không thể nằm trong chính gói đó. Reviewer chạy lại được bằng lệnh ở `phu_thuoc_python.md`, bỏ
  `--chan-pyarrow` nếu máy không bị chặn.

## 5. Bảng test (PASS / FAIL / SKIP / NOT_RUN)

| Phép chạy | Mã | Kết quả | Log |
|:--|:--|:--|:--|
| Script reviewer trước sửa (doubles; chỉ metadata) | `a03f6e1` | `bug_count 1`, exit 1 — **tái hiện** | `01` |
| `test_danh_gia_bdd_val.py` trong repo | sửa, chưa commit | **FAIL** 0/1: ImportError do SAC chặn pyarrow (môi trường) | `02`, `02a` |
| Test BDD, cây sạch, 9 file lõi | `5cbd434` | **22/22 PASS** | `05` (A) |
| Test BDD, cây sạch, 17 file | `5cbd434` | **22/22 PASS** | `05` (B) |
| Test mới trên entry point cũ | `a03f6e1` | 20/22 — **FAIL đúng dự kiến** (B5b, B5c) | `03` |
| Test mới trên biến thể sai | biến thể | 21/22 — **FAIL đúng dự kiến** (B5c) | `03` |
| Test mới trên mã sửa | `5cbd434` | **22/22 PASS** | `03` |
| Script reviewer sau sửa | `5cbd434` | `bug_count 0`, exit 0 | `04` |
| `test_detector_5lop.py`, cây sạch, không weights | không đổi | **76/76 PASS + 6 SKIP** (A7×2, E6, F1–F5, G, J3/J4/J6: CHƯA KIỂM) | `05` (B) |
| `test_detector_5lop.py`, cây sạch, weights ngoài gói | không đổi | **92/92 PASS** | `05` (C) |
| Kết quả BDD / checkpoint / file riêng không đổi | — | đạt | `07` |
| Gói cuối, cây sạch (`--zip`) | target | trong tin nhắn bàn giao | — |
| Test trên Linux / Python 3.12 (môi trường reviewer) | — | **NOT_RUN** (tác giả chỉ có Windows / 3.11.9) | — |
| Training, inference dữ liệu thật, full suite tracking / Nano | — | **NOT_RUN** (ngoài phạm vi; vòng này không đổi đường đó) | — |

## 6. Giới hạn

- Các lần kiểm chạy trên Windows, CPython 3.11.9, với pyarrow bị chặn có chủ ý. Chưa chạy trên môi trường của reviewer
  (Linux, 3.12).
- Script reviewer và nhóm B5 chỉ kiểm **tính nhất quán của ngưỡng** giữa metadata, scorer và chẩn đoán. Chúng không
  chứng minh AP đúng.
- Bằng chứng độc lập về cách tính AP vẫn là nhóm E7–E9 của test cũ, vốn so với pycocotools thật và cần thư viện đó.
- 92/92 cần weights **ngoài gói** (`DET5_MODELS_DIR`); không có weights thì 6 nhóm SKIP "CHUA KIEM", không tính PASS.
- Gói vẫn không có dataset, ảnh, weights hay NPZ. Reviewer chỉ có hash và đường dẫn của chúng.
- Các giới hạn của review `a03f6e1` ở mục 1 vẫn còn. UA-DETRAC chưa chạy, D1–D5 chưa chốt.

## 7. Lệnh (từ gốc repo) và mã thoát

```text
python handoff/review_evidence/CODEX_a03f6e1/REPRO_REVIEW_a03f6e1.py . --out <json>                    # truoc sua: exit 1; sau sua: exit 0
python src/test_danh_gia_bdd_val.py                                                                     # trong repo: exit 1 (SAC chan pyarrow)
python handoff/logs/review44/kiem_cay_sach.py --thu-muc <moi> --tu-git <commit> --duong-dan <file...> \
    [--thay src/danh_gia_detector_bdd_val.py=<file>] --chay <test> --chan-pyarrow [--models-dir <weights>] --out <json>
python handoff/logs/review44/tao_bien_the_sai.py <entry point 5cbd434> <bien the>                        # exit 0
python handoff/logs/review44/doi_chieu_hash_support.py --base 4d54744 --target HEAD --out <json> <17 file>  # exit 0, 17/17
python handoff/logs/review44/kiem_khong_doi.py --moc "2026-10-07 01:19:00"                              # exit 0
python src/make_review_delta.py --support <17 file>                                                     # goi + roundtrip
python handoff/logs/review44/kiem_cay_sach.py --zip handoff/review_delta_latest.zip --thu-muc <moi> --chay ... # kiem goi cuoi
```

## 8. Commit vòng 44

| Commit | Nội dung |
|:--|:--|
| `c9e673f` | Ghi nhận review `a03f6e1` + bằng chứng 28/28, archive gói; base giữ `4d54744` |
| `5cbd434` | Sửa R43-N1 + regression B5 |
| `b0c0ae7` | Bằng chứng: tái hiện, regression mã cũ / biến thể sai / mã mới, script reviewer, đo dependency, đối chiếu hash |
| commit chứa phụ lục này (target delta) | `07_kiem_khong_doi`, brief, phụ lục, handoff |
