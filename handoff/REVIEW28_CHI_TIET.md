# Vòng 28 — sửa R27-F1/F2/F3 của review `4937c85` (chi tiết)

Review: `handoff/CODEX_REVIEW_4937c85.md` (REQUEST CHANGES). **Base giữ `239b9c6`**. Trạng thái: **đã sửa, chờ Codex nghiệm thu**.
Phạm vi: chỉ sửa ba finding, kiểm thử, cập nhật tài liệu và delta. Không chạy training dài, không duyệt 15 epoch hay
ngân sách, không thêm dataset/model/thuật toán; không đụng Nano v16.

Nhãn nguồn bằng chứng dùng trong phụ lục:

- **chạy thật**: chạy thật trên PC với dữ liệu tổng hợp trong thư mục tạm;
- **fixture GPU**: training 1 epoch thật trên GTX 1650 với dữ liệu tổng hợp;
- **giả lập**: spy hoặc trainer giả, luôn ghi rõ;
- **bằng chứng GPU cũ**: số đo vòng 27, không chạy lại trong vòng này.

## 0. Trạng thái đầu vòng, file review, bản sao mã cũ

- **Trạng thái đầu vòng**: HEAD `4937c85`, 0 commit sau đó. Working tree chỉ có file riêng của người dùng (`.gitignore`,
  `src/demo/…`). Môi trường: Python 3.11.9, ultralytics 8.4.141, torch 2.6.0+cu124, pycocotools 2.0.11, OpenCV 5.0.0
  (`logs/review28/00_trang_thai_dau.log`).
- **File review**: ba file review **chưa có trên máy lúc bắt đầu vòng**. Người dùng gửi sau dưới dạng tài liệu đính
  kèm; Claude chép nguyên văn vào `handoff/` (`CODEX_REVIEW_4937c85.md`, `REPRO_REVIEW_4937c85.py`,
  `REPRO_REVIEW_4937c85.json`). Không có bản gốc trên đĩa để so byte (`00b_file_review_nhan_sau.log`).
- **Bản sao mã cũ**: chỉ checkout `src/` và `configs/` của `4937c85` vào scratchpad, vì checkout đầy đủ vượt giới hạn
  đường dẫn Windows do log vòng 25 quá dài. sha256 các file nguồn khớp `git show 4937c85` (`02_ban_sao_ma_cu.log`).
  Mọi phép tái hiện “trước sửa” chạy trên bản sao này, **không** trên working tree. Một phần sửa exporter đã bắt đầu
  trong working tree trước khi file review tới.

## 1. Tái hiện trước khi sửa (chạy thật, mã `4937c85`)

| Script | Kết quả | Log |
|:--|:--|:--|
| `REPRO_REVIEW_4937c85.py` (Codex) | **exit 1**; 6/6 cờ `bug_reproduced: true`; dataset gốc 77 ảnh train / 48 ảnh dev; F3: AP50 0,504950495 so với pycocotools 1,0; AP50–95 0,353465 so với 0,551485; TP@0,5 là 1 so với 2. Probe phụ `internal_duplicate_probe`: xuất được dù ảnh dev trùng nội dung ảnh train-fit | `03_codex_repro_ma_cu.{stdout,stderr,exit,json}` |
| `tai_hien_r27.py` (Claude tái dựng từ mô tả, trước khi có file Codex) | **exit 1**; 7/7 ca tái hiện. Runner còn báo 57 ảnh train khi YAML trỏ train vào dev (36 ảnh) | `01_tai_hien_ma_cu.*` |

## 2. R27-F1 — chốt đầu vào của runner

**Trước sửa**:

- `train_detector.kiem_dataset()` chỉ kiểm `val`, hash CSV và đếm hai thư mục cố định.
- Nó nhận YAML có `train: images/dev` và vẫn báo số ảnh train cũ.
- Nó cũng nhận dataset thiếu file nhãn.

**Thay đổi**:

- **Hợp đồng dùng chung**: `bdd_yolo_export.kiem_hop_dong_dataset()` được dùng cho exporter (cuối `--buoc xuat`,
  `--buoc kiem`), runner (`train_detector.kiem_dataset`) và evaluator (`chuan_bi_danh_gia`).
- **Kiểm đúng đường dẫn ultralytics resolve từ `data.yaml` nguyên văn**:
  - `train` = `images/train`, `val` = `images/dev`, không có `test`;
  - `path` là đường dẫn tuyệt đối tới chính thư mục chứa YAML, vì ultralytics resolve `path` tương đối theo
    `DATASETS_DIR`, không theo file YAML;
  - liệt kê ảnh như `BaseDataset.get_img_files` (glob đệ quy `**/*.*` + `IMG_FORMATS`);
  - suy file nhãn như `img2label_paths` (thay `/images/` cuối cùng).

  Runner so `IMG_FORMATS` của ultralytics đang chạy với bảng của hợp đồng; khác thì dừng.
- **Split đã khóa**:
  - train/dev thuộc đúng split đã khóa: bản sao `manifest/split_da_khoa.json`, cộng bản gốc nếu còn trên máy;
  - dataset đầy đủ phải dùng toàn bộ video của split;
  - train và dev không trùng video, tên ảnh hay sha256 ảnh (nguồn và xuất).
- **Danh sách file**:
  - ảnh thực tế khớp manifest cho từng split; ảnh thừa hoặc thiếu đều là lỗi;
  - file nhãn **rỗng** hợp lệ, file nhãn **thiếu** là lỗi;
  - file nhãn hoặc ignore thừa là lỗi.
- **Số ảnh báo cáo** lấy từ danh sách ảnh ultralytics sẽ dùng.
- **Thứ tự trong runner**: gọi hợp đồng trước `kiem_weights` (nạp YOLO), trước mọi `mkdir` và trước khi training.
  Lỗi thì trả exit 2 kèm thông điệp nêu file và lý do.
- **Exporter**: dừng khi ảnh dev trùng **nội dung** ảnh train-fit dù khác video (sha256 ảnh nguồn), trước khi ghi ảnh.
  Hành vi này không tự loại ảnh; người dùng quyết định cách xử lý.

**Kiểm sau sửa**:

- Script Codex: F1_train_is_dev và F1_missing_label đều `false` (`06_*`).
- Regression G đi qua **entry point** `train_detector.main()`, `--che-do smoke`. `ultralytics.YOLO` được thay bằng
  **spy (giả lập)** tại ranh giới nạp model/training; phần trước ranh giới chạy thật.
  - G1 (`train: images/dev`): exit 2, spy không được khởi tạo, không gọi train, không tạo thư mục lượt hay file kết quả.
  - G2 (thiếu nhãn của ảnh có box) và G3 (thiếu nhãn của ảnh âm tính): cùng kết quả như G1.
  - G0 (dataset hợp lệ): tới đúng ranh giới training, spy ghi 1 lần gọi train với đúng `data.yaml`.
  - G0b: số ảnh báo cáo 77/48 bằng số ảnh thật; nhãn rỗng hợp lệ được chấp nhận; danh tính dataset đã xác minh được
    ghi vào kết quả.
- Trên mã cũ: G1, G2, G3 FAIL — runner cũ chạy tới train (exit 3, spy bị gọi).

**Giới hạn**:

- Spy chỉ giả lập ranh giới; training thật với runner mới chỉ có smoke fixture (mục 6).
- Dataset gắn với vị trí thư mục (`path` tuyệt đối): chuyển thư mục thì phải xuất lại.

## 3. R27-F2 — khóa và xác minh nội dung dataset

**Trước sửa**:

- Manifest có hash ảnh nhưng bộ kiểm không so với ảnh trên đĩa.
- Nhãn chỉ được kiểm định dạng và số box; file ignore không được kiểm.
- Runner chỉ băm CSV; evaluator chép hash khai báo, không xác minh.

**Thay đổi**:

- **Manifest phiên bản 2** (`phien_ban_manifest: 2`):
  - mỗi ảnh có `nhan_ra` + `sha256_nhan` (kể cả nhãn rỗng: sha256 của 0 byte);
  - `ignore_ra` + `sha256_ignore` khi có vùng tô xám, rỗng khi không có;
  - thêm sha256 của `data.yaml` và `video.json`, và bản sao split đã khóa.

  Nhãn, ignore và `data.yaml` được ghi bằng byte `\n` cố định, nên hash không phụ thuộc hệ điều hành.
- **Hợp đồng** (mục 2) xác minh:
  - sha256 từng ảnh, nhãn và ignore;
  - kích thước ảnh và số box theo lớp;
  - phân biệt: file nhãn rỗng hợp lệ; ảnh không có vùng ignore (không được có file ignore); file ignore đã khai báo mà
    thiếu (lỗi).
- **Manifest v1 bị từ chối** với thông điệp yêu cầu xuất lại. Không bổ sung hash vào metadata cũ. Test H7 xác nhận
  `tom_tat.json` không bị ghi lại. Trên dataset fixture thật của vòng 27, runner mới trả exit 2
  (`fixture_gpu/01a_dry_run_dataset_v1.log`).
- **Evaluator**: `chuan_bi_danh_gia()` chạy hợp đồng, xác minh nội dung split được đánh giá, **trước** khi dự đoán, rồi
  ghi `dataset_da_xac_minh` vào `ket_qua.json`. `doc_gt(..., ignore_khai_bao=…)` đọc theo khai báo manifest: file đã khai
  báo mà thiếu thì lỗi, không đọc thành 0 vùng.
- **Runner** ghi danh tính đã xác minh vào kết quả, gồm hợp đồng, phiên bản, các sha256, số ảnh, split nào đã băm nội
  dung và thời điểm.
- **Cache nhãn của ultralytics**: cache chỉ kiểm **tổng kích thước + đường dẫn** file (`get_hash`), nên một nhãn bị sửa mà
  giữ số byte vẫn dùng cache cũ. Callback `on_pretrain_routine_end` (không nuốt lỗi) so ảnh + nhãn **ultralytics thực
  nạp** với file đã xác minh; khác thì dừng trước vòng train.

**Kiểm sau sửa**:

- **Script Codex**: F2_image_changed, F2_label_coordinates_changed và F2_ignore_removed đều `false`. Thông điệp lần lượt:
  “nội dung ảnh khác manifest”, “nội dung nhãn khác manifest”, “THIẾU file ignore đã khai báo (3 vùng)”.
- **Regression H** — tất cả PASS trên mã sửa:

  | Ca | Nội dung |
  |:--|:--|
  | H1 | Ảnh lật ngang, cùng kích thước |
  | H1b | Ảnh dev đổi, kiểm qua evaluator |
  | H2 | Tọa độ nhãn đổi, giữ lớp và số box |
  | H3 / H3b | File ignore đã khai báo bị xóa |
  | H4 | File ignore bị sửa |
  | H5 / H5b | Có file ignore cho ảnh không có vùng ignore |
  | H6 | Dataset hợp lệ: băm đủ 125 ảnh, 125 nhãn, 11 ignore; nhãn rỗng hợp lệ |
  | H7 | Manifest v1 |
  | H8 | Ảnh thừa trong `images/train` |
  | H9 | `path` tương đối |
  | H10 | Trùng nội dung train/dev lúc xuất; không tạo thư mục |

- **Regression I**:
  - I1–I3 dùng trainer **giả lập**;
  - **I4 dùng ultralytics thật**: tạo cache từ nhãn đã sửa cùng số byte, khôi phục file, hợp đồng file ĐẠT, nhưng
    `YOLODataset` thật dùng lại cache cũ và phép đối chiếu dữ liệu nạp phát hiện “nhãn khác 1”;
  - I5: cache mới thì khớp.
- **Smoke GPU fixture** (mục 6): 719/90 ảnh nạp khớp, 0 lệch.

**Giới hạn**:

- `tom_tat.json` là gốc tin của các hash. Ai sửa đồng thời file và `tom_tat.json` thì hợp đồng không phát hiện; không có ký
  số. Danh tính ghi vào kết quả gồm sha256 của `tom_tat.json` để đối chiếu về sau.
- Chi phí băm: 1,7 s cho 809 ảnh trên fixture. Chưa đo trên ~56K ảnh thật.

## 4. R27-F3 — ghép khi hòa IoU như pycocotools

**Trước sửa**: `np.argmax` chọn GT **đầu** khi hòa IoU. Trên fixture của review: TP@0,5 = 1, AP50 = 0,50495,
AP50–95 = 0,35347; pycocotools cho 2 / 1,0 / 0,55149.

**Thay đổi** (`ghep_anh_lop`):

- Theo đúng vòng lặp `COCOeval.evaluateImg` (`if iou < best: continue; best = iou; m = g`): khi hòa, chọn GT **sau**
  trong thứ tự duyệt (`np.flatnonzero(cand == max)[-1]`).
- IoU và IoA tính như `bbIou` của pycocotools: đổi box sang xywh, cạnh phải = x + w. Nhờ vậy các giá trị hòa giống
  pycocotools tới từng bit.
- **Không đổi** ngưỡng hay công thức AP. Bản làm việc từng thử dùng mảng ngưỡng float của pycocotools rồi **hoàn lại**
  theo yêu cầu giữ ngưỡng. Mảng ngưỡng đã chọn chỉ khác pycocotools ở 0,9 (pycocotools `0.8999999999999999`, lệch
  1 ulp), nên chỉ khác kết quả nếu có IoU nằm trong 1 ulp dưới 0,9.

**Kiểm sau sửa**:

- Script Codex: F3 `false`; AP50 1,0 = pycocotools; TP 2 / FP 0.
  - AP50–95 là 0,5514851485148514 so với 0,5514851485148515 (lệch 1 ulp, do thứ tự cộng số thực: pycocotools lấy trung
    bình cả mảng 10×101, ta lấy trung bình từng ngưỡng rồi trung bình).
- **E7** (fixture review): TP@0,5, AP50 và AP50–95 khớp **pycocotools chạy thật** (sai số < 1e-12).
- **E8** (IoU đúng bằng 0,6 và 0,7) và **E9** (GT hợp lệ chồng vùng ignore: det khớp GT là TP; det chỉ trong vùng bị bỏ
  qua; det xa là FP): khớp pycocotools.
- **E1–E5** giữ nguyên. E1 so ngẫu nhiên có vùng bỏ qua kiểu crowd; E2 lớp không có GT.
- Thiếu pycocotools thì các phép đối chiếu ghi **SKIP “CHƯA KIỂM”**, không tính PASS. Trên PC này pycocotools 2.0.11
  có mặt, và mọi phép đối chiếu đều chạy thật.
- Trên mã cũ: E7 FAIL (3 phép); E8 PASS vì không có hòa IoU nên không phải regression của F3.

**Giới hạn**: tương đương COCO được kiểm trên các ca tổng hợp: ngẫu nhiên, hòa IoU, IoU đúng ngưỡng, crowd. Chưa đối
chiếu trên dự đoán thật của dữ liệu BDD.

## 5. Kiểm chứng tổng hợp (chạy thật)

| Lượt | Mã | Kết quả | Log |
|:--|:--|:--|:--|
| Script Codex | cũ `4937c85` | exit 1, 6/6 `true` | `03_codex_repro_ma_cu.*` |
| Script Codex | sửa | **exit 0, 6/6 `false`**; probe trùng train/dev: bị từ chối | `06_codex_repro_ma_moi.*` |
| Script tái dựng (Claude) | cũ / sửa | exit 1 (7 ca) / **exit 0** | `01_*`, `07_*` |
| `test_detector_5lop` mới | sửa (working tree, rồi tại HEAD `a86ed5c`) | **83/83** / **83/83** | `05_*`, `05b_*` |
| `test_detector_5lop` mới | cũ `4937c85` | 58/83, 25 FAIL | `04_test_moi_tren_ma_cu.log` |
| `run_all_tests --only test_detector_5lop test_bdd100k_adapter test_bdd100k_coco` | sửa | 3/3 PASS (83/83, 42/42, 11/11) | `tests_20260930-1109.log` |

**Phân loại 25 FAIL trên mã cũ**:

- **Hành vi sai thật** của ba finding:
  - F3: E7 ×3;
  - F1: G1, G2, G3;
  - F2: H1, H2, H3, H4, H5, H7, H8, H10, cùng H1b/H3b/H5b — evaluator cũ không xác minh, đọc ignore thiếu thành 0 vùng;
  - I2–I4: runner cũ không đối chiếu dữ liệu đã nạp.
- **Chỉ vì tính năng mới chưa có** ở mã cũ, không tự nó là lỗi: G0b, H6, I1, I5.
- **H9 không tính là bằng chứng**: mã cũ cũng từ chối `path` tương đối, nhưng vì không tìm thấy manifest theo đường dẫn đó.
- **Lượt đầu trên mã cũ** (`04a_*`): 57/83, vì D9 FAIL do **câu chữ** thông điệp. Mã cũ vẫn bắt được thiếu nhãn ở
  exporter. D9 đã được sửa để chấp nhận cả hai câu chữ.

**Lượt phát triển trên mã mới** — không phải log bằng chứng, ghi lại để minh bạch:

- Một lượt 81/83: hai FAIL do chính phép kiểm cắt thông điệp trước khi so chuỗi.
- Hai lượt Codex/test giữa chừng được chạy trên thư mục tạm, không lưu.

**Không chạy lại full suite**: các module sửa (`bdd_yolo_export`, `train_detector`, `eval_detector_5lop`) chỉ được import
bởi nhau và bởi `test_detector_5lop`, không nằm trên đường tracking, Nano hay cache đã nghiệm thu.

## 6. Smoke GPU với runner mới (fixture GPU, không phải dữ liệu thật)

Log: `logs/review28/fixture_gpu/`.

- **(a)** Runner mới kiểm dataset fixture v1 của vòng 27 → **exit 2** “manifest phiên bản 1 … xuất lại”.
- **(b)** Xuất lại từ cùng nguồn fixture và cùng split đã khóa của vòng 27 → dataset v2 (719 ảnh train / 90 ảnh dev).
  Kiểm trùng nội dung train/dev: 0 cặp. BDD val thật đã được băm, không trùng.
- **(c)** Smoke 1 epoch YOLOv8n và YOLO11n bằng cấu hình **cuối** (`train_bdd5.yaml` sha256 `a466a4fa…`, chỉ khác bản
  commit trước ở chú thích) → cả hai **ĐẠT**:

  | | YOLOv8n | YOLO11n |
  |:--|:--|:--|
  | Hợp đồng | v2, nội dung 809 ảnh / 809 nhãn / 28 ignore đã băm | như YOLOv8n |
  | Dữ liệu ultralytics nạp (callback trong trainer thật) | 719/90, thiếu 0, thừa 0, nhãn khác 0 | như YOLOv8n |
  | Thời gian | 117,3 s | 131,3 s |
  | Tốc độ | 0,95 s/batch | 1,14 s/batch |
  | PyTorch reserved | 2.130 MiB | 2.346 MiB |
  | `best.pt` sha256 | `f0b1aea4…` | `aaa163dd…` |

- `nvidia-smi` đo cả GPU; đỉnh 3.965 MiB ở lượt YOLO11n gồm cả các ứng dụng desktop khác, không quy được cho tiến trình
  train (PyTorch reserved không đổi).
- Lượt (c) đầu tiên, trước khi sửa chú thích cấu hình (`01c_*`), cũng ĐẠT.
- Số đo batch 16/12/workers của vòng 27 là **bằng chứng GPU cũ**, không chạy lại.

## 7. Đính chính diễn giải theo review

- **Motorcycle** (BDD val): **915 annotation = 902 box không crowd/ignore + 13 box crowd**, không có ignore. Đã ghi trong
  `BDD100K_INVENTORY.md` và phần đính chính của `REVIEW27_CHI_TIET.md`.
- **Kiểm trùng nội dung** — không còn phát biểu “đã kiểm trùng ở mọi split”:
  - BDD val: tên video, tên ảnh và sha256 nội dung;
  - UA-DETRAC test: **chỉ tên video**;
  - train-fit/dev: khác video và, từ vòng 28, khác sha256 ảnh nguồn. Trên dữ liệu train thật chưa biết có trùng hay không.
- **Batch 16**:
  - “tràn VRAM” được sửa thành: batch 16 **hoàn tất nhưng chậm 2–4 lần**, không có OOM;
  - PyTorch reserved sát hoặc vượt VRAM còn trống; paging là **giả thuyết chưa đo**.

  Sửa trong `DETECTOR_TRAIN_PLAN.md` mục 4 và chú thích `train_bdd5.yaml`.
- `PLAN_DA_CHOT.md` vẫn ghi BDD val và UA-DETRAC test đã được xem ở vòng trước, không gọi là test chưa từng quan sát.

## 8. Giới hạn còn lại

- Chưa có BDD100K MOT train; mọi kiểm là trên fixture tổng hợp.
- Chưa duyệt epoch, bước frame và ngân sách.
- Gốc tin của hash là `tom_tat.json`, không có ký số.
- Dataset không di chuyển được, vì `path` là tuyệt đối.
- Ngưỡng 0,9 lệch 1 ulp so với pycocotools.
- Hai lượt tái hiện “trước sửa” chạy trên bản sao `src/` + `configs/`, không phải checkout đầy đủ.

## 9. File và commit

- `73e051c`: review và tái hiện trước sửa (`CODEX_REVIEW_4937c85.md`, `REPRO_REVIEW_4937c85.{py,json}`,
  `logs/review28/00_*`, `01_*`, `02_*`, `03_*`, `tai_hien_r27.py`).
- `a86ed5c`: mã và test — `src/bdd_yolo_export.py`, `src/train_detector.py`, `src/eval_detector_5lop.py`,
  `src/test_detector_5lop.py`, và `configs/phase2/detector/train_bdd5.yaml` (chỉ chú thích).
- `dbfe48d`: bằng chứng sau sửa (`logs/review28/00b_*`, `04*`–`07*`, `tests_*`, `fixture_gpu/`).
- Commit tài liệu (commit này): `REVIEW28_CHI_TIET.md`, `REVIEW_BRIEF.md`, `CODEX_HANDOFF.md` (hàng 23, 24),
  `DETECTOR_TRAIN_PLAN.md`, `BDD100K_INVENTORY.md`, `REVIEW27_CHI_TIET.md` (đính chính), `PLAN_DA_CHOT.md` (trạng thái).
