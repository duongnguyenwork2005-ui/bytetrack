# Kiểm kê BDD100K-MOT trên máy — trạng thái và phần còn thiếu

Kiểm ngày **2026-09-18**, máy `D:\LABOTARY\bytetrack`, nhánh `phase2-mixed-traffic`.
(Bản trước, 2026-09-17, ghi "chưa có dữ liệu"; nay **đã có ảnh**, **chưa có nhãn** — xem bảng dưới.)
**Cập nhật 2026-09-29 (vòng 27)**: nhãn val đã có dưới dạng COCO-video. Split **train** (ảnh + nhãn) vẫn chưa có, nên
chưa chuẩn bị được dataset huấn luyện detector. Chi tiết ở mục ngay dưới.
**Cập nhật 2026-10-01 (vòng 31)**: **ảnh** MOT train đã tải và giải nén đủ. **Nhãn** MOT train `box_track_20/train` vẫn
chưa lấy được từ nguồn chính thức nào, nên `kiem-nguon` vẫn exit 2.
**Cập nhật 2026-10-01 (vòng 32)**: tìm được **ứng viên** nhãn Scalabel `box_track_20` (zip đóng gói lại trên Hugging Face).
Ứng viên **không kiểm được md5 chính thức** nên đang **cách ly**, chưa nhập vào dữ liệu train; chờ review.
**Cập nhật 2026-10-02 (vòng 33)**: review `c4d64a1` CHANGES REQUESTED (R32-F1, R32-F2), đã sửa. Lớp/ignore của ứng viên
**khác** COCO val của dự án; nhãn vẫn cách ly; phương án C chờ chốt.
**Cập nhật 2026-10-02 (vòng 34)**: review `4e71d32` **PASS WITH LIMITATIONS**. Người dùng **chọn phương án C**. Đã nhập
nguyên byte 1.400 JSON native train; `kiem-nguon` đường chính **exit 0**.
**Cập nhật 2026-10-02 (vòng 35)**: review `ccd801f` **PASS WITH LIMITATIONS**. **Đã khóa split** `bdd5_train_v1`, xuất
tập con smoke (334 / 84 ảnh), smoke 2 model trên dữ liệu thật **đạt**. Dataset đầy đủ chưa xuất; training dài chưa duyệt.
**Cập nhật 2026-10-03 (vòng 38)**: review `0c0a07a` **PASS WITH LIMITATIONS**. **Đã xuất dataset đầy đủ** bước 5:
50.487 / 5.559 ảnh, 8,03 GB, hợp đồng v2 0 lỗi. Training dài chưa duyệt.

## Vòng 38 (2026-10-02/03) — dataset YOLO đầy đủ đã xuất (bước 5)

Chi tiết: `REVIEW38_CHI_TIET.md`; log ở `logs/review38/`.

- **Nghiệm thu `0c0a07a`**: PASS WITH LIMITATIONS (đính chính vòng 36 + pilot vòng 37). Base review → `0c0a07a`.
- **Nguồn không đổi**:
  - ảnh MOT train và snapshot nhãn train phương án C (vân tay `d2024493…`);
  - split `bdd5_train_v1` `b8a2848e…`;
  - cấu hình `bdd5_train.yaml` `ca5d602b…`.
- **`kiem-nguon` exit 0** (1.400 video, 278.079 frame) trước khi xuất.
- **Dataset đầy đủ** `data/yolo_bdd5_v1/` (có `.gitignore` riêng, không commit):

  | Mục | Giá trị |
  |:--|:--|
  | Bước frame | 5 (chỉ số 0-based % 5 == 0, trong khoảng frame có nhãn) |
  | Ảnh train-fit / dev | **50.487 / 5.559** (1.261 / 139 video) |
  | Ảnh âm tính / có vùng ignore | 686 / 17.040 (train-fit); 95 / 1.560 (dev) |
  | Dev motorcycle / bicycle | 226 / 407 box; 205 / 334 ảnh; 24 / 44 video |
  | Hash | `data.yaml` `1280e57b…`, manifest `1ee6174f…`, `tom_tat.json` `9eae5eb3…` |
  | Dung lượng | 8.034.288.965 B, 130.699 file; D: còn 54,83 GB |

- **Kiểm**:
  - hợp đồng `bdd5-dataset-v2`: 0 lỗi, cả lúc xuất và chạy lại; 0 `.npy`;
  - tập frame khớp từng phần tử với tập tính lại từ nguồn nhãn;
  - trùng nội dung train/dev: 0; BDD val: 0 trùng tên + 0 trùng nội dung;
  - UA-DETRAC test: **chỉ so tên video**.
- **Giữ nguyên**:
  - dataset smoke `data/yolo_bdd5_v1_smoke/`;
  - COCO val `074ff795…`, native val cách ly, kết quả lịch sử.
- **Bước tiếp** (chưa chạy, chờ duyệt): training dài local.

## Vòng 35 (2026-10-02) — split đã khóa, tập con smoke đã xuất, smoke 2 model đạt

Chi tiết: `REVIEW35_CHI_TIET.md`; log ở `logs/review35/`.

- **Nghiệm thu `ccd801f`**: PASS WITH LIMITATIONS cho bước nhập nhãn train theo phương án C. Base review → `ccd801f`.
- **Split đã khóa** `configs/phase2/detector/splits/bdd5_train_v1.json` (commit `4237149`, trước mọi train/AP):

  | Mục | Giá trị |
  |:--|:--|
  | train-fit / dev | **1.261 / 139 video**; 250.566 / 27.513 frame; 0 video chung |
  | Cách chia | Theo video, dev 10%, seed 0, phân tầng motorcycle/bicycle |
  | Vân tay nguồn nhãn | `d2024493…`, khớp manifest nhập vòng 34 (phương án C) |
  | Hash | `sha256_danh_sach` `a59a8d06…`; file `b8a2848e…` (giữ nguyên byte CRLF trong git từ `aeed110`) |
  | Trùng tên video với BDD val (200) / UA-DETRAC test (40) | 0 / 0 |

- **Tập con smoke đã xuất**: `data/yolo_bdd5_v1_smoke/` (không commit), 55,9 MB, 997 file.
  - 16 + 4 video, bước 10: **334 / 84 ảnh**.
  - Hợp đồng `bdd5-dataset-v2`: 0 lỗi.
  - Trùng nội dung train/dev: 0. BDD val: trùng tên 0, trùng nội dung 0 (đã băm 39.973 ảnh val).
  - Ultralytics tạo `labels/train.cache` và `labels/dev.cache` trong dataset smoke khi train. Runner đối chiếu nhãn thực nạp
    với dataset đã xác minh: 0 lệch.
- **Dataset đầy đủ** (bước 5): **chưa xuất**.
  - Đếm bằng chính hàm của bộ xuất: **50.487 ảnh train-fit / 5.559 ảnh dev**, 6,85 GB ảnh nguồn.
  - Dung lượng ước ~8,5 GB, vì ảnh tô xám ghi lại JPEG q95 nặng ~1,55× nguồn.
- **Smoke 1 epoch** trên dữ liệu thật: **đạt cho YOLOv8n và YOLO11n** (lượt `*_smoke_lan2`). Lượt `*_smoke` hỏng do công cụ
  đo, được giữ làm bằng chứng. Thư mục lượt ở `models/finetune_phase2/` (không commit).
- **Giữ nguyên**: COCO val `074ff795…`, native val cách ly, kết quả BDD lịch sử. Nhãn vẫn là snapshot bên thứ ba, **chưa xác
  thực** bằng archive gốc.
- **Bước tiếp theo** (chưa chạy, chờ duyệt): epoch / bước frame / ngân sách → xuất dataset đầy đủ → training dài local.

## Vòng 34 (2026-10-02) — phương án C: đã nhập nhãn train (snapshot bên thứ ba có điều kiện)

Chi tiết: `REVIEW34_CHI_TIET.md`; log ở `logs/review34/`.

- **Quyết định**: người dùng chọn **phương án C**, tức dùng **có điều kiện** bản nhãn bên thứ ba đã ghim revision và hash. Đây
  **không** phải bản gốc đã xác thực.
- **Provenance**:

  | Mục | Giá trị |
  |:--|:--|
  | Nguồn | HF `jaffe03195/bdd100k_sot`, file `bdd100k.zip`, revision **`694197099692fe1254e93215e73e68ece8b1f41f`** |
  | Manifest nguồn (vòng 32) | `logs/review32/09_manifest_nhan_ung_vien.tsv`, sha256 `a2e416f2…` |
  | Manifest nhập | `logs/review34/03_manifest_nhap_nhan_train.tsv`, sha256 `c5a548bc…` |
  | Thời điểm nhập | 2026-10-02 11:40:29–11:40:53 (+07:00) |

- **Kiểm trước khi nhập**: bản cách ly có đúng 1.400 JSON train; 0 thiếu, 0 thừa, 0 lệch byte/sha256 so với manifest. Thư
  mục đích chưa tồn tại.
- **Nhập**: chép **nguyên byte** vào `data/raw/bdd100k/labels/box_track_20/train/<video>.json`, ghi mới 1.400 file.
- **Kiểm sau khi nhập**:
  - đích có 1.400/1.400 file khớp sha256, không có file lạ;
  - không có file provenance trong thư mục nhãn (provenance để ở handoff);
  - bản cách ly giữ nguyên: 1.602/1.602 file vẫn khớp manifest, gồm native val và hai COCO phụ;
  - COCO val `074ff795…` không đổi.
- **`kiem-nguon` đường chính**: **exit 0** — `scalabel`, 1.400 video nhãn = 1.400 video ảnh, 278.079 frame, 0 ảnh thiếu,
  1.400 video dùng được.
- **Ý nghĩa và giới hạn**:
  - `kiem-nguon` đạt chỉ có nghĩa là **dữ liệu đủ đầu vào để đi tiếp**;
  - đối chiếu native val với COCO vẫn có **183 lệch lớp và 50 ngoại lệ ignore**, nguyên nhân **UNKNOWN**;
  - không điều nào ở trên xác thực nhãn là bản phát hành chính thức; **chưa xác thực bằng archive gốc** (md5 `6be40e0c…`);
  - train chưa có tham chiếu độc lập theo từng annotation.
- **Giữ nguyên**: COCO val đang dùng, GT đã khóa, kết quả BDD lịch sử, native val cách ly và hai COCO phụ của mirror.
- **Nếu sau này có archive gốc**: đối chiếu đường dẫn và sha256 từng file với manifest, báo khác biệt **trước** khi thay
  snapshot đã dùng. Không âm thầm thay snapshot sau khi đã khóa split hoặc training.
- **Bước tiếp theo** (chưa chạy): khóa split, xuất dataset, smoke hai model trên dữ liệu thật theo kế hoạch đã chốt. Training
  dài chưa được duyệt.

## Vòng 33 (2026-10-02) — sửa R32-F1 / R32-F2; nhãn **vẫn cách ly**

Chi tiết: `REVIEW33_CHI_TIET.md`; log ở `logs/review33/`.

- **Nguồn**: HF `jaffe03195/bdd100k_sot`, revision **`694197099692fe1254e93215e73e68ece8b1f41f`**.
  - sha256 từng file ở `logs/review32/09_manifest_nhan_ung_vien.tsv` (manifest sha256 `a2e416f2…`).
  - Băm lại trước và sau vòng này: **1.602/1.602 khớp**.
  - **Chưa xác thực bằng archive gốc** (md5 `6be40e0c…`).
- **R32-F1** (chạy lại nguyên `audit_real_labels.py` của reviewer trên PC; kết quả trùng bản reviewer, CSV trùng từng byte):

  | Kiểm | Kết quả |
  |:--|:--|
  | Ghép theo (frame, `scalabel_id`) | 465.512 cặp |
  | Box | 0 lệch |
  | Song ánh ID hai chiều | 0 xung đột |
  | Lớp khác ngoài ba ánh xạ | **183** annotation (46 track, 40 video): truck→car 73, pedestrian→car 62, car→pedestrian 20, car→bus 14, … |
  | Ngoại lệ ignore | **50**: 24 other person→pedestrian ignore=0, 21 pedestrian→pedestrian ignore=1, 5 other vehicle→car ignore=0 |

  - Hai phép kiểm riêng, **không cộng**.
  - Nguyên nhân **UNKNOWN**: repo không có mã chuyển đổi, cũng không ghi nguồn của file COCO val.
  - **Quan sát**: COCO của chính người đóng gói HF trùng native ở cả 183 vị trí lệch lớp, và dùng ignore=0 cho mọi
    annotation. Điều này không cho biết bên nào đúng.
- **R32-F2**: `kiem_nhan_ung_vien.py` đã sửa (exit 0/1/2/3; trạng thái schema và COCO tách riêng).
  - Regression 30/30, trong khi bản cũ chỉ 3/30.
  - Script Codex: bản cũ exit 1, `bug_count=5`; bản sửa exit 0, `bug_count=0`.
  - Trên nhãn thật: schema và độ đầy đủ **DAT**, COCO **DIFFERENCES** (exit 3).
  - Dữ liệu thật có 0 NaN/Inf, 0 ID trùng trong frame, 0 sai kiểu thuộc tính.
- **`kiem-nguon`**: bản cách ly exit 0; đường chính **exit 2** vì nhãn chưa nhập.
- **Đề xuất cho từng bước, chờ người dùng chốt phương án C**:

  | Bước | Nhãn dùng |
  |:--|:--|
  | Huấn luyện / khóa split / smoke | snapshot native train đã ghim, nhập nguyên byte |
  | Đánh giá BDD val đã bàn giao | **giữ COCO val** `074ff795…` |
  | Kết quả BDD lịch sử | không đổi |

  Nếu dùng native val ở bước nào thì phải ghi rõ, vì lớp/ignore khác COCO.

## Vòng 32 (2026-10-01) — ứng viên nhãn MOT 2020 đang **cách ly**, chờ quyết định

Chi tiết: `logs/review32/14_nguon_nhan_vong32.md`. Ảnh train và val giữ nguyên: không tải lại, không giải nén lại, không băm
lại.

- **md5 chính thức** `6be40e0ca56a83ddeba2ed6bff50f9e6`: lấy từ dòng 268 của `doc/source/download.rst`, commit
  `9ac17c6c` (blob `ae399a3a`, đã tính lại và khớp).
- **Lượt tìm** từ 18:24 đến 18:52 (~28 phút, vượt mức ~20 phút):
  - Internet Archive trả **HTTP 429** cho mọi yêu cầu;
  - tìm web không ra mirror;
  - không thử lại các link chính thức đã chết.
- **Không tìm được bản sao kiểm được bằng md5.**
- **Ứng viên duy nhất**: `jaffe03195/bdd100k_sot` trên Hugging Face, file `bdd100k.zip` (5,35 GB, đóng gói lại, người tải lên
  ẩn danh, repo ghi license `apache-2.0` sai so với BDD100K).
  - Chỉ đọc mục lục và tải riêng `bdd100k/labels/box_track_20/` bằng HTTP Range: **196,5 MB**.
  - Được 1.602 file, **CRC32 khớp toàn bộ**: train 1.400, val 200, cộng 2 file COCO val của người đóng gói.
  - Cách ly ở `data/raw/ung_vien_nhan/hf_jaffe03195_bdd100k_sot/`; manifest sha256 ở `logs/review32/09_manifest_nhan_ung_vien.tsv`.

| Kiểm chứng | Kết quả |
|:--|:--|
| JSON và schema Scalabel | 0 lỗi |
| Train ↔ ảnh chính thức | 1.400/1.400 video, 278.079/278.079 frame (0 thiếu, 0 thừa) |
| Val ↔ file COCO-video val của dự án | 465.512/465.512 nhãn ghép theo (frame, `scalabel_id`); box khớp tuyệt đối; song ánh ID không xung đột. *Đính chính vòng 33*: **lớp/ignore chưa khớp**, gồm 183 lệch lớp (46 track, 40 video) và 50 ngoại lệ ignore; nguyên nhân UNKNOWN |
| Ảnh val trong zip HF ↔ zip val chính thức | 39.973/39.973 khớp CRC32 |
| Thống kê ↔ bài báo BDD100K (Bảng 11, §3.6) | 130.573 track (“130.6K”), 3,32M box (“3.3M”). *Đính chính vòng 33*: chỉ **tương đồng tổng hợp, có sai lệch**; không xác thực từng annotation |
| `kiem_nguon` của dự án trên bản cách ly | **exit 0** |
| `kiem-nguon` chính thức | **exit 2**: nhãn chưa nhập |

**Giới hạn**: không kiểm được md5; phần train không có tham chiếu từng nhãn độc lập; tài liệu không ghi nguồn gốc file COCO val
của dự án.

**Cần quyết định**:

- **A.** Nhập ứng viên sau review.
- **B.** Chờ gói gốc; thư nháp tiếng Anh ở `logs/review32/15_thu_nhap_xin_nhan_goc_EN.md`, chưa gửi.
- **C.** Tạm dùng ứng viên, khi có gói gốc thì so sha256 từng file với manifest.

## Vòng 31 (2026-09-30 → 2026-10-01) — ảnh MOT train **đã có**, nhãn MOT 2020 **chưa lấy được**

Log và script nằm ở `logs/review31/`.

### Ảnh train: đã tải, xác minh, giải nén (7/7)

- **Nguồn**: `http://dl.yf.io/bdd100k/mot20/`, dùng HTTP vì HTTPS lỗi chứng chỉ. Danh sách 7 phần `images20-track-train-1..7.zip`
  đọc từ listing thật (`00_dl_yf_io_mot20_listing.html`), không đoán.
- **Tải**: `tai_mot20_train.py`, có resume bằng `curl -C -` và khóa theo từng phần. Kiểm thử cục bộ 27/27 PASS. Mỗi file đã
  kiểm:
  - HTTP 200 và Content-Type `application/zip`;
  - kích thước bằng Content-Length;
  - 4 byte đầu là chữ ký ZIP;
  - **md5 khớp file `.md5` của máy chủ**;
  - `testzip` (CRC từng file) OK.

  Bảng URL / byte / md5 / sha256: `20_manifest_tai_anh_train.{md,json}`. Tổng **33.902.985.620 B**. Zip giữ nguyên ở
  `data/raw/downloads/bdd100k_mot20/`.
- **Giải nén**: `giai_nen_mot20_train.py`. Kiểm từng member trước khi ghi; không có lồng `bdd100k/bdd100k`, không có đường dẫn
  `..`, không trùng tên giữa các archive. Ghi qua file tạm rồi đổi tên; file đã có thì chỉ bỏ qua khi trùng từng byte, khác
  nội dung thì không ghi đè. Kết quả **0 xung đột**.
- **Kiểm kê độc lập** (`kiem_ke_train.py` → `21_kiem_ke_train.json`):

  | Mục | Kết quả |
  |:--|:--|
  | Ảnh train `data/raw/bdd100k/images/track/train/<video>/<video>-<7 số>.jpg` | **1.400 video, 278.079 ảnh**, 33,87 GB, 101–263 ảnh/video |
  | So với hợp member của 7 archive | 0 thiếu, 0 thừa, 0 lệch kích thước, 0 file tạm |
  | Trùng tên video train ↔ val | 0 |
  | Ảnh val | giữ nguyên: 200 video, 39.973 ảnh |
  | Ổ `D:` sau khi giải nén | còn trống 66,0 GB |

### Nhãn MOT 2020 `box_track_20/train`: chưa lấy được

Chi tiết các nguồn đã thử: `13_nguon_nhan_da_thu.md`.

- Tài liệu chính thức (nguồn trên GitHub) mô tả gói 115 MB, md5 `6be40e0ca56a83ddeba2ed6bff50f9e6`, nội dung
  `labels/box_track_20/{train,val}`.
- Không nguồn chính thức nào còn phục vụ gói này:
  - mirror ETH `dl.cv.ethz.ch`: **NXDOMAIN**;
  - nút “MOT 2020 Labels” trên trang tải Berkeley đã bị comment, URL của nút trả **404**;
  - `dl.yf.io` không có gói nhãn MOT;
  - GitHub issue #356/#369 vẫn mở, chưa có maintainer trả lời.
- Không dùng Detection 100K, MOTS, nhãn COCO chuyển đổi hay mirror không chính thức để thay.
- Nếu người dùng có file zip gốc: đặt vào `data/raw/downloads/bdd100k_mot20/`, kiểm md5 ở trên, rồi chỉ giải nén phần
  `train` vào `data/raw/`.

### `kiem-nguon`

`python src/prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml --buoc kiem-nguon` → **exit 2**
(`22_kiem_nguon.log`). Chỉ còn liệt kê nhãn `data/raw/bdd100k/labels/box_track_20/train`; ảnh train không còn bị báo thiếu
như vòng 30.

### Sự cố trong lúc tải (đã xử lý)

- **Tốc độ**: tuyến quốc tế sang Mỹ nghẽn, chỗ nghẽn không nằm ở đường truyền nhà (`14_chan_doan_toc_do_*.txt`).
  - Cao điểm: 2–20 KB/s.
  - Khoảng 01:00–05:00: 1,5–2,3 MB/s.
- **Giới hạn của công cụ**: tác vụ nền bị dừng sau khoảng 30 phút nếu không đặt `timeout`, và tối đa khoảng 2 giờ kể cả khi
  đặt dài hơn. Các tiến trình tải được khởi động lại theo yêu cầu của người dùng. `--don-khoa-cu` (commit `02e7347`) xóa khóa
  của tiến trình đã chết trước mỗi lần chạy lại.
- **Lỗi script cắt mất dữ liệu tốt** (`15_su_co_0346_cat_part.md`, sửa ở `8b142e2`):
  - khi `curl` bị giết, Content-Type rỗng bị coi là “không phải zip”, nên phần 2 bị cắt mất khoảng 1,2 GB đã tải;
  - sau khi sửa, kiểm thử hồi quy FAIL trên bản cũ và PASS trên bản mới;
  - md5 của cả 7 file đều khớp máy chủ.

## Vòng 30 (2026-09-30) — kiểm lại trước khi khóa split: MOT train **vẫn chưa có**

**Đã kiểm** (không quét toàn ổ đĩa):

- `data/raw/bdd100k` (cây thư mục), `data/raw`;
- `D:\LABOTARY` — nơi chứa gói val đã tải;
- `~/Downloads`, kể cả file đang tải dở (`.crdownload`, `.part`): 0 file.

Log: `logs/review30/01_kiem_ke_bdd100k.{json,log}`, `02_kiem_nguon_bi_chan.log`.

**Kết quả**:

- Không đổi so với vòng 27: chỉ có ảnh MOT **val** (200 video, 39.973 ảnh) và nhãn val COCO-video.
- **Không có** ảnh `images/track/train`, nhãn `labels/box_track_20/train` hay bản COCO-video train.
- `prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml --buoc kiem-nguon` → **exit 2**, liệt kê
  đúng hai đường dẫn thiếu.

**Còn thiếu, cần đặt đúng chỗ**. Không tự tải gói lớn; không thay bằng val hay fixture:

| Gói (MOT 2020, split `train`) | Đặt / giải nén sao cho có |
|:--|:--|
| Ảnh train: gói *MOT 2020 Images — train*, nhiều phần, cùng kiểu tên với `images20-track-val-1.zip` đã tải | `data/raw/bdd100k/images/track/train/<video>/<video>-<7 chữ số>.jpg`. Giải nén vào `data/raw/` vì archive có sẵn tiền tố `bdd100k/` |
| Nhãn train: gói nhãn box tracking `box_track_20` (Scalabel), khuyến nghị | `data/raw/bdd100k/labels/box_track_20/train/<video>.json` |
| *hoặc* bản COCO-video train | `data/raw/bdd100k/labels_cocovid/<…train…>_cocofmt.json`, rồi đổi `bdd.nhan` trong `bdd5_train.yaml` sang `cocovid`. Cần nhiều RAM |

- Ổ `D:` còn **126 GB** trống. Ước tính ảnh train ~35 GB zip + ~34 GB sau giải nén, và dataset YOLO ~7 GB.
- Giữ nguyên bản tải về gốc.
- Sau khi đặt xong, chạy lại `--buoc kiem-nguon`; phải được exit 0.

## Vòng 27 (2026-09-29) — kiểm kê lại cho huấn luyện detector 5 lớp

Lệnh kiểm: `python src/kiem_ke_bdd100k.py --out handoff/logs/review27/01_kiem_ke_bdd100k.json --bam-archive`.
Script chỉ đọc (không giải nén, sửa hay xóa gì); log ở `handoff/logs/review27/01_kiem_ke_bdd100k.{json,log}`.

### Trên máy có gì

| Thành phần | Đường dẫn | Trạng thái |
|:--|:--|:--|
| Archive ảnh MOT 2020 val (phần 1) | `D:\LABOTARY\images20-track-val-1.zip` | có — 4.983.938.716 B, sha256 `4d678810a14095ab…`; 40.174 mục, gồm 39.973 ảnh của 200 video, **chỉ** `images/track/val` |
| Ảnh MOT val đã giải nén | `data/raw/bdd100k/images/track/val/<video>/` | có — 200 video, 39.973 ảnh (4,8 GB); header **toàn bộ 1280×720**, khớp kích thước khai báo trong nhãn |
| Nhãn MOT val (bản chuyển đổi **COCO-video**) | `data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json` | có — 117.094.032 B, sha256 `074ff795…`; cùng byte với bản ngoài repo `D:\LABOTARY\01-bdd_box_track_val_cocofmt.json` |
| Ảnh MOT **train** | `data/raw/bdd100k/images/track/train/` | ❌ **không có** |
| Nhãn MOT **train** (Scalabel `box_track_20` hoặc COCO-video) | `data/raw/bdd100k/labels/box_track_20/train/` hoặc `labels_cocovid/*train*` | ❌ **không có** |
| Nhãn MOT val gốc Scalabel | `data/raw/bdd100k/labels/box_track_20/val/` | không có (bản COCO-video ở trên thay thế cho đánh giá) |
| Gói detection 100K (`images/100k`, `labels/det_20`) | — | không có |
| Thư mục `datasets/BDD100K` của dự án cũ | `D:\LABOTARY\ByteTrack + EKF + Kitty\datasets\BDD100K` | rỗng (0 file) |

Kết luận của script: `du_de_chuan_bi_train = false`.

### Thống kê nhãn MOT val (nguồn đánh giá đã định danh — không dùng làm train/dev)

Tổng cộng: 200 video, 39.973 ảnh, 465.512 annotation. Mỗi video có 102–204 frame, trung bình 199,9.
Có 615 ảnh không có annotation nào: 488 ảnh nằm trong khoảng có nhãn của video, 127 ảnh ở đuôi video sau frame có nhãn cuối.

| Lớp | box (không crowd/ignore) | crowd | ignore | ảnh có lớp | video có lớp |
|:--|--:|--:|--:|--:|--:|
| car | 341.436 | 20.079 | 2.825 | 38.964 | 200 |
| truck | 27.343 | 481 | 424 | 15.862 | 160 |
| bus | 9.087 | 4 | 0 | 7.018 | 93 |
| **motorcycle** | **902** | 13 | 0 | **774** | **29** |
| bicycle | 4.135 | 271 | 0 | 2.738 | 63 |
| pedestrian *(ngoài 5 lớp)* | 57.065 | 1.842 | 816 | 14.792 | 152 |
| rider *(ngoài 5 lớp)* | 2.546 | 0 | 0 | 2.260 | 69 |
| train *(ngoài 5 lớp)* | 308 | 0 | 0 | 308 | 6 |

- **Cách đọc cột**:
  - các cột box, crowd và ignore tính theo annotation, và một annotation có thể vừa crowd vừa ignore;
  - cột “ảnh có lớp” và “video có lớp” chỉ đếm box không crowd/ignore.

  Ví dụ motorcycle (đính chính vòng 28): tổng **915 annotation = 902 box không crowd/ignore + 13 box crowd**, không có box
  ignore. 902 là số box được gán nhãn huấn luyện/đánh giá, không phải tổng số annotation motorcycle.
- Cờ `ignore` chỉ xuất hiện ở car, truck và pedestrian. Đây là các distractor `other vehicle`, `trailer`, `other person`
  đã được ánh xạ sang lớp đánh giá khi chuyển sang COCO-video.
- **Ảnh trùng nội dung**: 2 frame liên tiếp của video val `b1d7b3ac-5744370e` (`-0000079.jpg`, `-0000080.jpg`) giống
  nhau từng byte, nên toàn tập có 39.973 ảnh nhưng chỉ 39.972 hash khác nhau. Đây là đặc điểm của dữ liệu, không phải
  ảnh thiếu.

### MOT và detection là hai nguồn khác nhau, không thay thế trực tiếp

- **MOT 2020**: ảnh 5 fps theo video, có ID đối tượng theo thời gian. Split `train` là nguồn của train-fit và dev nội
  bộ trong plan.
- **Detection 100K**: mỗi ảnh là một khung hình tĩnh lấy từ một video, nhãn `det_20`, không có ID theo thời gian.
- Pipeline tracking và `configs/phase2/detector/bdd5_train.yaml` chỉ nhận MOT train. Muốn dùng gói detection (thay
  thế hoặc bổ sung) phải hỏi người dùng trước.

### Gói còn thiếu và đường dẫn cần đặt

Không dùng UA-DETRAC hay BDD val để thay thế.

1. **Ảnh MOT 2020 train**: gói *MOT 2020 Images — train* trên trang tải BDD100K chính thức.
   - Tên gói: gói val đã tải tên `images20-track-val-1.zip`; gói train được chia nhiều phần theo cùng kiểu tên. Máy không
     truy cập được trang tải lúc kiểm, nên cần xem lại tên và số phần trên trang.
   - Giải nén vào `data/raw/`, **không** vào `data/raw/bdd100k/`, vì archive đã có sẵn tiền tố `bdd100k/`.
   - Kết quả cần có: `data/raw/bdd100k/images/track/train/<video>/<video>-<7 chữ số>.jpg`.
2. **Nhãn MOT 2020 train, bản Scalabel (khuyến nghị)**: gói nhãn box tracking `box_track_20`.
   - Kết quả cần có: `data/raw/bdd100k/labels/box_track_20/train/<video>.json`, mỗi video một file.
   - Nếu chỉ có bản COCO-video (`*train*cocofmt.json`): đặt vào `data/raw/bdd100k/labels_cocovid/`, rồi đổi `bdd.nhan`
     trong `bdd5_train.yaml` sang `dinh_dang: cocovid`. Bản này đọc cả file vào RAM: file val 117 MB cho 200 video, bản
     train sẽ lớn hơn nhiều, trong khi máy có 7,8 GB RAM.
3. **Ước tính dung lượng**, suy từ val (5 GB zip ≈ 4,8 GB ảnh cho 200 video):
   - ảnh train khoảng 7 lần val, tức ~35 GB zip và ~34 GB sau giải nén;
   - dataset YOLO xuất ở bước 5 frame thêm khoảng 7 GB.

   Ổ `D:` còn ~125 GB trống. Số video thật chưa biết: `kiem-nguon` sẽ in số thật, không dùng con số ước tính.
4. **Kiểm sau khi đặt xong**: `python src/prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml --buoc kiem-nguon`
   phải exit 0. Lệnh này đối chiếu nhãn với ảnh theo **từng video**. Thiếu gì thì trả exit 2 và in đúng đường dẫn thiếu.

---

*Các mục 1–7 dưới đây là kiểm kê ngày 2026-09-18 (giữ nguyên để đối chiếu).*

## 1. Đã tải những gì — kết quả kiểm tra thực tế

**Archive đã tải**: `D:\LABOTARY\images20-track-val-1.zip` — 4,75 GB, 40.174 entry, thư mục gốc bên trong là
`bdd100k/images/…`. Tên gói cho biết đây là **MOT 2020 → Images → track → val → part 1**.

**Đã giải nén đúng vị trí** (không lồng nhầm `bdd100k/bdd100k`):

| Thành phần | Đường dẫn pipeline yêu cầu | Trạng thái |
|:--|:--|:--|
| **Ảnh MOT, split `val`** | `data/raw/bdd100k/images/track/val/<video>/<video>-<7 chữ số>.jpg` | ✅ **200 video, 39.973 ảnh**, đặt tên đúng quy ước adapter |
| **Ảnh MOT, split `train`** | `data/raw/bdd100k/images/track/train/…` | ❌ **không có** |
| **Nhãn tracking `val`** | `data/raw/bdd100k/labels/box_track_20/val/<video>.json` | ❌ **không có** — thư mục `labels/` không tồn tại, **0 file JSON** trong toàn bộ `data/raw/bdd100k` |
| **Nhãn tracking `train`** | `data/raw/bdd100k/labels/box_track_20/train/<video>.json` | ❌ **không có** |

> **"Tải xong dataset" không có nghĩa là đủ cả hai.** Gói đã tải là **Images**; gói **Labels** là một file
> tải **riêng**. Ảnh chiếm 4,75 GB, nhãn chỉ khoảng **50 MB**.

## 2. Hệ quả: chưa chạy được gì trên dữ liệu BDD thật

Adapter lấy **danh sách video, danh sách frame, tên file ảnh và `timestamp` TỪ file nhãn JSON**, không phải
từ thư mục ảnh ([bdd100k_mot.py:221–243](../src/mot_datasets/bdd100k_mot.py)):

```python
def list_videos(self, part):          # glob thu muc LABELS
    d = self.labels_dir(part)
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []

def frames(self, video, part="val"):  # doc JSON nhan -> frameIndex, name, timestamp
    fr = self.load_json(video, part)
```

Vì vậy **thiếu nhãn là chặn cứng** cho cả ba bước, kể cả bước chỉ cần ảnh:

| Lệnh | Kết quả hiện tại |
|:--|:--|
| `python src/check_bdd100k.py --part val` | **exit 1** — fail-loud, in đúng đường dẫn nhãn còn thiếu |
| `python src/det_cache.py --dataset bdd100k …` | không chạy được: cần `ds.frames()` → cần JSON nhãn |
| evaluator (TrackEval) | không chạy được: không có GT |

**Không** thay bằng nhãn detection (`labels/det_20`, `bdd100k_labels_images_*`) hay bằng fixture
`data/fixtures/bdd100k_mini` — fixture là dữ liệu **tổng hợp**, chỉ để kiểm tra pipeline (README của nó ghi
rõ), không dùng để báo cáo kết quả thực nghiệm.

## 3. Cần tải thêm gì — nguồn, vị trí, lệnh kiểm tra

Đăng ký tại **bdd-data.berkeley.edu** (nguồn chính thức, BAIR/Berkeley DeepDrive), phần **MOT 2020**:

| Gói | Bắt buộc? | Giải nén sao cho có |
|:--|:--|:--|
| **MOT 2020 Labels** | ✅ **bắt buộc** (~50 MB) | `data/raw/bdd100k/labels/box_track_20/{train,val}/<video>.json` |
| MOT 2020 Images — `train` | chỉ khi làm **Thí nghiệm 2** (tuning) | `data/raw/bdd100k/images/track/train/<video>/…jpg` |
| MOT 2020 Images — `val` | ✅ **đã có** | — |

Gói `test` **không cần**: BDD không công bố nhãn test.

Giải nén cẩn thận: archive chứa sẵn tiền tố `bdd100k/`, nên giải nén vào `data/raw/` (**không** vào
`data/raw/bdd100k/`, sẽ thành `data/raw/bdd100k/bdd100k/…`).

Kiểm tra theo đúng thứ tự — **mỗi bước phải exit 0 trước khi sang bước sau**:

```bash
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe

# 1. Schema + ảnh + thống kê lớp + phân bố Δt, đồng thời sinh manifest final (val)
$P src/check_bdd100k.py --part val --make-manifest
#    -> results/phase2/bdd100k_check/val_{class_stats,dt_stats}.csv, val_schema_report.json
#    -> results/phase2/manifests/bdd100k_final_val.json
#    phải in "[schema] N/N video hop le; 0 loi" và "[images] thieu 0 anh"

# 2. Manifest train-dev (~40 video, seed 0) - CHỈ dùng để tuning, KHÔNG dùng đánh giá
#    (chỉ chạy được khi đã có nhãn + ảnh của split `train`)
$P src/check_bdd100k.py --part train --make-manifest --n-dev 40 --seed 0
#    -> results/phase2/manifests/bdd100k_dev.json

# 3. Cache detection riêng cho BDD (CẦN GPU)
$P src/det_cache.py --dataset bdd100k --part val \
   --manifest results/phase2/manifests/bdd100k_smoke_val.json \
   --cache-id yolov8n-coco_conf0.05_smoke_bdd

# 4. Smoke 1-3 sequence bằng YAML + output RIÊNG (xem mục 4)
$P src/run_experiment.py --exp configs/phase2/experiments/bdd_smoke_val.yaml
```

> ⚠️ `run_experiment.py` **không có tham số `--videos`** (chỉ có `--exp/--stage/--dry-run/--no-skip/--force/
> --force-eval`). Muốn chạy trên một tập con thì phải **tạo manifest riêng + YAML riêng**, như mục 4 — bản
> trước của tài liệu này ghi `--videos <v1> <v2> <v3>` là **sai**, lệnh đó sẽ lỗi.

Tên manifest train-dev do `check_bdd100k.py` sinh ra thực tế là **`bdd100k_dev.json`** (không phải
`bdd100k_dev_train.json` như bản trước viết).

## 4. Smoke BDD — đã chuẩn bị sẵn, chạy được ngay khi có nhãn

Tách hoàn toàn khỏi đánh giá đầy đủ, **tên riêng ở cả bốn chỗ**:

| | giá trị |
|:--|:--|
| Manifest | `results/phase2/manifests/bdd100k_smoke_val.json` (1–3 sequence, **chọn và ghi ra file TRƯỚC khi xem metric**) |
| YAML | [`configs/phase2/experiments/bdd_smoke_val.yaml`](../configs/phase2/experiments/bdd_smoke_val.yaml) |
| `name` / thư mục kết quả | `bdd_smoke_val` → `results/phase2/bdd_smoke_val/`, `data/processed/trackers/BDD-val/p2-bdd_smoke_val-*` |
| `cache_id` | `yolov8n-coco_conf0.05_smoke_bdd` |

⇒ không ghi đè `bdd100k_final_val.json`, `bdd_transfer_val`, hay cache đánh giá đầy đủ.

Smoke dùng **CV và EKF** với cấu hình đã khóa, kiểm toàn đường chạy cache → tracking → evaluator → bảng tổng
hợp. **Không** chọn tham số hay bộ lọc từ metric của smoke.

## 5. Ánh xạ lớp, thuộc tính và vùng bỏ qua — đã định nghĩa, chờ dữ liệu để xác nhận

| Hạng mục | Quy ước đã chốt |
|:--|:--|
| Lớp đánh giá | 8 lớp BDD: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle |
| Ánh xạ detector | COCO → BDD: `1→bicycle, 2→car, 3→motorcycle, 5→bus, 7→truck`. **`person`(0) KHÔNG map** vì COCO gộp pedestrian + rider |
| Mục tiêu chính | `vehicle` = car + truck + bus + motorcycle + bicycle → class 1 (liên tục với UA-DETRAC) |
| Vùng bỏ qua | mọi box `crowd` (bất kỳ lớp), 3 lớp distractor (*other person, trailer, other vehicle*), và `train` khi target = vehicle → class 13, conf 0 |
| `occluded` | **cờ NHỊ PHÂN** "có bị che một phần hay không". BDD **không** ghi mức độ che. Interim gán `occlusion_ratio = 0,5` — đây là **GIÁ TRỊ QUY ƯỚC cho một cờ nhị phân, KHÔNG phải mức che 50 % đo được**: 0,5 được chọn vì vượt ngưỡng partial (0,10) nhưng **không bao giờ** vượt ngưỡng full (0,90), tức không bao giờ biến một cờ nhị phân thành sự kiện "che gần hoàn toàn". `visibility = 1 − occlusion_ratio` mang **cùng tính chất quy ước**. **Không so trực tiếp** với `occlusion_ratio` **đo được** của UA-DETRAC |
| Khoảng trống annotation (**gap**) | **KHÔNG** suy diễn thành che hoàn toàn. `occlusion_eval --gaps` tạo sự kiện gap với `has_full=False` và `gap_evidence ∈ {occluded_adjacent, unknown}`, báo cáo **tách tầng** |
| Thời gian | 5 fps danh nghĩa; Δt thực lấy từ `timestamp` (ms) khi **mọi** frame của video có trường đó, **ngược lại** suy ra theo `frameIndex / 5` — cột `dt_source` trong interim ghi rõ từng video dùng nguồn nào; bộ lọc vẫn ở **đơn vị bước** (dt = 1); `track_buffer_sec` 1,2 s = 6 bước @ 5 fps |
| `rider` | box **riêng** với xe hai bánh, không gộp; chỉ thống kê, **không** đánh giá (detector COCO không có lớp này) |

⚠️ **Chưa xác nhận trên dữ liệu thật**: tỉ lệ video có `timestamp` ở mọi frame, số video có khoảng lệch nhịp,
thống kê lớp thật (nhất là `motorcycle`/`bicycle` — phần liên quan nhất tới giao thông hỗn hợp), và số vùng
`crowd` thật. `check_bdd100k.py` in đúng các con số này ở bước 1 (`dt_source`, `n_intervals_irregular`,
`val_class_stats.csv`).

## 6. Kế hoạch đánh giá — **hai thí nghiệm tách biệt, không gộp kết luận**

### A. Chuyển cấu hình UA-DETRAC → BDD100K (`bdd_transfer_val.yaml`)
Giữ **nguyên** tham số nghiên cứu đã khóa trên UA-DETRAC, xử lý thời gian theo quy ước pipeline.
**Không tuning bằng nhãn BDD.** Ghi rõ khi báo cáo: `std_theta`/`std_omega` là **theo bước**, mà một bước ở
BDD dài gấp 5 lần ở DETRAC (0,2 s so với 0,04 s) — thí nghiệm này **cố ý** giữ nguyên giá trị theo bước,
đúng nghĩa "chuyển cấu hình"; **không** quy đổi bằng công thức.

### B. Tuning trên BDD train-dev rồi đánh giá trên val (`bdd_tuned_val.yaml`)
**Hiện chưa đủ điều kiện**: chỉ có ảnh `val`, không có `train`. Theo nguyên tắc đã chốt, **train-dev bắt buộc
lấy từ split `train`** — không được cắt val ra làm tập tuning.

Khi có `train`: công bố **trước** khi chạy — tập phát triển (~40 video train, seed 0,
`bdd100k_dev.json`), phạm vi tham số (`track_buffer_sec` ∈ {0,4; 0,8; 1,2; 2,0} và lưới C3 với
`std_theta`/`std_omega` nhân hệ số {1, 2, 5}), ngân sách tìm kiếm, tiêu chí chọn (HOTA ở c25 + CI, giống
`PREREG_moc2.md`). **Khóa** lựa chọn trong `LOCK_moc2.md` **trước** khi chạy val.

**Không trùng sequence**: hai tập lấy từ **hai split khác nhau** của BDD (`train` vs `val`) ⇒ giao rỗng theo
cấu tạo. Vẫn kiểm lại bằng một phép giao trên hai manifest trước khi chạy đánh giá.

Hai kết quả báo cáo **riêng**; **không** gộp thành một kết luận "khả năng chuyển miền".

## 7. Việc còn lại theo thứ tự

1. **Tải gói MOT 2020 Labels** → `data/raw/bdd100k/labels/box_track_20/val/*.json` (bắt buộc).
2. `check_bdd100k.py --part val --make-manifest` phải exit 0, 0 lỗi schema, 0 ảnh thiếu.
3. Đối chiếu thống kê lớp và phân bố Δt thật với quy ước ở mục 5.
4. Chọn 1–3 sequence, **ghi ra `bdd100k_smoke_val.json` trước khi xem metric**; tạo cache
   `yolov8n-coco_conf0.05_smoke_bdd` (cần GPU — đi qua guard provenance checkpoint đã thêm ở review vòng 3).
5. Chạy `bdd_smoke_val.yaml` (CV + EKF), kiểm 5 điểm ở mục 5, rồi mới bàn tới Thí nghiệm 1.
6. Chỉ khi có split `train` mới mở Thí nghiệm 2.
