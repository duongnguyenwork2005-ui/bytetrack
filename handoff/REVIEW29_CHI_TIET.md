# Vòng 29 — sửa R28-F1 (cache ảnh `.npy`) của review `07615ba` (chi tiết)

Review: `handoff/CODEX_REVIEW_07615ba.md` — REQUEST CHANGES, một finding P2. R27-F1/F2/F3 đã được reviewer xác nhận hết lỗi.
**Base giữ `239b9c6`**. Trạng thái: **đã sửa, chờ Codex nghiệm thu**.
Phạm vi: chỉ sửa R28-F1, kiểm phần bị ảnh hưởng, tài liệu và delta. Không training dài, không cloud, không thêm dataset hay
model; không đụng Nano v16.

Nhãn nguồn bằng chứng:

- **chạy thật**: chạy trên PC, với `YOLODataset` / runner / exporter thật và dữ liệu tổng hợp;
- **fixture GPU**: training 1 epoch thật trên GTX 1650 với dữ liệu tổng hợp;
- **giả lập**: spy tại ranh giới nạp model/training, ghi rõ ở từng chỗ.

## 0. Trạng thái đầu vòng và file review

- HEAD `07615ba` (= target review), 0 commit sau đó. Working tree chỉ có file riêng của người dùng. Môi trường: Python
  3.11.9, ultralytics 8.4.141, torch 2.6.0+cu124, OpenCV 5.0.0, pycocotools 2.0.11 (`logs/review29/00_trang_thai_dau.log`).
- **Đã có**: `CODEX_REVIEW_07615ba.md`, lưu vào `handoff/` (8.395 B, sha256 `73277f68…`, cùng byte với `~/Downloads`).
- **Chưa có trên máy**, tìm ở Downloads, `C:\Users` và `D:` lúc đầu vòng và kiểm lại hai lần trong vòng:
  `CODEX_REVIEW_07615ba_EVIDENCE.zip`, `REPRO_REVIEW_07615ba.py` và JSON kèm theo. Vì vậy **script Codex chưa chạy được
  (NOT_RUN)**; mục 3 dùng script tái dựng thay thế và ghi rõ như vậy.
- **Bản chép vòng 28**: ba file review vòng 28 được chép từ tài liệu đính kèm; bản gốc nay đã có trong `~/Downloads`. Cả ba
  **trùng từng byte**.

## 1. Nguyên nhân

- **Hành vi của ultralytics 8.4.141** (`data/base.py`, `BaseDataset.load_image`, khoảng dòng 247–262):
  - `npy_files` là đường dẫn ảnh đổi đuôi thành `.npy`, cùng thư mục và cùng tên;
  - khi ảnh chưa có trong RAM, nếu `.npy` tồn tại thì `np.load` được dùng **thay** JPEG, **bất kể** `self.cache` —
    `cache=False` cho `self.cache = None`;
  - `.npy` chỉ bị bỏ khi hỏng hoặc sai số kênh;
  - chế độ `cache="disk"` chỉ tạo `.npy` khi file **chưa có**, nên cũng dùng lại `.npy` cũ.
- **Hệ quả với chốt vòng 28**:
  - hợp đồng liệt kê ảnh theo `IMG_FORMATS` và băm JPEG, nhưng không nhìn `.npy`;
  - callback so đường dẫn, lớp và bbox nhưng không so nguồn pixel.

  Một `.npy` sai cạnh ảnh đã khóa lọt qua cả hai lớp.
- **Khác với cache nhãn**: `labels/<split>.cache` đã được xử lý ở vòng 28 bằng callback đối chiếu nhãn.

## 2. Sửa (commit `83482c2`)

**Chính sách**: `cache: false`, **không dùng cache ảnh**; pixel nạp phải là JPEG đã khóa sha256 trong manifest. Không đổi
giá trị cấu hình đã khóa, công thức AP, ngưỡng IoU, split hay các chốt vòng 28.

- **`src/bdd_yolo_export.py`** — `kiem_hop_dong_dataset()`, bước 5 “cache ảnh”:
  - với từng ảnh ultralytics sẽ nạp (liệt kê như trainer), có `<ảnh>.npy` thì báo lỗi, nêu split, đường dẫn (tối đa 3 ví
    dụ và tổng số) và lý do (“ultralytics `load_image()` nạp `.npy` THAY ảnh JPEG kể cả khi cache=False …”);
  - chỉ kiểm `.npy` cạnh ảnh của **dataset đang dùng**, không cấm `.npy` toàn repo, **không tự xóa**;
  - danh tính dataset thêm `cache_anh` (chính sách + số `.npy` cạnh ảnh theo split).

  Hợp đồng dùng chung nên cả exporter (`--buoc xuat` / `--buoc kiem`), runner và evaluator đều chặn.
- **`src/train_detector.py`**:
  - `doc_cfg()` từ chối `huan_luyen.cache` khác `false` (exit 2), vì `ram`/`disk`/`True` là chế độ hợp đồng chưa kiểm
    chứng nội dung;
  - `kiem_dau_vao_da_nap()` — callback trước vòng train, không nuốt lỗi — thêm kiểm nguồn pixel của dataset
    **ultralytics thật**: `ds.cache` phải là `None`, và không có file nào trong `ds.npy_files`. Phép này chặn `.npy`
    xuất hiện **sau** khi hợp đồng đã chạy;
  - kết quả ghi `cache_anh` và `npy_canh_anh` cho từng split.
- **`src/test_detector_5lop.py`**: thêm nhóm J (mục 3). Sửa cách đếm: thiếu weights thì A7, E6, F là **SKIP**, không tính
  PASS. Trước đây `F1..F5 bo qua` bị đếm PASS, còn A7/E6 bị bỏ qua mà không ghi gì.

Lỗi được trả rõ ràng: runner exit 2, `[LOI CAU HINH] dataset KHONG dat hop dong: … images/train: 1 file cache anh .npy canh anh
da khoa (vd ['images/train/…npy']) -- ultralytics load_image() nap .npy THAY anh JPEG ke ca khi cache=False …`.

## 3. Kiểm chứng

| Lượt (lệnh) | Mã | Kết quả | Loại | Log |
|:--|:--|:--|:--|:--|
| `python handoff/logs/review29/tai_hien_r28_f1.py . --out …` | chưa sửa `07615ba` | **exit 1, bug_count=2**. JPEG TB 90,4031 (train) / 89,6897 (dev), trùng số Codex; pixel nạp TB 0,0; hash JPEG đúng manifest; chốt runner và callback đều cho qua. Đối chứng sạch: pixel trùng JPEG ở mọi ảnh | chạy thật (`YOLODataset` thật) | `01_tai_hien_claude_ma_cu.*` |
| cùng lệnh | HEAD `83482c2` | **exit 0, bug_count=0**. Chốt runner và callback từ chối cả hai ca; đối chứng sạch vẫn đạt | chạy thật | `05_tai_hien_claude_ma_sua.*` |
| `python REPRO_REVIEW_07615ba.py . --out …` (Codex) | — | **NOT_RUN** — file không có trên máy | — | — |
| `python src/test_detector_5lop.py` | HEAD `83482c2` | **92/92 PASS**, 0 SKIP (có weights) | chạy thật + spy ở nhóm G, J3/J4/J6 | `04_test_tai_HEAD.*` |
| cùng test, chạy trên bản sao mã chưa sửa (`DET5_MODELS_DIR=…`) | `07615ba` | 84/92. **8 FAIL**: J1, J2, J3, J4, J5a, J5b, J6 = chốt cache ảnh và chính sách cache; J0b = khóa danh tính mới. Đối chứng J0 và toàn bộ nhóm A–I **PASS** | chạy thật | `03_test_moi_tren_ma_cu.*` |
| `python src/run_all_tests.py --only test_detector_5lop` | HEAD | PASS 1, FAIL 0, SKIPPED 0, NOT_RUN 36 | — | `tests_20260930-1232.log` |
| runner **thật**, không spy, trên fixture v2 | HEAD | `.npy` cạnh 1 ảnh train → **exit 2**; `.npy` cạnh 1 ảnh dev → **exit 2**. Không tạo thư mục lượt hay file kết quả. Dataset sạch → smoke YOLOv8n 1 epoch **ĐẠT**; callback: 719/90 khớp, `cache_anh: None`, `npy_canh_anh: 0` | fixture GPU | `fixture_gpu/01_*` |

**Nhóm J** (`test_detector_5lop`):

| Phép | Nội dung | Loại |
|:--|:--|:--|
| J0 | Dataset sạch: hợp đồng chấp nhận; `YOLODataset(cache=False, augment=False)` thật đọc pixel **trùng từng byte** với `cv2.imread(JPEG)` ở 125/125 ảnh train + dev, trước augmentation | chạy thật |
| J0b | Danh tính ghi chính sách cache và 0 `.npy` | chạy thật |
| J1 / J2 | `.npy` ảnh đen cạnh ảnh train / dev: ultralytics **thật** nạp ảnh đen (xác nhận nguy cơ); hợp đồng exporter và chốt runner từ chối, nêu split, đường dẫn, lý do | chạy thật |
| J3 / J4 | Qua entry point `train_detector.main()`: exit 2; spy **không** được khởi tạo, không gọi train; không có thư mục lượt hay file kết quả | **spy giả lập** ranh giới model/training |
| J5a / J5b | Callback với `YOLODataset` **thật** dựng khi còn sạch, sau đó `.npy` xuất hiện → `LoiDauVaoDaNap` | chạy thật |
| J6 | Cấu hình `cache: disk` → runner exit 2 trước khi nạp model | spy |

**Các regression cũ vẫn giữ và đạt**: G (YAML train trỏ dev; thiếu nhãn có box / nhãn rỗng), H (hash ảnh, nhãn, ignore;
manifest v1; …), I (cache nhãn cũ với `YOLODataset` thật), E7–E9 (pycocotools thật).

**Suite khác**: không chạy — `test_bdd100k_adapter` và `test_bdd100k_coco` không bị ảnh hưởng (NOT_RUN). Lý do không chạy
full suite: chỉ ba module của đường detector bị sửa, không nằm trên đường tracking/Nano đã nghiệm thu.

## 4. Giới hạn còn lại

- **Chưa chạy script Codex** `REPRO_REVIEW_07615ba.py` vì file chưa có trên máy. Khi có, chạy:
  `python handoff/REPRO_REVIEW_07615ba.py . --out handoff/logs/review29/repro_codex.json`. Kỳ vọng: exit 0, `bug_count=0`,
  đối chứng sạch đạt.
- **Phạm vi chốt**: chỉ kiểm `.npy` đúng tên cạnh ảnh train/dev mà trainer sẽ nạp; `.npy` ở nơi khác không ảnh hưởng
  loader nên không bị cấm. Cache RAM hay disk không được hỗ trợ: runner từ chối cấu hình.
- **Không băm lại pixel mỗi lần nạp**: bảo đảm dựa trên hash JPEG (hợp đồng), không có `.npy`, và `cache=None`. J0 chứng
  minh pixel nạp trùng JPEG trên fixture.
- **Nạp trước khi callback chạy**: dataloader worker có thể đọc trước vài batch trước `on_pretrain_routine_end`. Nếu khi đó
  có `.npy`, callback vẫn dừng trước vòng train, nên không có bước tối ưu nào.
- **Evaluator** cũng bị hợp đồng chặn khi có `.npy`, dù `predict` đọc JPEG. Đây là chính sách thống nhất; nếu cần tách
  riêng thì phải hỏi.
- Chưa có BDD100K MOT train; mọi kiểm là trên fixture. Epoch và ngân sách chưa duyệt.

## 5. File và commit

- `cf9350a`: trước sửa — `CODEX_REVIEW_07615ba.md`, `logs/review29/00_*`, `01_*`, `02_*`, `tai_hien_r28_f1.py`.
- `83482c2`: sửa — `src/bdd_yolo_export.py`, `src/train_detector.py`, `src/test_detector_5lop.py`.
- `48dc28a`: bằng chứng sau sửa — `logs/review29/03_*`–`05_*`, `tests_*`, `fixture_gpu/`.
- Commit tài liệu (commit này): `REVIEW29_CHI_TIET.md`, `REVIEW_BRIEF.md`, `CODEX_HANDOFF.md` (hàng 24, 25),
  `DETECTOR_TRAIN_PLAN.md` (chính sách cache), `PLAN_DA_CHOT.md` (trạng thái), `REVIEW28_CHI_TIET.md` (đính chính).
