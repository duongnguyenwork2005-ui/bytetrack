# Vòng 39 — kế hoạch kiểm resume và bộ giám sát (ghi TRƯỚC khi chạy kiểm chính thức)

Ngày 2026-10-03, người dùng duyệt hai việc trước lượt training dài: thêm resume cho runner huấn luyện và sửa bộ đo RAM.
Mọi phép kiểm dưới đây là kiểm kỹ thuật ngắn trên fixture hoặc dataset smoke đã khóa. Đây không phải pilot, không phải
training dài, và không chạy trên dataset đầy đủ. Kết quả chỉ dùng để kiểm cơ chế resume và giám sát, không dùng để so
model hay chọn tham số.

## 0. Khai báo trước khi đăng ký

- `src/test_chay_giam_sat.py` đã chạy trước khi viết kế hoạch này: **23/23 PASS**, log
  `02_test_chay_giam_sat_lan1.log`. Test này dùng tiến trình giả, không train.
- Trong lúc phát triển đã chạy **một lượt thử** trong scratchpad, ngoài repo, nên **không phải bằng chứng**:
  - fixture CPU, yolov8n;
  - C1 liên tục 3 epoch: 29 s, exit 0;
  - I bị kill sau epoch 1: exit 15;
  - dry-run tiếp tục: exit 0;
  - resume: exit 0, 13 mục runner tự kiểm đều đạt.

  Quan sát của lượt thử:
  - epoch 1 trùng;
  - sau resume LR và momentum trùng;
  - accumulate R = 8, C1 = 7;
  - bước tối ưu sau epoch 1: R ở ni 10, 18, 26; C1 ở ni 16, 23;
  - loss và thứ tự dữ liệu khác.
- Sau lượt thử, mã đã sửa thêm:
  - `mo_phong` trả danh sách bước;
  - K2 chỉ lọc dòng của lần 0;
  - số lần tiếp tục lấy max từ nhật ký và tên file;
  - so patience qua `json_safe`;
  - mỗi epoch ghi `mosaic_thuc_te`, đối chiếu ở K4.

  Tiêu chí dưới đây suy từ mã ultralytics 8.4.141, không chỉnh theo số của lượt thử. Lượt thử chỉ xác nhận dự đoán
  trước khi đăng ký.

## 1. Cách so (công bố trước, áp dụng cho mọi phần)

- **Trùng từng bit**:
  - CSV theo batch: chuỗi của từng ô phải giống hệt, trừ cột `t` (thời điểm) và `lan` (số lần chạy);
  - tensor: `torch.equal`;
  - số nguyên và chuỗi: `==`.
- **Không dùng tolerance cho bất kỳ phép so nào.** Lệch ở đâu thì báo số lượng và độ lớn, không nới ngưỡng.
- Resume **không** được kỳ vọng trùng từng bit với lượt liên tục, vì ultralytics không khôi phục RNG, thứ tự dữ liệu và
  gradient tích lũy dở. Phần khác biệt được dự đoán trước (K5b, K6) và báo đúng độ lớn.
- Ranh giới hỗ trợ: tiếp tục **sau một epoch đã hoàn tất** (đã ghi `last.pt` và bản sao `khoi_phuc/last_eNNN.pt`).
  Phần epoch đang chạy lúc bị ngắt, kể cả validate, phải chạy lại từ đầu epoch. Không tiếp tục giữa batch.

## 2. A — `src/test_tiep_tuc_huan_luyen.py` (CPU, fixture tổng hợp)

**Cấu hình**: `train_bdd5.yaml`, chỉ đổi `imgsz` 160, `workers` 0, `epochs` 3. Các khóa khác giữ nguyên: batch 8, nbs 64,
SGD lr0 0,01, warmup_epochs 1,0, close_mosaic 2, patience 5, seed 0, deterministic.

**Dataset**: fixture do exporter thật sinh, qua hợp đồng v2. Dự kiến nb = 10 batch/epoch, nw = 10.

**Hai nhóm lượt**:
- **CHẠY THẬT**: `train_detector.py` qua subprocess, ultralytics thật, CPU.
- **SPY**: `ultralytics.YOLO` bị thay bằng spy chặn `train()`. Mọi bước trước `train()` vẫn chạy thật.

### Nhóm U (đơn vị, không train)

| Mã | Kiểm | Dự kiến |
|:--|:--|:--|
| U1a | Nhật ký có dòng cuối ghi dở | bỏ dòng đó và cảnh báo; còn 2 sự kiện |
| U1b | Nhật ký có dòng hỏng ở giữa | `LoiTiepTuc` |
| U2 | `kiem_tuong_thich` | đường dẫn chuẩn hóa trùng → `[]`; lệch lr0 / epochs / device / data → 4 mục |
| U3a | Checkpoint đủ trạng thái, sha có trong nhật ký | hợp lệ |
| U3b | File có sha **không** có trong nhật ký | loại; **không** `torch.load` |
| U3 (×6) | strip / thiếu optimizer / thiếu EMA / `save_dir` khác / `name` khác / lệch nhật ký | loại, ghi đúng lý do |
| U4 | Số lần tiếp tục | = max(nhật ký, tên `nguon_tiep_tuc_N.pt`) = 2 |
| U5 | Chép nguyên tử | cùng sha, không còn file tạm |

### Nhóm R + K (mỗi model yolov8n, yolo11n; CHẠY THẬT)

Các lượt:
- **C1**, **C2**: liên tục 3 epoch;
- **I**: cùng cấu hình, **kill cả cây tiến trình** ngay khi nhật ký ghi `epoch_da_luu` epoch 1;
- **R**: `--tiep-tuc` I, chạy tới đủ 3 epoch.

| Mã | Phép so | Dự kiến | Loại |
|:--|:--|:--|:--|
| R0/R1/R2 | Mã thoát | C1, C2 exit 0; I bị kill, nhật ký chỉ có `bat_dau` + `epoch_da_luu 1`; R exit 0 | đạt / không đạt |
| K1 | C1 vs C2: mọi dòng theo batch; checkpoint e001..e003 (EMA, optimizer, updates, epoch, best_fitness) | trùng từng bit (CPU tất định) | đạt / không đạt |
| K2 | I epoch 1 vs C1 epoch 1 (10 dòng) và checkpoint e001 | trùng từng bit | đạt / không đạt |
| K3a | Runner tự kiểm sau resume (12 mục + EarlyStopping) | tất cả đạt | đạt / không đạt |
| K3b | Nguồn = e001 của I; sha trong nhật ký = sha `nguon_tiep_tuc_1.pt`; start_epoch 1, tổng 3 | đúng | đạt / không đạt |
| K4 | R (epoch 2–3, 20 dòng) vs C1: epoch, i, ni, LR 3 nhóm, momentum 3 nhóm, batch; LR không đổi trong epoch; mosaic thực tế theo epoch | trùng từng giá trị; mosaic {1: bật, 2: tắt, 3: tắt} | đạt / không đạt |
| K5a | accumulate và `ema.updates` từng batch của C1 **và** R vs mô phỏng `_do_train` | khớp; bước tối ưu sau epoch 1: C1 [16, 23], R [10, 18, 26] | đạt / không đạt |
| K5b | accumulate sau warmup | C1 = 7 (nw 10 < 15), R = 8; lượt dài nw 6.311 → 8 = 8 | đạt / không đạt |
| K6 | Khác biệt: số dòng loss khác, thứ tự dữ liệu khác, \|EMA cuối R − C1\| max, fitness cuối | loss khác > 0 và lệch EMA > 0 (không bit-identical); báo độ lớn, không đặt ngưỡng | kiểm dự đoán |
| K7 | Nhật ký I | đúng chuỗi `bat_dau(0)`, `epoch_da_luu(0,1)`, `bat_dau(1)`, `epoch_da_luu(1,2)`, `epoch_da_luu(1,3)`, `ket_thuc(1)`; provenance đủ; file cũ lưu vào `khoi_phuc/truoc_tiep_tuc_1/` trùng sha bản nguyên vẹn; `results.csv` 3 dòng | đạt / không đạt |
| K8 | Kết thúc R | `hoan_tat`, exit 0; `last.pt` đã strip (epoch −1, optimizer None); train_args epochs 3; checkpoint 5 lớp | đạt / không đạt |

### Nhóm S (SPY, yolov8n)

Mỗi ca từ chối phải thỏa đủ: đúng exit code, **không gọi `train()`**, thư mục lượt không đổi (so ảnh chụp sha256 từng
file), và thông báo nêu đúng lý do.

| Mã | Ca | Exit |
|:--|:--|:--|
| S1 | `--tiep-tuc` khi lượt không tồn tại; không tạo thư mục | 2 |
| S2 | Thư mục không có nhật ký (như lượt smoke/pilot cũ); không tạo metadata | 2 |
| S3 | Lượt đã hoàn tất (C1); không đổi file | 2 |
| S3b | Lượt đủ epoch nhưng mất dòng `ket_thuc` | 2 |
| S4 | `last.pt` bị cắt đôi, còn bản sao e001 | chọn bản sao; spy nhận `resume = khoi_phuc/nguon_tiep_tuc_1.pt` cùng sha e001; `last.pt` bị loại có lý do |
| S5 | `last.pt` hỏng và không còn bản sao | 2 |
| S6 | `last.pt` là checkpoint của lượt khác | 2 |
| S7a | Sửa một file nhãn | 2 (hợp đồng) |
| S7b | Bản xuất khác, hợp đồng vẫn đạt | 2 (vân tay) |
| S8 | lr0 0,02 | 2 |
| S9 | epochs 3 → 4 (không cộng thêm epoch) | 2 |
| S10 | Thiếu `--xac-nhan-train-dai` | 1 |
| S11 | `--che-do smoke` | 1 |
| S12 | Lượt mới trùng tên thư mục | 2 |
| S13 | Sai model | 2 |
| S14 | Nhật ký ghi `dung` | 2 |
| S15 | Phiên bản ultralytics của lượt gốc khác | 2 |
| S16 | `--batch 4` | 1 |
| S17 | `--dry-run --tiep-tuc`: in kế hoạch "tiếp tục từ epoch 2/3", không gọi train, không ghi gì | 0 |
| S18 | Ctrl+C (spy ném `KeyboardInterrupt`); kết quả ghi "bị ngắt" | 130 |

## 3. B — `handoff/logs/review39/kiem_resume_gpu_smoke.py` (GPU, dataset smoke đã khóa, cả hai model)

Phần này kiểm cùng cơ chế trên đúng phần cứng, dataset và cấu hình của lượt dài, chỉ khác số epoch. Đồng thời đây là phép
kiểm thật cho bộ giám sát RAM mới, với 2 worker DataLoader.

**Đầu vào**
- Cấu hình: `cau_hinh_kiem_resume_3epoch.yaml` = `train_bdd5.yaml`, **chỉ đổi** `epochs` 15 → 3 (thêm 4 dòng chú thích
  đầu file).
- Dataset: `data/yolo_bdd5_v1_smoke/data.yaml` (334 / 84 ảnh, hợp đồng v2).
- Thiết bị: `--device 0`; batch 8, imgsz 640, workers 2, AMP tắt.

**Lượt (tuần tự, mỗi model)**: C liên tục 3 epoch; I bị kill sau epoch 1; R `--tiep-tuc` I.
- Mọi lượt chạy qua `src/chay_giam_sat.py chay`.
- Với I, script kill **PID runner đọc từ `trang_thai.json` của công cụ** cùng các tiến trình con, không kill công cụ.

**Thư mục mới** (không ghi đè lượt / kết quả cũ):
- `models/finetune_phase2/kiem_resume_v39/<model>_{C,I}` — không commit;
- `results/phase2/kiem_resume_v39/` — không commit;
- `handoff/logs/review39/05_gpu/<model>_{C,I,R}/` — log, CSV RAM, tóm tắt của công cụ;
- `05_gpu/<model>_luot/` — bản chép nhật ký, theo batch, results, args, JSON runner; không chép `.pt`;
- tổng hợp: `06_kiem_resume_gpu.json`.

**Dự đoán theo mã** (nb = ⌈334 / 8⌉ = 42, nw = 42):
- accumulate cuối warmup C = 8 = accumulate sau resume R, nên ở smoke **không** có khác biệt accumulate;
- bước tối ưu sau epoch 1:
  - C: [44, 52, …, 124];
  - R: [42, 50, …, 122]. Bước đầu tiên của R chỉ mang gradient của 1 batch;
- `ema.updates`: sau epoch 1 = 12; cuối lượt = 23 ở cả C và R;
- mosaic: epoch 1 bật, epoch 2–3 tắt.

**Tiêu chí đạt / không đạt** (`danh_gia` trong script):

| Nhóm | Tiêu chí |
|:--|:--|
| Mã thoát | C exit 0; I bị kill với nhật ký chỉ `bat_dau` + `epoch_da_luu 1`; R exit 0 |
| Khôi phục | K3 ≥ 13 mục runner tự kiểm đều đạt; nguồn = e001, start_epoch 1, tổng 3 |
| Lịch | K4 trùng C từng giá trị (ni, LR, momentum, batch) + mosaic như dự đoán; K5 khớp mô phỏng ở cả C và R |
| Nhật ký, kết thúc | K7 đúng chuỗi sự kiện; K8 `hoan_tat`, exit 0, checkpoint 5 lớp |
| Giám sát C, I, R | PID công cụ bám = PID runner **tự ghi** trong nhật ký; CSV chỉ chứa PID đó; exit code giữ đúng qua công cụ (C 0, I 15, R 0); có mẫu, không mẫu nào không đọc được, RSS > 0; `tinh-lai` khớp `tom_tat.json` |

**Chỉ báo, không đặt tiêu chí**:
- K2: I trùng C ở epoch 1? Trên GPU **không giả định** tất định;
- K6: khác biệt loss, thứ tự, EMA, fitness;
- số tiến trình con tối đa: worker DataLoader và `nvidia-smi` của bộ đo VRAM trong runner;
- RSS và RAM trống toàn máy.

**Thời gian ước tính**: khoảng 15–20 phút cho cả hai model. Pilot vòng 37 mất khoảng 160 s cho 2 epoch mỗi model.

## 4. Suite bị ảnh hưởng trực tiếp (runner đã đổi)

- `src/test_detector_5lop.py`: vòng 37 đạt 92/92;
- `src/test_pilot_chan_doan.py`: vòng 37 đạt 34/34.

Cả hai phải PASS như trước. Không chạy lại suite tracking / Nano.

Log được lưu với tên mới và exit code thật. Lỗi thì giữ log, sửa, rồi chạy lại với tên log mới.

## 5. Không làm trong vòng này

- Không train trên dataset đầy đủ, không pilot, không chọn số epoch hay ngân sách.
- Không sửa dataset, split, checkpoint lịch sử, lượt smoke / pilot cũ.
- Checkpoint smoke / pilot cũ đã strip optimizer: giữ nguyên, báo là không resume được (S2), không tạo metadata hồi tố.

## 6. Phụ lục sửa đổi sau khi đăng ký (ghi thêm; mục 0–5 giữ nguyên)

- **Lần 1** (`03_..._lan1.log`, commit `231006c`): nhóm U 14/14 PASS, R0/R1 yolov8n PASS. Sau đó cả test thoát mã 2 ngay
  đầu nhóm S, không có traceback.
  - Nguyên nhân: lỗi của test. `argv()` cắt `[2:]` nên `sys.argv` còn sót đường dẫn `train_detector.py`, argparse gọi
    `SystemExit(2)`; `_chay` chỉ bắt `Exception`.
  - Sửa ở commit `26fdc81`. Tiêu chí không đổi.
- **Lần 2** (`03_..._lan2.log`, commit `26fdc81`): **57/59 PASS**. FAIL là K4 của cả hai model, **chỉ ở phần mosaic**;
  epoch, ni, LR, momentum, batch đều trùng (`lech []`).
  - Mosaic ghi được `False` ở mọi epoch, kể cả epoch 1 (dự đoán bật).
  - Nguyên nhân: **phép đo sai**. `YOLODataset` của ultralytics 8.4.141 **không có thuộc tính `mosaic`**, nên
    `getattr(ds, "mosaic", None)` luôn là `None` → `False`. `close_mosaic()` chỉ dựng lại transforms với `hyp.mosaic = 0`.
  - Sửa: đọc `p` của transform `Mosaic` đầu tiên trong `ds.transforms` (`xac_suat_mosaic`). Nhật ký ghi thêm `mosaic_p`.
    Thêm **U6**: dựng transforms thật bằng `v8_transforms` (p = 1,0 khi bật, 0,0 khi tắt) và xác nhận lớp `YOLODataset`
    không có thuộc tính `mosaic`. **Tiêu chí K4 không đổi**: mosaic {1: bật, 2: tắt, 3: tắt}, C1 = I/R.
  - Cùng cách đọc sai có ở `pilot_chan_doan.py` vòng 37 (trường `mosaic_thuc_te` trong JSON pilot luôn `False`, nên
    **không có giá trị đo**). Kết luận vòng 37 "mosaic tắt cả 2 epoch" vẫn có bằng chứng riêng: dòng log
    "Closing dataloader mosaic" xuất hiện đúng 1 lần trong mỗi log pilot, ngay epoch đầu, cộng với mã. Vòng này không sửa
    `pilot_chan_doan.py` (pilot đã xong, ngoài phạm vi); báo để người dùng quyết định.
- **Lần 3** (`03_..._lan3.log`, commit `53a328f`): **60/60 PASS**, exit 0. K4 đạt ở cả hai model; mosaic `p` ở C1 và I/R
  đều là {1: 1,0, 2: 0,0, 3: 0,0}.
- **Bổ sung sau lần 3** (ghi trước lần 4): xử lý một tình huống biên chưa có trong kế hoạch ban đầu.
  - Tình huống: bị ngắt **sau khi** ultralytics ghi `results.csv`, `last.pt`, `best.pt` của epoch k nhưng **trước khi**
    runner ghi `epoch_da_luu` k. Cửa sổ này dài khoảng 1–3 s mỗi epoch: chép bản sao, băm, fsync.
  - Khi đó resume chọn epoch k−1 (đúng), nhưng còn hai lỗi:
    - `best.pt` có thể thuộc **nhánh bị bỏ**, và `final_eval` của ultralytics đánh giá chính `best.pt`;
    - `results.csv` có **dòng trùng** epoch k, vì ultralytics resume ghi nối tiếp.
  - Mã mới:
    - `quy_dao()` dựng chuỗi epoch đã lưu dẫn tới checkpoint được chọn (lần tiếp tục → checkpoint nguồn → … → lần 0);
    - `ke_hoach_best()`: `best.pt` phải bằng best của quỹ đạo. Nếu lệch, khôi phục từ bản sao `last_eNNN.pt` cùng byte
      (ultralytics ghi `best.pt` từ cùng byte với `last.pt`). Không còn bản sao đúng byte → từ chối, exit 2;
    - `sao_luu_truoc_tiep_tuc` cắt `results.csv` về epoch ≤ epoch đã lưu, xử lý theo byte nên giữ CRLF.
    - Cả hai thao tác chạy **sau** khi đã lưu bản cũ vào `truoc_tiep_tuc_N/`.
  - Tiêu chí mới (đạt / không đạt):
    - **U7a**: quỹ đạo qua 2 lần tiếp tục đúng thứ tự; best của quỹ đạo không lấy nhánh bị bỏ; nguồn không truy được →
      `LoiTiepTuc`;
    - **U7b**: khớp → không làm gì; lệch → khôi phục từ bản sao; bản sao hỏng → `LoiTiepTuc`;
    - **U8**: `results.csv` chỉ còn epoch ≤ epoch đã lưu, đúng byte CRLF, bản gốc được lưu;
    - **S19** (spy): `best.pt` của nhánh bị bỏ cùng dòng `results.csv` epoch 2 → trước `train()`: `best.pt` = best e001,
      bản cũ được lưu, `results.csv` chỉ còn epoch 1;
    - **K7** thêm: mọi epoch ghi best có `best_sha256 == ban_sao_sha256` (giả định dùng để khôi phục); quỹ đạo của R là
      `[[0, 1]]`.
  - Trước khi đăng ký, đã chạy riêng nhóm U một lần để bắt lỗi cú pháp: 17/17. Đây là kiểm phát triển, không phải lần
    chạy chính thức.
- **Lần 4** (`03_..._lan4.log`, commit `2f9eee4`): **64/64 PASS**, exit 0.
- **Kiểm GPU** (`06_*`, commit `eca9008`): cả hai model **ĐẠT** 21/21 tiêu chí của mục 3.
- **Bổ sung sau lần 4** (ghi trước lần 5; **chỉ sửa file test**, runner không đổi nên bằng chứng GPU vẫn ứng với mã hiện
  tại):
  - Lý do: ở mọi phép kiểm trước, điểm tiếp tục trùng đúng epoch tắt mosaic (`start_epoch = epochs − close_mosaic`), nên
    nhánh `resume_training` tự tắt mosaic khi `start_epoch > epochs − close_mosaic` **chưa chạy**. Với lượt dài, nhánh này
    chạy nếu bị ngắt sau epoch 14.
  - **R1b**: I2 cùng cấu hình bị kill ngay sau khi nhật ký ghi epoch 2.
  - **K9** (đạt / không đạt), tiếp tục I2:
    - exit 0;
    - I2 epoch 1–2 trùng C1 từng bit;
    - runner tự kiểm ≥ 13 mục đều đạt; `start_epoch` 2;
    - epoch 3 trùng C1 ở ni, LR, momentum, batch; accumulate / `ema.updates` khớp mô phỏng;
    - mosaic `p` {1: 1,0, 2: 0,0, 3: 0,0};
    - log của lần tiếp tục có "Closing dataloader mosaic", tức nhánh `resume_training` đã chạy.
