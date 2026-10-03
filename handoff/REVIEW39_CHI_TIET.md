# Vòng 39 — ghi nhận nghiệm thu `b83dc12`; resume cho runner huấn luyện và bộ giám sát RAM bám PID (chi tiết)

**Phạm vi** (người dùng duyệt): thêm resume cho runner huấn luyện và sửa bộ đo RAM **trước** lượt training dài. Gồm triển
khai, kiểm thử kỹ thuật ngắn (fixture CPU và dataset smoke đã khóa trên GPU), bàn giao, rồi **dừng**.

> **Theo dõi vòng 40**: Codex `1454b3d` REQUEST CHANGES vì **R39-F1**: resume nối nhật ký ngay sau dòng cuối ghi dở,
> làm nhật ký hỏng ở giữa (liên quan mục 3.3–3.4). Đã sửa ở vòng 40; xem `REVIEW40_CHI_TIET.md`. Các số liệu khác của
> phụ lục này không đổi.

- Không training dài, không chạy trên dataset đầy đủ, không pilot.
- Không SSH Nano, không xuất lại dataset, không thêm model / dataset / thí nghiệm.
- Không đổi dataset, split, checkpoint lịch sử hay kết quả đã nghiệm thu.

## 0. Trạng thái đầu vòng

Log: `logs/review39/00_trang_thai_dau.log`.

- HEAD `b83dc12` (đúng target vòng 38), 0 commit sau đó. Working tree chỉ có file riêng của người dùng.
- Môi trường: ultralytics 8.4.141 (`engine/trainer.py` `7532ff0b…`), torch 2.6.0+cu124, psutil 7.2.2. RAM còn trống
  0,67 GB lúc kiểm.

## 1. Ghi nhận nghiệm thu `b83dc12` (ĐÃ CHẠY)

Commit `53dfc22`.

- **Báo cáo** `D:\LABOTARY\review_codex\CODEX_REVIEW_b83dc12.md`: 9.965 B, sha256 `d58d8f74…`.
  - **PASS WITH LIMITATIONS** cho xuất dataset đầy đủ vòng 38.
  - Hai ghi chú không chặn **R38-N1**, **R38-N2**, đã gộp vào tài liệu (mục 2).
- **Bằng chứng**: ZIP 585.769 B, sha256 `668f94eb…`. Giải nén vào thư mục mới; `SHA256SUMS` khớp **137/137**, cả trên blob
  đã stage. Lưu ở `handoff/review_evidence/CODEX_b83dc12/` (`-text`); ZIP gốc giữ nguyên.
- **Gói đã nghiệm thu**: 816.591 B, `f4efa478…`, khớp. Lưu ở `handoff/archive/review_delta_b83dc12.zip`.
- **Base review** `0c0a07a` → **`b83dc12`**; 13 khai báo NPZ giữ nguyên.
- **Phạm vi reviewer**: đã kiểm gói, log và số liệu (115/115 payload + 3 alias, 118/118 đường dẫn, 6/6 preimage, 188/188
  điều kiện đối chiếu). **Không** kiểm trực tiếp dataset 8,03 GB, **không** kiểm toàn cây Git.

## 2. Đính chính R38-N1, R38-N2 (chỉ diễn giải; số liệu không đổi)

- **R38-N1**: "789 bước tối ưu/epoch" là **ước tính với accumulate cố định** (⌈6.311 / 8⌉), chưa tính warmup (accumulate
  tăng dần từ 1 trong epoch đầu) và ranh giới cập nhật giữa các epoch. "94.665 batch cho 15 epoch" **chỉ đúng nếu chạy đủ
  15 epoch** (patience 5 có thể dừng sớm). Đã sửa ở:
  - `REVIEW38_CHI_TIET.md` mục 5: thêm chữ "khoảng" vào câu cũ, kèm một đoạn đính chính;
  - ghi chú `suy_ra()` của runner (trường `ghi_chu` của mọi lần dry-run sau này).
- **R38-N2**: phép kiểm danh sách frame ở vòng 38 **dùng lại `chon_frame()` của exporter** (cùng `doc_split()` /
  `chon_tap_con()`), nên **không phải một triển khai độc lập của quy tắc lấy mẫu**. Đổi tên gọi thành "kiểm chéo" ở
  `REVIEW38_CHI_TIET.md` mục 4, `DETECTOR_TRAIN_PLAN.md`, `PLAN_DA_CHOT.md` và hàng 34 của `CODEX_HANDOFF.md`. Tên file
  `05_kiem_doc_lap.py` giữ nguyên.
- Không xuất lại, không pilot.

## 3. Resume cho runner (`src/tiep_tuc_huan_luyen.py` mới, `src/train_detector.py`)

### 3.1 Cú pháp (đã kiểm qua entry point thật)

```text
# lượt MỚI (như trước; thư mục lượt đã tồn tại -> exit 2, không ghi đè)
python -u src/train_detector.py --model <M> --che-do day-du --xac-nhan-train-dai
# kiểm điều kiện tiếp tục: in epoch sẽ tiếp tục, checkpoint chọn, ứng viên bị loại -- KHÔNG train, KHÔNG ghi gì
python src/train_detector.py --model <M> --dry-run --tiep-tuc
# tiếp tục
python -u src/train_detector.py --model <M> --che-do day-du --xac-nhan-train-dai --tiep-tuc
```

- Tên lượt mặc định giống nhau cho lượt mới và lượt tiếp tục: `train_bdd5_v1_<M>_day_du`. Có thể chỉ định bằng `--ten`.
- `--tiep-tuc` chỉ dùng với `day-du` (hoặc `--dry-run`), và vẫn cần `--xac-nhan-train-dai`.
- Không dùng được `--batch` / `--workers` cùng `--tiep-tuc`.
- Mã thoát: 0 hoàn tất; 1 sai cờ; 2 từ chối trước khi train; 3 lỗi; 130 Ctrl+C.
- Kết quả của lần tiếp tục thứ N ghi vào `<tên>__tiep_tuc_N.json`, không ghi đè.

### 3.2 Cơ chế và ranh giới hỗ trợ

- Dùng **nguyên** resume của ultralytics 8.4.141, truyền **đường dẫn** checkpoint: `YOLO(ck).train(resume=<đường dẫn>)`.
  Không viết vòng train khác.
- **Không** dùng `resume=True`: khi checkpoint thiếu epoch / optimizer, ultralytics chỉ cảnh báo rồi **tự bắt đầu lượt
  mới**. Runner chặn mọi trường hợp đó trước.
- **Ranh giới**: tiếp tục **sau một epoch đã hoàn tất**, tức sau khi đã ghi `last.pt` và nhật ký epoch. Không tiếp tục giữa
  epoch, không giữa batch.

### 3.3 Nhật ký lượt, điểm khôi phục, provenance

- `<run_dir>/nhat_ky_luot.jsonl` (append + fsync) có ba loại sự kiện:
  - `bat_dau`: lượt mới hoặc tiếp tục;
  - `epoch_da_luu`: ghi mỗi epoch;
  - `ket_thuc`.
- `<run_dir>/khoi_phuc/last_eNNN.pt`: bản sao `last.pt` **mỗi epoch**, chép qua file tạm, kiểm sha rồi `os.replace`.
  Một lần ghi `last.pt` bị ngắt không làm mất điểm khôi phục trước đó.
- `<run_dir>/theo_batch.csv` ghi theo từng batch: lần, epoch, ni, loss, LR và momentum 3 nhóm, accumulate, `ema.updates`,
  batch, băm thứ tự ảnh.
- `epoch_da_luu` ghi:
  - sha `last.pt` và bản sao, sha `best.pt` (nếu epoch đó ghi best);
  - fitness, trạng thái EarlyStopping, quyết định dừng;
  - `ema.updates`, accumulate, LR, momentum;
  - băm thứ tự dữ liệu của epoch;
  - mosaic thực tế (`p` của transform Mosaic);
  - metrics.
- **Provenance** (sự kiện `bat_dau` của mỗi lần):
  - lệnh, cấu hình + sha256, tham số khai báo;
  - vân tay dataset (hợp đồng v2), checkpoint khởi tạo;
  - môi trường: python / torch / ultralytics / CUDA / cuDNN, sha `trainer.py`;
  - sha mã nguồn 4 file, git HEAD, thời điểm, PID.

  Với lần tiếp tục, ghi thêm:
  - checkpoint nguồn + sha, epoch đã lưu, epoch bắt đầu;
  - bản sao lưu, quỹ đạo, kế hoạch `best.pt`;
  - 13 kết quả tự kiểm khôi phục.

  Lịch sử các lần trước giữ nguyên trong nhật ký.

### 3.4 Kiểm trước khi train

Mọi kiểm dưới đây là chỉ đọc. Từ chối → exit 2, **không gọi train**, không tự khởi động lại.

1. Thư mục lượt tồn tại (S1).
2. Có nhật ký lượt (S2). Lượt smoke / pilot cũ không có nhật ký nên bị từ chối; không tạo metadata hồi tố.
3. Đúng một `bat_dau` lượt mới, đúng tên lượt.
4. Chưa có `ket_thuc hoan_tat` (S3).
5. Có ít nhất một epoch đã lưu. Nếu bị ngắt trong epoch 1 thì chạy lượt mới với `--ten` khác.
6. Cùng phiên bản ultralytics / torch và cùng sha `trainer.py` (S15).
7. Hợp đồng dataset v2 (S7a) và vân tay dataset trùng lượt gốc (S7b).
8. Checkpoint khởi tạo trùng lượt gốc (S13).
9. Chọn **checkpoint hợp lệ gần nhất** trong các ứng viên `weights/last.pt` và `khoi_phuc/last_e*.pt`:
   - chỉ `torch.load` file có sha nằm trong nhật ký (S6);
   - phải đủ epoch, optimizer, EMA + updates, khóa scaler;
   - `save_dir` / `name` đúng lượt;
   - nội dung khớp nhật ký (epoch, updates, best_fitness).

   `last.pt` hỏng → dùng bản sao (S4); không còn ứng viên nào → từ chối (S5).
10. Epoch đã chọn mang quyết định DỪNG (S14, S3b), hoặc đã đủ epoch → từ chối.
11. Cấu hình hiện tại trùng `train_args` của checkpoint ở **mọi khóa**: S8 (lr0), S9 (epochs; không cộng thêm epoch).
12. Dựng quỹ đạo và kế hoạch `best.pt` (mục 3.7). Không còn bản sao đúng byte → từ chối.

Sau khi qua các kiểm trên, runner ghi theo thứ tự:
- sao lưu (mục 3.7);
- khôi phục `best.pt` nếu cần;
- cắt `results.csv`;
- gọi ultralytics.

Trong `on_pretrain_routine_end`, trước vòng train:
- kiểm dữ liệu ultralytics **thực nạp** (callback có từ trước);
- **13 kiểm khôi phục** (mục 3.5). Sai → dừng.

### 3.5 Trạng thái khôi phục được — ultralytics nạp, runner kiểm lại sau khi nạp

| Trạng thái | Ultralytics 8.4.141 khôi phục | Runner kiểm (so với chính checkpoint nguồn, nạp lại độc lập) |
|:--|:--|:--|
| Epoch | `start_epoch = epoch + 1`; tổng epoch lấy từ checkpoint | `start_epoch`; tổng epoch = cấu hình |
| Optimizer | `load_state_dict` (momentum SGD lưu FP16 → nạp FP32) | momentum từng tham số, lệch max = 0; `param_groups` (lr, initial_lr, momentum, weight_decay, nesterov, dampening) |
| EMA | trọng số (FP16 → `.float()`) + `updates` | từng tensor `torch.equal`; `updates` |
| Model đang học | **dựng từ EMA của checkpoint** (ultralytics không lưu model thô) | từng tensor = EMA checkpoint |
| LR / scheduler | `scheduler.last_epoch = start_epoch − 1` | `last_epoch`; K4: LR từng batch trùng lượt liên tục |
| Scaler | AMP tắt → trạng thái rỗng | có khóa `scaler` (lọc ứng viên) |
| best_fitness | từ checkpoint | = checkpoint |
| EarlyStopping | **tạo mới** | runner khôi phục `best_fitness` / `best_epoch` / `possible_stop` từ nhật ký; patience khớp |
| Tham số | `train_args` của checkpoint | mọi khóa (trừ `resume`, `model`) trùng checkpoint |
| Batch | — | = cấu hình (kiểm cả mỗi epoch: chặn việc ultralytics tự giảm batch khi OOM) |
| Mosaic | `close_mosaic` theo epoch (và trong `resume_training`) | `p` thực tế từng epoch; K4 so với lượt liên tục |

### 3.6 Không khôi phục — giới hạn

- RNG python / numpy / torch: ultralytics gọi lại `init_seeds`, generator của DataLoader tạo mới. Vì vậy **thứ tự dữ liệu và
  augmentation ngẫu nhiên** sau resume khác lượt liên tục.
- **Gradient tích lũy dở** của các batch cuối epoch trước bị bỏ, vì `last_opt_step` đặt lại −1. Hệ quả:
  - bước tối ưu đầu tiên rơi vào batch đầu sau resume và **chỉ mang gradient của 1 batch**;
  - ranh giới các bước tối ưu sau đó lệch so với lượt liên tục.
- `accumulate` lấy lại `round(nbs / batch)` = 8. Giá trị này chỉ khác accumulate cuối warmup khi nw < 15: fixture CPU có
  nw = 10 cho 7 vs 8; smoke và lượt dài cho 8 = 8.
- Model đang học được dựng từ **EMA** của checkpoint, không phải từ trọng số thô. EMA và momentum được lưu FP16, nên có làm
  tròn.
- Hệ quả chung: tiếp tục **không trùng từng bit** với lượt liên tục. Độ lớn khác biệt đo ở mục 5 (K6).

### 3.7 Bảo toàn, quỹ đạo, phần phải làm lại sau gián đoạn

- **Trước khi ghi tiếp**, runner:
  - chép checkpoint nguồn thành `khoi_phuc/nguon_tiep_tuc_N.pt` (không bao giờ bị ghi đè); ultralytics nhận đúng đường dẫn
    này;
  - lưu `args.yaml`, `results.csv`, `theo_batch.csv`, `weights/last.pt`, `weights/best.pt` cũ vào
    `khoi_phuc/truoc_tiep_tuc_N/`.
- **Quỹ đạo**: chuỗi epoch đã lưu dẫn tới checkpoint được chọn (lần tiếp tục → checkpoint nguồn của nó → … → lần 0). Nhánh
  bị bỏ không thuộc quỹ đạo.
- **`best.pt`** phải là best của quỹ đạo. Nếu lệch, runner khôi phục từ bản sao `khoi_phuc/last_eNNN.pt` của epoch best:
  ultralytics ghi `best.pt` từ cùng byte với `last.pt`, và K7 xác nhận điều này ở mọi epoch ghi best. Không có bản sao
  đúng byte → từ chối.
- **`results.csv`** của ultralytics chỉ giữ epoch ≤ epoch đã lưu (xử lý theo byte, giữ CRLF), vì ultralytics resume ghi nối
  tiếp. `theo_batch.csv` giữ toàn bộ lịch sử và có cột `lan`.
- **Phải làm lại** sau gián đoạn:
  - epoch đang chạy lúc bị ngắt, kể cả validate, chạy lại từ đầu epoch;
  - nếu bị ngắt sau khi ultralytics ghi checkpoint nhưng trước khi runner ghi nhật ký epoch đó (khoảng 1–3 s mỗi epoch),
    epoch đó cũng chạy lại; `best.pt` / `results.csv` được sửa như trên;
  - ngắt trong epoch 1: không có điểm khôi phục, phải chạy lượt mới với `--ten` khác (thư mục cũ giữ nguyên);
  - ngắt sau epoch cuối, trong `final_eval` (ultralytics strip optimizer, ghi lại `last.pt` / `best.pt`, rồi validate
    `best.pt`): nhật ký đã ghi epoch cuối kèm quyết định DỪNG, nên **không resume**. Các bản sao `khoi_phuc/last_eNNN.pt`
    (kể cả của epoch best) còn nguyên để dựng lại `best.pt`. Đánh giá cuối phải chạy riêng; việc này ngoài phạm vi vòng
    này.

## 4. Bộ giám sát mới `src/chay_giam_sat.py` (thay bộ đo RAM ngoài)

**Lỗi vòng 38**: `do_tai_nguyen_ngoai.py ram --mau` chọn tiến trình **đầu tiên** có tên script trong command line, nên đã bám
nhầm shell của lệnh Monitor (pid 23788) thay vì exporter (pid 20304). Script cũ giữ nguyên làm bằng chứng.

**Thiết kế mới**

- `chay --thu-muc <mới> -- <lệnh>`: công cụ **tự khởi chạy** lệnh rồi bám PID của chính tiến trình con nó tạo, kèm
  `create_time`. PID được ghi vào `trang_thai.json` ngay khi tạo.
- **Mỗi mẫu** (mặc định 1 s) ghi:
  - timestamp tuyệt đối và tương đối;
  - RSS tiến trình chính;
  - số tiến trình con và RSS **từng** con (`PID:tên:RSS`), gồm worker DataLoader tạo / kết thúc giữa chừng và
    `nvidia-smi` của runner;
  - RAM trống toàn máy.

  `rss_tong` (chính + con) có thể **đếm trùng** trang chia sẻ; ghi chú này nằm trong tóm tắt. Không gộp VRAM (runner đo
  VRAM riêng bằng `nvidia-smi`).
- Mỗi mẫu kiểm lại `create_time`:
  - PID bị tái dùng → dừng lấy mẫu, ghi lý do;
  - tiến trình chính kết thúc → dừng, ghi lý do;
  - không đọc được (AccessDenied) → để trống + ghi chú, **không ghi 0**.
- Exit code của công cụ = exit code **thật** của tiến trình con, kể cả mã NTSTATUS 0xC000013A. `sys.exit` của Python trên
  Windows làm tràn mã ≥ 2³¹ thành 0xFFFFFFFF, nên công cụ đổi sang int32 có dấu. `log.txt` kết thúc bằng `# exit=N`.
- stdout đọc theo **byte** và không bao giờ ngừng đọc trước EOF, nên tiến trình con không bị đứt pipe (sự cố vòng 35).
  - Log ghi UTF-8 (`errors=replace`), mỗi dòng có mốc giờ.
  - Lỗi ghi màn hình chỉ tắt phần in ra màn hình.
  - Mặc định thêm `PYTHONIOENCODING=utf-8` và `PYTHONUNBUFFERED=1` cho tiến trình con; giá trị được ghi lại.
- Ctrl+C / Ctrl+Break: công cụ ghi nhận tín hiệu và **không** thoát trước tiến trình con.
- Không ghi đè thư mục kết quả.
- `gan --pid N --create-time T`: gắn vào tiến trình có sẵn; bắt buộc có `create_time`; sai → exit 2.
- `trang-thai`: trả `dang_chay`, `ket_thuc` (kèm exit) hoặc `bi_ngat_khong_co_exit_code`. Dùng lệnh này thay cho việc đoán
  trạng thái từ một monitor đã hết hạn. **Monitor hết hạn không có nghĩa là training thất bại**, và không được tự khởi động
  thêm lượt.
- `tinh-lai`: tính lại tóm tắt từ `ram_mau_tho.csv` và đối chiếu `tom_tat.json`.

**Test** `src/test_chay_giam_sat.py`: **23/23 PASS** (`logs/review39/02_*`). Đây là tiến trình giả, không train.

| Ca | Kết quả |
|:--|:--|
| M1 | Shell giả mang tên `train_detector.py` cùng một Python khác đang chạy: PID / `create_time` bám đúng tiến trình con (khác PID shell giả); mọi mẫu của PID đó; RSS max 170 MiB (con cấp phát 160 MiB) |
| M2 | Worker tạo rồi kết thúc: `so_con` lên 2, rồi về 0; `chi_tiet_con` liệt kê từng worker |
| M3, M4, M5 | Exit 3, bị kill (15), NTSTATUS 0xC000013A: công cụ trả đúng mã |
| M6 | Ctrl+Break tới cả nhóm: công cụ ghi tín hiệu, thoát sau tiến trình con, mã đúng |
| M7 | 50.007 dòng, `\r`, byte hỏng, dòng cuối không newline: log đủ, không đứt pipe |
| M8 | `gan`: PID không tồn tại / `create_time` sai → exit 2, không ghi mẫu; PID kết thúc giữa chừng → dừng kèm lý do |
| M9 | Thư mục kết quả đã có nội dung → exit 2, không chạy lệnh |
| M10 | `trang-thai`: kết thúc / bị ngắt không có exit code / đang chạy |
| M11 | `tinh-lai` khớp; sửa `tom_tat` thì bị phát hiện |

## 5. Kiểm resume trên CPU — `src/test_tiep_tuc_huan_luyen.py` (ĐÃ CHẠY)

**Kế hoạch đăng ký trước**: `logs/review39/01_ke_hoach_kiem_resume.md`, gồm cách so, dự đoán theo mã và tiêu chí.
- "Trùng từng bit" nghĩa là: chuỗi CSV giống hệt (trừ cột thời điểm và số lần), tensor `torch.equal`.
- **Không có tolerance** nào.
- Có khai báo lượt thử trong scratchpad trước khi đăng ký.

Phụ lục sửa đổi của kế hoạch ghi năm lần chạy; **cả 5 log đều giữ**. Mỗi bổ sung có tiêu chí ghi **trước** lần chạy kế
tiếp.

| Lần | Kết quả | Nguyên nhân / sửa |
|:--|:--|:--|
| 1 | Thoát mã 2 ở đầu nhóm S | Lỗi của test: argv của spy cắt sai nên argparse gọi `SystemExit(2)`. Sửa test |
| 2 | 57/59 | K4 FAIL **chỉ ở mosaic**: phép đo sai, vì `YOLODataset` 8.4.141 không có thuộc tính `mosaic`. Sửa cách đọc thành `p` của transform Mosaic; thêm U6 |
| 3 | 60/60, exit 0 | — |
| 4 | 64/64, exit 0 | Bổ sung trước khi chạy: `best.pt` theo quỹ đạo, cắt `results.csv`; thêm U7a/U7b/U8/S19, mở rộng K7 |
| 5 | **68/68, exit 0** (lần chính thức cuối) | Bổ sung trước khi chạy, chỉ sửa test: R1b/K9, tiếp tục **sau** epoch tắt mosaic |

**Thiết lập (lần 5)**
- Cấu hình: `train_bdd5.yaml` chỉ đổi imgsz 160, workers 0, epochs 3. Fixture do exporter thật sinh: nb = 10 batch/epoch,
  nw = 10.
- Mỗi model **chạy thật** trên CPU:
  - C1, C2 liên tục 3 epoch;
  - I bị **kill cả cây tiến trình** ngay khi nhật ký ghi epoch 1; R `--tiep-tuc` I;
  - I2 bị kill ngay khi nhật ký ghi epoch 2; tiếp tục I2 (K9).
- Thời gian (s):

  | Model | C1 | C2 | I (bị kill sau) | R |
  |:--|--:|--:|--:|--:|
  | yolov8n | 20,6 | 37,4 | 16,5 | 14,6 |
  | yolo11n | 32,9 | 22,2 | 10,8 | 17,2 |

**Bảng liên tục ↔ ngắt / tiếp tục**

| Phép so | yolov8n | yolo11n |
|:--|:--|:--|
| K1 đối chứng C1 vs C2 (30 dòng; checkpoint e001–e003) | trùng từng bit | trùng từng bit |
| K2 I trước điểm ngắt vs C1 (10 dòng; checkpoint e001) | trùng từng bit | trùng từng bit |
| K3 runner tự kiểm sau resume; nguồn = e001; start_epoch 1 / tổng 3 | 13/13 đạt | 13/13 đạt |
| K4 R (epoch 2–3, 20 dòng) vs C1: epoch, ni, LR 3 nhóm, momentum 3 nhóm, batch | trùng từng giá trị | trùng từng giá trị |
| K4 mosaic thực tế (`p`) theo epoch | {1: 1,0, 2: 0, 3: 0}, C1 = I/R = dự đoán | như yolov8n |
| K5 accumulate + `ema.updates` từng batch vs mô phỏng `_do_train` | khớp ở C1 và R | khớp |
| K5 bước tối ưu sau epoch 1 | C1 [16, 23]; R [10, 18, 26] | như yolov8n |
| K5b accumulate sau warmup (dự đoán) | C1 7, R 8 | C1 7, R 8 |
| K6 loss khác / thứ tự dữ liệu khác (epoch 2–3) | 20/20 / 20/20 | 20/20 / 20/20 |
| K6 \|EMA cuối R − C1\| max | 0,373 | 0,397 |
| K7 chuỗi sự kiện, provenance, sao lưu, quỹ đạo `[[0, 1]]`, `best.pt` cùng byte `last.pt` (6 epoch ghi best) | đạt | đạt |
| K8 `hoan_tat` exit 0; `last.pt` cuối đã strip; checkpoint 5 lớp | đạt | đạt |
| K9 I2 tiếp tục **sau** epoch tắt mosaic: epoch 1–2 trùng C1 từng bit; 13/13 tự kiểm; `start_epoch` 2; epoch 3 trùng C1 ở ni/LR/momentum/batch; khớp mô phỏng (bước 20, 28); mosaic `p` {1: 1,0, 2: 0, 3: 0}; log có "Closing dataloader mosaic" từ nhánh `resume_training` | đạt | đạt |

- Warmup **không chạy lại** sau resume: LR hằng trong mỗi epoch 2–3 và trùng lượt liên tục.
- Fitness cuối của C1 và R đều là 0,0. Fixture là ảnh tổng hợp 160 px; con số này không có ý nghĩa đánh giá.

**Ca từ chối và ca đặc biệt** (spy, yolov8n). Mỗi ca từ chối thỏa đủ: đúng exit, **không gọi `train()`**, thư mục lượt
không đổi (so sha256 từng file), thông báo nêu đúng lý do.

| Ca | Exit |
|:--|:--|
| S1 lượt không tồn tại · S2 không có nhật ký (lượt smoke/pilot cũ) · S3 lượt đã hoàn tất · S3b đủ epoch nhưng mất dòng `ket_thuc` | 2 |
| S5 `last.pt` hỏng, không còn bản sao · S6 `last.pt` của lượt khác | 2 |
| S7a sửa 1 file nhãn (hợp đồng) · S7b bản xuất khác (vân tay) | 2 |
| S8 lr0 0,02 · S9 epochs 3 → 4 · S12 lượt mới trùng thư mục · S13 sai model · S14 nhật ký ghi DỪNG · S15 ultralytics khác | 2 |
| S10 thiếu `--xac-nhan-train-dai` · S11 `--che-do smoke` · S16 `--batch 4` | 1 |
| S4 `last.pt` bị cắt đôi → chọn bản sao e001; ultralytics nhận `nguon_tiep_tuc_1.pt` cùng sha | train được gọi (spy chặn) |
| S19 `best.pt` + dòng `results.csv` của nhánh bị bỏ → trước `train()`: `best.pt` = best e001, bản cũ được lưu, `results.csv` chỉ còn epoch 1 | train được gọi (spy chặn) |
| S17 `--dry-run --tiep-tuc`: in "tiếp tục từ epoch 2/3", không ghi gì | 0 |
| S18 Ctrl+C trong lúc train (spy ném `KeyboardInterrupt`) | 130 |

## 6. Kiểm resume + bộ giám sát trên GPU, dataset smoke (ĐÃ CHẠY)

Đây là kiểm phần mềm, không phải pilot: không dùng để so model hay chọn tham số. Script:
`handoff/logs/review39/kiem_resume_gpu_smoke.py`. Tiêu chí đăng ký trước ở `01_*` mục 3.

**Thiết lập**
- Cấu hình `cau_hinh_kiem_resume_3epoch.yaml` = `train_bdd5.yaml`, **chỉ đổi** epochs 15 → 3. Dataset smoke đã khóa
  (334 / 84 ảnh).
- `--device 0`, batch 8, imgsz 640, workers 2, AMP tắt. nb = 42, nw = 42.
- Mỗi model chạy tuần tự C (liên tục 3 epoch), I (kill sau epoch 1), R (`--tiep-tuc` I).
- **Mọi lượt chạy qua `src/chay_giam_sat.py chay`**. Với I, script kill **PID runner đọc từ `trang_thai.json` của công cụ**
  cùng các tiến trình con.
- Thư mục mới:
  - `models/finetune_phase2/kiem_resume_v39/`, `results/phase2/kiem_resume_v39/` (không commit);
  - bằng chứng: `logs/review39/05_gpu/` (log, CSV RAM, tóm tắt của công cụ; bản chép nhật ký, theo batch, results, JSON
    runner; không có `.pt`) và `06_kiem_resume_gpu.{log,json}`.

**Kết quả: cả hai model ĐẠT 21/21 tiêu chí; exit 0** (`06_kiem_resume_gpu.json`, `danh_gia`).

| Phép so | yolov8n | yolo11n |
|:--|:--|:--|
| Thời gian C / I (bị kill sau) / R | 169 / 66 / 127 s | 193 / 79 / 141 s |
| I: kill PID runner + tiến trình con ngay sau epoch 1 | PID 12432 + 6 con; công cụ exit 15 | PID 15008 + 6 con; exit 15 |
| K3 runner tự kiểm sau resume; nguồn = e001 | 13/13 đạt | 13/13 đạt |
| K4 R vs C (84 dòng): epoch, ni, LR 3 nhóm, momentum 3 nhóm, batch | trùng từng giá trị | trùng từng giá trị |
| K4 mosaic `p` theo epoch | {1: 1,0, 2: 0, 3: 0}, C = I/R = dự đoán | như yolov8n |
| K5 accumulate + `ema.updates` vs mô phỏng | khớp ở C và R | khớp |
| K5 bước tối ưu sau epoch 1 | C 44, 52, …, 124; R 42, 50, …, 122; accumulate 8 = 8 | như yolov8n |
| K7 chuỗi sự kiện nhật ký; K8 `hoan_tat`, exit 0, checkpoint 5 lớp | đạt | đạt |
| *K2 (chỉ báo)*: I trước điểm ngắt vs C, epoch 1 | **lệch** 23/42 dòng, từ ni 19; checkpoint e001 khác | trùng 42/42; e001 trùng |
| *K6 (chỉ báo)*: loss khác / thứ tự dữ liệu khác | 84/84 / 84/84 | 84/84 / 84/84 |
| *K6 (chỉ báo)*: \|EMA cuối R − C\| max; fitness cuối C / R | 1,29; 0,134 / 0,126 | 0,408; 0,117 / 0,123 |

Diễn giải:
- **K2 trên GPU**: hai lượt cùng cấu hình (C và I trước điểm ngắt) đã khác nhau từ batch ni 19 với yolov8n, dù đã bật
  `deterministic`. GPU **không** tất định ổn định: yolo11n lần này lại trùng. Vì vậy, trên GPU không tách được phần khác
  biệt do resume khỏi phần do GPU. Phép so tách bạch là phép trên CPU (mục 5: K1, K2 trùng từng bit).
- **K6**: fitness cuối lệch theo hai chiều ngược nhau ở hai model. Với 3 epoch trên 334 ảnh, đây không phải bằng chứng về
  ảnh hưởng của resume lên chất lượng model; không rút kết luận.
- Bản sao checkpoint mỗi epoch (có optimizer): 12,3 MB (yolov8n), 10,7 MB (yolo11n). Với 15 epoch là khoảng 185 / 160 MB.

**Bằng chứng bộ giám sát bám đúng PID** (6/6 lượt; tính lại độc lập từ CSV ở `07_tinh_lai_ram_doc_lap.*`)

| Lượt | PID công cụ bám = PID runner tự ghi (nhật ký) | Exit | Mẫu / khoảng đo | RSS chính max | RSS tổng max | RAM trống toàn máy min |
|:--|:--|--:|:--|--:|--:|--:|
| yolov8n C | 24984 = 24984 | 0 | 164 / 167,4 s | 1.169 MiB | 2.945 MiB | 17 MiB |
| yolov8n I | 12432 = 12432 | 15 | 64 / 64,9 s | 1.277 | 3.086 | 45 |
| yolov8n R | 11340 = 11340 | 0 | 124 / 126,4 s | 1.263 | 3.394 | 101 |
| yolo11n C | 23488 = 23488 | 0 | 188 / 192,0 s | 1.195 | 3.533 | 163 |
| yolo11n I | 15008 = 15008 | 15 | 77 / 78,0 s | 1.123 | 3.236 | 263 |
| yolo11n R | 1328 = 1328 | 0 | 137 / 139,8 s | 1.393 | 3.865 | 33 |

- Mọi dòng CSV của mỗi lượt chỉ có đúng một PID. PID đó bằng PID trong `tom_tat.json` và bằng PID runner tự ghi bằng
  `os.getpid()` trong sự kiện `bat_dau` của nhật ký lượt, tức là một nguồn thứ ba độc lập với công cụ.
- `t_unix` tăng dần; không ô nào rỗng hay bằng 0. Mọi số trong bảng **tính lại bằng mã riêng** từ CSV (không dùng hàm tóm tắt
  của công cụ) và khớp `tom_tat.json`: **60/60 điều kiện** (`07_tinh_lai_ram_doc_lap.{py,log}`). `tinh-lai` của công cụ
  cũng khớp ở cả 6 lượt.
- Tiến trình con thấy được:
  - 6 worker Python: train dùng `workers` 2, val dùng `workers × 2` = 4 (đọc mã `DetectionTrainer.get_dataloader`), mỗi
    worker khoảng 320–430 MiB RSS;
  - `nvidia-smi` của bộ đo VRAM trong runner;
  - đôi khi một `cmd.exe` ngắn.
- **RSS tổng đếm trùng** trang chia sẻ (thư viện torch nạp ở mỗi worker), nên **không** phải bộ nhớ thực dùng. Chỉ số áp
  lực đáng tin là **RAM trống toàn máy**: xuống tới **17 MiB** (yolov8n C, giây 13, lúc tạo worker). Khi đó các ứng dụng
  khác vẫn mở, và lúc bắt đầu kiểm máy còn trống 1,8 GB. Không có lỗi hết bộ nhớ.

## 7. Suite bị ảnh hưởng trực tiếp (ĐÃ CHẠY)

Runner đã đổi, nên chạy lại hai suite dùng nó. Không chạy lại tracking / Nano / full suite.

| Suite | Kết quả | Log |
|:--|:--|:--|
| `src/test_chay_giam_sat.py` (mới) | 23/23 PASS | `logs/review39/02_*` |
| `src/test_tiep_tuc_huan_luyen.py` (mới) | lần 5: 68/68 PASS (lần 1–4 giữ, mục 5) | `logs/review39/03_*_lan1..5` |
| Kiểm GPU dataset smoke (2 model) | ĐẠT 21/21 tiêu chí mỗi model | `logs/review39/05_gpu/`, `06_*` |
| Tính lại RAM độc lập từ CSV | 60/60 điều kiện | `logs/review39/07_*` |
| `src/test_detector_5lop.py` | 92/92 PASS (vòng 37: 92/92) | `logs/review39/04_test_detector_5lop.log` |
| `src/test_pilot_chan_doan.py` | 34/34 PASS (vòng 37: 34/34) | `logs/review39/04_test_pilot_chan_doan.log` |

Hai test mới đã được đăng ký trong `src/run_all_tests.py`.

## 8. Khởi chạy trong terminal riêng (đề xuất — training dài CHƯA được duyệt)

- Chạy từ gốc repo `D:\LABOTARY\bytetrack`, trong một cửa sổ PowerShell hoặc cmd **riêng**.
- **Không** chạy qua tác vụ nền của trợ lý. Tác vụ nền giới hạn khoảng 2 giờ, và một tác vụ hay monitor hết hạn **không có
  nghĩa là training thất bại**.
- `$PY` là đường dẫn tuyệt đối tới Python của env, để tiến trình con không phụ thuộc PATH.
- Mỗi lần chạy dùng một thư mục giám sát **mới**; công cụ từ chối ghi đè.

**PowerShell — lượt mới**

```powershell
cd D:\LABOTARY\bytetrack
$PY = "C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe"
& $PY src\train_detector.py --model yolov8n --dry-run          # kiem truoc, KHONG train, KHONG ghi gi
& $PY src\chay_giam_sat.py chay --thu-muc results\phase2\detector_train\giam_sat\yolov8n_lan0 -- `
    $PY -u src\train_detector.py --model yolov8n --che-do day-du --xac-nhan-train-dai
$LASTEXITCODE
```

Ý nghĩa `$LASTEXITCODE`:
- 0: hoàn tất;
- 1: sai cờ;
- 2: từ chối trước khi train;
- 3: lỗi;
- 130: Ctrl+C. Runner ghi kết quả dở dang và `ket_thuc bi_ngat`; công cụ chờ runner thoát rồi trả đúng mã.

Nếu cửa sổ bị đóng hoặc máy mất điện thì không còn exit code. Khi đó dùng `trang-thai`, lệnh sẽ trả
`bi_ngat_khong_co_exit_code`.

**PowerShell — sau gián đoạn** (mất điện, đóng cửa sổ, Ctrl+C, máy khởi động lại)

```powershell
# a) trang thai lan chay truoc: dang_chay | ket_thuc (exit) | bi_ngat_khong_co_exit_code -- khong doan tu log
& $PY src\chay_giam_sat.py trang-thai --thu-muc results\phase2\detector_train\giam_sat\yolov8n_lan0
# b) kiem dieu kien tiep tuc: epoch se tiep tuc, checkpoint chon, ung vien bi loai, quy dao, best.pt -- KHONG train
& $PY src\train_detector.py --model yolov8n --dry-run --tiep-tuc
# c) tiep tuc (thu muc giam sat MOI)
& $PY src\chay_giam_sat.py chay --thu-muc results\phase2\detector_train\giam_sat\yolov8n_tiep_tuc_1 -- `
    $PY -u src\train_detector.py --model yolov8n --che-do day-du --xac-nhan-train-dai --tiep-tuc
$LASTEXITCODE
```

**cmd**: thay đổi so với PowerShell:
- đặt biến bằng `set PY=C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe`, gọi bằng `%PY%`, bỏ `& `;
- nối dòng bằng `^` thay cho `` ` ``;
- xem mã thoát bằng `echo %ERRORLEVEL%`.

**Đầu ra**

- **Thư mục giám sát**:
  - `log.txt`: mỗi dòng có mốc giờ; dòng cuối `# exit=<mã>`;
  - `ram_mau_tho.csv`, `tom_tat.json`, `trang_thai.json`.

  Tính lại tóm tắt RAM từ CSV: `& $PY src\chay_giam_sat.py tinh-lai --thu-muc <thư mục>`.
- **Thư mục lượt** `models\finetune_phase2\train_bdd5_v1_<M>_day_du\`:
  - `nhat_ky_luot.jsonl`, `theo_batch.csv`;
  - `khoi_phuc\last_eNNN.pt`: mỗi epoch một bản, cỡ bằng `last.pt` có optimizer (đo ở mục 6);
  - `weights\last.pt`, `weights\best.pt`.
- **Kết quả runner**: `results\phase2\detector_train\train_bdd5_v1_<M>_day_du.json`. Lần tiếp tục thứ N ghi
  `…__tiep_tuc_N.json`, không ghi đè.
- Chạy YOLO11n **sau khi** YOLOv8n kết thúc với exit 0. Không chạy đồng thời hai model.

## 9. Trạng thái sẵn sàng và quyết định chờ người dùng

**Về kỹ thuật: SẴN SÀNG** để chạy lượt dài khi được duyệt.
- Resume (lượt mới, tiếp tục, các ca từ chối) và bộ giám sát đã cài.
- Đã kiểm qua entry point thật trên fixture CPU (68/68) và trên đúng phần cứng, cấu hình, dataset smoke (GPU, cả hai model,
  ĐẠT).
- Các suite bị ảnh hưởng PASS.

**Về quyết định: CHƯA SẴN SÀNG**, vì lượt dài chưa được duyệt. Vòng này **chưa** được Codex nghiệm thu; base vẫn là
`b83dc12`.

Quyết định còn chờ:
1. **Số epoch** (đề xuất 15) và **ngân sách** training local. Kịch bản vòng 35/38 là 56,64 h, kịch bản chậm hơn 73,95 h, cho
   hai model tuần tự; đây là kịch bản theo giả định, không phải bảo đảm.
2. **Thời điểm bắt đầu**, chạy trong terminal riêng (mục 8), và có đóng bớt ứng dụng hay không. Lúc kiểm, RAM trống còn
   1,8 GB và xuống tới 17 MiB khi runner tạo 6 worker. Giảm `workers` sẽ đổi cấu hình đã khóa, nên cần duyệt riêng; vòng
   này không đổi.
3. Thứ tự hai model (đề xuất YOLOv8n trước) và thư mục giám sát cho lượt dài.
4. Có sửa trường `mosaic_thuc_te` trong `pilot_chan_doan.py` hay không (mục 10). Vòng này không sửa.

## 10. Chưa chạy, giới hạn

- **NOT_RUN**:
  - training dài; mọi lần chạy trên dataset đầy đủ;
  - pilot;
  - Nano, tracking, full suite.
- Resume không trùng từng bit với lượt liên tục (mục 3.6). Độ lớn khác biệt chỉ đo trên fixture CPU và smoke GPU 3 epoch;
  không suy ra ảnh hưởng lên chất lượng model của lượt dài.
- Kiểm GPU dùng dataset smoke 334 / 84 ảnh và 3 epoch; nó không đo được hành vi của 15 epoch.
- Nhánh `resume_training` tự tắt mosaic khi tiếp tục sau epoch tắt mosaic: **đã chạy trên CPU** (K9), **chưa chạy trên
  GPU**. Trên GPU, điểm tiếp tục trùng đúng epoch tắt mosaic, nên việc tắt xảy ra trong vòng lặp.
- Ngắt được mô phỏng bằng kill tiến trình (Windows trả mã 15 qua `TerminateProcess` của psutil). Chưa mô phỏng mất điện
  thật, tức cache đĩa chưa kịp ghi xuống; bù lại có chép nguyên tử, fsync và chọn bản sao khi file hỏng (S4).
- **Ctrl+C thật** trong lúc train thật (với worker DataLoader) **chưa kiểm**. S18 dùng spy ném `KeyboardInterrupt`; M6
  dùng tiến trình giả.
- Trên GPU, hai lượt cùng cấu hình có thể không trùng (K2 yolov8n). Khác biệt do resume chỉ tách bạch được trên CPU.
- **Sự cố trong vòng**: lúc kiểm GPU đang chạy, 4 file tài liệu (`REVIEW38_CHI_TIET.md`, `DETECTOR_TRAIN_PLAN.md`,
  `PLAN_DA_CHOT.md`, `CODEX_HANDOFF.md`) bị ghi.
  - Nguyên nhân: bước biến đổi script bằng heredoc hỏng; lệnh kế tiếp nằm ở dòng riêng, không nối `&&`, nên vẫn chạy bản
    script cũ.
  - Ảnh hưởng: chỉ là tài liệu, nội dung đúng như dự định. Runner chỉ ghi sha 4 file `src/` và git HEAD, nên bằng chứng
    GPU không bị ảnh hưởng.
  - Không commit cho tới khi kiểm GPU xong; ghi rõ trong commit `2ff9899`.
- **Phát hiện về vòng 37**:
  - trường `mosaic_thuc_te` trong JSON pilot đọc thuộc tính không tồn tại, nên luôn `False` và **không có giá trị đo**;
  - kết luận "mosaic tắt cả 2 epoch" của pilot vẫn có bằng chứng: log "Closing dataloader mosaic" xuất hiện đúng 1 lần mỗi
    log pilot, ngay epoch đầu, cộng với mã.
- Checkpoint smoke / pilot cũ đã strip optimizer: giữ nguyên, không resume được (S2), không tạo metadata hồi tố.

## 11. Commit vòng 39

| Commit | Nội dung |
|:--|:--|
| `53dfc22` | Ghi nhận nghiệm thu `b83dc12`, bằng chứng, base → `b83dc12` |
| `231006c` | Mã resume + bộ giám sát + test; kế hoạch kiểm đăng ký trước; log test giám sát 23/23 |
| `26fdc81` | Sửa test (argv spy), log lần 1 |
| `53a328f` | Sửa phép đo mosaic + U6, log lần 2 (57/59) |
| `bfd1e40` | Log lần 3 (60/60) |
| `2f9eee4` | `best.pt` theo quỹ đạo + cắt `results.csv` + test mới; phụ lục kế hoạch |
| `4e0f80d` | Log lần 4 (64/64) |
| `c183a12` | Log suite bị ảnh hưởng (92/92, 34/34) |
| `eca9008` | Kiểm GPU dataset smoke, hai model (ĐẠT) |
| `2ff9899` | Đính chính R38-N1/N2; kế hoạch, plan, handoff (ghi chú sự cố) |
| `2625997` | Test thêm R1b/K9 (chỉ sửa test); phụ lục kế hoạch |
| `b00d3fd` | Log lần 5 (68/68) |
| `04528c0` | Tính lại RAM độc lập từ CSV (60/60) |
| commit chứa phụ lục này (target delta) | Brief, phụ lục, handoff hàng 35 |
