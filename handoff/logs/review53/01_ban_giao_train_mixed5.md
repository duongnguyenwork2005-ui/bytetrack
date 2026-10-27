# R53A — nghiệm thu bàn giao hai detector mixed5 (YOLOv8n, YOLO11n)

Ngày 19/10/2026. Thực hiện bước `R53A_HANDOFF` của plan R53 (người dùng duyệt). Mục 0 của `handoff/PLAN_DA_CHOT.md` đã
được cập nhật trước báo cáo này. Bước kết thúc tại cổng review Codex, **chưa sang R53B**.

**Vòng 2 (19/10 23:34).** Review vòng 1 của Codex trả `revise` với hai lỗi R53A-F1 và R53A-F2. Cả hai đã sửa, xem mục 11.
Các số liệu bàn giao ở mục 2–7 không đổi.

## 1. Phạm vi và cách làm

- Chỉ đọc bằng chứng train sẵn có. Không train, không resume, không inference, không GPU, không chạy lại validation.
- Không nạp model bằng torch/ultralytics. Metadata checkpoint (`epoch`, `optimizer`, `train_metrics`, `date`, `names`,
  `train_args`) được đọc thẳng từ `data.pkl` trong file `.pt` (định dạng zip). Bộ unpickle hạn chế chỉ dựng
  dict/list/số; mọi lớp hay hàm khác được thay bằng đối tượng giả và **không** được gọi (có test riêng).
- Không sửa checkpoint, log, nhật ký hay dataset. Mọi file đầu vào được băm sha256 và liệt kê trong `nguon_sha256.csv`.
  Hash của 4 checkpoint cuối trùng hash runner ghi lúc kết thúc train (mục 4), nên các checkpoint này không đổi kể từ đó.
- Mã mới:
  - `src/r53_detector/ban_giao_train.py`: thư viện chuẩn, đặt UTF-8, tắt bytecode trước mọi import. Vòng 1 sha256
    `b8d9d9dd…a0a1`; sau sửa F2 sha256 `8cc69c871f26d2eeb87554089031c0aaf79ed245d572ff699d5da27c6304e56b`.
  - `src/r53_detector/test_ban_giao_train.py`. Vòng 1 sha256 `4d212f33…5181`; sau sửa F1 sha256
    `a818b937c7db9e1eee45f05b4171885200493953fe74f24015a39d7e390e9686`.

### Lệnh đã chạy (cwd = gốc repo)

Vòng 1 (18:58):

| Lệnh | Exit | Kết quả |
|:--|:-:|:--|
| `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/test_ban_giao_train.py` | 0 | 12/12 test đạt (`results/phase2/r53_detector/tests_handoff/ket_qua_test.json`) |
| `… src/r53_detector/ban_giao_train.py --model yolov8n` | 0 | `[DAT]`, 0 lỗi, 0 cảnh báo, 0 mục thiếu dữ liệu |
| `… src/r53_detector/ban_giao_train.py --model yolo11n` | 0 | `[DAT]`, 0 lỗi, 0 cảnh báo, 0 mục thiếu dữ liệu |

Vòng 2, sau sửa F1/F2 (theo thứ tự chạy; mọi thư mục `lan_*` nằm dưới `results/phase2/r53_detector/tests_handoff/`):

| Lệnh | Exit | Kết quả |
|:--|:-:|:--|
| `… test_ban_giao_train.py` (bản giữa, trước khi sửa `tao_thu_muc_lan`) | 1 | 14/15. `test_chay_lai_khong_xoa_fixture_cu` báo `FileExistsError`: hai lần gọi liên tiếp sinh cùng tên thư mục, vì đồng hồ Windows có độ phân giải thô. Cổng không tái dùng thư mục đã chặn đúng. Đã sửa bằng hậu tố `_k`. Bằng chứng giữ trong `lan_20261019_233411_798186_pid11804/` |
| `… test_ban_giao_train.py` | 0 | 15/15; kiểm nguyên 515 file có trước. `lan_20261019_233421_218114_pid24184/ket_qua_test.json` |
| `… test_ban_giao_train.py` (chạy lại, mã không đổi) | 0 | 15/15; kiểm nguyên 816 file có trước, gồm cả lần chạy 15/15 ngay trước. `lan_20261019_233426_638785_pid11772/ket_qua_test.json` |
| `… ban_giao_train.py --model yolov8n` | 0 | `[DAT]`, 0 lỗi / 0 cảnh báo / 0 thiếu; best e15, 15 epoch, quỹ đạo như mục 3 |
| `… ban_giao_train.py --model yolo11n` | 0 | `[DAT]`, 0 lỗi / 0 cảnh báo / 0 thiếu; best e13, 15 epoch |

Trước lần chạy chính thức có hai lần chạy thử, ghi vào `results/phase2/r53_detector/tests_handoff/thu_nghiem_som/`:

- Lần thử đầu (yolo11n) báo lỗi `best.pt: train_results ket thuc o epoch 15 != 13`. Nguyên nhân: `final_eval` của
  Ultralytics chép `train_results` của last.pt sang best.pt, nên trường này không xác định được epoch của best.pt. Đã đổi
  sang đối chiếu `train_metrics` và `date` (mục 3).
- Sau khi sửa, cả hai model chạy thử đều `[DAT]`, với cùng hash mã như lần chạy chính thức. Thư mục chạy thử còn lại
  trên đĩa (không có lệnh xóa trong quyền được cấp). Đây không phải đầu ra bàn giao.

### Test (fixture tổng hợp, không đọc dữ liệu thật)

Fixture có 3 epoch và 1 lần resume, cùng bố cục thư mục với repo. Vòng 1 đặt fixture dưới `tests_handoff/fixture/`. Từ
vòng 2, fixture nằm dưới `tests_handoff/lan_<thoi diem>_pid<pid>/fixture/`.

| Test | Kiểm |
|:--|:--|
| `test_hop_le` | Fixture sạch: đạt, 0 lỗi/cảnh báo. Quỹ đạo `[[0,1],[0,2],[1,3]]`. Best e3 xác định từ train_metrics và date. Peak toàn lượt là max chứ không cộng. Phát hiện khoảng mất mẫu 101 s. final_eval tách khỏi val. Ghi đủ 5 file đầu ra |
| `test_thieu_epoch` | Bỏ e2 → lỗi `thieu epoch [2]` và quỹ đạo không liên tục; mục 8 epoch không đạt |
| `test_trung_epoch_sau_resume` | Lan 1 ghi lại e2 → lỗi `trung/xung dot epoch sau resume {2: [0, 1]}` |
| `test_start_epoch_resume_lech` | `start_epoch` của lần resume lệch số epoch đã lưu → lỗi |
| `test_hash_lech` | Sửa 1 byte `last_e001.pt` và `best.pt` → lỗi hash so với nhật ký, JSON runner và kỳ vọng plan |
| `test_ket_thuc_loi` | `ket_thuc` = `loi`, exit 1, log `[LOI]` → 3 điều kiện mục 8 không đạt |
| `test_thieu_ket_thuc_lan_cuoi` | Lần cuối không có `ket_thuc` → lỗi |
| `test_thieu_du_lieu_tai_nguyen` | Thiếu `ram_mau_tho.csv` cả hai lần và CSV nvidia-smi lan 0 → ghi rõ trong `thieu_du_lieu`. Ô CSV để trống chứ không ghi 0. Cực trị toàn lượt chỉ lấy từ lần có dữ liệu |
| `test_results_csv_lech` | results.csv lệch nhật ký → lỗi |
| `test_best_lech_ky_vong` | Best khác kỳ vọng → lỗi |
| `test_thieu_dong_DAT` | Log giám sát không có `[DAT]` → mục 8 không đạt |
| `test_unpickle_khong_goi_ham_la` | Pickle có `__reduce__` trỏ `os.getcwd` → nhận đối tượng giả, hàm không được gọi. File không phải zip → `doc_duoc=False` |
| `test_hop_le` (bổ sung vòng 2) | Ghi lại đầu ra vào cùng thư mục → bản cũ được chép sang `_ban_truoc/<thoi diem>/`, giống từng byte (so hash 5 file) |
| `test_chay_lai_khong_xoa_fixture_cu` (mới, F1) | Mô phỏng hai lần chạy: fixture `hop_le` của lần A còn nguyên hash sau khi lần B tạo fixture cùng tên. Tạo lại cùng tên trong cùng lần → `FileExistsError`, không xóa |
| `test_out_ngoai_vung_bi_tu_choi` (mới, F2) | `kiem_out` từ chối 10 đường dẫn: gốc `r53_detector/`, `chuan_bi/`, `smoke/`, `full/…`, thư mục anh em cùng tiền tố `ban_giao_khac`, `tests_handoff_x`, `ban_giao/../smoke`, `tests_handoff/../chuan_bi`, `results/`, ngoài repo. Chấp nhận `ban_giao/`, `tests_handoff/` và thư mục con. `ghi_dau_ra` với out ngoài vùng → `ValueError`, không tạo thư mục. CLI thật với `--out` vào `smoke/`, `chuan_bi/`, `full/` hoặc gốc `r53_detector/`: exit 2, stdout rỗng, không tạo thư mục hay file. Danh sách mục trong `r53_detector/` không đổi |
| `test_giu_bang_chung_luot_truoc` (mới, F1, chạy cuối) | Băm lại mọi file có trong `tests_handoff/` trước lần chạy: không mất, không đổi. Mọi file mới chỉ nằm trong thư mục `lan_*` của lần đó |

## 2. Mục 8 (kế hoạch 02): điều kiện kết thúc thành công

Áp dụng cho lần chạy cuối của mỗi model. Chi tiết nằm ở `ban_giao_<model>.json` → `muc_8`.

| Điều kiện | YOLOv8n (lan 3) | YOLO11n (lan 0) |
|:--|:--|:--|
| Bộ giám sát exit 0 (`tom_tat`/`trang_thai`/`# exit=` cuối log) | đạt: 0 / ket_thuc / 0 | đạt: 0 / ket_thuc / 0 |
| Runner `[DAT] day-du`, `loi=None`, `loi_callback=None`, loss hữu hạn (log + JSON + mọi epoch) | đạt | đạt |
| last/best nạp được, 5 lớp, tham số trùng cấu hình | đạt | đạt |
| Nhật ký `ket_thuc` `hoan_tat`, `kiem_sau_train_dat=true`, exit 0 | đạt (18/10 04:30:05) | đạt (19/10 16:52:37) |
| Đủ 15 epoch | đạt, 15 | đạt, 15 |

Về điều kiện nạp lại: bằng chứng chính là kiểm tra runner thực hiện sau train (`checkpoint.{best,last}.nap_lai`,
`dung_5_lop`, `train_args_lech_cau_hinh = {}`). R53A không nạp lại bằng torch, chỉ đọc lại metadata. Kết quả đọc lại:
`names` = car / truck / bus / motorcycle / bicycle, không có optimizer, `epoch = -1`, `train_args` trùng `train_kwargs`
của runner.

## 3. Epoch, quỹ đạo resume và best

- **YOLOv8n:** lan 0 e1–7 → lan 1 e8–10 → lan 2 e11 → lan 3 e12–15. Ba lần resume đều sau Ctrl+C có chủ đích theo
  yêu cầu người dùng (`bi_ngat`, exit 130). Mỗi lần resume khớp các điểm sau:
  - `start_epoch` bằng số epoch đã lưu;
  - checkpoint nguồn bằng bản sao epoch cuối của lần trước;
  - `nguon_tiep_tuc_N.pt` khớp hash, 3/3;
  - `kiem_khoi_phuc` có mọi mục `dat`.
- **YOLO11n:** chạy 1 lần e1–15, không resume.
- Mỗi model đúng 15 epoch, không trùng. results.csv có 15 hàng, khớp nhật ký và JSON runner trên 7 cột metric
  (dung sai 1e-5). Cột `time` của results.csv cộng dồn trong từng lần chạy, nên không dùng làm thời gian epoch.

| Model | Best | mAP50 / mAP50-95 val epoch | Bằng chứng |
|:--|:-:|:--|:--|
| YOLOv8n | **e15** | 0,49254 / 0,33452 | (1) fitness (= mAP50-95) max ở e15; (2) sự kiện cuối có `best_ghi_epoch_nay=true` là e15; (3) `train_metrics` của best.pt chỉ khớp e15, và `date` 18/10 04:29:09 ứng với sự kiện lưu e15 |
| YOLO11n | **e13** | 0,49640 / 0,34021 | (1) fitness max ở e13 (e14 0,33729, e15 0,33816); (2) sự kiện ghi best cuối là e13; (3) `train_metrics` của best.pt chỉ khớp e13, và `date` 19/10 12:25:37 ứng với sự kiện lưu e13 |

## 4. Checkpoint và sha256 (`checkpoint_sha256.csv`)

| Model | File | sha256 thực (= JSON runner) | Bytes | Ghi chú |
|:--|:--|:--|--:|:--|
| YOLOv8n | `weights/best.pt` | `542b1381c33a673531bccdbca769f26408a16e9ba6ed89e49238a0f16b9b0694` | 6.222.691 | = hash plan nêu; e15 |
| YOLOv8n | `weights/last.pt` | `35b2ff822f8e1d5019464b06e5fdf7391ab9bd1e1909af038d41c68338738598` | 6.222.691 | e15 |
| YOLO11n | `weights/best.pt` | `9a5e34a12c1d97cd1b0e49325d2eadbd443a965633b464461980df91fea386fb` | 5.445.075 | = hash plan nêu; e13 |
| YOLO11n | `weights/last.pt` | `db330f8f36f817414afb1316514cdc324e2e755ecbac25d6f3bb3f6fb1983f57` | 5.445.075 | e15 |

- Bốn file trên là **bản cuối đã bỏ optimizer** (`strip_optimizer`, `epoch = -1`, không có EMA/optimizer).
- **Bản lưu trong epoch** (`khoi_phuc/last_eNNN.pt`, còn optimizer, `epoch` 0-based) là file khác, hash khác. Ví dụ
  best e13 của YOLO11n lúc ghi có hash `d2def41b…e117`, best e15 của YOLOv8n có hash `7e6f3492…cc51`.
  30/30 bản lưu trong epoch khớp `ban_sao_sha256` và `last_sha256` của nhật ký, đồng thời khớp `nhat_ky_epoch` của
  JSON runner.
- Quan sát, chưa giải thích: với YOLOv8n, best.pt và last.pt cùng là e15, cùng kích thước và cùng `date`, nhưng hash
  khác nhau. R53A không suy ra nguyên nhân. Hai file được strip riêng; checkpoint dùng cho R53B là best.pt với hash ở bảng trên.

## 5. Kết quả dev: val từng epoch và validation cuối (tách riêng)

- **Val từng epoch:** `dev_tung_epoch.csv`, 30 hàng. Lấy từ nhật ký lượt và đối chiếu results.csv cùng JSON runner.
  `val_epoch_s` là thời gian val trong vòng train, lấy từ JSON runner. Tổng val epoch: 1.073,6 s (YOLOv8n) và
  1.054,1 s (YOLO11n).
- **Validation cuối** `danh_gia_cuoi_ultralytics`: Ultralytics `final_eval` validate best.pt FP16 nạp từ đĩa.

| Model | P | R | mAP50 | mAP50-95 | Khoảng sau train epoch cuối |
|:--|--:|--:|--:|--:|--:|
| YOLOv8n | 0,61439 | 0,45204 | 0,49255 | 0,33455 | 118,47 s |
| YOLO11n | 0,62637 | 0,45552 | 0,49646 | 0,34015 | 138,71 s |

- Cột cuối **không phải thời gian thuần validation**. Khoảng này gồm lưu checkpoint, dọn dẹp cuối epoch và validate
  best.pt.
- Chênh lệch so với val epoch best: YOLOv8n +0,000009 mAP50 / +0,000032 mAP50-95; YOLO11n +0,000063 / −0,000062.
  Phần này chỉ để mô tả: val trong epoch dùng EMA FP32 trong bộ nhớ, còn final_eval dùng file FP16.
- Đây là dev nội bộ của mixed5, không phải kết quả đánh giá R53. Không dùng để xếp hạng.

## 6. Lịch sử chạy, thời gian và tạm dừng

| Model | Lan | Giám sát bắt đầu → kết thúc | Thời lượng (s) | Kết thúc | Epoch | Tạm dừng trước lan (s) |
|:--|:-:|:--|--:|:--|:--|--:|
| YOLOv8n | 0 | 16/10 18:32:42 → 17/10 07:40:22 | 47.260,3 | bi_ngat, exit 130 | 1–7 | — |
| YOLOv8n | 1 | 17/10 08:16:50 → 14:02:37 | 20.746,8 | bi_ngat, exit 130 | 8–10 | 2.188 |
| YOLOv8n | 2 | 17/10 17:18:19 → 19:16:25 | 7.086,7 | bi_ngat, exit 130 | 11 | 11.742 |
| YOLOv8n | 3 | 17/10 20:59:16 → 18/10 04:30:07 | 27.051,1 | hoan_tat, exit 0 | 12–15 | 6.171 |
| YOLO11n | 0 | 18/10 04:33:47 → 19/10 16:52:40 | 130.733,1 | hoan_tat, exit 0 | 1–15 | — |

**Toàn lượt.**

- YOLOv8n: đồng hồ 122.245 s. Tổng thời lượng chạy 102.144,9 s, tạm dừng giữa các lần 20.101 s. Tổng train epoch
  99.803 s.
- YOLO11n: đồng hồ 130.733 s, không tạm dừng. Tổng train epoch 129.397 s.
- Ba lần ngắt của YOLOv8n mất phần epoch dở sau lần lưu cuối: 69,9 / 71,0 / 76,1 s, tính từ sự kiện lưu cuối đến
  `ket_thuc` (`mat_sau_epoch_luu_cuoi_s`). Ở lần hoàn tất, khoảng này là 55,6 s (YOLOv8n) và 68,5 s (YOLO11n), gồm
  final_eval và kiểm tra sau train.

**Mất mẫu trong lần chạy** (khoảng > 60 s giữa hai mẫu RAM, chu kỳ 1 s):

- YOLOv8n: 0 khoảng.
- YOLO11n: 1 khoảng 10.692 s, 18/10 19:08:31 → 22:06:43, khớp lần máy ngủ do mất điện ghi ở mục 0. CSV nvidia-smi
  cũng mất mẫu 10.714 s cùng lúc. Vì vậy `train_s` của e7 YOLO11n là 18.913 s, so với khoảng 7.900 s ở các epoch khác.

## 7. Tài nguyên (`tai_nguyen.csv`; cực trị toàn lượt lấy max/min qua các lần, **không cộng peak**)

| Model | Phạm vi | RSS PID max (MiB) | RSS PID + con max | Số con max | RAM máy còn trống min | VRAM nvidia-smi max | torch alloc / reserved max |
|:--|:--|--:|--:|--:|--:|--:|--:|
| YOLOv8n | lan 0 | 1.215,8 | 4.078,1 | 8 | 16 | 2.425 | 1.746 / 2.152 |
| YOLOv8n | lan 1 | 1.576,1 | 4.143,8 | 8 | 6 | 2.815 | 1.728 / 2.166 |
| YOLOv8n | lan 2 | 1.538,9 | 3.943,1 | 7 | 10 | 2.380 | 1.723 / 2.132 |
| YOLOv8n | lan 3 | 1.456,8 | 4.308,6 | 7 | 15 | 2.361 | 1.728 / 2.160 |
| YOLOv8n | toàn lượt | 1.576,1 (lan 1) | 4.308,6 (lan 3) | 8 | 6 (lan 1) | 2.815 (lan 1) | 1.746 / 2.166 |
| YOLO11n | lan 0 = toàn lượt | 2.361,1 | 3.991,7 | 8 | 6 | 3.020 | 1.983 / 2.402 |

Nguồn và phạm vi đo:

- **RSS:** working set Windows của PID runner, lấy từ `giam_sat/<lan>/ram_mau_tho.csv`. "PID + con" cộng mọi tiến trình
  con (worker dataloader), có đếm trùng trang chia sẻ.
- **RAM máy:** `psutil available` của toàn máy.
- **VRAM nvidia-smi:** `memory.used` của cả GPU, gồm desktop.
- **torch alloc / reserved:** allocator PyTorch của tiến trình train theo từng epoch, lấy từ JSON runner.

Số tính lại từ CSV khớp `tom_tat.json` của bộ giám sát (RSS max, RAM min) và `vram_nvidia_smi.max_mib` của runner.
Không có mục tài nguyên nào bị thiếu. Nếu thiếu, ô sẽ để trống và được ghi trong `thieu_du_lieu`, không điền 0.

## 8. Đầu ra

`results/phase2/r53_detector/ban_giao/`:

- `ban_giao_yolov8n.json`, `ban_giao_yolo11n.json`: đủ các mục `muc_8`, `muc_10`, `quy_dao_epoch`, `best`,
  `checkpoint_cuoi`, `danh_gia_cuoi_ultralytics`, `results_csv`, `lich_su_chay`, `toan_luot`, `loi`, `canh_bao`,
  `thieu_du_lieu`.
- `dev_tung_epoch.csv`, `checkpoint_sha256.csv`, `tai_nguyen.csv`.
- `nguon_sha256.csv`: bảng hash nguồn của mọi file đã đọc.

Các bảng gộp được dựng lại từ mọi `ban_giao_*.json` sau mỗi lệnh. Test và fixture nằm ở
`results/phase2/r53_detector/tests_handoff/`.

Từ vòng 2, mỗi lệnh chép các file đầu ra sắp bị ghi lại sang `ban_giao/_ban_truoc/<thoi diem>_pid<pid>/` trước khi ghi:

- `20261019_233432_520233_pid20512/`: đầu ra vòng 1 (18:58), gồm `ban_giao_yolov8n.json` cùng bốn CSV gộp.
- `20261019_233436_818688_pid2788/`: `ban_giao_yolo11n.json` của vòng 1, cùng bốn CSV gộp do lệnh yolov8n vòng 2 vừa
  ghi.

So bằng `git diff --no-index`: hai `ban_giao_<model>.json` mới chỉ khác bản vòng 1 ở `tao_luc` và
`ma_nguon_sha256`. Bốn CSV gộp mới giống bản vòng 1 từng byte.

Mục 10 của kế hoạch 02 đã có đủ: epoch / thời gian / exit (mục 3, 6), checkpoint sha256 (mục 4), dev từng epoch và
đánh giá cuối (mục 5), tài nguyên (mục 7), lịch sử resume (mục 3, 6).

## 9. Giới hạn

- R53A không nạp model bằng torch. Bằng chứng "nạp được" là kiểm tra runner thực hiện sau train, cộng với metadata đọc lại.
- Exit code của bộ giám sát được suy từ `tom_tat.json`, `trang_thai.json` và dòng `# exit=`. `chay_giam_sat.py` trả
  đúng exit của tiến trình con, nhưng không có file nào ghi riêng mã thoát của tiến trình công cụ.
- Thời gian tạm dừng giữa các lần đo từ lúc giám sát lần trước kết thúc đến lúc giám sát lần sau bắt đầu. Khoảng này
  gồm cả thời gian dry-run `--tiep-tuc` và thời gian máy ngủ (ví dụ 17/10 14:03–17:15).
- RSS "PID + con" đếm trùng trang chia sẻ, nên có thể cao hơn bộ nhớ thực dùng.
- Không cài gói, không ghi ngoài repo, không commit hay stage.

## 10. Trạng thái

R53A đã sửa F1/F2 và tự kiểm (vòng 2), đang chờ Codex review lại. **Chưa làm:** R53B (khóa giao thức, tiền kiểm
UA-DETRAC test / BDD val, smoke), đánh giá đầy đủ ba detector, tracking.

## 11. Sửa sau review vòng 1 (R53A-F1, R53A-F2)

Nguồn: review Codex vòng 1, verdict `revise`. Phần bằng chứng của review xác nhận hash nguồn, 15 epoch, best e15/e13 và
tài nguyên khớp bàn giao, nên chỉ sửa hai lỗi sau.

- **R53A-F1 (test xóa bằng chứng lượt trước).** Ở vòng 1, `tao_fixture` gọi `shutil.rmtree` khi fixture cùng tên đã
  tồn tại. Ba lần chạy test vòng 1 vì vậy đã ghi đè fixture, `dau_ra_*` và `ket_qua_test.json` của lần trước. Bản lần
  cuối vòng 1 vẫn còn, nhưng nội dung hai lần đầu không khôi phục được.
  - Sửa: `test_ban_giao_train.py` không còn lệnh xóa nào.
  - Mỗi lần chạy tạo thư mục mới `tests_handoff/lan_<thoi diem>_pid<pid>/`, tạo với `exist_ok=False`, trùng tên thì
    thêm hậu tố. Fixture, đầu ra và `ket_qua_test.json` (mở chế độ `x`) đều ghi trong thư mục này.
  - `tao_fixture` gặp thư mục đã có thì báo `FileExistsError`.
  - Trước khi chạy test, script chụp sha256 mọi file trong `tests_handoff/`. Test cuối `test_giu_bang_chung_luot_truoc`
    kiểm các file đó còn nguyên. `ket_qua_test.json` ghi số file và sha256 của ảnh chụp.
  - Bố cục vòng 1 (`fixture/`, `dau_ra_hop_le/`, `dau_ra_thieu_tai_nguyen/`, `ket_qua_test.json` 12/12,
    `thu_nghiem_som/`) được giữ nguyên và nằm trong ảnh chụp đã kiểm.
- **R53A-F2 (cổng `--out` quá rộng).** Hàm mới `kiem_out` resolve đường dẫn rồi chỉ chấp nhận khi đường dẫn bằng hoặc
  nằm dưới `results/phase2/r53_detector/ban_giao/` hoặc `…/tests_handoff/`.
  - So khớp theo thành phần đường dẫn, không theo tiền tố chuỗi, nên `ban_giao_khac` bị từ chối.
  - `main` gọi `kiem_out` trước `tong_hop`, nên khi từ chối thì chưa đọc nguồn và chưa ghi gì; exit 2.
  - `ghi_dau_ra` gọi lại `kiem_out` trước `mkdir`, để lời gọi trực tiếp từ test cũng qua cổng.
- **Bổ sung cùng nguyên tắc giữ bằng chứng** (không đổi số liệu). Ghi lại vào `ban_giao/` không còn thay thế âm thầm
  bản trước: bản trước được chép sang `_ban_truoc/` và kiểm byte trước khi ghi (mục 8).
- **Giới hạn.**
  - sha256 của các file `_ban_truoc/` không được tính lại trong lượt này: quyền Bash không có công cụ băm ngoài hai
    script. Bằng chứng giống từng byte gồm kiểm hash lúc chép (`_luu_ban_truoc`) và `git diff --no-index`.
  - Reviewer có thể đối chiếu sha256 với `review_points/0002_review/hashes.json` của orchestrator, ví dụ
    `ban_giao_yolov8n.json` vòng 1 = `6d6233ab…92bd4`.
  - Lần chạy 14/15 dùng bản test trung gian; hash của bản đó nằm trong `ket_qua_test.json` của thư mục lần chạy.
