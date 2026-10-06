# Vòng 42 — ghi nhận nghiệm thu `6c8e37a`; đính chính R41-N1; training dài YOLOv8n → YOLO11n (chi tiết)

**Phạm vi** (người dùng duyệt 2026-10-03):
- Training dài trên PC local: YOLOv8n rồi YOLO11n, tuần tự, **mỗi model tổng cộng 15 epoch** (kể cả khi resume).
- Dataset đầy đủ `data/yolo_bdd5_v1/`, bước frame 5, split đã khóa. Giữ nguyên batch, workers, seed, optimizer,
  augmentation và các tham số đã duyệt; không tuning.
- Ngân sách dự toán 56,64 h, kịch bản chậm hơn 73,95 h. Đây không phải bảo đảm.
- Chỉ training và validation dev theo cấu hình hiện có; **dừng trước giai đoạn đánh giá test**.
- Base review: **`6c8e37a`**. Vòng 42 **chưa** được nghiệm thu.

> **Đính chính vòng 43** (review `4d54744`: PASS WITH LIMITATIONS cho hồ sơ training; chỉ sửa câu chữ / bảng trình
> bày, CSV và log giữ nguyên):
>
> - **R42-N1**: RAM trống thấp nhất ghi theo **từng nguồn** (bộ giám sát / bộ lấy mẫu trong runner; cùng đo
>   `psutil.virtual_memory().available` nhưng ở thời điểm khác nhau). VRAM là `nvidia-smi memory.used` của **cả GPU**;
>   tổng RSS có thể đếm trùng trang bộ nhớ chia sẻ.
> - **R42-N2**: NaN xuất hiện từ batch 4597/6311 (≈73 %); dòng tiến độ cuối trước khi dừng là 4786/6311 (≈76 %). Dòng
>   log đó không phải bằng chứng chính xác batch cuối đã thực thi. Toàn bộ phần epoch 12 chưa hoàn tất bị bỏ.
> - **R42-N3**: sửa 5 ô làm tròn lệch trong bảng dev theo epoch (làm tròn một lần từ `results.csv`).

## 0. Trạng thái đầu vòng

Log: `logs/review42/00_trang_thai_dau.log`.
- HEAD `6c8e37a`, 0 commit sau đó. Working tree chỉ có file riêng của người dùng.
- Môi trường: Python 3.11.9, ultralytics 8.4.141 (`trainer.py` `7532ff0b…`), torch 2.6.0+cu124, CUDA 12.4,
  cuDNN 90100, GTX 1650 4 GiB, driver 555.97.
- Nguồn: đang cắm sạc (pin 97 %); gói High performance, sleep / hibernate = không bao giờ; không có khởi động lại đang chờ.
- RAM trống 0,26 GiB / 7,84 GiB; ổ C trống 49,0 GiB, ổ D trống 50,8 GiB.

## 1. Ghi nhận nghiệm thu `6c8e37a` (commit `6b39817`)

- **Báo cáo** `D:\LABOTARY\review_codex\CODEX_REVIEW_6c8e37a.md`: 10.963 B, sha256 `0c6932ca…`.
- **Kết luận**: **PASS WITH LIMITATIONS**, đóng R40-N1 (`results.csv`) và R40-N2 (`theo_batch.csv`) trong phạm vi chuẩn bị
  resume thật. R41-N1 là đính chính mô tả mức thấp.
- **Giới hạn ghi đúng như review**:
  - review không phê duyệt training dài;
  - không xác nhận chịu được mọi dạng mất điện;
  - không nghiệm thu guard ghi nối / khóa lượt chạy;
  - suite của tác giả được đối chiếu log, không chạy lại; reviewer không kiểm toàn cây Git.
- **Bằng chứng**: ZIP 188.110 B, sha256 `95e5672b…`. Giải nén vào thư mục mới, ZIP gốc giữ nguyên. `SHA256SUMS` khớp
  **43/43** trên đĩa và trên blob đã stage. Lưu ở `handoff/review_evidence/CODEX_6c8e37a/` (`-text`).
- **Gói đã nghiệm thu**: 690.663 B, `8920742e…`, khớp mục 1 của review. Lưu ở
  `handoff/archive/review_delta_6c8e37a.zip`.
- **Base review** `421dd53` → **`6c8e37a`** (vòng 41); giữ đủ 13 khai báo đầu vào không bàn giao.

## 2. Đính chính R41-N1 — chỉ câu chữ (commit `48dc026`)

Đúng như review: `sua_duoi_csv` kiểm nội dung mới trong bộ nhớ và hash file tạm **trước** khi `os.replace` thay file đích.
Hàm **không** đọc lại CSV đích sau `os.replace`; test đọc lại output để xác nhận.
- `src/tiep_tuc_huan_luyen.py`: **chỉ docstring** bước 4 của `sua_duoi_csv`. Kiểm:
  - AST sau khi bỏ docstring **trùng** bản `44045186…`;
  - 4 dòng văn bản khác nhau đều nằm trong docstring;
  - `py_compile` đạt; sha256 mới `d5b875bc…`.
- `REVIEW41_CHI_TIET.md`: sửa mục 3 bước 4, thêm dòng theo dõi vòng 42.
- `logs/review41/02_ke_hoach_kiem_r40n.md`: thêm đính chính vào phụ lục mục 4. Mục 1–3 giữ nguyên như đã đăng ký.
- Không thêm mã đọc lại, không thêm test, không guard append / khóa lượt.

## 3. Chuẩn bị lượt dài (commit `85645a0`)

- **Dry-run hai model** (`logs/review42/01_dry_run_<M>.log`): exit 0.
  - Hợp đồng dataset `bdd5-dataset-v2`, 0 lỗi: manifest v2; đã băm 56.046 ảnh + 56.046 nhãn + 18.600 file ignore;
    50.487 ảnh train-fit / 5.559 ảnh dev; split `b8a2848e…` khớp bản gốc; 0 `.npy`.
  - Định danh trùng bản xuất vòng 38: `data.yaml` `1280e57b…`, manifest `1ee6174f…`.
  - Checkpoint khởi tạo đúng sha256: YOLOv8n `f59b3d83…`, YOLO11n `0ebbc80d…`; COCO 80 lớp.
  - Tham số hiệu lực đúng cấu hình đã duyệt. Hai model chỉ khác `model` và `name`. Không xuất lại dataset.
- **Cấu hình** `configs/phase2/detector/train_bdd5.yaml` giữ nguyên byte (`a466a4fa…`), vì cấu hình pilot ghim sha256 của
  nó và test P1a kiểm giá trị đó. Chú thích "DE XUAT (chua duyet)" cạnh `epochs: 15` là lịch sử.
- **Kế hoạch đăng ký trước**: `logs/review42/02_ke_hoach_luot_dai.md`. Gồm tên / thư mục mới, cách khởi chạy, theo
  dõi, phục hồi, điều kiện để bắt đầu YOLO11n, rủi ro.
- **Khởi chạy độc lập với phiên Claude** (`khoi_chay_wmi.ps1`): WMI `Win32_Process.Create` → `chay_giam_sat.py chay` →
  `train_detector.py --che-do day-du --xac-nhan-train-dai [--tiep-tuc]`.
  - Tiến trình thăm dò vô hại cho thấy: tiến trình tạo ra có cha `WmiPrvSE.exe`, không nằm trong Job Object, sống sau
    khi lệnh tạo trả về.
  - Môi trường người dùng chuẩn: cùng USERPROFILE / APPDATA / TEMP và file settings Ultralytics; tìm thấy `git` và
    `nvidia-smi`.
  - Đây không phải tác vụ nền của Claude.
- **Theo dõi chỉ đọc** (`theo_doi_luot.py`): ảnh chụp mỗi 30 phút và mỗi epoch. Thêm một bộ canh NaN riêng đọc
  `log.txt` mỗi 30 giây, bật sau sự cố NaN ở mục 4.2.
- **Dừng có kiểm soát** (`dung_luot.py`): gửi `CTRL_C_EVENT` vào console ẩn của lượt, dùng nhánh `KeyboardInterrupt` sẵn
  có của runner. Nếu không được thì dừng cứng cây tiến trình. Cả hai lần dừng đều đi nhánh Ctrl+C và runner thoát sau
  8 s.

## 4. YOLOv8n — **HOÀN TẤT 15/15 epoch** (3 lần chạy)

### 4.1 Diễn biến

| Lần | Thư mục giám sát | Chạy | Epoch lưu | Kết thúc |
|:--|:--|:--|:--|:--|
| 0 | `yolov8n_lan0` | 2026-10-03 19:27 → 10-04 08:41 (13,24 h) | 1–7 | **Ctrl+C theo yêu cầu "tạm dừng"** của người dùng ("Dừng training ngay"); exit 130, `bi_ngat`; dòng tiến độ cuối trước khi dừng 6027/6311 (≈95 %); toàn bộ phần epoch 8 chưa hoàn tất bị bỏ |
| 1 | `yolov8n_tiep_tuc_1` | 10-04 14:38 → 23:29 (8,84 h) | 8–11 | **Loss NaN** từ batch 4597/6311 (≈73 %) của epoch 12; người dùng chọn "Dừng ngay, resume sau"; Ctrl+C, exit 130; dòng tiến độ cuối trước khi dừng 4786/6311 (≈76 %); toàn bộ phần epoch 12 chưa hoàn tất bị bỏ |
| 2 | `yolov8n_tiep_tuc_2` | 10-05 02:48 → 09:37 (6,82 h) | 12–15 | **hoàn tất**, exit 0, `[DAT]` |

Trước mỗi lần resume:
- chạy `--dry-run --tiep-tuc` (`07_*`, `13_*`): đạt;
- kiểm nguồn: đang cắm sạc, gói High performance;
- sau khi nạp: `kiem_khoi_phuc` **13/13 đạt** ở cả lần 1 và lần 2. Bản sao nguyên byte nằm ở
  `khoi_phuc/truoc_tiep_tuc_N/`; CSV và nhật ký sạch, không cắt hàng nào.

### 4.2 Sự cố và cách xử lý (quan sát, có log)

- **Nguồn điện**:
  - 10-04 13:25: máy chạy pin và gói nguồn chuyển sang Power saver (ngủ sau 15 / 10 phút). Đã báo người dùng, chờ
    tới khi cắm sạc và về High performance (14:37) mới resume. Không tự đổi cài đặt.
  - 17:15: rút sạc → 17:39 pin cạn (Kernel-Power 524) → máy ngủ hai lần (khoảng 3 phút và 1,5 phút) → 17:44 cắm lại. Tiến
    trình đứng yên trong lúc ngủ rồi chạy tiếp, **không bị ngắt**; epoch 9 hoàn tất (`10_*`).
  - 19:12–19:35: rút sạc thêm một lần (pin 94 % → 10 %), không ngủ.
- **NaN (epoch 12, lần 1)** — `11_*`:
  - Tốc độ bình thường khoảng 1,0 it/s tới batch khoảng 4400 (23:12). Sau đó chậm khoảng 2,6 s/it. Loss trung bình
    thành NaN từ batch 4597 (23:20:40).
  - Log System 14:30–23:22 không có nvlddmkm / TDR 4101 / WHEA. RAM trống lúc đó 590–690 MiB.
  - **Nguyên nhân chưa xác định.**
  - Checkpoint epoch 11 nạp lại trên CPU: 0/355 tensor EMA và 0/183 momentum không hữu hạn. `theo_batch.csv` không có
    hàng NaN, vì hàng của epoch 12 chưa được ghi.
  - Lần 2 chạy lại epoch 12–15 **không có NaN**, có bộ canh NaN theo dõi.
- **Tốc độ**: epoch thường khoảng 1,63–1,68 h. Epoch 9 (2 lần ngủ), 10 và 11 chậm hơn, tới 2,0 h. Cùng lúc thấy tải khác
  trên máy (java.exe khoảng 2 lõi CPU, Wallpaper Engine dùng GPU); đó chỉ là quan sát, chưa xác định nguyên nhân.

### 4.3 Kết quả

**Điều kiện thành công** (`15_kiem_thanh_cong_yolov8n.log`): **12/12 đạt**.
- bộ giám sát exit 0; `[DAT] day-du yolov8n`; `loi = None`; loss hữu hạn;
- last + best đúng 5 lớp, `train_args` trùng cấu hình;
- nhật ký `ket_thuc` lần 2: `hoan_tat`, `kiem_sau_train_dat`, exit 0;
- đủ 15 epoch.

**Dev nội bộ** (validation của Ultralytics trên 5.559 ảnh dev; **không phải test**), mAP50-95 / mAP50:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| mAP50-95 | 0,143 | 0,176 | 0,220 | 0,224 | 0,240 | 0,251 | 0,252 | 0,253 | 0,266 | 0,266 | 0,275 | 0,275 | 0,278 | 0,278 | **0,280** |
| mAP50 | 0,233 | 0,282 | 0,343 | 0,350 | 0,375 | 0,388 | 0,393 | 0,393 | 0,412 | 0,413 | 0,428 | 0,427 | 0,431 | 0,432 | 0,435 |

Epoch 14–15 tắt mosaic, đúng `close_mosaic 2`.

**Đánh giá cuối `best.pt`** (epoch 15, `final_eval` của Ultralytics trên dev):

| Lớp | Ảnh | Đối tượng | P | R | mAP50 | mAP50-95 |
|:--|--:|--:|:--|:--|:--|:--|
| tất cả | 5.559 | 46.799 | 0,605 | 0,420 | 0,435 | **0,280** |
| car | 5.407 | 42.211 | 0,790 | 0,775 | 0,824 | 0,569 |
| truck | 1.957 | 2.971 | 0,667 | 0,412 | 0,487 | 0,356 |
| bus | 780 | 984 | 0,443 | 0,305 | 0,266 | 0,177 |
| motorcycle | 205 | 226 | 0,654 | 0,319 | 0,330 | 0,172 |
| bicycle | 334 | 407 | 0,469 | 0,290 | 0,269 | 0,127 |

**Checkpoint và tài nguyên**:
- `weights/best.pt` sha256 `5f66776d3202389ca484f3d9e2f107dddb354d21a9f1762d19401d690a4a0d8a`.
- `weights/last.pt` sha256 `66e53715c5310b55e35d85db80ecd888d7a276fbf7de043fb4b7bdca40f0f422`.
- Mỗi file 6.222.691 B (Ultralytics đã bỏ optimizer); sha256 tính lại từ file trùng giá trị runner ghi.
- 15 bản sao `khoi_phuc/last_eNNN.pt` khớp sha256 nhật ký.
- Thời gian runner 3 lần: 103.525 s (28,8 h), gồm cả phần epoch 8 và 12 bị bỏ. Một epoch: 5.864–7.221 s train +
  52–80 s val.
- VRAM tối đa 2.933 MiB: `nvidia-smi memory.used` của **cả GPU**, gồm cả tiến trình khác.
- Tổng RSS / working set của runner và tiến trình con tối đa 4.083 MiB; có thể đếm trùng trang bộ nhớ chia sẻ.
- RAM máy trống thấp nhất, theo từng nguồn:
  - bộ giám sát (`tom_tat.json`, `ram.ram_he_thong_con_trong_mib.min`): 22 / **5** / 162 MiB ở lần 0 / 1 / 2. Mức
    5 MiB rơi vào lúc máy ngủ vì pin cạn. Trung vị 554–870 MiB tùy lần;
  - bộ lấy mẫu trong runner (`vram_nvidia_smi.ram_con_trong_min_mib`): 27 / 23 / 73 MiB.

## 5. YOLO11n — **HOÀN TẤT 15/15 epoch** (1 lần chạy, không resume)

- Chạy: 2026-10-05 09:39 → 10-06 15:55 (30,27 h), thư mục giám sát `yolo11n_lan0`. Bắt đầu sau khi YOLOv8n đạt 12/12.
- Không NaN, không ngủ, luôn cắm sạc với pin 100 % (83 ảnh chụp).
- Log System chỉ có 2 sự kiện đổi nguồn lúc 21:44:07–08, tức rút và cắm lại trong 1 giây (`21_*`).
- **Điều kiện thành công** (`19_*`): **12/12 đạt**.

**Dev nội bộ** (không phải test), mAP50-95 / mAP50:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| mAP50-95 | 0,140 | 0,181 | 0,200 | 0,207 | 0,222 | 0,238 | 0,242 | 0,243 | 0,252 | 0,256 | 0,266 | 0,268 | **0,272** | 0,269 | 0,270 |
| mAP50 | 0,229 | 0,280 | 0,314 | 0,323 | 0,346 | 0,367 | 0,373 | 0,375 | 0,388 | 0,395 | 0,413 | 0,411 | 0,416 | 0,410 | 0,411 |

Epoch 14–15 tắt mosaic, không vượt epoch 13. `best.pt` là epoch 13.

**Đánh giá cuối `best.pt`** (epoch 13, `final_eval` trên dev):

| Lớp | Ảnh | Đối tượng | P | R | mAP50 | mAP50-95 |
|:--|--:|--:|:--|:--|:--|:--|
| tất cả | 5.559 | 46.799 | 0,575 | 0,382 | 0,416 | **0,272** |
| car | 5.407 | 42.211 | 0,777 | 0,771 | 0,820 | 0,567 |
| truck | 1.957 | 2.971 | 0,672 | 0,413 | 0,484 | 0,357 |
| bus | 780 | 984 | 0,463 | 0,317 | 0,285 | 0,188 |
| motorcycle | 205 | 226 | 0,634 | 0,204 | 0,312 | 0,164 |
| bicycle | 334 | 407 | 0,329 | 0,206 | 0,182 | 0,085 |

**Checkpoint và tài nguyên**:
- `weights/best.pt` sha256 `d7d1c7e86d03c69d913febe79e4cbb6c2e1bcf9f17f034724af4de2ccfba7923`.
- `weights/last.pt` sha256 `3dbd9fb00208e99471513b45ee7f9ff6149633f0f14597d67ecbc52db49c26f4`.
- Mỗi file 5.445.075 B; sha256 tính lại trùng giá trị runner ghi. 15 bản sao `khoi_phuc` khớp nhật ký.
- Thời gian runner 108.820 s (30,2 h). Một epoch: 7.180–7.226 s train + 52–56 s val.
- VRAM tối đa 3.123 MiB: `nvidia-smi memory.used` của **cả GPU**, gồm cả tiến trình khác.
- Tổng RSS / working set của runner và tiến trình con tối đa 4.200 MiB, trung vị 1.074 MiB; có thể đếm trùng trang
  bộ nhớ chia sẻ.
- RAM máy trống thấp nhất, theo từng nguồn:
  - **81 MiB theo bộ giám sát** (`tom_tat.json`), trung vị 951 MiB; 107.295 mẫu, 0 mẫu không đọc được;
  - **6 MiB theo bộ lấy mẫu trong runner** (`vram_nvidia_smi.ram_con_trong_min_mib`).

  Hai nguồn đo cùng một đại lượng nhưng ở thời điểm khác nhau, nên không gọi 81 MiB là mức thấp nhất chung của cả
  lượt.

**Thời gian cả hai model**: thời gian runner tổng 59,0 h. Con số này nằm giữa kịch bản 56,64 h và 73,95 h, và gồm khoảng
3 h epoch dở bị bỏ của YOLOv8n. Thời gian đồng hồ từ lúc khởi chạy đầu tới lúc kết thúc là khoảng 68,5 h, vì có các
khoảng tạm dừng.

## 6. Bằng chứng trong gói

- `logs/review42/00`–`23`: trạng thái máy, dry-run, kế hoạch, khởi chạy, dừng, sự kiện nguồn, kiểm điều kiện thành
  công, tóm tắt.
- `theo_doi_<M>.log`: ảnh chụp JSON.
- `ket_qua/<M>/`:
  - kết quả runner (`train_bdd5_v1_<M>_day_du*.json`);
  - nhật ký lượt, `results.csv`, `args.yaml`;
  - `tom_tat.json` / `trang_thai.json` của bộ giám sát;
  - `log_loc.txt`: `log.txt` đã bỏ các dòng tiến độ từng batch, giữ dòng 100 % cuối mỗi pha, bảng validation, cảnh
    báo và mã thoát.
- `ket_qua/tep_lon_khong_ban_giao.csv`: 54 file lớn **không đóng gói**, kèm đường dẫn, bytes, sha256:
  - `log.txt` đầy đủ, `ram_mau_tho.csv`, `theo_batch.csv`, CSV nvidia-smi;
  - `weights/*.pt`, `khoi_phuc/**/*.pt`.
- Script: `khoi_chay_wmi.ps1`, `theo_doi_luot.py`, `dung_luot.py`, `kiem_thanh_cong.py`, `tom_tat_luot.py`,
  `dong_goi_ket_qua.py`, `ghi_trang_thai_may.py`.

## 7. Giới hạn

- **Chỉ dev nội bộ**:
  - `best.pt` được chọn theo fitness trên **chính** tập dev này, nên điểm dev của `best.pt` thiên lạc quan.
  - Chưa có test (BDD val / UA-DETRAC).
  - Mỗi model chỉ một lượt (một seed). Chưa kiểm định thống kê.
  - Chênh lệch giữa hai model trên dev (0,280 / 0,272 mAP50-95) **chưa đủ để kết luận** model nào tốt hơn.
- **YOLOv8n qua 2 lần resume**: không trùng từng bit với lượt liên tục (RNG, thứ tự dữ liệu, gradient tích lũy dở;
  giới hạn đã công khai từ vòng 39). Phần chưa hoàn tất của epoch 8 (dòng tiến độ cuối ≈95 %) và của epoch 12 (NaN từ
  ≈73 %, dòng tiến độ cuối ≈76 %) bị bỏ rồi chạy lại.
- **NaN**: nguyên nhân chưa xác định. Chưa kiểm cơ chế tự phục hồi NaN của Ultralytics, vì đã dừng trước khi epoch kết
  thúc. Nếu cơ chế đó chạy, runner sẽ coi lượt là lỗi (exit 3, `hoan_tat`), nên resume bị chặn.
- **Thời gian epoch** chịu ảnh hưởng của tải khác trên máy và các lần ngủ; không dùng làm số đo tốc độ chuẩn.
- **Ảnh chụp theo dõi** của trợ lý trống hai lần khi lượt YOLO11n đang chạy:
  - 10-05 23:49 → 10-06 01:08 (79 phút);
  - 10-06 11:53 → 14:22 (149 phút).

  Với YOLOv8n, ảnh chụp chỉ trống trong lúc lượt đã dừng. Bộ giám sát lấy mẫu RAM với chu kỳ 1 giây, nhưng CSV thô không
  nằm trong gói, và số mẫu cũng như "0 mẫu không đọc được" không tự chứng minh không có khoảng trống giữa các mẫu
  (đính chính vòng 43).
- **Kết quả runner theo từng lần**: file JSON của lần cuối chỉ chứa các epoch của lần đó. Nhật ký lượt chứa đủ 15 epoch.

## 8. Lệnh (từ gốc repo) và mã thoát

```text
python src/train_detector.py --model <M> --dry-run                                  # exit 0 (ca hai)
powershell -NoProfile -ExecutionPolicy Bypass -File handoff\logs\review42\khoi_chay_wmi.ps1 -Model <M> -GiamSat <thu muc> [-TiepTuc]
python src/train_detector.py --model yolov8n --dry-run --tiep-tuc                    # exit 0 (truoc lan 1, lan 2)
python handoff/logs/review42/dung_luot.py --giam-sat <thu muc giam sat> --cho-s 120   # 2 lan: Ctrl+C, runner exit 130
python handoff/logs/review42/kiem_thanh_cong.py <M> <thu muc giam sat lan cuoi>      # exit 0 (12/12) ca hai
python handoff/logs/review42/tom_tat_luot.py --model <M> --out <json>
python handoff/logs/review42/dong_goi_ket_qua.py --out handoff/logs/review42/ket_qua
```

| Lượt | Exit bộ giám sát |
|:--|:--|
| YOLOv8n lần 0 | 130 |
| YOLOv8n lần 1 | 130 |
| YOLOv8n lần 2 | **0** |
| YOLO11n lần 0 | **0** |

## 9. Chưa làm — chờ người dùng duyệt

- **Giai đoạn đánh giá test** (BDD val / UA-DETRAC) cho `best.pt` của hai model; so sánh bộ lọc; tracking; tối ưu Nano.
  Theo phạm vi, vòng này dừng ở đây.
- Đề xuất cũ vẫn chờ duyệt: guard ở hàm append; khóa thư mục lượt.

## 10. Commit vòng 42

| Commit | Nội dung |
|:--|:--|
| `6b39817` | Ghi nhận nghiệm thu `6c8e37a`, bằng chứng 43/43, archive gói, base → `6c8e37a`, trạng thái đầu |
| `48dc026` | Đính chính R41-N1 (chỉ câu chữ) |
| `85645a0` | Chuẩn bị: dry-run 2 model, kế hoạch, khởi chạy độc lập, theo dõi, tài liệu |
| `51fbac4` | YOLOv8n tạm dừng theo yêu cầu sau 7 epoch |
| `170448a` | Trước resume lần 1: dry-run `--tiep-tuc` đạt; điều kiện nguồn chưa đạt |
| `323f9ce` | YOLOv8n lần 1: epoch 8–11; NaN epoch 12 → dừng |
| `025a893` | YOLOv8n hoàn tất 15/15, 12/12 điều kiện đạt |
| `4f64bfb` | YOLO11n hoàn tất 15/15, 12/12 điều kiện đạt; đóng gói bằng chứng |
| commit chứa phụ lục này (target delta) | Brief, phụ lục, handoff |
