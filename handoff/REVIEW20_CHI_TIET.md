# Phụ lục vòng 20 — đóng benchmark Nano: sao lưu artefact v5, tính lại từ dữ liệu thô, tình trạng thẻ SD

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Base review **giữ `a5eec39`**. Trạng thái trước đợt đóng: HEAD `127a066` (vòng 19–20: dựng lại
env, Mốc 1 v5 ĐẠT, benchmark lần 2 — `handoff/NANO_STATUS.md` mục 4f).

**Phạm vi đợt này**: chỉ sao lưu, kiểm chỉ đọc, tính lại và viết tài liệu. **Không** chạy TensorRT/detector, training, tuning, benchmark
mới; không `apt`, không cài gói, không xóa file, không `fsck`, không sudo; không sửa mã trong `src/`, tracker, YAML khóa, GT, kết quả
UA-DETRAC/BDD, `pc-ref-*`, Mốc 1 v5, v3/v4 hay file riêng. Lệnh trên Nano trong đợt này **chỉ đọc**. Mật khẩu không được dùng và không
ghi ở đâu; SSH dùng khóa.

Giờ trong log: **giờ Nano = giờ PC + 1 h** (Nano UTC+8, PC UTC+7).

## 1. Dòng thời gian ngày 24/9 (giờ Nano)

| Giờ | Sự kiện | Bằng chứng |
|:--|:--|:--|
| 09:31 | boot (thẻ 1, `/dev/mmcblk1p1`) | `01_trang_thai_dau_phien.log` |
| 09:36–09:40 | dựng Miniforge + env `nano`, giải nén v5, **Mốc 1 v5 ĐẠT** (`nano_run_20260924-093948`) | `02`–`04_*.log`, `nano_reference/` |
| 09:41:30 | warm-up tách riêng: 1 CV + 1 EKF vào split `NANO-bench-warmup` | `bench_lan2/bench_run.log` |
| 09:42:03–09:48:13 | **benchmark chính**: 6 cặp = 12 lượt, split `NANO-bench` | `bench_lan2/` |
| 09:49 | parity đại diện + hash bất biến + trạng thái sau | `06_sau_bench.log` |
| 10:13:40 | **tắt máy sạch** (`last -x`: `shutdown system down`) → boot 10:13:59. Không do lệnh nào trong phiên này | `07b_*.log` |
| 10:35:30 | kiểm thẻ chỉ đọc | `07_tinh_trang_the_sd.log` |
| 10:35:50–56 | lập danh sách + SHA256/số byte nguồn, chép về PC | `sao_luu_nano/nguon_*.txt` |
| ≈10:36 | **tắt không sạch** (phiên ghi `crash`, không có bản ghi shutdown) → boot 10:36:47 | `07b_*.log` |
| 10:46–10:52 | kiểm bản sao (08), hash output 12 lượt (09a), kiểm lại thẻ (07b), băm lại nguồn (08b) | `08*`, `09a`, `07b` |

Cả hai lần khởi động lại đều **sau** cửa sổ đo (09:41–09:48). Journal không lưu lâu dài (`/var/log/journal` không có) nên log không
cho biết nguyên nhân. Lệnh của phiên này đều chỉ đọc và không dùng sudo. Ngày 24/9 không có thay đổi gói nào (`dpkg.log` 0 dòng
install/upgrade/remove, `apt/history.log` 0 lần chạy).

## 2. Sao lưu artefact v5 từ Nano (`handoff/logs/review20/sao_luu_nano/`)

| Nhóm | Nguồn trên Nano | Số file | Byte | Kiểm |
|:--|:--|--:|--:|:--|
| `goi_v5/` | `~/nano_run/v5/nano_transfer/`: `reference/*` (Mốc 1: JSON/log chạy, parity, hash `pc-ref`), `results_nano/bench_cv_ekf/lan2/*`, `bench_v5_lan2.sh`, `PACKAGE_INFO.json`, `MANIFEST_SHA256.txt`, `NANO-full/pc-ref-{cv,ekf}` (data), `NANO-bench/*` (12 lượt: `run_meta.json`, `summary.csv`, `video_meta`, `timing`; thêm `data` + `data_cls` cho r1, r6), `NANO-bench-warmup/*` | 95 | 5.262.092 | SHA256 95/95 khớp, số byte 95/95 khớp, danh sách nguồn == đích |
| `home/` | `~/dung_env.log`, `~/miniforge_install.log`, `~/conda_create.log`; `env_nano_goi.txt` (liệt kê `conda-meta` + `dist-info`, 57 dòng) | 3 + 1 | 34.976 | SHA256 3/3 khớp, số byte khớp |

- Danh sách nguồn: `danh_sach_{goi_v5,home}.txt`; SHA256/byte nguồn: `nguon_{sha256,byte}_{goi_v5,home}.txt`; kiểm: `08_kiem_sao_luu.log`.
- Đối chiếu với bản đã commit ở `127a066`: `nano_reference/` 8 file, `bench_lan2/` 13 file → 0 khác.
- **Sau lần tắt không sạch**: băm lại nguồn trên Nano → 95/95 và 3/3 **không đổi** so với danh sách lấy trước đó (`08b_*.log`).
- 7 file mang CRLF (`pc-ref-{cv,ekf}` data, 4 `parity_pc_np_vs_ul_*.json`, `PACKAGE_INFO.json`, vì đóng gói từ PC). Repo dùng
  `* text=auto` nên Git sẽ đổi chúng sang LF. `sao_luu_nano/.gitattributes` (`* -text`, chỉ trong thư mục này) giữ nguyên byte. Sau
  khi stage, blob trong index khớp SHA256 nguồn 95/95 và 3/3. Nếu giải nén delta bằng công cụ có đổi kết thúc dòng, hãy so theo
  blob Git hoặc payload ZIP.
- **Không sao lưu**: cache NPZ và engine (đã nằm trong gói v5, hash trong `MANIFEST_SHA256.txt`); output `data/` của 8 lượt
  r2–r5 (hash ở mục 5 cho thấy chúng giống hệt r1/r6 theo cấu hình). Không có **log tegrastats thô** để sao lưu: công cụ chỉ lưu tóm
  tắt (`n_mau`, `nhiet_max_C`, `nhiet_tb_C`) trong JSON.

## 3. Tình trạng thẻ SD (chỉ đọc; `07_tinh_trang_the_sd.log`, `07b_the_sd_sau_khoi_dong_lai.log`)

| Mục | 10:35 (trước tắt không sạch) | 10:52 (sau) |
|:--|:--|:--|
| root | `/dev/mmcblk1p1` ext4 **rw**, relatime | như cũ |
| `df -h /` | 58G, dùng 27G (49 %) | như cũ |
| inode dùng | 304.188 | 304.183 |
| ext4 `errors_count` / `first_error_time` / `last_error_time` | 0 / 0 / 0 | 0 / 0 / 0 |
| `dmesg` mmc/ext4 | không có lỗi I/O/ext4 | `EXT4-fs (mmcblk1p1): 18 orphan inodes deleted`, `recovery complete` (phục hồi journal sau tắt không sạch), không có lỗi I/O |

Thẻ đang cắm vẫn là **thẻ 1** (inode dùng đầu phiên 268.484, gần với fsck thẻ 1 là 267.687; phần tăng thêm đến từ Miniforge/env/v5 dựng lại).
Không chạy fsck trên root đang mount. Thẻ này từng có nhiều lỗi fsck (vòng 19), nên dù hiện tại không có lỗi, rủi ro filesystem vẫn còn.
Bản sao ở mục 2 là bản chính để review.

## 4. Tính lại benchmark từ dữ liệu thô (`09_tinh_lai_tu_raw.{log,json}`, script `tinh_lai_bench.py`)

Script đọc trực tiếp `timing/MVI_39031.csv` của 12 lượt trong bản sao, không import mã repo. Nó bỏ 50 frame đầu mỗi lượt, tự ghép cặp
theo `cap` và so từng số với `bench_cv_vs_ekf.json` (sha256 `4d63d060…`). **Kết quả: ĐẠT**, mọi đại lượng lệch 0 (≤ 1e-9)
so với công cụ.

- Mỗi lượt: **1470 frame nguồn → 1420 frame đo**; cột `frame` = 1..1470; `n_steps` = 1 ở mọi frame; `n_dets` theo frame giống hệt ở 12
  lượt; `n_tracks` giống hệt trong 6 lượt cùng cấu hình; `t_assoc_ms` = `t_update_ms − t_predict_ms` (sai khác trung bình ≤ 9e-16).
- Thứ tự chạy theo mốc thời gian tăng ngặt; cặp 1–3 là **AB** (CV → EKF), cặp 4–6 là **BA** (EKF → CV). 12/12 lượt exit 0,
  không loại lượt nào.
- Warm-up tách riêng (không vào số liệu): CV exit 0, wall 15,9 s; EKF exit 0, wall 16,9 s (lượt nguội chậm hơn rõ so với 9,6–10,9 s của
  lượt chính).

### 4a. 12 lượt (ms/frame trên 1420 frame; wall gồm khởi động `conda run`)

| Chạy | Cặp | Cấu hình | update tb | update trung vị | predict tb | assoc tb | wall (s) | tegrastats nhiệt max / tb (°C) |
|--:|:--|:--|--:|--:|--:|--:|--:|:--|
| 1 | 1 AB | CV | 4,657 | 4,061 | 0,613 | 4,044 | 9,64 | 42,0 / 40,3 |
| 2 | 1 AB | EKF | 5,512 | 4,659 | 1,246 | 4,266 | 10,90 | 42,5 / 40,5 |
| 3 | 2 AB | CV | 4,686 | 4,086 | 0,618 | 4,069 | 9,70 | 42,5 / 40,6 |
| 4 | 2 AB | EKF | 5,445 | 4,577 | 1,230 | 4,215 | 10,78 | 43,0 / 41,0 |
| 5 | 3 AB | CV | 4,693 | 4,107 | 0,618 | 4,075 | 9,70 | 42,0 / 40,8 |
| 6 | 3 AB | EKF | 5,498 | 4,580 | 1,248 | 4,250 | 10,87 | 43,5 / 41,3 |
| 7 | 4 BA | EKF | 5,414 | 4,566 | 1,227 | 4,187 | 10,73 | 43,0 / 41,5 |
| 8 | 4 BA | CV | 4,774 | 4,175 | 0,633 | 4,141 | 9,88 | 44,0 / 41,6 |
| 9 | 5 BA | EKF | 5,491 | 4,631 | 1,263 | 4,228 | 10,86 | 44,0 / 41,5 |
| 10 | 5 BA | CV | 4,757 | 4,210 | 0,626 | 4,131 | 9,83 | 43,5 / 41,7 |
| 11 | 6 BA | EKF | 5,443 | 4,605 | 1,232 | 4,211 | 10,77 | 44,0 / 41,9 |
| 12 | 6 BA | CV | 4,697 | 4,065 | 0,618 | 4,079 | 9,71 | 43,5 / 41,7 |

Theo cấu hình (6 lượt): CV update tb trung vị **4,695** (4,657–4,774), predict 0,618, assoc 4,077; EKF **5,468** (5,414–5,512),
predict 1,239, assoc 4,221.

### 4b. Hiệu theo cặp EKF − CV

| Cặp | Thứ tự | d_update tb | d_predict | d_assoc | d_update (trung vị frame) | d_wall (s) | tỷ lệ update |
|--:|:--|--:|--:|--:|--:|--:|--:|
| 1 | AB | +0,855 | +0,633 | +0,222 | +0,598 | +1,259 | +18,4 % |
| 2 | AB | +0,758 | +0,612 | +0,146 | +0,491 | +1,081 | +16,2 % |
| 3 | AB | +0,806 | +0,630 | +0,176 | +0,473 | +1,172 | +17,2 % |
| 4 | BA | +0,640 | +0,593 | +0,046 | +0,391 | +0,849 | +13,4 % |
| 5 | BA | +0,734 | +0,637 | +0,097 | +0,420 | +1,027 | +15,4 % |
| 6 | BA | +0,746 | +0,614 | +0,132 | +0,539 | +1,059 | +15,9 % |

### 4c. Tổng hợp 6 cặp (thống kê mô tả)

| Đại lượng | Trung vị | Q1–Q3 (exclusive, như công cụ) | Q1–Q3 (inclusive) | min / max | Cặp dương |
|:--|--:|:--|:--|:--|:--|
| d_update tb (ms/frame) | **+0,752** | [+0,710; +0,818] | [+0,737; +0,794] | +0,640 / +0,855 | **6/6** |
| d_predict (ms/frame) | +0,622 | [+0,608; +0,634] | [+0,613; +0,632] | +0,593 / +0,637 | 6/6 |
| d_assoc = update − predict | +0,139 | [+0,084; +0,187] | [+0,105; +0,168] | +0,046 / +0,222 | 6/6 |
| d_update (trung vị frame) | +0,482 | [+0,413; +0,554] | [+0,433; +0,527] | +0,391 / +0,598 | 6/6 |
| d_wall (s/lượt) | +1,070 | [+0,983; +1,194] | [+1,035; +1,149] | +0,849 / +1,259 | 6/6 |
| tỷ lệ update | +16,0 % | [+14,9; +17,5] % | [+15,5; +16,9] % | +13,4 / +18,4 % | 6/6 |

Cách đọc: trên sequence và cấu hình này, EKF-CTRV tốn thêm khoảng **0,75 ms/frame** chi phí `tracker.update()` so với KF-CV, tức
khoảng **+16 %**. Phần lớn đến từ bước predict (+0,62 ms); phần còn lại (+0,14 ms) nằm trong ghép cặp và cập nhật. Cả 6 cặp cùng dấu,
ở cả thứ tự AB và BA. Hai cách tính tứ phân vị cho kết quả khác nhau vì n = 6. **IQR ở đây chỉ mô tả độ phân tán của 6 hiệu số**:
nó không phải khoảng tin cậy, và 6 cặp không đủ để nói "có ý nghĩa thống kê". d_wall ≈ 1,07 s/lượt cùng cỡ với 1470 × 0,75 ms ≈ 1,1 s.
Đây chỉ là kiểm tra độ nhất quán; wall còn gồm khởi động `conda run`, import và I/O.

## 5. Tính đúng của output và bất biến

- **Hash output** `data/MVI_39031.txt` trên Nano (`09a_hash_output_12_luot.log`, chỉ đọc): **cả 6 lượt CV + warm-up CV + `nano-cv`
  Mốc 1 = `09419c20…`**, bằng `pc-ref-cv` sau chuẩn hóa LF. **Cả 6 lượt EKF + warm-up + `nano-ekf` = `63e209df…`**, bằng
  `pc-ref-ekf` (LF). Tức là việc đo không làm đổi kết quả tracking.
- **Parity đại diện** `bench-{cv,ekf}-r{1,6}` so với `NANO-full/pc-ref-{cv,ekf}`: ĐẠT, bbox/conf lệch 0, ID khớp (5842 / 5845 dòng)
  (`bench_lan2/parity_bench_*.json`, `06_sau_bench.log`).
- **Hash trước/sau**: `pc-ref-*` 24 file trước == sau == `reference/pc_ref_sha256.txt`; `nano-*` Mốc 1 24 file trước == sau
  (`bench_lan2/{pc_ref,moc1_nano}_sha256_{truoc,sau}.txt`).
- **Cờ `bit_identical_data_txt=False`** trong JSON Mốc 1 chỉ do kết thúc dòng: `pc-ref` sinh trên Windows (pandas `to_csv` dùng
  `os.linesep` = CRLF), Nano ghi LF. Sau chuẩn hóa CRLF → LF, byte giống hệt (hash ở trên).

## 6. Điều kiện đo (`bench_lan2/00_trang_thai_truoc.txt`, `99_trang_thai_sau.txt`, `05_*.log`, `06_*.log`, JSON từng lượt)

Jetson Nano 4 GB, L4T R32.7.5; thẻ 1; env `nano` (Py 3.8.20, numpy 1.24.4, pandas 2.0.3, pyyaml 6.0.2, lap 0.5.13),
`BYTETRACK_BACKEND=np`; gói v5 `ae5fd57b…` (dựng tại `eeb1daf`); script `bench_v5_lan2.sh` sha256 `03d1d5e2…`. Cache
`yolov8n-coco_conf0.05_all`, `MVI_39031`, `--conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --fps 25 --track-buffer-sec 1.2
--elapsed-mode steps`. `nvpmodel` MAXN, governor `schedutil` ở mọi lượt; **không** chạy `jetson_clocks` (cần root, và không đổi
cấu hình theo yêu cầu). Nghỉ 20 s giữa các lượt; tegrastats 500 ms: CV 19 mẫu, EKF 21 mẫu mỗi lượt; nhiệt max 42,0–44,0 °C; CPU-therm
41 → 43 °C từ đầu đến cuối; swap 0; RAM ~1,36/3,96 GB. Chạy nền: desktop (Xorg, compiz), `update-manager`, `packagekitd`,
`gnome-software` (~2–4 % CPU mỗi tiến trình lúc nghỉ, trước và sau đo).

## 7. Giới hạn

1. **Tracker-from-cache**: đo `tracker.update()` khi replay det cache. Không gồm decode video, detector (TensorRT), vẽ hay ghi video.
   Đây **không phải FPS end-to-end**. Không được cộng số này với thời gian detector đo ở vòng 16 để ra FPS hệ thống.
2. Chỉ **một sequence** (`MVI_39031`, 1470 frame), một cache, một cấu hình ngưỡng, **6 cặp**. Không khái quát sang cảnh khác, mật độ
   xe khác hay thiết bị khác.
3. Không lưu tần số CPU theo lượt. Tegrastats chỉ còn tóm tắt nhiệt, **không có dòng thô**. Governor `schedutil` không khóa xung
   nhịp; có tác vụ nền (mục 6).
4. Thẻ 1 từng có lỗi fsck; sau benchmark có một lần tắt không sạch (ext4 đã phục hồi journal, `errors_count` 0). Nguyên nhân hai lần
   khởi động lại không xác định được từ log.
5. IQR (n = 6) phụ thuộc cách tính tứ phân vị. Nó không phải khoảng tin cậy và không phải kiểm định.
6. Không suy diễn sang **HOTA/IDF1** hay độ chính xác: benchmark này không đo độ chính xác. Output trùng `pc-ref` chỉ cho thấy tracker
   trên Nano cho **cùng kết quả** với PC.
7. Trên thẻ 1 hiện chưa dựng lại v3/v4, engine TensorRT và Mốc 2 (bằng chứng vòng 16 nằm trên PC); đợt này cũng không mở lại.

## 8. File mới của đợt đóng (không có mã trong `src/`)

`handoff/logs/review20/`: `07_tinh_trang_the_sd.log`, `07b_the_sd_sau_khoi_dong_lai.log`, `08_kiem_sao_luu.log`,
`08b_hash_nguon_sau_khoi_dong_lai.log`, `09_tinh_lai_tu_raw.{log,json}`, `09a_hash_output_12_luot.log`, `10_kiem_doc_duoc.log`,
`tinh_lai_bench.py` (script phân tích độc lập, không nằm trong `src/`, không được test suite import), `sao_luu_nano/` (mục 2, kèm
`.gitattributes` cục bộ). Commit bằng chứng: `cf4261f`.
Tài liệu: phụ lục này, `handoff/NANO_STATUS.md` (mục 4f), `CODEX_HANDOFF.md` (hàng 16), `handoff/REVIEW_BRIEF.md`.
Không sửa mã nên không chạy lại full suite; lần gần nhất là 33/33 trên `eeb1daf`, và mã `src/` không đổi kể từ đó.
