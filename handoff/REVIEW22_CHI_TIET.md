# Phụ lục vòng 22 — sửa N21-F1/F2/F3 (review `9f210e2`), kiểm chứng và đo lại luồng trực tiếp trên Nano

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Review: `handoff/CODEX_REVIEW_9f210e2.md` (REQUEST CHANGES, ba lỗi P2). Base review **giữ `45e086a`**.
Trạng thái đầu vòng: HEAD `9f210e2`, 0 commit sau đó, working tree chỉ có file riêng của người dùng (`logs/review22/00_trang_thai_dau.log`).
Dữ liệu vòng 21, benchmark vòng 20, kết quả/lock UA-DETRAC và BDD **giữ nguyên**. Không training/tuning; không đổi `run_tracking`,
`det_cache`, tracker, YAML khóa. Engine vòng 21 được dùng lại (sha256 `c32690cd…` kiểm lại trước khi đo).

## 1. Tái hiện trước khi sửa (`logs/review22/01_repro_truoc_khi_sua.log`, script `repro_n21.py`, mã `9f210e2` trong clone tạm)

Entry point thật, detector giả lập, 8 ảnh (mỗi frame chậm 20 ms để vượt độ phân giải 15,6 ms của `time.monotonic` trên Windows).
Đây là fixture, **không** phải benchmark.

| Finding | Kịch bản | Quan sát trên mã cũ |
|:--|:--|:--|
| N21-F1 | chèn trễ 1 s vào `load_tracker_cfg()` sau HELLO | `wall_vong_lap_s` **0,067 s** trong khi frame 1 bắt đầu đọc → frame 8 xong tracker **1,094 s**; FPS báo **118,8**, theo khoảng bao phủ **7,31** |
| N21-F2 | `--save-cache review_custom --cache-root "<tmp>/custom root/det cache"` | live exit 0, NPZ đúng thư mục; chạy nguyên `replay_argv` → **exit 1** `FileNotFoundError …\data\interim\det_cache\review_custom\meta.json` |
| N21-F3 | `--tracker-yaml` không tồn tại / YAML hỏng | detector **đã khởi động** (`detector.pid`), cha **exit 1** traceback, không `run_meta` |
| (thêm) | `--set khong_co_dau_bang` | **được chấp nhận im lặng** (exit 0) |

## 2. Sửa (`7dd74be`, `b0c645b`)

**N21-F1 — định nghĩa thời gian chốt trước khi sửa, rồi đồng bộ bắt đầu** (`nano_stream_proto` giao thức 2, `nano_detect_stream`, `nano_live`):
con khởi tạo xong → gửi HELLO → **chờ lệnh BẮT ĐẦU** (`G`); cha chuẩn bị (cấu hình, telemetry) → **bật timer rồi mới gửi `G`** → con mới đọc
frame 1. `run_meta.thoi_gian`:

| Trường | Phạm vi |
|:--|:--|
| `khoi_tao_s` | spawn con → nhận HELLO (Python con, import, nạp engine, làm nóng) |
| `chuan_bi_cha_s` | nhận HELLO → ngay trước khi gửi BẮT ĐẦU; chưa frame nào được đọc |
| `xu_ly_s`, `fps_xu_ly` | ngay trước khi gửi BẮT ĐẦU → `track_video` trả về (frame cuối qua tracker, nhận EOS hoặc dừng); FPS = số frame / `xu_ly_s` |
| `ket_thuc_s` | sau vòng lặp → con thoát (xả frame đang bay khi dừng, reap con, dừng telemetry) |
| `ghi_s` | ghi MOT/CSV/cache |
| `fps_on_dinh` | (N − 25) / (t_xong[N] − t_xong[25]), không đổi |
| `kiem_cua_so_do` | từ mốc `monotonic`: frame 1 bắt đầu đọc ≥ lúc gửi BẮT ĐẦU và frame cuối xong ≤ hết cửa sổ |

Tên cũ `wall_vong_lap_s`/`fps_vong_lap` bị thay (không trộn hai phạm vi). `S` trước `G` → EOS 0 frame. Vòng tracking dùng lại nguyên.

**N21-F2** — `src/nano_replay.py`: đặt `det_cache.CACHE_ROOT` rồi gọi **chính `run_tracking.main()`** (runner không đổi; mọi chỗ đọc
cache dùng thuộc tính module lúc gọi). `replay_argv` = `python src/nano_replay.py --cache-root <đường dẫn tuyệt đối> -- <đối số run_tracking>`
khi root khác mặc định (một phần tử argv, giữ dấu cách); root mặc định giữ lệnh `run_tracking.py` như cũ.

**N21-F3** — `kiem_truoc_khi_chay()` trước spawn: YAML (tồn tại, đọc được, `tracker_type` có trong registry), `--set` (dạng `key=value`,
khóa có trong YAML, kiểu giá trị), tham số số học, `--class-map`, engine/cache giả lập/.pt, interpreter, `LD_PRELOAD`, regions, cache
đích, stop-file → **mã 2, không khởi động detector**. Sau spawn: `try/finally` **luôn** dừng/reap con và dừng telemetry; lỗi trước BẮT ĐẦU →
mã 2 `loi_chuan_bi`/`loi_khoi_tao` (có `run_meta`); lỗi ở cha giữa vòng lặp → mã 3 `loi_cha`, ghi phần đã nhận (pipeline, cache), không
"hoàn tất". Giới hạn: khi `tracker.update()` ném lỗi, `run_tracking.track_video` không trả hàng MOT đã tích lũy (không sửa runner dùng chung).

## 3. Sau khi sửa

- **Tái hiện lại** (`logs/review22/02_repro_sau_khi_sua.log`): F1 `xu_ly_s` **0,1708 s** vs bao phủ **0,172 s** (46,84 vs 46,51 FPS, trong
  một tick 15,6 ms), frame 1 đọc sau BẮT ĐẦU, `chuan_bi_cha_s` 1,01 s không vào FPS; F2 `replay_argv` nguyên → exit 0, `nano_parity` ĐẠT;
  F3 → mã 2, không `detector.pid`.
- **Regression** `src/test_nano_stream.py` 15a/15b (chuẩn bị chậm 1 s; hoàn tất và dừng sớm: con chờ BẮT ĐẦU ≥ 900 ms, mọi frame trong
  cửa sổ, `xu_ly_s` ≥ bao phủ, FPS không thổi phồng), 16 (root ngoài mặc định có dấu cách, chạy nguyên `replay_argv`, parity + byte), 17a
  (YAML thiếu/hỏng/`tracker_type` lạ, `--set` sai dạng/khóa lạ/sai kiểu, `--hang-doi 0` → mã 2, không spawn), 17b/17c (lỗi ở cha sau spawn:
  mã 2 / mã 3, con được reap). Phép 10 đổi cách gây lỗi khởi tạo con (tên video sai, vì thư mục cache thiếu giờ bị chặn trước spawn).
  PC **20/20** (1 SKIP SIGINT/Windows). Cùng file test với mã `9f210e2` (clone tạm): 1–13 PASS, **15–17 FAIL** (`regression_bat_ma_cu.log`).
- Test liên quan PC: `test_nano_detector` 6/6, `test_nano_parity` 20/20, `test_nano_run_cache` 9/9, `test_track_video_file` 27/27,
  `test_run_tracking_replay` 10/10. Full suite: `**34/34 PASS** (`logs/review22/tests_20260928-1246.log`, trên `b0c645b`)`.

## 4. Trên Nano (gói v8 `f4da9f08…`, 63.858.472 B, `~/nano_run/v8`; v5–v7 giữ nguyên; engine `c32690cd…` dùng lại)

Máy ổn định suốt đợt (boot 11:53:15, không khởi động lại, ext4 `errors_count` 0).

1. Manifest 554/554, biên dịch Python 3.6.9/3.8.20, `nano_env_check` ĐẠT; `test_nano_parity` 20/20, `test_nano_run_cache` 9/9,
   **`test_nano_stream` 20/20** (con 3.6 ↔ cha 3.8, kể cả 5b SIGINT và 15–17), 1 SKIP (torch) — `logs/review22/03_trien_khai_test_v8.log`.
2. **Parity 10/10** (`nano_live_v8/…/parity/`): giả lập == `pc-ref` smoke; **TensorRT CV/EKF live == replay** (parity exit 0, byte giống
   hệt); dừng stop-file (56 frame, con thoát 0, replay phần đã xử lý khớp), SIGINT, `kill -9` con, engine sai → đúng mã thoát.
   **MOT replay bàn giao để tự so byte**: `parity/B_replay_cv/`, `B_replay_ekf/`, `D3_replay_dung/` cạnh `B_trt_cv/`, `B_trt_ekf/`,
   `D3_stop_file/` (sha256 LF: CV `15ab38b4…`, EKF `238dbb2a…`, dừng `fd410cf4…` — live == replay, và bằng output live vòng 21).
   Frame 1 bắt đầu đọc 0,27–0,34 ms sau khi gửi BẮT ĐẦU.
3. **Đo lại** (điều kiện công bố trước `b0c645b`, `dieu_kien_do_toan_luong_v22.md`; 6/6 lượt exit 0, không loại lượt, cả 6 có
   `moi_frame_trong_cua_so = true`):

| Lượt | Cấu hình | `fps_xu_ly` | `fps_on_dinh` | decode trung vị (ms) | trễ trung vị (p90) ms | `khoi_tao_s` | `chuan_bi_cha_s` |
|--:|:--|--:|--:|--:|--:|--:|--:|
| 1 | CV | 9,07 | 9,03 | 16,4 | 118,1 (122,4) | 9,06 | 0,019 |
| 2 | EKF | 9,42 | 9,42 | 11,4 | 113,9 (117,8) | 8,35 | 0,020 |
| 3 | EKF | 9,20 | 9,17 | 16,2 | 117,6 (121,8) | 8,26 | 0,020 |
| 4 | CV | 9,13 | 9,10 | 16,4 | 118,6 (121,5) | 8,11 | 0,019 |
| 5 | CV | 9,19 | 9,19 | 15,9 | 114,9 (122,0) | 8,65 | 0,018 |
| 6 | EKF | 9,36 | 9,38 | 11,4 | 114,3 (117,7) | 8,24 | 0,020 |

Theo cấu hình (trung vị, min–max 3 lượt): **CV 9,13 (9,07–9,19)**, **EKF 9,36 (9,20–9,42)**. Các stage khác gần như cố định giữa các lượt:
infer 73,5–73,7 ms (execute ~68,4), pre 12,4–13,4, post 7,1–7,3; `tracker.update()` CV 5,57–5,78 / EKF 6,34–6,48 ms; `ket_thuc_s` ~0,79 s;
`ghi_s` ~0,1 s. RAM tối đa ~2,68 GB, swap 15–16 MB, CPU 1224–1479 MHz, nhiệt CPU ≤ 39 °C, GPU ≤ 36 °C.

**Cách đọc**: decode của OpenCV 3.2 có **hai mức ~11,3 và ~16,3 ms/frame**; tỷ lệ frame ở mức chậm dao động 3–95 % giữa các lượt
(vòng 21: 0–88 %), rơi vào cả 3 lượt CV nhưng chỉ 1 lượt EKF ở vòng này. Chênh lệch FPS CV/EKF (~0,2) cùng cỡ biến thiên do decode, nên
**không** kết luận EKF nhanh hơn CV; nguyên nhân hai mức decode chưa xác định (có thể do xung CPU `schedutil`, chưa đo). Thông lượng vẫn do
tiến trình detector quyết định; tracker chạy song song ở cha.

## 5. Đính chính số vòng 21 (giữ số cũ, không sửa timestamp gốc) — `logs/review22/06_dinh_chinh_fps_v21_doi_chieu_v22.log`

`fps_vong_lap` vòng 21 được tính trên cửa sổ bắt đầu **sau** khi con đã đọc/infer phần đầu; tính lại từ timestamp sẵn có với phạm vi
"frame 1 bắt đầu đọc → frame 300 xong tracker" (không gồm EOS/ghi): CV 9,0960 / 9,3435 / 9,2697; EKF 9,3520 / 9,4157 / 9,2629
(thiếu 68–90 ms mỗi lượt, lệch FPS 0,21–0,28 %) — khớp bảng của review. Cùng phép tính cho vòng 22: cửa sổ `xu_ly_s` luôn **dài hơn**
khoảng bao phủ 18–20 ms (gồm nhận EOS), nên `fps_xu_ly` không bị thổi phồng. Kết luận ~9,3 FPS của vòng 21 vẫn đứng; `fps_on_dinh`
vòng 21 không bị ảnh hưởng.

## 6. Mức bằng chứng

- **Reviewer tự kiểm được**: test PC (fixture), roundtrip delta, và so byte **live vs replay** của TensorRT thật qua 6 file MOT trong
  `nano_live_v8/` (không cần NPZ).
- **Chạy thật trên Nano, bàn giao log**: test thiết bị, parity/biên, 6 lượt đo, telemetry.
- **Không làm**: camera, render, toàn sequence 1470 frame qua luồng trực tiếp; nguyên nhân mất nguồn ngày 28/9 (trước vòng này) vẫn chưa rõ.

## 7. File

Mã: `src/nano_stream_proto.py`, `src/nano_detect_stream.py`, `src/nano_live.py`, `src/nano_replay.py` (mới), `src/nano_live_kiem.py`,
`src/test_nano_stream.py`. Bằng chứng `handoff/logs/review22/`: 00–06, `repro_n21.py`, `regression_bat_ma_cu.log`,
`dieu_kien_do_toan_luong_v22.md`, `tinh_lai_cua_so_v21_v22.py`, full suite, `nano_live_v8/` (174 file từ Nano, sha256/byte khớp nguồn,
`.gitattributes` `* -text`; 30 BMP đen tổng hợp không chép về — `05_khong_chep_ve.log`; 4 NPZ cache live bị loại khỏi delta, khai báo hash).
