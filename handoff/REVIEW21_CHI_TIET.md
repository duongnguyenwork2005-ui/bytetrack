# Phụ lục vòng 21 — chốt review `45e086a`, sửa R20-P2/P3, luồng trực tiếp video → TensorRT → tracker trên Nano

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Review: `handoff/CODEX_REVIEW_45e086a.md` (PASS WITH LIMITATIONS cho benchmark tracker-from-cache
vòng 20). Base review **chuyển sang `45e086a`**. Trạng thái đầu vòng: HEAD `45e086a`, 0 commit sau đó, working tree chỉ có file riêng
của người dùng (`logs/review21/00_trang_thai_dau.log`). Benchmark vòng 20, kết quả/lock UA-DETRAC, BDD và `pc-ref-*` **không đổi**.
Không training, không tuning, không `apt`, không flash/boot/DTB, không đổi `nvpmodel`/xung. Mật khẩu không được dùng; SSH bằng khóa.

## 1. Chốt review, base (`a3ff001`, `8184bfe`)

- Gói đã review `review_delta_latest.zip` = **4.396.467 B**, sha256 `0b42dc93…` **khớp báo cáo**; lưu cục bộ
  `handoff/archive/review_delta_45e086a.zip` (thư mục bị ignore). Gói Nano v5 `ae5fd57b…` và v6 `b0712b0f…` cũng được lưu trước khi dựng lại.
- `make_review_delta.py --set-base 45e086a --round 20 --giu-khai-bao <NPZ vòng 16>`: NPZ
  `handoff/logs/review16/nano_out/nano_trt_MVI_39031_f0001-0300/MVI_39031.npz` (13.469 B, sha256 `d18dc4ef…`) được **khai báo giữ
  nguyên** trong `review_base.json` và luôn in trong `DEPENDENCIES.md` của các gói sau (cơ chế mới, mục 2).

## 2. R20-P2 — patch giữ nguyên byte (`281117f`)

**Tái hiện trước khi sửa** (`logs/review21/repro_crlf_truoc_khi_sua.log`, script `repro_crlf.py`, repo tạm, mọi lệnh git qua
`-c core.autocrlf=false -c core.eol=lf`): file 15 B hai dòng CRLF, không đánh dấu binary. `git diff` thô 139 B / 2 CR;
`make_patch()` cũ trả **str 137 B / 0 CR**; `git apply` **exit 0** nhưng file tái dựng **13 B**, sha256 khác target — trùng số của Codex.

**Sửa** `src/make_review_delta.py`:
- `git_bytes()`; `make_patch()` trả **bytes nguyên từ stdout Git** đến `z.writestr`. Kiểm đường dẫn trong patch dùng bản xem
  `decode("utf-8","replace")` riêng, tách dòng bằng `split("\n")` (không `splitlines()`, vốn cắt ở CR lẻ). `git()` dạng text chỉ còn
  dùng cho tên file/log.
- **Roundtrip tự động** sau khi dựng gói, và `--kiem-roundtrip [zip]`: clone `--shared --no-checkout` tạm, `GIT_INDEX_FILE` tạm +
  `read-tree <base>` + `apply --cached` bản patch **lấy từ ZIP**, so blob id + sha256 từng đường dẫn trong patch với target (xóa →
  vắng mặt). Không đụng index/config/worktree của repo chính.
- `--set-base … --giu-khai-bao PATH…` (mục 1); `.engine` vào `SKIP_SUFFIXES`. Kiểm file riêng (tên thành viên ZIP + đường dẫn
  trong patch) giữ nguyên.

**Regression** `src/test_review_delta.py` phép 14–18 (repo tạm, `git -c`, **không** `.gitattributes`, không `git config`, không binary):
thêm file CRLF; sửa file CRLF; xóa file CRLF; file trộn LF/CRLF + CR lẻ + không LF cuối; sửa + thêm file binary (NUL, `\r\n`); một
file `src/demo/` phải bị loại.

| Phép | Mã mới | Mã `45e086a` (`--packager`, `logs/review21/regression_bat_ma_cu.log`) |
|:--|:--|:--|
| 14. patch trong ZIP == `git diff --binary` từng byte | PASS (1916 B, 10 CR) | **FAIL** (1907 B, 0 CR) |
| 15. áp lên base sạch → byte/sha256 khớp target + MANIFEST (6 đường dẫn) | PASS | **FAIL**: `corrupt patch at p.diff:47` (CR lẻ thành xuống dòng), 6/6 lệch |
| 16. `--kiem-roundtrip` đạt với gói thật, **bắt** gói có patch bị mất CR | PASS | FAIL (chưa có tính năng) |
| 17. file riêng không lọt ZIP/patch | PASS | PASS |
| 18. `--giu-khai-bao` in DEPENDENCIES + mang sang lần đổi base sau | PASS | FAIL (chưa có tính năng) |

Mã mới **19/19** (13 phép cũ + 14–18). Bản sao bằng chứng lịch sử (7 file `binary` vòng 20) giữ nguyên byte.

## 3. R20-P3 — hai diễn giải (`1efae6f`, chỉ tài liệu, không đổi số)

`handoff/REVIEW20_CHI_TIET.md` (dòng "Đính chính vòng 21" + mục 4, 4a, 4c) và brief cũ: `wall_s` 12 lượt = subprocess
`[sys.executable, "src/run_tracking.py", …]` (khởi động Python con, import, đọc cache, tracking, ghi kết quả), **không** gồm
`conda run` của tiến trình cha; wall warm-up 15,9/16,9 s đo bằng `date` quanh `conda run …` nên có gồm — phạm vi khác. Câu thống kê
thay bằng "chưa thực hiện kiểm định hoặc xây dựng khoảng tin cậy; IQR chỉ mô tả độ phân tán của 6 hiệu số"; `assoc` gồm ghép cặp,
cập nhật bộ lọc và vòng đời track. `09_tinh_lai_tu_raw.log`/`tinh_lai_bench.py` vòng 20 giữ nguyên byte (còn câu cũ ở tiêu đề mục 3).
Số **+0,752 ms/frame** và giới hạn đã công bố không đổi.

## 4. Luồng trực tiếp (`f63903c`, `47ef97e`)

TensorRT chỉ có binding Python 3.6, tracker cần ≥ 3.8 → **hai tiến trình**, mỗi frame truyền ngay qua pipe:

| Thành phần | Nội dung |
|:--|:--|
| `src/nano_stream_proto.py` (cú pháp 3.6) | HELLO/FRAME/EOS/ERROR nhị phân: frame_idx, time_s, cờ unreadable/truncated, n×6 float32, thời gian stage của con, mốc `monotonic`; con ← cha: byte credit `+` / dừng `S`. EOF giữa thông điệp, magic sai, payload sai kích thước → lỗi |
| `src/nano_detect_stream.py` (con, `/usr/bin/python3` 3.6 + `LD_PRELOAD` libgomp) | tái sử dụng `nano_detector` (letterbox, `TRTEngine` cudart **đồng bộ**, `decode_nms`, `scale_boxes`, `liet_ke_anh`, `mask_regions`); thư mục ảnh hoặc video (`cv2.VideoCapture`), `time_s=(idx−1)/fps`; **cửa sổ credit K** (mặc định 4); kênh giao thức = bản sao stdout, fd 1 trỏ sang stderr; bỏ qua SIGINT (cha điều khiển dừng); kiểm chữ ký/kích thước ảnh trước `imread` (mục 5) |
| `src/nano_live.py` (cha, env `nano` 3.8, backend np) | adapter `NanoStreamDetections` → **`run_tracking.track_video()` dùng chung** với replay; lọc `det_cache.filter_dets`; cấu hình hiệu lực dựng như `run_tracking.main`; mặc định c25; ghi `data/ data_cls/ timing/ pipeline/ run_meta.json`, `--save-cache` (chỉ các frame đã vào tracker) + `replay_argv`; tegrastats **dòng thô**, xung sysfs, thiết bị trước/sau |
| `src/nano_detector.py` | chỉ thêm `t_last` (h2d/exec/d2h) ở nhánh cudart; hành vi không đổi (`test_nano_detector` 6/6) |
| `src/nano_live_kiem.py`, `scripts/nano/04–06_*.sh`, `NANO_START_HERE.md` mục 6b | khởi động/dừng, kiểm chứng trên thiết bị, đo |

Mã thoát: **0** hoàn tất · **2** lỗi trước vòng lặp (không kết quả) · **3** lỗi giữa luồng (con chết/ERROR/EOF không EOS, mất frame,
ảnh hỏng, chạm trần `det_max_det`, thiếu frame so với khai báo) — ghi đủ phần đã xử lý, **không** "hoàn tất" · **4** dừng theo yêu cầu.
Không cần GT; `--regions` (vùng bỏ qua từ nhãn) chỉ cho chế độ đối chiếu, ghi cờ + sha256 vào metadata.

**Khởi động / dừng trên Nano** (từ gốc gói `~/nano_run/v7/nano_transfer`, engine `~/nano_run/engine/yolov8n_fp16.engine`):

```bash
ENGINE=~/nano_run/engine/yolov8n_fp16.engine bash scripts/nano/04_live_nano.sh cv  results_nano/live/<ten-moi>
ENGINE=~/nano_run/engine/yolov8n_fp16.engine bash scripts/nano/04_live_nano.sh ekf results_nano/live/<ten-moi> --source data/smoke/MVI_39031_f0001-0300.mp4
touch results_nano/live/<ten-moi>/STOP        # hoặc Ctrl+C -> dừng có kiểm soát, mã 4
```

**Test PC** `src/test_nano_stream.py` (14 phép; đăng ký `run_all_tests`): **14/14 PASS**, 1 SKIP (SIGINT không gửi riêng được trên
Windows). Phép 14 dùng detector **thật** `TorchPtInfer` (yolov8n.pt, CPU, 8 frame DETRAC) → live == replay từ cache.

## 5. Chạy thật trên Nano

Thiết bị: Jetson Nano 4 GB, L4T R32.7.5, thẻ 1 (`mmcblk1p1` rw), MAXN, `schedutil`; Python hệ thống 3.6.9 (TensorRT 8.2.1.9,
numpy 1.19.5, **OpenCV 3.2.0**), env `nano` 3.8.20. Giờ log Nano = PC + 1 h.

1. **Gói v6** `b0712b0f…` (63.851.330 B): manifest 553/553, file phía con biên dịch được bằng Python 3.6.9 thật, `nano_env_check` ĐẠT.
   `test_nano_stream --python-detector /usr/bin/python3`: **12/14** — (a) phép 7 giết con **trước** HELLO vì Python 3.6 + cv2 khởi
   động chậm hơn PC (chương trình báo đúng `loi_khoi_tao`, mã 2; test sai thời điểm); (b) phép 9: **OpenCV 3.2 abort cả tiến trình**
   (`std::out_of_range`) với BMP rác thay vì trả `None` (`logs/review21/03_test_tren_nano.log`). Sửa `47ef97e`: con kiểm chữ ký/kích
   thước ảnh trước `imread` (không đạt → frame không đọc được); test đợi HELLO; thêm ca BMP bị cắt.
2. **Gói v7** `6dc4100d…` (63.852.880 B) → `~/nano_run/v7` (v5, v6 giữ nguyên): manifest 553/553, compile 3.6/3.8 OK, env check ĐẠT.
   `test_nano_parity` 20/20, `test_nano_run_cache` 9/9, **`test_nano_stream` 14/14** (kể cả 5b SIGINT, 9 ảnh hỏng), 1 SKIP (torch).
3. **Engine** FP16 build trên Nano (`01_build_engine.sh`, trtexec, 483 s, PASSED): 12.626.138 B, sha256 `c32690cd…`, ONNX `a66a5e03…`.
   `trtexec` GPU compute trung vị 67,96 ms (mô tả, không phải luồng).
4. **Mất nguồn**: khoảng 11:52 (giờ Nano) máy tắt không sạch (`last -x`: `crash`), khởi động lại 11:53; ext4 phục hồi journal (15 orphan
   inode), `errors_count` 0, không có thay đổi gói. Người dùng xác nhận Nano bị sập nguồn. Xảy ra **sau** các test v6 và **trước** khi
   có engine/v7; lần build engine đầu vì thế không chạy. Mọi thứ dùng để kết luận (v7, engine, parity, đo) đều tạo và kiểm **sau** lần
   khởi động lại (`logs/review21/06b_khoi_dong_lai.log`, `07_sau_mat_nguon.log`). Không có lần khởi động lại nào trong lúc kiểm chứng/đo.

### 5a. Kiểm chứng trước khi đo (`05_parity_live.sh` → `nano_live_v7/results_nano/live_v7/parity/`): **10/10 ĐẠT** + 3 mô tả

| Phép | Kết quả |
|:--|:--|
| A. giả lập (300 JPG thật + cache smoke) CV / EKF == `pc-ref` smoke | byte giống hệt (LF): `e3aa27bb…` / `f22f615a…` = hash Mốc 1 |
| B. **TensorRT thật** CV: live → `--save-cache` → replay `run_tracking` bằng chính `replay_argv` → `nano_parity` | 300/300 frame, 1272 hàng, 14 track; parity exit 0, **byte giống hệt** (`15ab38b4…`) |
| B. TensorRT thật EKF | 300/300, 1270 hàng, 15 track; parity exit 0, byte giống hệt (`238dbb2a…`) |
| D1. 30 ảnh đen | exit 0, 0 hàng MOT, timing đủ 30 frame (vòng đời vẫn cập nhật) |
| D2. mp4 smoke (OpenCV 3.2 FFMPEG) | 300 = khai báo, hết nguồn, hoàn tất (chỉ kiểm chức năng) |
| D3. stop-file giữa luồng | mã 4, 56 frame xử lý, 1 frame đang bay được xả (không xử lý), con thoát 0 (EOS dừng), replay phần đã xử lý khớp |
| D4. SIGINT (Ctrl+C) vào cha | mã 4, 50 frame, con thoát 0 |
| D5. `kill -9` con (PID từ `detector.pid`) | mã 3, `loi_tien_trinh_con`, 49 frame được ghi, **không** hoàn tất |
| D6. engine sai đường dẫn | mã 2, không có kết quả |

Tolerance giữ nguyên (bbox 1e-3, conf 1e-6, ID toàn sequence), không nới. **Mô tả, không phải tiêu chí**: detection TRT (lượt đối chiếu
`--regions`) so với cache PC PyTorch ở c25: 584 vs 581 box, ghép 578 (0,990/0,995), IoU tb 0,989, |Δconf| tb 0,009 (max 0,19), trùng lớp
0,993; mức thô 869 vs 845, IoU tb 0,983. Detection TRT của lượt CV và lượt EKF **giống hệt bit** (2938 box). Chế độ sử dụng thông
thường (không `--regions`) cho nhiều track hơn (1272 hàng so với 545 của `pc-ref`) vì các vùng UA-DETRAC bỏ qua không bị tô; hai con
số này **không** so với nhau như độ chính xác.

### 5b. Đo toàn luồng (`06_bench_live.sh`, điều kiện công bố trước `42dff0c`, sha256 `640890b8…`) — 6/6 lượt exit 0, không loại lượt

300 JPG 960×540, không `--regions`, không render, K = 4, c25; khởi tạo đo riêng; tóm tắt bỏ 25 frame đầu; thứ tự CV, EKF, EKF, CV, CV, EKF,
nghỉ 20 s; MAXN, `schedutil`, không `jetson_clocks`. Chi tiết từng lượt: `nano_live_v7/results_nano/live_v7/bench/bang_luot.csv`, `tong_hop_bench.json`.

| Lượt | Cấu hình | FPS vòng lặp (300 frame) | FPS ổn định (bỏ 25) | trễ/frame trung vị (p90) ms | khởi tạo s |
|--:|:--|--:|--:|--:|--:|
| 1 | CV | 9,11 | 9,06 | 118,6 (121,8) | 8,5 |
| 2 | EKF | 9,37 | 9,37 | 114,4 (118,6) | 8,4 |
| 3 | EKF | 9,44 | 9,40 | 114,2 (118,2) | 9,3 |
| 4 | CV | 9,36 | 9,32 | 114,8 (118,2) | 8,6 |
| 5 | CV | 9,29 | 9,28 | 114,6 (117,5) | 9,0 |
| 6 | EKF | 9,29 | 9,27 | 115,8 (118,6) | 8,7 |

Theo cấu hình (trung vị, min–max của 3 lượt): **CV 9,29 FPS (9,11–9,36)**, **EKF 9,37 FPS (9,29–9,44)**; FPS ổn định CV 9,28, EKF 9,37.

Trung vị theo frame (6 lượt, sau warm-up): con — decode 11,4–11,6 ms (lượt 1: 16,3), preprocess 13,3–13,5, **infer 73,6–74,2**
(H2D 2,5–2,6 / execute 68,4–68,8 / D2H 1,7), postprocess 7,2–7,4, chờ credit ≈ 0; truyền pipe ≈ 0,37; cha — chờ đọc 98–103, lọc 0,62–0,65,
`tracker.update()` **CV 5,65–5,76 / EKF 6,39–6,56**, xử lý ở cha CV 6,0–6,1 / EKF 6,7–6,9; khoảng giữa hai frame xong 106–111 ms.
Ghi kết quả 0,10–0,11 s. Thiết bị: RAM tối đa ~2,67 GB, swap 15–16 MB, CPU 1224–1479 MHz, GR3D tb 62–67 % (max 99 %), nhiệt CPU ≤ 36,5 °C,
GPU ≤ 33 °C (tegrastats thô trong từng thư mục lượt).

**Cách đọc (mô tả)**: thông lượng bị giới hạn bởi tiến trình detector (~106 ms/frame tuần tự: decode + pre + infer + post); tracker ở
tiến trình cha chạy song song với frame kế tiếp nên chênh lệch `tracker.update()` giữa EKF và CV (~0,8 ms) **không** thể hiện trong FPS ở
cấu hình này. Con không phải chờ credit (cha nhanh hơn). Không kiểm định, không khoảng tin cậy; 6 lượt trên 300 frame của một sequence.

## 6. Phân loại bằng chứng

- **Chạy thật trên Nano**: engine, test thiết bị (v6 phát hiện 2 lỗi, v7 đạt), parity A/B, biên D1–D6, mô tả C, 6 lượt đo.
- **Chỉ kiểm trên PC**: sửa packager + regression (mã cũ fail), `TorchPtInfer` live→replay, full suite `lần 1 33/34 (`test_runner_backend_np` FAIL: Smart App Control chặn `fastparquet\cencoding.pyd` lúc 11:31:26, `logs/review21/09_*`; chạy riêng 14/14), lần 2 **34/34 PASS** (`logs/review21/tests_20260928-1133.log`), cùng mã`.
- **Chưa làm / còn chặn**: camera, hiển thị/render trực tiếp, toàn sequence 1470 frame qua luồng trực tiếp (gói chỉ mang 300 ảnh
  smoke), đo công suất; nguyên nhân mất nguồn chưa rõ (nguồn điện/thẻ) — rủi ro ổn định cho các lần đo dài.

## 7. Giới hạn

FPS là của **luồng hai tiến trình không render** trên 300 frame smoke, cấu hình c25, không `--regions`; không khái quát ra toàn dataset,
cảnh khác hay camera. Không cộng/so với latency detector vòng 16 hay benchmark tracker vòng 20 (phạm vi khác). `schedutil` không khóa xung;
có desktop chạy nền. Lượt 1 có decode chậm hơn (16,3 ms) — không loại. `--kiem-roundtrip` chỉ kiểm các đường dẫn trong patch; file bị loại
có chủ ý (NPZ, cache live `.npz`) được khai báo hash, không tái dựng.

## 8. File

Mã: `src/make_review_delta.py`, `src/test_review_delta.py`, `src/nano_stream_proto.py`, `src/nano_detect_stream.py`, `src/nano_live.py`,
`src/nano_live_kiem.py`, `src/test_nano_stream.py`, `src/nano_detector.py` (t_last), `src/run_all_tests.py`, `scripts/nano/04–06_*.sh`,
`scripts/nano/NANO_START_HERE.md`. Bằng chứng: `handoff/logs/review21/` (00–08, repro, regression, test, điều kiện đo, hai lượt full suite, `09_*` sự kiện
Smart App Control) và
`handoff/logs/review21/nano_live_v7/` (156 file từ Nano, sha256/byte khớp nguồn; `.gitattributes` cục bộ `* -text` giữ byte CRLF của
`bang_luot.csv`; 30 BMP đen tổng hợp **không** chép về, hash ở `08_khong_chep_ve.log`).
