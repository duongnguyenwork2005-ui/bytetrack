# REVIEW_BRIEF — vòng 22: sửa N21-F1/F2/F3, kiểm chứng và đo lại luồng trực tiếp trên Nano

<!-- BASE_TARGET -->

Base **`45e086a`** (review `handoff/CODEX_REVIEW_9f210e2.md`). Mã lõi tracking, benchmark vòng 20, UA-DETRAC/BDD và dữ liệu vòng 21
không đổi. Chi tiết: **`handoff/REVIEW22_CHI_TIET.md`**.

## Ba finding — tái hiện trước/sau qua entry point thật (`logs/review22/01_*`, `02_*`)

| | Mã `9f210e2` | Sau sửa |
|:--|:--|:--|
| **F1** cha chuẩn bị chậm 1 s | cửa sổ 0,067 s, frame thật trải 1,094 s (118,8 vs 7,31 FPS) | 0,1708 s vs 0,172 s; frame 1 đọc sau BẮT ĐẦU |
| **F2** `--cache-root` riêng | `replay_argv` → exit 1 (`FileNotFoundError`) | exit 0, `nano_parity` ĐẠT |
| **F3** YAML thiếu/hỏng | detector đã chạy, exit 1, không `run_meta`; `--set` sai được nhận | mã 2, không khởi động detector |

- **F1**: giao thức 2 — con gửi HELLO rồi chờ lệnh BẮT ĐẦU; cha chuẩn bị, bật timer rồi mới gửi. Thời gian tách `khoi_tao_s`,
  `chuan_bi_cha_s`, `xu_ly_s`/`fps_xu_ly`, `ket_thuc_s`, `ghi_s`, kèm kiểm cửa sổ bằng mốc `monotonic`.
- **F2**: `nano_replay.py` đặt `det_cache.CACHE_ROOT` rồi gọi chính `run_tracking.main()`; `replay_argv` mang root tuyệt đối.
- **F3**: kiểm YAML/`tracker_type`/`--set` (dạng, khóa, kiểu)/tham số trước spawn. Sau spawn `try/finally` luôn reap con; lỗi trước BẮT
  ĐẦU → mã 2, giữa vòng lặp → mã 3, ghi phần đã có.
- Regression `test_nano_stream` 15–17: PC **20/20**; trên mã cũ 15–17 **FAIL**. Full suite PC: `**34/34 PASS** (`logs/review22/tests_20260928-1246.log`, trên `b0c645b`)`.

## Nano thật (gói v8 `f4da9f08…`, engine `c32690cd…` dùng lại, không khởi động lại suốt đợt)

- Test thiết bị 20/20, 9/9, **20/20** (con 3.6 ↔ cha 3.8, kể cả SIGINT và 15–17).
- **Parity 10/10**: TensorRT CV/EKF live == replay, byte giống hệt; dừng, SIGINT, `kill -9` con, engine sai → đúng mã thoát.
- **MOT replay bàn giao** cạnh MOT live để Codex tự so byte: `nano_live_v8/…/parity/B_replay_{cv,ekf}`, `D3_replay_dung`.

## Đo lại (điều kiện công bố trước `b0c645b`; 6 lượt, không loại lượt, cả 6 có mọi frame trong cửa sổ)

| | `fps_xu_ly` (3 lượt) | `tracker.update()` | infer |
|:--|:--|:--|:--|
| CV | **9,13** (9,07–9,19) | 5,57–5,78 ms | ~73,7 ms |
| EKF | **9,36** (9,20–9,42) | 6,34–6,48 ms | ~73,7 ms |

Decode OpenCV 3.2 có hai mức (~11,3 / ~16,3 ms). Tỷ lệ frame ở mức chậm dao động 3–95 % giữa các lượt và rơi vào cả 3 lượt CV, nên
chênh lệch CV/EKF **không** được diễn giải là EKF nhanh hơn. Detector vẫn quyết định thông lượng.

**Đính chính vòng 21** (giữ số cũ): tính lại từ timestamp có sẵn, frame 1 bắt đầu đọc → frame 300 xong: CV 9,10/9,34/9,27,
EKF 9,35/9,42/9,26 (lệch 0,21–0,28 %). Kết luận ~9,3 FPS vẫn đứng.

## Giới hạn

300 frame smoke, một sequence, không camera/render. Nguyên nhân hai mức decode và lần mất nguồn 28/9 chưa rõ. Delta từ `45e086a` đã kiểm
roundtrip theo byte/hash; NPZ (vòng 16, cache live) khai báo, không bàn giao.
