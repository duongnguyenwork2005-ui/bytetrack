# REVIEW_BRIEF — vòng 17: xử lý review `20207ec` — preflight chặn đầu vào/môi trường không hợp lệ trước khi tracker chạy

<!-- BASE_TARGET -->

Base review **giữ `a5eec39`**. `handoff/CODEX_REVIEW_20207ec.md` đã lưu. Không tuning/train; tracker, cấu hình khóa, GT, kết
quả UA-DETRAC/BDD, 4 run `v2`, `pc-ref-*`, file riêng **không đổi**. Chi tiết: **`handoff/REVIEW17_CHI_TIET.md`**.

**Thứ tự thời gian**: review `20207ec` viết trên delta vòng 15. Ở vòng 16 Nano **đã được chạy thật** theo yêu cầu (gói v3/v4:
mốc 1 parity ĐẠT lệch 0, mốc 2 chạy trọn — `REVIEW16_CHI_TIET.md`). Vòng này chỉ siết cổng preflight, không đổi tracker; gói
**v5 chưa chuyển sang Nano** (chờ nghiệm thu).

## Tái hiện trên `20207ec` (`logs/review17/repro_truoc_khi_sua.log`)

N15-P1 sidecar `{ bad json` → exit 1, `badside-{cv,ekf}` dở dang · N15-P2 interpreter không `lap` → env_check exit 1 nhưng
`nano_run_cache` vẫn vào smoke · N15-P3 xóa 2 NPZ + 4 tham chiếu → `nano_env_check` **`DAT (1-4)`, exit 0** (cả khi chỉ xóa NPZ).

## Sửa (`eeb1daf`)

- **`src/nano_preflight.py`** (mới, cú pháp 3.6, chỉ numpy): **một** hợp đồng dùng chung — đúng hai cache theo tên (`meta.json`
  parse + khóa; NPZ đủ 7 mảng, Σ`n_per_frame` = N, `frame_idx`/`time_s` tăng ngặt; **sidecar parse + schema**), **đủ bốn** file
  `pc-ref` đọc được (≥ 1 dòng MOTChallenge hợp lệ), YAML, mã. "Có thư mục" không đủ.
- `nano_env_check.py`: mục 4 dùng validator → exit 0 **chỉ khi** tất cả đạt (vẫn chạy được bằng Python 3.6).
- `nano_run_cache.py`: bước 0 = validator + `nano_env_check` **nonzero → exit 2** (tail vào log/JSON) + import np chặn torch;
  không hash/không tracker/không `<tag>-*` khi thiếu. Giữ hash `pc-ref-*` trước/sau.
- `make_nano_transfer.py verify`: thêm 4 negative case + kiểm `pc-ref-*` không đổi sau tất cả.

## Kiểm chứng

- **`test_nano_run_cache` 9/9** (entry point thật, thư mục dạng gói tạm, 4 `pc-ref` tạo bằng `run_tracking` np): hợp lệ → 4 parity
  ĐẠT + hash trước==sau; thiếu pc-ref / sidecar hỏng / thiếu `lap` (gói giả) / sidecar sai schema / tag `pc-ref` → exit 2, không
  output, không tracker; `nano_env_check` thiếu NPZ+tham chiếu → nonzero, không `DAT (1-4)`. Trên **mã cũ**: phép 3, 4, 5, 6
  **FAIL** (`regression_bat_ma_cu.log`). Đăng ký `run_all_tests`.
- `test_nano_parity` 20/20, `test_review_delta` 14/14, `test_nano_detector` 6/6; **full suite 33/33** trên `eeb1daf`
  (`logs/review17/tests_20260922-1512.log`), không commit trong lúc chạy.
- **Gói v5** `handoff/nano_transfer_latest.zip`: **63.801.260 B**, sha256
  `ae5fd57b6fc99590cae7c4c1860ad3ce03a51c67133288f967fffd0e83331dac`, dựng tại `eeb1daf`. Verify **chính ZIP** trong thư mục tạm,
  **2 stack** (PC; numpy 1.24/lap 0.5.12/không scipy): đường hợp lệ parity ĐẠT bit-identical ×4, `pc-ref` không đổi; (i) thiếu
  pc-ref, (ii) sidecar hỏng, (iii) thiếu `lap`, (iv) env_check thiếu NPZ+pc-ref → đúng exit 2 / nonzero, không output, không
  `DAT (1-4)`; cây `pc-ref-*` == bản đóng gói sau tất cả → **ĐẠT**.

## Trạng thái Nano / bước tiếp theo

Gói v5 **chưa chạy trên Nano**; bằng chứng thiết bị hiện có là vòng 16 (v3/v4). Commit: `eeb1daf` mã + test; sau đó log/tài liệu.
Delta từ `a5eec39` không chứa ZIP Nano. Sau nghiệm thu: chuyển v5 sang `~/nano_run/v5`, chạy `nano_env_check` → `nano_run_cache`
(mốc 1) xác nhận lại với preflight mới.
