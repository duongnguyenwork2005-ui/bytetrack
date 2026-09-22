# Phụ lục vòng 17 — xử lý review `20207ec` (N15-P1/P2/P3): preflight chặn mọi đầu vào/môi trường không hợp lệ trước khi tracker chạy

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Review: `handoff/CODEX_REVIEW_20207ec.md` (lấy từ `~/Downloads`, lưu vòng này). Base review
**giữ `a5eec39`**. Trạng thái trước: HEAD `343fa91`, tree sạch (`logs/review17/trang_thai_truoc.log`).

> **Thứ tự thời gian cần nói rõ**: review `20207ec` được viết trên delta vòng 15 (trước khi có thiết bị). Ở vòng 16 Nano
> **đã được chạy thật** theo yêu cầu của người dùng (gói v3/v4, mốc 1 parity ĐẠT lệch 0, mốc 2 chạy trọn —
> `handoff/REVIEW16_CHI_TIET.md`). Các sửa ở vòng này **không đổi tracker** và không đổi kết quả đó; chúng chỉ siết cổng an toàn
> (preflight). Gói v5 dựng ở vòng này **chưa** chạy trên Nano; sẽ chuyển sau khi Codex nghiệm thu.

## 1. Tái hiện trên mã `20207ec` (`logs/review17/repro_truoc_khi_sua.log`)

`nano_run_cache.py`/`nano_env_check.py` tại HEAD y hệt `20207ec` (`git diff` rỗng). Chạy trong bản giải nén sạch của gói v4:

| Finding | Kịch bản | Quan sát trên mã cũ |
|:--|:--|:--|
| N15-P1 | sidecar smoke `{ bad json` → `nano_run_cache --tag badside` | **exit 1**, thư mục `NANO-smoke/badside-{cv,ekf}` **dở dang** (tracker đã chạy rồi mới `JSONDecodeError`) |
| N15-P2 | interpreter không có `lap` (venv scratch numpy 1.24.4/pandas 1.5.3, **không lap**) | `nano_env_check` exit 1 nhưng `nano_run_cache` **vẫn vào vòng smoke**, exit 1 |
| N15-P3 | xóa 2 NPZ + 4 `pc-ref/data/MVI_39031.txt` → `nano_env_check.py` | **exit 0, `DAT (1-4)`**, in "pc-ref-* (0 file)" |
| N15-P3b | chỉ xóa 2 NPZ | exit 0, `DAT (1-4)` |

## 2. Sửa (`eeb1daf`)

| File | Thay đổi |
|:--|:--|
| `src/nano_preflight.py` (**mới**, cú pháp 3.6, chỉ numpy; vermin: tối thiểu 3.3) | **Một** hợp đồng đầu vào dùng chung: đúng hai cache theo tên (`meta.json` parse + khóa `conf/max_det/classes`; `MVI_39031.npz` đọc được, đủ 7 mảng, cùng độ dài ≥ 1, `data (N,6)`, Σ`n_per_frame` = N, `frame_idx`/`time_s` tăng ngặt, `orig_shape (2,) > 0`; sidecar parse được và khớp `video`/`n_frames`/`n_boxes`), **đủ bốn** file `pc-ref` đọc được (≥ 1 dòng, mỗi dòng ≥ 6 trường số, frame ≥ 1), 2 YAML, 3 file mã. Trả `loi` (rỗng ⇔ đạt) + mô tả |
| `src/nano_env_check.py` | mục 4 gọi validator chung; **exit 0 chỉ khi** 1–4 đều đạt (N15-P3). Vẫn chạy được bằng Python 3.6 hệ thống |
| `src/nano_run_cache.py` | bước 0: validator chung (N15-P1: sidecar hỏng → exit 2 **trước** hash/tracker, không tạo `<tag>-*`) + `nano_env_check.py` **nonzero = lỗi đầu vào → exit 2**, tail ghi vào log/JSON `0_env_check` (N15-P2); backend np import với torch/ultralytics chặn; giữ hash `pc-ref-*` trước/sau, chỉ ghi `<tag>-{cv,ekf}` |
| `src/test_nano_run_cache.py` (**mới**), `run_all_tests.py` | regression 9 phép (mục 3) |
| `src/make_nano_transfer.py` | `verify` thêm negative (i) thiếu pc-ref, (ii) sidecar hỏng, (iii) thiếu `lap`, (iv) `nano_env_check` thiếu NPZ+pc-ref, và kiểm cây `pc-ref-*` không đổi sau tất cả |
| `scripts/nano/NANO_START_HERE.md` | mô tả hợp đồng preflight và điều kiện `exit 0` |

Không đổi `run_tracking.py`, `det_cache.py`, tracker, cấu hình khóa, GT, kết quả, `v2`, `pc-ref-*`, file riêng.

## 3. Regression `src/test_nano_run_cache.py` — entry point thật trong thư mục dạng gói tạm

Gói tạm: `src/` + `configs/` + `third_party/ultralytics_trackers/` chép từ repo; 2 cache **tổng hợp** đúng tên hợp đồng (3 và
5 frame, có frame rỗng, box conf thấp bị lọc ở c25); 4 `pc-ref-{cv,ekf}` tạo **bằng chính `run_tracking.py`** (np, chặn
torch); `reference/pc_ref_sha256.txt`.

| # | Phép kiểm | Mới (`eeb1daf`) | Cũ (`20207ec`, bản sao test không dừng, `regression_bat_ma_cu.log`) |
|:--|:--|:--|:--|
| 1 | hợp lệ `--tag kiem`: exit 0, 4 parity ĐẠT, 4 thư mục `kiem-*`, hash `pc-ref-*` trước == sau (tính độc lập), khớp bản đóng gói | PASS | PASS |
| 2 | thiếu `NANO-full/pc-ref-ekf/data/MVI_39031.txt` → exit 2, không `bad1-*`, không bước `2_*` | PASS | PASS |
| 3 | (N15-P1) sidecar `{ bad json` → exit 2 trước khi chạy tracker, không `bad2-*`, log nêu "sidecar" | PASS | **FAIL** (exit 1, `bad2-cv/ekf` dở dang) |
| 4 | (N15-P2) thiếu `lap` (gói giả `lap.py` ném `ImportError` đặt trước PYTHONPATH) → env_check nonzero; `nano_run_cache` exit 2, không tracker, không `bad3-*`, JSON có tail nêu `lap` | PASS | **FAIL** (chạy tiếp, exit 1) |
| 5 | (N15-P3) `nano_env_check` xóa NPZ + tham chiếu → nonzero, không "DAT (1-4)"; chỉ thiếu NPZ → nonzero; khôi phục → 0 | PASS | **FAIL** (0/0/0) |
| 6 | sidecar sai schema (`n_frames` 99) → env_check nonzero, `nano_run_cache` exit 2, không `bad4-*` | PASS | **FAIL** |
| 7 | `--tag pc-ref-x` → exit 2 | PASS | PASS |
| — | cây `pc-ref-*` y hệt sau toàn bộ phép kiểm | PASS | PASS |

`test_nano_parity` 20/20, `test_review_delta` 14/14, `test_nano_detector` 6/6 (`logs/review17/tests_lien_quan.log`).
**Full suite** trên code chốt `eeb1daf`: **33/33 PASS** (`logs/review17/tests_20260922-1512.log`, exit 0; gồm `test_nano_run_cache` 9/9, `test_pc_launcher` 17/17, `test_nano_detector` 6/6); không commit/sửa source trong lúc chạy.

## 4. Gói v5 và kiểm gói

`handoff/nano_transfer_latest.zip` **v5** — **63.801.260 byte**, sha256 `ae5fd57b6fc99590cae7c4c1860ad3ce03a51c67133288f967fffd0e83331dac`,
546 thành viên, dựng tại `eeb1daf` (tree sạch). Thêm so với v4: `src/nano_preflight.py`, `src/test_nano_run_cache.py`,
`nano_env_check.py`/`nano_run_cache.py` mới, `NANO_START_HERE.md` mới. **Chưa chuyển sang Nano** (chờ Codex nghiệm thu).

Kiểm **chính file ZIP mới** trong thư mục tạm ngoài repo (`logs/nano_transfer/kiem_goi_sach{,_oldstack}.{log,json}`), hai stack:
PC (Py 3.11.9, numpy 2.4.6, lap 0.5.13, pandas 2.3.3) và cũ (numpy 1.24.4, pandas 1.5.3, lap 0.5.12, **không scipy/cv2**):

| Bước | PC | Stack cũ |
|:--|:--|:--|
| manifest 545/545; `nano_env_check` ĐẠT; import np `torch=false`; `test_nano_parity` 20/20 | ✅ | ✅ |
| **`nano_run_cache --tag kiem`** (đường hợp lệ): smoke 545 ×2 + full 5842/5845 parity ĐẠT, bit-identical ×4, `pc-ref` khớp bản đóng gói và không đổi sau | ✅ | ✅ |
| (i) thiếu `NANO-full/pc-ref-ekf/data/MVI_39031.txt` → exit **2**, không `neg1-*`, không bước `2_*` | ✅ | ✅ |
| (ii) sidecar `{ bad json` → exit **2**, không `neg2-*`, không tracker (N15-P1) | ✅ | ✅ |
| (iii) thiếu `lap` (gói giả ném ImportError) → `nano_env_check` exit 1; `nano_run_cache` exit **2**, không `neg3-*` (N15-P2) | ✅ | ✅ |
| (iv) `nano_env_check` với NPZ + tham chiếu bị xóa → exit 1, **không** in `DAT (1-4)` (N15-P3) | ✅ | ✅ |
| cây `pc-ref-*` sau **toàn bộ** kịch bản == bản đóng gói (`4d`) | ✅ | ✅ |
| `det_compare` cắt-vs-gốc 845/845; ONNX; 300 JPG + mp4; render 300 frame | ✅ | (env PC) |

Kết luận verify: **ĐẠT** — chỉ chứng minh gói đủ và các cổng chặn đúng trên PC; không chứng minh tương thích Nano (đã có bằng chứng
thiết bị riêng ở vòng 16 với gói v3/v4).

## 5. Bước tiếp theo

Chờ Codex nghiệm thu vòng này; sau đó chuyển v5 sang Nano (`~/nano_run/v5`, không ghi đè), chạy `nano_env_check` →
`nano_run_cache` (mốc 1) để xác nhận lại trên thiết bị với preflight mới; mốc 2 (`03_e2e_nano.sh`) đã có bằng chứng vòng 16.
