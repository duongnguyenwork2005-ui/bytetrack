# REVIEW_BRIEF — vòng 18–20: Nano v5 — Mốc 1 ĐẠT, benchmark tracker CV↔EKF từ cache, đóng benchmark (sao lưu + tính lại)

<!-- BASE_TARGET -->

Base review **giữ `a5eec39`**. Delta từ đó gồm vòng 17 (preflight, chờ nghiệm thu) và các vòng **18–20: chỉ thêm log, bằng chứng và tài liệu**.
Từ `eeb1daf` đến nay **không có dòng mã nào đổi** (`git diff eeb1daf HEAD` chỉ chạm `handoff/`, `CODEX_HANDOFF.md`). Tracker, YAML khóa, GT,
kết quả UA-DETRAC/BDD và `pc-ref-*` không đổi. Chi tiết: **`handoff/REVIEW20_CHI_TIET.md`**, `handoff/NANO_STATUS.md` mục 4f.

## Diễn biến

- **Vòng 18** (22/9): gói v5 `ae5fd57b…` trên Nano thật → **Mốc 1 ĐẠT** (`logs/review18/`).
- **Vòng 19**: benchmark lần 1 **thất bại 0/6 cặp** do lỗi gọi lệnh (thiếu `BYTETRACK_BACKEND=np`; warm-up `--pairs 2` bị từ chối).
  Sau đó `apt upgrade` hỏng DTB; máy boot bằng **thẻ 1** (bản sao cũ), mất dữ liệu 22/9.
- **Vòng 20** (24/9): dựng lại env, **Mốc 1 lại ĐẠT** (4/4 parity lệch 0); benchmark lần 2 (`logs/review20/bench_lan2/`).
- **Đóng benchmark** (chỉ đọc trên Nano): sao lưu 95 + 3 file (SHA256 + byte khớp), tính lại từ CSV thô, kiểm thẻ.

## Benchmark tracker-from-cache (`MVI_39031`, 1470 frame, bỏ 50 frame đầu → 1420 frame đo, warm-up tách riêng)

6 cặp (3 AB + 3 BA), 12/12 lượt exit 0, không loại lượt nào. `update` = tổng `tracker.update()` (ms/frame):

| | CV (6 lượt) | EKF (6 lượt) | **EKF − CV theo cặp** |
|:--|:--|:--|:--|
| update tb | 4,695 (4,657–4,774) | 5,468 (5,414–5,512) | trung vị **+0,752**, min/max +0,640/+0,855, **6/6 dương**, ≈ +16 % |
| predict | 0,618 | 1,239 | +0,622 (6/6) |
| assoc = update − predict | 4,077 | 4,221 | +0,139 (6/6) |
| wall/lượt (gồm `conda run`) | 9,6–9,9 s | 10,7–10,9 s | +1,07 s |

Q1–Q3 của d_update: [+0,710; +0,818] (exclusive, như công cụ), [+0,737; +0,794] (inclusive). **IQR chỉ mô tả độ phân tán
của 6 hiệu số**, không phải khoảng tin cậy hay kiểm định. `logs/review20/tinh_lai_bench.py` tính lại độc lập từ `timing/*.csv`: khớp JSON, lệch 0.

**Tính đúng**: output của cả 12 lượt, warm-up và Mốc 1 cùng hash với `pc-ref` (LF) theo từng cấu hình. Parity `r1`/`r6` ĐẠT lệch 0.
`pc-ref-*` và `nano-*` Mốc 1 có hash trước == sau. Cờ `bit_identical_data_txt=False` chỉ do CRLF (pandas trên Windows) so với LF (Nano).

**Điều kiện**: MAXN, `schedutil`, không `jetson_clocks`; nhiệt max 42–44 °C; swap 0; desktop + `update-manager`/`packagekitd` chạy nền.

## Thẻ SD và sao lưu

Thẻ 1 `mmcblk1p1` rw, 27G/58G, ext4 `errors_count` 0, không lỗi I/O. **Sau** benchmark có hai lần khởi động lại (10:13 sạch;
≈10:36 không sạch → ext4 phục hồi journal, vẫn 0 lỗi; nguyên nhân không có trong log). Băm lại nguồn sau đó: không đổi. Không có thay đổi gói.

## Giới hạn

Tracker từ cache, **không phải FPS end-to-end**, không cộng detector; một sequence, 6 cặp; không có xung nhịp CPU theo lượt hay
tegrastats thô; thẻ 1 từng lỗi fsck; không suy sang HOTA/IDF1. TensorRT/Mốc 2 chưa dựng lại trên thẻ 1 (bằng chứng vòng 16 trên PC).

Không sửa mã trong đợt này nên không chạy lại full suite (33/33 trên `eeb1daf`). Delta không chứa ZIP Nano, cache NPZ, weights hay video.
**Áp patch**: dùng `git -c core.autocrlf=false -c core.eol=lf apply`. `make_review_delta` làm patch mất CR của file `-text` mang CRLF;
7 file như vậy được đánh dấu `binary` để tránh lỗi này (phụ lục mục 9, chưa sửa mã).
