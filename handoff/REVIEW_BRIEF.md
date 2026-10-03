# REVIEW_BRIEF — vòng 41: sửa R40-N1 / R40-N2 (đuôi ghi dở của `results.csv` / `theo_batch.csv` khi resume)

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW41_CHI_TIET.md`**. Kế hoạch đăng ký trước: `handoff/logs/review41/02_ke_hoach_kiem_r40n.md`.

**Phạm vi** (người dùng duyệt): sửa R40-N1 và R40-N2 của review `421dd53`, kiểm hồi quy ngắn trên fixture CPU, bàn giao,
dừng. Không train dài, không GPU, không dataset đầy đủ.

## Review `421dd53` — đã ghi nhận

PASS WITH LIMITATIONS, đóng R39-F1 trong phạm vi sửa nhật ký; không có nghĩa toàn bộ resume đã an toàn. Bằng chứng khớp
52/52. Base → **`421dd53`**, giữ đủ 13 khai báo.

## Tái hiện trước khi sửa

- `observe_csv_risks.py` của Codex chạy nguyên trên `421dd53`: **`known_risks_reproduced = 3`**, exit 0 (chỉ nghĩa là
  phép quan sát chạy xong).
- Đường gọi thật: `save_metrics` (Ultralytics) và `het_epoch_train` ghi nối; bước sao lưu chỉ cắt `results.csv` theo
  epoch bằng `int(float(...))`, không kiểm cột, bỏ qua `theo_batch.csv`.

## Sửa — chỉ ở bước chuẩn bị ghi của resume thật

- `phan_tich_duoi_csv` (chỉ đọc): kiểm header và từng hàng theo định dạng thật (đủ cột, ô bắt buộc đúng kiểu, ô tùy chọn
  được trống).
  - hàng cuối không hợp lệ → cắt từ đầu hàng;
  - hàng cuối hợp lệ nhưng thiếu newline → giữ, thêm đúng ký tự xuống dòng;
  - hỏng ở giữa, header sai, file rỗng → `LoiTiepTuc`, exit 2 trước khi train.
- `sua_duoi_csv` (trong bước sao lưu): sha256 phải trùng preflight và bản sao; ghi qua file tạm + fsync + `os.replace`;
  đọc lại kiểm; biên nhận `xu_ly_<file>.json`, cũng ghi vào `bat_dau`.
  - `results.csv`: giữ quy tắc epoch, áp trên hàng đã kiểm dạng;
  - `theo_batch.csv`: giữ mọi hàng lịch sử hợp lệ và cột `lan`.
- `--dry-run --tiep-tuc` in kế hoạch CSV, không ghi gì.

## Kiểm — ĐÃ CHẠY

| Phép | Kết quả |
|:--|:--|
| `test_tiep_tuc_huan_luyen`, mã mới | lần 2: **138/138**. Lần 1 không hoàn tất vì **máy tự ngủ do pin cạn** (log System); không đổi mã |
| Nhóm C trên **mã cũ** `421dd53` | **11/31**, đúng dự đoán: NUL làm resume kẹt ở exit 1; đuôi `1` và hàng thiếu newline làm **mất hàng epoch 2**; `theo_batch` sai cột; không từ chối ca hỏng ở giữa |
| `test_detector_5lop` / `test_pilot_chan_doan` | 92/92 / 34/34 |
| Script Codex sau sửa | **1**: hai ca `results.csv` đã hết; ca `theo_batch` gọi thẳng writer, bỏ qua bước chuẩn bị (đúng dự đoán) |
| Đối chứng: ca `theo_batch` đi qua bước chuẩn bị | **0** |

Nhóm C chạy thật qua `train_detector.py` (yolov8n, CPU): 4 ca × {dry-run, resume + ghi tiếp thật, bản sao/biên nhận,
checkpoint, resume lần 2} + 2 ca từ chối. Kết quả: hai CSV đọc được, đúng cột, `results` [1, 2, 3], đủ lịch sử
`theo_batch`. Nhóm N (R39-F1) vẫn đạt; lỗi `os.replace` (MOCK) không làm mất bản gốc.

## Giới hạn

- Hàng bị cắt đúng sau một ô mà vẫn hợp lệ cú pháp thì không phân biệt được với hàng hoàn chỉnh thiếu newline.
- `results.csv` không được Ultralytics fsync; mất hẳn hàng thì không khôi phục được.
- Writer không đổi; chưa mô phỏng mất điện thật.

## Đề xuất ngoài phạm vi — chờ duyệt

Guard ở hàm append; khóa thư mục lượt.

## Trạng thái

R40-N1/N2 đã sửa và kiểm; **chờ Codex review**. Training dài, số epoch, ngân sách và thời điểm **vẫn chưa duyệt**.

**Nên xem**: `logs/review41/01_*`, `03_*_lan2.log`, `04_*`, `06*`.
