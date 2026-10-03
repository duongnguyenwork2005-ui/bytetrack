# Vòng 40 — kế hoạch sửa và kiểm R39-F1 (ghi TRƯỚC khi chạy kiểm)

Người dùng duyệt ngày 2026-10-03: chỉ sửa R39-F1 của review `1454b3d`, kiểm hồi quy ngắn trên fixture CPU, bàn giao rồi
dừng. Không train dài, không chạy dataset đầy đủ, không lặp training GPU. Base review giữ `b83dc12`.

## 1. Tái hiện trước khi sửa (đã chạy)

`01_repro_truoc_sua.{log,json}`: chạy nguyên `REPRO_REVIEW_1454b3d.py` của Codex (sha256 trong log) trên mã `1454b3d`.
- Kết quả **exit 1, `bug_count=3`**; ca `clean_control` đạt.
- Ba biến thể lỗi đều cho "dòng 3 hỏng ở GIỮA": JSON cuối ghi dở không newline, JSON cuối ghi dở có newline, JSON cuối hợp
  lệ thiếu newline.

Đường gọi thật đã đọc trong mã:
1. **Preflight**: `chuan_bi_tiep_tuc` gọi `doc_nhat_ky`. Hàm này bỏ qua dòng cuối hỏng, chỉ cảnh báo, không sửa file.
2. **Sao lưu**: `sao_luu_truoc_tiep_tuc` sao lưu checkpoint và CSV nhưng **không** sao lưu, không xử lý nhật ký.
3. **Lần append đầu tiên**: `GhiLuotDayDu.sau_chuan_bi` ghi `bat_dau`. Nếu train lỗi trước callback, lần append đầu là
   `ket_thuc` trong `main`.
4. **Ghi epoch**: `luu_model` ghi `epoch_da_luu`.

## 2. Sửa

Chỉ sửa ở **bước chuẩn bị ghi của resume thật** (bước 2), vì bước này chạy trước mọi lần append và không chạy ở dry-run.

- `phan_tich_duoi_nhat_ky(b)` (mới, chỉ đọc), cùng quy tắc với `doc_nhat_ky`:
  - `khong`: dòng nội dung cuối hợp lệ và có newline;
  - `them_newline`: JSON cuối **hợp lệ** nhưng thiếu newline → **giữ sự kiện**, chỉ thêm `\n`;
  - `cat_duoi`: dòng nội dung cuối không parse được (có hoặc không newline) → cắt từ đầu dòng đó tới hết file;
  - dòng hỏng ở **giữa** → `LoiTiepTuc`, không tự sửa.
- Preflight thêm `nhat_ky_duoi` = kế hoạch trên (chỉ đọc). `--dry-run --tiep-tuc` in kế hoạch và không ghi gì.
- `sua_duoi_nhat_ky` (mới), gọi trong `sao_luu_truoc_tiep_tuc`, sau khi sao lưu checkpoint:
  1. đọc nguyên byte; sha256 phải trùng lúc preflight, nếu không thì dừng;
  2. **luôn** sao lưu nguyên byte vào `khoi_phuc/truoc_tiep_tuc_N/nhat_ky_luot.jsonl` (ghi qua file tạm, kiểm sha256);
  3. sửa bằng file tạm + fsync + `os.replace`: nếu bị ngắt giữa chừng, nhật ký cũ còn nguyên;
  4. đọc lại: không cảnh báo, kết thúc bằng newline, sự kiện hợp lệ **trùng từng sự kiện** với trước khi sửa;
  5. biên nhận `xu_ly_nhat_ky.json` cùng thư mục; cùng nội dung được ghi vào sự kiện `bat_dau` của lần tiếp tục
     (`tiep_tuc.sao_luu.nhat_ky`).

  Không thêm sự kiện nào, không tạo lịch sử giả.
- **Không đổi** `ghi_nhat_ky` / `doc_nhat_ky`, cơ chế resume của Ultralytics hay các kiểm checkpoint.
- Hệ quả biết trước: script cấp hàm của Codex vẫn sẽ cho exit 1, `bug_count=3` sau khi sửa, vì nó gọi thẳng hai hàm
  này. Review đã nêu: khi sửa ở đường chuẩn bị riêng, **test tích hợp** là tiêu chí nghiệm thu.

## 3. Kiểm và tiêu chí

Phân biệt ba loại phép kiểm:
- **CHẠY THẬT**: `train_detector.py` qua subprocess, ultralytics thật, CPU, fixture;
- **SPY**: `ultralytics.YOLO` bị thay để chặn `train()`;
- **MOCK**: thay một hàm hệ thống.

| Mã | Loại | Tiêu chí đạt |
|:--|:--|:--|
| U9 | hàm thật | `phan_tich_duoi_nhat_ky` cho đúng hành động ở 5 ca (sạch, ghi dở không/có newline, byte NUL, hợp lệ thiếu newline), đúng ranh giới byte; dòng hỏng ở giữa → `LoiTiepTuc` |
| U10 | hàm thật, thư mục tạm | `sua_duoi_nhat_ky` ở 5 ca: bản sao nguyên byte; nhật ký sau sửa đúng byte mong đợi; sự kiện hợp lệ trùng; biên nhận = giá trị trả về; không còn file tạm |
| U10b | **MOCK** `os.replace` lỗi khi thay nhật ký | `LoiTiepTuc`; nhật ký cũ còn nguyên byte; bản sao đã có |
| S20 | **SPY** | Dòng hỏng ở giữa → exit 2, không gọi `train()`, thư mục lượt không đổi |
| N0 | CHẠY THẬT | Lượt gốc `yolov8n_NK` bị kill ngay sau epoch 1 |
| N-x-0a | CHẠY THẬT | Mỗi ca: dry-run exit 0, **không đổi file** (so sha256 từng file) |
| N-x-0b | CHẠY THẬT | Dry-run báo đúng hành động (`khong` / `cat_duoi` / `cat_duoi` / `them_newline`) |
| N-x-A | CHẠY THẬT | Resume thật, kill sau epoch 2: nhật ký **đọc lại được**, không cảnh báo, sự kiện hợp lệ cũ giữ nguyên + `bat_dau` lần 1 + `epoch_da_luu` 2 |
| N-x-B | CHẠY THẬT | Bản sao nhật ký trước sửa trùng từng byte; biên nhận = bản ghi trong `bat_dau`, đúng hành động và số byte bỏ |
| N-x-C | CHẠY THẬT | `last_e001.pt`, `nguon_tiep_tuc_1.pt` = e001 gốc; các file cũ được lưu trước khi ghi tiếp trùng gốc |
| N-x-D | CHẠY THẬT | Preflight tiếp theo exit 0 "tiếp tục từ epoch 3/3"; resume lần 2 exit 0 tới hết; nhật ký đọc lại đủ, kết thúc `hoan_tat` |
| N-x-B2 | CHẠY THẬT | Bản sao trước lần 2 trùng byte; biên nhận `khong` |
| N-giua | CHẠY THẬT | Dòng hỏng ở giữa: dry-run và resume thật đều exit 2 "hong o GIUA"; thư mục lượt không đổi |

`x` là 4 ca: `sach`, `ghi_do_khong_newline`, `ghi_do_co_newline`, `hop_le_thieu_newline`.

**Kết quả dự kiến**
- **Mã mới**: `src/test_tiep_tuc_huan_luyen.py` đạt toàn bộ **102 phép** = 68 phép cũ + U9, U10, U10b, S20, N0,
  4 × 7 phép N, N-giua.
- **Mã cũ** (`git archive 1454b3d src`, chạy `--chi-nhom-n --ma-nguon <src cũ>`, tổng 30 phép):
  - đạt 12: N0, 0a × 4, A của ca sạch, C × 4, D của ca sạch, N-giua;
  - không đạt 18:
    - **A × 3 và D × 3 ở ba biến thể** = đúng lỗi R39-F1;
    - 0b × 4, B × 4, B2 × 4, vì mã cũ không báo kế hoạch và không sao lưu nhật ký (yêu cầu mới).
- **Suite bị ảnh hưởng trực tiếp** (runner đổi): `test_detector_5lop` 92/92 và `test_pilot_chan_doan` 34/34, như vòng 39.
- **Script Codex sau sửa**: dự kiến vẫn exit 1, `bug_count=3` (mục 2); chỉ ghi nhận.

Không dùng tolerance. Lỗi nào phát sinh thì giữ log, tìm nguyên nhân có bằng chứng, sửa, rồi chạy lại với tên log mới.
Không sửa hay commit trong lúc test đang chạy.
