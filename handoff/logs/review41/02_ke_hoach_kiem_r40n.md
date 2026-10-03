# Vòng 41 — kế hoạch sửa và kiểm R40-N1 / R40-N2 (ghi TRƯỚC khi chạy kiểm chính thức)

Người dùng duyệt ngày 2026-10-03: sửa hai vấn đề CSV R40-N1 (`results.csv`) và R40-N2 (`theo_batch.csv`) của review
`421dd53`, kiểm hồi quy ngắn trên fixture CPU, bàn giao rồi dừng. Không train dài, không GPU, không dataset đầy đủ. Base đã
chuyển sang `421dd53`.

## 1. Tái hiện trước khi sửa (đã chạy)

`01_quan_sat_csv_truoc_sua.{log,json}`: chạy nguyên `observe_csv_risks.py` của Codex trên mã `421dd53`.
`known_risks_reproduced = 3`, exit 0. Exit 0 chỉ có nghĩa phép quan sát chạy xong. Ba rủi ro:
- `results.csv` đuôi byte NUL → `ValueError` thô, không phải `LoiTiepTuc`;
- `results.csv` sau epoch 1–9, đuôi `1` → được giữ thành hàng thiếu trường (`time` = None);
- `theo_batch.csv` đuôi `0,1,` + writer thật `het_epoch_train` → hàng mới dính vào, ra `0,1,1` cùng trường thừa.

Đường gọi thật (đọc mã):
- `results.csv` do Ultralytics `save_metrics` ghi nối (chế độ text, mỗi ô `%.6g`, header chỉ khi file chưa có; trên
  Windows là CRLF);
- `theo_batch.csv` do `GhiLuotDayDu.het_epoch_train` ghi nối (`csv.DictWriter`, CRLF);
- `sao_luu_truoc_tiep_tuc` sao lưu cả hai rồi chỉ cắt `results.csv` theo epoch bằng `int(float(trường đầu))`, không kiểm
  cột;
- lần ghi nối đầu tiên sau resume nằm ở epoch tiếp tục đầu tiên. Chỗ sửa là bước chuẩn bị này, chạy trước mọi lần ghi nối
  và không chạy ở dry-run.

## 2. Sửa

- `phan_tich_duoi_csv(b, kieu, ten, epoch_da_luu)` (mới, **chỉ đọc**):
  - header phải ở dòng đầu và đúng định dạng thực tế: `results.csv` bắt đầu bằng `epoch,time`; `theo_batch.csv` trùng
    `COT_BATCH`;
  - mỗi hàng dữ liệu phải đủ số cột, và các ô bắt buộc phải đúng kiểu:
    - `results.csv`: mọi ô là số (Ultralytics ghi `%.6g`), epoch nguyên ≥ 1;
    - `theo_batch.csv`: các cột `lan`, `epoch`, `i`, `ni`, `accumulate`, `ema_updates`, `batch_size` (int), `t` (float),
      `loss_huu_han` (True/False) bắt buộc; các cột loss / lr / momentum là số **hoặc ô trống hợp lệ**; `thu_tu_batch`
      trống hoặc đúng 16 ký tự hex;
  - chỉ hàng nội dung **cuối** được phép hỏng (ghi dở) → `cat_duoi` từ đầu hàng;
  - hàng cuối hợp lệ nhưng thiếu newline → `them_newline`: giữ hàng, thêm đúng ký tự xuống dòng của file (CRLF/LF; nếu đã
    có `\r` thì chỉ thêm `\n`);
  - hàng hỏng ở **giữa**, header sai hoặc file rỗng → `LoiTiepTuc`, nên preflight exit 2 trước khi train.
- Preflight thêm `csv_duoi` (kế hoạch cho cả hai file). `--dry-run --tiep-tuc` in kế hoạch và không ghi gì.
- `sua_duoi_csv` (mới), gọi trong `sao_luu_truoc_tiep_tuc` sau khi sao lưu nguyên byte:
  1. sha256 phải trùng lúc preflight và trùng bản sao;
  2. cắt đuôi dở hoặc thêm newline;
  3. với `results.csv`, **giữ quy tắc cũ**: chỉ giữ hàng epoch ≤ epoch đã lưu, áp trên các hàng đã qua kiểm định dạng;
     với `theo_batch.csv`, **giữ mọi hàng lịch sử hợp lệ** (cả cột `lan`), không cắt theo epoch;
  4. ghi bằng file tạm + fsync + `os.replace`; đọc lại và kiểm: sạch, đủ cột, hàng hợp lệ đúng như mong đợi;
  5. biên nhận `khoi_phuc/truoc_tiep_tuc_N/xu_ly_<file>.json`, cũng ghi vào `bat_dau` (`tiep_tuc.sao_luu.csv`).
- Giới hạn nhận biết (ghi rõ, không tuyên bố hơn): một hàng bị cắt đúng sau một ô mà vẫn hợp lệ cú pháp thì không phân biệt
  được với hàng hoàn chỉnh thiếu newline. Ví dụ: số ở cột cuối `results.csv` bị cắt bớt chữ số; hàng `theo_batch.csv` có
  `thu_tu_batch` trống bị cắt ngay sau dấu phẩy cuối.

## 3. Kiểm và tiêu chí

Ba loại phép kiểm:
- **CHẠY THẬT**: `train_detector.py` qua subprocess, ultralytics thật, CPU, fixture;
- **SPY**: chặn `train()`;
- **MOCK**: thay `os.replace`.

| Mã | Loại | Tiêu chí đạt |
|:--|:--|:--|
| U11 | hàm thật | `phan_tich_duoi_csv`, 14 ca CRLF/LF (sạch; hoàn chỉnh thiếu newline; CRLF ghi dở một nửa; NUL; số dở dang `1`; thiếu cột; `0,1,`; hex bị cắt; ô tùy chọn trống hợp lệ): đúng hành động, số byte bỏ, ký tự xuống dòng; hỏng ở giữa / header sai / file rỗng → `LoiTiepTuc` |
| U12 | hàm thật + **writer thật** | `sao_luu_truoc_tiep_tuc` ở 4 ca (CRLF và LF): bản sao nguyên byte; biên nhận = bản ghi trả về; sau đó ghi tiếp bằng `BaseTrainer.save_metrics` của Ultralytics và `het_epoch_train`: đúng cột, `results` epoch 1..10, `theo_batch` giữ lịch sử (cả ô trống) cộng hàng mới |
| U12b | **MOCK** `os.replace` lỗi khi thay `results.csv` | `LoiTiepTuc`; file cũ nguyên byte; bản sao đã có |
| S21 / S22 | **SPY** | `results.csv` hỏng ở giữa / header `theo_batch.csv` sai → exit 2, không gọi `train()`, thư mục lượt không đổi |
| C0 | CHẠY THẬT | Lượt gốc `yolov8n_CK` bị kill sau epoch 1; hai CSV kết thúc CRLF |
| C-x-0a / 0b | CHẠY THẬT | Dry-run exit 0, không đổi file; báo đúng hành động cho từng file |
| C-x-A | CHẠY THẬT | Resume thật, kill sau epoch 2, ghi tiếp bằng writer thật: hai CSV đọc được, đủ cột, sạch; `results` epoch [1, 2]; `theo_batch` giữ nguyên các hàng lịch sử + nb hàng lần 1 epoch 2 |
| C-x-B | CHẠY THẬT | Bản sao hai CSV trước sửa trùng từng byte; biên nhận = bản ghi trong `bat_dau`, đúng hành động |
| C-x-C | CHẠY THẬT | `last_e001.pt`, `nguon_tiep_tuc_1.pt` = e001 gốc; args / last / best cũ được lưu |
| C-x-D | CHẠY THẬT | Preflight tiếp exit 0 "tiếp tục từ epoch 3/3"; resume lần 2 tới hết; `results` epoch [1, 2, 3]; `theo_batch` = lịch sử + lần 1 + lần 2, đủ cột |
| C-x-B2 | CHẠY THẬT | Bản sao trước lần 2 trùng byte; biên nhận `khong` ở cả hai file |
| C-results_hong_giua / C-batch_header_sai | CHẠY THẬT | Dry-run và resume thật đều exit 2 với đúng lý do; thư mục lượt không đổi |

Bốn ca `x`, mỗi ca ghi dạng (`results.csv` → hành động; `theo_batch.csv` → hành động):

| Ca | `results.csv` | `theo_batch.csv` |
|:--|:--|:--|
| `sach` | không đổi → `khong` | không đổi → `khong` |
| `nul_va_hang_do` | NUL → `cat_duoi` | `0,1,` → `cat_duoi` |
| `so_dang_do_va_thieu_newline` | `1` → `cat_duoi` | hàng cuối thiếu CRLF → `them_newline` |
| `thieu_newline_va_hex_cat` | hàng epoch 1 thiếu CRLF → `them_newline` | hàng có `thu_tu_batch` bị cắt → `cat_duoi` |

**Kết quả dự kiến**
- **Mã mới**: `src/test_tiep_tuc_huan_luyen.py` đạt **138 phép** = 102 phép cũ (gồm nhóm N của R39-F1) + U11, U12, U12b,
  S21, S22 + nhóm C 31 phép (C0 + 4 × 7 + 2 ca từ chối).
- **Mã cũ** (`git -c core.autocrlf=false archive 421dd53 src`, chạy `--chi-nhom-c --ma-nguon <src cũ>`, tổng 31 phép):
  - **đạt 11**: C0, 0a × 4, A và D của ca sạch, C × 4;
  - **không đạt 20**:
    - **A × 3 và D × 3 ở ba ca hỏng**, đúng R40-N1 / R40-N2: `ValueError` làm resume dừng; đuôi `1` dính thành epoch 12;
      hàng thiếu newline dính với hàng mới;
    - 0b × 4, B × 4, B2 × 4, vì mã cũ không báo kế hoạch CSV, không ghi biên nhận;
    - 2 ca từ chối, vì mã cũ không từ chối.
- **Script Codex sau sửa**: dự kiến `known_risks_reproduced = 1`. Hai ca `results.csv` đi qua `sao_luu_truoc_tiep_tuc`
  nên đã được sửa. Ca `theo_batch.csv` gọi thẳng writer, bỏ qua bước chuẩn bị, nên vẫn quan sát thấy rủi ro. Tiêu chí là
  test tích hợp; có kèm đối chứng đi qua bước chuẩn bị.
- **Suite bị ảnh hưởng trực tiếp**: `test_detector_5lop` 92/92, `test_pilot_chan_doan` 34/34.

Không dùng tolerance. Lỗi nào phát sinh thì giữ log, tìm nguyên nhân có bằng chứng, sửa, rồi chạy lại với tên log mới.
Không sửa hay commit trong lúc test đang chạy.
