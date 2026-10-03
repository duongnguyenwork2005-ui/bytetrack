# Vòng 41 — ghi nhận nghiệm thu `421dd53`; sửa R40-N1 (`results.csv`) và R40-N2 (`theo_batch.csv`) (chi tiết)

**Phạm vi** (người dùng duyệt): sửa hai vấn đề CSV R40-N1 và R40-N2 của review `421dd53`, kiểm hồi quy ngắn trên fixture CPU,
bàn giao, rồi **dừng**.
- Không train dài, không GPU, không dataset đầy đủ.
- Không đổi split, batch, workers, epoch hay ngưỡng; không sửa pilot mosaic.
- Không thêm guard ở hàm append, không khóa chạy đồng thời; không SSH Nano, không đóng ứng dụng.
- Base review: **`421dd53`** (Codex nghiệm thu vòng 40). Vòng 41 **chưa** được nghiệm thu.

## 0. Trạng thái đầu vòng

Log: `logs/review41/00_trang_thai_dau.log`.
- HEAD `421dd53`, đúng target đã review; 0 commit sau đó. Working tree chỉ có file riêng của người dùng.
- Môi trường: Python 3.11.9, ultralytics 8.4.141 (`trainer.py` `7532ff0b…`), torch 2.6.0+cu124. RAM còn trống 1,17 GB
  lúc kiểm.
- sha256 trước khi sửa: `tiep_tuc_huan_luyen.py` `826f233a…`, `train_detector.py` `48e9cabf…`.

## 1. Ghi nhận nghiệm thu `421dd53` (ĐÃ CHẠY)

Commit `508cfad`.
- **Báo cáo** `D:\LABOTARY\review_codex\CODEX_REVIEW_421dd53.md`: 12.063 B, sha256 `96b1ad70…`.
- **Kết luận**: **PASS WITH LIMITATIONS**, **đóng R39-F1** trong phạm vi sửa nhật ký. Ghi nhận này **không** có nghĩa toàn
  bộ cơ chế resume đã an toàn.
- **Giới hạn ghi đúng như review**:
  - không xác nhận toàn bộ cơ chế phục hồi chịu được mọi dạng mất điện;
  - không phải phê duyệt training dài;
  - reviewer không lặp lại test CPU/GPU của tác giả và không kiểm toàn cây Git;
  - phép thử `os.replace` lỗi là giả lập, không phải mất điện vật lý.
- **Hai rủi ro CSV** do tác giả công khai, được reviewer tái hiện: R40-N1, R40-N2 (mục 2).
- **Bằng chứng**: ZIP 238.263 B, sha256 `f57b53c3…`. Giải nén vào thư mục mới, ZIP gốc giữ nguyên; `SHA256SUMS` khớp
  **52/52**, cả trên blob đã stage. Lưu ở `handoff/review_evidence/CODEX_421dd53/` (`-text`).
- **Gói đã nghiệm thu**: 2.177.781 B, `806b1e01…`, khớp. Lưu ở `handoff/archive/review_delta_421dd53.zip`.
- **Base review** `b83dc12` → **`421dd53`** (vòng 40); giữ đủ 13 khai báo đầu vào không bàn giao.

## 2. Tái hiện trước khi sửa và đường gọi thật

`observe_csv_risks.py` của Codex, chạy nguyên trên mã `421dd53` (`logs/review41/01_quan_sat_csv_truoc_sua.{log,json}`):
**`known_risks_reproduced = 3`**, exit 0. Exit 0 chỉ có nghĩa phép quan sát chạy xong.

| Tình huống | Quan sát |
|:--|:--|
| `results.csv` đuôi 4 byte NUL | `ValueError` thô, không phải `LoiTiepTuc` (bản sao vẫn nguyên byte) |
| `results.csv` sau epoch 1–9, đuôi `1` | được giữ thành hàng thiếu trường (`time` = None) |
| `theo_batch.csv` đuôi `0,1,` + writer thật `het_epoch_train` | hàng mới dính vào thành `0,1,1`, kèm 2 trường thừa |

**Đường gọi thật** (đọc mã):
- `results.csv` do Ultralytics `BaseTrainer.save_metrics` ghi nối: chế độ text, mỗi ô `%.6g`, header chỉ khi file chưa có,
  trên Windows là CRLF. Không fsync.
- `theo_batch.csv` do `GhiLuotDayDu.het_epoch_train` ghi nối: `csv.DictWriter`, CRLF, header chỉ khi file chưa có, có
  fsync.
- `sao_luu_truoc_tiep_tuc` sao lưu cả hai. Sau đó nó chỉ cắt `results.csv` theo epoch bằng `int(float(trường đầu))`
  (gây `ValueError` với NUL, giữ `1` như epoch 1) và không kiểm số cột. `theo_batch.csv` không được xử lý.
- Lần ghi nối đầu tiên sau resume nằm ở epoch tiếp tục đầu tiên. Bước sao lưu chạy trước mọi lần ghi nối và không chạy ở
  dry-run, nên đó là chỗ sửa.

## 3. Sửa (commit `8d5d3c8`) — chỉ ở bước chuẩn bị ghi của resume thật

Cùng nguyên tắc đã dùng cho nhật ký ở vòng 40. sha256 sau sửa: `tiep_tuc_huan_luyen.py` `44045186…`, `train_detector.py`
`250302fb…`.

**`phan_tich_duoi_csv(b, kieu, ten, epoch_da_luu)`** (mới, **chỉ đọc**):

| Kiểm | `results.csv` | `theo_batch.csv` |
|:--|:--|:--|
| Header | dòng đầu, bắt đầu `epoch,time`, không ô trống | dòng đầu, trùng `COT_BATCH` (20 cột) |
| Hàng hợp lệ | đủ cột; **mọi ô là số** (`%.6g`); epoch nguyên ≥ 1 | đủ 20 cột; bắt buộc `lan, epoch, i, ni, accumulate, ema_updates, batch_size` (int), `t` (float), `loss_huu_han` (True/False); loss / lr / momentum là số **hoặc ô trống hợp lệ**; `thu_tu_batch` trống hoặc đúng 16 ký tự hex |

| Phần đuôi | Hành động |
|:--|:--|
| Hàng cuối hợp lệ, có newline | `khong` |
| Hàng cuối **hợp lệ** nhưng thiếu newline | `them_newline`: giữ hàng, thêm đúng ký tự xuống dòng của file (CRLF/LF; nếu đã có `\r` thì chỉ thêm `\n`) |
| Hàng cuối không hợp lệ (thiếu trường, byte NUL, số dở dang, hex bị cắt…) | `cat_duoi`: cắt từ đầu hàng đó |
| Hàng hỏng ở **giữa**, header sai, file rỗng | `LoiTiepTuc`: preflight exit 2 trước khi train, không tự sửa |

**Preflight** thêm `csv_duoi` (kế hoạch cho cả hai file). `--dry-run --tiep-tuc` in kế hoạch và **không ghi gì**.

**`sua_duoi_csv`** (mới), gọi trong `sao_luu_truoc_tiep_tuc`, sau khi đã sao lưu nguyên byte:
1. File phải trùng sha256 lúc preflight và trùng bản sao; nếu mất hoặc đổi giữa chừng → `LoiTiepTuc`.
2. Cắt hàng cuối dở, hoặc thêm newline.
3. Riêng từng file:
   - `results.csv`: **giữ quy tắc cũ**, chỉ giữ hàng epoch ≤ epoch đã lưu, nhưng nay áp trên các hàng **đã qua kiểm định
     dạng**, nên không còn hàng dở lọt qua chỉ vì trường đầu chuyển được thành số;
   - `theo_batch.csv`: **giữ mọi hàng lịch sử hợp lệ**, kể cả cột `lan` và các hàng của epoch bị ngắt; không cắt theo epoch.
4. Ghi qua file tạm + fsync + `os.replace`; đọc lại và kiểm: sạch, đủ cột, các hàng hợp lệ đúng như mong đợi.
5. Biên nhận `khoi_phuc/truoc_tiep_tuc_N/xu_ly_<file>.json`: sha256 trước/sau, hành động, lý do, số byte bỏ, sha256 phần
   bỏ, số hàng bỏ theo epoch, vị trí bản sao. Cùng nội dung được ghi vào sự kiện `bat_dau` (`tiep_tuc.sao_luu.csv`).

Khối cắt `results.csv` cũ bị thay. Tóm tắt `results_csv` (`dong_giu` / `dong_bo`) giữ lại để đối chiếu với vòng 39.

**Giới hạn nhận biết** (không tuyên bố hơn):
- Một hàng bị cắt đúng sau một ô mà phần còn lại vẫn hợp lệ cú pháp thì không phân biệt được với hàng hoàn chỉnh thiếu
  newline. Ví dụ:
  - số ở cột cuối `results.csv` bị cắt bớt chữ số;
  - hàng `theo_batch.csv` có `thu_tu_batch` trống, bị cắt ngay sau dấu phẩy cuối.
- Với `results.csv`, hàng của epoch bị ngắt (epoch > epoch đã lưu) vẫn bị bỏ theo quy tắc epoch, kể cả khi trông hoàn chỉnh.
- Ultralytics không fsync `results.csv`. Nếu mất điện làm **mất hẳn** hàng (không phải ghi dở) thì bước này không khôi phục
  được hàng đó.

## 4. Kiểm hồi quy (ĐÃ CHẠY)

**Kế hoạch đăng ký trước**: `logs/review41/02_ke_hoach_kiem_r40n.md`, gồm tiêu chí và kết quả dự kiến trên mã mới lẫn mã cũ,
có phụ lục sửa đổi.

Ba loại phép kiểm:
- **CHẠY THẬT**: `train_detector.py` qua subprocess, ultralytics thật, CPU, fixture;
- **SPY**: chặn `train()`;
- **MOCK**: thay `os.replace`.

### 4.1 Mã mới — `src/test_tiep_tuc_huan_luyen.py`

| Lần | Kết quả | Ghi chú |
|:--|:--|:--|
| 1 (`03_…_lan1.log`) | **không hoàn tất** | 107 phép đạt (cả U11 / U12 / U12b, S21 / S22, nhóm N), 1 không đạt: C0 hết thời gian đợi. Sau đó tác vụ nền bị dừng ở mốc 1 giờ. Nguyên nhân có bằng chứng: **máy tự ngủ vì pin cạn**, từ 16:54:10 tới 17:53:42 (log System Kernel-Power 524 / 42, Power-Troubleshooter: `03_su_kien_nguon_lan1.log`). C0 đã xong epoch 1 rồi đứng yên trong lúc máy ngủ (`03_moc_thoi_gian_lan1.log`). Không sửa mã, không đổi tiêu chí |
| 2 (`03_…_lan2.log`) | **138/138, exit 0** | Chạy khi đang cắm sạc; đúng số phép đăng ký trước. Giữa hai lần chỉ thêm log và phụ lục kế hoạch (commit `04cf6ea`), không đổi mã |

**Nhóm C** (CHẠY THẬT, yolov8n, lần 2): mọi phép đạt. Lượt gốc C0 bị kill sau epoch 1; `results.csv` có 1 hàng,
`theo_batch.csv` có 10 hàng, đều CRLF (file thật của Ultralytics và runner).

| Ca | Dry-run (0a / 0b) | A: resume thật, kill sau epoch 2, ghi tiếp thật | B: bản sao + biên nhận | C | D: preflight + resume tới hết | B2 |
|:--|:--|:--|:--|:--|:--|:--|
| `sach` | không đổi file / `khong`, `khong` | đúng cột, sạch; `results` [1, 2]; `theo_batch` 10 lịch sử + 10 mới | trùng byte | đạt | [1, 2, 3]; 10 + 10 + 10 | đạt |
| NUL + `0,1,` | không đổi / `cat_duoi`, `cat_duoi` | như trên | trùng byte, `cat_duoi` / `cat_duoi` | đạt | như trên | đạt |
| `1` + hàng batch thiếu CRLF | không đổi / `cat_duoi`, `them_newline` | như trên, giữ nguyên hàng thiếu newline | `cat_duoi` / `them_newline` | đạt | như trên | đạt |
| hàng results thiếu CRLF + hex bị cắt | không đổi / `them_newline`, `cat_duoi` | như trên, giữ nguyên hàng epoch 1 | `them_newline` / `cat_duoi` | đạt | như trên | đạt |
| `results.csv` hỏng ở giữa | dry-run exit 2 | resume thật exit 2, thư mục không đổi | — | — | — | — |
| header `theo_batch.csv` sai | dry-run exit 2 | resume thật exit 2, thư mục không đổi | — | — | — | — |

Phép kiểm khác:
- **U11**: 14 ca phân tích đúng (CRLF và LF, ô tùy chọn trống hợp lệ). Hỏng ở giữa, header sai, file rỗng → `LoiTiepTuc`.
- **U12**: 4 ca qua `sao_luu_truoc_tiep_tuc`, gồm CRLF và LF; ghi tiếp bằng `BaseTrainer.save_metrics` thật và
  `het_epoch_train` thật. Đủ cột, `results` epoch 1..10, `theo_batch` giữ lịch sử kể cả ô trống.
- **U12b** (MOCK `os.replace` lỗi): `LoiTiepTuc`, `results.csv` cũ nguyên byte, bản sao đã có.
- **S21 / S22** (SPY): exit 2, không gọi `train()`, thư mục lượt không đổi.
- 102 phép cũ vẫn đạt, gồm toàn bộ nhóm N (regression R39-F1), K1–K9, S1–S20.

### 4.2 Test mới bắt được lỗi trên mã cũ

Mã cũ: `git -c core.autocrlf=false archive 421dd53 src`, trùng blob (`tiep_tuc_huan_luyen.py` `826f233a…`,
`train_detector.py` `48e9cabf…`). Chạy nhóm C của test **mới** trên mã đó (`--chi-nhom-c --ma-nguon`, CHẠY THẬT):
**11/31, exit 1**, khớp đúng dự đoán đăng ký trước (`logs/review41/04_nhom_c_tren_ma_cu_421dd53.log`).

- **Đạt 11**: C0, 0a × 4, A và D của ca sạch, C × 4.
- **Không đạt 20**: A × 3 và D × 3 ở ba ca hỏng (bảng dưới); 0b × 4, B × 4, B2 × 4 (mã cũ không báo kế hoạch CSV, không
  ghi biên nhận); 2 ca từ chối.

| Ca | A trên mã cũ | D trên mã cũ |
|:--|:--|:--|
| NUL + `0,1,` | resume **exit 1** (`ValueError` thô) trước khi train; `results` epoch `['1', '\x00\x00\x00\x00']`; `theo_batch` 11 hàng, sai cột | preflight exit 0, nhưng resume lại exit 1: **kẹt**, không resume được |
| `1` + hàng batch thiếu CRLF | train tới epoch 2, nhưng `results` epoch `['1', '12']`: đuôi `1` dính với hàng epoch 2; `theo_batch` 19 hàng, sai cột | `results` `['1', '3']`: quy tắc epoch cũ bỏ hàng `12`, **mất hàng epoch 2** |
| hàng results thiếu CRLF + hex bị cắt | `results` chỉ còn `['1']`: hàng epoch 2 dính vào hàng epoch 1; `theo_batch` sai cột | `results` `['1', '3']`, hàng epoch 2 vẫn mất |
| `results.csv` hỏng ở giữa | dry-run exit 0, resume exit 1 (`ValueError`): không từ chối có kiểm soát | — |
| header `theo_batch.csv` sai | dry-run exit 0, resume exit 0: không từ chối | — |

Ca sạch đạt trên cả hai mã.

### 4.3 Suite bị ảnh hưởng trực tiếp và script của Codex

| Phép | Kết quả |
|:--|:--|
| `test_detector_5lop` | **92/92**, exit 0 (`05_test_detector_5lop.log`) |
| `test_pilot_chan_doan` | **34/34**, exit 0 (`05_test_pilot_chan_doan.log`) |
| `observe_csv_risks.py` của Codex, chạy nguyên (sha256 `e45e5c9d…`), trước sửa | `known_risks_reproduced = 3`, exit 0 (`01_*`) |
| như trên, **sau sửa** | **1**, exit 0 (`06_*`), đúng dự đoán |
| Đối chứng `06b_quan_sat_qua_buoc_chuan_bi.py` | **0**, exit 0 (`06b_*`) |

- Sau sửa, hai ca `results.csv` của script (đi qua `sao_luu_truoc_tiep_tuc`) không còn lỗi. File kết thúc ở hàng
  `9,90,0.5\r\n`, bản sao nguyên byte.
- Ca `theo_batch.csv` của script gọi thẳng `het_epoch_train` trên file có đuôi dở, không qua bước chuẩn bị. Vì vậy vẫn thấy
  hàng dính `0,1,1` cùng 2 trường thừa. Điều này đúng dự đoán: bản sửa nằm ở bước chuẩn bị, writer không đổi (guard ở writer
  là đề xuất ngoài phạm vi, mục 7).
- Đối chứng 06b dùng cùng hai ca `results.csv`. Ca `theo_batch.csv` đi qua bước chuẩn bị thật, rồi mới gọi writer thật:
  - (a) **dữ liệu mẫu của script**: hàng lịch sử chỉ có `lan/epoch/i`, các ô bắt buộc khác trống. Runner không bao giờ ghi
    hàng như vậy, nên theo định dạng thật nó không hợp lệ. Hàng 2 thành hàng hỏng ở giữa → `LoiTiepTuc` "dòng 2 hỏng ở GIỮA
    (ô bắt buộc 'ni' = '')", file không đổi, writer không chạy;
  - (b) **hàng đầy đủ như runner ghi**: cắt đuôi `0,1,`, writer ghi hàng mới đúng `[1, 2, 0]`, không có trường thừa.
- 06b không thay thế test tích hợp (nhóm C).

## 5. Lệnh (từ gốc repo) và mã thoát đã ghi

```text
# quan sat cua Codex (chay nguyen): truoc sua (HEAD 421dd53) 3, exit 0; sau sua 1, exit 0 (muc 4.3)
python D:/LABOTARY/review_codex/CODEX_REVIEW_421dd53_EVIDENCE/observe_csv_risks.py --repo D:/LABOTARY/bytetrack --out <json>
# doi chung qua buoc chuan bi ghi cua resume: 0, exit 0
python handoff/logs/review41/06b_quan_sat_qua_buoc_chuan_bi.py --out <json>
# suite resume day du (CPU, fixture, ~8 phut): lan 2 138/138, exit 0; lan 1 khong hoan tat (may ngu vi pin can)
python -u src/test_tiep_tuc_huan_luyen.py
# test moi tren ma cu: trich nguyen blob 421dd53 roi chi chay nhom C: 11/31, exit 1 (dung du doan)
git -c core.autocrlf=false archive 421dd53 src | tar -x -C <thu muc>
python -u src/test_tiep_tuc_huan_luyen.py --chi-nhom-c --ma-nguon <thu muc>/src
# suite bi anh huong truc tiep: 92/92 va 34/34, exit 0
python -u src/test_detector_5lop.py
python -u src/test_pilot_chan_doan.py
```

Cú pháp resume cho người dùng **không đổi**: `--che-do day-du --xac-nhan-train-dai --tiep-tuc`. `--dry-run --tiep-tuc` nay
in thêm kế hoạch CSV cho từng file (`khong` / `cat_duoi` / `them_newline`, kèm số hàng `results.csv` sẽ bỏ theo epoch), vẫn
không ghi gì.

## 6. Giới hạn

- **Giới hạn nhận biết** (mục 3): hàng bị cắt đúng sau một ô mà vẫn hợp lệ cú pháp được coi là hàng hoàn chỉnh thiếu
  newline, nên được giữ. Không tuyên bố phát hiện được mọi dạng ghi dở.
- Chỉ hàng nội dung **cuối** được xử lý. Hàng hỏng ở giữa, header sai, file rỗng → dừng (exit 2) trước khi train, cần người
  xử lý tay; file không bị đổi.
- `theo_batch.csv`: hàng lịch sử thiếu ô bắt buộc (runner không ghi dạng này) bị coi là hỏng và bị từ chối, không tự sửa
  (đối chứng 06b-a).
- `results.csv`:
  - Ultralytics không fsync file này; mất hẳn hàng (không phải ghi dở) thì không khôi phục được;
  - hàng hoàn chỉnh của epoch bị ngắt vẫn bị bỏ theo quy tắc epoch có từ vòng 39;
  - checkpoint của Ultralytics có kèm `train_results` (đọc từ `results.csv` lúc lưu); vòng này không dùng nó để đối chiếu
    hay khôi phục.
- Writer (`save_metrics`, `het_epoch_train`) không đổi; gọi thẳng writer trên file có đuôi dở vẫn dính hàng (script Codex
  sau sửa vẫn 1). Mọi đường ghi hiện có chạy sau bước chuẩn bị:
  - lượt mới tạo thư mục mới; runner từ chối thư mục đã có;
  - lượt tiếp tục được xử lý ở bước sao lưu.

  Một đường ghi khác thêm sau này sẽ chưa được bảo vệ ở mức hàm (mục 7).
- Không khóa chạy đồng thời: hai lần chạy cùng lúc trên một thư mục lượt chưa bị chặn (mục 7).
- Ngắt được mô phỏng bằng kill tiến trình và bằng đuôi hỏng ghi sẵn; lỗi `os.replace` là MOCK. Chưa mô phỏng mất điện thật.
- Nhóm C chỉ chạy yolov8n trên fixture CPU (3 epoch). Logic CSV không phụ thuộc model; không chạy GPU, không dataset đầy đủ.
- Các giới hạn của vòng 39–40 giữ nguyên:
  - resume không trùng từng bit;
  - Ctrl+C thật chưa kiểm;
  - RAM trống thấp nhất 17 MiB ở GPU smoke;
  - trường mosaic của pilot vòng 37 không đáng tin.

## 7. Đề xuất ngoài phạm vi — CHƯA làm, chờ duyệt

1. **Guard ở hàm append** (`het_epoch_train`, `ghi_nhat_ky`): kiểm đuôi file trước khi ghi nối, từ chối nếu không sạch. Bảo
   vệ cả những đường ghi thêm sau này, không chỉ đường resume.
2. **Khóa thư mục lượt**: chặn hai lần chạy cùng lúc trên một thư mục.
3. **Training dài**: số epoch, ngân sách, thời điểm **vẫn chờ duyệt**. Khi chạy dài cần cắm sạc. Lần 1 của vòng này cho thấy
   máy tự ngủ khi pin cạn, làm tiến trình đứng yên khoảng 1 giờ.

## 8. Commit vòng 41

| Commit | Nội dung |
|:--|:--|
| `508cfad` | Ghi nhận nghiệm thu `421dd53` (PASS WITH LIMITATIONS, đóng R39-F1), bằng chứng 52/52, archive gói, base → `421dd53`; tái hiện trước sửa (3) |
| `8d5d3c8` | Sửa R40-N1/N2 + test (U11, U12, U12b, S21, S22, nhóm C) + kế hoạch đăng ký trước |
| `04cf6ea` | Lần 1 không hoàn tất (máy ngủ vì pin cạn): log, sự kiện nguồn, mốc thời gian; phụ lục kế hoạch |
| `da1aac2` | Log lần 2 (138/138) |
| `6e29adb` | Test mới trên mã cũ (11/31, đúng dự đoán) |
| `9db5532` | Suite bị ảnh hưởng (92/92, 34/34); quan sát Codex sau sửa (1); đối chứng 06b (0) |
| commit chứa phụ lục này (target delta) | Brief, phụ lục, handoff |
