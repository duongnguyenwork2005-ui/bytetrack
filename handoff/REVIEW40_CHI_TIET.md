# Vòng 40 — sửa R39-F1 (nhật ký resume sau dòng cuối ghi dở) của review `1454b3d` (chi tiết)

**Phạm vi** (người dùng duyệt): chỉ sửa **R39-F1**, kiểm hồi quy ngắn trên fixture CPU, bàn giao, rồi **dừng**.
- Không train dài, không chạy dataset đầy đủ, không lặp training GPU.
- Không xuất lại dataset; không đổi split, batch, workers, số epoch hay ngưỡng đánh giá.
- Không sửa trường mosaic của pilot vòng 37, không SSH Nano, không đóng ứng dụng.
- Base review giữ **`b83dc12`**; vòng 39 chưa được nghiệm thu.

> **Theo dõi vòng 41**: Codex `421dd53` PASS WITH LIMITATIONS, đóng R39-F1 trong phạm vi sửa nhật ký. Hai rủi ro CSV
> ở mục 7 (R40-N1 `results.csv`, R40-N2 `theo_batch.csv`) được sửa ở vòng 41; xem `REVIEW41_CHI_TIET.md`. Các số liệu
> khác của phụ lục này không đổi.

## 0. Trạng thái đầu vòng

Log: `logs/review40/00_trang_thai_dau.log`.
- HEAD `1454b3d`, đúng target đã review; 0 commit sau đó. Working tree chỉ có file riêng của người dùng.
- Môi trường: Python 3.11.9, ultralytics 8.4.141 (`trainer.py` `7532ff0b…`), torch 2.6.0+cu124. RAM còn trống 0,39 GB lúc
  kiểm.
- sha256 trước khi sửa: `tiep_tuc_huan_luyen.py` `2ecf26cd…`, `train_detector.py` `ba8f7f4e…`.

## 1. Ghi nhận review `1454b3d` (ĐÃ CHẠY)

Commit `0e58cf8`.
- **Báo cáo** `D:\LABOTARY\review_codex\CODEX_REVIEW_1454b3d.md`: 12.122 B, sha256 `44200551…`.
  - **REQUEST CHANGES**, chỉ vì **R39-F1** (P1).
  - Không rút lại kết quả CPU/GPU vòng 39. Chưa duyệt training dài.
- **Bằng chứng**: ZIP 403.784 B, sha256 `bb836d0b…`. Giải nén vào thư mục mới, ZIP gốc giữ nguyên; `SHA256SUMS` khớp
  **97/97**, cả trên blob đã stage. Lưu ở `handoff/review_evidence/CODEX_1454b3d/` (`-text`).
- Gói đã review (1.607.478 B, `b7674449…`) lưu ở `handoff/archive/review_delta_1454b3d.zip`, **chưa nghiệm thu**.
  `review_base.json` **không đổi** (base `b83dc12`).

## 2. Tái hiện trước khi sửa và đường gọi thật

**Script của Codex**, chạy nguyên trên mã `1454b3d` (`logs/review40/01_repro_truoc_sua.{log,json}`): **exit 1,
`bug_count=3`**. Ca `clean_control` đạt. Cả ba biến thể đều cho "dòng 3 hỏng ở GIỮA":
- JSON cuối ghi dở, không newline;
- JSON cuối ghi dở, có newline;
- JSON cuối hợp lệ nhưng thiếu newline.

**Đường gọi thật** (đọc mã):
1. **Preflight**: `chuan_bi_tiep_tuc` gọi `doc_nhat_ky`. Hàm này bỏ qua dòng cuối hỏng, chỉ cảnh báo, không sửa file.
2. **Sao lưu**: `sao_luu_truoc_tiep_tuc` lưu checkpoint và CSV, nhưng **không** sao lưu, không xử lý nhật ký.
3. **Lần append đầu tiên**: `GhiLuotDayDu.sau_chuan_bi` ghi `bat_dau`. Nếu train lỗi trước callback, lần append đầu là
   `ket_thuc` trong `main`.
4. **Ghi epoch**: `luu_model` ghi `epoch_da_luu`.

Ở bước 3, append nối ngay vào phần ghi dở. Đuôi không có newline thì hai JSON dính nhau; đuôi có newline thì dòng hỏng
thành dòng giữa. Đến lần đọc sau, `LoiTiepTuc` chặn resume.

## 3. Sửa (commit `6fc979f`, bổ sung `074c486`)

Chỉ sửa ở **bước chuẩn bị ghi của resume thật**, tức `sao_luu_truoc_tiep_tuc`, vì bước này chạy trước mọi lần append và
không chạy ở dry-run. **Không đổi** `ghi_nhat_ky` / `doc_nhat_ky`, cơ chế resume của Ultralytics hay các kiểm checkpoint.

**`phan_tich_duoi_nhat_ky(b)`** (mới, **chỉ đọc**), cùng quy tắc với `doc_nhat_ky`:

| Phần đuôi | Hành động |
|:--|:--|
| Dòng nội dung cuối hợp lệ, có newline | `khong` |
| JSON cuối **hợp lệ** nhưng thiếu newline | `them_newline`: **giữ sự kiện**, chỉ thêm `\n` |
| Dòng nội dung cuối không parse được (có hoặc không newline; kể cả byte NUL) | `cat_duoi`: cắt từ đầu dòng đó tới hết file |
| Dòng hỏng ở **giữa** | `LoiTiepTuc`: không tự sửa, từ chối trước khi train |

**Preflight** thêm `nhat_ky_duoi` (kế hoạch trên). `--dry-run --tiep-tuc` in kế hoạch và **không ghi gì**.

**`sua_duoi_nhat_ky`** (mới), gọi trong `sao_luu_truoc_tiep_tuc` sau khi sao lưu checkpoint và **trước lần append đầu
tiên**:
1. Nhật ký phải còn và sha256 phải **trùng lúc preflight**; nếu không thì `LoiTiepTuc`.
2. **Luôn** sao lưu nguyên byte vào `khoi_phuc/truoc_tiep_tuc_N/nhat_ky_luot.jsonl`, ghi qua file tạm và kiểm sha256.
3. Sửa bằng file tạm + fsync + `os.replace`. Nếu bị ngắt giữa chừng, nhật ký cũ còn nguyên (U10b); lần tiếp tục sau sẽ xử
   lý lại với số lần mới.
4. Đọc lại và kiểm: không cảnh báo, kết thúc bằng newline, sự kiện hợp lệ **trùng từng sự kiện** với trước khi sửa. Phần
   giữ lại trùng từng byte với bản gốc.
5. Biên nhận `khoi_phuc/truoc_tiep_tuc_N/xu_ly_nhat_ky.json` ghi: sha256 và số byte trước/sau, hành động, lý do, số byte
   bỏ, sha256 phần bỏ, 200 byte đầu của phần bỏ, vị trí bản sao. Cùng nội dung được ghi vào sự kiện `bat_dau` của lần
   tiếp tục (`tiep_tuc.sao_luu.nhat_ky`) và vào JSON kết quả runner.

Không thêm sự kiện nào, không tạo lịch sử giả. Phần bị cắt chỉ nằm trong bản sao.

## 4. Kiểm hồi quy (ĐÃ CHẠY)

**Kế hoạch đăng ký trước**: `logs/review40/02_ke_hoach_kiem_r39f1.md`, gồm tiêu chí và kết quả dự kiến trên mã mới lẫn mã
cũ, có phụ lục sửa đổi.

Ba loại phép kiểm:
- **CHẠY THẬT**: `train_detector.py` qua subprocess, ultralytics thật, CPU, fixture;
- **SPY**: chặn `train()`;
- **MOCK**: thay `os.replace`.

### 4.1 Mã mới — `src/test_tiep_tuc_huan_luyen.py`

| Lần | Kết quả | Ghi chú |
|:--|:--|:--|
| 1 (`03_…_lan1.log`) | 98/99, exit 1 | Nhóm N 30/30 và S20 đạt. Nhóm U dừng ở U8: test cũ gọi bước sao lưu trên thư mục tạm không có nhật ký → `FileNotFoundError`, nên U9 / U10 / U10b chưa chạy. Sửa phần thiết lập U8 và đổi lỗi thành `LoiTiepTuc` rõ ràng; tiêu chí không đổi |
| 2 (`03_…_lan2.log`) | **102/102, exit 0** | Đúng số phép đăng ký trước |

Nhóm N (CHẠY THẬT, yolov8n, lần 2): mọi phép đạt.

| Ca | Dry-run (0a không đổi file / 0b kế hoạch) | A đọc lại sau resume | B bản sao + biên nhận | C checkpoint cũ | D preflight + resume tiếp | B2 |
|:--|:--|:--|:--|:--|:--|:--|
| sạch | đạt / `khong` | 2 + 2 sự kiện | trùng byte, `khong` | đạt | 3/3 → hoàn tất | đạt |
| ghi dở, không newline | đạt / `cat_duoi` | 2 + 2 | trùng byte, bỏ 56 B | đạt | đạt | đạt |
| ghi dở, có newline | đạt / `cat_duoi` | 2 + 2 | trùng byte, bỏ 57 B | đạt | đạt | đạt |
| JSON hợp lệ thiếu newline | đạt / `them_newline` | **3** + 2 (giữ sự kiện) | trùng byte, bỏ 0 B | đạt | đạt | đạt |
| dòng hỏng ở giữa | dry-run exit 2 | resume thật exit 2, thư mục không đổi (không sao lưu, không train) | — | — | — | — |

Phép kiểm khác:
- **U9**: hàm phân tích đúng ở 5 ca, kể cả byte NUL.
- **U10**: sửa đúng byte, bản sao nguyên byte, biên nhận đúng.
- **U10b** (MOCK `os.replace` lỗi): `LoiTiepTuc`, nhật ký cũ nguyên byte.
- **S20** (SPY): dòng hỏng ở giữa → exit 2, không gọi `train()`.
- 68 phép cũ của vòng 39 vẫn đạt (K1–K9, S1–S19…).

### 4.2 Test mới bắt được lỗi trên mã cũ

`logs/review40/04_nhom_n_tren_ma_cu_1454b3d.log`. Mã cũ được trích bằng
`git -c core.autocrlf=false archive 1454b3d src`, trùng blob (`tiep_tuc_huan_luyen.py` `2ecf26cd…`). Lệnh:
`--chi-nhom-n --ma-nguon <src cũ>`.

Kết quả: **12/30, exit 1, khớp đúng dự đoán đã đăng ký** (đạt 12, không đạt 18).

| Ca (mã cũ) | 0a | 0b | A | B | C | D | B2 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| sạch | đạt | không | đạt | không | đạt | đạt | không |
| ghi dở, không newline | đạt | không | **không**: "dòng 3 hỏng ở GIỮA" | không | đạt | **không**: preflight / resume lần 2 exit 2 | không |
| ghi dở, có newline | đạt | không | **không** (như trên) | không | đạt | **không** | không |
| JSON hợp lệ thiếu newline | đạt | không | **không** (như trên) | không | đạt | **không** | không |

N0 và N-giua đạt.
- **A và D không đạt ở đúng 3 biến thể, trong khi đạt ở ca sạch: đây chính là R39-F1** qua đường resume thật. Lần resume
  lần 1 của mã cũ vẫn chạy được, nhưng nhật ký sau đó không đọc lại được, và lần preflight tiếp theo bị chặn.
- 0b, B, B2 không đạt ở mọi ca vì mã cũ không báo kế hoạch và không sao lưu nhật ký (yêu cầu mới của vòng này).

### 4.3 Suite bị ảnh hưởng trực tiếp và script của Codex

| Phép | Kết quả | Log |
|:--|:--|:--|
| `src/test_detector_5lop.py` (runner đổi) | 92/92 PASS, exit 0 (vòng 39: 92/92) | `05_test_detector_5lop.log` |
| `src/test_pilot_chan_doan.py` | 34/34 PASS, exit 0 (vòng 39: 34/34) | `05_test_pilot_chan_doan.log` |
| Script Codex, chạy nguyên trên mã đã sửa | **exit 1, `bug_count=3`**, như dự đoán | `06_repro_sau_sua.*` |
| Đối chứng: cùng 4 ca của Codex, chèn bước chuẩn bị ghi thật | **`bug_count=0`, exit 0**; bản sao trùng gốc 4/4 | `06b_repro_qua_buoc_chuan_bi.*` |

Script của Codex gọi **thẳng** `ghi_nhat_ky` / `doc_nhat_ky`, hai hàm mà vòng này cố ý không đổi. Chính docstring của script
nói rằng khi sửa ở đường resume thì test tích hợp mới là tiêu chí. Script `06b` giữ nguyên dữ liệu, thao tác và tiêu chí
của Codex, chỉ chèn `sua_duoi_nhat_ky` trước lần append đầu. Đây là bằng chứng bổ trợ, không thay thế nhóm N.

## 5. Lệnh (từ gốc repo) và mã thoát đã ghi

```text
# tai hien cua Codex (cap ham): truoc sua exit 1 / bug_count 3; sau sua van exit 1 / bug_count 3 (muc 4.3)
python D:/LABOTARY/review_codex/CODEX_REVIEW_1454b3d_EVIDENCE/REPRO_REVIEW_1454b3d.py --repo D:/LABOTARY/bytetrack --out <json>
# doi chung qua buoc chuan bi ghi cua resume: exit 0, bug_count 0
python handoff/logs/review40/06b_repro_qua_buoc_chuan_bi.py --out <json>
# suite resume day du (CPU, fixture, ~9 phut): 102/102, exit 0
python -u src/test_tiep_tuc_huan_luyen.py
# test moi tren ma cu: trich nguyen blob 1454b3d roi chi chay nhom N: 12/30, exit 1 (dung du doan)
git -c core.autocrlf=false -c core.eol=lf archive 1454b3d src | tar -x -C <thu muc>
python -u src/test_tiep_tuc_huan_luyen.py --chi-nhom-n --ma-nguon <thu muc>/src
# suite bi anh huong truc tiep: 92/92 va 34/34, exit 0
python -u src/test_detector_5lop.py
python -u src/test_pilot_chan_doan.py
```

Cú pháp resume cho người dùng **không đổi**: `--che-do day-du --xac-nhan-train-dai --tiep-tuc`. `--dry-run --tiep-tuc` nay
in thêm kế hoạch nhật ký (`khong` / `cat_duoi` / `them_newline`), vẫn không ghi gì.

## 6. Giới hạn

- Chỉ dòng nội dung **cuối** được xử lý. Nhiều dòng hỏng, hay dòng hỏng ở giữa, vẫn bị từ chối và cần người xử lý tay.
- `ghi_nhat_ky` / `doc_nhat_ky` không đổi. Mọi đường append hiện có chạy sau bước chuẩn bị:
  - lượt mới chỉ ghi vào nhật ký mới; runner từ chối nếu nhật ký đã có;
  - lượt tiếp tục được xử lý ở bước sao lưu.

  Một đường ghi khác thêm sau này sẽ chưa được bảo vệ ở mức hàm (mục 7).
- Ngắt được mô phỏng bằng kill tiến trình và bằng đuôi hỏng ghi sẵn. Chưa mô phỏng mất điện thật.
- Nhóm N chỉ chạy yolov8n trên fixture CPU (imgsz 160, 3 epoch). Logic nhật ký không phụ thuộc model; không lặp training
  GPU.
- Đuôi ghi dở của `results.csv` / `theo_batch.csv` **chưa** được xử lý (mục 7).
- Các giới hạn của vòng 39 giữ nguyên:
  - resume không trùng từng bit;
  - Ctrl+C thật chưa kiểm;
  - RAM trống thấp nhất 17 MiB ở GPU smoke;
  - trường mosaic của pilot vòng 37 không đáng tin.

## 7. Đề xuất ngoài phạm vi — CHƯA làm, chờ duyệt

Cả ba điểm cùng lớp "đuôi ghi dở", ở cùng bước chuẩn bị ghi. Đã kiểm bằng thí nghiệm trên thư mục tạm
(`logs/review40/07_de_xuat_duoi_csv.{py,log}`), không sửa mã:
1. **`results.csv`** (phép cắt theo epoch có từ vòng 39):
   - đuôi byte NUL → `ValueError` thô thay cho `LoiTiepTuc`, nên resume dừng bằng traceback, sau khi đã tạo bản sao;
   - đuôi `1` dở dang được giữ lại thành dòng rác.

   Đề xuất: xử lý dòng cuối dở như nhật ký (cắt, có sao lưu) và đổi lỗi phân tích thành `LoiTiepTuc`.
2. **`theo_batch.csv`**: nếu bị ngắt giữa lúc ghi một dòng, lần ghi nối sau làm dòng bị dính (thừa trường). Đề xuất cắt
   dòng CSV dở ở cùng bước, có sao lưu. File này chỉ dùng chẩn đoán, không dùng để quyết định resume.
3. **Phòng thủ thêm** (không bắt buộc cho R39-F1):
   - `ghi_nhat_ky` từ chối append khi đuôi không sạch;
   - khóa thư mục lượt để chặn hai lần chạy cùng lúc trên một thư mục.

## 8. Commit vòng 40

| Commit | Nội dung |
|:--|:--|
| `0e58cf8` | Ghi nhận review `1454b3d` (REQUEST CHANGES), bằng chứng 97/97, archive gói; tái hiện trước sửa (exit 1, 3) |
| `6fc979f` | Sửa R39-F1 + test (U9, U10, U10b, S20, nhóm N) + kế hoạch đăng ký trước |
| `074c486` | Log lần 1 (98/99); sửa phần thiết lập U8 + `LoiTiepTuc` khi mất nhật ký |
| `42b44fc` | Log lần 2 (102/102) |
| `81fe10c` | Test mới trên mã cũ (12/30, đúng dự đoán) |
| `72ac1bf` | Suite bị ảnh hưởng (92/92, 34/34); repro sau sửa; đối chứng 06b; bằng chứng đề xuất |
| commit chứa phụ lục này (target delta) | Brief, phụ lục, handoff |
