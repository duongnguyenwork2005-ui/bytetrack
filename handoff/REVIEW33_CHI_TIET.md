# Vòng 33 — sửa R32-F1 và R32-F2 của review `c4d64a1` (chi tiết)

- Base review giữ **`bcb92f8`**; target trước sửa là `c4d64a1`.
- Phạm vi: sửa bộ kiểm, đối chiếu lại nhãn đang cách ly, đính chính báo cáo.
- **Không** nhập nhãn vào dữ liệu chính; không khóa split, xuất dataset, smoke hay training.
- Không SSH Nano, không sửa `src/` hay `configs/`, không đụng file riêng của người dùng.

## 0. Trạng thái đầu vòng và bằng chứng reviewer

- **Trạng thái** (`logs/review33/00_trang_thai_dau.log`):
  - nhánh `phase2-mixed-traffic`, HEAD `c4d64a1` (đúng dự kiến, 0 commit sau đó);
  - working tree chỉ có file riêng của người dùng, cộng thư mục `CODEX_REVIEW_c4d64a1_EVIDENCE/` chưa track ở gốc repo do
    người dùng đặt; thư mục này để nguyên;
  - Windows 10.0.26200, Python 3.11.9.
- **Review** `CODEX_REVIEW_c4d64a1.md` (12.912 B, sha256 `0a7dd8bd…`): **CHANGES REQUESTED** với R32-F1 và R32-F2, đều P2.
  Đã lưu `handoff/CODEX_REVIEW_c4d64a1.md`.
- **Bằng chứng** `CODEX_REVIEW_c4d64a1_EVIDENCE.zip` (132.958 B, sha256 `a4102d80…`):
  - testzip sạch; `SHA256SUMS.txt` khớp **42/42**, cả trên file giải nén lẫn trên blob đã commit;
  - giải nén vào `handoff/review_evidence/CODEX_c4d64a1/`, dùng `.gitattributes -text` để giữ nguyên byte 2 CSV có CRLF;
  - ZIP gốc để ở `handoff/archive/` (git ignore);
  - `reviewed_source/` trùng byte với bản `c4d64a1`.
- Commit: `1034695`.

## 1. Trước khi sửa (ĐÃ CHẠY)

| Lệnh | Kết quả | Log |
|:--|:--|:--|
| `python handoff/review_evidence/CODEX_c4d64a1/REPRO_REVIEW_c4d64a1.py --repo D:\LABOTARY\bytetrack --out handoff/logs/review33/codex_repro_truoc_sua` (script **giữ nguyên**) | **exit 1, `bug_count=5`**. Năm ca bắt buộc đều trả exit 0 trên mã cũ: thiếu root, thiếu train, NaN, `labels={}`, ID trùng trong frame. Đối chứng sạch đạt. Bốn ca COCO là **quan sát** (exit 0), không tính vào `bug_count` | `01_*`, `codex_repro_truoc_sua/` |
| `python handoff/review_evidence/CODEX_c4d64a1/audit_real_labels.py --nhan <cách ly>/box_track_20 --coco data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json --out handoff/logs/review33/02_audit_real_labels_tren_pc` | `real_labels_audit.json` trùng nội dung bản reviewer; `class_mismatches.csv` và `ignore_exceptions.csv` **trùng từng byte** với bằng chứng reviewer | `02_*` |

## 2. R32-F1 — diễn giải lớp/ignore

**Số liệu đã xác minh**, ghép theo (frame, `scalabel_id`):

| Kiểm | Kết quả |
|:--|:--|
| Cặp annotation | 465.512/465.512 |
| Box (w = x2−x1+1) | 0 lệch |
| Song ánh `scalabel_id ↔ instance_id` trong từng video | 0 xung đột ở cả hai chiều |
| **Lớp khác ngoài ba ánh xạ** (`other vehicle`→car, `other person`→pedestrian, `trailer`→truck) | **183** annotation, **46** track, **40** video: truck→car 73, pedestrian→car 62, car→pedestrian 20, car→bus 14, bus→truck 4, pedestrian→rider 3, car→rider 2, bus→pedestrian 2, bicycle→rider 1, truck→pedestrian 1, truck→bus 1. Toàn bộ có COCO ignore=0 |
| **Ngoại lệ ignore** so với quy tắc từng mô tả (“ba lớp phụ có ignore=1”) | **50**: 24 other person→pedestrian có ignore=0; 21 pedestrian→pedestrian có ignore=1; 5 other vehicle→car có ignore=0 |

- 183 và 50 là **hai phép kiểm**, không cộng thành số annotation khác nhau.
- Đã đối chiếu từng hàng với CSV của reviewer: **183/183, 50/50**, trùng byte. Cả script của reviewer lẫn bộ kiểm đã sửa của
  ta đều cho đúng các hàng này.
- Ví dụ thực: video `b1c9c847-3bda4659`, ID `00114202`, frame 88–91: native là `truck`, COCO là `car`.

**Nguyên nhân: UNKNOWN.**

- Repo không có mã chuyển đổi; tài liệu (`configs/phase2/LOCK_bdd_transfer.md`, `BDD_TRANSFER_NOTES.md`) chỉ ghi file COCO
  val là “COCO-video chuyển đổi”, sha256 `074ff795…`, không ghi nguồn.
- **Quan sát** (`09_quan_sat_coco_nguoi_dong_goi.json`): hai file COCO val do người đóng gói HF thêm vào cùng zip, đã cách ly
  từ vòng 32, có:
  - **ignore=0 cho mọi annotation**;
  - lớp **trùng native ở cả 183 vị trí lệch**.

  Điều này cho thấy gói HF nhất quán nội bộ, và COCO của dự án là một bản chuyển đổi khác. Nó **không** cho biết bên nào
  đúng hơn. Không mở thêm cuộc tìm nguồn.

**Đã đính chính**:

- `logs/review32/14_nguon_nhan_vong32.md`: thêm khối “Đính chính vòng 33”; sửa kết luận, các hàng 2/4/6 của bảng kiểm chứng,
  mục giới hạn và mục quyết định;
- `BDD100K_INVENTORY.md`: hai hàng ở mục Vòng 32, cộng mục Vòng 33;
- `REVIEW_BRIEF.md`.

Nội dung đính chính:

- chỉ kết luận **box và ID khớp**;
- **lớp/ignore còn khác**, chưa xác định bên nào đúng;
- thống kê so với bài báo chỉ là **tương đồng tổng hợp, có sai lệch**, không xác thực từng annotation.

**Giữ nguyên**: native cách ly, COCO val `074ff795…`, GT đã khóa và kết quả BDD lịch sử. Không sửa nhãn để ép khớp.

## 3. R32-F2 — bộ kiểm không được báo đạt khi thiếu bằng chứng

**Sửa** `handoff/logs/review32/kiem_nhan_ung_vien.py`, phạm vi tối thiểu. CLI cũ và các khóa JSON mà script Codex đọc đều
giữ nguyên.

| Yêu cầu | Thay đổi |
|:--|:--|
| Thiếu root, split bắt buộc, bộ nhãn rỗng | Báo `LoiDauVao`, **exit 2**. Kiểm một phần phải chọn rõ bằng `--split`; split không chọn ghi `NOT_RUN` |
| `labels=[]` hay sai kiểu | `[]` hợp lệ; `null` hoặc thiếu khóa hợp lệ theo đặc tả Scalabel nhưng **đếm riêng**; dict, chuỗi, số là lỗi `labels_khong_phai_list` (không biến thành frame rỗng) |
| Tọa độ | Phải là `int`/`float` thật và hữu hạn: chặn NaN/Inf (`box_khong_huu_han`) và boolean (`box_toa_do_bool`) |
| Trùng lặp | ID trùng trong frame bị chặn (`id_trung_trong_frame`). Frame trùng và khóa COCO trùng bị bắt **trước** khi đưa vào dictionary |
| Song ánh ID | Hai chiều, theo **từng video**: `id_mot_native_nhieu_coco` và `id_mot_coco_nhieu_native` |
| File COCO được yêu cầu mà thiếu hoặc hỏng | **exit 2**, trạng thái `LOI_DAU_VAO`. Bỏ qua phải chọn rõ bằng `--khong-doi-chieu-coco` (ghi `NOT_RUN` kèm lý do) |
| Trạng thái tách riêng | `trang_thai.schema_va_do_day_du` (`DAT`/`LOI`/`NOT_RUN`) tách khỏi `trang_thai.doi_chieu_coco` (`MATCH`/`DIFFERENCES`/`NOT_RUN`/`LOI_DAU_VAO`) |
| Mã thoát | 0 = đạt và COCO MATCH (hoặc NOT_RUN do chọn rõ); 1 = lỗi schema; 2 = thiếu đầu vào; **3 = COCO có khác biệt**. Exit 0 không bao giờ có nghĩa “có khác biệt” |

Ngoài ra:

- lớp/ignore dùng đúng định nghĩa của reviewer; thêm kiểm `iscrowd = crowd hoặc ignore`;
- ghi CSV từng hàng, cùng cột với bằng chứng reviewer;
- khác biệt lớp đã biết **không** bị ép thành lỗi schema.

## 4. Sau khi sửa (ĐÃ CHẠY)

| Lệnh | Kết quả | Log |
|:--|:--|:--|
| `python handoff/logs/review33/test_kiem_nhan_ung_vien.py` (regression mới, 30 ca, fixture tạm) | **30/30 PASS** | `03_test_ban_sua.log` |
| Cùng test với `--script <bản c4d64a1>` | **3/30 PASS**, 27 FAIL. Phần lớn là lỗi thật: exit 0 khi thiếu root/train, NaN, boolean, `labels={}`, ID trùng, COCO thiếu, khóa trùng, gộp ID, lệch box/lớp/ignore/iscrowd, annotation lẻ. Một phần do bản cũ chưa có tính năng mới: không có `trang_thai`, `--split`, `--khong-doi-chieu-coco`, bộ đếm riêng; ca 06 thì sập (traceback) khi thiếu thư mục ảnh | `03_test_ban_cu_c4d64a1.log` |
| Script Codex, giữ nguyên, trên bản sửa | **exit 0, `bug_count=0`**. Đối chứng sạch đạt; 5 ca bắt buộc trả exit 2/1. Quan sát: thiếu COCO exit 2; lệch bbox, đổi lớp, gộp track đều exit 3 (gộp track phát hiện 1 xung đột ID) | `04_*`, `codex_repro_sau_sua/` |
| Test cũ của vòng 32 (3 ca), chạy bản sao trong scratchpad trên bản sửa | **3/3 PASS** | `05_*` |
| Băm lại nhãn cách ly theo manifest, **trước** khi chạy | **1.602/1.602 khớp**; manifest `a2e416f2…`; COCO val `074ff795…` | `06_*` (`kiem_hash_nhan.py`) |
| Bộ kiểm đã sửa trên **1.400 train + 200 val + COCO val** | Đầu vào DAT; schema và độ đầy đủ **DAT** (train 278.079 frame = ảnh, val 39.973); COCO **DIFFERENCES**; **exit 3**. 0 lệch box, 0 xung đột ID, 0 annotation lẻ, 183 lệch lớp, 50 ngoại lệ ignore, 0 iscrowd ngoài quy tắc, 0 khóa trùng. CSV trùng byte với reviewer | `07_*` |
| Băm lại **sau** khi chạy | **1.602/1.602 khớp**; COCO không đổi | `08_*` |
| `kiem_nguon` trên bản cách ly | **exit 0**: 1.400 video, 278.079 frame, 0 ảnh thiếu | `10_*` |
| `kiem-nguon` chính thức | **exit 2**, báo thiếu `data/raw/bdd100k/labels/box_track_20/train`; nhãn chưa nhập | `11_*` |

**Lỗi công cụ và lỗi dữ liệu**: các ca NaN/Inf, boolean, `labels={}`, ID trùng, frame trùng là lỗi mà **công cụ cũ** có thể bỏ
lọt. Trên **1.600 file native thật**, cả reviewer lẫn bộ kiểm đã sửa đều cho kết quả:

- 0 tọa độ không hữu hạn;
- 0 ID trùng trong frame, 0 frame trùng;
- 0 sai kiểu `occluded`/`truncated`/`crowd`;
- 0 `labels` null hoặc thiếu khóa;
- 0 track đổi lớp.

## 5. NOT_RUN và giới hạn

- **NOT_RUN**: full suite tracking/Nano, vì đường chạy đó không đổi (`src/`, `configs/` không đổi). Nhập nhãn, khóa split,
  xuất, smoke, training đều chưa chạy vì ngoài phạm vi.
- **Nguồn nhãn**: snapshot bên thứ ba đã ghim revision HF `694197099692…` và hash từng file; **chưa xác thực bằng archive
  gốc** (md5 `6be40e0c…`).
- **Lớp/ignore** khác COCO val của dự án, nguyên nhân UNKNOWN. Phần train không có tham chiếu từng nhãn độc lập.
- **Phương án C** (review đề xuất): dùng có điều kiện, nhập nguyên byte 1.400 JSON train, giữ cách ly và manifest, không ghi
  đè COCO val. **Đang chờ người dùng chốt.**

## 6. Commit (cục bộ, không push)

| Commit | Nội dung |
|:--|:--|
| `1034695` | Ghi nhận review và bằng chứng |
| `5cf3653` | Sửa R32-F2, regression, script Codex trước/sau |
| `3c7c387` | Xác nhận R32-F1 trên PC, chạy trên nhãn thật, hash, quan sát COCO, `kiem-nguon` |
| Commit tài liệu | Phụ lục này, brief, `BDD100K_INVENTORY.md`, `CODEX_HANDOFF.md` (hàng 27–29), đính chính `14_nguon_nhan_vong32.md` |
