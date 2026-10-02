# Vòng 36 — ghi nhận nghiệm thu `2351c67`, đính chính R35-F1 và R35-F2 (chi tiết)

**Phạm vi**: ghi nhận nghiệm thu, rồi **chỉ đính chính diễn giải** và bàn giao.

- Không train, không inference, không pilot, không xuất dataset đầy đủ, không chạy lại smoke.
- Split đã khóa, dataset smoke, checkpoint, ngưỡng đánh giá, dữ liệu và kết quả lịch sử **giữ nguyên**.
- `src/` không đổi. Không Nano. Không đụng file riêng của người dùng.

## 0. Trạng thái đầu vòng

Log: `logs/review36/00_trang_thai_dau.log`.

- Nhánh `phase2-mixed-traffic`, HEAD `2351c67`, **0 commit** sau đó. Base review lúc đầu vòng: `ccd801f`.
- Working tree chỉ có file riêng của người dùng (`.gitignore`, `src/demo/**`, …) và thư mục chưa track như các vòng trước.
- Ghi thêm **tham chiếu mã nguồn** ultralytics 8.4.141 đang cài: `engine/trainer.py` sha256 `7532ff0b…`, hàm
  `_get_warmup_iterations`. Đây là đọc mã, **không phải log thực chạy**.

## 1. Ghi nhận nghiệm thu `2351c67` (ĐÃ CHẠY)

Commit `8a9dec8`.

- **Báo cáo** `D:\LABOTARY\review_codex\CODEX_REVIEW_2351c67.md`: 9.139 B, sha256 `e65a90a4…`.
  - Kết luận **PASS WITH LIMITATIONS** cho khóa split, xuất tập con và smoke 1 epoch.
  - Có hai điểm diễn giải mức P2: **R35-F1** và **R35-F2**.
  - Lưu nguyên byte thành `handoff/CODEX_REVIEW_2351c67.md`.
- **Bằng chứng** `CODEX_REVIEW_2351c67_EVIDENCE.zip`: 294.707 B, sha256 `2f8c66bd…`, CRC sạch.
  - Giải nén vào thư mục mới cạnh ZIP; ZIP gốc giữ nguyên.
  - `SHA256SUMS.txt` khớp **69/69**. Báo cáo trong ZIP trùng byte bản đứng riêng.
  - Lưu ở `handoff/review_evidence/CODEX_2351c67/` với `.gitattributes -text`; blob đã stage cũng khớp 69/69.
- **Gói đã nghiệm thu**: 378.096 B, sha256 `76e7d2ad…`, khớp báo cáo. Lưu ở `handoff/archive/review_delta_2351c67.zip`
  (git ignore).
- **Base review**: `ccd801f` → **`2351c67`** (vòng 35). 13 khai báo NPZ giữ nguyên.
- **Phạm vi reviewer**:
  - đã kiểm: 75 đường dẫn thay đổi (74 payload + 1 alias, 6 preimage); split và tập con (tính lại từ split và manifest nhãn
    đã nghiệm thu); test công cụ log (12/12 trên Linux UTF-8; **không** phải tự tái hiện errno 22 của Windows); đối chiếu log
    smoke; tính lại dự toán (JSON trùng byte với bản bàn giao);
  - **không** làm: lặp lại training GPU, kiểm byte checkpoint, xác minh toàn cây Git 3.849 file, tính lại thành viên từng
    tầng từ 1.400 JSON, đếm box từ ảnh/nhãn nguồn.

## 2. R35-F1 — phân biệt quan sát với nguyên nhân (ĐÃ SỬA, chỉ tài liệu)

Đối chiếu lại **chỉ từ log và JSON đã có**, không nạp model: `logs/review36/04_doi_chieu_R35-F1.{py,log}`.

| Điểm cần tách | Bằng chứng hiện có | Hệ quả |
|:--|:--|:--|
| Phép so của `13_*` | **COCO 80 lớp** (file khởi tạo) so với **checkpoint smoke 5 lớp** (`best.pt`) | Chênh confidence gộp hai ảnh hưởng |
| Model 5 lớp sau khởi tạo, **trước** bước tối ưu đầu | **Không có** checkpoint hay số đo. Log chỉ ghi `nc=80 → 5`, "Remapped 5/5 cls head rows", chuyển 322/355 (v8n) và 451/499 (11n) tensor | Ảnh hưởng của bước chuyển sang 5 lớp chưa đo riêng |
| Warmup/LR thực chạy | Runner chỉ chụp LR lúc `on_train_start` và cuối epoch (đều 0,01). "Không warmup khi `epochs=1`" chỉ từ **đọc mã nguồn** | Chưa có log thực chạy |
| Conf 1e-6 (`13b_*`) | YOLO11n: car AP50 ≈ 0,233 (25.199 dự đoán); truck/bus AP50 0; motorcycle/bicycle không có GT | Chỉ về **car**, trên 84 ảnh dev smoke |

**Câu đã đính chính**: checkpoint smoke có confidence thấp hơn checkpoint COCO 80 lớp; **nguyên nhân chưa xác định**. Warmup
và trạng thái khởi tạo model 5 lớp là các yếu tố cần phân biệt.

**Rút lại** (vượt bằng chứng):

1. "Smoke 1 epoch không warmup làm điểm lớp giảm mạnh" và các biến thể của câu này.
2. "Ánh xạ lớp và tọa độ đúng; không phải lỗi đường ống". Kết quả conf 1e-6 chỉ cho thấy có dự đoán car khớp GT ở điểm thấp.
3. Lý do "pilot có warmup để xem điểm lớp có hồi lại không". Chỉ tăng epoch không tạo phép đối chứng nguyên nhân sạch.

**Giữ nguyên**:

- Đánh giá chính ở **conf 0,001** (YOLOv8n mAP50 0,098 trên 3 lớp có GT; YOLO11n 0). Kết quả conf 1e-6 vẫn là chẩn đoán riêng.
- Số liệu `13_*`, `13b_*`, checkpoint và log.
- Giới hạn: dev smoke không có motorcycle hay bicycle.
- Script `13_*`, `13b_*` và JSON của chúng giữ nguyên làm bằng chứng của lần đã chạy. Docstring của chúng ghi ý định lúc đó
  ("vì sao …", "phân biệt … lỗi đường ống"); diễn giải được đính chính tại đây.

**Đề xuất chờ duyệt** (chưa làm): nếu cần chẩn đoán, đo model 5 lớp trước bước học và lưu số warmup iteration/LR thực tế
từng bước.

**Nơi đã sửa**:

- `REVIEW35_CHI_TIET.md`: khối đính chính đầu trang, mục 7, mục 10;
- `DETECTOR_TRAIN_PLAN.md`: đầu trang, mục 5a, mục 8;
- `PLAN_DA_CHOT.md`: mốc 2;
- `CODEX_HANDOFF.md`: hàng 31, thêm hàng 32;
- `REVIEW_BRIEF.md`: viết mới.

## 3. R35-F2 — dự toán là kịch bản, không phải bảo đảm (ĐÃ SỬA)

Commit `2b15b88`.

- **Sửa tại nguồn** `logs/review35/15_du_toan_training.py`: **chỉ chuỗi mô tả** (docstring, `loai`, 3 `ghi_chu` trong
  `GIA_DINH`, các dòng `print`). Bỏ nghĩa "cận trên"/"tối đa"; ghi rõ thời gian, RAM và VRAM thực tế có thể vượt cả kịch bản
  chậm hơn. Công thức, tên khóa và đầu vào **không đổi**, thấy được trong `git diff`.
- **Sinh lại** vào `logs/review36/02_du_toan_sinh_lai.{json,log}`. JSON và log gốc vòng 35 giữ nguyên; sha256 kiểm lại khớp.
- **Đối chiếu số** bằng `logs/review36/doi_chieu_so_json.py`. Bộ đối chiếu tự kiểm 7 ca (`01_*`): khác số, khác kiểu, khác cấu
  trúc thì exit 1; chỉ khác chuỗi thì exit 0. Kết quả (`03_*`):
  - so với JSON gốc vòng 35: **162 giá trị số và cấu trúc giống hệt**; chỉ 4 chuỗi đổi;
  - so với JSON reviewer tính lại: cũng **162/162** giống hệt.
- **Bảng số giữ nguyên**:

  | Bước frame / epoch | Tổng hai model | Kịch bản chậm hơn |
  |:--|--:|--:|
  | 5 / 15 | **56,64 h** | **73,95 h** |
  | 5 / 30 | 113,08 h | 147,61 h |
  | 10 / 15 | 29,10 h | 37,99 h |
  | 10 / 30 | 58,08 h | 75,81 h |

- **Tài liệu**:
  - "(cao …)" và "mức cao" đổi thành "**kịch bản chậm hơn**";
  - bỏ "cận trên khuyến nghị";
  - RAM/VRAM: bỏ dấu `≤`, ghi thành khoảng kịch bản ước tính từ smoke, kèm câu thực tế có thể vượt (kể cả OOM nếu lô nhiều
    box hơn smoke);
  - câu "RSS … là cận trên" đổi thành "tổng có thể đếm trùng trang chia sẻ".

## 4. Kiểm sau đính chính (ĐÃ CHẠY)

`logs/review36/05b_kiem_khong_doi.log`: **ĐẠT**.

- File split sha256 `b8a2848e…`; dataset smoke (`anh.csv`, `data.yaml`) khớp `tom_tat.json`.
- Kết quả runner hai model trùng byte bản đã commit. 4 checkpoint (`last`/`best` × 2) khớp sha256 đã ghi nhận.
- So với `2351c67`, trong `logs/review35/`, `configs/` và `src/` (trừ file riêng `src/demo/**`), chỉ
  `15_du_toan_training.py` đổi (chuỗi mô tả). Không log, JSON hay số đo gốc nào đổi.
- `05a_*` là lần kiểm đầu, báo "CO KHAC" sai do phép kiểm bắt cả file riêng `src/demo/rerun_with_state_log.py` của người
  dùng (đã sửa từ trước vòng 35). Giữ lại, có ghi chú.
- Tìm lại các câu bị rút trong mọi trang bàn giao: chỉ còn trong chính báo cáo review và trong câu trích ở đoạn đính chính.
- Test: chỉ tự kiểm bộ đối chiếu mới (7/7 đúng kỳ vọng). **Full suite NOT_RUN** vì `src/` không đổi. Không chạy lại smoke.

## 5. Chưa chạy, giới hạn

- **NOT_RUN**: training, pilot, inference, xuất dataset đầy đủ, chạy lại smoke, full suite.
- Nguyên nhân confidence thấp ở checkpoint smoke **vẫn chưa xác định**.
- Dự toán vẫn là kịch bản theo giả định, ngoại suy từ một epoch 42 batch.
- Nhãn vẫn là snapshot bên thứ ba (phương án C). 183 lệch lớp và 50 ngoại lệ ignore của native val so với COCO lịch sử chưa
  giải quyết; ngoài phạm vi vòng này.

## 6. Quyết định chờ người dùng duyệt

1. **Pilot thêm epoch**: có chạy hay không. Nếu mục đích là tìm nguyên nhân confidence thấp, cần thêm phép đo model 5 lớp trước
   bước học và log warmup/LR (đề xuất ở mục 2).
2. **Xuất dataset đầy đủ** (bước 5, ~8,5 GB, kịch bản ước tính).
3. **Số epoch** (đề xuất 15).
4. **Bước frame** (đề xuất 5).
5. **Ngân sách training local**: kịch bản 56,64 h, kịch bản chậm hơn 73,95 h cho bước 5 / 15 epoch; không phải bảo đảm.
   Nên đóng bớt ứng dụng vì RAM thấp.

## 7. Commit vòng 36

| Commit | Nội dung |
|:--|:--|
| `8a9dec8` | Ghi nhận nghiệm thu `2351c67`, bằng chứng, base → `2351c67` |
| `2b15b88` | R35-F2 tại nguồn: chuỗi mô tả script dự toán, sinh lại, đối chiếu 162 số |
| (commit tài liệu) | R35-F1/F2 trong tài liệu, đối chiếu R35-F1, kiểm không đổi, brief, phụ lục này |
