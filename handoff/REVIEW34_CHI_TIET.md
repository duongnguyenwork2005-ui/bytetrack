# Vòng 34 — ghi nhận nghiệm thu `4e71d32` và nhập nhãn train theo phương án C (chi tiết)

**Phạm vi**: ghi nhận nghiệm thu, nhập nguyên byte 1.400 JSON native train, chạy `kiem-nguon` trên PC.

- **Không** khóa split, xuất dataset, smoke hay training.
- Không SSH Nano; không sửa `src/` hay `configs/`.
- Không đụng file riêng của người dùng.

## 0. Trạng thái đầu vòng

Log: `logs/review34/00_trang_thai_dau.log`.

- Nhánh `phase2-mixed-traffic`, HEAD `4e71d32` (đúng dự kiến, 0 commit sau đó).
- Working tree chỉ có file riêng của người dùng và thư mục `CODEX_REVIEW_c4d64a1_EVIDENCE/` chưa track ở gốc repo; để
  nguyên.

## 1. Ghi nhận nghiệm thu (ĐÃ CHẠY)

- **Báo cáo**: đọc thẳng từ `D:\LABOTARY\review_codex\`.
  - `CODEX_REVIEW_4e71d32.md`: 8.772 B, sha256 `1a4b6106…`, **PASS WITH LIMITATIONS**, đóng R32-F1 và R32-F2.
  - Đã lưu thành `handoff/CODEX_REVIEW_4e71d32.md`.
- **Bằng chứng**: `CODEX_REVIEW_4e71d32_EVIDENCE.zip`, 280.474 B, sha256 `85141100…`, testzip sạch.
  - Giải nén vào thư mục mới `D:\LABOTARY\review_codex\CODEX_REVIEW_4e71d32_EVIDENCE\`; ZIP gốc giữ nguyên.
  - `SHA256SUMS.txt` khớp **56/56**, không file thừa. Báo cáo trong ZIP trùng byte với bản đứng riêng.
  - Lưu vào `handoff/review_evidence/CODEX_4e71d32/` (`.gitattributes -text` cho các CSV CRLF); blob đã commit vẫn khớp 56/56.
- **Gói delta đã nghiệm thu**: `bcb92f8` → `4e71d32`, **1.156.972 B**, sha256 `a34037a1…`, khớp báo cáo.
  - Lưu thành `handoff/archive/review_delta_4e71d32.zip` (git ignore).
- **Base review**: chuyển `bcb92f8` → **`4e71d32`** bằng `make_review_delta.py --set-base 4e71d32 --round 33`, mang sang 13 khai
  báo NPZ. Base không chạy theo HEAD.
- Commit: `6aaaed7`.

## 2. Kiểm bản cách ly trước khi nhập (ĐÃ CHẠY)

Script `logs/review34/nhap_nhan_train.py --chi-kiem`, kiểm thử fixture **9/9** (`01_*`). Log: `02_kiem_truoc_nhap.{json,log}`.

| Kiểm | Kết quả |
|:--|:--|
| Hàng train trong manifest vòng 32 (sha256 `a2e416f2…`) | 1.400 |
| JSON train trong bản cách ly | **1.400**: 0 thiếu, 0 thừa, **0 lệch byte/sha256** |
| Nguồn | HF `jaffe03195/bdd100k_sot`, revision `694197099692fe1254e93215e73e68ece8b1f41f` |
| COCO val `data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json` | sha256 `074ff795…`, khớp giá trị đã nghiệm thu |
| Thư mục đích | chưa tồn tại, 0 xung đột |

## 3. Nhập nhãn train (ĐÃ CHẠY)

Lệnh `python handoff/logs/review34/nhap_nhan_train.py --ra … --ra-tsv …`, chạy từ 11:40:29 đến 11:40:53. Log:
`03_nhap_nhan_train.{json,log}`.

- Chép nguyên byte 1.400 file vào `data/raw/bdd100k/labels/box_track_20/train/<video>.json`, qua file tạm rồi đổi tên.
  **Ghi mới 1.400**, không có file nào có sẵn.
- **Kiểm sau nhập**:
  - đích có 1.400 file `.json`, **1.400/1.400 khớp sha256 manifest**, 0 file lạ;
  - nguồn cách ly vẫn đủ 1.400 file và vẫn khớp;
  - COCO val không đổi.
- **Manifest nhập**: `03_manifest_nhap_nhan_train.tsv` (sha256 `c5a548bc…`). Provenance để ở handoff, **không** đặt trong
  thư mục nhãn.
- **Bản cách ly sau nhập**: băm lại cả 1.602 file (train, native val, hai COCO phụ) bằng `logs/review33/kiem_hash_nhan.py`,
  kết quả **1.602/1.602 khớp** (`04_*`).
- **Chỉ nhập native train**. Native val, hai COCO phụ của mirror và COCO val của dự án giữ nguyên chỗ cũ.

## 4. `kiem-nguon` đường chính (ĐÃ CHẠY)

```text
python src/prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml --buoc kiem-nguon
```

**exit 0** (`05_kiem_nguon_chinh.log`):

| Mục | Kết quả |
|:--|:--|
| Định dạng | `scalabel` |
| Video có nhãn / video có ảnh | **1.400 / 1.400** |
| Video có nhãn không có ảnh | 0 |
| Video có ảnh không có nhãn | 0 |
| Video thiếu ảnh lẻ | 0 |
| Frame trong nhãn | **278.079** |
| Ảnh thiếu | 0 |
| Video dùng được | **1.400** |

**Phân biệt**:

1. `kiem-nguon` đạt nghĩa là **dữ liệu đủ đầu vào để đi tiếp** theo hợp đồng của bộ kiểm.
2. Đối chiếu native val với COCO val **vẫn có 183 lệch lớp và 50 ngoại lệ ignore**; nguyên nhân UNKNOWN.
3. Không điều nào ở trên xác thực nhãn là bản phát hành chính thức.

## 5. Provenance và quyết định sử dụng

- Người dùng **chọn phương án C**: dùng có điều kiện snapshot bên thứ ba đã ghim revision và hash từng file.
- **Chưa xác thực bằng archive gốc** (md5 `6be40e0c…`); train chưa có tham chiếu độc lập theo từng annotation.
- Giữ nguyên COCO val và các kết quả lịch sử.
- Nếu sau này có archive gốc: đối chiếu đường dẫn và sha256 từng file, báo khác biệt **trước** khi thay snapshot đã dùng.
- Đã cập nhật `BDD100K_INVENTORY.md` (mục Vòng 34), `PLAN_DA_CHOT.md` (mốc 2) và `CODEX_HANDOFF.md` (hàng 29 sang trạng thái
  đã nghiệm thu, thêm hàng 30).
- Giữ nguyên lựa chọn huấn luyện local trên PC và kế hoạch đã chốt.

## 6. Chưa chạy

- Test suite: **NOT_RUN** vì không đổi mã dự án. Script mới chỉ nằm trong `logs/review34/` và có kiểm thử fixture 9/9.
- Khóa split, xuất dataset, smoke 2 model, training: **CHƯA CHẠY**; đây là bước tiếp theo theo kế hoạch.
- Training dài, epoch và ngân sách: **chưa duyệt**.
