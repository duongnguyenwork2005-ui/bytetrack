# REVIEW_BRIEF — vòng 31–32: dữ liệu BDD100K MOT train (ảnh đủ; nhãn ứng viên đang cách ly)

<!-- BASE_TARGET -->

**Phạm vi**: chỉ chạy trên PC. Không SSH hay đổi Nano, không training, **không sửa mã dự án** (`src/`, `configs/` không đổi).
Chỉ có script dữ liệu / kiểm chứng trong `handoff/logs/review31/`, `review32/` và tài liệu. Base review giữ **`bcb92f8`**.
Không commit dataset hay weights.

## Vòng 31 — ảnh MOT train: ĐÃ CHẠY

- **Tải đủ 7/7 `images20-track-train-*.zip`** từ `http://dl.yf.io/bdd100k/mot20/`, danh sách lấy từ listing thật. Mỗi file:
  - HTTP 200, `application/zip`;
  - kích thước bằng Content-Length;
  - **md5 khớp `.md5` của máy chủ**, `testzip` OK.

  sha256: `logs/review31/20_manifest_tai_anh_train.md`. Đối chiếu độc lập ở vòng 32: mirror HF `vanthanh/BDD100kMOT` có sha256
  LFS trùng 7/7.
- **Giải nén**: kiểm từng member trước khi ghi, ghi qua file tạm rồi đổi tên, không ghi đè nội dung khác. Kết quả
  **1.400 video, 278.079 ảnh**. Kiểm kê độc lập khớp 100 % với member của archive; val giữ nguyên.
- **Sự cố**: khi `curl` bị giết, script cắt nhầm `.part` của phần 2 (mất khoảng 1,2 GB, phải tải lại).
  - Sửa ở `8b142e2`. Regression trên bản cũ được 22/25, FAIL đúng ca sự cố; bản sửa 25/25, và 27/27 sau khi thêm ca dọn
    khóa.
  - Tác vụ nền bị giới hạn khoảng 2 giờ, nên thêm `--don-khoa-cu` để dọn khóa của tiến trình đã chết (`02e7347`).

## Vòng 32 — nhãn MOT 2020 train: ĐÃ CHẠY, kết quả là ứng viên đang **cách ly**

- **md5 chính thức** `6be40e0c…`: lấy từ `doc/source/download.rst` dòng 268, commit `9ac17c6c`; đã tính lại blob hash.
- **Lượt tìm** (~28 phút, vượt mức ~20 phút): Internet Archive trả 429 cho mọi yêu cầu; tìm web không ra mirror; đã xem 57
  dataset trên Hugging Face. **Không có bản sao kiểm được md5.**
- **Ứng viên**: `box_track_20/{train,val}` trong `jaffe03195/bdd100k_sot/bdd100k.zip`, zip đóng gói lại, người tải lên ẩn danh.
  - Chỉ tải riêng phần nhãn bằng HTTP Range: 196,5 MB trên tổng 5,35 GB.
  - 1.602 file, CRC32 khớp. Cách ly ở `data/raw/ung_vien_nhan/`; **chưa nhập**.
- **Kiểm chứng** (`logs/review32/14_nguon_nhan_vong32.md`):
  - schema 0 lỗi;
  - train khớp 1:1 với ảnh chính thức: 1.400 video, 278.079 frame;
  - val khớp **465.512/465.512** nhãn với file COCO val của dự án (box tuyệt đối, track id nhất quán);
  - ảnh val trong zip khớp CRC32 39.973/39.973 với zip val chính thức;
  - thống kê theo lớp khớp Bảng 11 của bài báo BDD100K (130.573 track so với “130.6K”).
- **`kiem-nguon`**:
  - chính thức: **exit 2**, chỉ còn thiếu nhãn (`13_kiem_nguon_chinh.log`);
  - cùng `kiem_nguon` trên bản cách ly: **exit 0** (`12_*`).

## Chưa chạy

- **Chờ quyết định A/B/C**: nhập nhãn, khóa split, xuất dataset, smoke, training. A = nhập sau review; B = chờ gói gốc
  (thư nháp tiếng Anh `15_*`, chưa gửi); C = tạm dùng ứng viên rồi so sha256 khi có gói gốc.
- **Test suite**: NOT_RUN vì `src/` không đổi. Kiểm thử cục bộ của các script trong `logs/`:

  | Script | Kết quả |
  |:--|:--|
  | tải | 27/27 |
  | giải nén | 18/18 |
  | kiểm kê | 4/4 |
  | đọc zip từ xa | 7/7 |
  | kiểm nhãn | 3/3 |

## Đề nghị reviewer xem

1. Bằng chứng ở mục “Kiểm chứng” có đủ để nhập ứng viên (A hoặc C) không, hay phải chờ md5 (B).
2. Giới hạn đã ghi:
   - phần train không có tham chiếu từng nhãn độc lập;
   - tài liệu không ghi nguồn gốc file COCO val của dự án;
   - nhãn license `apache-2.0` trên repo HF không có giá trị.
3. `liet_ke_zip_tu_xa.py` và `kiem_nhan_ung_vien.py`: tải đúng tiền tố, CRC, không ghi đè; cách ghép nhãn theo `scalabel_id`.
