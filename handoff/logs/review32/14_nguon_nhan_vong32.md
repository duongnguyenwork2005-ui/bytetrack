# Vòng 32 — tìm và kiểm chứng nhãn BDD100K MOT 2020 train

Thời điểm: 2026-10-01, HEAD đầu vòng `dce28dd`, base review giữ `bcb92f8`.

## Kết luận

- **Chưa có bản sao kiểm được bằng md5 chính thức** `6be40e0ca56a83ddeba2ed6bff50f9e6`. Không nguồn nào truy cập được còn
  gói `box_track_labels_trainval.zip` / `bdd100k_box_track_20_labels_trainval.zip`.
- **Có một ứng viên**: file Scalabel `box_track_20/{train,val}` nằm trong một zip **đóng gói lại** trên Hugging Face
  (`jaffe03195/bdd100k_sot`, `bdd100k.zip`).
  - Chỉ tải **riêng phần nhãn**: 1.602 member, CRC32 khớp.
  - Đang **cách ly** ở `data/raw/ung_vien_nhan/hf_jaffe03195_bdd100k_sot/`.
  - **Chưa nhập** vào `data/raw/bdd100k/labels/box_track_20/train`.
- Có bằng chứng gián tiếp mạnh rằng đây là nhãn gốc (mục “Kiểm chứng”), nhưng **không phải md5**. Theo yêu cầu, bản này phải
  được người dùng / reviewer xem trước khi nhập.
- `kiem-nguon` chính thức vẫn **exit 2** (`13_kiem_nguon_chinh.log`). Chạy cùng `kiem_nguon` trên bản cách ly được **exit 0**
  (`12_kiem_nguon_ban_cach_ly.log`).

## Nguồn của mã md5

`bdd100k/bdd100k`, file `doc/source/download.rst`, nhánh `master`, commit `9ac17c6c7c51d2fc83065fccd707cd5b1882a293`
(2024-03-09, “Update download link (#339)”).

- Blob `ae399a3ade3de172f9cd28681529c636c5016ed1`: đã tính lại blob hash của git từ bản raw, khớp API.
- Bản raw trùng từng byte với bản lưu ở vòng 31.
- Dòng 257 là “MOT 2020 Labels”; **dòng 268** là `| md5 | 6be40e0ca56a83ddeba2ed6bff50f9e6 |`; mục này ghi kích thước 115MB.
- PR #339 đổi nơi tải từ cổng đăng nhập Berkeley sang `dl.cv.ethz.ch/bdd100k/data/`.

Log: `01_download_rst_at_<commit>.rst`, `01_github_contents_download_rst.json`, `01_github_commit_download_rst_tom_tat.json`
(đã bỏ email tác giả commit).

## Lượt tìm

Từ 18:24 đến 18:52, khoảng **28 phút**, **vượt mức ~20 phút** vì phải đọc mục lục từ xa của các zip ứng viên. Không thử lại
các link chính thức đã chết ở vòng 31.

| # | Nguồn | Đã làm | Kết quả |
|:--|:--|:--|:--|
| 1 | Nguồn chính thức: ETH, nút Berkeley, `dl.yf.io`, `doc.bdd100k.com` | **Không thử lại** | Đã thất bại ở vòng 31 (`../review31/13_nguon_nhan_da_thu.md`) |
| 2 | Internet Archive: Wayback CDX và API “available” cho `dl.cv.ethz.ch/bdd100k/data/` và các URL zip nhãn | 8 yêu cầu, có nghỉ giữa các lần | **HTTP 429 Too Many Requests** cho mọi yêu cầu, lần cuối lúc 18:52. **Chưa xác minh** được Wayback có lưu file hay không (`02_*`) |
| 3 | Hugging Face: tìm “bdd100k”, “bdd-100k” (57 dataset); đọc danh sách file của 10 dataset ứng viên | API công khai | Chi tiết từng dataset ở bảng ngay dưới (`03_*`, `04_muc_luc_tu_xa/`) |
| 4 | Tìm web theo đúng hai tên file | 2 truy vấn | Không có mirror |
| 5 | `SysCV/pcan` `docs/GET_STARTED.md` (cùng nhóm maintainer) | Đọc bản raw | Chỉ trỏ về trang chính thức (`05_*`) |
| 6 | `Erotemic/bdd-data` README | 2 nhánh | 404 |
| — | Kaggle, Google Drive, Baidu, torrent | **Không thử** | Cần đăng nhập, hoặc không có link cụ thể đã xác minh |

Kết quả trên Hugging Face:

| Dataset | Kết quả |
|:--|:--|
| `vanthanh/BDD100kMOT` | Chỉ là mirror **ảnh** MOT. sha256 LFS của train-1..7 **trùng 7/7** với sha256 vòng 31 tính trên ảnh của ta. Không có nhãn |
| `jaffe03195/bdd100k_sot`, `bdd100k.zip` | **Có `bdd100k/labels/box_track_20/`** → ứng viên |
| `jaffe03195/bdd100k_sot`, `bdd_created.zip` | Mục lục chỉ có thư mục `img/` và `vis/` theo từng video (có vẻ là dữ liệu tự tạo); **0** file JSON hay nhãn |
| `bdd100kmot_vehicle.zip` | 38,7 GB, có ở hai repo. Tên gợi ý bản chuyển đổi chỉ gồm xe; **không mở** |
| `2inf/bdd100k` | Gated, chỉ có `solesensei_bdd100k.zip`; tên gợi ý gói 100K; **không mở** |

## Ứng viên

- **URL ghim theo revision**:
  `https://huggingface.co/datasets/jaffe03195/bdd100k_sot/resolve/694197099692fe1254e93215e73e68ece8b1f41f/bdd100k.zip`
- **Repo**:
  - tạo ngày 2025-12-12; `bdd100k.zip` được thêm ở commit `982b8d0d` (2025-12-15T02:17:53Z);
  - README chỉ có `license: apache-2.0`, khác license BDD100K;
  - không có mô tả hay nguồn gốc; trông như không gian làm việc của một dự án SOT (có pptx, docx, weights).
- **Zip**: 5.352.373.031 B, LFS sha256 `6f5b9d47…`. **Không tải cả file**: chỉ đọc mục lục và các member `box_track_20/`
  bằng HTTP Range, tổng **196,5 MB** (mục lục 30,8 MB, member 165,7 MB). Script: `liet_ke_zip_tu_xa.py`, kiểm thử cục bộ
  7/7.
- **Nội dung zip** (200.291 file):

  | Nhóm | Số file |
  |:--|--:|
  | `labels/100k` | 80.000 |
  | `images/track/val` | 39.973 |
  | `labels/track` và `filter_labels/track` (có thể do người đóng gói tạo) | 39.358 mỗi nhóm |
  | `labels/box_track_20` | **train 1.400 + val 200** + 2 file COCO val của người đóng gói |

- **Cách ly**: `data/raw/ung_vien_nhan/hf_jaffe03195_bdd100k_sot/` gồm phần nhãn và mục lục đầy đủ `muc_luc_bdd100k_zip.tsv`.
  Thư mục bị git ignore.
- **Manifest**: `09_manifest_nhan_ung_vien.tsv` (1.602 file: byte, CRC32, sha256, md5). Vân tay bộ nhãn `a317cdce…`
  (`09_manifest_nhan_ung_vien_tom_tat.json`).

## Kiểm chứng

| # | Kiểm | Kết quả |
|:--|:--|:--|
| 1 | CRC32 từng member, kiểm khi đọc rồi tính lại | **1.602/1.602 khớp** |
| 2 | JSON và schema Scalabel (`kiem_nhan_ung_vien.py`, test 3/3): frame `name/videoName/frameIndex`; nhãn `id/category/box2d/attributes`; box hợp lệ | **0 lỗi** ở train và val |
| 3 | Train ↔ ảnh **chính thức** đã xác minh ở vòng 31 | Video **1.400/1.400**, frame **278.079/278.079**: 0 thiếu, 0 thừa. 3.930 frame không có nhãn, chỉ ghi nhận, **không coi là âm tính** |
| 4 | Val ↔ file COCO-video val của dự án (sha256 `074ff795…`), ghép theo (frame, `scalabel_id`) | **465.512/465.512** cặp; box **khớp tuyệt đối** với quy ước w = x2−x1+1; 0 track id bất nhất. Ánh xạ lớp đúng kiểu chuyển đổi của BDD (`other vehicle`→car, `other person`→pedestrian, `trailer`→truck, đều ignore=1). 4.018 nhãn có crowd khác iscrowd, **toàn bộ** là nhãn ignore=1 (`07b_*`) |
| 5 | Ảnh val trong zip HF ↔ zip val **chính thức** `images20-track-val-1.zip` | **39.973/39.973** khớp tên, kích thước và CRC32 (`08_*`) |
| 6 | Thống kê train+val ↔ bài báo BDD100K (CVPR 2020, arXiv 1805.04687): §3.6 và Bảng 11 | 1.600 video, 318.052 frame, 130.573 track (bài báo: “130.6K”), box 8 lớp 3.320.597 (“3.3M”). Từng lớp khớp ở độ chính xác bài báo công bố: motorcycle 685/685 track, truck 177.294 box (“177K”). Lệch nhỏ duy nhất ở lớp train: 30 so với 34 track, 547 so với 557 truncated (`11_*`) |
| 7 | `kiem_nguon` của dự án trên bản cách ly (đổi đường dẫn nhãn trong bộ nhớ, `kiem_nguon_ban_cach_ly.py`) | **exit 0**: 1.400 video, 278.079 frame, 0 ảnh thiếu |
| 8 | `kiem-nguon` chính thức (chưa nhập nhãn) | **exit 2**, chỉ thiếu nhãn |

## Giới hạn

- **Không kiểm được md5 chính thức**: không có gói gốc; ứng viên nằm trong zip đóng gói lại của người tải lên ẩn danh.
- **Phần train không có tham chiếu từng nhãn độc lập.** Chỉ có tính nhất quán nội bộ (khớp 1:1 với ảnh chính thức) và thống
  kê tổng hợp đã làm tròn của bài báo.
- **Phép đối chiếu val** dựa vào file COCO val của dự án. File này có từ trước vòng 27 (trùng byte với bản
  `D:\LABOTARY\01-bdd_box_track_val_cocofmt.json`), nhưng **tài liệu dự án không ghi nơi tải gốc**.
- **License**: dữ liệu vẫn thuộc license BDD100K; nhãn `apache-2.0` trên repo HF không có giá trị.
- Nếu sau này có gói gốc khớp md5: so sha256 từng file với `09_manifest_nhan_ung_vien.tsv` là xác nhận dứt khoát được ngay.

## Cần người dùng / reviewer quyết định

- **A.** Chấp nhận ứng viên sau review: chép 1.400 JSON train vào `data/raw/bdd100k/labels/box_track_20/train/`, ghi nguồn và
  manifest sha256, rồi chạy `kiem-nguon` (dự kiến exit 0).
- **B.** Chờ gói gốc từ maintainer hoặc lab đã tải trước đây (thư nháp: `15_thu_nhap_xin_nhan_goc_EN.md`).
- **C.** Kết hợp: tạm dùng ứng viên cho bước khóa split / smoke; khi có gói gốc thì so sha256 từng file với manifest và thay
  nếu khác.
