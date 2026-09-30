# Vòng 31 — nguồn nhãn MOT 2020 (`box_track_20/train`) đã thử

Thời điểm: 2026-09-30, từ PC (mạng gia đình, DNS mặc định 192.168.0.1). Chỉ đọc listing, header và tài liệu công khai; không
đăng nhập, không tải gói nhãn nào.

**Kết luận: CHƯA lấy được nhãn Scalabel gốc `box_track_20/train`.** Không nguồn chính thức nào còn phục vụ gói này. Không dùng
Detection 2020 (`det_20`), MOTS, nhãn 2018 hay bản COCO chuyển đổi để thay.

## Gói cần có, theo tài liệu chính thức

Theo `bdd100k/bdd100k` (GitHub, nhánh `master`), file `doc/source/download.rst`, mục “MOT 2020 Labels”; bản lưu:
`07_github_bdd100k_doc_source_download.rst`.

| Thuộc tính | Giá trị |
|:--|:--|
| Kích thước | 115 MB |
| md5 | `6be40e0ca56a83ddeba2ed6bff50f9e6` |
| Nội dung | `bdd100k/labels/box_track_20/{train,val}` |
| Định dạng | Scalabel, 5 Hz |
| Nơi tải ghi trong tài liệu | `https://dl.cv.ethz.ch/bdd100k/data/` |

Tên file không ghi trong tài liệu:
- trang tải Berkeley ghi `bdd100k_box_track_20_labels_trainval.zip`;
- một người dùng GitHub (issue #369) ghi `box_track_labels_trainval.zip` theo listing ETH cũ — lời kể của bên thứ ba, chưa
  xác minh.

## Các nguồn đã thử

| # | Nguồn | Đã làm | Kết quả | Bằng chứng |
|:--|:--|:--|:--|:--|
| 1 | Mirror ETH `https://dl.cv.ethz.ch/bdd100k/data/` (người dùng nêu; tài liệu chính thức ghi) | curl HTTP và HTTPS; `nslookup` qua resolver nhà và 8.8.8.8 | **NXDOMAIN** ở cả hai resolver; curl lỗi 6 (không phân giải được tên). `cv.ethz.ch` cũng NXDOMAIN; `www.ethz.ch`, `vision.ee.ethz.ch` vẫn phân giải bình thường, nên DNS máy hoạt động | `04_eth_mirror_thu.txt`, `04_dns_thu.txt` |
| 2 | Trang tải chính thức `bdd-data.berkeley.edu/download.html` | GET qua HTTP. HTTPS lỗi chứng chỉ `SEC_E_WRONG_PRINCIPAL` | HTTP 200, là trang tĩnh. Nút **“MOT 2020 Labels” bị comment trong HTML**. Các nút đang hiện: video, ảnh 100K/10K, “Labels”, segmentation/drivable bản cũ, ảnh MOT 2020, ảnh MOTS 2020, Detection 2020 Labels — **không có nhãn MOT** | `05_bdd_data_berkeley_download.html` (đã ẩn một email cá nhân, xem `05_..._nguon_goc.json`), `05_trang_chinh_thuc_thu.txt` |
| 3 | URL nhãn trong nút bị comment: `http://128.32.162.150/bdd100k/2021/bdd100k_box_track_20_labels_trainval.zip` | HEAD qua IP và qua `dl.yf.io` | **HTTP 404**; thư mục `/bdd100k/2021/` cũng 404 | `06_nhan_mot20_head.txt` |
| 4 | Máy chủ ảnh `http://dl.yf.io/bdd100k/` (cùng IP 128.32.162.150) | Đọc listing gốc và các thư mục `mot20/`, `mots20/`, `images/`, `images/track/`, `images/seg_track_20/`, `codalab/`, `legacy/`, `models/` | **Không có gói nhãn MOT**. Chỉ có ảnh `images20-track-*`, `bdd100k_labels.zip` (nút “Labels” của trang chính thức — bộ nhãn ảnh cũ, không phải MOT 2020; chưa tải để mở), `bdd100k_det_20_labels.zip` (Detection 2020), segmentation, drivable, video. Các gói nhãn này đều ngoài phạm vi | `00_*`, `02_*` |
| 5 | Tài liệu `doc.bdd100k.com/download.html` | curl HTTP/HTTPS | Không phân giải được tên máy (curl lỗi 6). Đã đọc nguồn tài liệu trên GitHub thay thế (dòng trên) | `05_trang_chinh_thuc_thu.txt` |
| 6 | `SysCV/bdd100k-models` `doc/PREPARE_DATASET.md` (người dùng nêu) | GET bản raw GitHub | Chỉ hướng về trang chính thức (“MOT 2020 Labels”), không có URL riêng | `03_github_bdd100k_models_PREPARE_DATASET.md` |
| 7 | GitHub issues của `bdd100k/bdd100k` | API tìm issue theo 3 truy vấn; đọc #356 và #369 kèm comment | #356 (mở, 9 comment, cập nhật 2026-08-19) và #369 (mở, 2 comment, 2026-07-15) cùng báo nhãn MOT 2020 không tải được ở cả Berkeley lẫn ETH. **Không có trả lời của maintainer**: mọi comment có `author_association` = NONE | `12_github_issues_tom_tat.json` (đã ẩn email người bình luận) |
| 8 | Tìm web: `"box_track_labels_trainval.zip" bdd100k` và “bdd100k MOT 2020 labels box_track_20 mirror huggingface” | Chỉ đọc kết quả tìm kiếm | Không thấy nguồn công khai đã xác minh cho gói gốc; chỉ có trang chính thức (đang chết) và các gói khác loại. Không mở hay tải mirror không chính thức nào | — |

## Không làm

- **Không đăng nhập** portal: không có tài khoản trong phạm vi này và không xử lý mật khẩu. Trang tải hiện là trang tĩnh,
  nút nhãn MOT đã bị comment và URL của nó trả 404, nên đăng nhập nhiều khả năng không mở thêm nguồn. Đây là **suy luận**,
  chưa kiểm chứng.
- **Không dùng** mirror không chính thức (Hugging Face, Kaggle, Google Drive…) và **không dùng** dữ liệu khác thay thế.

## Người dùng có thể làm để có nhãn

1. Lấy gói gốc từ người hoặc lab đã tải trước đây, hoặc xin maintainer (issue #356/#369, hoặc liên hệ trực tiếp).
2. Đặt file zip vào `data/raw/downloads/bdd100k_mot20/`. Tôi sẽ:
   - kiểm md5 `6be40e0ca56a83ddeba2ed6bff50f9e6` và testzip;
   - chỉ giải nén `bdd100k/labels/box_track_20/train/` vào `data/raw/`;
   - chạy lại `kiem-nguon`.

   md5 khớp nghĩa là file trùng byte với gói tài liệu chính thức mô tả, bất kể lấy từ đâu. Dùng nguồn nào vẫn do người dùng
   quyết định.
