# R48B / M2 — Audit dữ liệu UA-DETRAC + BDD100K MOT (kèm kiểm cấu hình augmentation của M5)

Ngày 2026-10-10. HEAD `eaf2e877081938420508e696de22dbd2aa0bff97` (`eaf2e87`), nhánh `phase2-mixed-traffic`.
Người thực hiện: Claude (tự kiểm Tier 1). Codex chưa nghiệm thu mốc này. Chưa commit.

**Sửa hậu audit 2026-10-11 (review_003, F1–F4)**: chi tiết ở mục 13. F1 là lỗi mã ở phần tổng hợp quỹ đạo: track
`nghi_nhieu` (theo max) từng bị tính vào tập hợp lệ, và chỉ số p90 phân loại độc lập. Lỗi đã sửa; bốn phần `nhan` và
phần `tong_hop` đã chạy lại. Mục 8 và các số liên quan dùng số **sau sửa**. F2–F4 là đính chính diễn giải
(mục 3.2, 6.3, 2, 6.4, 10). Tiêu chí và ngưỡng công bố trước **không đổi**.

Script **chỉ đọc**: `src/kc48/audit_du_lieu.py`. Mọi file ghi đi qua `_duong_ghi()` và chỉ được nằm trong
`results/phase2/kc48_audit/`. Ảnh dataset và overlay chỉ nằm trong `results/phase2/kc48_audit/overlay/` (`.gitignore` `*`).
Bộ nhớ đệm lớn (sha256 từng ảnh, quỹ đạo từng track) nằm trong `results/phase2/kc48_audit/_dem/`, có `.gitignore` `*`
do script tạo. Không file nào trong `data/`, `models/` bị ghi.

Quy ước: **quan sát** = số đếm từ file; **giả thuyết** = cách giải thích chưa kiểm; **đề xuất** = chưa được duyệt,
không thực hiện.

## 0. Tóm tắt

- Ba mốc của slide lần 2 **khớp đúng** khi đếm lại từ 2–3 nguồn độc lập: BDD train-fit 1.261 video / 50.487 ảnh,
  dev nội bộ 139 / 5.559, BDD val 200 / 39.973 (mục 2).
- Tập chỉ có **vài chục video**: UA-DETRAC dev 12 video, test 40, train 60. Lớp hiếm của BDD cũng chỉ nằm trong vài chục
  video ở dev/val: motorcycle có 24 video ở dev, 29 ở val. Sau lọc theo tiêu chí công bố trước, UA-DETRAC dev chỉ còn
  32 track cua / cua gắt và 14 track bus (mục 5, 8). Dev12 không có track `nghi_nhieu`, nên sửa F1 không đổi các số này.
- Quỹ đạo BDD (sau sửa F1, track `nghi_nhieu` bị loại khỏi tập hợp lệ của cả max lẫn p90): 37,5–42,0 % track hợp lệ
  đổi hướng trên ảnh ≥ 30° theo max, 28,2–32,2 % theo p90. Trước sửa, báo cáo ghi khoảng 30 % (mục 8, 13).
- Đã đọc toàn bộ 458.183 ảnh (sha256, header, SOI/EOI): 0 lỗi. Giải mã đầy đủ 6.783 ảnh mẫu: 0 lỗi. Kiểm 320 mẫu chuyển
  nhãn YOLO, phân tầng 8 tầng × 40: 320/320 khớp nhãn gốc. 1.400/1.400 file nhãn BDD train khớp byte và sha256 với
  manifest nhập vòng 34 (mục 4, 6).
- **Vấn đề ưu tiên số 1**: dev nội bộ BDD không độc lập hoàn toàn với train-fit. 70/139 video dev có chung tiền tố tên
  với 173 video train-fit. Một cặp gần trùng xuyên split đã xác nhận bằng mắt là cùng cảnh. Giữa BDD train và BDD val,
  **chưa phát hiện** chồng lấn **trong phạm vi đã kiểm**: 0 tiền tố chung, 0 nhóm sha256 trùng giữa hai video, 0 cặp gần
  trùng trên 4 frame mẫu mỗi video. Phạm vi này không loại trừ chồng lấn ngoài các frame mẫu (mục 6.4, 10).
- Augmentation thực sự chạy khi huấn luyện detect: mosaic 1,0 (tắt 2 epoch cuối), translate 0,1, scale 0,5,
  HSV (0,015 / 0,7 / 0,4), fliplr 0,5. Các tham số mixup, cutmix, copy_paste, degrees, shear, perspective, flipud đều
  bằng 0. Albumentations không chạy vì env không cài gói. `auto_augment` và `erasing` có trong args nhưng chỉ dùng cho
  classify (mục 3.2).

## 1. Lệnh đã chạy (theo thứ tự thời gian, foreground, mỗi lệnh < 10 phút)

Python: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe`. Tiền tố lệnh: `python src/kc48/audit_du_lieu.py`.
Các lệnh 0–15 đều kết thúc bình thường (exit 0). Mỗi lần chạy `audit_du_lieu.py` được ghi thêm một dòng vào
`results/phase2/kc48_audit/nhat_ky_chay.jsonl`. Lệnh 16–26 là lượt sửa hậu audit ngày 2026-10-11 (mục 13). Trong đó
lệnh 18 cố ý chạy kiểm trên đầu ra cũ và trả exit 1, làm bằng chứng lỗi trước khi sửa.

| # | Lệnh | Thời gian | Đầu ra chính |
|:--|:--|--:|:--|
| 0 | `-m py_compile src/kc48/audit_du_lieu.py` | — | biên dịch sạch |
| 1 | `--phan cong_bo` (**trước mọi phép tính**) | <1 s | `tieu_chi_cong_bo.json`, sha256 `d7543fffa6aeeccd8ba864bc189025bc4c99e308bcb0347708e1aa6d7615f3ec` |
| 2 | `--phan nhan --tap detrac_train` | 26,1 s | `_dem/nhan/detrac_train.json`, `_dem/quy_dao/…`, `_dem/canh_xe_lon/…`, `_dem/mau/…` |
| 3 | `--phan nhan --tap detrac_test` | 33,8 s | như trên |
| 4 | `--phan nhan --tap bdd_val` | 13,7 s | như trên |
| 5 | `--phan nhan --tap bdd_train` | 64,8 s | như trên (1.400 video) |
| 6 | `--phan anh --tap detrac --gioi_han_giay 480` | 115 s | 140.131 ảnh, 9,84 GB đọc đầy đủ |
| 7 | `--phan anh --tap bdd_val --gioi_han_giay 480` | 46 s | 39.973 ảnh, 5,00 GB |
| 8 | `--phan anh --tap bdd_train --gioi_han_giay 500` | 315 s | 278.079 ảnh, 33,87 GB (`day_du`: sha256 mọi ảnh) |
| 9 | `--phan nguon` | — | `_dem/nguon.json` |
| 10 | `--phan checkpoint` | — | `checkpoint_huan_luyen.csv`, `augmentation_hieu_luc.csv` |
| 11 | `--phan yolo` | — | `yolo_kiem_mau.csv`, `_dem/yolo.json` |
| 12 | `--phan trung` | — | `gan_trung_ung_vien.csv`, `gan_trung_khoang_cach_gan_nhat.csv`, `_dem/trung.json` |
| 13 | `--phan truc_quan` | — | `mau_truc_quan.csv`, `overlay/mau_box_*.jpg`, `overlay/mau_khung_*.jpg`, `overlay/gan_trung_cap_ung_vien.jpg` |
| 14 | `--phan tong_hop` | — | `phan_bo_lop.csv`, `chat_luong.csv`, `quy_dao_cua.csv`, `moc_doi_chieu.csv`, … , 2 biểu đồ PNG |
| 15 | `--phan tong_hop` (chạy lại sau khi thêm mục 5b, xem mục 12) | — | thêm `tien_to_video_bdd.csv`; các CSV khác sinh lại từ cùng bộ đệm |
| 16 | `python src/kc48/chup_hash_dau_ra.py --ghi truoc_sua_F1` (lời gọi trước, **trước** khi sửa mã) | — | `_dem/hash_truoc_sua_F1.json`, 1.752 file |
| 17 | `-m py_compile src/kc48/kiem_loc_quy_dao.py src/kc48/audit_du_lieu.py src/kc48/chup_hash_dau_ra.py` | — | exit 0 |
| 18 | `python src/kc48/kiem_loc_quy_dao.py` trên đầu ra **cũ** | — | **exit 1**, 10/16 đạt: CSV từng track thiếu `loai_p90_tho`; 4.687 track `nghi_nhieu` theo max mà p90 không bị loại, trong đó 3.405 track train-fit p90 tính là cua / cua gắt |
| 19–22 | `--phan nhan --tap` detrac_train / detrac_test / bdd_val / bdd_train | 25,1 / 29,7 / 10,5 / 55,5 s | exit 0; số track không đổi (5.952 / 2.337 / 14.382 / 86.731) |
| 23 | `--phan tong_hop` | 5,8 s | exit 0; 66 dòng `quy_dao_cua.csv`; không có dòng "bieu do loi" |
| 24 | `python src/kc48/chup_hash_dau_ra.py --ghi sau_sua_F1` | — | exit 0, 1.752 file |
| 25 | `python src/kc48/chup_hash_dau_ra.py --so truoc_sua_F1 sau_sua_F1` | — | exit 0; 1.740 giữ nguyên byte, 4 chỉ đổi khóa thời gian, 8 đổi nội dung (mục 13) |
| 26 | `python src/kc48/kiem_loc_quy_dao.py --ghi` | — | **exit 0, 21/21 đạt**; `kiem_loc_quy_dao.json` |

Các lệnh 9–12 chạy song song vì chỉ đọc và độc lập với nhau. Bước 6–8 tiếp tục theo video được: video đã có file đệm thì
bỏ qua. Lần này mỗi tập xong trong một lượt chạy.

## 2. Đối chiếu mốc slide lần 2 (`results/phase2/kc48_audit/moc_doi_chieu.csv`)

| Mốc (slide lần 2) | Giá trị slide | Đếm lại (nguồn) | Kết luận |
|:--|--:|:--|:--|
| BDD train-fit: số video | 1.261 | `split.train_fit` 1.261; đếm lại từ nhãn 1.261; tên file `data/yolo_bdd5_v1/images/train` 1.261 | khớp |
| BDD train-fit: số ảnh | 50.487 | tái suy quy tắc `buoc_frame` 5 trong khoảng có nhãn: 50.487; `manifest/anh.csv` (split=train) 50.487; file trong `images/train` 50.487 | khớp |
| Dev nội bộ: số video | 139 | `split.dev` 139; từ nhãn 139; từ tên file `images/dev` 139 | khớp |
| Dev nội bộ: số ảnh | 5.559 | tái suy 5.559; manifest 5.559; file 5.559 | khớp |
| BDD val: số video | 200 | `videos` trong COCO-video 200; thư mục `images/track/val` 200 | khớp |
| BDD val: số ảnh | 39.973 | `images` trong COCO-video 39.973; file jpg trên đĩa 39.973 | khớp |

Ghi chú:
- Số ảnh 50.487 / 5.559 là **frame lấy mẫu** (`buoc_frame` 5 ≈ 1 ảnh/giây video, nhãn BDD MOT 5 fps). Số frame
  có nhãn đầy đủ lần lượt là 250.566 và 27.513; toàn bộ 1.400 video có 278.079 frame. Nhiều frame liền nhau trong một
  video **không** phải bối cảnh độc lập. Video là **đơn vị chia split** và đơn vị lấy mẫu bootstrap (1.261 / 139 / 200).
  Tuy vậy, video **không chắc là đơn vị độc lập thống kê**: mục 6.4 đã thấy các video cùng cảnh (cặp train-fit / dev
  cùng tiền tố tên, hai video DETRAC test cùng camera). Vì vậy số video chỉ là cận trên của số bối cảnh độc lập. *(Đính
  chính F3, 2026-10-11.)*
- **Tập chỉ có vài chục video**: UA-DETRAC train 60 video (82.085 frame có `<frame>`), dev 12 video (16.347 frame) và
  test 40 video (56.167 frame có `<frame>`, 56.340 ảnh). "Dev" trong slide là dev nội bộ BDD 139 video của detector,
  không phải dev tracking UA-DETRAC 12 video (`results/phase2/manifests/detrac_dev.json`).

## 3. Nguồn, split, checkpoint và augmentation thực sự bật

### 3.1. Checkpoint (`checkpoint_huan_luyen.csv`, `_dem/checkpoint.json`)

| Checkpoint | Trọng số (sha256 so với kế hoạch khóa R47) | Dữ liệu huấn luyện | Epoch / chọn | Bước optimizer |
|:--|:--|:--|:--|:--|
| `yolov8n_coco` | `models/yolov8n.pt` `f59b3d83…` — **khớp** | COCO, trọng số có sẵn của Ultralytics. Trên máy **không có** log huấn luyện, split hay augmentation (unknown). Dùng làm detector của cache tracking `yolov8n-coco_conf0.05_all` | — | — |
| `yolov8n_ft` | `…/train_bdd5_v1_yolov8n_day_du/weights/best.pt` `5f66776d…` — **khớp** | `data/yolo_bdd5_v1` = BDD MOT train chính thức, split `bdd5_train_v1` (train-fit 1.261 / dev 139 video, 50.487 / 5.559 ảnh, `buoc_frame` 5) | chạy đủ 15/15 epoch, qua 2 lần tiếp tục (3 file log); fitness cao nhất trên dev ở epoch 15 (0,28015) | `ema_updates` cuối 12.988 |
| `yolo11n_ft` | `…/train_bdd5_v1_yolo11n_day_du/weights/best.pt` `d7d1c7e8…` — **khớp** | như trên | 15/15 epoch; fitness cao nhất ở epoch 13 (0,27225) | 12.987 |

Hai model fine-tune có cùng cấu hình: `imgsz` 640, `batch` 8, `nbs` 64 (accumulate 8), 6.311 batch mỗi epoch,
SGD với `lr0` 0,01, `seed` 0, `deterministic`. Chênh 1 ở `ema_updates` (12.988 so với 12.987) xuất hiện khi yolov8n phải
tiếp tục huấn luyện. Hiện chưa rõ cơ chế của khoản chênh này; mức này không đáng kể. Ultralytics 8.4.141, torch
2.6.0+cu124 (`results/phase2/detector_train/*.json`, khóa `moi_truong`).

### 3.2. Augmentation hiệu lực (`augmentation_hieu_luc.csv`; so `configs/phase2/detector/train_bdd5.yaml` với args ghi trong log)

Mọi tham số trong YAML trùng với giá trị hiệu lực trong log ở **cả hai** model. Pipeline detect của ultralytics 8.4.141
nằm ở `site-packages/ultralytics/data/augment.py:2773-2854` (`v8_transforms`):

| Biến đổi | Giá trị | Có chạy khi train detect? |
|:--|:--|:--|
| Mosaic | `mosaic` 1,0; `close_mosaic` 2 | **có**. Cấu hình tắt mosaic ở 2 epoch cuối; log của cả hai model có dòng "Closing dataloader mosaic", xuất hiện ngay sau một lượt val trên dev (`results/phase2/detector_train/giam_sat/yolov8n_tiep_tuc_2/log.txt:13384`; `…/yolo11n_lan0/log.txt:86668`). Chưa đối chiếu số epoch của dòng log này; JSON chỉ ghi `mosaic_thuc_te` ở các mốc tiếp tục (epoch 7, 11: bật) |
| RandomPerspective | `translate` 0,1; `scale` 0,5; `degrees` = `shear` = `perspective` = 0 | **có**: dịch và co giãn; không xoay, không cắt xiên |
| CopyPaste / MixUp / CutMix | 0 / 0 / 0 | không |
| Albumentations (Blur, CLAHE, …) | mặc định ultralytics | **không**: env `khoaluan-mot` không cài `albumentations` (`importlib.util.find_spec` trả `None`), lớp này bỏ qua khi `ImportError` (`augment.py:2140`, `2179-2180`). Hai log huấn luyện cũng không có dòng `albumentations:` |
| RandomHSV | 0,015 / 0,7 / 0,4 | **có** |
| Lật | `fliplr` 0,5; `flipud` 0 | **có** lật ngang |
| `auto_augment` randaugment, `erasing` 0,4 | mặc định, có trong args | **không**: chỉ `classify_augmentations` dùng (`augment.py:2914-3002`), gọi từ `ClassificationDataset` (`dataset.py:1133`, `1203-1204`) |
| `bgr`, `multi_scale`, `rect` | 0 / 0,0 / false | không |

Cách lấy mẫu frame: `buoc_frame` 5, chỉ giữ frame trong khoảng [frame có nhãn đầu, frame có nhãn cuối]
(`src/bdd_yolo_export.py:374-387`). Vùng crowd / distractor được tô xám (114, 114, 114). Có 17.040 ảnh train-fit và
1.560 ảnh dev bị tô xám (`data/yolo_bdd5_v1/manifest/tom_tat.json`). Mẫu YOLO ở mục 6.3 (320 ảnh) tái suy độc lập từ
nhãn gốc và xác nhận phần **siêu dữ liệu** của xử lý tô xám / sao chép: tọa độ vùng trong file ignore, phân loại sao
chép / tô xám, sha256 ảnh xuất so với manifest, kích thước ảnh; ảnh "sao chép" giữ nguyên byte. **Chưa kiểm pixel** của
ảnh tô xám, tức chưa kiểm vùng được tô đúng (114, 114, 114), pixel ngoài vùng được giữ, và đối tượng có nhãn chồng lên
vùng tô vẫn còn nhìn thấy. Do đó chưa thể kết luận việc tô xám đúng ở mức ảnh. *(Đính chính F2, 2026-10-11.)*

## 4. Nguồn nhãn BDD bên thứ ba (`_dem/nguon.json`; hồ sơ vòng 31–34)

| Mục | Trạng thái |
|:--|:--|
| Nguồn nhãn train (phương án C) | snapshot HF `jaffe03195/bdd100k_sot`, revision `694197099692fe1254e93215e73e68ece8b1f41f` (`handoff/REVIEW34_CHI_TIET.md:40`); thư mục cách ly `data/raw/ung_vien_nhan/hf_jaffe03195_bdd100k_sot` vẫn còn |
| Từng file so với manifest nhập vòng 34 (`handoff/logs/review34/03_manifest_nhap_nhan_train.tsv`) | **1.400/1.400 khớp byte và sha256**; 0 thiếu, 0 thừa |
| Hash tổng (`sha256` của các dòng `<tên>\t<sha256>` đã sắp xếp) | `d20244932880e4d4…` — khớp file split khóa và `tom_tat.json` của bản xuất YOLO: nhãn không đổi kể từ khi khóa split |
| COCO-video val | sha256 `074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d`, khớp tiền tố `074ff795` đã nghiệm thu ở R34 |
| Ánh xạ lớp BDD → 5 lớp | car / truck / bus / motorcycle / bicycle không crowd → 0..4. Crowd thuộc 5 lớp, `other vehicle`, `trailer` → vùng tô xám. pedestrian, rider, train, other person → không gán nhãn. Số đếm gốc theo loại × crowd: `nhan_goc_theo_loai.csv` (vd. train-fit: `car|crowd=1` 99.552, `other vehicle` 16.904, `trailer` 3.742) |
| Ánh xạ UA-DETRAC → 5 lớp (D1–D5) | car → car, bus → bus; van → vùng bỏ qua (D1), others → vùng bỏ qua (D2). Nhãn DETRAC **không có** truck / motorcycle / bicycle |
| **Chưa xác thực** | (i) chưa đối chiếu với archive gốc BDD (md5 `6be40e0c…`, không tải được); (ii) không có tham chiếu độc lập ở mức từng annotation cho train; (iii) native val và COCO val vẫn lệch 183 lớp và 50 ngoại lệ ignore ở R34, nguyên nhân UNKNOWN. Audit này chỉ chứng minh nhãn trên máy đúng là bản đã nhập và đã khóa, **không** chứng minh bản đó trùng với nhãn gốc của BDD |

Quan sát liên quan: trong COCO-video val có 17/14.382 track đổi lớp (car↔truck, truck↔bus; `chat_luong.csv`, dòng
`bdd_val,track_doi_lop`). Trong nhãn Scalabel train-fit và dev có 0/78.877 và 0/7.854 track đổi lớp. Có thể đây là cùng
hiện tượng với 183 lệch lớp ở R34, nhưng chưa kiểm.

## 5. Phân bố theo lớp (`phan_bo_lop.csv`, `phan_bo_detrac_loai_goc.csv`, `mat_do_frame.csv`, `dieu_kien_canh.csv`)

Định nghĩa:
- `kt_rel` = √(diện tích box / diện tích ảnh); bảng ghi p50 [p25–p75].
- "nhỏ" = diện tích < 32² px ở độ phân giải gốc (BDD 1280×720, DETRAC 960×540).
- Mật độ = số box của lớp trên mỗi frame, tính trên mọi frame của tập con.
- Che khuất / cắt khung:
  - BDD Scalabel: cờ nhị phân `occluded` / `truncated` (không phải mức che; `src/mot_datasets/bdd100k_mot.py:24`).
  - BDD val COCO: không có trường này, ghi **unknown**.
  - DETRAC: "che" = target có `region_overlap`; "cắt" = `truncation_ratio` > 0.

### 5.1. BDD, frame dùng huấn luyện / chọn checkpoint (`_b5`) và val

| Tập | Lớp | bbox | track | video có lớp | kt_rel p50 [p25–p75] | nhỏ | che | cắt khung | box/frame |
|:--|:--|--:|--:|--:|:--|--:|--:|--:|--:|
| train-fit b5 | car | 383.387 | 67.656 | 1.261 | 0,051 [0,030–0,098] | 30,2 % | 64,4 % | 11,7 % | 7,59 |
| | truck | 27.358 | 3.513 | 906 | 0,062 [0,036–0,121] | 21,8 % | 74,3 % | 14,4 % | 0,54 |
| | bus | 10.708 | 1.133 | 532 | 0,065 [0,036–0,135] | 21,6 % | 74,3 % | 16,2 % | 0,21 |
| | motorcycle | 2.235 | 531 | 230 | 0,055 [0,032–0,097] | 27,7 % | 68,0 % | 10,4 % | 0,044 |
| | bicycle | 4.747 | 1.211 | 397 | 0,048 [0,030–0,078] | 30,8 % | 75,5 % | 10,1 % | 0,094 |
| dev b5 | car | 42.211 | 6.869 | 139 | 0,053 [0,031–0,102] | 28,4 % | 64,7 % | 12,4 % | 7,59 |
| | truck | 2.971 | 334 | 94 | 0,070 [0,038–0,153] | 20,2 % | 72,6 % | 19,4 % | 0,53 |
| | bus | 984 | 100 | 55 | 0,048 [0,028–0,084] | 31,1 % | 83,8 % | 12,3 % | 0,18 |
| | motorcycle | 226 | 41 | **24** | 0,060 [0,034–0,088] | 23,5 % | 60,6 % | 12,0 % | 0,041 |
| | bicycle | 407 | 104 | **44** | 0,044 [0,029–0,078] | 34,6 % | 66,3 % | 9,1 % | 0,073 |
| val | car | 341.436 | 13.189 | 200 | 0,048 [0,028–0,093] | 33,0 % | unknown | unknown | 8,54 |
| | truck | 27.343 | 726 | 160 | 0,058 [0,033–0,114] | 25,0 % | unknown | unknown | 0,68 |
| | bus | 9.087 | 196 | 93 | 0,064 [0,035–0,154] | 23,5 % | unknown | unknown | 0,23 |
| | motorcycle | 902 | 44 | **29** | 0,052 [0,030–0,077] | 29,6 % | unknown | unknown | 0,023 |
| | bicycle | 4.135 | 244 | 63 | 0,046 [0,031–0,067] | 29,8 % | unknown | unknown | 0,10 |

Số box và số video có lớp của train-fit b5 và dev b5 trùng **từng số** với `tom_tat.json` của bản xuất (tính độc lập).
Toàn bộ frame có nhãn (`_all`) cũng trùng với `thong_ke_nguon_theo_split` của file split, ví dụ train-fit car 1.891.481,
motorcycle 11.051. Mật độ toàn khung (5 lớp, `mat_do_frame.csv`): train-fit 8,5 box/frame (p95 18), dev 8,4 (p95 17),
val 9,6 (p95 19). Điều kiện cảnh (thời tiết, ngày/đêm) của BDD là **unknown**: nhãn MOT không có thuộc tính cảnh.

Số track ở bảng tính theo cặp (lớp, ID). Vì vậy 17 track đổi lớp của val được đếm ở cả hai lớp, và số này lệch rất nhẹ
so với `quy_dao_cua.csv` (dùng lớp đa số).

### 5.2. UA-DETRAC (loại gốc; 5 lớp chỉ có car, bus)

| Tập (video / frame có `<frame>`) | Loại | bbox | track | video có loại | kt_rel p50 | nhỏ | che (`region_overlap`) | che ≥ 0,5 | cắt (`trunc` > 0) | box/frame |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| train60 (60 / 82.085) | car | 503.853 | 5.192 | 60 | 0,081 | 14,3 % | 24,3 % | 8,9 % | 10,2 % | 6,14 |
| | bus | 33.651 | 106 | 47 | 0,204 | 0,7 % | 31,3 % | 1,7 % | 48,6 % | 0,41 |
| | van | 57.051 | 611 | 53 | 0,095 | 9,8 % | 27,7 % | 3,6 % | 11,0 % | 0,70 |
| | others | 3.726 | 43 | 27 | 0,132 | 3,6 % | 30,8 % | 4,8 % | 25,5 % | 0,045 |
| **dev12** (12 / 16.347) | car | 96.866 | 1.021 | 12 | 0,100 | 2,2 % | 38,1 % | 13,3 % | 11,0 % | 5,93 |
| | bus | 4.861 | **14** | **8** | 0,193 | 0 % | 46,2 % | 4,0 % | 25,9 % | 0,30 |
| | van | 14.250 | 138 | 11 | 0,137 | 1,4 % | 47,9 % | 6,3 % | 11,2 % | 0,87 |
| | others | 868 | 12 | 6 | 0,135 | 0 % | 40,0 % | 1,8 % | 19,0 % | 0,053 |
| test40 (40 / 56.167) | car | 548.555 | 1.975 | 40 | 0,080 | 7,9 % | 32,0 % | 7,4 % | 11,6 % | 9,77 |
| | bus | 71.785 | 205 | 37 | 0,176 | 0,3 % | 48,4 % | 2,9 % | 31,2 % | 1,28 |
| | van | 38.519 | 126 | 35 | 0,098 | 7,0 % | 26,4 % | 3,7 % | 7,0 % | 0,69 |
| | others | 16.915 | 31 | 21 | 0,190 | 0,02 % | 28,6 % | 1,6 % | 9,2 % | 0,30 |

- Mật độ toàn khung (cả 4 loại): train60 7,3 box/frame (p95 18), dev12 7,1 (p95 21), test40 12,0 (p95 27).
- Độ dài track car p50: train60 2,84 s, dev12 2,44 s, test40 6,44 s.
- Điều kiện cảnh (`sequence_attribute`, số video):

  | Tập | cloudy | night | rainy | sunny | camera `unstable` |
  |:--|--:|--:|--:|--:|--:|
  | train60 | 19 | 16 | 10 | 15 | 38 |
  | dev12 | 2 | 3 | 5 | 2 | 9 |
  | test40 | 11 | 12 | 9 | 8 | 9 |

- Giá trị `occlusion_status` thô (−1 / 0 / 1) đếm trong `nhan_goc_theo_loai.csv`; ý nghĩa từng giá trị chưa đối chiếu
  tài liệu gốc.

So sánh miền (quan sát):
- Box car DETRAC lớn hơn BDD (kt_rel p50 0,080–0,100 so với 0,048–0,053). Ở đầu vào 640, chiều cao p50 là 29–37 px so với
  20,6–22,8 px (`h640px_p50`).
- Bus DETRAC lớn hơn hẳn: p50 0,18–0,20 so với 0,05–0,07.
- BDD có tỉ lệ box nhỏ cao hơn nhiều: car 30 % so với 2–14 %.

## 6. Chất lượng (`chat_luong.csv`, `yolo_kiem_mau.csv`, `gan_trung_*.csv`, `tien_to_video_bdd.csv`)

### 6.1. Ảnh

| Kiểm | BDD train-fit | BDD dev | BDD val | DETRAC dev12 | DETRAC train khác | DETRAC test40 |
|:--|--:|--:|--:|--:|--:|--:|
| Ảnh đọc đầy đủ + sha256 | 250.566 | 27.513 | 39.973 | 16.919 | 66.872 | 56.340 |
| Lỗi đọc / header | 0 | 0 | 0 | 0 | 0 | 0 |
| Thiếu SOI / thiếu EOI (ảnh cắt cụt) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Giải mã đầy đủ trên mẫu (4 frame/video) | 5.029, lỗi 0 | 555, lỗi 0 | 799, lỗi 0 | 48, lỗi 0 | 192, lỗi 0 | 160, lỗi 0 |
| Kích thước thực | 1280×720 tất cả | 1280×720 | 1280×720 | 960×540 | 960×540 | 960×540 |
| File lạ không phải jpg | 0 | 0 | 0 | 0 | 0 | 0 |

### 6.2. Nhãn ↔ ảnh, bbox, track

| Kiểm | Kết quả |
|:--|:--|
| Frame có nhãn nhưng thiếu ảnh | 0 ở mọi tập |
| Ảnh không có frame nhãn | BDD: 0. DETRAC: train60 1.706 (2,0 %), trong đó **dev12 572** (= 16.919 ảnh − 16.347 frame), test40 173 (= D4 của R47) |
| Box suy biến (w ≤ 0 hoặc h ≤ 0), cạnh < 1 px, không hữu hạn | 0 ở mọi tập |
| Box vượt khung > 1 px | 0 ở mọi tập |
| Box vượt khung ≤ 1 px | BDD train-fit 78/2.602.992; dev 1; val 19.733/465.512 (4,2 %); DETRAC train60 49.228/598.281 (8,2 %), test40 80.953/675.774 (12,0 %). Ví dụ: `MVI_20011` target 1 có `top` + `height` = 541,0 > 540. Exporter và evaluator đều cắt box vào khung |
| ID trùng trong cùng frame | 0 ở mọi tập |
| Track đổi lớp | BDD train-fit 0, dev 0, val 17 (mục 4); DETRAC 0 |
| Nhãn DETRAC nằm trong `ignored_region` (IoA ≥ 0,5) | train60 20.767 (3,5 %), test40 44.081 (6,5 %) — xem vấn đề 4 |
| `region_overlap` vượt box > 1 px | train60 15.044/174.220 (8,6 %), test40 2.815/290.682 (1,0 %). Ý nghĩa chưa rõ; phép tính tỉ lệ che ở mục 5 đã giao với box nên không bị ảnh hưởng |

### 6.3. Bản chuyển nhãn YOLO `data/yolo_bdd5_v1` (`yolo_kiem_mau.csv`, `_dem/yolo.json`)

Lấy mẫu theo tiêu chí công bố trước (`tieu_chi_cong_bo.json`, khóa `mau_yolo`, seed 48): 8 tầng = split (train/dev) ×
xử lý (sao chép / tô xám) × có lớp hiếm. Mỗi tầng 40 ảnh, tổng **320 ảnh**. Kết quả kỳ vọng được **tái suy độc lập** từ
JSON Scalabel gốc theo quy tắc trong docstring `src/bdd_yolo_export.py`, không gọi mã exporter.

| Kiểm | Đạt |
|:--|--:|
| Lớp từng box; tọa độ chuẩn hóa (dung sai 1e-6); cột `n_<lớp>` của manifest | 320/320; 320/320; 320/320 |
| Vùng tô xám trong file ignore (dung sai 0,006 px); phân loại sao chép / tô xám | 320/320; 320/320 |
| sha256 file nhãn và ảnh xuất so với manifest; W×H | 320/320; 320/320; 320/320 |
| Ảnh "sao chép" giữ nguyên byte so với ảnh gốc | 160/160 |
| Ảnh "tô xám" cùng kích thước với ảnh gốc | 160/160 |
| Số file ảnh / file nhãn: `images/train`, `labels/train`, `images/dev`, `labels/dev` | 50.487 / 50.487 / 5.559 / 5.559 (= manifest) |

Phạm vi: đây là mẫu, chưa phải toàn bộ 56.046 ảnh. Toàn bộ tập vẫn được kiểm gián tiếp qua bộ đếm theo lớp ở mục 5.1:
số box từng lớp tính độc lập trùng từng số với manifest.

Phạm vi với ảnh tô xám (đính chính F2): kiểm `anh_to_xam_cung_kich_thuoc` chỉ so W×H. Kiểm `sha_anh_ra_khop_manifest`
chỉ cho biết file trên đĩa là file exporter đã ghi, không cho biết nội dung đúng
(`src/kc48/audit_du_lieu.py`, hàm `phan_yolo`, các khóa `vung_khop`, `xu_ly_khop`, `sha_anh_ra_khop_manifest`,
`anh_to_xam_cung_kich_thuoc`). Không có phép so pixel nào giữa ảnh tô xám và ảnh gốc. Ảnh tô xám được mã hóa JPEG lại,
nên nếu kiểm sau này thì cần dung sai.

### 6.4. Trùng và gần trùng xuyên split (`_dem/trung.json`, `gan_trung_*.csv`, `overlay/gan_trung_cap_ung_vien.jpg`)

- **Trùng chính xác (sha256)** trên **toàn bộ 458.183 ảnh** của BDD train (1.400 video), BDD val và DETRAC (100 video):
  3 nhóm, **mỗi nhóm nằm trong một video**, đều là frame liền nhau giống hệt byte (đứng hình). **Hai** nhóm thuộc BDD
  train-fit: `01460ec4-a12a1552` frame 137–138 và `02269962-0e3f5114` frame 103–106. **Một** nhóm thuộc BDD val:
  `b1d7b3ac-5744370e` frame 79–80 (`_dem/trung.json`; `chat_luong.csv`, dòng `trung_chinh_xac:trong_cung_video`).
  **0** nhóm trùng giữa hai video khác nhau hay giữa hai split. *(Đính chính F4, 2026-10-11: bản trước ghi cả 3 nhóm
  thuộc train-fit. Nguyên nhân là cột `vi_du` của CSV chỉ in 2 nhóm đầu (`[:2]`); nay in tối đa 5 nhóm, xem mục 13.)*
- **Gần trùng (dHash 256 bit)**: mỗi video lấy 4 frame (công bố trước), tổng 6.383 frame BDD và 400 frame DETRAC. Chỉ
  so cặp frame thuộc **hai video khác nhau** trong cùng miền. Khoảng cách gần nhất sang video khác có trung vị
  67–81 bit. Có **3 cặp** ≤ 20 bit, **cả 3 đều xác nhận bằng mắt là cùng cảnh** (`overlay/gan_trung_cap_ung_vien.jpg`):
  1. BDD dev `02ddd956-37234391` f21 và train-fit `02ddd956-5440cc89` f183: cùng xe trắng phía trước, cùng ngã tư và
     dãy nhà. Đây là **cặp xuyên split train-fit / dev**.
  2. DETRAC dev12 `MVI_20062` f721 và `MVI_20063` f29: hai sequence liền nhau của cùng camera. `MVI_20063` thuộc nhóm
     train không dùng cho tracking, nên không ảnh hưởng đánh giá.
  3. DETRAC test40 `MVI_40773` f99 và `MVI_40775` f878: cùng camera, cùng cảnh đêm; cả hai trong test, nên 40 video test
     không phải 40 bối cảnh độc lập.
- **Tiền tố tên video BDD** (đoạn trước dấu `-`; phân tích **thêm sau** khi thấy cặp 1, không có trong tiêu chí công bố
  trước, `tien_to_video_bdd.csv`):
  - 63 tiền tố chung giữa train-fit và dev, liên quan 173 video train-fit và **70/139 video dev**.
  - **0** tiền tố chung giữa BDD train (train-fit, dev) và BDD val.
  - Giả thuyết (chưa xác minh từ tài liệu BDD100K): video cùng tiền tố thuộc cùng một lần ghi hình hoặc cùng thiết bị.
    Cặp 1 ủng hộ giả thuyết này nhưng chỉ là một trường hợp.
- Phạm vi gần trùng: chỉ 4 frame/video, ngưỡng 20 bit. Kiểm này **không** loại trừ hai video có đoạn trùng ngoài các
  frame mẫu.
- Kết luận đúng phạm vi (đính chính F3): giữa BDD train (train-fit, dev) và BDD val, cũng như giữa DETRAC train và DETRAC
  test, **chưa phát hiện** chồng lấn trong phạm vi đã kiểm. Phạm vi gồm sha256 toàn bộ ảnh, dHash trên 4 frame mẫu mỗi
  video và tiền tố tên video BDD. Như vậy **không** chứng minh được các tập này độc lập; chẳng hạn cùng địa điểm nhưng
  khác tiền tố, hoặc đoạn trùng ngoài frame mẫu, vẫn chưa bị loại trừ. Ngay trong một tập cũng đã thấy video cùng cảnh
  (cặp 2, 3), nên video là đơn vị chia split chứ không chắc là đơn vị độc lập thống kê.

## 7. Kiểm trực quan mẫu phân tầng (`mau_truc_quan.csv`, `overlay/mau_box_*.jpg`, `overlay/mau_khung_*.jpg`)

**Quy tắc chọn mẫu** (viết vào `tieu_chi_cong_bo.json`, khóa `mau_truc_quan`, **trước** khi tính):
- Bốn tập con: `bdd_trainfit_b5`, `bdd_val`, `detrac_train60`, `detrac_test40`.
- Ô mẫu = lớp × kích thước × điều kiện:
  - Lớp: BDD 5 lớp; DETRAC car, bus, van, others.
  - Kích thước: nhỏ / vừa / lớn theo diện tích px (quy ước COCO).
  - Điều kiện "khó": BDD train = occluded hoặc truncated; DETRAC = có `region_overlap`, hoặc `trunc` > 0, hoặc đêm /
    mưa; BDD val = unknown.
- Mỗi ô lấy 1 box bằng reservoir sampling đều, seed 48. Thêm 2 frame toàn khung cho mỗi tập.

**Số lượng**: 91 box trong 91 ô có dữ liệu, cộng 8 frame toàn khung, tổng 99 mục.
- `bdd_trainfit_b5`: 30/30 ô.
- `bdd_val`: 15/15 ô.
- `detrac_train60`: 23/24 ô; ô bus | nhỏ | thường không có box.
- `detrac_test40`: 23/24 ô; ô others | nhỏ | thường không có box.

Claude xem các ô ở kích thước khoảng 240×180 px.

**Loại lỗi thấy được** (số `#` theo `mau_truc_quan.csv`; đánh giá bằng mắt trên ảnh thu nhỏ, chưa xem ở độ phân giải gốc):

| Loại | Mục | Ghi chú |
|:--|:--|:--|
| Không kiểm được vì quá tối, mờ chuyển động hoặc quá nhỏ | BDD train #3, #5, #9, #27; BDD val #44, #46, #47; DETRAC #69 | không kết luận đúng hay sai |
| Nghi box không chứa đối tượng nhìn thấy được, hoặc box lệch sang vật khác | BDD train #16 (car nhỏ: vùng trời + cột điện), #23 (motorcycle: vùng chủ yếu là ô tô đỏ); DETRAC #63 (others nhỏ ở mép vật thể khác) | cần xem ảnh gốc |
| Ranh giới lớp mơ hồ | BDD #26 (bán tải → truck), #30 (xe dạng van → truck), trong khi frame toàn khung `02c26e9d-02585e70` f35 có xe van trắng → car; DETRAC #90 (xe trắng loại cảnh sát → others), #67 (xe tối màu giống sedan → van) | ranh giới car / truck / van chưa đối chiếu hướng dẫn gán nhãn |
| Xe bị cắt mạnh ở biên ảnh | DETRAC #52 (bus còn một dải hẹp), #81 | phù hợp với `truncation_ratio` |
| Box nằm một phần trong `ignored_region` | frame toàn khung DETRAC test `MVI_40892` (xe xám góc phải dưới) | khớp với số đếm ở mục 6.2 |
| `others` của DETRAC chứa xe tải | #61, #62, #64–#66, #87, #88, #91 (xe tải thùng, xe ben, xe cứu hộ) | thành phần lớp `others` mới quan sát trên mẫu |

Không thấy box lệch hình học rõ ràng trên 91 ô. Đây là **mẫu nhỏ**: chỉ dùng để phát hiện **loại** lỗi, **không** kết
luận toàn tập sạch hay ước lượng tỉ lệ lỗi.

## 8. Quỹ đạo đổi hướng (`quy_dao_cua.csv`, `bieu_do_quy_dao_cua.png`; tiêu chí `tieu_chi_cong_bo.json` khóa `quy_dao`)

**Tiêu chí công bố trước** (sha256 file `d7543fff…`, ghi trước lệnh `nhan` đầu tiên):
- Hướng ψ(t) tính từ tâm box giữa t−0,4 s và t+0,4 s, chỉ khi độ dịch ≥ max(8 px, 0,25·h).
- Δψ = |wrap(ψ(t + 1 s) − ψ(t))|.
- Loại `loai_ngan` nếu track < 2 s. Loại `loai_cham` nếu tỉ lệ hướng hợp lệ < 0,5 hoặc không có Δψ.
- `nghi_nhieu` nếu max Δψ > 150°.
- Ngưỡng: thẳng < 15° ≤ cong nhẹ < 30° ≤ **cua** < 60° ≤ **cua gắt**.
- Chỉ số chính: max Δψ; độ nhạy: p90 Δψ.
- Với BDD, kết quả là "**đổi hướng trên ảnh**": camera gắn trên xe đang chạy (ego-motion), chưa hiệu chỉnh nên không
  phải "xe cua".

**Tập hợp lệ (sau sửa F1)**: bỏ `loai_ngan`, `loai_cham` và `nghi_nhieu`. `nghi_nhieu` xét theo **max** Δψ, đúng
câu chữ của tiêu chí. Tập hợp lệ này **dùng chung** cho chỉ số chính (max) và độ nhạy (p90), nên trong mọi dòng của
`quy_dao_cua.csv`: `n_hop_le` = thẳng + cong nhẹ + cua + cua gắt = `n_track` − ngắn − chậm − nghi nhiễu. Bất biến này được
kiểm bằng `src/kc48/kiem_loc_quy_dao.py` (mục 13). Cột `n_truoc_loc_nhieu` và `truoc_loc_*` giữ số **trước** khi loại
nghi nhiễu, tức số bản trước đã báo cáo.

| Tập con | track | qua lọc ngắn / chậm | nghi nhiễu (theo max) | **hợp lệ** | thẳng | cong nhẹ | cua | cua gắt | (cua + gắt) / hợp lệ — max | — p90 | trước lọc nhiễu — max / p90 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DETRAC train60 | 5.952 | 4.406 | 4 (0,1 %) | 4.402 | 3.606 | 523 | 207 | 66 | 6,2 % | 2,2 % | 6,2 % / 2,3 % |
| **DETRAC dev12** | 1.185 | 715 | 0 | **715** | 617 | 66 | **23** | **9** | **4,5 %** | **1,1 %** | 4,5 % / 1,1 % |
| DETRAC test40 | 2.337 | 1.787 | 15 (0,8 %) | 1.772 | 1.183 | 426 | 137 | 26 | 9,2 % | 3,1 % | 9,1 % / 3,1 % |
| BDD train-fit (trên ảnh) | 78.877 | 41.329 | 8.162 (19,7 %) | 33.167 | 12.447 | 8.283 | 6.293 | 6.144 | 37,5 % | 28,2 % | 30,1 % / 30,9 % |
| BDD dev (trên ảnh) | 7.854 | 4.260 | 830 (19,5 %) | 3.430 | 1.284 | 853 | 639 | 654 | 37,7 % | 28,3 % | 30,4 % / 30,1 % |
| BDD val (trên ảnh) | 14.382 | 6.905 | 1.486 (21,5 %) | 5.419 | 1.823 | 1.320 | 1.162 | 1.114 | 42,0 % | 32,2 % | 33,0 % / 34,6 % |

Thẳng / cong nhẹ / cua / cua gắt là phân loại theo max. Phân loại theo p90 trên cùng tập hợp lệ nằm ở các dòng
`chi_so = loai_p90` của `quy_dao_cua.csv`. Ví dụ train-fit theo p90: 15.246 / 8.567 / 5.695 / 3.659. Phần trăm trong
ngoặc ở cột nghi nhiễu lấy mẫu số là số track qua lọc ngắn / chậm (`ti_le_nghi_nhieu_tren_truoc_loc`).

Quan sát:
- DETRAC dev12 chỉ có **32 track** cua / cua gắt theo max, và chỉ **8** theo p90. Bus trong dev12 có 0 track cua. Dev12
  không có track `nghi_nhieu`, nên các số này giống bản trước.
- Tỉ lệ cua ở test40 cao gấp khoảng 2 lần dev12, nên dev12 có ít track cua hơn test40.
- Với BDD, 19,5–21,5 % track qua lọc ngắn / chậm bị gắn `nghi_nhieu` (đảo chiều > 150°; train-fit 8.162/41.329, dev
  830/4.260, val 1.486/6.905). Sau khi loại nhóm này, 37,5–42,0 % track hợp lệ đổi hướng trên ảnh ≥ 30° theo max và
  28,2–32,2 % theo p90. Giả thuyết: phần lớn do ego-motion (xe quay đầu, rẽ) và chuyển động tương đối khi dừng / chạy.
  Vì vậy **không** nên dùng số này làm "xe cua" trong câu hỏi PF.
- Tác động của sửa F1 tập trung ở BDD: mẫu số giảm 19,5–21,5 %, và tỉ lệ theo max tăng 7,3–9,0 điểm phần trăm. Ở p90, việc lọc
  bỏ những track `nghi_nhieu` từng bị tính là cua / cua gắt (train-fit 3.405, dev 310, val 645; cột
  `nghi_nhieu_bi_tinh_cua_hoac_gat_truoc_loc`), nên tỉ lệ p90 giảm 1,8–2,7 điểm. Ở DETRAC, thay đổi ≤ 15 track mỗi tập.
- Số `pct_curved` cũ trong `results/phase2/manifests/detrac_dev.json` dùng tiêu chí khác, lỏng hơn và không được công bố
  ở đây, nên **không so sánh trực tiếp** với bảng trên.

## 9. Nhóm xe nhỏ cạnh / bị che bởi xe lớn — dữ liệu cho R48C (`canh_xe_lon_tom_tat.csv`, `canh_xe_lon_video_detrac.csv`)

**Tiêu chí công bố trước** (khóa `canh_xe_lon`):
- Xe nhỏ: BDD car / motorcycle / bicycle; DETRAC car.
- Xe lớn: BDD bus / truck; DETRAC bus / others.
- Điều kiện: diện tích box lớn ≥ 2 × box nhỏ **và** IoA(box nhỏ) ≥ 0,10, trong cùng frame.
- Sự kiện = chuỗi frame liên tiếp của cùng cặp track.
- DETRAC tính thêm một chỉ số riêng "theo nhãn che": `region_overlap` có `occlusion_id` trỏ tới target lớn.

| Tập con | video | cặp-frame | sự kiện | track nhỏ liên quan | video có sự kiện | Theo nhãn che DETRAC: sự kiện / track / video |
|:--|--:|--:|--:|--:|--:|:--|
| DETRAC dev12 | 12 | 11.194 | **88** | **65** | 9 | 26 / 22 / 8 |
| DETRAC train60 | 60 | 25.531 | 480 | 335 | 45 | 91 / 72 / 30 |
| DETRAC test40 | 40 | 110.850 | 1.263 | 622 | 38 | 342 / 221 / 31 |
| BDD dev | 139 | 6.446 | 1.285 | 773 | 94 | unknown |
| BDD train-fit | 1.261 | 56.701 | 12.783 | 7.731 | 896 | unknown |
| BDD val | 200 | 13.066 | 3.017 | 1.761 | 158 | unknown |

Số theo từng video DETRAC (thời tiết, số track, số sự kiện) nằm ở `canh_xe_lon_video_detrac.csv`, dùng để R48C chọn
nhóm theo tiêu chí dữ liệu và ghi các trường hợp bị loại. Ở dev12, cỡ mẫu là 88 sự kiện hình học / 65 track nhỏ /
9 video. Tiêu chí hình học không phân biệt "cạnh" với "che": IoA ≥ 0,1 bao gồm cả hai trường hợp.

## 10. Danh sách vấn đề ưu tiên

| Ưu tiên | Vấn đề (quan sát) | Ảnh hưởng | Cách xử lý trong phạm vi / đề xuất (**chưa duyệt**) |
|:--|:--|:--|:--|
| 1 | Dev nội bộ BDD dùng chung tiền tố tên video với train-fit (70/139 video dev); 1 cặp gần trùng xuyên split đã xác nhận (mục 6.4) | Giả thuyết: chỉ số dev dùng chọn epoch / dừng sớm có thể lạc quan. Với BDD val và UA-DETRAC test: **chưa phát hiện** chồng lấn với tập huấn luyện **trong phạm vi đã kiểm** (0 tiền tố chung BDD train / val; 0 nhóm sha256 trùng giữa hai video; 0 cặp gần trùng xuyên split trên 4 frame mẫu mỗi video). Phạm vi này chưa đủ để kết luận "không ảnh hưởng" *(đính chính F3)* | Khi diễn giải, ghi rằng dev không độc lập theo lần ghi hình. Đề xuất: split phiên bản mới nhóm theo tiền tố; việc này cần huấn luyện lại nên ngoài lượt này và phải hỏi người dùng |
| 2 | UA-DETRAC dev12 rất nhỏ cho câu hỏi chính: 12 video; 32 track cua (8 theo p90); 14 track bus / 8 video; 88 sự kiện xe nhỏ cạnh xe lớn trong 9 video (mục 5.2, 8, 9) | Khoảng tin cậy rộng; dễ kết luận nhầm từ vài video | R48C: dùng mô phỏng để kiểm cơ chế, bootstrap theo video, báo cỡ mẫu từng nhóm; không suy rộng từ dev12 |
| 3 | 572 ảnh dev12, 1.706 ảnh train60 và 173 ảnh test40 không có `<frame>` (mục 6.2) | Chưa biết evaluator tracking xử lý frame này thế nào. Nếu coi là "không có GT" thì mọi track ở đó tính là FP | R48C kiểm và ghi cách xử lý trong giao thức. Không sửa evaluator (khóa) |
| 4 | Nhãn DETRAC nằm ≥ 50 % trong `ignored_region`: 3,5 % (train60), 6,5 % (test40) | Theo D5, GT trong vùng bỏ qua vẫn được chấm, nên detection bỏ sót ở đó tính là FN | Chỉ ghi nhận khi diễn giải; không đổi D1–D5 |
| 5 | Đổi hướng trên ảnh BDD bị ego-motion chi phối: 19,5–21,5 % track qua lọc ngắn / chậm bị gắn nghi nhiễu; sau khi loại nhóm này, 37,5–42,0 % track hợp lệ ≥ 30° theo max, 28,2–32,2 % theo p90 (mục 8; số sau sửa F1) | Không dùng được làm nhóm "cua" của BDD | Gọi đúng là "đổi hướng trên ảnh". Đề xuất: hiệu chỉnh chuyển động camera (ngoài phạm vi) |
| 6 | DETRAC không có truck / motorcycle / bicycle; `others` chứa xe tải (quan sát trên mẫu) | So sánh theo lớp giữa hai miền chỉ làm được ở car, bus | Giữ D3 (chỉ car, bus) |
| 7 | 17 track đổi lớp trong COCO-video val; nhãn train chưa xác thực với archive gốc (mục 4) | Nhiễu nhãn val chưa đo được; nguồn train phụ thuộc snapshot bên thứ ba | Ghi phạm vi chưa xác thực trong mọi báo cáo dùng BDD |
| 8 | Ranh giới car / truck / van mơ hồ trên mẫu (BDD #26, #30 và frame toàn khung; DETRAC #67, #90) | Có thể góp phần vào nhầm lớp bus / truck / car (điểm chưa rõ R47 về recall bus); **chưa kiểm** | Đề xuất: đo trực tiếp ma trận nhầm lớp (ngoài R48B) |
| 9 | Box vượt khung ≤ 1 px phổ biến ở DETRAC (8–12 %) và BDD val (4 %) | Không đáng kể: exporter và evaluator đã cắt box vào khung | Không cần xử lý |
| 10 | Frame đứng hình trong cùng video: 3 nhóm sha256 trùng, gồm 2 nhóm ở BDD train-fit và 1 nhóm ở BDD val (mục 6.4) | Không đáng kể | Không cần xử lý |

## 11. Giới hạn

- Kiểm trực quan: 91 box + 8 frame, xem ở độ phân giải thấp, do một người xem (Claude). Không ước lượng được tỉ lệ lỗi.
- Gần trùng: 4 frame/video, dHash 256 bit, ngưỡng 20 bit. Không bắt được hai video trùng một đoạn nằm ngoài frame mẫu,
  và chưa so BDD với DETRAC (khác miền).
- Ý nghĩa của tiền tố tên video BDD, `occlusion_status` và `region_overlap` vượt box của DETRAC chưa đối chiếu tài liệu
  gốc.
- Tỉ lệ che DETRAC = tổng diện tích `region_overlap` giao box / diện tích box, chặn ở 1. Các vùng chồng nhau có thể bị
  tính hai lần.
- Không có log huấn luyện COCO của `yolov8n_coco`, nên nguồn, augmentation và split của checkpoint này là unknown.
- Tiêu chí quỹ đạo dùng tâm box trên ảnh, nên phụ thuộc phối cảnh (DETRAC) và ego-motion (BDD).
- Bản chuyển YOLO: ảnh tô xám chỉ được kiểm siêu dữ liệu (tọa độ vùng, sha256 so với manifest, kích thước), **chưa kiểm
  pixel** (mục 3.2, 6.3).
- Tính độc lập: video là đơn vị chia split, không chắc là đơn vị độc lập thống kê. Các kiểm chồng lấn chỉ cho phép kết
  luận "chưa phát hiện trong phạm vi đã kiểm" (mục 2, 6.4).

## 12. Lệch kế hoạch và ghi chú

- Phân tích tiền tố tên video BDD (`tien_to_video_bdd.csv`) được **thêm sau** khi thấy cặp gần trùng 1. Đây không phải
  tiêu chí công bố trước. Mã thêm vào chỉ nằm trong phần `tong_hop`; các phần `nhan`, `anh`, `trung` không đổi sau khi
  chạy. Lệnh `tong_hop` chạy lại lần 2 sinh lại cùng các CSV từ cùng bộ đệm, cộng thêm file tiền tố.
- `anh --tap bdd_train` chạy chế độ `day_du` (sha256 mọi ảnh) thay vì `hon_hop`, vì thông lượng đo được khoảng
  107 MB/s đủ cho một lượt 315 s.
- Không sửa mã tracker, evaluator, dữ liệu gốc, split hay quyết định D1–D5. Không train, không chạy GPU, không truy cập
  mạng.
- **Ghi ngoài `write_paths` (F5, `review_005`).** Lời gọi sửa hậu audit `0010_coder` đã tạo
  `C:\Users\Mikuno\.claude\projects\D--LABOTARY-bytetrack\memory\ket-luan-tu-nguon-day-du.md` và sửa
  `C:\Users\Mikuno\.claude\projects\D--LABOTARY-bytetrack\memory\MEMORY.md`. Đây là memory của Claude Code, nằm ngoài
  repo. Hai thao tác này không đổi file nào trong repo. Thao tác, bằng chứng và giới hạn xác minh: mục 13.3.

## 13. Sửa lỗi hậu audit (review_003, review_005; 2026-10-11)

Nguồn: review Codex `review_003` (kết luận *revise*), phát hiện F1–F4. Lượt sửa chỉ nằm trong R48B; ngoại lệ là hai file
memory ghi ngoài `write_paths` (F5, mục 13.3). **Không đổi** tiêu chí công bố trước (`tieu_chi_cong_bo.json`, sha256
`d7543fff…` giữ nguyên, kiểm C0), ngưỡng, quy tắc mẫu, dữ liệu trong `data/` hay mã tracker. Các phần `anh`, `trung`,
`yolo`, `nguon`, `checkpoint`, `truc_quan` không chạy lại.
Review vòng 2 `review_005` (kết luận *revise*) ghi F1–F4 đã đóng và nêu thêm F5 (mục 13.3).

### 13.1. F1 — lỗi mã: track `nghi_nhieu` vẫn nằm trong tập hợp lệ

**Lỗi ở bản trước** gồm hai chỗ:
- Phần tổng hợp tính `n_hop_le = n_track − loai_ngan − loai_cham`, nên track `nghi_nhieu` vẫn nằm trong mẫu số.
- Hàm `quy_dao()` áp ngưỡng `nghi_nhieu` lên p90 **một cách độc lập**. Track có max > 150° nhưng p90 ≤ 150° vì thế quay
  lại nhóm thẳng / cong / cua của chỉ số p90. Ví dụ `MVI_20064` track 34: max 179,18°, p90 41,43°; theo max là
  `nghi_nhieu`, theo p90 là `cua`.

Cả hai trái với tiêu chí "max Δψ > 150° → nghi_nhieu" và với yêu cầu M2 loại quỹ đạo nhiễu.

**Sửa** trong `src/kc48/audit_du_lieu.py`:
- `phan_loai_track()`: `nghi_nhieu` xét trên max và áp cho cả hai chỉ số. Cột `loai_p90_tho` giữ phân loại p90 chưa
  lọc, chỉ dùng để báo cáo trước / sau.
- `tong_hop_quy_dao()`:
  - `n_hop_le = n_truoc_loc_nhieu − nghi_nhieu`.
  - Kiểm bất biến `n_hop_le` = tổng 4 nhóm; hỏng thì dừng với lỗi.
  - Thêm cột `n_truoc_loc_nhieu`, `truoc_loc_*`, `ti_le_*_truoc_loc`, `nghi_nhieu_bi_tinh_cua_hoac_gat_truoc_loc`.
- CSV từng track `_dem/quy_dao/*.csv` thêm cột `loai_p90_tho`. Biểu đồ vẽ tỉ lệ trên tập hợp lệ sau lọc và ghi số track
  bị loại.

Vì đầu ra từng track thay đổi, đã chạy lại bốn phần `nhan` rồi `tong_hop` (lệnh 19–23).

**Kiểm** bằng `src/kc48/kiem_loc_quy_dao.py` (lệnh 18, 26). Script chỉ đọc; khi có `--ghi` thì ghi
`kiem_loc_quy_dao.json`.
- A, hàm thuần:
  - `phan_loai_track(179, 41)` cho `nghi_nhieu` / `nghi_nhieu`, p90 thô là `cua`.
  - Max = 150 nằm đúng biên (ngưỡng là `>`), nên không bị loại.
  - `tong_hop_quy_dao` trên 7 track tổng hợp cho đúng số đếm trước / sau lọc, và bất biến đúng.
- B, quỹ đạo tổng hợp qua hàm `quy_dao()` thật, 25 fps:
  - Cung tròn đều 41°/s: max = p90 = 41,0°, cả hai chỉ số là `cua`, track không bị loại.
  - Cùng cung đó thêm một đoạn quay đầu: max 178,36°, p90 41,0°. Track bị loại ở cả hai chỉ số; p90 thô là `cua`.
- C, đầu ra thật:
  - CSV từng track đủ cột; 0 track vi phạm luật lọc.
  - Tái tính nhãn từ max / p90 theo ngưỡng công bố trên 109.402 track: 0 sai. Trong số đó, 39 track không tái tính vì
    giá trị đã làm tròn nằm trong ±0,005° quanh một ngưỡng.
  - `quy_dao_cua.csv` trùng đúng với tổng hợp lại từ CSV từng track: 66 dòng, 0 ô lệch.
  - Bất biến đúng ở mọi dòng; max và p90 dùng cùng tập hợp lệ; ví dụ của review đúng.
  - Số train-fit khớp số review_003 tự tính: 41.329 / 8.162 / 33.167 / 12.437 / 3.405.
- Kết quả:
  - Trước sửa (lệnh 18, chạy trên đầu ra cũ): exit 1, 10/16 đạt. C1 hỏng vì thiếu cột. C2 hỏng: 4.687 track vi phạm,
    trong đó 3.405 track train-fit bị p90 tính là cua / cua gắt.
  - Sau sửa (lệnh 26): exit 0, 21/21 đạt, ghi `results/phase2/kc48_audit/kiem_loc_quy_dao.json`.

**So hash đầu ra trước / sau** bằng `src/kc48/chup_hash_dau_ra.py` (lệnh 16, 24, 25; `_dem/hash_truoc_sua_F1.json`,
`_dem/hash_sau_sua_F1.json`). Bản chụp "trước" đúng là trạng thái Codex đã review: sha256 `chat_luong.csv` trong đó
(`790f9b21…`) trùng hash ghi lúc review
(`D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\review_points\0008_review\hashes.json`). Tổng
1.752 file:
- 1.740 file giữ nguyên byte, gồm overlay, `_dem/mau/*.json`, `_dem/canh_xe_lon/*.csv` và các CSV phân bố, mật độ,
  điều kiện cảnh. Như vậy chạy lại `nhan` tái tạo đúng các đầu ra không liên quan.
- 4 file `_dem/nhan/*.json` chỉ đổi khóa thời gian (`tao_luc`, `giay`).
- 8 file đổi nội dung:
  - `_dem/quy_dao/{bdd_train,bdd_val,detrac_train,detrac_test}.csv`, `quy_dao_cua.csv`, `bieu_do_quy_dao_cua.png`:
    đúng như dự kiến của F1.
  - `nhat_ky_chay.jsonl`: thêm 5 dòng chạy.
  - `chat_luong.csv`: nặng hơn 124 byte, đúng bằng độ dài chuỗi của nhóm trùng thứ ba (BDD val `b1d7b3ac-5744370e`
    f79–80) trong ô `vi_du` của dòng `trung_chinh_xac:trong_cung_video`. Lời gọi sửa trước đã đổi mã tổng hợp từ `[:2]`
    thành `[:5]` để ô này in đủ 3 nhóm, phục vụ F4. Các nguồn của file (JSON nhãn, CSV ảnh, `trung.json`, `yolo.json`,
    `nguon.json`) giữ nguyên byte. Không còn bản sao file cũ để so từng ô, nên "chỉ ô này đổi" là **suy luận** từ ba
    điểm trên, chưa phải phép so từng ô.

### 13.2. F2–F4 — đính chính diễn giải (không chạy lại gì)

- F2 (mục 3.2, 6.3, 11): bỏ câu "xác nhận độc lập xử lý tô xám". Mẫu YOLO chỉ kiểm siêu dữ liệu; pixel ảnh tô xám chưa
  kiểm.
- F3 (mục 0, 2, 6.4, 10, 11): tách **đơn vị chia split** (video) khỏi **tính độc lập thống kê**. Câu "không ảnh hưởng BDD
  val / UA-DETRAC test" đổi thành "chưa phát hiện chồng lấn trong phạm vi đã kiểm", kèm phạm vi kiểm.
- F4 (mục 6.4, 10): 3 nhóm sha256 trùng gồm 2 nhóm BDD train-fit và 1 nhóm BDD val, khớp `_dem/trung.json`.

### 13.3. F5 — ghi ngoài phạm vi vào memory của Claude Code (review_005; chỉ ghi nhận)

Nguồn: `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_005.json`, mục `issues` (F5,
P2). Theo review, lượt này chỉ ghi nhận: không sửa, xóa hay khôi phục memory; không chạy lại audit hay test. Trong mục
này, `stdout.log` là `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\calls\0010_coder\stdout.log`.

**Hai đường dẫn ngoài `write_paths` của R48B.** Cả hai nằm trong thư mục memory của Claude Code, ngoài repo và ngoài
git.

| # | Đường dẫn | Thao tác | Bằng chứng trong `stdout.log` (giờ UTC theo log) |
|:--|:--|:--|:--|
| 1 | `C:\Users\Mikuno\.claude\projects\D--LABOTARY-bytetrack\memory\ket-luan-tu-nguon-day-du.md` | Tạo mới bằng `Write`: memory loại `feedback` về bài học F4 (đếm trên nguồn đầy đủ; chụp hash trước / sau khi chạy lại) | Dòng 842: lệnh `Write` (2026-10-10T20:13:48.789Z). Dòng 843: kết quả `File created successfully`, `"type":"create"`, `"originalFile":null` |
| 2 | `C:\Users\Mikuno\.claude\projects\D--LABOTARY-bytetrack\memory\MEMORY.md` | Sửa bằng `Edit`: chèn một dòng trỏ tới file 1, ngay trước dòng `App Codex desktop khoa DLL sandbox` | Dòng 844: lệnh `Edit` (2026-10-10T20:13:51.883Z). Dòng 845: kết quả `has been updated successfully`; `structuredPatch` chỉ thêm 1 dòng, không xóa dòng nào |

- Dòng 842–845 cùng `session_id` `88f5fbd3-2ded-46e7-9a4f-eccbc331e3a9`, trùng `originSessionId` ghi trong file 1.
- Nguyên nhân (suy luận từ log): ngay trước lệnh `Write`, dòng 841 ghi ý định lưu bài học F4 vào memory. Lời gọi đã
  dùng cơ chế memory của Claude Code mà không đối chiếu với `write_paths`. Lớp quyền không chặn: lời gọi chạy với
  `--permission-mode dontAsk` và `--allowedTools` cho `Edit`, `Write` không giới hạn đường dẫn
  (`calls\0010_coder\command.json` dòng 14–27).
- Báo cáo JSON của lời gọi có nhắc một câu (`calls\0010_coder\validated.json` dòng 3: "Ngoài repo: đã thêm memory
  `ket-luan-tu-nguon-day-du.md` và một dòng trỏ trong `MEMORY.md`") nhưng không gọi đó là lệch phạm vi. Tài liệu bàn
  giao (mục 12–13 này, mục 0 `PLAN_DA_CHOT.md`) chưa ghi lại cho tới lượt này.

**Ảnh hưởng.** Hai thao tác không đổi file nào trong repo, `data/` hay kết quả. Memory chỉ chứa bài học quy trình,
không có số liệu hay quyết định mới. Tuy vậy, `MEMORY.md` được nạp vào ngữ cảnh của các lời gọi Claude sau, nên dòng
trỏ đã vào ngữ cảnh của chúng; lời gọi ghi nhận này cũng thấy dòng đó.

**Trạng thái lúc ghi nhận** (2026-10-11, chỉ đọc):
- File 1 còn nguyên. Nội dung trùng bản ghi ở `stdout.log` dòng 843 (`tool_use_result.content`), tức bản đã được Claude
  Code thêm `node_type`, `originSessionId` và `modified: 2026-10-10T20:13:48.797Z`. Phép so này bằng mắt, chưa so byte
  hay hash.
- `MEMORY.md` dòng 32 là dòng đã chèn, nằm giữa dòng 31 (`Luot R48 bao cao 17/10`) và dòng 33
  (`App Codex desktop khoa DLL sandbox`).
- Không sửa, xóa hay khôi phục. Người dùng quyết định giữ hay gỡ hai thay đổi này.

**Phạm vi rà soát** (Grep, chỉ đọc, trên `calls\*\stdout.log` và `calls\*\stderr.log` của lượt
`bytetrack_r48_bao_cao_20261017_1825`):
- Tìm lệnh `Write` / `Edit` / `NotebookEdit` có `file_path` ngoài `D:\LABOTARY\bytetrack`: chỉ có hai lệnh trên (dòng
  842, 844). Hai kết quả khớp ở `calls\0011_review\stderr.log` dòng 1947, 1949 là Codex in lại dòng 842, 844 bằng `rg`
  khi review (tiền tố `842:`, `844:`), không phải lần ghi mới.
- Tìm lệnh `Bash` / `PowerShell` của Claude có chứa `.claude`, `memory` hoặc `MEMORY`: 0 kết quả.
- Phép rà chỉ xét lời gọi công cụ của Claude có trong log. Nó không bắt được file do script hay tiến trình con ghi mà
  lệnh không nêu đường dẫn. Mẫu tìm dựa trên định dạng JSON của log: khóa `file_path` phải nằm trong 40 ký tự đầu của
  `input`; với lệnh shell, chỉ xét 600 ký tự đầu, trước dấu `"` đầu tiên.

**Giới hạn xác minh trạng thái trước sửa.**
- Thư mục memory nằm ngoài repo và ngoài git, nên không có lịch sử, diff hay bản sao để so.
- Orchestrator không chụp vùng này. `calls\0010_coder\changes.json` không có đường dẫn nào chứa `.claude`, `memory`,
  `MEMORY` hay `Mikuno` (Grep, 0 kết quả). `review_points\*\hashes.json` chỉ chứa đường dẫn tương đối trong repo: Grep
  các chuỗi trên trong `review_points` chỉ khớp nội dung file repo được sao vào `files\`, không khớp `hashes.json`.
- Trạng thái trước sửa chỉ được ghi trong kết quả của chính công cụ đã sửa, cùng log: dòng 843 (`"originalFile":null`,
  tức file 1 chưa tồn tại) và dòng 845 (`originalFile` là toàn văn `MEMORY.md` trước sửa; `structuredPatch` thêm đúng
  1 dòng). Đây là tự khai của công cụ, không phải bằng chứng độc lập. Vì vậy chưa chứng minh độc lập được rằng file 1
  chưa từng tồn tại, hay rằng `MEMORY.md` không đổi ở chỗ khác.

## 14. Phụ lục R48E — kiểm pixel 160 ảnh tô xám (2026-10-11; mốc người dùng duyệt qua email 11/10 07:06)

Phụ lục này bổ sung cho mục 6.3 và đính chính F2 (mục 13.2: "pixel ảnh tô xám chưa kiểm"). Các mục 0–13 không sửa.

**Phạm vi.** Chỉ kiểm đúng 160 ảnh `xu_ly = to_xam` trong mẫu 320 ảnh của mục 6.3 (seed 48, 4 tầng × 40 ảnh). **Không
kết luận cho toàn bộ 18.600 ảnh tô xám** của `data/yolo_bdd5_v1`. Script chỉ đọc ảnh. Mọi file ghi đều qua `_duong_ghi`,
chỉ vào `results/phase2/kc48_audit/`; không ghi gì vào `data/`.

### 14.1. Lệnh (foreground, HEAD `eaf2e87`, theo thứ tự thời gian; `pixel_to_xam_nhat_ky.jsonl`)

| # | Lệnh (`C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe ...`) | Bắt đầu | Thời gian | Exit |
|:--|:--|:--|--:|:--|
| 1 | `-m py_compile src/kc48/kiem_pixel_to_xam.py` | — | — | 0 |
| 2 | `src/kc48/kiem_pixel_to_xam.py --phan cong_bo` → `tieu_chi_pixel_to_xam.json` | 21:13:28 | 0,0 s | 0 |
| 3 | `src/kc48/kiem_pixel_to_xam.py --phan tinh` → `pixel_to_xam_tung_anh.csv`, `pixel_to_xam_tom_tat.json` | 21:13:30 | 64,3 s | 0 |
| 4 | `src/kc48/kiem_pixel_to_xam.py --phan chan_doan` (hậu kiểm, xem 14.4) → `pixel_to_xam_chan_doan_K.{csv,json}` | 21:31 | — | 0 |

Lệnh 1–3 chạy ở lời gọi Claude trước (bị mất kết nối trước khi trả báo cáo). Exit 0 của chúng được suy từ hai dòng nhật
ký: script chỉ ghi nhật ký sau khi phần chạy xong mà không lỗi. Lệnh 4 và một lần `py_compile` lại chạy ở lời gọi này,
exit 0.

### 14.2. Tiêu chí công bố trước (`tieu_chi_pixel_to_xam.json`, sha256 `4f62d1654e1b8c84e54868220759b81ab4cb1f26e1d9a6346708646147da0ca0`)

File tiêu chí được ghi lúc 21:13:28, trước lệnh tính lúc 21:13:30. `--phan cong_bo` từ chối ghi đè.
`--phan tinh` chỉ tự kiểm hai mục của file tiêu chí so với hằng số trong mã: chuỗi dung sai xám và bán kính lõi
(`src/kc48/kiem_pixel_to_xam.py:157–160`). Lệnh dừng nếu hai mục này lệch. Các ngưỡng T1/T2/T3 (0,99; 35 dB; MAE 2,0)
là hằng số trong mã (dòng 161), **không** được đối chiếu tự động với file tiêu chí. Đối chiếu thủ công cho thấy chúng
khớp giá trị đã công bố trong file. `pixel_to_xam_tom_tat.json` ghi lại sha256 trên. Tóm tắt:

- **Mẫu**: chọn lại bằng đúng thủ tục `phan_yolo` (seed, tầng, thứ tự). Đối chiếu từng (split, video, ảnh) với
  `yolo_kiem_mau.csv`; lệch thì dừng. Kết quả: khớp, 320 / 160.
- **Vùng** (tái suy độc lập từ JSON Scalabel gốc theo quy tắc 4–5 của docstring `src/bdd_yolo_export.py`, không gọi
  exporter): V = hợp vùng tô xám (crowd thuộc 5 lớp, other vehicle, trailer); G = V trừ hợp box có nhãn (kỳ vọng xám);
  K = V ∩ box có nhãn (kỳ vọng giữ nguyên); O = ngoài V (kỳ vọng giữ nguyên). Phần **lõi** bỏ dải biên JPEG bán kính
  r = 8 px: lõi G là pixel G có cửa sổ 17×17 toàn G; lõi O/K là pixel O/K có cửa sổ không chứa pixel G.
- **Xấp xỉ xám**: max theo kênh |I_ra − 114| ≤ 6. Lý do: JPEG q95 do cv2 ghi (`src/bdd_yolo_export.py:574`), thêm
  chênh lệch giữa hai bộ giải mã (cv2 lúc ghi, PIL lúc kiểm).
- **Ngưỡng**: T1 (vùng tô): lõi G khác rỗng và tỉ lệ xám ≥ 0,99. T2 (ngoài vùng): lõi O có PSNR ≥ 35 dB và MAE ≤ 2,0.
  T3 (box chồng vùng): nếu lõi K khác rỗng thì PSNR ≥ 35 dB và MAE ≤ 2,0; lõi K rỗng thì `khong_ap_dung`.
  Ảnh `dat` khi T1, T2 đạt và T3 ∈ {đạt, không áp dụng}. Ảnh `khong_dat` khi có tiêu chí không đạt. Ảnh
  `chua_ket_luan` khi lõi G rỗng và không tiêu chí nào trượt.

### 14.3. Kết quả (`pixel_to_xam_tung_anh.csv`: 160 dòng; `pixel_to_xam_tom_tat.json`)

| Kết quả theo ảnh | train hiếm=0 | train hiếm=1 | dev hiếm=0 | dev hiếm=1 | Tổng |
|:--|--:|--:|--:|--:|--:|
| `dat` | 32 | 31 | 33 | 35 | **131** |
| `chua_ket_luan` (lõi G rỗng) | 5 | 9 | 7 | 5 | **26** |
| `khong_dat` (đều do T3) | 3 | 0 | 0 | 0 | **3** |

| Tiêu chí | Đạt | Không đạt | Không áp dụng | Phân vị trên các ảnh có giá trị (min / p50 / max) |
|:--|--:|--:|--:|:--|
| T1, tỉ lệ xám trên lõi G của ảnh xuất | 134 | 0 | 26 | 1,0 / 1,0 / 1,0 (n = 134) |
| (đối chứng) cùng phép đo trên ảnh gốc | — | — | — | 0,0 / 0,0005 / 0,1006: ảnh gốc không xám sẵn, nên T1 phân biệt được |
| T2, lõi O: PSNR dB; MAE; p99 \|Δ\| | 160 | 0 | 0 | 45,54 / 50,88 / 57,50; 0,081 / 0,293 / 0,710; 2 / 3 / 5 |
| T3, lõi K: PSNR dB; MAE | 83 | 3 | 74 | 35,39 / 45,28 / 54,53; 0,146 / 0,973 / 2,704 (n = 86) |

Kiểm phụ, không có ngưỡng: 160/160 ảnh cùng kích thước; 160/160 sha256 ảnh xuất khớp manifest; 160/160 vùng tái suy
khớp file ignore (dung sai 0,006 px).

- **(1) Trong vùng ignore.** Ở cả 134 ảnh có lõi G, 100 % pixel lõi nằm trong dung sai xám. Tính cả dải biên (toàn bộ
  G, 157 ảnh có G) thì tỉ lệ thấp hơn: p5 0,869, p50 0,981, min 0,0. Bốn giá trị thấp nhất (0,0; 0,037; 0,375; 0,5)
  đều ở ảnh có G chỉ 16–106 px và lõi G rỗng (CSV dòng 115, 137, 120, 160), tức vùng mỏng nằm gần như toàn bộ trong dải biên.
  Giả thuyết (chưa kiểm riêng): tỉ lệ thấp là do nén JPEG làm nhòe biên vùng. Đây cũng là lý do tiêu chí dùng phần lõi.
- **(2) Ngoài vùng.** 160/160 ảnh đạt T2. PSNR lõi O thấp nhất 45,5 dB, MAE cao nhất 0,71/255. Tính cả dải biên quanh G,
  số gần như không đổi (PSNR O min 45,535 dB).
- **(3) Box có nhãn chồng vùng ignore.** 127/160 ảnh có ít nhất một box có nhãn chạm V, tổng 282 box. Phần chồng (K)
  **không bị tô xám**: tỉ lệ pixel xám trên K của ảnh xuất có p50 0,002, max 0,094, xấp xỉ ảnh gốc (p50 0,0011, max
  0,078). Điều này khớp quy tắc exporter: xóa mặt nạ ở mọi box có nhãn trước khi tô (`src/bdd_yolo_export.py:571–573`).
  Lõi K khác rỗng ở 86 ảnh: 83 đạt T3, **3 không đạt**. Cả 3 đều ở `train|hiếm=0`: `00ffeef4-ea139a4e-0000176`
  (42 px, MAE 2,43), `023b022b-723860c9-0000041` (136 px, MAE 2,31), `026e1e49-771df1d9-0000131` (513 px, MAE 2,70).
  Cả 3 trượt vì MAE > 2,0; PSNR 35,4–37,2 dB vẫn trên ngưỡng 35. 26 ảnh `chua_ket_luan` đều đạt T2, T3 nếu áp dụng; vùng
  ignore của chúng quá mỏng để tách khỏi biên JPEG theo r = 8 px. Có 3 ảnh mà V bị box có nhãn phủ hết nên không có G.

### 14.4. Hậu kiểm 3 ảnh T3 không đạt (`pixel_to_xam_chan_doan_K.{csv,json}`)

Phần này viết **sau khi đã thấy kết quả 14.3**. Nó không đổi T1–T3 hay kết quả theo ảnh đã công bố. Câu hỏi: sai số trên
lõi K do exporter đụng vào box có nhãn, hay do nén JPEG lại?

- Đối chứng 1: nén lại ảnh gốc trong bộ nhớ bằng cv2 q95, **không tô xám** (cv2 5.0.0 hiện có; không ghi file), rồi đo
  MAE trên cùng lõi K. Với 3 ảnh trượt: 2,373 / 2,323 / 2,644, so với ảnh xuất 2,429 / 2,311 / 2,704. Chênh lệch
  +0,056 / −0,012 / +0,061. Trên cả 86 ảnh có lõi K: chênh p50 +0,004, min −0,083, max +0,347; 81/86 ảnh có
  |chênh| ≤ 0,1.
- Đối chứng 2: trên cùng ảnh xuất, MAE ở phần box có nhãn nằm ngoài V (lõi) là 0,20 / 0,80 / 0,66.
- Diễn giải (quan sát, không phải chứng minh): chỉ riêng việc nén JPEG lại ảnh gốc đã cho mức sai số gần như bằng ảnh
  xuất trên các vùng nhỏ này (42–513 px). Vì vậy 3 ca trượt T3 **phù hợp với** mất mát do nén lại JPEG ở vùng có nhiều
  chi tiết; hậu kiểm **chưa chứng minh** đây là nguyên nhân duy nhất. Trong phép đo này không thấy dấu hiệu exporter tô
  hay sửa phần box có nhãn. Giả thuyết (chưa kiểm): ngưỡng MAE 2,0 công bố trước có thể quá chặt cho vùng nhỏ nhiều chi
  tiết. Ngưỡng **không** được sửa sau khi xem; 3 ảnh vẫn ghi `khong_dat`.
- Giới hạn của hậu kiểm: bản cv2 lúc export chưa được xác minh trong phụ lục này. Ảnh nén lại không trùng byte với ảnh
  xuất (MAE giữa hai ảnh trên lõi K là 0,06–0,67), nên đối chứng 1 chỉ là xấp xỉ.

### 14.5. Kết luận trong phạm vi mẫu

Trên 160 ảnh mẫu: 131 đạt, 26 chưa kết luận (vùng mỏng), 3 không đạt theo ngưỡng MAE của T3. Hậu kiểm cho thấy sai số
của 3 ca này phù hợp với mất mát do nén lại JPEG; chưa chứng minh nguyên nhân, nên chưa loại trừ được lỗi xử lý vùng
(giới hạn ở 14.4). Vùng ignore được tô (114,114,114) ở phần lõi; ngoài vùng
giữ nguyên trong mức nén lại q95; box có nhãn chồng vùng ignore được giữ, không bị tô. Kết luận chỉ áp dụng cho mẫu này,
không suy ra toàn bộ 18.600 ảnh tô xám. Đính chính F2 ("pixel chưa kiểm") nay được thay bằng: "đã kiểm pixel trên 160
ảnh mẫu, xem mục 14".
