# R53C — Kết quả đánh giá đầy đủ ba detector trên UA-DETRAC test và BDD val

Ngày 2026-10-20, Claude (người triển khai). Bước `R53C_REPORT_FULL`, kết thúc tại GPT review. PLAN mục 0 đã cập nhật
trước báo cáo này. Lượt này **chỉ đọc** kết quả và bằng chứng: không chạy lại đánh giá, suy luận, chấm lại, test, GPU,
WMI hay tác vụ nền; không tracking; không commit hay stage.

**Căn cứ:**
- Giao thức khóa: `handoff/logs/review53/02_ke_hoach_khoa_danh_gia.md` (mục 1–10, PL1–PL3). Tiêu chí hậu kiểm: mục 10.
- R53B approve ở review vòng 5:
  `D:/LABOTARY/automation/runs_local/bytetrack_r53b_lock_smoke_20261020_0356/reports/review_005.json`.
- Nghiệm thu A của GPT sau lượt đầy đủ, verdict **approve**:
  `D:/LABOTARY/automation/runs_local/hoi_codex_sau_danh_gia_full_20261020/answer.json`.
- Bộ chạy tuần tự và nhật ký: `D:/LABOTARY/automation/danh_gia_r53_full/` (`chay_tuan_tu.py`, `nhat_ky.log`,
  `trang_thai_chuoi.json`, `cham_lai.log`).

## 1. Phạm vi và nguồn số liệu

- Ba detector, đúng mục 1 tài liệu 02: `yolov8n_coco` (baseline, `models/yolov8n.pt`, ánh xạ `coco80`),
  `yolov8n_mixed5` (best e15, `lop5`), `yolo11n_mixed5` (best e13, `lop5`).
- Mọi số trong mục 2–4 đọc từ `results/phase2/r53_detector/full/<tập>/<tên>/` (thư mục lượt gốc, không phải
  `__cham_lai`):
  - UA-DETRAC test: `theo_lop.csv` (AP từng lớp), `tong_hop.csv` (`mAP (car, bus)`), khớp khối `ket_qua_D3` của
    `ket_qua.json`;
  - BDD val: `theo_lop.csv` (AP50, AP50-95, P/R từng lớp) và `mo_hinh.<tên>.macro` của `ket_qua.json` (mAP).
- Smoke (`results/phase2/r53_detector/smoke/`), dev nội bộ và các lượt R43/R47 **không** có trong bảng.
- Số làm tròn 4 chữ số. Chênh lệch = detector mixed5 − baseline, cùng thang 0–1 của AP, tính trên giá trị đầy đủ rồi mới
  làm tròn.
- Giao thức (mục 4 tài liệu 02): imgsz 640, conf 0,001, NMS IoU 0,7 theo lớp, max_det 300, FP32; ghép COCO, AP nội suy
  101 điểm trên IoU 0,50:0,05:0,95; điểm vận hành conf ≥ 0,25, IoU 0,5 (P/R ký hiệu `vh`).

## 2. UA-DETRAC test — theo D3 (56.167 ảnh, 40 video)

GT: car 548.555, bus 71.785. Van, others, `ignored_region` là vùng bỏ qua (D1, D2); 173 ảnh không có `<frame>` bị loại
(D4). Theo D3 chỉ báo car, bus và `mAP (car, bus)`; truck, motorcycle, bicycle là **N/A** trong JSON lẫn CSV. Bộ đếm thô
của ba lớp này nằm ở `so_lieu_tho_khong_phai_metric.csv` và không phải metric, nên không trình bày ở đây.

| Detector | Lớp | AP50 | AP75 | AP50-95 | P vh | R vh |
|:--|:--|--:|--:|--:|--:|--:|
| `yolov8n_coco` | car | 0,8577 | 0,6640 | 0,5699 | 0,8893 | 0,7840 |
| `yolov8n_coco` | bus | 0,8825 | 0,7413 | 0,6294 | 0,8373 | 0,8510 |
| `yolov8n_coco` | **mAP (car, bus)** | **0,8701** | 0,7026 | **0,5997** | 0,8633 | 0,8175 |
| `yolov8n_mixed5` | car | 0,8995 | 0,7311 | 0,6298 | 0,8524 | 0,8692 |
| `yolov8n_mixed5` | bus | 0,8576 | 0,6447 | 0,5925 | 0,8607 | 0,8000 |
| `yolov8n_mixed5` | **mAP (car, bus)** | **0,8785** | 0,6879 | **0,6111** | 0,8566 | 0,8346 |
| `yolo11n_mixed5` | car | 0,9055 | 0,7346 | 0,6357 | 0,8346 | 0,8846 |
| `yolo11n_mixed5` | bus | 0,8713 | 0,6896 | 0,6086 | 0,8384 | 0,8282 |
| `yolo11n_mixed5` | **mAP (car, bus)** | **0,8884** | 0,7121 | **0,6222** | 0,8365 | 0,8564 |
| cả ba | truck, motorcycle, bicycle | N/A | N/A | N/A | N/A | N/A |

Nguồn: `full/detrac_test/<tên>/theo_lop.csv`, `full/detrac_test/<tên>/tong_hop.csv` (`n_anh` = 56167 cả ba).

## 3. BDD100K MOT val, COCO-video — năm lớp (39.973 ảnh, 200 video)

GT theo `bdd_yolo_export.vai_tro`; crowd/ignore là vùng bỏ qua. mAP = trung bình năm lớp (`macro.so_lop_co_gt` = 5).
`theo_lop.csv` của BDD không có cột AP75 (AP75 từng lớp chỉ có trong `ket_qua.json`) và khối `macro` chỉ có mAP50,
mAP50-95, nên bảng chỉ dùng AP50 và AP50-95.

| Lớp | n_gt | Detector | AP50 | AP50-95 | P vh | R vh |
|:--|--:|:--|--:|--:|--:|--:|
| car | 341.436 | `yolov8n_coco` | 0,5577 | 0,3369 | 0,7819 | 0,4407 |
| | | `yolov8n_mixed5` | 0,7778 | 0,5154 | 0,8203 | 0,7040 |
| | | `yolo11n_mixed5` | 0,7740 | 0,5137 | 0,8041 | 0,7019 |
| truck | 27.343 | `yolov8n_coco` | 0,1831 | 0,1345 | 0,3923 | 0,1993 |
| | | `yolov8n_mixed5` | 0,3863 | 0,2749 | 0,7106 | 0,3310 |
| | | `yolo11n_mixed5` | 0,3726 | 0,2653 | 0,7098 | 0,3117 |
| bus | 9.087 | `yolov8n_coco` | 0,2487 | 0,2012 | 0,4960 | 0,2679 |
| | | `yolov8n_mixed5` | 0,3869 | 0,2938 | 0,6171 | 0,3579 |
| | | `yolo11n_mixed5` | 0,3666 | 0,2818 | 0,5854 | 0,3432 |
| motorcycle | 902 | `yolov8n_coco` | 0,1175 | 0,0582 | 0,3931 | 0,1264 |
| | | `yolov8n_mixed5` | 0,2174 | 0,0993 | 0,4619 | 0,2284 |
| | | `yolo11n_mixed5` | 0,2016 | 0,0959 | 0,4351 | 0,2007 |
| bicycle | 4.135 | `yolov8n_coco` | 0,1126 | 0,0546 | 0,7950 | 0,0609 |
| | | `yolov8n_mixed5` | 0,1793 | 0,0885 | 0,4986 | 0,1308 |
| | | `yolo11n_mixed5` | 0,1788 | 0,0905 | 0,4930 | 0,1277 |
| **mAP (5 lớp)** | | `yolov8n_coco` | **0,2439** | **0,1571** | 0,5716 | 0,2190 |
| | | `yolov8n_mixed5` | **0,3896** | **0,2544** | 0,6217 | 0,3504 |
| | | `yolo11n_mixed5` | **0,3787** | **0,2494** | 0,6055 | 0,3371 |

Nguồn: `full/bdd_val/<tên>/theo_lop.csv`; dòng mAP từ `full/bdd_val/<tên>/ket_qua.json` → `mo_hinh.<tên>.macro`
(`mAP50`, `mAP50_95`, `precision_vh`, `recall_vh`).

## 4. Chênh lệch mô tả so với baseline `yolov8n_coco`

Chỉ mô tả hiệu số trên một lần chạy của từng detector. Không bootstrap, không khoảng tin cậy, không kiểm định; không
kết luận nguyên nhân (trộn dữ liệu, kiến trúc, không gian lớp hay ngân sách huấn luyện) và không suy ra chất lượng
tracking.

**UA-DETRAC test (D3):**

| Chỉ số | `yolov8n_mixed5` − baseline | `yolo11n_mixed5` − baseline |
|:--|--:|--:|
| car AP50 / AP75 / AP50-95 | +0,0417 / +0,0671 / +0,0599 | +0,0478 / +0,0706 / +0,0658 |
| car P vh / R vh | −0,0369 / +0,0852 | −0,0547 / +0,1006 |
| bus AP50 / AP75 / AP50-95 | −0,0249 / −0,0966 / −0,0370 | −0,0112 / −0,0517 / −0,0208 |
| bus P vh / R vh | +0,0234 / −0,0510 | +0,0011 / −0,0228 |
| mAP (car, bus) mAP50 / mAP75 / mAP50-95 | +0,0084 / −0,0148 / +0,0115 | +0,0183 / +0,0095 / +0,0225 |
| mAP (car, bus) P vh / R vh | −0,0067 / +0,0171 | −0,0268 / +0,0389 |

Mô tả: so với baseline, cả hai detector mixed5 có AP car cao hơn ở cả ba ngưỡng và AP bus thấp hơn ở cả ba ngưỡng. Ở
điểm vận hành, car có recall cao hơn, precision thấp hơn; bus có recall thấp hơn, còn precision bus cao hơn với
`yolov8n_mixed5` (+0,0234) và gần như không đổi với `yolo11n_mixed5` (+0,0011). `mAP (car, bus)` mAP50 và mAP50-95 của hai
detector mixed5 cao hơn baseline; mAP75 của `yolov8n_mixed5` thấp hơn baseline.

**BDD val (năm lớp):**

| Lớp | AP50: v8n mixed5 / 11n mixed5 − baseline | AP50-95: v8n mixed5 / 11n mixed5 − baseline |
|:--|--:|--:|
| car | +0,2201 / +0,2163 | +0,1785 / +0,1769 |
| truck | +0,2032 / +0,1895 | +0,1404 / +0,1308 |
| bus | +0,1382 / +0,1179 | +0,0926 / +0,0806 |
| motorcycle | +0,0999 / +0,0841 | +0,0411 / +0,0378 |
| bicycle | +0,0667 / +0,0662 | +0,0339 / +0,0359 |
| **mAP (5 lớp)** | **+0,1456 / +0,1348** | **+0,0973 / +0,0924** |

mAP P vh / R vh: `yolov8n_mixed5` +0,0501 / +0,1314; `yolo11n_mixed5` +0,0338 / +0,1180.

Mô tả: trên BDD val, AP50 và AP50-95 của cả năm lớp ở hai detector mixed5 đều cao hơn baseline. Ở điểm vận hành, P và R
của từng lớp đều cao hơn baseline, trừ precision của bicycle (baseline 0,7950; mixed5 0,4986 / 0,4930). Hai detector
mixed5 chênh nhau ít hơn nhiều so với chênh lệch của mỗi detector so baseline. Báo cáo không xếp hạng hai detector mixed5
với nhau.

## 5. Tiêu chí hậu kiểm mục 10 — bằng chứng từng tiêu chí

Đường dẫn tương đối tính từ `results/phase2/r53_detector/full/`. `<tập>__<tên>` là thư mục giám sát trong `_giam_sat/`.

| # | Tiêu chí (mục 10 tài liệu 02) | Bằng chứng | Kết quả đọc được |
|:--|:--|:--|:--|
| 1 | exit 0 thật theo bộ giám sát; không Ctrl+C | `_giam_sat/<tập>__<tên>/tom_tat.json` (`exit_code`, `exit_code_hex`, `ctrl_c`, `loi_in_man_hinh`), `trang_thai.json`; `D:/LABOTARY/automation/danh_gia_r53_full/trang_thai_chuoi.json`, `nhat_ky.log` | 6/6: `exit_code` 0, `ctrl_c` rỗng, `loi_in_man_hinh` 0; chuỗi `"xong 6/6, tat ca exit 0"`; lượt sau chỉ khởi chạy sau khi lượt trước có exit (nhat_ky.log) |
| 2 | `so_anh_da_du_doan` = 56.167 / 39.973; `gioi_han_anh` = 0 | `<tập>/<tên>/ket_qua.json` (`mo_hinh.<tên>.so_anh_da_du_doan`, `dataset.gioi_han_anh`) | DETRAC 56167 ×3, BDD 39973 ×3; `gioi_han_anh` 0 ×6 |
| 3 | NPZ cùng sha256; DETRAC có `nguon_du_doan_<tên>.json` ghi trước khi chấm, `npz_sha256` khớp | DETRAC: `detrac_test/<tên>/nguon_du_doan_<tên>.json` (`npz_sha256`, `tao_luc`); `detrac_test/<tên>__cham_lai/ket_qua.json` (`npz_sha256` nhận lại). BDD: `_hau_kiem/bdd_val__<tên>__npz.txt` (`Get-FileHash`); `bdd_val/<tên>__cham_lai/ket_qua.json` → `cham_lai.dau_vao_sha256` | DETRAC `a48966b7…b707` / `af7530b5…6449` / `fbfb358f…7b57`; BDD `534a0cbf…eedf` (88.266.007 B) / `2c6f571f…ed67` (83.165.805 B) / `c8409075…576a` (82.963.892 B). GPT (nghiệm thu A) tự băm sáu NPZ: khớp |
| 4 | JSON/CSV hợp lệ, đủ file | DETRAC `detrac_test/<tên>/`: `ket_qua.json`, `danh_sach_anh.json`, `theo_lop.csv`, `tong_hop.csv`, `so_lieu_tho_khong_phai_metric.csv`; BDD `bdd_val/<tên>/`: `ket_qua.json`, `theo_lop.csv` | đủ file cho 6/6 (Glob); các file đọc được trong lượt này đều phân tích được |
| 5 | sha256 checkpoint, danh tính dữ liệu bằng khóa; ánh xạ lớp đúng | `<tập>/<tên>/ket_qua.json` (`mo_hinh.<tên>.sha256`, `anh_xa_lop.loai_checkpoint`, khối `dataset`) so với mục 1–2 tài liệu 02 | checkpoint `f59b3d83…3b36` / `542b1381…0694` / `9a5e34a1…86fb` đúng tên; `coco80` cho baseline, `lop5` cho hai mixed5; mỗi `ket_qua.json` DETRAC chứa đủ 6 hash danh tính (manifest, fingerprint, bảng hash, D4, danh sách ảnh, nội dung), mỗi BDD đủ 5 (nhãn, manifest, danh sách ảnh, nội dung, bảng hash) |
| 6 | AP từng lớp: DETRAC theo D3, BDD năm lớp; so sánh mô tả với baseline | mục 2–4 báo cáo này; `ket_qua_D3` trong `detrac_test/<tên>/ket_qua.json` (truck/motorcycle/bicycle = `"N/A"`) | đạt |
| 7 | chấm lại từ NPZ bằng scorer cũ, số liệu trùng | DETRAC `detrac_test/<tên>__cham_lai/` (`--cham-lai`); BDD `bdd_val/<tên>__cham_lai/` (`--buoc cham-lai-bdd`, `log.txt`, `exit.json`); `D:/LABOTARY/automation/danh_gia_r53_full/cham_lai.log` | 6/6 exit 0. BDD: `goi_suy_luan` 0, `module_da_nap` ultralytics/torch false, `dau_vao_khong_doi` true, `doi_chieu_ket_qua_goc.truong_khac` rỗng, chuỗi khóa 4 mức. GPT (A): chín CSV DETRAC và ba CSV BDD chấm lại trùng từng byte với gốc |
| 8 | smoke không vào bảng kết quả | mục 1 báo cáo này; bảng chỉ lấy từ `full/` | đạt |

**Kiểm khóa trong lượt dài:** trước mỗi lượt, `kiem-khoa` exit 0 (`trang_thai_chuoi.json` → `kiem_khoa.dong_cuoi`: 25
file khớp `SHA256SUMS` `e1442ea6…250d`, chuỗi `phu_luc_1, phu_luc_2, phu_luc_3`). `dong_cuoi` bị bộ chạy cắt ở 300 ký
tự (`chay_tuan_tu.py:132`, `last[0][:300]`), nên danh sách "file doi sau khoa" ở đó dừng sau `chuan_bi_danh_gia.py`;
lần `kiem-khoa` của lượt R53C (mục 8) in đủ, thêm `test_chuan_bi_danh_gia.py`, khớp việc test đổi ở PL1–PL3. Sau khởi chạy, `nhat_ky.log` ghi
`pyc_moi_trong_src` rỗng và `ultralytics_goc_repo` false cho 6/6, cwd giám sát `D:\LABOTARY\bytetrack` cho 6/6. Lệnh
trong `tom_tat.json` chạy python với `-B -u` (tôi đọc toàn văn ở `_giam_sat/detrac_test__yolov8n_coco/tom_tat.json`;
với BDD, `cham-lai-bdd` kiểm lệnh giám sát = lệnh bàn giao, `cham_lai.log` ghi đạt 3/3).

## 6. Thời gian chạy và tài nguyên (máy GTX 1650 của dự án, một lần chạy)

Chuỗi: 2026-10-20 11:30:46 → 13:04:19 (+07), khoảng 1 giờ 34 phút cho 6 lượt. Dự toán mục 9 tài liệu 02 là kịch bản
≈ 1,9 giờ (không phải cận trên).

| Lượt | Bắt đầu → kết thúc | Toàn lượt (bộ giám sát, s) | Dự đoán + chấm (`thoi_gian_s` adapter, s) | RSS max (MiB) | RAM máy trống thấp nhất (MiB) |
|:--|:--|--:|--:|--:|--:|
| DETRAC `yolov8n_coco` | 11:30:48 → 11:47:58 | 1.030,5 | 967,7 | 1.418,6 | 80 |
| DETRAC `yolov8n_mixed5` | 11:48:07 → 12:05:28 | 1.040,4 | 975,0 | 1.303,6 | 369 |
| DETRAC `yolo11n_mixed5` | 12:05:59 → 12:20:48 | 889,1 | 829,0 | 1.348,6 | 319 |
| BDD `yolov8n_coco` | 12:21:17 → 12:37:53 | 996,6 | 884,0 | 1.336,5 | 432 |
| BDD `yolov8n_mixed5` | 12:38:15 → 12:51:28 | 792,9 | 705,3 | 904,4 | 150 |
| BDD `yolo11n_mixed5` | 12:51:33 → 13:04:05 | 752,1 | 675,7 | 1.368,7 | 191 |

Nguồn: `_giam_sat/<tập>__<tên>/tom_tat.json` (`bat_dau`, `ket_thuc`, `thoi_gian_s`, `ram.rss_chinh_mib.max`,
`ram.ram_he_thong_con_trong_mib.min`); `<tập>/<tên>/ket_qua.json` (`mo_hinh.<tên>.thoi_gian_s`). RSS là working set
Windows, không gồm VRAM; không có tiến trình con (`so_con` max 0). Đây là thời gian tường của lượt đánh giá có kiểm dữ
liệu, không phải benchmark FPS.

Chấm lại (`cham_lai.log`, thời gian tường giữa các mốc): DETRAC 13:04:33 → 13:12:08 (171 / 143 / 141 s, gồm kiểm dữ
liệu 59 / 56 / 57 s); BDD 13:12:08 → 13:18:14 (119 / 122 / 125 s, gồm kiểm dữ liệu 76 / 74 / 79 s). Không suy luận lại (BDD ghi `goi_suy_luan` 0, không nạp ultralytics/torch).

## 7. Giới hạn

- **BDD val không hoàn toàn chưa xem.** `ket_qua.json` của cả sáu thư mục BDD ghi `sequence_da_xem_truoc` =
  `b1c66a42-6f7d68ca`, `b1c81faa-3df17267`, `b1c81faa-c80764c5` (metric đã được xem ở bước smoke trước đây), kèm ghi chú
  "KHONG mo ta toan bo val la tap hoan toan chua xem". Ngoài ra mẫu smoke R53B 64 ảnh (`chuan_bi/mau_smoke_bdd_val.json`,
  mục 5 tài liệu 02) lấy ảnh của `b1c66a42-6f7d68ca` và ảnh `b1c9c847-3bda4659-0000001.jpg`; AP smoke đã được xem nhưng
  không dùng để chọn hay chỉnh. BDD val cũng đã được dùng ở R43 để đánh giá các detector trước. Theo khóa, không chọn
  checkpoint và không tuning bằng BDD val, UA-DETRAC test hay smoke.
- **D3 cho UA-DETRAC:** chỉ car và bus có GT; van/others là vùng bỏ qua. Kết quả UA-DETRAC không nói gì về truck,
  motorcycle, bicycle; ba lớp này chỉ được đánh giá trên BDD val.
- **Một lần chạy, không bất định:** không bootstrap, không khoảng tin cậy, không lặp seed. Ảnh là các frame liên tiếp của
  40 / 200 video nên không độc lập; hiệu số nhỏ (ví dụ vài phần nghìn của `mAP (car, bus)`) không được diễn giải là
  khác biệt chắc chắn.
- **Không nhân quả:** baseline là model COCO 80 lớp dùng ánh xạ theo tên; hai mixed5 là đầu 5 lớp, huấn luyện trên dữ
  liệu trộn, khác kiến trúc giữa v8n và 11n. Bảng không tách được đóng góp của từng yếu tố và không đo tác động riêng
  của việc trộn dữ liệu.
- **Không suy ra tracking:** AP detector theo frame không cho biết HOTA / IDF1 / AssA của ByteTrack + KF/EKF/UKF/PF.
  Tracking trên test40 thuộc kế hoạch khác, chưa chạy.
- **Tái lập:** chấm lại từ NPZ tất định (trùng kết quả). Suy luận lại trên GPU có thể lệch nhẹ do phiên bản thư viện
  hoặc cuDNN; không kiểm trong R53.
- **Giới hạn kiểm tra của nghiệm thu A** (`answer.json`): GPT chỉ đọc và đối chiếu, không chạy lại scorer, không băm lại
  toàn bộ ảnh. **R53B review vòng 5** (`review_005.json`): review tĩnh; cổng nguồn không chứng minh mã chưa từng bị sửa
  rồi khôi phục trong lượt chạy. Điều này dựa vào `kiem-khoa` trước mỗi lượt.
- **Giới hạn của lượt R53C này:** tôi không băm NPZ, CSV hay ảnh; nhận định "trùng từng byte" của CSV chấm lại và "sáu NPZ
  khớp" là kết quả GPT tự kiểm ở nghiệm thu A. Tôi tự đọc và so giá trị (không so byte) của bốn file chấm lại với bản
  gốc: `detrac_test/yolov8n_coco__cham_lai/tong_hop.csv`, `detrac_test/yolov8n_mixed5__cham_lai/theo_lop.csv`,
  `detrac_test/yolo11n_mixed5__cham_lai/tong_hop.csv`, `bdd_val/yolo11n_mixed5__cham_lai/theo_lop.csv`: mọi giá trị trùng.
  Hash danh tính dữ liệu trong 12 `ket_qua.json` (gốc + chấm lại) được tìm bằng Grep theo chuỗi hash của mục 2 tài
  liệu 02.

## 8. Lệnh đã chạy trong lượt R53C

| # | Lệnh | Exit | Ghi chú |
|---|---|---|---|
| 1 | `git status --short` | 0 | trạng thái cây trước khi sửa; mọi thay đổi chưa commit có sẵn được giữ |
| 2 | `python src/r53_detector/chuan_bi_danh_gia.py --buoc kiem-khoa` (sau khi sửa hai file) | 0 | `[DAT] kiem-khoa: 25 file khop SHA256SUMS (e1442ea6…250d), khoa luc 2026-10-20T04:14:32; chuoi phu luc dat (phu_luc_1, phu_luc_2, phu_luc_3), file doi sau khoa: handoff/logs/review53/02_ke_hoach_khoa_danh_gia.md, src/r53_detector/chuan_bi_danh_gia.py, src/r53_detector/test_chuan_bi_danh_gia.py`. Chỉ đọc; hai file của lượt này không nằm trong khóa |

(`python` = `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe`.) Ngoài hai lệnh trên chỉ dùng Read / Grep / Glob.
Chỉ ghi hai file: `handoff/PLAN_DA_CHOT.md` (mục 0) và file này.

## 9. Bước tiếp

- GPT review báo cáo này và mục 0.
- Sau khi approve: lượt C — GPT soạn kế hoạch tracking ByteTrack + KF/EKF/UKF/PF trên test40 với detector mixed5, trình
  người dùng duyệt trước khi triển khai. Không chọn detector theo AP test, không tuning trên test40.
- Stage/commit hoãn; người dùng quyết định riêng.
