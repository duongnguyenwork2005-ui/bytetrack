# Kế hoạch khóa — đánh giá detector trên UA-DETRAC test (bước 3 của trình tự sau D1–D5)

**Khóa lúc**: 2026-10-10, trước lượt suy luận đầu tiên. Sau khi xem kết quả, không sửa đè file này. Mọi sửa đổi chỉ ghi vào phụ lục cuối file.
**Người duyệt chạy**: người dùng, 2026-10-10 khoảng 02:20 (yêu cầu chạy xong bước 4 trước 10:00).
**Căn cứ**:
- Dự thảo `handoff/logs/review45/06_du_thao_giao_thuc_detrac.md`, mục 3–8.
- Quyết định D1–D5: `handoff/REVIEW46_CHI_TIET.md` mục 11.4.
- Mốc A: `handoff/logs/review47/01_fingerprint_detrac_test.md`.
- Base repo: `e4a0311`.

## 1. Dữ liệu (đã xác minh ở mốc A, adapter kiểm lại trước mỗi lượt)

| Mục | Giá trị |
|:--|:--|
| Manifest `results/phase2/manifests/detrac_test.json` | 40 video, sha256 `ff2b4b9c658aced54567247c16b5a0fc43ae1e9836af70e7ea656fce3077f47c` |
| Fingerprint `data/interim/hash_tap_danh_gia/detrac_test_fingerprint.json` | `dd0f2a0e9dcaf11f26fe397ff4c4eb7ef8a8ecfd248d4682aa652b3af156343a` |
| Bảng hash ảnh `detrac_test_sha256.csv` | `3cea34881d90031a5ed2161a774154e87691228da9b25b1ac0b07dbe8717b166` (56.340 ảnh, 0 ảnh lệch 960×540) |
| Danh sách loại D4 `detrac_test_loai_D4.csv` | `4d7312530505651b92da414b2dcf82c4ec74486dc408fa37dc9f927d85723551` (173 ảnh, khớp vòng 45) |
| Tập đánh giá sau D4 | 56.167 ảnh. Danh sách có thứ tự: `46204978758016cb16627b2035d34a40c9b31c5eb477db6dd442afab3eba92d7`. Tổng hợp nội dung: `e4fff621ae775ad7d571d67e23c5f75db100434b4905c1e405fea24dec696cb6` |
| XML | 40 file, sha256 từng file ghi trong fingerprint, khớp vòng 45 |

Adapter đọc lại các số sau ở chế độ `--chi-kiem` trên dữ liệu thật (exit 0, 64 s):
- GT: car 548.555, bus 71.785;
- vùng bỏ qua: `ignored_region` 220.290 lượt theo ảnh, van 38.519, others 16.915.

## 2. Mã (sha256)

| File | sha256 | Ghi chú |
|:--|:--|:--|
| `src/danh_gia_detector_detrac_test.py` | `6540832e8cae42d1b129f3c4aaed65422196ffdb502f0db19d183b640709100b` | adapter mới (mốc B) |
| `src/test_danh_gia_detector_detrac_test.py` | `a9e8fc0e48c30c8876249a215b3fd7c9fe3931d1775cb705d7ebd7a52b8113c8` | 10/10 PASS (fixture, dự đoán giả lập) |
| `src/eval_detector_5lop.py` | `30a1fd5a63641f1cebc22e4e7ef7e4600a48a45ed6f7d533bf319af0fa156f7f` | evaluator; không đổi, trùng dự thảo |
| `src/danh_gia_detector_bdd_val.py` | `de2c06f4db2eb99c15b44d873cc63b6b6b446bc632000701671dedd65677020a` | chỉ import `quy_tac_hieu_luc`, `chan_doan` |
| `src/fingerprint_detrac_test.py` | `37b62da99b77880a27aa234a887158e51498b6822921b430c1366d271f621255` | mốc A; adapter dùng các hàm đọc/băm |
| `src/bdd_yolo_export.py` | `1ca964b0fc81701ac746335fa20ac6e5231322a4cdd11ad47753a2b25fc31493` | chỉ import `_hop_le_box` |
| `src/detector_classes.py` | `7bba3439987377cb2f161eb31e2cf0afafae3e814f165279087b7a3780699c3e` | 5 lớp đã khóa |

Môi trường: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe` (Python 3.11.9), ultralytics 8.4.141, torch 2.6.0+cu124, GPU NVIDIA GeForce GTX 1650. Giống vòng 43.

## 3. Checkpoint (giống kế hoạch khóa vòng 43)

| Tên | Đường dẫn | sha256 |
|:--|:--|:--|
| `yolov8n_coco` (baseline) | `models/yolov8n.pt` | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `yolov8n_ft` | `models/finetune_phase2/train_bdd5_v1_yolov8n_day_du/weights/best.pt` | `5f66776d3202389ca484f3d9e2f107dddb354d21a9f1762d19401d690a4a0d8a` |
| `yolo11n_ft` | `models/finetune_phase2/train_bdd5_v1_yolo11n_day_du/weights/best.pt` | `d7d1c7e86d03c69d913febe79e4cbb6c2e1bcf9f17f034724af4de2ccfba7923` |

## 4. Quy tắc (không đổi sau khi khóa)

**GT theo D1–D5:**
- car, bus là GT.
- D1: van là vùng bỏ qua cho mọi lớp.
- D2: others là vùng bỏ qua cho mọi lớp.
- `ignored_region` là vùng bỏ qua của mọi ảnh trong video.
- D4: loại 173 ảnh không có `<frame>`.
- D5: quy tắc bỏ qua của evaluator như BDD.
  - Detection được ghép với GT hợp lệ trước.
  - Chỉ detection chưa ghép mới xét bỏ qua, với IoA (giao / diện tích detection, lấy max theo từng vùng) ≥ ngưỡng IoU đang xét.
  - GT trong vùng bỏ qua vẫn được chấm. Không tô xám ảnh.
- Box được cắt vào khung ảnh; box nhỏ hơn 1 px sau khi cắt thì bỏ.

**Dự đoán:** dùng nguyên hàm của evaluator. `imgsz` 640, `conf` 0,001, NMS IoU 0,7 theo lớp, `max_det` 300, FP32, lọc lớp theo tên trước NMS, batch 16, `device 0`. Dự đoán cả 5 lớp.

**Chấm:** ghép COCO; AP nội suy 101 điểm × 10 ngưỡng IoU 0,50:0,05:0,95. Điểm vận hành: conf ≥ 0,25, IoU 0,5.

**Báo cáo theo D3:**
- Kết quả đánh giá là **car, bus** và `mAP (car, bus)`.
- truck, motorcycle, bicycle ghi N/A.
- Bộ đếm thô nằm riêng ở `so_lieu_tho_khong_phai_metric`.
- Tập này **không** dùng để chọn checkpoint. Tập đã được dùng trong các thí nghiệm tracking trước đây, nên không mô tả là "chưa từng xem".

## 5. Lệnh (tuần tự, mỗi model một lượt, chạy độc lập qua WMI; thư mục mới, không ghi đè)

```
python -u src/danh_gia_detector_detrac_test.py --model yolov8n_coco=models/yolov8n.pt --sha256 yolov8n_coco=f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 --out results/phase2/detector_eval/detrac_test/20261010_yolov8n_coco
python -u src/danh_gia_detector_detrac_test.py --model yolov8n_ft=models/finetune_phase2/train_bdd5_v1_yolov8n_day_du/weights/best.pt --sha256 yolov8n_ft=5f66776d3202389ca484f3d9e2f107dddb354d21a9f1762d19401d690a4a0d8a --out results/phase2/detector_eval/detrac_test/20261010_yolov8n_ft
python -u src/danh_gia_detector_detrac_test.py --model yolo11n_ft=models/finetune_phase2/train_bdd5_v1_yolo11n_day_du/weights/best.pt --sha256 yolo11n_ft=d7d1c7e86d03c69d913febe79e4cbb6c2e1bcf9f17f034724af4de2ccfba7923 --out results/phase2/detector_eval/detrac_test/20261010_yolo11n_ft
```

Log và trạng thái ghi ở `results/phase2/detector_eval/detrac_test/20261010_logs/`, cùng bản sao script chạy.

**Lượt hợp lệ khi:**
- exit 0;
- kiểm trước dự đoán đều đạt;
- `so_anh_da_du_doan` = 56.167;
- có đủ `du_doan_<tên>.npz`, `danh_sach_anh.json`, `ket_qua.json`, `theo_lop.csv`, `tong_hop.csv`, `so_lieu_tho_khong_phai_metric.csv`.

Lượt lỗi thì ghi lại nguyên nhân rồi chạy lại vào thư mục mới, không xóa thư mục cũ.

## 6. Chỗ khác giao thức (ghi rõ)

1. **Chưa commit trước suy luận.** Giao thức yêu cầu commit, nhưng commit do người dùng quyết định, và người dùng đang ngủ. Thay vào đó:
   - file này được khóa bằng sha256;
   - sha256 được gửi email cho người dùng **trước** lượt suy luận đầu tiên (thời điểm email là dấu thời gian).

   Người dùng commit sau nếu muốn.
2. **Codex review adapter và kế hoạch song song với suy luận, không làm trước.**
   - Codex hết hạn mức tới 05:50; phiên Claude dừng từ khoảng 03:30 tới 06:30. Để kịp mốc 10:00 người dùng đặt, suy luận khởi chạy ngay khi khóa, còn review Codex chạy cùng lúc.
   - Trước khi chạy, Claude đã tự kiểm: test fixture 10/10, `--chi-kiem` trên dữ liệu thật, mẫu 60 ảnh băm lại khớp. Đây **không** phải review độc lập.
   - Lỗi chấm điểm mà Codex tìm ra thì chấm lại từ NPZ đã lưu bằng `--cham-lai`, không cần suy luận lại. Lỗi ở bước dự đoán thì chạy lại. Cả hai trường hợp đều ghi vào phụ lục.

## Phụ lục (chỉ thêm sau khi khóa)

### PL1 — 2026-10-10, ghi sau khi chạy xong (Claude, khoảng 08:00)

Phần thân bên trên giữ nguyên: 7.817 byte đầu của file này vẫn có sha256 `7028a291d2595ee55ead9b055fa1001225733ff8075eec0773680178842d4dfd`.

1. **Dấu thời gian khóa** (mục 6.1):
   - Email vào hàng đợi lúc 06:33:59.
   - Gmail SMTP nhận lúc 06:34:06 (`250 2.0.0 OK`).
   - Bản ghi: `D:\LABOTARY\automation\notify\sent\bytetrack_r47_fingerprint_adapter_20261010_021706-khoa-ke-hoach-7028a291d259.json`.
   - Runner khởi động 06:34:01 và kiểm dữ liệu trong 66 s. Thư mục lượt đầu tạo lúc 06:35:09, tức dự đoán bắt đầu sau khi Gmail đã nhận email.
2. **Chạy**: cả 3 lượt đạt tiêu chí ở mục 5. Mỗi lượt exit 0, dự đoán 56.167 ảnh, đủ 6 file, sha256 checkpoint đúng mục 3.

   | Lượt | Thời gian |
   |:--|:--|
   | `yolov8n_coco` | 06:34:01–06:52:39 |
   | `yolov8n_ft` | 06:52:39–07:10:11 |
   | `yolo11n_ft` | 07:10:11–07:27:41 |

   Kết quả: `04_ket_qua_danh_gia_detrac_test.md`.
3. **sha256 NPZ**: ghi sau khi chạy, vì adapter đã khóa chưa ghi giá trị này.

   | NPZ | sha256 |
   |:--|:--|
   | `yolov8n_coco` | `599cdec3500377e8236add99642a69e658b19beb807fb44237601babff2e46f1` |
   | `yolov8n_ft` | `82fee2885cf93b3e93ac85660a5cc72da068cdb763246a19eadff8b73b019c5f` |
   | `yolo11n_ft` | `960fd880c082e504106056c715ab288f9d012f6262664f06eae2c4ead2eb6fac` |

4. **Review Codex song song** (mục 6.2):
   - Lần 1 (06:35:44–06:39:25): revise, 2 issue P2 về `--cham-lai`. Cả hai issue không ảnh hưởng tới dự đoán, ghép hay AP.
   - Claude sửa trên bản sao. Lần 2 (06:43:33–06:44:46): approve.
   - Bản sửa v2 (adapter `971f50a9…`, test `2945cd70…`, 14/14) được chép vào repo lúc 07:39:35, sau lượt cuối. Lệnh `cp -p` giữ mtime của bản sao.
   - Ba lượt chạy bằng bản đã khóa `6540832e…`: không thư mục kết quả nào có `nguon_du_doan_*.json`, file mà v2 luôn ghi trước khi chấm. Vì vậy không chạy lại.
5. **Lỗi hiệu năng của `--cham-lai`** (phát hiện khi chấm lại kiểm chéo, ngoài kế hoạch):
   - Lượt chấm lại `yolov8n_coco` khởi chạy 07:39:54, ghi vào `20261010_yolov8n_coco_kiem_cham_lai/`.
   - `doc_du_doan` đọc `z[...]` của NpzFile trong vòng lặp theo ảnh, nên mỗi lần lặp lại giải nén cả mảng (1,23 s). Ước tính lượt này mất khoảng 24 giờ.
   - Claude dừng lượt này lúc khoảng 07:48. Thư mục (chỉ có `danh_sach_anh.json`) và log `20261010_logs/kiem_cham_lai_yolov8n_coco.log` được giữ nguyên.
   - Lỗi có từ bản khóa, nhưng chỉ ảnh hưởng `--cham-lai`. Ba lượt chính chấm trong bộ nhớ.
   - Bản sửa v3 đọc mỗi mảng một lần: adapter `bb4b497b4f89a4464777144a35d66cbd109d03d3c485c4176cde0b67e358ccb4`, test 14/14.
6. **Chấm lại kiểm chéo cả 3 model** bằng v3 (07:49:27–07:59:32, ghi vào `20261010_<tên>_kiem_cham_lai_2/`):
   - Cả 3 lượt exit 0.
   - Số liệu trùng khớp tuyệt đối với lượt gốc: `ket_qua_D3`, chẩn đoán thô và danh tính dữ liệu đều có 0 chỗ khác; 3 CSV trùng từng byte.
   - sha256 của 3 NPZ và 3 `ket_qua.json` gốc không đổi.
7. **Codex nghiệm thu** kết quả, báo cáo 04, bản sửa v3 và phụ lục này: ghi ở PL2.

### PL2 — 2026-10-10, Codex nghiệm thu bước 4 (ghi khoảng 08:07)

- Phạm vi nghiệm thu: số liệu báo cáo 04 so với `ket_qua.json`/CSV, tính hợp lệ theo mục 5, diễn giải, bản sửa v3, và PL1.
- Lần 1 (08:01:10–08:03:12): **revise**. Không có lỗi số liệu. Có 2 issue diễn đạt trong báo cáo 04:
  - [P2] Kết luận bus→truck vượt quá dữ liệu, vì GT bus trong vùng bỏ qua vẫn được chấm. Đã sửa thành "chưa xác định".
  - [P3] Danh sách trường JSON khác nhau thiếu `lenh`. Đã bổ sung.
- Lần 2, xác nhận phạm vi hẹp (08:04:30–08:05:21): **approve**.
- Hồ sơ: `D:\LABOTARY\automation\runs\bytetrack_r47_fingerprint_adapter_20261010_021706\codex_nghiem_thu_buoc4*\`.
- Hạn mức Codex sau hai lời gọi: 5 giờ 44 %, tuần 85 %.

Bước 4 hoàn tất. Commit do người dùng quyết định.
