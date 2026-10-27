# R53B — Kế hoạch khóa đánh giá ba detector trên UA-DETRAC test và BDD val

**Trạng thái:** viết 2026-10-20, trước khóa và trước mọi smoke. File này nằm trong `SHA256SUMS` của bước khóa. Sau khi
khóa, phần thân không được sửa đè. Mọi thay đổi chỉ ghi vào bản phụ lục riêng, và phải review lại trước lượt dài.
Thời điểm khóa được ghi ở `results/phase2/r53_detector/chuan_bi/thoi_diem_khoa.json`, không ghi trong file này.
**Chưa commit**, theo yêu cầu hiện tại; người dùng quyết định commit.

**Căn cứ:**
- Plan orchestrator R53, bước `R53B_LOCK_SMOKE`. Người dùng duyệt chạy lượt mới chỉ gồm R53B.
- R53A đã được Codex approve: `reports/review_002.json` của lượt
  `bytetrack_r53_mixed5_handoff_eval_prep_20261019_1845`. Đầu ra nằm ở `results/phase2/r53_detector/ban_giao/`.
- Quyết định D1–D5: `handoff/REVIEW46_CHI_TIET.md` mục 11.4.
- Kế hoạch và kết quả R47 (UA-DETRAC test): `handoff/logs/review47/03_ke_hoach_khoa_danh_gia_detrac_test.md`, `04_...`.
- Kế hoạch và thực thi R43 (BDD val): `handoff/logs/review43/04_ke_hoach_danh_gia.md`, `handoff/REVIEW43_CHI_TIET.md`.

## 1. Ba detector (chỉ ba tên này)

| Tên | Checkpoint | sha256 | Loại ánh xạ lớp | Nguồn |
|:--|:--|:--|:--|:--|
| `yolov8n_coco` | `models/yolov8n.pt` | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | `coco80` (car 2, truck 7, bus 5, motorcycle 3, bicycle 1, theo tên) | baseline, như R43/R47 |
| `yolov8n_mixed5` | `models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/best.pt` | `542b1381c33a673531bccdbca769f26408a16e9ba6ed89e49238a0f16b9b0694` | `lop5` (ID 0..4) | best e15 trên dev nội bộ; R53A |
| `yolo11n_mixed5` | `models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt` | `9a5e34a12c1d97cd1b0e49325d2eadbd443a965633b464461980df91fea386fb` | `lop5` (ID 0..4) | best e13 trên dev nội bộ; R53A |

- Hash của hai best.pt khớp `ban_giao/checkpoint_sha256.csv` (R53A). Bước `lenh-day-du` băm lại cả ba file: cả ba khớp.
- Không chọn lại checkpoint. Không tuning bằng BDD val, UA-DETRAC test hay smoke. Không dùng trọng số BDD-only.

## 2. Dữ liệu (tiền kiểm 2026-10-20 bằng cổng adapter, exit 0)

Tiền kiểm gọi `kiem_va_doc_gt` của adapter gốc với `kiem_noi_dung=True`, tức băm lại nội dung **mọi** ảnh. Sau đó so
từng trường với hồ sơ: R47 cho UA-DETRAC, `ket_qua.json` các lượt `r43_*` cho BDD. Có lệch thì exit 2 và không ghi
gì. Fingerprint và bảng hash chỉ được đọc, không được cập nhật.

**UA-DETRAC test** (`tien_kiem_detrac_test.json`, kiểm 64 s):
- 40 video, 56.340 ảnh trên đĩa. Loại D4 173 ảnh, còn **56.167** ảnh đánh giá.
- Manifest `ff2b4b9c658aced54567247c16b5a0fc43ae1e9836af70e7ea656fce3077f47c`.
- Fingerprint `dd0f2a0e9dcaf11f26fe397ff4c4eb7ef8a8ecfd248d4682aa652b3af156343a`.
- Bảng hash ảnh `3cea34881d90031a5ed2161a774154e87691228da9b25b1ac0b07dbe8717b166`.
- Danh sách D4 `4d7312530505651b92da414b2dcf82c4ec74486dc408fa37dc9f927d85723551`.
- Danh sách ảnh có thứ tự `46204978758016cb16627b2035d34a40c9b31c5eb477db6dd442afab3eba92d7`.
- Tổng hợp nội dung `e4fff621ae775ad7d571d67e23c5f75db100434b4905c1e405fea24dec696cb6`.
- Tất cả khớp R47.

**BDD100K MOT val, COCO-video** (`tien_kiem_bdd_val.json`, kiểm 82 s):
- 200 video, **39.973** ảnh, kể cả ảnh không có annotation.
- Nhãn `074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d`.
- Manifest `2071c15bd300e5dfff156fd708393b7018b074777ec4ef83da236409eb34c3ba`.
- Danh sách ảnh `4bdf4889984ad9b9f37602f711fd066622f5473364e033fda0b886e7829666d5`.
- Tổng hợp nội dung `1e209af9a7077c173d46006b25c3eb6f7f66b9aed484b9d2a8dd281f61317555`.
- Bảng hash `60fea2a9ce623450c1526953b7e427016ed917cfd702a7c6275525fca3582d20`.
- Đối chiếu 39.973/39.973 ảnh, 0 lệch. Tất cả khớp R43.

## 3. Mã và môi trường

| File | sha256 | Ghi chú |
|:--|:--|:--|
| `src/danh_gia_detector_detrac_test.py` | `bb4b497b4f89a4464777144a35d66cbd109d03d3c485c4176cde0b67e358ccb4` | adapter UA-DETRAC, = v3 R47 PL1 |
| `src/danh_gia_detector_bdd_val.py` | `de2c06f4db2eb99c15b44d873cc63b6b6b446bc632000701671dedd65677020a` | adapter BDD, = R47 mục 2 |
| `src/eval_detector_5lop.py` | `30a1fd5a63641f1cebc22e4e7ef7e4600a48a45ed6f7d533bf319af0fa156f7f` | dự đoán / ghép / AP, = R47 |
| `src/detector_classes.py` | `7bba3439987377cb2f161eb31e2cf0afafae3e814f165279087b7a3780699c3e` | 5 lớp đã khóa, = R47 |
| `src/fingerprint_detrac_test.py` | `37b62da99b77880a27aa234a887158e51498b6822921b430c1366d271f621255` | hàm băm mốc A, = R47 |
| `src/bdd_yolo_export.py` | `1ca964b0fc81701ac746335fa20ac6e5231322a4cdd11ad47753a2b25fc31493` | `_hop_le_box`, `vai_tro`, = R47 |
| `src/config.py` | `37e4f56247b7be0896c29d74026b863435b2eec5f53c11fc1a8cd073da70d88b` | `PROJECT_ROOT` |
| `src/r53_detector/chuan_bi_danh_gia.py` | `ffef2f8474248dea340b733ef7e46e4cbad5014a656d843f12a802f322b7f794` | R53B: tiền kiểm / khóa / smoke |
| `src/r53_detector/test_chuan_bi_danh_gia.py` | `806980c6c3401618cda90e2be9aec0169ede0fa24896cc10cd7bd14b101ec2f3` | test fixture R53B, 38/38 |

Môi trường thực tế (ghi trong tiền kiểm):
- `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe`, Python 3.11.9, Windows 10.0.26200.
- ultralytics 8.4.141, torch 2.6.0+cu124, torchvision 0.21.0+cu124, numpy 2.4.6, pillow 12.3.0, opencv-python 5.0.0.93.
- GPU GTX 1650, như R43/R47. Tên GPU thực tế được ghi vào `ket_qua.json` của từng smoke.
- ultralytics và torch giống R43/R47.

## 4. Quy tắc (giữ nguyên; không đổi sau khóa)

- **Dự đoán**: hàm `du_doan_ultralytics` của evaluator.
  - imgsz 640, conf 0,001, NMS IoU 0,7 theo lớp (không agnostic), max_det 300, FP32 (`half=False`).
  - Lọc lớp theo tên trước NMS. batch 16, device 0.
- **Chấm**: `eval_detector_5lop.danh_gia`.
  - Ghép theo COCO. AP nội suy 101 điểm trên 10 ngưỡng IoU 0,50:0,05:0,95.
  - Điểm vận hành: conf ≥ 0,25, IoU 0,5. Ngưỡng lấy qua `quy_tac_hieu_luc(0.25)`.
- **UA-DETRAC (D1–D5)**:
  - car, bus là GT.
  - D1 `van` và D2 `others` là vùng bỏ qua cho mọi lớp. `ignored_region` cũng là vùng bỏ qua.
  - D4: loại 173 ảnh không có `<frame>`.
  - D5: ghép GT hợp lệ trước. Chỉ detection chưa ghép mới xét bỏ qua (IoA ≥ ngưỡng IoU). GT trong vùng bỏ qua vẫn được chấm.
  - Box cắt vào khung; nhỏ hơn 1 px thì bỏ.
- **Báo cáo D3**: chỉ AP car, bus và `mAP (car, bus)`. truck, motorcycle, bicycle ghi N/A trong cả JSON lẫn CSV. Bộ
  đếm thô của ba lớp này nằm riêng ở `so_lieu_tho_khong_phai_metric`.
- **BDD val**: GT theo `bdd_yolo_export.vai_tro`; crowd/ignore là vùng bỏ qua. Báo đủ năm lớp. Motorcycle và
  bicycle chỉ được đánh giá từ BDD val.
- **Không làm**:
  - không bootstrap;
  - không phân tích nguyên nhân;
  - không kết luận nhân quả về tác động riêng của việc trộn dữ liệu;
  - không dùng AP smoke để xếp hạng hay chỉnh tham số.
- Kết quả cuối chỉ so sánh mô tả ba detector theo giao thức này.

## 5. Mẫu smoke (lập và lưu trước suy luận)

**Quy tắc chọn:**
1. Đi theo thứ tự ảnh của adapter.
2. Với từng lớp cần kiểm, lấy ảnh đầu tiên có GT lớp đó: car/bus trên UA-DETRAC, năm lớp trên BDD.
3. Loại trùng.
4. Bổ sung các ảnh đầu chưa được chọn cho tới khi đủ 64 ảnh. Giữ thứ tự adapter.
- Sha256 từng ảnh được băm lại và đối chiếu với bảng hash của tập.
- Mỗi lệnh smoke tính lại mẫu và phải trùng mẫu đã khóa: cùng danh sách, cùng sha256 từng ảnh. Ba model dùng cùng một
  mẫu.

| Tập | File | Ảnh | Ảnh bắt buộc | GT trong mẫu |
|:--|:--|--:|:--|:--|
| UA-DETRAC test | `chuan_bi/mau_smoke_detrac_test.json` | 64 | car: `MVI_39031/img00001.jpg` (chỉ số 0); bus: `MVI_39051/img00616.jpg` (chỉ số 2085) | car 104, bus 1 |
| BDD val | `chuan_bi/mau_smoke_bdd_val.json` | 64 | car/truck/motorcycle: `b1c66a42-6f7d68ca-0000001.jpg` (0); bus: `b1c66a42-6f7d68ca-0000056.jpg` (55); bicycle: `b1c9c847-3bda4659-0000001.jpg` (606) | car 1.349, truck 48, bus 8, motorcycle 63, bicycle 1 |

## 6. Khóa và smoke (thực thi trong lượt này)

Khóa:

```
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc khoa
```

Bước khóa tạo ba file trong `results/phase2/r53_detector/chuan_bi/`:
- `khoa_danh_gia.json`;
- `SHA256SUMS`, gồm sha256 của: khóa, file này, tiền kiểm, mẫu, 9 file mã, 3 checkpoint, manifest/fingerprint/bảng
  hash/D4/nhãn;
- `thoi_diem_khoa.json`.

Bước khóa từ chối nếu một trong các điều sau xảy ra:
- đã có smoke;
- thiếu tiền kiểm;
- checkpoint lệch;
- file này thiếu lệnh hoặc hash, hoặc còn dấu hiệu chưa xong;
- ổ đĩa còn dưới 15 GB.

Sáu smoke, chạy riêng từng lệnh, GPU tuần tự:

```
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc smoke --tap detrac_test --model yolov8n_coco
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc smoke --tap detrac_test --model yolov8n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc smoke --tap detrac_test --model yolo11n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc smoke --tap bdd_val --model yolov8n_coco
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc smoke --tap bdd_val --model yolov8n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc smoke --tap bdd_val --model yolo11n_mixed5
```

Mỗi smoke làm theo thứ tự, trước khi nạp model:
1. Kiểm khóa: băm lại mọi file trong `SHA256SUMS`.
2. Kiểm sha256 checkpoint.
3. Kiểm dữ liệu bằng cổng adapter, đủ nội dung ảnh, và so danh tính với khóa.
4. Tính lại mẫu và so với mẫu đã khóa.
5. Ghi `danh_sach_anh.json`.

Sau đó smoke dự đoán đúng 64 ảnh mẫu, rồi ghi:
- `du_doan_<tên>.npz` và `nguon_du_doan_<tên>.json`;
- `ket_qua.json`, có cảnh báo "smoke kỹ thuật";
- CSV kết quả;
- `log.txt` và `exit.json`.

Thư mục: `results/phase2/r53_detector/smoke/<tập>/<model>/`. Thư mục đã có thì exit 1. Chạy lại vào `<model>__lan<N>`.

Cấu hình thư viện (ultralytics/matplotlib) đặt ở `smoke/_cau_hinh_thu_vien/`, không ghi vào hồ sơ người dùng.
`smoke/.gitignore` loại NPZ khỏi Git.

**Tiêu chí smoke:**
- exit 0;
- khóa kiểm đạt trước khi nạp model;
- đúng checkpoint;
- danh tính dữ liệu bằng khóa;
- tối đa 64 ảnh, cùng mẫu giữa các model;
- loại ánh xạ lớp đúng: `coco80` cho baseline, `lop5` cho mixed5;
- NPZ có sha256 và nguồn dự đoán;
- chấm lại từ NPZ cho kết quả trùng;
- UA-DETRAC: chuẩn hóa D3 đạt; BDD: đủ năm lớp có GT trong mẫu.

AP smoke **không** dùng để xếp hạng hay chỉnh tham số.

## 7. Sáu lệnh đánh giá đầy đủ (chỉ bàn giao — KHÔNG chạy trong lượt R53)

Dùng nguyên hai adapter gốc. Mỗi lệnh chạy một model vào một thư mục mới; adapter từ chối nếu thư mục đã tồn tại.

```
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_detrac_test.py --model yolov8n_coco=models/yolov8n.pt --sha256 yolov8n_coco=f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/detrac_test/yolov8n_coco
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_detrac_test.py --model yolov8n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/best.pt --sha256 yolov8n_mixed5=542b1381c33a673531bccdbca769f26408a16e9ba6ed89e49238a0f16b9b0694 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/detrac_test/yolov8n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_detrac_test.py --model yolo11n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt --sha256 yolo11n_mixed5=9a5e34a12c1d97cd1b0e49325d2eadbd443a965633b464461980df91fea386fb --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/detrac_test/yolo11n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_bdd_val.py --model yolov8n_coco=models/yolov8n.pt --sha256 yolov8n_coco=f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/bdd_val/yolov8n_coco
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_bdd_val.py --model yolov8n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/best.pt --sha256 yolov8n_mixed5=542b1381c33a673531bccdbca769f26408a16e9ba6ed89e49238a0f16b9b0694 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/bdd_val/yolov8n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_bdd_val.py --model yolo11n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt --sha256 yolo11n_mixed5=9a5e34a12c1d97cd1b0e49325d2eadbd443a965633b464461980df91fea386fb --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/bdd_val/yolo11n_mixed5
```

Mỗi adapter tự kiểm trước khi dự đoán (sai → exit 2, không dự đoán):
- sha256 checkpoint bằng `--sha256`;
- manifest, fingerprint, bảng hash, D4, nhãn;
- sha256 nội dung **mọi** ảnh.

## 8. Quy trình chạy riêng ngoài orchestrator (theo R43/R47)

Lượt dài chỉ khởi chạy sau khi người dùng cho phép, trong phiên thiết lập sau nghiệm thu R53B.

**Trước khi chạy:**
1. Chạy `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc kiem-khoa`.
   Lệnh này chỉ đọc, băm lại mọi file trong `SHA256SUMS`, và phải exit 0. Lệch thì dừng; không sửa khóa để hợp thức hóa.
2. Xác nhận không có tiến trình train hay đánh giá nào đang dùng GPU. GPU chỉ chạy một model tại một thời điểm.
3. Ổ D còn ít nhất 15 GB. Máy cắm sạc và không ngủ (bài học R42/R52). Đóng ứng dụng nặng: R43 có lúc RAM trống
   thấp nhất chỉ 50 MiB.
4. Các thư mục `results/phase2/r53_detector/full/<tập>/<model>/` chưa tồn tại.

**Khởi chạy:**
- Mỗi lệnh ở mục 7 chạy qua bộ giám sát `src/chay_giam_sat.py chay`. Bộ giám sát ghi PID và create_time, log có mốc
  giờ, mẫu RAM, và exit code thật.
- Tiến trình được khởi chạy độc lập bằng WMI `Win32_Process.Create`, cùng cơ chế với
  `handoff/logs/review43/khoi_chay_danh_gia_wmi.ps1`.
- Chạy tuần tự, theo thứ tự `detrac_test` (coco → v8n mixed5 → 11n mixed5) rồi `bdd_val` (cùng thứ tự).
- Thư mục giám sát riêng, mới cho từng lệnh:

```
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/chay_giam_sat.py chay --thu-muc results/phase2/r53_detector/full/_giam_sat/detrac_test__yolov8n_coco -- C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_detrac_test.py --model yolov8n_coco=models/yolov8n.pt --sha256 yolov8n_coco=f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/detrac_test/yolov8n_coco
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/chay_giam_sat.py chay --thu-muc results/phase2/r53_detector/full/_giam_sat/detrac_test__yolov8n_mixed5 -- C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_detrac_test.py --model yolov8n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/best.pt --sha256 yolov8n_mixed5=542b1381c33a673531bccdbca769f26408a16e9ba6ed89e49238a0f16b9b0694 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/detrac_test/yolov8n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/chay_giam_sat.py chay --thu-muc results/phase2/r53_detector/full/_giam_sat/detrac_test__yolo11n_mixed5 -- C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_detrac_test.py --model yolo11n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt --sha256 yolo11n_mixed5=9a5e34a12c1d97cd1b0e49325d2eadbd443a965633b464461980df91fea386fb --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/detrac_test/yolo11n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/chay_giam_sat.py chay --thu-muc results/phase2/r53_detector/full/_giam_sat/bdd_val__yolov8n_coco -- C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_bdd_val.py --model yolov8n_coco=models/yolov8n.pt --sha256 yolov8n_coco=f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/bdd_val/yolov8n_coco
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/chay_giam_sat.py chay --thu-muc results/phase2/r53_detector/full/_giam_sat/bdd_val__yolov8n_mixed5 -- C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_bdd_val.py --model yolov8n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/best.pt --sha256 yolov8n_mixed5=542b1381c33a673531bccdbca769f26408a16e9ba6ed89e49238a0f16b9b0694 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/bdd_val/yolov8n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/chay_giam_sat.py chay --thu-muc results/phase2/r53_detector/full/_giam_sat/bdd_val__yolo11n_mixed5 -- C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_bdd_val.py --model yolo11n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt --sha256 yolo11n_mixed5=9a5e34a12c1d97cd1b0e49325d2eadbd443a965633b464461980df91fea386fb --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/bdd_val/yolo11n_mixed5
```

**Theo dõi:**
- Dùng `src/chay_giam_sat.py trang-thai --thu-muc <thư mục giám sát của lệnh>`. Không đoán trạng thái từ monitor đã hết hạn.
- Lệnh tiếp theo chỉ bắt đầu sau khi lệnh trước có exit code.

**Khi lỗi:**
- Giữ nguyên thư mục kết quả, log và thư mục giám sát của lượt lỗi; không xóa, không ghi đè.
- Ghi nguyên nhân vào phụ lục.
- Chạy lại vào thư mục mới, ví dụ `results/phase2/r53_detector/full/detrac_test/yolov8n_mixed5__lan2`, kèm thư mục giám sát mới.
- Nếu **chỉ bước chấm** lỗi (NPZ đã ghi):
  - không suy luận lại;
  - UA-DETRAC: chấm lại bằng `--cham-lai`. Adapter đọc `nguon_du_doan_<tên>.json`, kiểm danh tính dữ liệu, cấu hình
    dự đoán và sha256 NPZ.
  - BDD: chấm lại từ NPZ bằng `bv.kiem_va_doc_gt` + `eval_detector_5lop.danh_gia` trong công cụ hậu kiểm của phiên sau.
    Không dùng trực tiếp `handoff/logs/review43/tong_hop_danh_gia.py`, vì công cụ này ghi cứng đường dẫn R43.

Lệnh chấm lại đối chiếu cho UA-DETRAC (sau lượt đầy đủ, không cần GPU):

```
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_detrac_test.py --cham-lai results/phase2/r53_detector/full/detrac_test/yolov8n_coco --out results/phase2/r53_detector/full/detrac_test/yolov8n_coco__cham_lai
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_detrac_test.py --cham-lai results/phase2/r53_detector/full/detrac_test/yolov8n_mixed5 --out results/phase2/r53_detector/full/detrac_test/yolov8n_mixed5__cham_lai
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_detrac_test.py --cham-lai results/phase2/r53_detector/full/detrac_test/yolo11n_mixed5 --out results/phase2/r53_detector/full/detrac_test/yolo11n_mixed5__cham_lai
```

## 9. Dự toán (kịch bản theo giả định, không phải cận trên)

**Căn cứ:** các lượt đầy đủ đã chạy cùng adapter, cùng GPU GTX 1650, cùng kiến trúc (YOLOv8n, YOLO11n), cùng imgsz.
Bước `lenh-day-du` đọc số liệu từ `ket_qua.json` và kích thước NPZ.

| Tập | Lượt căn cứ | Dự đoán + chấm (s) | Toàn lượt | NPZ |
|:--|:--|:--|:--|:--|
| UA-DETRAC test (56.167 ảnh) | R47 `20261010_yolov8n_coco` / `_yolov8n_ft` / `_yolo11n_ft` | 1.047 / 983 / 975 | 18,6 / 17,5 / 17,5 phút (gồm kiểm dữ liệu ~66 s; lần này 64 s) | 247 / 284 / 288 MB |
| BDD val (39.973 ảnh) | R43 `r43_yolov8n_coco` / `_yolov8n_ft` / `_yolo11n_ft` | 1.010 / 1.080 / 1.049 | 1.143 / 1.206 / 1.181 s ≈ 19,1–20,1 phút (gồm kiểm 122–129 s; lần này 82 s) | 88 / 83 / 83 MB |

**Giả định:** `yolov8n_mixed5` chạy như `yolov8n_ft`, `yolo11n_mixed5` như `yolo11n_ft`. Cùng kiến trúc và cùng
đầu 5 lớp; số detection có thể khác nên thời gian chấm và kích thước NPZ có thể khác.

**Kịch bản:**
- UA-DETRAC: 3 lượt ≈ 54 phút.
- BDD val: 3 lượt ≈ 59 phút.
- Tổng tuần tự ≈ 1,9 giờ. Kịch bản chậm ×1,5 ≈ 2,8 giờ.
- Nên chạy thành hai đợt, mỗi tập một đợt (~1 giờ), để mỗi đợt nằm trong giới hạn tác vụ nền ~2 giờ.
- NPZ cả 6 lượt ≈ 1,1 GB theo giả định trên. Ổ D hiện còn 42,6 GB.
- RAM theo R43: RSS tối đa 0,94–1,13 GiB/lượt.

## 10. Tiêu chí hậu kiểm lượt dài

Từng lượt:
1. exit 0, đúng exit code thật theo bộ giám sát; không có Ctrl+C.
2. `so_anh_da_du_doan` = 56.167 (UA-DETRAC) hoặc 39.973 (BDD); `gioi_han_anh` = 0.
3. Có `du_doan_<tên>.npz` cùng sha256. UA-DETRAC có thêm `nguon_du_doan_<tên>.json`, ghi trước khi chấm, với
   `npz_sha256` khớp file.
4. JSON và CSV hợp lệ:
   - UA-DETRAC: `ket_qua.json`, `danh_sach_anh.json`, `theo_lop.csv`, `tong_hop.csv`, `so_lieu_tho_khong_phai_metric.csv`;
   - BDD: `ket_qua.json`, `theo_lop.csv`.
5. sha256 checkpoint và danh tính dữ liệu trong `ket_qua.json` bằng khóa (mục 1–2). Loại ánh xạ lớp đúng (`coco80` / `lop5`).
6. Báo AP từng lớp:
   - UA-DETRAC theo D3: car, bus, `mAP (car, bus)`; ba lớp còn lại N/A;
   - BDD: năm lớp.
   So sánh **mô tả** với baseline `yolov8n_coco`.
7. Chấm lại từ NPZ bằng scorer cũ để đối chiếu, mục 8. Số liệu phải trùng. Nếu chỉ lỗi chấm thì chấm lại, không suy
   luận lại.
8. Kết quả smoke không đưa vào bảng kết quả.

Tracking trên test40 và mọi bước sau đánh giá detector thuộc kế hoạch khác, cần người dùng duyệt riêng.

## Phụ lục 1 — sửa sau review R53B vòng 1 (F1–F3), 2026-10-20

Phần trên mốc `## Phụ lục 1` giữ nguyên **từng byte** của bản đã khóa lúc 2026-10-20T04:14:32
(sha256 `7d1f42e37c9cf4dae48d0a2ba0eddd9e46350e1312add80125ace5e1df24e2f5`). Phụ lục này **không** đổi quy tắc (mục 4),
ba checkpoint (mục 1), danh tính dữ liệu (mục 2), mẫu smoke (mục 5) hay sáu lệnh đầy đủ (mục 7). Khóa gốc
(`chuan_bi/khoa_danh_gia.json`, `SHA256SUMS`, `thoi_diem_khoa.json`) và toàn bộ bằng chứng smoke dưới `smoke/` giữ nguyên.
Chưa commit, theo yêu cầu hiện tại.

### PL1.1 Lý do

Review Codex R53B vòng 1 (`bytetrack_r53b_lock_smoke_20261020_0356/reports/review_001.json`, verdict revise):
- F1: test chỉ gọi wrapper, chưa có fixture cho CLI của chính hai adapter gốc.
- F2: mục 8 giao phần chấm lại BDD cho một công cụ chưa có; adapter BDD không ghi hash/nguồn NPZ trước khi chấm.
- F3: mục 8 chưa đặt môi trường cho tiến trình khởi chạy độc lập (bytecode, cấu hình thư viện, offline).

### PL1.2 Khóa phụ lục và cách kiểm

Chỉ ba file được đổi sau khóa: tài liệu này (chỉ nối thêm sau mốc), `src/r53_detector/chuan_bi_danh_gia.py`
(`ffef2f8474248dea340b733ef7e46e4cbad5014a656d843f12a802f322b7f794` → `6fc0c72a4d0bc34fc8c616f77dc44b47f949575f93152f4bc7812540249194e4`)
và `src/r53_detector/test_chuan_bi_danh_gia.py`
(`806980c6c3401618cda90e2be9aec0169ede0fa24896cc10cd7bd14b101ec2f3` → `4e1f2235803151527d8985bb136874e6abc7225e1b2836ab2cdb392ca5622fbf`).
Không file nào khác của mục 3 đổi; hai adapter, scorer, `detector_classes`, `fingerprint_detrac_test`, `bdd_yolo_export`,
`config` vẫn đúng hash khóa.

Lệnh `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc phu-luc`:
1. Kiểm `SHA256SUMS` gốc, `khoa_danh_gia.json` và `thoi_diem_khoa.json` không đổi.
2. Mọi file khác trong `SHA256SUMS` gốc phải còn đúng hash. File ngoài tập ba file trên bị đổi → exit 2.
3. Giữ bản gốc: tài liệu 02 — phần byte trước mốc phải có sha256 bằng bản khóa. Hai file mã — chép từ snapshot review
   của orchestrator (`D:\LABOTARY\automation\runs_local\bytetrack_r53b_lock_smoke_20261020_0356\review_points\0002_review\files\`)
   vào `chuan_bi/phu_luc_1/ban_khoa_goc/`, chỉ khi sha256 bằng hash khóa gốc.
4. Phần phụ lục phải chứa nguyên văn 6 dòng lệnh ở PL1.5 và 7 dòng `set` biến môi trường; không còn dấu hiệu chưa xong.
5. Đòi một lần test trong `tests_eval/` đạt toàn bộ, ghi đúng hash mã hiện tại.
6. Ghi `chuan_bi/phu_luc_1/khoa_phu_luc_1.json`, `chuan_bi/phu_luc_1/SHA256SUMS`, `chuan_bi/phu_luc_1/thoi_diem_phu_luc_1.json`.
   Đã có thư mục `phu_luc_1/` → exit 1, không ghi đè.

`--buoc kiem-khoa` (mục 8, bước 1) giờ kiểm cả chuỗi. Không có thư mục phụ lục: kiểm như cũ, mọi file phải đúng hash
khóa gốc. Có phụ lục: `SHA256SUMS` gốc và `SHA256SUMS` phụ lục đều phải khớp; ba file đổi phải đúng hash phụ lục; bản
gốc giữ lại phải còn khớp hash khóa gốc. Phụ lục dở dang (thiếu `thoi_diem_phu_luc_1.json`) → exit 2. Test L0–L11 kiểm
các nhánh này trên fixture.

### PL1.3 F1 — fixture CLI của hai adapter gốc

`test_chuan_bi_danh_gia.py` nhóm C/R/F gọi **`main()` và parser của chính hai adapter**, dự đoán giả lập ở ranh giới
nạp model. Đối số là `doi_so_day_du(tap, model)`; test E1 kiểm chuỗi này nằm nguyên văn trong 6 lệnh mục 7. Test chỉ
thay giá trị checkpoint, sha256 và `--out` bằng fixture, rồi trỏ dữ liệu vào fixture tổng hợp:
- C1/C2: `--batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0` → exit 0. Đủ ảnh, đủ file đầu ra; D3 trên
  UA-DETRAC; năm lớp có GT trên BDD.
- C3: sai sha256 → exit 2, không dự đoán, không tạo thư mục. C4: `--out` đã có → exit 1, không đổi.
  C5: nội dung ảnh đổi → exit 2. Mỗi kiểm chạy cho cả hai adapter.
- Log và exit từng lần gọi: `tests_eval/lan_<…>/fixture_cli/log_cli/*.txt`.

### PL1.4 F2 — khi NPZ đã lưu nhưng bước chấm lỗi

Chung cho cả hai tập: không suy luận lại; không xóa, không ghi đè thư mục lỗi và thư mục giám sát.

**UA-DETRAC.** Adapter ghi `nguon_du_doan_<tên>.json` (có `npz_sha256`) ngay sau NPZ, trước khi chấm. Chấm lại bằng ba
lệnh `--cham-lai` ở mục 8, ra thư mục `__cham_lai` mới. Fixture R1–R3 kiểm:
- chấm lỗi → còn NPZ và nguồn, không có `ket_qua.json`;
- `--cham-lai` → exit 0, không dự đoán, D3 trùng lượt bình thường;
- NPZ bị sửa → exit 2.

**BDD val.** Adapter lưu NPZ rồi chấm; không ghi hash hay nguồn trước khi chấm, và không có `--cham-lai` (R6).
Quy trình, làm ngay khi `chay_giam_sat.py trang-thai` báo `ket_thuc` với exit khác 0, trước mọi thao tác khác:
1. Ghi bằng chứng NPZ bằng PowerShell, chỉ đọc NPZ, ghi vào thư mục mới `results\phase2\r53_detector\full\_hau_kiem\`:
   ```
   New-Item -ItemType Directory -Path results\phase2\r53_detector\full\_hau_kiem -Force
   Get-Item results\phase2\r53_detector\full\bdd_val\yolov8n_coco\du_doan_yolov8n_coco.npz | Select-Object FullName, Length, LastWriteTime | Format-List | Out-File results\phase2\r53_detector\full\_hau_kiem\bdd_val__yolov8n_coco__npz.txt -Encoding utf8
   Get-FileHash -Algorithm SHA256 results\phase2\r53_detector\full\bdd_val\yolov8n_coco\du_doan_yolov8n_coco.npz | Format-List | Out-File results\phase2\r53_detector\full\_hau_kiem\bdd_val__yolov8n_coco__npz.txt -Append -Encoding utf8
   ```
   Với `yolov8n_mixed5` và `yolo11n_mixed5`, thay tên model ở cả ba chỗ (thư mục, tên NPZ, tên file ghi). Nếu thư mục
   `_hau_kiem` đã có file cùng tên thì thêm hậu tố `__lan2`; không ghi đè.
2. Bằng chứng nguồn của NPZ:
   - `kiem-khoa` exit 0 trước khởi chạy;
   - dòng `# lenh:` trong `log.txt` của bộ giám sát, có checkpoint và sha256. Adapter đã kiểm sha256 checkpoint trước khi
     dự đoán (sai → exit 2);
   - dòng `[du lieu] 39973 anh, 200 video, …` (in sau cổng dữ liệu, trước suy luận), traceback và `# exit=`;
   - PID và create_time trong `trang_thai.json`;
   - `LastWriteTime` của NPZ nằm giữa dòng `[du lieu]` và lúc kết thúc.
3. Chấm lại. Fixture R4–R5 và F4 cho thấy: NPZ còn nguyên; tên ảnh trong NPZ đúng thứ tự adapter; chấm lại bằng hàm cũ
   (`bv.kiem_va_doc_gt` → `dg.doc_du_doan` → `eval_detector_5lop.danh_gia` với `bv.quy_tac_hieu_luc(0.25)`) **trùng
   từng trường** với lượt bình thường. **Chưa có lệnh chạy bước này trên dữ liệu thật.** Người dùng đã từ chối đề xuất
   P1517bbbd (công cụ hậu kiểm chấm lại BDD từ NPZ), nên lượt này không tạo công cụ đó. Vì vậy, khi BDD lỗi ở bước
   chấm: dừng sau bước 2 và báo người dùng. Người dùng chọn: (a) duyệt một công cụ chấm lại; hoặc (b) cho phép chạy
   lại adapter vào thư mục mới `__lan2`. Phương án (b) là suy luận lại, khoảng 18–20 phút GPU cho mỗi model (mục 9).
   Không tự chọn.
4. Mục 10 tiêu chí 7 (chấm lại từ NPZ để đối chiếu): UA-DETRAC làm được bằng `--cham-lai`. Với BDD, chưa có công cụ
   thực thi vì cùng quyết định trên. Người dùng cần quyết trước lượt dài: duyệt công cụ, hoặc bỏ tiêu chí 7 cho BDD.

### PL1.5 F3 — môi trường của tiến trình khởi chạy độc lập

WMI `Win32_Process.Create` cấp cho tiến trình môi trường chuẩn của hệ thống, không kế thừa biến của phiên đang mở. Vì
vậy biến phải được đặt **ngay trong dòng lệnh được khởi chạy**. `cmd.exe` đặt 7 biến sau:

```
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "YOLO_CONFIG_DIR=D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien"
set "MPLCONFIGDIR=D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien"
set "YOLO_OFFLINE=1"
```

Sau đó dòng lệnh tạo thư mục cấu hình nếu chưa có, rồi chạy bộ giám sát và adapter. Cả hai đều chạy bằng cùng python,
có `-B`. Bộ giám sát truyền nguyên môi trường cho adapter. Thư mục cấu hình phải được tạo **trước** khi python chạy:
test F2 vòng này cho thấy ultralytics gặp `YOLO_CONFIG_DIR` chưa tồn tại thì lặng lẽ dùng `<cwd>\Ultralytics\`. Lần
chạy test lúc 05:13 vì thế đã tạo `D:\LABOTARY\bytetrack\Ultralytics\settings.json` ở gốc repo, ngoài vùng ghi. File
này giữ nguyên, chờ người dùng quyết định xóa. Đã sửa bằng bước `mkdir` trong dòng lệnh; test F2 sau sửa thấy cấu hình
nằm trong `<YOLO_CONFIG_DIR>\Ultralytics\`.

Sáu dòng `CommandLine`, khởi chạy với `CurrentDirectory = D:\LABOTARY\bytetrack`, tuần tự theo thứ tự mục 8. In lại
bằng bước chỉ đọc `--buoc lenh-day-du`.

```
cmd.exe /d /c set "PYTHONDONTWRITEBYTECODE=1" && set "PYTHONUTF8=1" && set "PYTHONIOENCODING=utf-8" && set "PYTHONUNBUFFERED=1" && set "YOLO_CONFIG_DIR=D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien" && set "MPLCONFIGDIR=D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien" && set "YOLO_OFFLINE=1" && (if not exist "D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien\" mkdir "D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien") && "C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe" -B -u src\chay_giam_sat.py chay --thu-muc results\phase2\r53_detector\full\_giam_sat\detrac_test__yolov8n_coco --khong-in -- "C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe" -B -u src\danh_gia_detector_detrac_test.py --model yolov8n_coco=models/yolov8n.pt --sha256 yolov8n_coco=f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/detrac_test/yolov8n_coco
cmd.exe /d /c set "PYTHONDONTWRITEBYTECODE=1" && set "PYTHONUTF8=1" && set "PYTHONIOENCODING=utf-8" && set "PYTHONUNBUFFERED=1" && set "YOLO_CONFIG_DIR=D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien" && set "MPLCONFIGDIR=D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien" && set "YOLO_OFFLINE=1" && (if not exist "D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien\" mkdir "D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien") && "C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe" -B -u src\chay_giam_sat.py chay --thu-muc results\phase2\r53_detector\full\_giam_sat\detrac_test__yolov8n_mixed5 --khong-in -- "C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe" -B -u src\danh_gia_detector_detrac_test.py --model yolov8n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/best.pt --sha256 yolov8n_mixed5=542b1381c33a673531bccdbca769f26408a16e9ba6ed89e49238a0f16b9b0694 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/detrac_test/yolov8n_mixed5
cmd.exe /d /c set "PYTHONDONTWRITEBYTECODE=1" && set "PYTHONUTF8=1" && set "PYTHONIOENCODING=utf-8" && set "PYTHONUNBUFFERED=1" && set "YOLO_CONFIG_DIR=D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien" && set "MPLCONFIGDIR=D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien" && set "YOLO_OFFLINE=1" && (if not exist "D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien\" mkdir "D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien") && "C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe" -B -u src\chay_giam_sat.py chay --thu-muc results\phase2\r53_detector\full\_giam_sat\detrac_test__yolo11n_mixed5 --khong-in -- "C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe" -B -u src\danh_gia_detector_detrac_test.py --model yolo11n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt --sha256 yolo11n_mixed5=9a5e34a12c1d97cd1b0e49325d2eadbd443a965633b464461980df91fea386fb --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/detrac_test/yolo11n_mixed5
cmd.exe /d /c set "PYTHONDONTWRITEBYTECODE=1" && set "PYTHONUTF8=1" && set "PYTHONIOENCODING=utf-8" && set "PYTHONUNBUFFERED=1" && set "YOLO_CONFIG_DIR=D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien" && set "MPLCONFIGDIR=D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien" && set "YOLO_OFFLINE=1" && (if not exist "D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien\" mkdir "D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien") && "C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe" -B -u src\chay_giam_sat.py chay --thu-muc results\phase2\r53_detector\full\_giam_sat\bdd_val__yolov8n_coco --khong-in -- "C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe" -B -u src\danh_gia_detector_bdd_val.py --model yolov8n_coco=models/yolov8n.pt --sha256 yolov8n_coco=f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/bdd_val/yolov8n_coco
cmd.exe /d /c set "PYTHONDONTWRITEBYTECODE=1" && set "PYTHONUTF8=1" && set "PYTHONIOENCODING=utf-8" && set "PYTHONUNBUFFERED=1" && set "YOLO_CONFIG_DIR=D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien" && set "MPLCONFIGDIR=D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien" && set "YOLO_OFFLINE=1" && (if not exist "D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien\" mkdir "D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien") && "C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe" -B -u src\chay_giam_sat.py chay --thu-muc results\phase2\r53_detector\full\_giam_sat\bdd_val__yolov8n_mixed5 --khong-in -- "C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe" -B -u src\danh_gia_detector_bdd_val.py --model yolov8n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/best.pt --sha256 yolov8n_mixed5=542b1381c33a673531bccdbca769f26408a16e9ba6ed89e49238a0f16b9b0694 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/bdd_val/yolov8n_mixed5
cmd.exe /d /c set "PYTHONDONTWRITEBYTECODE=1" && set "PYTHONUTF8=1" && set "PYTHONIOENCODING=utf-8" && set "PYTHONUNBUFFERED=1" && set "YOLO_CONFIG_DIR=D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien" && set "MPLCONFIGDIR=D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien" && set "YOLO_OFFLINE=1" && (if not exist "D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien\" mkdir "D:\LABOTARY\bytetrack\results\phase2\r53_detector\full\_cau_hinh_thu_vien") && "C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe" -B -u src\chay_giam_sat.py chay --thu-muc results\phase2\r53_detector\full\_giam_sat\bdd_val__yolo11n_mixed5 --khong-in -- "C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe" -B -u src\danh_gia_detector_bdd_val.py --model yolo11n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt --sha256 yolo11n_mixed5=9a5e34a12c1d97cd1b0e49325d2eadbd443a965633b464461980df91fea386fb --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/bdd_val/yolo11n_mixed5
```

Khởi chạy bằng WMI, cùng cơ chế `handoff/logs/review43/khoi_chay_danh_gia_wmi.ps1`. Gán cho `$cmd` nguyên văn một dòng
ở trên, bọc trong dấu nháy đơn; các dòng không chứa dấu nháy đơn.

```
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = 'D:\LABOTARY\bytetrack'; ProcessStartupInformation = $si }
"ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
```

Kiểm sau khởi chạy:
- `trang_thai.json` của thư mục giám sát có PID và create_time;
- `results\phase2\r53_detector\full\_cau_hinh_thu_vien\Ultralytics\settings.json` xuất hiện;
- không có `.pyc` mới trong `src\`.

Fixture F1–F4 chạy đúng chuỗi này: cùng hàm `dong_lenh_doc_lap`; môi trường đã bỏ 7 biến; `cmd.exe` → `chay_giam_sat`
→ tiến trình adapter giả lập. Riêng tiến trình con là test ở chế độ `--con-adapter` thay cho adapter; không có WMI.
Kết quả:
- bytecode tắt, cả cờ và biến;
- UTF-8 bật;
- 7 biến đúng giá trị;
- cấu hình ultralytics nằm trong vùng đầu ra;
- không có `.pyc` mới;
- chấm lỗi → exit 1 thật qua bộ giám sát, log giữ traceback.

### PL1.6 Lệnh đã chạy trong vòng sửa (không GPU, không dữ liệu thật ngoài kiểm khóa)

| Lệnh | Exit | Bằng chứng |
|:--|:--|:--|
| `test_chuan_bi_danh_gia.py` lần 1 (05:13) | 1 (73/74; F2 lộ lỗi `YOLO_CONFIG_DIR`) | `tests_eval/lan_20261020_051353_550120_pid20388/` |
| `test_chuan_bi_danh_gia.py` lần 2 (05:14) | 1 (73/74; điều kiện F2 sai dạng thư mục con) | `tests_eval/lan_20261020_051447_323753_pid13972/` |
| `test_chuan_bi_danh_gia.py` lần 3 (05:15) | 0 (74/74), trên mã trước thay đổi in lệnh | `tests_eval/lan_20261020_051506_648198_pid2788/` |
| `chuan_bi_danh_gia.py --buoc kiem-khoa` trước phụ lục | 2 (`chuan_bi_danh_gia.py: sha256 khac khoa`) — cổng vẫn chặn | stdout |
| `chuan_bi_danh_gia.py --buoc lenh-day-du` | 0 | stdout (6 dòng PL1.5) |
| `test_chuan_bi_danh_gia.py` lần 4 (05:16), mã cuối | 0 (74/74) | `tests_eval/lan_20261020_051609_986864_pid5104/` |

Bước `phu-luc` và `kiem-khoa` sau phụ lục: xem báo cáo 03.

## Phụ lục 2 — chấm lại BDD val từ NPZ (bước hẹp Pdaa0fd9d, đóng F2), 2026-10-20

Phần trên mốc `## Phụ lục 2` giữ nguyên **từng byte** của bản ở phụ lục 1 (sha256
`5e27b4b53589d8c12ce38057b55f1eaecc59477c1f0ab4f39364bc79cbe5ddb5`); phần trên mốc `## Phụ lục 1` vẫn là bản khóa gốc
`7d1f42e37c9cf4dae48d0a2ba0eddd9e46350e1312add80125ace5e1df24e2f5`. Phụ lục 2 **không** đổi quy tắc (mục 4), ba
checkpoint (mục 1), dữ liệu (mục 2), mẫu smoke (mục 5), sáu lệnh đầy đủ (mục 7), sáu dòng khởi chạy độc lập (PL1.5),
khóa gốc, phụ lục 1 hay bằng chứng smoke. Không kiểm DETRAC trong bước này (UA-DETRAC dùng `--cham-lai` của adapter,
PL1.4). Không lập bảng so sánh detector: đề xuất P1517bbbd vẫn bị từ chối.

### PL2.1 Lý do và phạm vi

- Review Codex R53B vòng 2 (`review_002.json`) giữ F2 mở: BDD val chưa có cách thực thi tiêu chí 7 (mục 10) khi NPZ đã
  lưu mà bước chấm lỗi. Vòng 3 dừng ở trạng thái **blocked** vì hết ngân sách 3/3 vòng sửa-review.
- Người dùng (Codex chọn thay theo ủy quyền khi ngủ) **duyệt bước hẹp Pdaa0fd9d**, kèm ngân sách bổ sung; ngân sách
  hiệu lực của bước này là vòng 4/4. Đề xuất cấp thêm 2 vòng (P56df4991) bị từ chối.
- Thêm đúng một thao tác chỉ đọc vào `src/r53_detector/chuan_bi_danh_gia.py`: `--buoc cham-lai-bdd --model <tên>`,
  cùng bước khóa `--buoc phu-luc-2`. Không sửa adapter, scorer, `detector_classes.py`, `fingerprint_detrac_test.py`.

### PL2.2 Thao tác `cham-lai-bdd` (chỉ đọc, không suy luận)

Đầu vào cố định theo tên model (không nhận đường dẫn tùy ý):
- thư mục full `results/phase2/r53_detector/full/bdd_val/<tên>/` và NPZ `du_doan_<tên>.npz` trong đó;
- bằng chứng hash NPZ `results/phase2/r53_detector/full/_hau_kiem/bdd_val__<tên>__npz.txt` (PL2.4);
- tóm tắt bộ giám sát `results/phase2/r53_detector/full/_giam_sat/bdd_val__<tên>/tom_tat.json`;
- `ket_qua.json` gốc của thư mục full nếu có (lượt chạy thường).

Thứ tự kiểm; lỗi ở bất kỳ bước nào → exit 2, không ghi `ket_qua.json`:
1. Cổng khóa theo chuỗi: khóa gốc → phụ lục 1 → phụ lục 2 (`kiem_khoa`); thiếu phụ lục 2 → exit 2.
2. Nguồn và hash NPZ: sha256 NPZ băm lại = dòng `Hash` trong bằng chứng, dòng `Path` trỏ đúng NPZ; lệnh trong
   `tom_tat.json` chạy đúng `src\danh_gia_detector_bdd_val.py` với **đúng** đối số bàn giao của model (`--model` = checkpoint
   đã khóa, `--sha256` = hash đã khóa, `--batch 16`, `--device 0`, `--conf-van-hanh 0.25`, `--gioi-han-anh 0`, `--out` =
   thư mục đang chấm; dữ liệu = cổng đầy đủ); không có đối số lạ. Nếu có `ket_qua.json` gốc: chỉ một model, đúng sha256,
   `gioi_han_anh` 0.
3. Cổng dữ liệu đầy đủ của adapter: `bv.kiem_va_doc_gt` (nhãn, ảnh, manifest, bảng hash ảnh); danh tính = khóa.
4. NPZ: `dg.doc_du_doan`; số ảnh và **tên + thứ tự** ảnh trong NPZ phải trùng thứ tự ảnh của adapter.
5. Quy tắc `bv.quy_tac_hieu_luc(0.25)` = khóa; chấm bằng `eval_detector_5lop.danh_gia` và `bv.chan_doan` (hàm cũ).
   Trong lúc chấm, `du_doan_ultralytics` của hai adapter bị thay bằng hàm chặn (gọi → lỗi); số lần gọi ghi vào kết quả.

Đầu ra: thư mục **mới** `results/phase2/r53_detector/full/bdd_val/<tên>__cham_lai/` (đã có → exit 1, không ghi đè;
chạy lại dùng `--lan N` → `<tên>__cham_lai_lanN`), gồm `log.txt`, `exit.json`, `ket_qua.json` (cùng cấu trúc với adapter
BDD, thêm `npz`, `npz_sha256` và khối `cham_lai`: hàm đã gọi, nguồn, exit gốc, sha256 mọi file đầu vào trước chấm,
`dau_vao_khong_doi`, `goi_suy_luan`), `theo_lop.csv` (cùng cột, cùng cách ghi với adapter). Hai trường adapter ghi lúc
suy luận không có trong NPZ nên không tái tạo: `anh_xa_lop`, `thoi_gian_s`. Có `ket_qua.json` gốc → so từng trường còn
lại; khác → exit 3. Đầu vào đổi trong lúc chấm → exit 3.

Hàm cũ được gọi (nguyên văn, không sửa): `danh_gia_detector_bdd_val.kiem_va_doc_gt`, `danh_gia_detector_detrac_test.doc_du_doan`,
`danh_gia_detector_bdd_val.quy_tac_hieu_luc(0.25)`, `eval_detector_5lop.danh_gia`, `danh_gia_detector_bdd_val.chan_doan`.

### PL2.3 Ba lệnh chấm lại BDD trên dữ liệu thật (chỉ bàn giao — KHÔNG chạy trong lượt này)

Chạy sau lượt đầy đủ, từng model, khi đã có NPZ, bằng chứng PL2.4 và `tom_tat.json`. Không cần GPU; mỗi lệnh chạy cổng dữ
liệu đầy đủ (82,1 s ở tiền kiểm BDD, `chuan_bi/tien_kiem_bdd_val.json`), cổng khóa, đọc NPZ và chấm 39.973 ảnh (phần chấm
chưa đo trên dữ liệu thật; không ước lượng thêm). Chạy foreground trong phiên được duyệt cho lượt dài:

```
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc cham-lai-bdd --model yolov8n_coco
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc cham-lai-bdd --model yolov8n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc cham-lai-bdd --model yolo11n_mixed5
```

Dùng cho cả hai trường hợp: (a) BDD chấm lỗi sau NPZ (exit gốc khác 0, không `ket_qua.json`) → đây là cách phục hồi,
không suy luận lại; (b) BDD chạy thường → đối chiếu tiêu chí 7 (mục 10), từng trường phải trùng. Việc này thay bước 3–4
của PL1.4 cho BDD; phương án (b) "chạy lại adapter vào `__lan2`" của PL1.4 chỉ còn dùng khi NPZ hỏng hoặc thiếu, và vẫn
cần người dùng duyệt riêng.

### PL2.4 Bằng chứng NPZ sau mỗi lệnh BDD (thay hai dòng PL1.4 bước 1)

Ngay khi bộ giám sát báo `ket_thuc` cho lệnh BDD của model (exit 0 **hoặc** khác 0), trước mọi thao tác khác, chạy hai
dòng PowerShell (CurrentDirectory = repo; tạo `_hau_kiem` bằng dòng `New-Item` của PL1.4 nếu chưa có). `-Width 500` để
dòng `Path` không bị ngắt; `-NoClobber` để không ghi đè. Nếu file đã có thì dòng đầu báo lỗi: dừng và báo người dùng,
không đổi tên file (bước chấm lại chỉ đọc đúng tên này).

```
Get-Item results\phase2\r53_detector\full\bdd_val\yolov8n_coco\du_doan_yolov8n_coco.npz | Select-Object FullName, Length, LastWriteTime | Format-List | Out-File results\phase2\r53_detector\full\_hau_kiem\bdd_val__yolov8n_coco__npz.txt -Encoding utf8 -Width 500 -NoClobber
Get-FileHash -Algorithm SHA256 results\phase2\r53_detector\full\bdd_val\yolov8n_coco\du_doan_yolov8n_coco.npz | Format-List | Out-File results\phase2\r53_detector\full\_hau_kiem\bdd_val__yolov8n_coco__npz.txt -Append -Encoding utf8 -Width 500
Get-Item results\phase2\r53_detector\full\bdd_val\yolov8n_mixed5\du_doan_yolov8n_mixed5.npz | Select-Object FullName, Length, LastWriteTime | Format-List | Out-File results\phase2\r53_detector\full\_hau_kiem\bdd_val__yolov8n_mixed5__npz.txt -Encoding utf8 -Width 500 -NoClobber
Get-FileHash -Algorithm SHA256 results\phase2\r53_detector\full\bdd_val\yolov8n_mixed5\du_doan_yolov8n_mixed5.npz | Format-List | Out-File results\phase2\r53_detector\full\_hau_kiem\bdd_val__yolov8n_mixed5__npz.txt -Append -Encoding utf8 -Width 500
Get-Item results\phase2\r53_detector\full\bdd_val\yolo11n_mixed5\du_doan_yolo11n_mixed5.npz | Select-Object FullName, Length, LastWriteTime | Format-List | Out-File results\phase2\r53_detector\full\_hau_kiem\bdd_val__yolo11n_mixed5__npz.txt -Encoding utf8 -Width 500 -NoClobber
Get-FileHash -Algorithm SHA256 results\phase2\r53_detector\full\bdd_val\yolo11n_mixed5\du_doan_yolo11n_mixed5.npz | Format-List | Out-File results\phase2\r53_detector\full\_hau_kiem\bdd_val__yolo11n_mixed5__npz.txt -Append -Encoding utf8 -Width 500
```

Giới hạn: bằng chứng này ghi **sau** khi adapter kết thúc, không phải trong tiến trình adapter. Nó chứng minh NPZ không đổi
từ lúc ghi bằng chứng tới lúc chấm lại; khoảng giữa lúc adapter ghi NPZ và lúc ghi bằng chứng được đối chiếu bằng
`LastWriteTime` với log giám sát (PL1.4 bước 2), không bằng hash.

### PL2.5 Khóa phụ lục 2 và phiên bản mã

`--buoc phu-luc-2` (chạy một lần, đã có → exit 1): kiểm khóa gốc và phụ lục 1 còn nguyên; hash "trước" của mỗi file = mức
gần nhất ghi nó (SHA256SUMS phụ lục 1, nếu không thì khóa gốc); chỉ ba file được đổi (tài liệu này — chỉ nối thêm sau mốc
`## Phụ lục 2`; hai file mã); bản phụ lục 1 của hai file mã lấy từ snapshot review của orchestrator (đúng hash) và chép vào
`chuan_bi/phu_luc_2/ban_khoa_truoc/`; đòi một lần test đạt trên đúng mã hiện tại; ghi `chuan_bi/phu_luc_2/`
(`khoa_phu_luc_2.json`, `SHA256SUMS`, `thoi_diem_phu_luc_2.json`). `kiem-khoa` (và `smoke`, `cham-lai-bdd`) kiểm cả chuỗi ba
mức. Không commit theo yêu cầu hiện tại.

Phiên bản mã khóa ở phụ lục 2 (sha256):
- `src/r53_detector/chuan_bi_danh_gia.py`: `535d53cca5cebac7c8d47179f715f77b284efca05b1316876a938d2878f756cf`
  (phụ lục 1: `6fc0c72a4d0bc34fc8c616f77dc44b47f949575f93152f4bc7812540249194e4`);
- `src/r53_detector/test_chuan_bi_danh_gia.py`: `8d705a88804e25fa339e0754c1f63ecfee72a344b3bd7f49ca51f9bcee2f3022`
  (phụ lục 1: `4e1f2235803151527d8985bb136874e6abc7225e1b2836ab2cdb392ca5622fbf`);
- adapter, scorer, `detector_classes.py`, `fingerprint_detrac_test.py`, `bdd_yolo_export.py`, `config.py`: không đổi, đúng
  hash mục 3.

### PL2.6 Fixture tổng hợp (test_chuan_bi_danh_gia.py nhóm P, X)

Thư mục full giả lập dựng bằng `main()` của adapter BDD gốc (dự đoán giả lập bám GT, AP fixture > 0 cả năm lớp; tọa độ
nguyên, conf biểu diễn đúng ở float32): `yolov8n_coco` chấm lỗi sau NPZ; hai model mixed5 chạy thường; `tom_tat.json`
tổng hợp theo dạng `chay_giam_sat`; bằng chứng hash theo dạng Format-List (BOM, CRLF). Kiểm:
- X0 chưa có phụ lục 2 → exit 2; X1 phục hồi sau lỗi chấm → exit 0, `ket_qua.json` trùng từng trường và `theo_lop.csv`
  trùng từng byte với lượt thường (trừ `anh_xa_lop`, `thoi_gian_s`); X2 không gọi dự đoán, đầu vào và dữ liệu không đổi;
- X3 đích đã có → exit 1; X4 thư mục chạy thường → đối chiếu không trường nào khác; X5 NPZ bị sửa → exit 2; X6 NPZ đảo
  thứ tự / thiếu ảnh (hash bằng chứng khớp) → exit 2; X7 khôi phục NPZ → thư mục mới `__lan4` exit 0; X8 lệnh giám sát
  sai `--gioi-han-anh` / `--sha256` / `--out` → exit 2; X9 thiếu bằng chứng → exit 2; X10 khóa hỏng → exit 2; X11 tham số;
- P1–P8: phụ lục 2 không đổi gì / thiếu bản phụ lục 1 / sửa phần đã khóa / thiếu chuỗi bắt buộc → exit 2; đạt → exit 0;
  tạo lại → exit 1; sửa mã, tài liệu 02, bản sao phụ lục 1 hoặc 2, SHA256SUMS phụ lục 1 → `kiem-khoa` exit 2; smoke vẫn qua
  cổng ba mức.

Kết quả test, `phu-luc-2`, `kiem-khoa`: xem báo cáo 03.

## Phụ lục 3 — sửa F2 [P2] sau review R53B vòng 4: cổng nguồn so đường dẫn adapter, 2026-10-20

Phần trên mốc `## Phụ lục 3` giữ nguyên **từng byte** của bản ở phụ lục 2 (sha256
`b18a7636e030a907ae34ead16a354dfcba3c2d89f2763bb3f73ee091953c3276`). Phụ lục 3 **không** đổi quy tắc, ba checkpoint, dữ
liệu, mẫu smoke, sáu lệnh đầy đủ, sáu dòng khởi chạy độc lập, ba lệnh chấm lại (PL2.3), sáu dòng bằng chứng NPZ (PL2.4),
khóa gốc, phụ lục 1, phụ lục 2 hay bằng chứng smoke. Vẫn không kiểm DETRAC, không lập bảng so sánh (P1517bbbd bị từ chối).

### PL3.1 Lý do

Review Codex R53B vòng 4 (`review_004.json`, blocked): `kiem_nguon_bdd` chỉ so **tên file** của adapter trong lệnh giám
sát, nên một `danh_gia_detector_bdd_val.py` ở bản checkout khác vẫn được chấp nhận; cổng nguồn chưa bảo đảm NPZ đến từ
adapter đã khóa. Lượt sửa này là vòng 5/5 của ngân sách bổ sung cho bước Pdaa0fd9d (vòng 3 = blocked do hết 3/3; người
dùng duyệt bước hẹp kèm ngân sách bổ sung; vòng 4 = blocked với F2 [P2]).

### PL3.2 Thay đổi trong `cham-lai-bdd` (bước 2 của PL2.2)

- Thư mục làm việc của lượt BDD lấy từ `full/_giam_sat/bdd_val__<tên>/trang_thai.json` (trường `cwd`, do
  `src/chay_giam_sat.py` ghi bằng `os.getcwd()`; tiến trình adapter kế thừa cwd này). `tom_tat.json` không có `cwd`.
  Thiếu `trang_thai.json`, thiếu `cwd` tuyệt đối, hoặc `lenh` của hai file khác nhau → exit 2. Không còn ngầm lấy cwd = repo.
- Phần tử lệnh trùng tên adapter BDD được resolve (tương đối → theo `cwd` trên) và phải **bằng** đường dẫn đã resolve của
  `src/danh_gia_detector_bdd_val.py` trong repo này; khác (kể cả bản sao nguyên byte ở checkout khác) → exit 2. Nội dung
  file adapter trong repo do cổng khóa kiểm (SHA256SUMS gốc). Các đối số đường dẫn khác (`--model`, `--out`, dữ liệu)
  cũng resolve theo cùng `cwd`.
- `ket_qua.json` chấm lại ghi thêm `cham_lai.adapter_goc` và `cham_lai.cwd_giam_sat`.
- Lượt dài không đổi: chạy WMI với CurrentDirectory = repo như mục 8 / PL1.5; bộ giám sát tự ghi `trang_thai.json`.

Ba lệnh chấm lại vẫn nguyên văn PL2.3 (chỉ bàn giao, **không** chạy trong lượt này):

```
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc cham-lai-bdd --model yolov8n_coco
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc cham-lai-bdd --model yolov8n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc cham-lai-bdd --model yolo11n_mixed5
```

### PL3.3 Khóa phụ lục 3 và phiên bản mã

`--buoc phu-luc-3` (một lần, đã có → exit 1) dùng cùng cơ chế với `phu-luc-2` (mã gộp thành `buoc_phu_luc_noi(ch, k)`):
kiểm khóa gốc, phụ lục 1, phụ lục 2 còn nguyên và mọi bản trước giữ lại vẫn đúng hash; hash "trước" = mức gần nhất ghi
file; chỉ ba file được đổi (tài liệu này, chỉ nối thêm sau mốc `## Phụ lục 3`, và hai file mã); bản phụ lục 2 của hai
file mã lấy từ snapshot review vòng 4 của orchestrator (đúng hash) và chép vào `chuan_bi/phu_luc_3/ban_khoa_truoc/`; cần
một lần test đạt trên đúng mã hiện tại. `kiem-khoa`, `smoke`, `cham-lai-bdd` kiểm cả chuỗi bốn mức. Không commit theo
yêu cầu hiện tại.

Phiên bản mã khóa ở phụ lục 3 (sha256):
- `src/r53_detector/chuan_bi_danh_gia.py`: `dfa2665dddb9363df05266542ec7321a4f588011fe26d3a0c3143394eb6163dc`
  (phụ lục 2: `535d53cca5cebac7c8d47179f715f77b284efca05b1316876a938d2878f756cf`);
- `src/r53_detector/test_chuan_bi_danh_gia.py`: `2c59a6e49b471545a303588a1b92fea6daff7832812dd1c1058b84da0acc0a12`
  (phụ lục 2: `8d705a88804e25fa339e0754c1f63ecfee72a344b3bd7f49ca51f9bcee2f3022`);
- adapter, scorer, `detector_classes.py`, `fingerprint_detrac_test.py`, `bdd_yolo_export.py`, `config.py`: không đổi.

### PL3.4 Fixture tổng hợp (nhóm X12, Q)

- X12, qua CLI `main()` của wrapper: bản sao nguyên byte của adapter BDD ở `checkout_khac/src/` gọi bằng đường dẫn tuyệt
  đối → exit 2; đường dẫn tương đối `src\danh_gia_detector_bdd_val.py` nhưng `cwd` giám sát = checkout khác → exit 2;
  `trang_thai.json` thiếu `cwd` / thiếu file / ghi lệnh khác `tom_tat.json` → exit 2; đường dẫn tuyệt đối đúng adapter
  trong repo → exit 0. Không ca nào gọi dự đoán; ca lỗi không có `ket_qua.json`.
- Q1–Q8: phụ lục 3 không đổi gì / thiếu bản phụ lục 2 trong snapshot / sửa phần trước mốc / thiếu chuỗi → exit 2; đạt →
  exit 0 (bản trước = hash phụ lục 2); tạo lại phụ lục 3 hoặc 2 → exit 1; sửa mã, tài liệu 02, bản sao phụ lục 3,
  SHA256SUMS phụ lục 2 → `kiem-khoa` exit 2; `cham-lai-bdd` sau phụ lục 3 → exit 0, chuỗi khóa bốn mức, cùng số liệu.
- Fixture `tom_tat.json` / `trang_thai.json` nay theo đúng dạng `chay_giam_sat` (cwd chỉ ở `trang_thai.json`).

Kết quả test, `phu-luc-3`, `kiem-khoa`: xem báo cáo 03.
