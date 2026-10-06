# Vòng 43 — ghi nhận nghiệm thu `4d54744`; đính chính R42-N1/N2/N3; đánh giá detector trên BDD100K MOT val (chi tiết)

**Phạm vi** (người dùng duyệt 2026-10-06):
- Đánh giá detector trên PC local cho **ba checkpoint**: baseline YOLOv8n COCO; `best.pt` của YOLOv8n và YOLO11n đã
  fine-tune ở vòng 42.
- Hai tập: **BDD100K MOT val** (COCO-video, 200 video / 39.973 ảnh) và **UA-DETRAC test** (manifest khóa).
- Không train thêm, không tuning theo test, không SSH Nano. Giữ nguyên split, dataset, checkpoint, cấu hình tracking và
  kết quả lịch sử.
- Báo cáo rồi dừng; chưa so sánh bộ lọc, thuật toán tracking hay tối ưu Nano.
- Base review: **`4d54744`**. Vòng 43 **chưa** được nghiệm thu.

**Trạng thái**:
- **BDD val: ĐÃ CHẠY** 3/3 lượt; cả ba đạt tiêu chí lượt hợp lệ của kế hoạch khóa (exit 0, 39.973/39.973 ảnh, kiểm dữ
  liệu và checkpoint đạt).
- **UA-DETRAC test: CHƯA CHẠY**, bị chặn bởi quyết định ánh xạ lớp của người dùng (mục 8).

## 0. Trạng thái đầu vòng

Log: `logs/review43/00_trang_thai_dau.log`.
- HEAD `4d54744` (target vòng 42). Working tree chỉ có file riêng của người dùng.
- Môi trường: Python 3.11.9, ultralytics 8.4.141 (`trainer.py` `7532ff0b…`), torch 2.6.0+cu124, CUDA 12.4,
  cuDNN 90100, numpy 2.4.6, OpenCV 5.0.0; GTX 1650 4 GiB, driver 555.97.
- Nguồn: cắm sạc (pin 100 %), gói High performance, sleep / hibernate = không bao giờ; không có khởi động lại đang chờ.
- RAM trống 0,96 GiB / 7,84 GiB.

## 1. Ghi nhận nghiệm thu `4d54744` (commit `4fa1839`)

- **Báo cáo** `handoff/CODEX_REVIEW_4d54744.md`: trùng byte bản trong `D:\LABOTARY\review_codex` (11.336 B, sha256
  `45161d67…`).
- **Kết luận**: **PASS WITH LIMITATIONS**, chỉ trong phạm vi **hồ sơ training** vòng 42. Đây không phải nghiệm thu chất
  lượng trên test.
- **Giới hạn ghi đúng như review**:
  - reviewer không nạp checkpoint, không chạy training / inference / test, không kiểm lại ảnh / nhãn;
  - không kiểm toàn cây Git;
  - `audit_training.py` 128/128 là phép đối chiếu hồ sơ, không phải test của dự án;
  - nguyên nhân NaN chưa xác định; guard ghi nối / khóa lượt vẫn là đề xuất.
- **Bằng chứng**: `CODEX_REVIEW_4d54744_EVIDENCE.zip` 33.018 B, sha256 `83aedc5b…`. Giải nén vào thư mục mới
  `handoff/review_evidence/CODEX_4d54744/`, ZIP gốc giữ nguyên. `SHA256SUMS` khớp **10/10** trên đĩa và trên blob đã
  stage (`-text`).
- **Gói đã nghiệm thu**: 741.857 B, `f144a834…`, khớp mục 1 của review. Lưu ở `handoff/archive/review_delta_4d54744.zip`.
- **Base review** `6c8e37a` → **`4d54744`** (vòng 42); giữ đủ 13 khai báo đầu vào không bàn giao.

## 2. Đính chính R42-N1 / N2 / N3 — chỉ câu chữ và bảng trình bày (commit `4fa1839`)

Sửa trong `REVIEW42_CHI_TIET.md`, có ghi chú "Đính chính vòng 43" ở đầu. CSV và log giữ nguyên.
- **R42-N1**: RAM trống thấp nhất ghi theo **từng nguồn**:
  - YOLO11n: 81 MiB theo bộ giám sát, 6 MiB theo bộ lấy mẫu trong runner;
  - YOLOv8n, ba lần chạy: 22 / 5 / 162 MiB theo bộ giám sát, 27 / 23 / 73 MiB theo runner.

  VRAM 2.933 / 3.123 MiB là `nvidia-smi memory.used` của **cả GPU**; tổng RSS có thể đếm trùng trang bộ nhớ chia sẻ.
- **R42-N2**: NaN từ batch 4597/6311 (≈73 %). Dòng tiến độ cuối trước khi dừng là 4786/6311 (≈76 %); dòng này không phải
  bằng chứng batch cuối đã thực thi. Epoch 8: dòng tiến độ cuối 6027/6311 (≈95 %).
- **R42-N3**: sửa 5 ô — YOLOv8n e2 mAP50-95 0,176; e4 / e7 mAP50 0,350 / 0,393; YOLO11n e1 / e7 mAP50 0,229 / 0,373.
  Cả 4 bảng epoch (2 model × 2 chỉ số × 15 epoch) được đối chiếu lại, khớp `results.csv` làm tròn một lần.
- Bỏ câu "log RAM liên tục mỗi giây", vì CSV thô không nằm trong gói.

## 3. Kiểm trực tiếp ba checkpoint (commit `1df012f`)

Script `logs/review43/kiem_checkpoint.py` → `01_kiem_checkpoint.{log,json}`, exit 0. Chỉ đọc file.

| Tên | File | sha256 tính từ file | Kiểu, dtype | Tham số | Tensor hữu hạn | Lớp |
|:--|:--|:--|:--|--:|--:|:--|
| `yolov8n_coco` | `models/yolov8n.pt` | `f59b3d83…` khớp | DetectionModel, FP16 | 3.157.200 | 298/298 | COCO 80; ánh xạ theo tên: 2 car, 7 truck, 5 bus, 3 motorcycle, 1 bicycle |
| `yolov8n_ft` | `…/train_bdd5_v1_yolov8n_day_du/weights/best.pt` | `5f66776d…` khớp | DetectionModel, FP16 | 3.011.823 | 298/298 | 5 lớp: 0 car, 1 truck, 2 bus, 3 motorcycle, 4 bicycle |
| `yolo11n_ft` | `…/train_bdd5_v1_yolo11n_day_du/weights/best.pt` | `d7d1c7e8…` khớp | DetectionModel, FP16 | 2.590.815 | 418/418 | như trên |

- Nạp bằng hàm sẵn có `tiep_tuc_huan_luyen.doc_checkpoint` (CPU).
- Tensor hữu hạn: `torch.isfinite` trên mọi tensor số thực của `model.state_dict()`. Đây là bước mới; dự án chưa có công
  cụ này.
- Tên lớp: `ultralytics.YOLO(path).names`, cùng cách evaluator đọc, cộng `detector_classes.loai_checkpoint` /
  `mo_ta_anh_xa`.
- Không dùng `last.pt` hay checkpoint khác.

## 4. Điểm vào đánh giá BDD val — sửa lỗi kỹ thuật chặn đánh giá (commit `1df012f`)

**Tái hiện điểm chặn** (`02_tai_hien_chan_bdd_val.log`): evaluator đã nghiệm thu `eval_detector_5lop.py` chỉ đọc được
hợp đồng dataset train / dev.
- `--split test` → `[LOI DATASET] split 'test' khong thuoc hop dong`, exit 2, không tạo thư mục ra.
- Chỉ có `data.yaml` của train / dev. BDD val chỉ có ảnh gốc và một JSON COCO-video.
- `chuan_bi_danh_gia()` chỉ ánh xạ `{'val': 'dev', 'train': 'train'}`.

**Sửa tối thiểu**: `src/danh_gia_detector_bdd_val.py` (mới, sha256 `d1e2fd22…`) **chỉ thêm phần đọc GT**.
- GT theo đúng quy tắc exporter 5 lớp: `NguonCocoVid` → `vai_tro` → `_hop_le_box`, import trực tiếp từ
  `bdd_yolo_export`, không chép lại.
- Kiểm **trước** khi dự đoán; sai thì exit 2 và không dự đoán:
  - sha256 checkpoint (`--sha256` bắt buộc);
  - sha256 nhãn = manifest; tập video = manifest; không có ảnh liệt kê trùng;
  - tập ảnh từ nhãn = tập `.jpg` trên đĩa; w/h trong header ảnh = w/h của nhãn;
  - sha256 nội dung **từng ảnh** = bảng hash đã lưu.
- Dự đoán, ghép, AP, điểm vận hành và lưu dự đoán dùng **nguyên hàm** của evaluator: `du_doan_ultralytics`, `danh_gia`,
  `luu_du_doan`, `QUY_TAC`.
- Không tô xám ảnh val; giữ ảnh không có annotation. Thư mục ra đã tồn tại → exit 1, không ghi đè.
- Mã cũ không sửa: `eval_detector_5lop.py` `30a1fd5a…`, `bdd_yolo_export.py` `1ca964b0…`.

**Test hồi quy** `src/test_danh_gia_bdd_val.py` (mới): **19/19 PASS** (`03_test_danh_gia_bdd_val.log`).
- B1a–f: quy tắc GT và số đếm trên fixture 2 video × 3 frame.
- **B2**: GT đọc trực tiếp **trùng** đường exporter → evaluator đã nghiệm thu, 6/6 ảnh.
- B3a–f: 6 ca từ chối — sha nhãn, tập video, ảnh thừa, ảnh thiếu, w/h, nội dung ảnh.
- B4a–f: `main()` với dự đoán giả lập. Metric trùng `danh_gia` gọi trực tiếp, mAP50 = 1,0 > 0. Các ca exit 1 / 2.

Suite liên quan `test_detector_5lop`: **92/92** (`03_test_detector_5lop.log`).

## 5. Kế hoạch khóa và khởi chạy (commit `1df012f`, `b9ea1d8`)

- `logs/review43/04_ke_hoach_danh_gia.md` được commit **trước** lượt đánh giá đầu tiên trên dữ liệu thật, và không bị
  sửa sau đó (`git diff 1df012f HEAD -- handoff/logs/review43/04_ke_hoach_danh_gia.md` rỗng). Phụ lục mục 10 vẫn trống:
  không có sửa đổi giao thức nào.
- Nội dung kế hoạch: checkpoint + hash; dữ liệu và fingerprint; quy tắc GT / ánh xạ lớp / crowd / ignore; thiết lập dự
  đoán và chấm; cách báo cáo; cách chạy; UA-DETRAC chờ quyết định; tiêu chí lượt hợp lệ.
- `logs/review43/khoi_chay_danh_gia_wmi.ps1`:
  - checkpoint và sha256 cố định theo kế hoạch; không nhận đường dẫn tùy ý;
  - từ chối khi thư mục ra đã tồn tại, hoặc đang có tiến trình đánh giá / train;
  - WMI `Win32_Process.Create` (cha `WmiPrvSE`, cửa sổ ẩn) → `chay_giam_sat.py chay` → điểm vào. Cùng cơ chế vòng 42.
- Thiết lập **hiệu lực** (ghi trong `ket_qua.json` của mỗi lượt): `imgsz 640`, `conf 0.001`, NMS IoU `0.7` theo lớp,
  `max_det 300`, batch 16, `device 0`. Tham số `half=False` sinh cảnh báo "'half' is deprecated" ở mỗi lần gọi predict
  (2.499 dòng/lượt = ⌈39.973/16⌉ batch). Trong ultralytics 8.4.141, `half=False` được chuyển thành `quantize=None`
  (`cfg/__init__.py` dòng 590–596), dự đoán dùng `fp16 = (quantize == 16)` = False (`engine/predictor.py` dòng 436),
  và backend PyTorch gọi `model.float()` khi `fp16` là False (`nn/backends/pytorch.py` dòng 68). Vì vậy dự đoán chạy
  **FP32**, đúng kế hoạch, dù checkpoint lưu FP16. Đây là đọc mã nguồn ultralytics, không phải phép đo riêng.

## 6. Ba lượt chạy BDD val (commit `6dc5b29`)

Chạy tuần tự theo kế hoạch: `yolov8n_coco` → `yolov8n_ft` → `yolo11n_ft`. Mỗi lượt một model, vào thư mục mới. Không có
smoke trên dữ liệu thật, không chạy lại lượt nào, không đổi tham số.

| Lượt | Log | Bắt đầu → kết thúc (2026-10-06) | Thời gian (s) | Exit | Ảnh đã dự đoán | Kiểm dữ liệu (s) | RSS tối đa (MiB) | RAM máy trống thấp nhất (MiB) |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| `yolov8n_coco` | `05`, `06` | 20:27:33 → 20:46:36 | 1.143,1 | **0** | 39.973/39.973 | 129 | 941,5 | 50 |
| `yolov8n_ft` | `07`, `08` | 20:47:16 → 21:07:22 | 1.206,4 | **0** | 39.973/39.973 | 122 | 971,0 | 130 |
| `yolo11n_ft` | `09`, `10` | 21:08:08 → 21:27:49 | 1.181,4 | **0** | 39.973/39.973 | 129 | 1.126,1 | 122 |

RSS là working set của tiến trình đánh giá; RAM máy trống là `psutil.virtual_memory().available` theo bộ giám sát (mẫu
1 s). Hai số này lấy từ `tom_tat.json` của từng lượt.

**Kiểm hợp lệ** (`tong_hop_danh_gia.py` → `11_tong_hop_kiem_hop_le.log`, `ket_qua/kiem_hop_le.json`): **39/39 phép kiểm
đạt**.
- **Tiêu chí mục 9** của kế hoạch, cho từng lượt:
  - exit code thật 0 theo bộ giám sát, không có Ctrl+C;
  - 39.973 ảnh dự kiến = 39.973 ảnh đã dự đoán, `gioi_han_anh` 0;
  - kiểm dữ liệu đạt: sha256 nhãn `074ff795…` = manifest `2071c15b…`, 200 video, tập ảnh = đĩa, w/h, nội dung ảnh
    39.973/39.973 khớp bảng hash `60fea2a9…`, 0 lệch;
  - sha256 checkpoint và loại ánh xạ lớp đúng mục 2.
- **Lệnh thật** (`tom_tat.json`, `ket_qua.json`) đúng mẫu khóa, không có tham số phụ: không `--gioi-han-anh`; batch và
  device dùng mặc định 16 / `0`.
- **Giống nhau giữa ba lượt**: `quy_tac` (dự đoán, ghép, bỏ qua, AP, vận hành), `quy_tac_gt` và toàn bộ
  `dataset_da_xac_minh` (mọi sha256, mọi số đếm). Riêng `quy_tac.du_doan` và `van_hanh` còn được so với giá trị mục 5.
- **Số đếm GT** (giống nhau ở ba lượt; khớp thống kê ở mục 3 của kế hoạch):
  - GT: car 341.436, truck 27.343, bus 9.087, motorcycle 902, bicycle 4.135;
  - 20.848 vùng bỏ qua trên 15.720 ảnh;
  - 615 ảnh không có annotation; 778 ảnh không có GT 5 lớp;
  - 61.761 annotation lớp khác (pedestrian, rider, train, other person);
  - 17.980 box bị cắt vào khung; 0 box < 1 px sau khi cắt.
- `theo_lop.csv` == `ket_qua.json` (đọc lại số thực, so bằng `==`).
- sha256 mã đánh giá lúc tổng hợp == lúc trước lượt đầu (`05`).
- **Tái tính từ dự đoán đã lưu**: đọc lại GT (không băm lại nội dung) và `du_doan_<ten>.npz`, rồi gọi lại `danh_gia` /
  `chan_doan` của evaluator. Kết quả **trùng tuyệt đối** (`==`) mọi số trong `ket_qua.json` của cả ba model. Vậy NPZ đủ
  để tái tính metric.

Lần chạy đầu của bước tổng hợp bị tôi dừng sau khoảng 19 phút (`11a_tong_hop_lan1_dung_vi_cham.log`). Nguyên nhân là lỗi
hiệu năng trong script tổng hợp: script truy cập mảng `NpzFile` trong vòng lặp từng ảnh, nên mảng bị giải nén lại 39.973
lần. Tôi đã sửa và chạy lại. Lỗi này không ảnh hưởng kết quả đánh giá, vì script tổng hợp chỉ đọc.

## 7. Kết quả BDD100K MOT val

Bảng sinh tự động từ `ket_qua.json` (`ket_qua/bang_ket_qua.md`; số đầy đủ trong `bang_theo_lop.csv`, `bang_macro.csv`).
Ba model dùng **cùng ảnh, cùng GT, cùng thiết lập**.

**AP theo lớp** — BDD100K MOT val (COCO-video), 39.973 ảnh, 200 video. AP nội suy 101 điểm; AP50–95 = trung bình 10 ngưỡng IoU 0,50:0,05:0,95. Chênh lệch Δ = model − baseline `yolov8n_coco`, tính bằng **điểm AP** (×100).

| Lớp | Số GT | Số ảnh có lớp | Model | AP50 | AP50–95 | ΔAP50 | ΔAP50–95 |
|:--|--:|--:|:--|--:|--:|--:|--:|
| car | 341.436 | 38.964 | `yolov8n_coco` | 0,558 | 0,337 | — | — |
| car | 341.436 | 38.964 | `yolov8n_ft` | 0,779 | 0,518 | +22,1 | +18,1 |
| car | 341.436 | 38.964 | `yolo11n_ft` | 0,776 | 0,514 | +21,8 | +17,8 |
| truck | 27.343 | 15.862 | `yolov8n_coco` | 0,183 | 0,135 | — | — |
| truck | 27.343 | 15.862 | `yolov8n_ft` | 0,377 | 0,268 | +19,4 | +13,3 |
| truck | 27.343 | 15.862 | `yolo11n_ft` | 0,369 | 0,263 | +18,6 | +12,9 |
| bus | 9.087 | 7.018 | `yolov8n_coco` | 0,249 | 0,201 | — | — |
| bus | 9.087 | 7.018 | `yolov8n_ft` | 0,395 | 0,298 | +14,7 | +9,7 |
| bus | 9.087 | 7.018 | `yolo11n_ft` | 0,366 | 0,280 | +11,7 | +7,9 |
| motorcycle | 902 | 774 | `yolov8n_coco` | 0,117 | 0,058 | — | — |
| motorcycle | 902 | 774 | `yolov8n_ft` | 0,178 | 0,080 | +6,1 | +2,2 |
| motorcycle | 902 | 774 | `yolo11n_ft` | 0,173 | 0,080 | +5,5 | +2,2 |
| bicycle | 4.135 | 2.738 | `yolov8n_coco` | 0,113 | 0,055 | — | — |
| bicycle | 4.135 | 2.738 | `yolov8n_ft` | 0,188 | 0,090 | +7,5 | +3,5 |
| bicycle | 4.135 | 2.738 | `yolo11n_ft` | 0,196 | 0,096 | +8,3 | +4,1 |

**mAP (macro, 5 lớp: car, truck, bus, motorcycle, bicycle — mọi lớp đều có GT)**

| Model | mAP50 | mAP50–95 | ΔmAP50 | ΔmAP50–95 | P macro @0,25 | R macro @0,25 |
|:--|--:|--:|--:|--:|--:|--:|
| `yolov8n_coco` | 0,244 | 0,157 | — | — | 0,572 | 0,219 |
| `yolov8n_ft` | 0,384 | 0,251 | +14,0 | +9,4 | 0,613 | 0,347 |
| `yolo11n_ft` | 0,376 | 0,247 | +13,2 | +9,0 | 0,613 | 0,338 |

**Điểm vận hành** conf ≥ 0,25, IoU 0,5 (ngưỡng tracker c25). Detection không ghép mà IoA với vùng bỏ qua ≥ ngưỡng thì không tính FP.

| Lớp | Model | TP | FP | FN | Precision | Recall | F1 |
|:--|:--|--:|--:|--:|--:|--:|--:|
| car | `yolov8n_coco` | 150.469 | 41.977 | 190.967 | 0,782 | 0,441 | 0,564 |
| car | `yolov8n_ft` | 239.996 | 51.967 | 101.440 | 0,822 | 0,703 | 0,758 |
| car | `yolo11n_ft` | 239.438 | 56.797 | 101.998 | 0,808 | 0,701 | 0,751 |
| truck | `yolov8n_coco` | 5.450 | 8.444 | 21.893 | 0,392 | 0,199 | 0,264 |
| truck | `yolov8n_ft` | 8.724 | 3.369 | 18.619 | 0,721 | 0,319 | 0,442 |
| truck | `yolo11n_ft` | 8.271 | 3.247 | 19.072 | 0,718 | 0,302 | 0,426 |
| bus | `yolov8n_coco` | 2.434 | 2.473 | 6.653 | 0,496 | 0,268 | 0,348 |
| bus | `yolov8n_ft` | 3.297 | 1.995 | 5.790 | 0,623 | 0,363 | 0,459 |
| bus | `yolo11n_ft` | 3.149 | 2.311 | 5.938 | 0,577 | 0,347 | 0,433 |
| motorcycle | `yolov8n_coco` | 114 | 176 | 788 | 0,393 | 0,126 | 0,191 |
| motorcycle | `yolov8n_ft` | 189 | 294 | 713 | 0,391 | 0,210 | 0,273 |
| motorcycle | `yolo11n_ft` | 167 | 224 | 735 | 0,427 | 0,185 | 0,258 |
| bicycle | `yolov8n_coco` | 252 | 65 | 3.883 | 0,795 | 0,061 | 0,113 |
| bicycle | `yolov8n_ft` | 573 | 555 | 3.562 | 0,508 | 0,139 | 0,218 |
| bicycle | `yolo11n_ft` | 629 | 547 | 3.506 | 0,535 | 0,152 | 0,237 |

**Chẩn đoán mô tả** (không đổi metric)

| Model | Ảnh đã dự đoán | Ảnh không có GT 5 lớp | Detection conf ≥ 0,25 trên các ảnh đó |
|:--|--:|--:|--:|
| `yolov8n_coco` | 39.973/39.973 | 778 | 118 |
| `yolov8n_ft` | 39.973/39.973 | 778 | 121 |
| `yolo11n_ft` | 39.973/39.973 | 778 | 99 |

**Chẩn đoán bổ sung** (đếm từ `du_doan_<ten>.npz` lúc tái tính; KHÔNG có trong kế hoạch khóa, chỉ mô tả, không đổi metric): số detection sau NMS mỗi ảnh (conf ≥ 0,001, trần `max_det` 300).

| Model | Tổng detection | Trung bình / ảnh | Tối đa / ảnh | Số ảnh chạm trần 300 |
|:--|--:|--:|--:|--:|
| `yolov8n_coco` | 4.920.095 | 123,09 | 300 | 330 |
| `yolov8n_ft` | 4.624.928 | 115,7 | 300 | 155 |
| `yolo11n_ft` | 4.661.420 | 116,61 | 300 | 163 |

**Nhận xét mô tả** (một lượt mỗi model; không có khoảng tin cậy hay kiểm định):
- Cả hai model fine-tune có AP cao hơn baseline ở **cả 5 lớp**, cả AP50 lẫn AP50–95. mAP50–95 tăng +9,4 điểm
  (`yolov8n_ft`) và +9,0 điểm (`yolo11n_ft`); mAP50 tăng +14,0 và +13,2 điểm.
- AP50–95 tăng nhiều nhất ở car (+18,1 / +17,8 điểm), truck (+13,3 / +12,9) và bus (+9,7 / +7,9). motorcycle (+2,2 /
  +2,2) và bicycle (+3,5 / +4,1) tăng ít. AP50–95 tuyệt đối của hai lớp này vẫn thấp (≤ 0,096), và số GT ít (902 và
  4.135).
- **Giữa hai model fine-tune**: `yolov8n_ft` có mAP50–95 cao hơn `yolo11n_ft` 0,4 điểm (0,251 / 0,247). Theo lớp,
  `yolov8n_ft` cao hơn ở car (+0,3 điểm), truck (+0,4), bus (+1,8), motorcycle (+0,05); `yolo11n_ft` cao hơn ở bicycle
  (+0,6). Mỗi model chỉ một lượt, nên các chênh lệch này **không đủ** để nói model nào tốt hơn.
- **Điểm vận hành** conf 0,25:
  - car: recall 0,441 → 0,703 / 0,701; precision 0,782 → 0,822 / 0,808;
  - truck: precision 0,392 → 0,721 / 0,718; recall 0,199 → 0,319 / 0,302;
  - bicycle: recall tăng 0,061 → 0,139 / 0,152, nhưng precision giảm 0,795 → 0,508 / 0,535 (FP 65 → 555 / 547).

  Đây là đánh đổi tại một ngưỡng cố định, phụ thuộc độ hiệu chỉnh confidence của từng model. Không phải so sánh chất
  lượng thuần.
- **Chẩn đoán**:
  - Trên 778 ảnh không có GT 5 lớp, số detection conf ≥ 0,25 là 118 / 121 / 99.
  - Số ảnh chạm trần `max_det` 300 là 330 / 155 / 163, tức tối đa 0,83 % số ảnh. Trần chỉ cắt các detection confidence
    thấp nhất của những ảnh đó. Chẩn đoán này không có trong kế hoạch khóa và không đổi metric.
- Không so các số này với dev nội bộ vòng 42 (mục 10). Không so với UA-DETRAC: tập lớp khác, và UA-DETRAC chưa chạy.

## 8. UA-DETRAC test — **CHƯA CHẠY**, chờ quyết định ánh xạ lớp

Theo yêu cầu: chỉ báo AP cho lớp thực sự có nhãn và được ánh xạ hợp lệ, không tự đồng nhất van với truck, không tạo ánh
xạ mới. Ánh xạ hiện **chưa đủ rõ**, nên vòng này dừng ở BDD val và nêu chính xác các quyết định còn thiếu. Không viết bộ
đọc GT UA-DETRAC, không chạy dự đoán nào trên UA-DETRAC test.

**Dữ kiện** (đếm từ XML, kế hoạch khóa mục 8; không phải metric):
- Manifest khóa `results/phase2/manifests/detrac_test.json` (sha256 `ff2b4b9c…`): 40 video test.
- `data/extracted/DETRAC-Test-Annotations-XML/`: 56.167 frame có `<frame>`, trong khi trên đĩa có 56.340 ảnh. Tức
  **173 ảnh không có `<frame>`** trong XML.
- `vehicle_type`: car 548.555, **van 38.519**, bus 71.785, **others 16.915**. **Không có** nhãn xe máy hay xe đạp.
- 163 vùng `ignored_region`.
- Ánh xạ duy nhất có trong mã là `prepare_mixed_yolo_dataset.DETRAC_MAP` (van → car, others → truck). Chính mã này ghi
  đó là "GIẢ ĐỊNH", và cấu hình chứa nó (`mixed5.yaml`) ghi "không dùng trong plan đã chốt". Vì vậy không dùng.

**Quyết định cần người dùng** (mỗi mục chọn một):

| # | Câu hỏi | Lựa chọn | Hệ quả khi chấm |
|:--|:--|:--|:--|
| D1 | `van` (38.519 box) | (a) vùng bỏ qua cho mọi lớp; (b) GT lớp car; (c) cách khác | (a) không tạo GT; detection không ghép được mà nằm trên van (IoA ≥ ngưỡng) thì không bị tính FP, dù dự đoán lớp nào. (b) van là GT của car: detection car trên van có thể là TP, còn detection truck trên van thường là FP |
| D2 | `others` (16.915 box) | (a) vùng bỏ qua; (b) GT lớp truck; (c) cách khác | (a) không báo AP truck. (b) báo thêm AP truck, nhưng AP đó phụ thuộc định nghĩa "others" của UA-DETRAC |
| D3 | Lớp báo AP | car, bus, và thêm truck chỉ khi D2 = (b) | motorcycle và bicycle là **N/A**: không có GT; dự đoán của hai lớp này không được chấm và không tính FP |
| D4 | 173 ảnh không có `<frame>` | (a) giữ làm ảnh âm tính; (b) loại và ghi rõ danh sách | Chưa biết các ảnh này không có xe, hay có xe mà chưa được gán nhãn. (a) mọi detection trên các ảnh này là FP, trừ phần nằm trong `ignored_region`. (b) không tính các ảnh này |
| D5 | `ignored_region` (163) | đề xuất: vùng bỏ qua theo IoA, như BDD val | detection không ghép được mà IoA ≥ ngưỡng thì bỏ qua |

Sau khi có quyết định:
1. Ghi quyết định vào phụ lục kế hoạch và **khóa trước khi chạy**.
2. Viết bộ đọc GT UA-DETRAC, giống cách làm ở mục 4: chỉ thêm phần đọc GT, kèm test hồi quy và kiểm fingerprint (manifest,
   XML, nội dung ảnh).
3. Chạy tuần tự ba model như BDD val.

mAP UA-DETRAC sẽ tính trên tập lớp khác BDD val, nên **không** so trực tiếp với mAP BDD val.

## 9. Bằng chứng trong gói

- `logs/review43/`:
  - `00_trang_thai_dau.log`; `01_kiem_checkpoint.{log,json}` + `kiem_checkpoint.py`; `02_tai_hien_chan_bdd_val.log`;
    `03_test_danh_gia_bdd_val.log`, `03_test_detector_5lop.log`; `04_ke_hoach_danh_gia.md` (kế hoạch khóa);
  - mỗi lượt: log trạng thái máy + sha256 mã đánh giá, log khởi chạy (`05`/`06` baseline, `07`/`08` YOLOv8n fine-tune,
    `09`/`10` YOLO11n fine-tune);
  - `11_tong_hop_kiem_hop_le.log`: kiểm hợp lệ, tái tính từ NPZ, tổng hợp bảng; `11a_tong_hop_lan1_dung_vi_cham.log`:
    lần 1 của bước tổng hợp, bị dừng vì lỗi hiệu năng của script (mục 6);
  - script `khoi_chay_danh_gia_wmi.ps1`, `tong_hop_danh_gia.py`.
- `logs/review43/ket_qua/`:
  - `bang_theo_lop.csv`, `bang_macro.csv`, `bang_ket_qua.md` (bảng mục 7, sinh tự động từ `ket_qua.json`);
  - `kiem_hop_le.json` (từng phép kiểm, chẩn đoán bổ sung);
  - `<ten>/ket_qua.json`, `<ten>/theo_lop.csv`: chép nguyên byte từ thư mục kết quả;
  - `<ten>/giam_sat/tom_tat.json`, `trang_thai.json`, `log_loc.txt`. `log_loc.txt` là `log.txt` của bộ giám sát đã bỏ
    2.499 dòng cảnh báo lặp "'half' is deprecated"; số dòng bỏ được ghi ở cuối file;
  - `tep_lon_khong_ban_giao.csv`: 9 file lớn **không đóng gói**, kèm đường dẫn, bytes, sha256 — 3 file dự đoán
    `du_doan_<ten>.npz`, 3 `log.txt` đầy đủ, 3 `ram_mau_tho.csv`.
- **Không đóng gói**, chỉ ghi đường dẫn và hash:
  - dữ liệu BDD val: nhãn COCO-video `data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json` (117.094.032 B,
    `074ff795…`), ảnh `data/raw/bdd100k/images/track/val/` (39.973 ảnh; sha256 tổng hợp nội dung `1e209af9…`, sha256
    danh sách ảnh `4bdf4889…`), bảng hash `data/interim/hash_tap_danh_gia/bdd100k_mot_val_sha256.csv` (`60fea2a9…`),
    manifest `results/phase2/manifests/bdd100k_transfer_cocovid_val.json` (`2071c15b…`);
  - checkpoint (mục 3);
  - dự đoán NPZ, log đầy đủ, CSV RAM (`tep_lon_khong_ban_giao.csv`).

## 10. Giới hạn

- **Mỗi model một lượt, một seed.** Không có khoảng tin cậy hay kiểm định. Mọi chênh lệch ở mục 7 là **mô tả**; không
  kết luận "cải thiện có ý nghĩa thống kê". Đặc biệt, chênh lệch giữa `yolov8n_ft` và `yolo11n_ft` chưa đủ để nói model
  nào tốt hơn.
- **So với baseline** là so một model COCO (ánh xạ theo tên lớp) với model đã fine-tune trên BDD train cùng miền. Chênh
  lệch gộp cả khác biệt miền dữ liệu lẫn khác biệt định nghĩa lớp giữa COCO và BDD; đây không phải so sánh kiến trúc.
  Vòng này không tách riêng các yếu tố đó.
- **BDD val không phải tập chưa từng xem**: đã dùng trong thí nghiệm transfer tracking; 3 sequence đã xem metric ở smoke
  (vòng 5–6). Tập này **không** được dùng để chọn checkpoint.
- **Nhãn** là bản COCO-video của dự án. 183 khác biệt lớp / 50 ngoại lệ ignore so với native (nguồn gốc UNKNOWN) giữ
  nguyên, không sửa.
- **Dev nội bộ** (dùng chọn `best.pt`, vòng 42) có ảnh đã tô xám vùng ignore, đo bằng validator của Ultralytics. BDD val
  không tô xám và đo bằng evaluator 5 lớp. Hai phép đo khác nhau; không so hay trộn với kết quả vòng này.
- **Điểm vận hành** dùng cùng conf 0,25 cho cả ba model. Độ hiệu chỉnh confidence khác nhau giữa các model, nên P/R tại
  một ngưỡng cố định không phải thước đo chất lượng thuần.
- **AP** tự cài theo logic pycocotools; khác `Params.iouThrs` duy nhất ở ngưỡng 0,9 (1 ulp, giới hạn đã ghi từ vòng 28).
- **Trần `max_det` 300** áp dụng trước khi chấm, như kế hoạch. Số ảnh chạm trần được ghi ở chẩn đoán bổ sung (mục 7);
  chẩn đoán này không có trong kế hoạch khóa và không đổi metric.
- **Log trạng thái trước lượt `yolov8n_ft`** không được ghi. Tôi khởi chạy ngay sau khi kiểm nguồn, rồi bù bằng
  `08_trang_thai_khi_chay_yolov8n_ft.log` (chỉ đọc, khoảng 4 phút sau khi khởi chạy). Sha256 mã đánh giá ghi trước lượt
  đầu (`05`), trong lượt thứ hai (`08`), trước lượt thứ ba (`09`) và lúc tổng hợp (`11`) đều trùng; HEAD không đổi trong
  suốt ba lượt.
- **RAM máy thấp**: RAM trống theo bộ giám sát có lúc chỉ còn 50 MiB (lượt baseline). Cả ba lượt vẫn exit 0 và đủ ảnh;
  RAM thấp có thể ảnh hưởng thời gian chạy. Thời gian lượt không phải số đo tốc độ chuẩn.
- **Reviewer không có byte** dữ liệu, checkpoint hay NPZ trong gói, chỉ có hash và đường dẫn.
- **UA-DETRAC test chưa chạy** (mục 8).

## 11. Lệnh (từ gốc repo) và mã thoát

```text
python handoff/logs/review43/kiem_checkpoint.py --out handoff/logs/review43/01_kiem_checkpoint.json     # exit 0
python src/eval_detector_5lop.py --data data/yolo_bdd5_v1/data.yaml --split test ...                     # exit 2 (tai hien diem chan)
python src/test_danh_gia_bdd_val.py                                                                      # exit 0, 19/19
python src/test_detector_5lop.py                                                                         # exit 0, 92/92
powershell -NoProfile -ExecutionPolicy Bypass -File handoff\logs\review43\khoi_chay_danh_gia_wmi.ps1 -Ten <ten>   # 3 lan, exit 0
  # -> python -u src\chay_giam_sat.py chay --thu-muc results\phase2\detector_eval\giam_sat\r43_<ten> --khong-in --
  #    python -u src\danh_gia_detector_bdd_val.py --model <ten>=<duong dan> --sha256 <ten>=<sha256> \
  #        --out results\phase2\detector_eval\bdd_val_cocovid\r43_<ten>
python handoff/logs/review43/tong_hop_danh_gia.py --out handoff/logs/review43/ket_qua --tai-tinh          # exit 0 (lan 2)
  # lan 1: toi dung bang Stop-Process sau ~19 phut (loi hieu nang cua script, 11a_*); thu muc ket_qua rong da xoa
```

| Lượt | Exit bộ giám sát (= exit điểm vào) |
|:--|:--|
| `yolov8n_coco` | **0** |
| `yolov8n_ft` | **0** |
| `yolo11n_ft` | **0** |

## 12. Chưa làm — chờ người dùng

- **UA-DETRAC test**: cần 5 quyết định D1–D5 ở mục 8.
- So sánh bộ lọc, thuật toán tracking, tối ưu Nano: ngoài phạm vi vòng này.
- Đề xuất cũ vẫn chờ duyệt: guard ở hàm append; khóa thư mục lượt.

## 13. Commit vòng 43

| Commit | Nội dung |
|:--|:--|
| `4fa1839` | Ghi nhận nghiệm thu `4d54744`, bằng chứng 10/10, archive gói, base → `4d54744`; đính chính R42-N1/N2/N3; trạng thái đầu |
| `1df012f` | Khóa giao thức trước khi chạy: kiểm 3 checkpoint, tái hiện điểm chặn, điểm vào BDD val + test 19/19, suite 92/92, kế hoạch |
| `b9ea1d8` | Launcher WMI cho lượt đánh giá (checkpoint + sha256 cố định; từ chối ghi đè / chạy chồng) |
| `6dc5b29` | 3 lượt BDD val (log, kết quả nhỏ), kiểm hợp lệ 39/39, tái tính từ NPZ, bảng kết quả |
| commit chứa phụ lục này (target delta) | Brief, phụ lục, handoff, trạng thái kế hoạch |
