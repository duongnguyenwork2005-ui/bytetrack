# Vòng 45 — DỰ THẢO giao thức đánh giá detector trên UA-DETRAC test (CHƯA khóa, CHƯA chạy)

**Trạng thái**: dự thảo để người dùng duyệt D1–D5. Đây **không** phải kế hoạch khóa. Chưa viết bộ đọc GT, chưa chạy dự
đoán. Sau khi người dùng duyệt:
1. ghi quyết định vào một **kế hoạch khóa mới** (file mới, commit **trước** lượt đầu);
2. viết bộ đọc GT UA-DETRAC kèm test fixture (mục 8);
3. dựng bảng hash nội dung ảnh;
4. chạy tuần tự ba model.

Dữ kiện dưới đây đo bằng thao tác chỉ đọc ở vòng 45 (`logs/review45/01`–`05`).

## 1. Phạm vi

- Đánh giá detector trên PC local cho **ba checkpoint của vòng 43**, trên **UA-DETRAC test** (40 video, manifest khóa).
- Không train, không tuning theo test, không chọn checkpoint hay ngưỡng theo kết quả test.
- Không SSH Nano. Không so sánh bộ lọc / tracking / Nano.
- Kết quả UA-DETRAC báo **riêng**, không gộp hay so mAP trực tiếp với BDD val: tập lớp đánh giá khác nhau.

## 2. Checkpoint

Hash kiểm lại từ file ở vòng 45 (`05_kiem_hash_checkpoint.log`), trùng kế hoạch khóa vòng 43 mục 2:

| Tên | File | sha256 | Bytes | Ánh xạ lớp |
|:--|:--|:--|--:|:--|
| `yolov8n_coco` (baseline) | `models/yolov8n.pt` | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | 6.549.796 | COCO 80 → 5 lớp theo tên: 2 car, 7 truck, 5 bus, 3 motorcycle, 1 bicycle |
| `yolov8n_ft` | `models/finetune_phase2/train_bdd5_v1_yolov8n_day_du/weights/best.pt` | `5f66776d3202389ca484f3d9e2f107dddb354d21a9f1762d19401d690a4a0d8a` | 6.222.691 | 5 lớp 0..4 |
| `yolo11n_ft` | `models/finetune_phase2/train_bdd5_v1_yolo11n_day_du/weights/best.pt` | `d7d1c7e86d03c69d913febe79e4cbb6c2e1bcf9f17f034724af4de2ccfba7923` | 5.445.075 | 5 lớp 0..4 |

- Ánh xạ theo **tên**, như vòng 43 (`detector_classes.anh_xa_model_sang_lop5`).
- Không checkpoint nào được train trên UA-DETRAC. Hai model fine-tune chỉ học BDD train, baseline là COCO.
- Lượt chạy tự kiểm lại sha256 trước khi nạp; lệch thì exit 2.

## 3. Dữ liệu

- **Manifest khóa** `results/phase2/manifests/detrac_test.json` (sha256 `ff2b4b9c…`): 40 video, trùng
  `DETRAC-MOT-toolkit/evaluation/seqs/testlist-full.txt`.
- **Nhãn**: `data/extracted/DETRAC-Test-Annotations-XML/<video>.xml`. sha256 từng file nằm ở
  `logs/review45/phan_tich/theo_video.csv`, sẽ được ghim lại trong kế hoạch khóa.
- **Ảnh**: `data/extracted/DETRAC-Images/<video>/img%05d.jpg`. Cả 40 video đều 960×540 (đọc header ảnh đầu và cuối).
- **Thống kê** (đếm lại ở vòng 45, khớp số vòng 43 — đây không phải metric):
  - 56.167 `<frame>` trong XML; 56.340 ảnh trên đĩa; 173 ảnh không có `<frame>`; 0 `<frame>` không có ảnh;
  - 675.774 box: car 548.555, bus 71.785, **van 38.519**, **others 16.915**. **Không có** nhãn `truck`, xe máy hay
    xe đạp;
  - 163 `ignored_region`, trùng file `igrs/*_IgR.txt` của toolkit ở 40/40 video.
- **Tập này đã được dùng trước đây**: các thí nghiệm tracking giai đoạn 2 (detection YOLOv8n COCO) đã chạy trên 40 video
  test. Vì vậy không mô tả đây là tập chưa từng xem. Tập này không dùng để chọn checkpoint.
- **Không tái dùng** cache detection tracking cũ (conf 0,05, ảnh đã tô xám, định dạng khác).

## 4. Quy tắc GT — phụ thuộc D1–D5 (đề xuất **in đậm**, chờ duyệt)

| Mục | Đề xuất | Phương án khác |
|:--|:--|:--|
| car, bus | **GT lớp car, bus** (dùng nhãn trực tiếp) | — |
| D1 `van` | **vùng bỏ qua cho mọi lớp** | GT lớp car |
| D2 `others` | **vùng bỏ qua cho mọi lớp** | GT lớp truck |
| D3 lớp báo AP | **car, bus**; mAP = trung bình 2 lớp, ghi "mAP (car, bus)"; truck / motorcycle / bicycle = **N/A** | thêm truck nếu D2 = GT truck |
| D4 173 ảnh không có `<frame>` | **loại khỏi đánh giá, ghi danh sách** (`phan_tich/anh_khong_frame.csv`) | giữ làm ảnh âm tính |
| D5 `ignored_region` | **vùng bỏ qua theo đúng quy tắc BDD hiện dùng** (mục 6), GT trong vùng vẫn là GT, không tô xám ảnh | quy tắc kiểu toolkit DETRAC (cần đổi evaluator, ngoài kế hoạch) |

Các quy tắc khác kế thừa vòng 43:
- box GT cắt vào khung 960×540; box nhỏ hơn 1 px sau khi cắt thì bỏ và đếm riêng;
- toạ độ pixel theo kích thước ảnh thật; w/h khác 960×540 thì lỗi;
- không sửa XML, không tự gán lại nhãn theo quan sát bằng mắt.

## 5. Dự đoán (kế thừa nguyên giao thức vòng 43)

- Hàm `du_doan_ultralytics` của evaluator: `imgsz 640`, `conf 0.001`, NMS IoU `0.7` theo lớp (không agnostic), `max_det
  300`, FP32, lọc lớp theo **tên** trước NMS; batch 16, `device 0`; ultralytics 8.4.141, torch 2.6.0+cu124.
- **Khác biệt bắt buộc do dataset**:
  - ảnh 960×540 thay vì 1280×720. Cùng `imgsz 640`, nên ảnh được thu theo tỉ lệ khác;
  - **không tô xám** vùng bỏ qua trên ảnh, nhất quán với BDD val. Điều này khác pipeline tracking / Nano, vốn tô xám
    `ignored_region` trước khi detect (`baseline_track.mask_ignored_regions`). Kết quả detector vì vậy không thay cho
    detection mà tracker nhận.
- Dự đoán cả 5 lớp như BDD; chỉ các lớp ở D3 được chấm.

## 6. Chấm điểm (evaluator `eval_detector_5lop.py`, sha256 `30a1fd5a…`, không đổi)

- **Ghép** như `COCOeval.evaluateImg`: trong từng (ảnh, lớp), detection xếp conf giảm dần (mergesort); ghép với GT chưa
  ghép có IoU cao nhất ≥ ngưỡng; hòa IoU thì chọn GT sau; IoU như `bbIou`.
- **Vùng bỏ qua** (`ghep_anh_lop`):
  - mảng `ign` dùng chung cho mọi lớp của ảnh, gồm `ignored_region` cộng các box chuyển thành vùng bỏ qua ở D1/D2;
  - **chỉ** detection không ghép được GT hợp lệ mới bị xét;
  - nó bị bỏ qua (không TP, không FP) nếu `IoA = diện tích(det ∩ vùng_k) / diện tích(det)`, lấy **max theo từng vùng k**
    và tính bằng hình học số thực, **≥ ngưỡng IoU đang xét** (0,50…0,95). Ở điểm vận hành, IoU 0,5 nên ngưỡng IoA là 0,5;
  - GT hợp lệ nằm trong vùng bỏ qua **vẫn là GT**: detect được là TP, bỏ sót là FN.
- **AP**: nội suy 101 điểm × 10 ngưỡng IoU 0,50:0,05:0,95; lớp không có GT → `null` (không phải 0), không vào mAP.
- **Điểm vận hành**: P / R / F1 tại conf ≥ 0,25, IoU 0,5. Macro chỉ tính trên các lớp có GT.

## 7. Lưu và fingerprint

- Mỗi lượt ghi:
  - lệnh, thời điểm, exit code thật;
  - số ảnh dự kiến và số ảnh đã dự đoán;
  - sha256 checkpoint, manifest, từng XML;
  - danh sách ảnh (sha256 của danh sách), sha256 tổng hợp nội dung ảnh đối chiếu bảng hash;
  - danh sách ảnh bị loại theo D4;
  - `quy_tac`, `quy_tac_gt` hiệu lực.
- **Bảng hash nội dung ảnh** (mới, dựng trước khi khóa): `data/interim/hash_tap_danh_gia/detrac_test_sha256.csv`, băm
  lại toàn bộ ảnh, không dùng mtime. Lệch thì exit 2.
- Đầu ra `results/phase2/detector_eval/detrac_test/<lượt>/`: `du_doan_<ten>.npz`, `ket_qua.json`, `theo_lop.csv`.
  Không ghi đè thư mục cũ.

## 8. Cách chạy (sau khi duyệt)

- **Điểm vào mới** `src/danh_gia_detector_detrac_test.py`, theo đúng mẫu vòng 43:
  - **chỉ thêm phần đọc GT** từ XML theo D1–D5;
  - dự đoán, ghép, AP và lưu dùng nguyên hàm của evaluator;
  - kiểm trước dự đoán: sha256 checkpoint, manifest, XML, tập ảnh, w/h, nội dung ảnh.
- **Test fixture** (XML + ảnh nhỏ tự tạo) cho từng quy tắc D1–D5, đi qua entry point thật với dự đoán giả lập; kèm
  `--support` khi bàn giao.
- Khởi chạy độc lập qua WMI, tuần tự baseline → `yolov8n_ft` → `yolo11n_ft`, thư mục mới.
- Thời gian theo kịch bản tốc độ BDD vòng 43 (khoảng 19–20 phút cho 39.973 ảnh mỗi model): khoảng 25–35 phút cho
  56.167 ảnh mỗi model. Đây là kịch bản theo giả định, không phải bảo đảm.

## 9. Báo cáo

- Mỗi lớp được chấm: AP50, AP75, AP50–95, số GT, số ảnh có lớp; mAP trên các lớp ở D3, ghi rõ danh sách lớp.
- Chênh lệch so với baseline tính bằng điểm AP; P / R / F1 ở điểm vận hành.
- Lớp không chấm ghi **N/A**.
- Không so mAP với BDD val. Không so với bảng xếp hạng DETRAC chính thức: toolkit chấm một lớp "vehicle", AP kiểu khác,
  quy tắc ignore khác (mục 10).
- Không kết luận "có ý nghĩa thống kê": mỗi model một lượt.

## 10. Khác biệt với toolkit DETRAC chính thức (đọc mã `data/extracted/DETRAC-MOT-toolkit`, không chạy)

| Điểm | Giao thức dự án (đề xuất) | Toolkit |
|:--|:--|:--|
| Lớp | car, bus riêng từng lớp | một lớp "vehicle" (mọi `vehicle_type`) |
| IoA với vùng bỏ qua | max theo từng vùng, hình học số thực | tỉ lệ pixel trong **hợp** các vùng (raster) |
| Ngưỡng IoA | bằng ngưỡng IoU đang xét (0,50…0,95) | cố định 0,5 |
| Thời điểm | sau khi ghép GT hợp lệ | loại detection **trước** khi chấm (`dropDetections.m`) |
| GT trong vùng bỏ qua | giữ làm GT | đường MOT: bỏ box GT IoA ≥ 0,5 ở hai đầu quỹ đạo (`parseGT.m`); đường DET: nằm trong `AP_DET_EVAL.exe` (nhị phân, không đọc được) |
| AP | COCO 101 điểm × 10 IoU | PR theo ngưỡng score 0:0,05:1, nội suy tuyến tính |

Quy mô khác biệt, đo trên GT (`03_phan_tich_vung_bo_qua.log`):
- GT car có IoA ≥ 0,5 với vùng bỏ qua: 35.248 box theo max từng vùng, 35.757 box theo hợp pixel. 699 + 190 box chỉ đạt
  ngưỡng ở một trong hai cách tính;
- 7.779 box car nằm trọn trong vùng bỏ qua;
- 4 cặp vùng bỏ qua chồng nhau.

## 11. Tiêu chí lượt hợp lệ (dự kiến)

- exit 0;
- số ảnh dự đoán = số ảnh của tập đánh giá theo D4;
- sha256 checkpoint / manifest / XML khớp;
- tập ảnh và w/h đạt; nội dung ảnh khớp bảng hash.
