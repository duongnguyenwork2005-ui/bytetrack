# Vòng 37 — kế hoạch pilot đã khóa (ghi TRƯỚC khi train)

Người dùng duyệt ngày 2026-10-02: pilot chẩn đoán **YOLOv8n và YOLO11n, mỗi model một lượt 2 epoch, chạy lần lượt** trên
GPU PC, dùng đúng dataset smoke đã khóa. Ngân sách là 2 model × 2 epoch. Phép đo trước bước học không phải lượt train thêm.

## Đầu vào (khóa: `01_khoa_cau_hinh_pilot_<model>.json`, tạo bằng `--che-do kiem-pilot`, không train)

| Mục | Giá trị |
|:--|:--|
| Cấu hình pilot | `configs/phase2/detector/train_bdd5_pilot.yaml`, sha256 `5d195826…` |
| Kế thừa | `train_bdd5.yaml`, sha256 (nội dung LF) `a466a4fa…`, cùng file đã dùng cho smoke vòng 35 |
| Khác smoke | **chỉ** `epochs` 1 → 2; runner từ chối nếu tham số hiệu lực khác ở bất kỳ khóa nào khác |
| Checkpoint khởi tạo | YOLOv8n `models/yolov8n.pt` `f59b3d83…`; YOLO11n `models/candidates/yolo11n.pt` `0ebbc80d…` (COCO 80 lớp, như smoke). Không resume từ checkpoint smoke |
| Dataset | smoke đã khóa: 334 ảnh train / 84 ảnh dev; `data.yaml` `9744ba88…`, manifest `c4899826…`, split `b8a2848e…`; hợp đồng v2 kiểm lại trước khi nạp model |
| Tham số giữ như smoke | batch 8, imgsz 640, SGD lr0 0,01 / lrf 0,01 / momentum 0,937 / wd 0,0005, nbs 64, warmup_epochs 1,0 / warmup_bias_lr 0,1 / warmup_momentum 0,8, augmentation (mosaic 1,0, close_mosaic 2 …), AMP tắt, workers 2, `cache: false`, seed 0, deterministic, patience 1 |
| Thư viện | ultralytics 8.4.141 (`engine/trainer.py` sha256 `7532ff0b…`), torch 2.6.0+cu124 |

Lệnh: `bash handoff/logs/review37/chay_pilot.sh`, chạy tuần tự YOLOv8n rồi YOLO11n. Mỗi model:

```text
python src/train_detector.py --config configs/phase2/detector/train_bdd5_pilot.yaml --model <M> --che-do kiem-pilot
python handoff/logs/review37/so_khop_khoa.py handoff/logs/review37/01_khoa_cau_hinh_pilot_<M>.json <log kiem-pilot>
python -u src/train_detector.py --config configs/phase2/detector/train_bdd5_pilot.yaml --model <M> --che-do pilot --xac-nhan-pilot
```

Bước so khớp đứng trước để bảo đảm lượt chạy dùng đúng khóa này; lệch khóa thì không train.

## Dự đoán theo mã ultralytics (trước khi chạy; lượt chạy ghi giá trị thực)

- 42 batch/epoch. `BaseTrainer._get_warmup_iterations` (gọi chính hàm đang cài) cho **nw = 42**, nghĩa là warmup phủ trọn
  epoch 1:
  - LR nhóm weight/bn tăng từ 0 lên 0,01; bias giảm từ 0,1 xuống 0,01;
  - momentum từ 0,8 lên 0,937;
  - accumulate từ 1 lên 8.
- Epoch 2 không warmup: LR = 0,01 × lf(1) = **0,00505**; accumulate giữ giá trị warmup cuối.
- **Mosaic**: ultralytics tắt mosaic khi `epoch == epochs − close_mosaic` = 2 − 2 = 0. Vì vậy **cả hai epoch pilot
  không dùng mosaic**, còn smoke 1 epoch có mosaic (1 − 2 = −1, không bao giờ tắt). Giá trị cấu hình giữ nguyên như yêu cầu;
  augmentation **hiệu lực** vẫn khác smoke ở điểm này. Lượt chạy ghi trạng thái mosaic thực tế từng epoch.
- Patience 1 không thể dừng trước epoch 2: EarlyStopping chỉ tăng delta từ epoch thứ hai.

## Phép đo (`src/pilot_chan_doan.py`; test `src/test_pilot_chan_doan.py` 34/34 PASS trước khi khóa)

- **Trước học** (`on_train_start`): đo chính model 5 lớp trainer đã dựng, sau đổi nc và nạp/remap, trước bước cập nhật đầu.
  Ghi kèm:
  - tên và thứ tự lớp;
  - tensor chép nguyên / remap / khởi tạo mới, đối chiếu độc lập với checkpoint COCO;
  - optimizer chưa có trạng thái, `ema.updates` = 0, không gradient, model trùng EMA.
- **Sau mỗi epoch** (`on_fit_epoch_end` trong vòng lặp): đo **model hiện tại** và **EMA**, ghi tách hai nguồn.
  - Lần gọi thứ hai trong `final_eval` chỉ được ghi mốc thời gian.
- **Sau train**: đo `last.pt` và `best.pt` nạp từ đĩa; nạp lại và kiểm đúng 5 lớp.
- **Mỗi phép đo** dùng nguyên evaluator: imgsz 640, **conf 0,001**, NMS IoU 0,7, max_det 300, FP32, 84 ảnh dev smoke. Ghi:
  - số detection, phân bố confidence, AP theo lớp có GT;
  - **chẩn đoán riêng**: điểm lớp lớn nhất trước NMS/ngưỡng. Không dùng tính AP, không hạ ngưỡng.
- **Bất biến**: chụp trạng thái huấn luyện trước và sau mỗi phép đo, khôi phục RNG và cờ. Nếu còn khác thì dừng train.
  A/B trên CPU đã cho thấy lượt có đo trùng lượt không đo từng bit.
- **Diễn biến**:
  - mỗi batch: loss, LR và momentum từng nhóm, accumulate, `ema.updates`;
  - mỗi epoch: mosaic, số lần cập nhật, đỉnh bộ nhớ torch;
  - mốc thời gian khởi tạo / train / val / lưu / đo;
  - RAM/VRAM: `DoVram` (nvidia-smi cả GPU + psutil available, 1 s, mẫu thô) và bộ đo ngoài (RSS tiến trình + con, 1 s, mẫu thô).

## Tiêu chí hoàn chỉnh (không đặt điều kiện AP)

Đầu vào đúng; đủ 2 epoch; mỗi epoch có cập nhật trọng số; loss hữu hạn mọi batch; checkpoint nạp lại đúng 5 lớp; đủ 7 phép đo
và mọi phép đo trong lúc train không đổi trạng thái.

## Ngoài phạm vi

- Không train thêm epoch, đổi batch / ảnh / LR / warmup / augmentation, thêm model / seed / dataset, xuất dataset đầy đủ hay
  training dài.
- Không dùng pilot để xếp hạng hay chọn model.
- Dev smoke không có motorcycle/bicycle, nên không đánh giá được AP hai lớp này.
- Lỗi: giữ log và kết quả dở dang, không chạy lại để tìm kết quả tốt hơn.
