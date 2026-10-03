# Vòng 42 — kế hoạch lượt training dài (ghi TRƯỚC khi khởi chạy)

## 1. Phê duyệt của người dùng (2026-10-03)

- Training dài trên PC local: **YOLOv8n rồi YOLO11n**, chạy tuần tự, mỗi model **tổng cộng 15 epoch**.
- Dataset đầy đủ `data/yolo_bdd5_v1/`, bước frame 5, split đã khóa.
- Giữ batch, workers, seed, optimizer, augmentation và các tham số đã duyệt. Không tự tuning.
- Ngân sách dự toán hai model: **56,64 giờ**; kịch bản chậm hơn **73,95 giờ**. Đây là kịch bản theo giả định, **không phải bảo
  đảm**. Theo `DETECTOR_TRAIN_PLAN.md` mục 5a:
  - YOLOv8n: 1,74 h/epoch → 26,5 h (chậm hơn: 34,6 h);
  - YOLO11n: 1,98 h/epoch → 30,2 h (chậm hơn: 39,4 h).
- Chỉ training và validation dev theo cấu hình hiện có. Chưa chạy test BDD val / UA-DETRAC, so sánh bộ lọc, tracking hay
  tối ưu Nano.

## 2. Cấu hình hiệu lực (không đổi)

- `configs/phase2/detector/train_bdd5.yaml`, sha256 `a466a4fa…`, **giữ nguyên byte**.
  - Chú thích "DE XUAT (chua duyet)" cạnh `epochs: 15` là lịch sử. Phê duyệt được ghi ở tài liệu này và ở handoff.
  - Không sửa file vì cấu hình pilot ghim sha256 của nó (`ke_thua_sha256`), và test P1a kiểm giá trị đó.
- Dry-run hai model (`01_dry_run_<M>.log`) in ra tham số hiệu lực:
  - `epochs 15`, `imgsz 640`, `batch 8`;
  - `SGD`, `lr0 0.01`, `lrf 0.01`, `momentum 0.937`, `weight_decay 0.0005`, `nbs 64` → `accumulate 8`, weight decay hiệu
    lực 0,0005;
  - `warmup_epochs 1.0`, `close_mosaic 2` (epoch 14–15 không mosaic), `patience 5`;
  - `seed 0`, `deterministic`, `amp false`, `workers 2`, `cache false`;
  - augmentation như cấu hình;
  - 6.311 batch/epoch, 94.665 batch nếu chạy đủ 15 epoch.
- **`epochs = 15` là TỔNG số epoch của lượt, kể cả khi resume.**
  - Resume dùng resume của Ultralytics với đường dẫn checkpoint; `epochs` lấy từ checkpoint.
  - Runner kiểm cấu hình hiệu lực trùng checkpoint, không cộng thêm epoch, và kiểm `tong_epoch` sau khi nạp.
- **Dừng sớm**: `patience 5` theo fitness của Ultralytics (mAP50-95 trên dev) có thể kết thúc lượt trước epoch 15.
  - Đó là kết thúc hợp lệ của cấu hình đã duyệt.
  - Không chạy thêm epoch, không đổi patience.
- Checkpoint khởi tạo:
  - YOLOv8n: `models/yolov8n.pt`, sha256 `f59b3d83…`, COCO 80 lớp;
  - YOLO11n: `models/candidates/yolo11n.pt`, sha256 `0ebbc80d…`.

  Runner kiểm hash và loại checkpoint trước khi nạp.
- Hợp đồng dataset kiểm lại khi dry-run và một lần nữa khi khởi chạy thật:
  - manifest v2, sha256 từng ảnh / nhãn / file ignore, split đã khóa;
  - 50.487 ảnh train-fit / 5.559 ảnh dev;
  - đối chiếu với nhãn Ultralytics thực nạp ở callback, trước vòng train.

  Không xuất lại dataset.

## 3. Tên lượt và thư mục (đều mới, đã kiểm chưa tồn tại)

| Model | Thư mục lượt | Kết quả runner | Thư mục giám sát |
|:--|:--|:--|:--|
| YOLOv8n | `models/finetune_phase2/train_bdd5_v1_yolov8n_day_du/` | `results/phase2/detector_train/train_bdd5_v1_yolov8n_day_du.json` | `results/phase2/detector_train/giam_sat/yolov8n_lan0/` |
| YOLO11n | `models/finetune_phase2/train_bdd5_v1_yolo11n_day_du/` | `results/phase2/detector_train/train_bdd5_v1_yolo11n_day_du.json` | `results/phase2/detector_train/giam_sat/yolo11n_lan0/` |

- Resume lần N ghi `…__tiep_tuc_N.json`, giám sát ở `giam_sat/<M>_tiep_tuc_<N>/`.
- Không ghi đè smoke / pilot / `kiem_resume_v39` hay kết quả cũ. Runner và bộ giám sát đều tự từ chối ghi đè.

## 4. Khởi chạy

- Runner hiện có, cờ xác nhận training dài, qua bộ giám sát (`khoi_chay_wmi.ps1`, cwd = gốc repo):

  ```text
  python.exe -u src\chay_giam_sat.py chay --thu-muc results\phase2\detector_train\giam_sat\<M>_lan0 --khong-in -- ^
      python.exe -u src\train_detector.py --model <M> --che-do day-du --xac-nhan-train-dai
  ```

- **Tiến trình độc lập với phiên Claude**: tạo bằng WMI `Win32_Process.Create`.
  - Tiến trình cha là `WmiPrvSE.exe`, nằm ngoài cây tiến trình `claude.exe` / VSCode.
  - Cùng session người dùng; cửa sổ ẩn.
  - Không phải tác vụ nền của Claude (bị cắt sau khoảng 2 giờ).
  - Đã thử bằng tiến trình thăm dò vô hại: không nằm trong Job Object, sống sau khi lệnh tạo trả về. Môi trường là môi
    trường người dùng chuẩn: cùng USERPROFILE / APPDATA / TEMP, cùng file settings Ultralytics, tìm thấy `git` và
    `nvidia-smi`.
- Sau khi tạo, chỉ báo "đã bắt đầu" khi đã thấy:
  - PID công cụ và PID runner (`trang_thai.json`) còn sống, command line đúng;
  - `log.txt` tăng;
  - thư mục lượt được tạo;
  - sự kiện `bat_dau` trong nhật ký lượt;
  - tiến độ epoch 1 trong log.

## 5. Theo dõi (chỉ đọc)

- `theo_doi_luot.py` ghi ảnh chụp JSON vào `theo_doi_<M>.log`: lúc bắt đầu, mỗi epoch mới, mỗi 30 phút, khi kết thúc /
  bị ngắt / đứng yên.
- Mỗi ảnh chụp gồm:
  - trạng thái, PID, exit code;
  - epoch đã lưu, fitness;
  - tiến độ epoch/batch, tuổi của `log.txt`;
  - RSS runner + tiến trình con, RAM toàn máy, pagefile;
  - VRAM / nhiệt GPU, pin, dung lượng đĩa.
- Bộ giám sát ghi:
  - stdout + stderr vào `log.txt`;
  - RAM đúng PID và tiến trình con, cùng RAM toàn máy, mỗi giây vào `ram_mau_tho.csv`;
  - exit code thật vào `tom_tat.json`.
- Runner ghi:
  - checkpoint mỗi epoch vào `khoi_phuc/last_eNNN.pt`;
  - nhật ký lượt, `theo_batch.csv`, VRAM.
- Claude kiểm định kỳ bằng các lượt chờ ở nền, mỗi lượt tối đa khoảng 110 phút. Một lượt chờ hết giờ **không** có nghĩa là
  training thất bại.
- Không sửa mã, không commit trong lúc train. Không mở thêm thí nghiệm sau khi xem metric.

## 6. Phục hồi khi bị ngắt

1. Giữ nguyên mọi log. Xác định trạng thái bằng `chay_giam_sat.py trang-thai`, không đoán.
2. Chạy `python src\train_detector.py --model <M> --dry-run --tiep-tuc`, ghi log.
3. Chỉ khi kiểm tra đạt mới resume đúng lượt bằng cơ chế đã nghiệm thu: `khoi_chay_wmi.ps1 -Model <M> -GiamSat
   <M>_tiep_tuc_<N> -TiepTuc`.
4. Không chạy lại từ đầu, không tăng epoch, không đổi cấu hình hay workers / batch.
5. Lỗi lặp lại cùng nguyên nhân, hoặc dry-run từ chối: **dừng và báo nguyên nhân**, có bằng chứng.
6. Điều kiện ngoài tầm kiểm soát (nguồn điện, RAM, đĩa, cập nhật Windows): báo bằng chứng và thao tác người dùng cần làm.
   Không tự đóng ứng dụng, không đổi cài đặt hệ thống.

## 7. Điều kiện để bắt đầu YOLO11n

YOLOv8n phải kết thúc thành công:
- bộ giám sát exit 0;
- runner in `[DAT] day-du yolov8n`; kết quả có `loi = None`, không lỗi callback, loss hữu hạn;
- `last.pt` / `best.pt` nạp được, đúng 5 lớp, tham số trùng cấu hình;
- nhật ký `ket_thuc`: `hoan_tat`, `kiem_sau_train_dat = true`, `exit_code = 0`;
- đủ 15 epoch, hoặc ít hơn nếu dừng sớm (`dung = true` ở epoch cuối).

YOLO11n chạy theo đúng quy trình trên, tên và thư mục riêng. Không chạy đồng thời hai model.

## 8. Rủi ro đã biết lúc chuẩn bị

- RAM trống thấp: 0,26 GiB / 7,84 GiB lúc ghi trạng thái đầu vòng, do nhiều ứng dụng đang mở. GPU smoke vòng 39 từng
  xuống 17 MiB mà vẫn đạt. Paging có thể làm chậm so với dự toán. Không đóng ứng dụng.
- Nguồn:
  - đang cắm sạc;
  - gói High performance: sleep và hibernate khi rảnh là "không bao giờ";
  - không có khởi động lại nào đang chờ.

  Các tình huống vẫn làm dừng lượt: đóng nắp máy, rút sạc tới mức pin cạn, cập nhật Windows khởi động lại.
- Bị ngắt giữa epoch thì mất phần epoch đang chạy; resume chỉ từ sau epoch đã hoàn tất.
- Kết quả resume không trùng từng bit với lượt liên tục (giới hạn đã công khai từ vòng 39).

## 9. Bàn giao từng model

Mỗi model bàn giao: epoch hoàn tất, thời gian, exit code, checkpoint và sha256, kết quả dev (validation Ultralytics mỗi
epoch và đánh giá cuối `best.pt`), tài nguyên (RAM đúng PID và tiến trình con, RAM toàn máy, VRAM), lịch sử resume nếu
có. Sau đó dừng trước giai đoạn đánh giá test.
