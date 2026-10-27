# R52B — cấu hình, tập smoke, smoke 1 epoch và kế hoạch lượt training dài mixed5 (ghi TRƯỚC khi khởi chạy)

Ngày 2026-10-16. Lượt dài **chưa chạy**: chỉ khởi chạy sau khi R52B được nghiệm thu và người dùng cho phép, theo đúng
mục 5–8 dưới đây. Mẫu: `handoff/logs/review42/02_ke_hoach_luot_dai.md`.

## 1. Cơ sở (quyết định người dùng 16/10/2026)

- Huấn luyện lại **YOLOv8n rồi YOLO11n**, tuần tự, trên dataset trộn `data/yolo_mixed5_v1/` (R52A).
- Khởi tạo COCO, **15 epoch**, siêu tham số **y hệt** `train_bdd5.yaml`. Không tự tuning.
- Chọn checkpoint / dừng sớm chỉ trên dev nội bộ (`val` = BDD dev 139 video + UA-DETRAC dev12). BDD val và 40 video
  UA-DETRAC test không tham gia.
- Không thuộc lượt này: đánh giá detection fine-tune, tracking.

## 2. Cấu hình `configs/phase2/detector/train_mixed5.yaml`

- sha256 `2dcb31ef…bdd00`. Là bản sao của `train_bdd5.yaml` (sha256 `a466a4fa…553b`, không sửa). Chỉ đổi:
  - `ten`: `train_bdd5_v1` → `train_mixed5_v1`;
  - `dataset.day_du`: → `data/yolo_mixed5_v1/data.yaml`;
  - `dataset.smoke`: → `data/yolo_mixed5_v1_smoke/data.yaml`.
- Phần chú thích đầu file được viết lại cho mixed5. Chú thích "DE XUAT (chua duyet) … ~27 h" cạnh `epochs: 15` chép
  nguyên từ bản gốc và là lịch sử: phê duyệt nằm ở quyết định người dùng ngày 16/10, dự toán mới ở mục 4.
- Bằng chứng so tự động: `python -X utf8 src/mixed5/so_cau_hinh_train.py`, exit 0 →
  `results/phase2/mixed5/so_cau_hinh_train.json`.
  - Cả hai file có 54 khóa lá; 51 khóa trùng cả giá trị lẫn kiểu (1.0 khác 1).
  - 3 khóa khác đúng là `ten`, `dataset.day_du`, `dataset.smoke`, và giá trị mới đúng kế hoạch.
  - Các giá trị chốt được kiểm riêng: epochs 15, batch 8, SGD, lr0 0,01, lrf 0,01, momentum 0,937, weight_decay 0,0005,
    patience 5, seed 0, deterministic true, amp false, cache false, imgsz 640, pretrained true, smoke.epochs 1.
  - sha256 checkpoint khởi tạo trên đĩa trùng khai báo: `models/yolov8n.pt` `f59b3d83…`,
    `models/candidates/yolo11n.pt` `0ebbc80d…`. Runner kiểm thêm: checkpoint COCO 80 lớp.
  - Test: `TestR52BCauHinh`, 3 test, trong `src/mixed5/test_mixed5.py`. Test bắt được khóa thêm, giá trị sai, đổi kiểu,
    và khóa được phép nhưng sai giá trị.

## 3. Tập smoke `data/yolo_mixed5_v1_smoke/` (CHỈ kiểm đường chạy)

- Mã: `src/mixed5/xuat_smoke_mixed5.py --buoc xuat` rồi `--buoc kiem`, đều exit 0. Bằng chứng:
  `results/phase2/mixed5/smoke_dataset/kiem_hop_dong_smoke.json`.
- Quy tắc:
  - **BDD**: dùng nguyên tập con smoke đã định nghĩa trong `bdd5_train.yaml` `tap_con.smoke` (16 video train-fit / 4 video
    dev, bước frame 10, seed 0), lấy từ `data/yolo_bdd5_v1_smoke`.
    - Tập video được tính lại bằng `bx.chon_tap_con` và trùng `video.json` nguồn.
    - Dataset nguồn đạt hợp đồng trước khi dùng (manifest `c4899826…`). Split BDD nguồn có sha256 `b8a2848e…`, trùng
      phần BDD của `mixed5_v1.json`.
  - **UA-DETRAC**: 3 video train + 1 video dev. Chọn bằng `np.random.default_rng(0).permutation` trên danh sách đã sắp
    xếp của `mixed5_v1.json`, cùng cách với `bx.chon_tap_con`.
    - Train: `MVI_20032`, `MVI_20034`, `MVI_40131`. Dev: `MVI_40992`, thuộc dev12.
    - Lấy mọi dòng của các video này trong manifest `mixed5_v1`, nên giữ đúng quy tắc của dataset đầy đủ: `num % 10 == 0`
      và có `<frame>`.
  - File: 916/916 ảnh là hardlink tới dataset nguồn, 0 file sao chép. Có kiểm 0 JPEG thiếu EOI trước khi hardlink. Dòng
    manifest giữ nguyên giá trị cột của manifest nguồn.
  - Split khóa của manifest: `mixed5_v1.json` (sha256 `0acc018b…664b`).
- Kết quả `bx.kiem_hop_dong_dataset`: `loi: 0`, `tap_con: smoke`, manifest `92d4cd2f…289d`, data.yaml `5a3f9194…9b61`.
  - Train 621 ảnh / 19 video; dev 295 ảnh / 5 video. Có bảng chi tiết ở mục 3a.
  - Đã băm 916 ảnh, 916 nhãn và 650 file ignore.
  - So với nguồn: 0 dòng lệch, 0 file khác byte.
  - Rò rỉ với UA-DETRAC test (56.340 ảnh / 40 video) và BDD val (39.973 / 200): trùng sha256 nguồn 0, trùng sha256 xuất 0,
    trùng video 0. Trùng train/dev: 0.
  - Kiểm lại **sau** hai lượt smoke GPU vẫn `loi: 0` và 0 file khác byte. Ultralytics chỉ thêm `labels/train.cache` và
    `labels/dev.cache`; hợp đồng không tính hai file này.

### 3a. Thống kê tập smoke (box theo lớp)

| nguồn | split | video | ảnh | âm tính | tô xám | car | truck | bus | motorcycle | bicycle |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bdd | train | 16 | 334 | 0 | 125 | 2.316 | 204 | 100 | 31 | 65 |
| detrac | train | 3 | 287 | 0 | 287 | 2.318 | 0 | 360 | 0 | 0 |
| bdd | dev | 4 | 84 | 4 | 27 | 542 | 6 | 4 | 0 | 0 |
| detrac | dev | 1 | 211 | 0 | 211 | 501 | 0 | 0 | 0 | 0 |

Ba video UA-DETRAC train trong smoke không có box `truck` (`others`). Smoke chỉ kiểm đường chạy, không kiểm phân bố lớp.

## 4. kiem-cau-hinh, smoke 1 epoch, dự toán

### 4a. kiem-cau-hinh trên dataset đầy đủ (không train)

```text
python -X utf8 src/train_detector.py --config configs/phase2/detector/train_mixed5.yaml --model yolov8n --che-do kiem-cau-hinh --project models/finetune_mixed5 --ket-qua-dir results/phase2/mixed5/smoke
python -X utf8 src/train_detector.py --config configs/phase2/detector/train_mixed5.yaml --model yolo11n --che-do kiem-cau-hinh --project models/finetune_mixed5 --ket-qua-dir results/phase2/mixed5/smoke
```

- Cả hai lệnh exit 0 (`[OK] kiem cau hinh xong, KHONG train`). Theo runner, chế độ này chỉ in ra stdout và không ghi
  file. Số liệu dưới đây chép từ stdout.
- Hợp đồng `mixed5_v1`: `loi 0`, manifest `f2df97da…7c13`, data.yaml `d7828268…6285`, split `0acc018b…664b` (bản gốc
  khớp).
  - 57.043 ảnh train / 1.309 video; 7.190 ảnh dev / 151 video.
  - Đã băm 64.233 ảnh + 64.233 nhãn + 25.867 file ignore, mất 154,5 / 154,7 s.
- Tham số hiệu lực đúng cấu hình. Giá trị suy ra: accumulate 8, weight decay hiệu lực 0,0005, 7.131 batch/epoch,
  106.965 batch nếu chạy đủ 15 epoch.
- Thư mục lượt sẽ dùng chưa tồn tại: `models/finetune_mixed5/train_mixed5_v1_<M>_day_du`.

### 4b. Smoke 1 epoch (GPU)

```text
python -X utf8 src/train_detector.py --config configs/phase2/detector/train_mixed5.yaml --model yolov8n --che-do smoke --project models/finetune_mixed5 --ket-qua-dir results/phase2/mixed5/smoke
python -X utf8 src/train_detector.py --config configs/phase2/detector/train_mixed5.yaml --model yolo11n --che-do smoke --project models/finetune_mixed5 --ket-qua-dir results/phase2/mixed5/smoke
```

Kết quả: `results/phase2/mixed5/smoke/train_mixed5_v1_<M>_smoke.json`. Tóm tắt có kiểm tự động:
`src/mixed5/du_toan_luot_dai.py` → `results/phase2/mixed5/smoke/tom_tat_smoke.json` (`dat: true`).

| | YOLOv8n | YOLO11n |
|:--|--:|--:|
| exit / dòng cuối runner | 0 / `[DAT] smoke yolov8n` | 0 / `[DAT] smoke yolo11n` |
| `loi` / lỗi callback | None / None | None / None |
| loss epoch 1 (box / cls / dfl), hữu hạn | 1,482 / 2,681 / 1,127 | 1,523 / 3,123 / 1,207 |
| batch | 78 | 78 |
| s/batch (trung vị, bỏ 5 batch đầu) | 0,9548 | 1,1406 |
| train_s / val_s / toàn lượt | 79,3 / 17,6 / 124,7 s | 90,25 / 3,69 / 114,6 s |
| torch max allocated / reserved | 1.704 / 2.134 MiB | 1.948 / 2.372 MiB |
| nvidia-smi max (cả GPU) | 2.442 MiB | 2.663 MiB |
| GPU util trung vị | 96 % | 96 % |
| RAM hệ thống còn trống thấp nhất | **119 MiB** | **49 MiB** |
| last.pt / best.pt nạp lại, đúng 5 lớp `car,truck,bus,motorcycle,bicycle` | có / có | có / có |
| tham số checkpoint lệch cấu hình | 0 | 0 |
| đánh giá dev smoke (`eval_detector_5lop`, best.pt, 295 ảnh) | chạy được, mAP50 0,167 | chạy được, mAP50 0,127 |

- Kiểm cấu hình khi nạp: runner gọi lại hợp đồng dataset trước khi nạp model. Callback `on_pretrain_routine_end` đối chiếu
  ảnh + nhãn Ultralytics thực nạp với dataset đã xác minh, và cả hai lượt đều qua.
- Ultralytics in "Remapped 5/5 cls head rows from pretrained weights by class name".
- mAP dev smoke chỉ chứng minh đánh giá **chạy được**. Không dùng để so sánh model hay dự đoán chất lượng.
- `val_s` của YOLOv8n (17,6 s) lớn hơn YOLO11n (3,69 s). Có thể lượt đầu phải dựng cache nhãn / khởi động; chưa tách
  nguyên nhân. Dự toán không dùng `val_s` smoke.

### 4c. Dự toán lượt dài — **kịch bản theo giả định, không phải cận trên hay bảo đảm**

Script `src/mixed5/du_toan_luot_dai.py`, exit 0 → `results/phase2/mixed5/smoke/du_toan_luot_dai.json`. Mọi giả định
ghi trong file.

- 7.131 batch/epoch. Val/epoch = trung vị (val_s / ảnh dev) của lượt dài BDD-only cùng model × 7.190 ảnh dev.
- Khởi tạo mỗi lần chạy runner: 2 × 155 s băm hợp đồng + 5 phút quét nhãn = 610 s.
- Kịch bản A dùng s/batch smoke. Kịch bản B dùng s/batch hiệu dụng (train_s / so_batch, trung vị) của lượt dài BDD-only
  đã chạy trên cùng máy: `results/phase2/detector_train/train_bdd5_v1_<M>_day_du*.json`.
- Kịch bản chậm hơn: ×1,3 cho rủi ro chưa đo (giảm xung khi chạy nhiều giờ, RAM thấp / paging, mất epoch dở khi ngắt).
- **Chưa tính**: dừng sớm (`patience 5`), thời gian chờ giữa các lần resume.

| Model | h/epoch (A / B) | 15 epoch A | 15 epoch B | chậm hơn (A / B) |
|:--|--:|--:|--:|--:|
| YOLOv8n | 1,91 / 1,89 | 28,9 h | 28,5 h | 37,5 / 37,1 h |
| YOLO11n | 2,28 / 2,28 | 34,3 h | 34,3 h | 44,7 / 44,6 h |
| **Tổng tuần tự** | | **63,2 h** | **62,8 h** | **82,2 / 81,7 h** |

Đối chiếu: lượt BDD-only 50.487 ảnh có dự toán 56,64 h. Dataset trộn nhiều hơn 13 % ảnh, nên khoảng 2,6 ngày GPU liên tục
là quy mô hợp lý. Thời gian thực vẫn có thể vượt cả kịch bản chậm hơn.

### 4d. Đĩa

- Tập smoke dùng toàn hardlink, gần như không tốn thêm đĩa.
- Mỗi lượt dài ghi vào `models/finetune_mixed5/`: checkpoint ~6 MB, bản khôi phục mỗi epoch `khoi_phuc/last_eNNN.pt`
  ~15 × vài chục MB, cùng các file `.cache` nhãn.
- Kiểm `D:` còn ≥ 15 GB trước khi khởi chạy. Bộ theo dõi ghi dung lượng đĩa.

## 5. Tên lượt và thư mục (đều mới; runner và bộ giám sát tự từ chối ghi đè)

| Model | Thư mục lượt | Kết quả runner | Thư mục giám sát |
|:--|:--|:--|:--|
| YOLOv8n | `models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/` | `results/phase2/mixed5/detector_train/train_mixed5_v1_yolov8n_day_du.json` | `results/phase2/mixed5/detector_train/giam_sat/yolov8n_lan0/` |
| YOLO11n | `models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/` | `results/phase2/mixed5/detector_train/train_mixed5_v1_yolo11n_day_du.json` | `results/phase2/mixed5/detector_train/giam_sat/yolo11n_lan0/` |

- Resume lần N ghi `…__tiep_tuc_N.json`; giám sát ở `giam_sat/<M>_tiep_tuc_<N>/`.
- Không ghi vào `models/finetune_phase2/`, `results/phase2/detector_train/` hay `results/phase2/detector_smoke/` cũ.
- Mọi lệnh phải truyền **cùng** `--config`, `--project`, `--ket-qua-dir`, kể cả khi resume. Nếu thiếu, runner rơi về mặc
  định BDD-only.

## 6. Khởi chạy (sau khi được duyệt)

1. Trước khi chạy, kiểm và ghi lại:
   - cắm sạc, gói nguồn không sleep;
   - không có Windows Update khởi động lại đang chờ;
   - `D:` còn ≥ 15 GB;
   - không có `train_detector.py` nào đang chạy;
   - RAM trống. Smoke R52B xuống 49–119 MiB; nên đóng bớt ứng dụng. Không tự đóng ứng dụng của người dùng.
2. Kiểm cấu hình lần cuối, exit 0:

   ```text
   python -X utf8 src/train_detector.py --config configs/phase2/detector/train_mixed5.yaml --model yolov8n --che-do kiem-cau-hinh --project models/finetune_mixed5 --ket-qua-dir results/phase2/mixed5/detector_train
   ```

3. Khởi chạy độc lập qua WMI, dùng `handoff/logs/review52/khoi_chay_wmi_mixed5.ps1`. Script này là bản sao của script
   vòng 42, chỉ thêm 3 cờ mixed5 và đổi thư mục giám sát. Cwd là gốc repo:

   ```text
   powershell -NoProfile -ExecutionPolicy Bypass -File handoff\logs\review52\khoi_chay_wmi_mixed5.ps1 -Model yolov8n -GiamSat yolov8n_lan0
   ```

   Lệnh mà script tạo:

   ```text
   python.exe -u src\chay_giam_sat.py chay --thu-muc results\phase2\mixed5\detector_train\giam_sat\yolov8n_lan0 --khong-in -- ^
       python.exe -u src\train_detector.py --config configs\phase2\detector\train_mixed5.yaml --model yolov8n ^
       --che-do day-du --xac-nhan-train-dai --project models\finetune_mixed5 --ket-qua-dir results\phase2\mixed5\detector_train
   ```

   - Cha của tiến trình là `WmiPrvSE.exe`, ngoài cây tiến trình Claude / VSCode, không nằm trong Job Object. Đây không
     phải tác vụ nền của Claude, vốn bị cắt sau khoảng 2 giờ.
   - Script tự từ chối chạy khi thư mục giám sát đã tồn tại hoặc đang có `train_detector.py` chạy.
4. Chỉ báo "đã bắt đầu" khi thấy đủ:
   - PID công cụ và PID runner còn sống (`trang_thai.json`), command line đúng;
   - `log.txt` tăng;
   - thư mục lượt được tạo;
   - sự kiện `bat_dau` trong `nhat_ky_luot.jsonl`;
   - tiến độ epoch 1 trong log.
5. YOLO11n: cùng quy trình với `-Model yolo11n -GiamSat yolo11n_lan0`, **chỉ sau** khi YOLOv8n đạt mục 8. Không chạy
   đồng thời hai model.

## 7. Theo dõi (chỉ đọc) và phục hồi

- Theo dõi bằng `handoff/logs/review42/theo_doi_luot.py`. Script nhận đường dẫn qua tham số, nên dùng được cho mixed5:

  ```text
  python handoff/logs/review42/theo_doi_luot.py --giam-sat results/phase2/mixed5/detector_train/giam_sat/yolov8n_lan0 --run-dir models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du --log results/phase2/mixed5/detector_train/theo_doi_yolov8n.log
  ```

  - Ảnh chụp ghi lúc bắt đầu, mỗi epoch, mỗi 30 phút, khi kết thúc / bị ngắt / đứng yên.
  - Mỗi ảnh chụp gồm PID, epoch đã lưu, fitness, tuổi `log.txt`, RAM, VRAM, nhiệt độ, pin, đĩa.
  - Một lượt chờ hết giờ **không** có nghĩa training thất bại.
- `kiem_thanh_cong.py`, `tom_tat_luot.py`, `dong_goi_ket_qua.py` của vòng 42 **ghi cứng** `models/finetune_phase2` và
  `results/phase2/detector_train`, nên **không** dùng trực tiếp cho mixed5. Cần bản tham số hóa (đề xuất, chưa làm) hoặc
  kiểm tay theo mục 8.
- Trong lúc train: không sửa mã, không commit, không mở thêm thí nghiệm sau khi xem metric.
- **Phục hồi khi bị ngắt**:
  1. Giữ nguyên mọi log. Xác định trạng thái bằng `python src/chay_giam_sat.py trang-thai …`, không đoán.
  2. Kiểm điều kiện tiếp tục (không train, không ghi):

     ```text
     python -X utf8 src/train_detector.py --config configs/phase2/detector/train_mixed5.yaml --model <M> --dry-run --tiep-tuc --project models/finetune_mixed5 --ket-qua-dir results/phase2/mixed5/detector_train
     ```

  3. Chỉ khi bước 2 đạt mới chạy
     `khoi_chay_wmi_mixed5.ps1 -Model <M> -GiamSat <M>_tiep_tuc_<N> -TiepTuc`. Script này thêm `--tiep-tuc` vào cùng bộ
     cờ mixed5.
  4. Không chạy lại từ đầu, không tăng epoch (`epochs = 15` là tổng của lượt, kể cả khi resume), không đổi cấu hình /
     batch / workers.
- **Điều kiện dừng**, dừng rồi báo kèm bằng chứng:
  - lỗi lặp lại cùng nguyên nhân;
  - dry-run `--tiep-tuc` từ chối;
  - loss không hữu hạn;
  - hợp đồng dataset hoặc callback đối chiếu nhãn báo lệch;
  - `D:` dưới 15 GB;
  - nguyên nhân ngoài tầm kiểm soát: điện, RAM, cập nhật Windows. Khi đó báo thao tác người dùng cần làm.
- **Kết thúc hợp lệ**: `patience 5` dừng sớm (`dung = true` ở epoch cuối), hoặc chạy đủ 15 epoch. Không chạy thêm.

## 8. Điều kiện để bắt đầu YOLO11n

YOLOv8n phải kết thúc thành công:
- bộ giám sát exit 0;
- runner in `[DAT] day-du yolov8n`; kết quả có `loi = None`, không lỗi callback, loss hữu hạn;
- `last.pt` / `best.pt` nạp được, đúng 5 lớp, tham số trùng cấu hình;
- `nhat_ky_luot.jsonl` có sự kiện `ket_thuc` với `hoan_tat`, `kiem_sau_train_dat = true`, `exit_code = 0`;
- đủ 15 epoch, hoặc ít hơn nếu dừng sớm hợp lệ.

## 9. Rủi ro đã biết

- **RAM**: còn trống thấp nhất 119 MiB (v8n) và 49 MiB (11n) trong smoke, dù máy 7,8 GB. Lượt BDD-only từng chạy được trong
  điều kiện tương tự, có resume. Paging có thể làm chậm hơn dự toán.
- **Nguồn**: rút sạc → Power saver → máy ngủ (bài học vòng 42). Đóng nắp, cạn pin, cập nhật Windows đều làm dừng lượt.
- Bị ngắt giữa epoch thì mất phần epoch đó; resume chỉ chạy tiếp từ sau epoch đã hoàn tất. Kết quả resume không trùng từng bit
  với lượt liên tục.
- Phân bố nguồn: UA-DETRAC chiếm 11,49 % ảnh train (R52A). Xe hai bánh trên UA-DETRAC không có nhãn (quyết định người
  dùng); thiệt hại đo ở bước đánh giá sau.

## 10. Bàn giao từng model (sau lượt dài)

Mỗi model bàn giao:
- epoch đã hoàn tất, thời gian, exit code;
- checkpoint kèm sha256;
- kết quả dev: validation Ultralytics mỗi epoch và đánh giá cuối `best.pt`;
- tài nguyên: RAM đúng PID và tiến trình con, RAM toàn máy, VRAM;
- lịch sử resume, nếu có.

Sau đó dừng, trước giai đoạn đánh giá BDD val / UA-DETRAC test.
