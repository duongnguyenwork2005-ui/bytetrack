# Vòng 38 — ghi nhận nghiệm thu `0c0a07a`; xuất dataset YOLO đầy đủ (bước 5) và kiểm chứng (chi tiết)

**Phạm vi** (người dùng duyệt): xuất dataset YOLO đầy đủ từ BDD100K MOT train, **1 frame mỗi 5 frame**, theo split đã khóa;
kiểm chứng; bàn giao; **dừng**.

- Không training, không inference, không Nano.
- Không đổi model, split, chính sách nhãn hay dữ liệu nguồn.
- Không thêm seed hay ablation.

## 0. Trạng thái đầu vòng

Log: `logs/review38/00_trang_thai_dau.log`.

- HEAD `0c0a07a` (đúng target đã nghiệm thu), 0 commit sau đó.
- Working tree chỉ có file riêng của người dùng.
- `data/yolo_bdd5_v1` và `data/yolo_bdd5_v1.dang_tao` chưa tồn tại. D: còn 63,21 GB, RAM còn trống 1,15 GB.

## 1. Ghi nhận nghiệm thu `0c0a07a` (ĐÃ CHẠY)

Commit `9347141`.

- **Báo cáo** `D:\LABOTARY\review_codex\CODEX_REVIEW_0c0a07a.md`: 11.643 B, sha256 `b3330e63…`.
  - **PASS WITH LIMITATIONS** cho đính chính vòng 36 và pilot vòng 37.
  - Đóng **R35-F1 và R35-F2** về báo cáo.
  - Câu hỏi nghiên cứu warmup/mosaic chưa xác định, nhưng **không phải finding**; không mở thêm pilot hay ablation.
- **Bằng chứng**: ZIP 360.375 B, sha256 `86c596a4…`.
  - Giải nén vào thư mục mới; `SHA256SUMS.txt` khớp **91/91**, cả trên blob đã stage.
  - Lưu ở `handoff/review_evidence/CODEX_0c0a07a/` (`-text`).
- **Gói đã nghiệm thu**: 967.546 B, `a2985214…`, khớp. Lưu ở `handoff/archive/review_delta_0c0a07a.zip`.
- **Base review** `2351c67` → **`0c0a07a`** (vòng 37); 13 khai báo NPZ giữ nguyên.
- **Phạm vi reviewer**:
  - đã kiểm: 132 đường dẫn (áp ngược/xuôi), 11/11 kiểm cô lập hàm runner, 468 điều kiện đối chiếu số liệu/receipt, chạy lại
    tóm tắt và công cụ so khóa;
  - **không** làm: lặp lại training/inference GPU, mở `.pt`, xác minh toàn cây Git.

## 2. Kiểm nguồn và cấu hình xuất (ĐÃ CHẠY)

Commit `efd6c01`.

- `prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml --buoc kiem-nguon` → **exit 0**: 1.400 video
  nhãn = 1.400 video ảnh, 278.079 frame, 0 ảnh thiếu.
- **Hash**, chỉ đọc (`02_*`):

  | Mục | Giá trị |
  |:--|:--|
  | Cấu hình `bdd5_train.yaml` | `ca5d602b…` (không đổi từ vòng 35) |
  | File split | `b8a2848e…` |
  | `sha256_danh_sach` (tính lại) | `a59a8d06…` |
  | Vân tay nguồn nhãn (1.400 file) | `d2024493…`, khớp bản ghi trong split |
  | Số video | train-fit 1.261, dev 139, 0 chung |

- **Bước frame hiệu lực = 5**.
  - Xuất đầy đủ không dùng `--tap-con`, nên lấy `lay_mau.buoc_frame`.
  - Quy tắc của `bdd_yolo_export.chon_frame` (mã nguồn ghi kèm log, sha256 `65fc5681…`): giữ frame có chỉ số **0-based**
    theo video chia hết cho 5 (file `…-0000001.jpg`, `…-0000006.jpg`, …), **và** nằm trong khoảng [frame có nhãn đầu,
    frame có nhãn cuối] của video.
  - Không dịch điểm bắt đầu, không đổi quy tắc.
- **Tập đánh giá cấm**: BDD100K MOT val (bản COCO-video) và UA-DETRAC test. Không dùng làm dữ liệu huấn luyện.

## 3. Xuất dataset đầy đủ (ĐÃ CHẠY)

Commit `06cc7ed`.

```text
bash handoff/logs/review38/chay_xuat_day_du.sh
  -> python -u src/prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml --buoc xuat
```

- **exit 0**, wall **777 s**.
- Exporter đã nghiệm thu chạy theo thứ tự:
  1. lượt 1: chọn frame, băm ảnh nguồn và kiểm trùng **trước khi ghi**;
  2. lượt 2: ghi vào `data/yolo_bdd5_v1.dang_tao`;
  3. kiểm hợp đồng v2 (129,7 s);
  4. đổi tên thành `data/yolo_bdd5_v1/`.
- Theo dõi tiến trình và số ảnh mỗi phút; không chạy trùng.
- **Số lượng so với dự kiến**: **train-fit 50.487 ảnh / 1.261 video; dev 5.559 ảnh / 139 video**, khớp đúng số đếm vòng 35.
- **Đếm frame**: 278.079 frame nguồn = **56.046** chọn + 221.729 lệch bước + 304 ngoài khoảng có nhãn.
- **Đếm nhãn**:
  - 82.568 nhãn lớp khác, không gán nhãn (pedestrian, rider, train, other person);
  - 14 box bị cắt vào khung; 0 box < 1 px sau cắt;
  - 0 vùng tô xám rỗng; 160.178.277 pixel tô xám.
- **Hash đầu ra**: `data.yaml` `1280e57b…`, manifest `anh.csv` `1ee6174f…`, `tom_tat.json` `9eae5eb3…`, `video.json`
  `05140872…`.
- **Dung lượng**: **8.034.288.965 B (8,03 GB), 130.699 file**. D: 63,20 → **54,83 GB**. Kịch bản ước tính vòng 35 là
  ~8,46 GB.
- **RAM trong lúc xuất**: xem mục 4, ghi chú bộ đo.

## 4. Kiểm chứng dataset đã xuất (ĐÃ CHẠY)

Commit `c925b21`.

- **Hợp đồng dùng chung** (`--buoc kiem`, `04_*`): **exit 0, 0 lỗi**.
  - Đường dẫn Ultralytics resolve: `images/train`, `images/dev`.
  - Băm lại **56.046 ảnh, 56.046 nhãn, 18.600 file ignore**, khớp manifest v2. Nhãn rỗng hợp lệ (686 / 95) được phân biệt
    với file thiếu.
  - 0 `.npy` cạnh ảnh; bản sao split khớp `b8a2848e…`.
- **Kiểm độc lập** (`05_kiem_doc_lap.py`, chỉ đọc): **ĐẠT**.
  - Tập (split, video, frame) trong manifest **khớp từng phần tử** với tập tính lại bằng `chon_frame` từ nguồn nhãn:
    56.046 / 56.046, 0 thiếu, 0 thừa.
  - Mọi video ở đúng split, 0 video chung, không video nào của split thiếu ảnh.
  - 475.234 dòng box: lớp 0..4, tọa độ trong khung, 0 lỗi; 0 ảnh lệch số box với manifest.
  - `data.yaml`: `nc 5`, lớp `car, truck, bus, motorcycle, bicycle`.

**Thống kê**:

| | train-fit | dev |
|:--|--:|--:|
| Video / ảnh | 1.261 / 50.487 | 139 / 5.559 |
| Ảnh âm tính (nhãn rỗng hợp lệ) | 686 | 95 |
| Ảnh có vùng ignore (tô xám) | 17.040 | 1.560 |
| Box car | 383.387 | 42.211 |
| Box truck | 27.358 | 2.971 |
| Box bus | 10.708 | 984 |
| **Box motorcycle** | 2.235 | **226** |
| **Box bicycle** | 4.747 | **407** |
| Ảnh có car / truck / bus | 49.530 / 16.882 / 8.160 | 5.407 / 1.957 / 780 |
| **Ảnh có motorcycle / bicycle** | 1.822 / 3.598 | **205 / 334** |
| Video có car / truck / bus | 1.261 / 906 / 532 | 139 / 94 / 55 |
| **Video có motorcycle / bicycle** | 230 / 397 | **24 / 44** |

- **Số video có lớp** sau lấy mẫu thấp hơn thống kê split (đếm trên mọi frame). Ví dụ: dev motorcycle 26 → 24, bicycle 45 →
  44; train motorcycle 241 → 230. Lý do: một số video chỉ có lớp đó ở frame không được chọn. **Tập dev không bị thay đổi**
  dựa trên các con số này.
- **Kiểm trùng**, phạm vi đúng như chính sách:
  - train-fit ↔ dev: so **nội dung** (sha256 ảnh nguồn) trước khi ghi → **0 cặp**;
  - BDD100K MOT val: so **tên video, tên ảnh và nội dung** (đã băm 39.973 ảnh val) → **0 / 0**;
  - UA-DETRAC test: **chỉ so tên video** (40 video), **không** so nội dung ảnh.
- **Ghi chú bộ đo RAM** (`03_ghi_chu_ram_xuat.txt`):
  - bộ đo ngoài chọn tiến trình đầu tiên có tên script trong dòng lệnh. Nó đã bám **nhầm** shell của lệnh Monitor chạy song
    song (pid 23788), không phải exporter (pid 20304);
  - vì vậy cột RSS tiến trình của lần xuất **không hợp lệ**;
  - cột RAM hệ thống còn trống, đo toàn máy, vẫn hợp lệ: thấp nhất 449 MiB, trung vị 727 MiB;
  - không chạy lại xuất để đo lại.

## 5. Sẵn sàng cho quyết định training dài

Commit `553961f`.

- **Dry-run** (`--dry-run` = `kiem-cau-hinh`, **không train**, không tạo thư mục lượt): **exit 0** cho YOLOv8n và YOLO11n.
  - Runner nhận `data/yolo_bdd5_v1/data.yaml`, hợp đồng 0 lỗi, 50.487 / 5.559 ảnh, manifest `1ee6174f…`.
  - Tham số theo `train_bdd5.yaml`: **epochs 15 (đề xuất, chưa duyệt)**, patience 5, batch 8, imgsz 640, SGD lr0 0,01,
    warmup 1 epoch, close_mosaic 2, AMP tắt, workers 2, `cache: false`, seed 0.
  - **6.311 batch / 789 bước tối ưu mỗi epoch**; 94.665 batch cho 15 epoch.
- **Tài nguyên hiện tại** (`07_*`):
  - RAM còn trống **1.258 MiB / 8.029 MiB**. VS Code ~2,0 GB, ChatGPT ~1,1 GB (RSS cộng theo tên, có thể đếm trùng). Không đóng
    ứng dụng nào.
  - D: còn **54,83 GB**. VRAM đang dùng 393 MiB.
- **Dự toán**: số ảnh thực tế **trùng** số đã dùng ở vòng 35 (50.487 / 5.559), nên kịch bản **56,64 h** (chậm hơn **73,95 h**)
  cho hai model tuần tự, bước 5, 15 epoch, **không đổi** và không tính lại. Đây là kịch bản theo giả định, không phải bảo đảm.
  Pilot vòng 37 có s/batch 0,93 / 1,11, gần số smoke dùng trong dự toán.

**Lệnh training dự kiến** — CHƯA chạy, chờ duyệt epoch / ngân sách / thời điểm:

```bash
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
# 0. ngay truoc khi chay: nguoi dung dong bot ung dung; ghi RAM; kiem cau hinh lai (KHONG train)
$P src/train_detector.py --model yolov8n --dry-run
$P src/train_detector.py --model yolo11n --dry-run
# 1. YOLOv8n -- epochs theo train_bdd5.yaml (15, de xuat); bat buoc --xac-nhan-train-dai
$P -u src/train_detector.py --model yolov8n --che-do day-du --xac-nhan-train-dai 2>&1 \
  | $P handoff/logs/review37/do_tai_nguyen_ngoai.py moc-gio --ra <log>/train_yolov8n.log
# 2. YOLO11n -- SAU khi YOLOv8n xong (tuan tu, khong chay dong thoi)
$P -u src/train_detector.py --model yolo11n --che-do day-du --xac-nhan-train-dai 2>&1 \
  | $P handoff/logs/review37/do_tai_nguyen_ngoai.py moc-gio --ra <log>/train_yolo11n.log
# 3. danh gia dev noi bo cung quy tac cho 3 detector (sau khi train)
$P src/eval_detector_5lop.py --data data/yolo_bdd5_v1/data.yaml --split val \
   --model yolov8n_coco=models/yolov8n.pt \
   --model yolov8n_ft=models/finetune_phase2/train_bdd5_v1_yolov8n_day_du/weights/best.pt \
   --model yolo11n_ft=models/finetune_phase2/train_bdd5_v1_yolo11n_day_du/weights/best.pt \
   --out results/phase2/detector_eval/dev_bdd5_v1
```

**Rủi ro vận hành cần người dùng quyết định** (chưa làm gì):

1. **Thời lượng**: mỗi lượt khoảng 26–30 h theo kịch bản, dài hơn nhiều so với giới hạn ~2 h của tác vụ nền trong phiên
   Claude Code. Lượt dài nên do người dùng khởi chạy trong terminal riêng, hoặc chọn cách khác.
2. **Không có chế độ resume**: runner từ chối thư mục lượt đã có. Nếu bị ngắt (sleep, cập nhật Windows, lỗi), phải chạy lại
   với tên mới từ đầu. Thêm resume là thay đổi mã, cần duyệt.
3. **RAM**: hiện còn trống 1,26 GB; pilot từng xuống 7–59 MiB. Người dùng nên đóng bớt ứng dụng và ghi RAM trước khi chạy.
4. **Bộ đo RAM ngoài** cần chọn đúng tiến trình Python (hoặc nhận PID) trước lượt dài. Đây là đề xuất sửa nhỏ, chưa làm.

## 6. Quyết định chờ người dùng

1. Số epoch (đề xuất 15) và ngân sách training local (kịch bản 56,64 h / 73,95 h).
2. Thời điểm bắt đầu, cách khởi chạy (terminal riêng), có thêm resume hay không.
3. Đóng bớt ứng dụng trước khi chạy.
4. Sửa bộ đo RAM ngoài trước lượt dài (đề xuất).

## 7. Chưa chạy, giới hạn

- **NOT_RUN**: training dài (không bật `--xac-nhan-train-dai`), inference, Nano, full suite.
  - Vòng này không sửa mã, nên không có test nào cần chạy lại.
  - Chỉ thêm script kiểm chỉ đọc `05_kiem_doc_lap.py` và script chạy `chay_xuat_day_du.sh`.
- Kiểm trùng với UA-DETRAC test chỉ ở mức tên video.
- Nhãn train vẫn là snapshot bên thứ ba (phương án C), chưa xác thực bằng archive gốc. Lệch lớp/ignore ở native val so với COCO
  val lịch sử (183/50) chưa giải quyết; ngoài phạm vi.
- Dataset (8,03 GB) nằm ở `data/yolo_bdd5_v1/` (có `.gitignore` riêng), không commit. Dataset smoke và mọi checkpoint/kết quả
  trước đó giữ nguyên.

## 8. Commit vòng 38

| Commit | Nội dung |
|:--|:--|
| `9347141` | Ghi nhận nghiệm thu `0c0a07a`, bằng chứng, base → `0c0a07a` |
| `efd6c01` | `kiem-nguon` + kiểm hash split/nguồn nhãn, bước 5 |
| `06cc7ed` | Xuất dataset đầy đủ (log, script chạy) |
| `c925b21` | Kiểm hợp đồng chạy lại, kiểm độc lập, ghi chú bộ đo RAM |
| `553961f` | Dry-run hai model, tài nguyên hiện tại |
| commit chứa phụ lục này (target delta) | Brief, phụ lục, kế hoạch, inventory, handoff |
