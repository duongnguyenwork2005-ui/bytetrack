# Vòng 30 — ghi nhận nghiệm thu `bcb92f8` và kiểm kê BDD100K MOT train (chi tiết)

Base review mới: **`bcb92f8`**. Phạm vi: PC; không SSH hay đổi Nano v16, không training dài, không cloud, không thêm dataset,
model hay thí nghiệm. **Vòng này không sửa mã**, nên không có test nào cần chạy lại.

## 0. Ghi nhận nghiệm thu (đã chạy)

- **Trạng thái đầu vòng**: HEAD `bcb92f8` (= target được nghiệm thu), 0 commit sau đó. Working tree chỉ có file riêng của
  người dùng (`logs/review30/00_trang_thai_dau.log`).
- **Kết luận Codex** (`handoff/CODEX_REVIEW_bcb92f8.md`, 7.354 B, sha256 `26496ddc…`): **PASS WITH LIMITATIONS**, đóng R28-F1.
- **Bằng chứng Codex**:
  - `CODEX_REVIEW_bcb92f8_EVIDENCE.zip` (22.287 B, sha256 `ac5d066a…`): testzip sạch; `SHA256SUMS.txt` khớp **11/11**
    file, và blob đã commit cũng khớp 11/11;
  - giải nén vào `handoff/review_evidence/CODEX_bcb92f8/`; bản ZIP gốc để trong `handoff/archive/` (bị ignore);
  - script gốc `REPRO_REVIEW_07615ba.py` được lưu thêm vào `handoff/`.
- **Script Codex**: Codex đã **tự chạy nguyên** `REPRO_REVIEW_07615ba.py` trên target `bcb92f8`: **exit 0, `bug_count=0`**.
  - Lượt **NOT_RUN** trên máy tác giả ở vòng 29 được **giữ nguyên làm lịch sử**; phụ lục vòng 29 chỉ thêm một ghi chú bổ
    sung.
  - Không chạy lại smoke tổng hợp chỉ để đổi trạng thái log.
- **Delta đã nghiệm thu**: lưu thành `handoff/archive/review_delta_bcb92f8.zip`, 717.372 B, sha256 `5a931976…`, khớp báo
  cáo. Base review chuyển `239b9c6` → **`bcb92f8`** (`make_review_delta.py --set-base bcb92f8 --round 29`); 13 khai báo NPZ
  được mang sang.
- **Giới hạn reviewer đã ghi**:
  - chốt không băm lại pixel ở từng batch;
  - reviewer không chạy GPU hay checkpoint: lượt của reviewer là 76/76 cộng 6 SKIP;
  - phần “92/92” và smoke GPU là bằng chứng bàn giao được Codex đối chiếu.

## 1. Kiểm kê dữ liệu thật (đã chạy)

Chỉ kiểm các nơi đã biết, không quét toàn ổ đĩa:

- `data/raw/bdd100k` (cây thư mục) và `data/raw`;
- `D:\LABOTARY` — nơi chứa gói val đã tải;
- `~/Downloads`, kể cả file tải dở.

Log: `logs/review30/01_kiem_ke_bdd100k.{json,log}`.

| Thành phần | Trạng thái |
|:--|:--|
| Ảnh MOT val (`images/track/val`) | có: 200 video, 39.973 ảnh; archive `images20-track-val-1.zip` 4.983.938.716 B |
| Nhãn MOT val (COCO-video) | có: `labels_cocovid/bdd_box_track_val_cocofmt.json` |
| **Ảnh MOT train** (`images/track/train`) | ❌ **không có**; không có archive train ở các nơi đã kiểm |
| **Nhãn MOT train** (`labels/box_track_20/train` hoặc COCO-video train) | ❌ **không có** |
| Gói detection 100K | không có; cũng không nằm trong plan |
| File đang tải dở trong Downloads | 0 |
| Dung lượng `D:` | trống 126 GB / 228 GB |

`python src/prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml --buoc kiem-nguon` → **exit 2**,
liệt kê đúng hai đường dẫn thiếu (`logs/review30/02_kiem_nguon_bi_chan.log`). Không dùng val hay fixture thay thế và không tự
tải gói lớn. Gói cần tải và đường dẫn đặt: `BDD100K_INVENTORY.md`, mục “Vòng 30”.

## 2. Trạng thái các bước

| Bước | Trạng thái |
|:--|:--|
| Ghi nhận nghiệm thu, lưu bằng chứng, lưu trữ delta, chuyển base | **ĐÃ CHẠY** |
| Kiểm kê dữ liệu, `kiem-nguon` | **ĐÃ CHẠY**, kết quả là thiếu dữ liệu (exit 2) |
| Kiểm ảnh–nhãn, frame, bbox, ánh xạ 5 lớp, thống kê lớp/video của train | **BỊ CHẶN** — thiếu MOT train |
| Chia train-fit/dev theo video và khóa split + hash | **BỊ CHẶN** |
| Kiểm trùng: BDD val (tên video, tên ảnh, sha256), UA-DETRAC test (chỉ tên video), train-fit/dev (video + sha256 ảnh nguồn) | **BỊ CHẶN** — sẽ chạy trong bước xuất |
| Xuất dataset manifest v2, hợp đồng dữ liệu, `cache: false` | **BỊ CHẶN** |
| Smoke dữ liệu thật YOLOv8n + YOLO11n, 1 epoch, cùng tập con | **BỊ CHẶN** |
| Cập nhật ước lượng thời gian bằng số ảnh thật và tốc độ thật | **BỊ CHẶN**; ước lượng hiện tại vẫn là fixture |
| Test | **CHƯA CHẠY / không cần** — vòng này không sửa mã |
| Training dài, chọn nơi chạy, duyệt 15 epoch / ngân sách | **CHƯA CHẠY** — ngoài phạm vi, chờ người dùng |

## 3. Khi dữ liệu có mặt

Chạy theo đúng thứ tự trong `DETECTOR_TRAIN_PLAN.md` mục 6 (`kiem-nguon` → `chia` → commit split → `xuat --tap-con smoke` →
`xuat` → `kiem` → `--dry-run` → smoke hai model). Các điểm dừng đã định:

- **Trùng dữ liệu**: ảnh dev trùng nội dung ảnh train-fit, hoặc trùng tập đánh giá → exporter dừng. Báo bằng chứng và
  phương án; không tự loại ảnh, đổi split hay sửa nhãn.
- **Kích thước tập con smoke**: tối đa khoảng 336 ảnh train-fit và 84 ảnh dev theo cấu tạo, trong giới hạn ≤ 1.000 / ≤ 200.
  Xác nhận bằng `tom_tat.json` trước khi train; vượt thì dừng và hỏi. Hai model dùng cùng tập con, chọn trước khi xem AP.
- **Ghi lại trong kết quả smoke**: dữ liệu ultralytics thực nạp khớp manifest, `cache_anh None`, 0 `.npy`, loss hữu hạn,
  checkpoint nạp lại đúng 5 lớp, evaluator chạy trên dev smoke, cấu hình hiệu lực, hash, thời gian, ảnh/giây, VRAM và RAM.
  Runner đã ghi sẵn các mục này.

## 4. Giới hạn

- Ngân sách ~27 h (YOLOv8n) + ~32 h (YOLO11n) cho 15 epoch vẫn là **ước tính từ fixture** và số video BDD công bố, ±30%;
  **chưa duyệt**.
- Chưa biết số video/ảnh train thật, thống kê motorcycle trên train, hay nguồn thật có trùng train/dev không.
- Chưa chọn nơi chạy training dài.

## 5. File và commit

- `4d9c3f2`: ghi nhận nghiệm thu — `CODEX_REVIEW_bcb92f8.md`, `REPRO_REVIEW_07615ba.py`, `review_evidence/CODEX_bcb92f8/`,
  `review_base.json`, ghi chú bổ sung trong `REVIEW29_CHI_TIET.md`, `CODEX_HANDOFF.md` hàng 25, `logs/review30/00_*`.
- `4d24521`: kiểm kê — `logs/review30/01_*`, `02_*`.
- Commit tài liệu (commit này): `REVIEW30_CHI_TIET.md`, `REVIEW_BRIEF.md`, `BDD100K_INVENTORY.md` (mục Vòng 30),
  `DETECTOR_TRAIN_PLAN.md` (trạng thái, tập con smoke), `PLAN_DA_CHOT.md`, `CODEX_HANDOFF.md` hàng 26.
