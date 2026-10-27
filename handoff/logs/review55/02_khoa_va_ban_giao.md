# R55B — Tiền kiểm, khóa R55 và bàn giao (chỉ chuẩn bị)

Ngày 2026-10-21. Nhánh `phase2-mixed-traffic`, HEAD `117d62d`, không commit. Mục 0 `handoff/PLAN_DA_CHOT.md` đã cập
nhật **trước** file này. R55 chỉ chuẩn bị; **không** có cache test40, suy luận, tracking, chấm, MOT hay metric dữ liệu
thật mới. Chờ Codex review R55B; không chuyển sang full.

## 1. Căn cứ

- Plan do Codex đề xuất sau lượt `bytetrack_r54_test40_prep_20261020_145514` (lời gọi `0016_planner`), người dùng duyệt.
- R55A được Codex approve: `D:/LABOTARY/automation/runs_local/bytetrack_r55_prepare_20261021_051232/reports/review_003.json`
  (F1, F6 đóng; trước đó `review_001` revise F1–F5, `review_002` revise F1 + F6).
- R54: `review_003.json` approve R54A (khóa `khoa_20261020_211635_887200_pid10816`, `khoa.json`
  `e2d1b0d52e1ca45fcb98de46e273f4dc16e48a3bcc7a9be1cfc7a9a418e16b4f`), `review_005.json` approve smoke foreground R54B,
  `review_006.json` approve tài liệu Paad1df4f — trong
  `D:/LABOTARY/automation/runs_local/bytetrack_r54_test40_prep_20261020_145514/reports/`. Không review nào duyệt full.
- Quyết định chuyển tiếp: P1980cb21 hoãn phép đo PF500/UKF đoạn dày khỏi R54 và yêu cầu đưa vào plan lượt dài;
  Pc5fc63da/P482c0392 từ chối thêm bước vào R54, không hủy nội dung chuyển tiếp; P7e7ee3c8 từ chối riêng kiểm
  `setup_model` trên CPU trước smoke — R55 không chạy kiểm đó. Quyết định ở lượt này: P6fb1cad7 (ca tiêm module), P9857b26c
  (marker nguyên tử), Pb73a427f (đo chi phí kiểm module mỗi đơn vị) đều **bị từ chối** — không làm.

## 2. Lệnh đã chạy (foreground, tuần tự, chờ exit)

| # | Lệnh (`C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r55_tracking/dieu_phoi_test40.py …`) | Exit | Bằng chứng |
|---|---|---|---|
| 1 | `--buoc tien-kiem` | 0 (`dat`, 0 lệch, 20,5 s) | `results/phase2/r55_tracking/chuan_bi/tien_kiem_20261021_121436_943200_pid22720/` |
| 2 | `--buoc khoa --tien-kiem tien_kiem_20261021_121436_943200_pid22720 --tests lan_20261021_115928_508900_pid21264` | 0 | `results/phase2/r55_tracking/chuan_bi/khoa_20261021_121500_644100_pid22868/` |
| 3 | `--buoc kiem-khoa --khoa khoa_20261021_121500_644100_pid22868 --luu chuan_bi` | 0 (376 + 368 file, 0 lệch) | `results/phase2/r55_tracking/chuan_bi/kiem_khoa_20261021_121515_485600_pid17620/ket_qua.json` |

Tự kiểm thêm sau khi sửa mục 0 và viết file này: `--buoc kiem-khoa --khoa khoa_20261021_121500_644100_pid22868` (không
`--luu`, chỉ đọc, không ghi file) → exit 0, 376 + 368 file, 0 lệch.

Tên phiên ở bước 2–3 lấy từ dòng in của bước trước; phiên test là phiên fixture R55A đã được review
(`lan_20261021_115928_508900_pid21264`, 24/24, exit 0). Không chọn "phiên mới nhất".

## 3. Tiền kiểm (`tien_kiem.json` SHA256 `a88426a1338bc8d6692267d07085abb8ddc55b8532a4e2f7249a99ee07b946a0`)

- **Khóa R54:** băm lại 368 file, 0 lệch, `khoa.json` `e2d1b0d5…a418e16b4f`.
- **Checkpoint/names (đọc CPU, không suy luận, không `setup_model`):** SHA256 trùng giao thức và khóa R53:
  `yolov8n_coco` `models/yolov8n.pt` `f59b3d83…fc83b36` (coco80, ánh xạ {1,2,3,5,7} → 5 lớp);
  `yolov8n_mixed5` `…/train_mixed5_v1_yolov8n_day_du/weights/best.pt` `542b1381…6b0694` (lop5);
  `yolo11n_mixed5` `…/train_mixed5_v1_yolo11n_day_du/weights/best.pt` `9a5e34a1…a386fb` (lop5).
- **Dữ liệu:** manifest `results/phase2/manifests/detrac_test.json` `ff2b4b9c…3077f47c`, fingerprint R47 `dd0f2a0e…156343a`,
  bảng hash ảnh R47 `3cea3488…8717b166`, bảng D4 `4d731253…85723551`; `bang_tien_kiem.csv` 40 video
  (`c026e173…87d2c793`). Tổng: **56.340 ảnh** (stat + size đúng cả 56.340), seqLength 56.340, **173 ảnh D4**, 56.167 frame
  có GT, 675.774 dòng GT, 163 vùng ignored; GT/seqinfo/ignored_regions của 40 video băm và đối chiếu khóa R54.
- **Ảnh — phạm vi kiểm:** kế thừa kiểm stat/size theo bảng hash R47 và **băm lại 256 ảnh mẫu cố định R54**
  (MVI_39031 và MVI_39051, frame 1–128; `mau_anh.json` `a902a88d…76844dc`, danh sách `644f14d7…aff7b`).
  **Không băm lại toàn bộ 56.340 ảnh** ở tiền kiểm; công việc cache (đường thực thi) băm từng ảnh ngay trước suy luận.
- **Cấu hình hiệu lực:** backend tracker `ultralytics`, `track_buffer` 30 bước, 4 bộ lọc (CV/EKF/UKF/PF500 seed 0/1/2)
  trùng khóa R54. `BYTETRACK_BACKEND` không đặt.
- **Thư viện (trùng khóa R54):** Python 3.11.9, ultralytics 8.4.141, torch 2.6.0+cu124, torchvision 0.21.0+cu124,
  trackeval 1.3.0, numpy 2.4.6, pandas 2.3.3, opencv-python 5.0.0.93, scipy 1.17.1, lap 0.5.13, fastparquet 2026.5.0;
  filterpy không cài. Cấu hình Ultralytics chỉ trong `_cau_hinh_thu_vien/` của từng phiên.
- **Vùng ghi:** `results/phase2/r55_tracking/thuc_thi/` chưa tồn tại (không tạo ở R55B).
- **Dung lượng đĩa (kịch bản theo giả định, không bảo đảm):** kích thước file smoke R54 lần 2
  (`results/phase2/r54_tracking/smoke/lan_20261021_002054_245100_pid16684`, chỉ `stat`, **không đọc NPZ**: cache 64.412 B,
  trackers 6.286.577 B, trackeval 792.522 B) × 220,08 (tỉ lệ ảnh) → cần ~3,13 GB; trống ~44,5 GB. Giả định: tuyến tính theo
  số ảnh, 128 frame đầu đại diện mật độ (chưa kiểm), chấm sao MOT (×2), không tính `lap-tracker`/`cham-lai`/`hau-kiem`.

## 4. Khóa R55

- Phiên `results/phase2/r55_tracking/chuan_bi/khoa_20261021_121500_644100_pid22868/`, `khoa.json` SHA256
  **`5a07efb8485466ef71366b94c6ae10051df22dacf606e8b6558eef204cf6e42b`** (ghi trong `SHA256SUMS`, `thoi_diem_khoa.json`).
- Nối tới: khóa R54 (thư mục + `khoa_json_sha256` `e2d1b0d5…a418e16b4f`, 368 file), giao thức R54
  `configs/phase2/r54_tracking/giao_thuc_test40.json` `eb1f8f3087401eaf3641c1bc215a864ad23850ec5ddd6970d7d74280715431ad`,
  hợp đồng `configs/phase2/r55_tracking/hop_dong_dieu_phoi.json` `626dd24dd9d753dc7c370cbc7cc3a4809f3447b0e5a3e1ab1789db9d7df12c76`;
  ma trận 36 run; thứ tự công việc cache → xác nhận cache → tracking → chấm → hậu kiểm.
- **376 file** = 368 file khóa R54 (băm lại, phải trùng; gồm checkpoint, dữ liệu/GT/seqinfo/ignored 40 video, YAML
  tracker, mã TrackEval/Ultralytics/phụ thuộc, RECORD thư viện) + 8 file R55:

| Vai trò | File | SHA256 |
|---|---|---|
| dieu_phoi_r55 | `src/r55_tracking/dieu_phoi_test40.py` | `b9e7f19941296bee8ff236fd67ee2687313b8d6c68765d52dc0b7b013f23c047` |
| test_r55 | `src/r55_tracking/test_dieu_phoi_test40.py` | `9f8a1bcd09e7140a1ea511aea72f0cd3a585485d1d0f99c461bd0a7f6859af05` |
| hop_dong_r55 | `configs/phase2/r55_tracking/hop_dong_dieu_phoi.json` | `626dd24dd9d753dc7c370cbc7cc3a4809f3447b0e5a3e1ab1789db9d7df12c76` |
| thiet_ke_r55 | `handoff/logs/review55/01_thiet_ke_dieu_phoi.md` | `9687aae739813f10ca4be5e010f4abe57b319c4047ead1898b3c65432419a598` |
| ket_qua_test_r55 | `results/phase2/r55_tracking/tests/lan_20261021_115928_508900_pid21264/ket_qua_test.json` (24/24) | `f7972339908f8c2dd71cc7cbc0b6be4d6d7f57234d8cac54a3e72ecab938e7a3` |
| tien_kiem_r55 | `…/tien_kiem_20261021_121436_943200_pid22720/tien_kiem.json` | `a88426a1338bc8d6692267d07085abb8ddc55b8532a4e2f7249a99ee07b946a0` |
| tien_kiem_r55 | `…/bang_tien_kiem.csv` | `c026e173f80a213c8bb80a196b3d49fca6348fef1de32092a5ec22da87d2c793` |
| tien_kiem_r55 | `…/mau_anh.json` | `a902a88d1459e3254bb6fc689adb1fb531e6237a07f873d1bd4e3b7da76844dc` |

- Kết quả test chỉ định có SHA mã/test/hợp đồng trùng bản hiện hành (bộ điều phối kiểm trước khi khóa); thiết kế được
  băm ở bản hiện hành (gồm mục 13–14). Module src/Ultralytics/TrackEval nạp khi import mã thực thi và RECORD thư viện
  đều đã nằm trong khóa R54 — khóa R55 không thêm file mã thư viện mới.
- **Giới hạn import lười:** khóa không khởi tạo predictor/`setup_model`; module Ultralytics nạp lười trong
  `setup_model`/`stream_inference` chỉ có trong khóa nhờ kế thừa khóa R54 (rà tĩnh + smoke R54B). Đường thực thi kiểm
  module nạp **trước** công việc (ngoài khóa/lệch hash → exit 2, không suy luận) và **sau từng đơn vị** trước marker
  (ngoài khóa → không marker, exit 4). Chi phí kiểm sau mỗi đơn vị chưa đo (đề xuất đo Pb73a427f bị từ chối).
- Sửa bất kỳ file nào trong khóa sau thời điểm này → phải tạo phiên khóa mới (giữ phiên cũ); sửa mã/cấu hình R55A →
  quay lại R55A, bằng chứng fixture mới.

## 5. Điều kiện cho lượt thực thi (chưa có, cần plan + duyệt riêng)

1. **Plan thực thi test40 do người dùng duyệt**, cấp mã `--duyet` cho `--buoc thuc-thi`; R55 không cấp quyền này.
2. **Phép đo thời gian PF500/UKF trên đoạn dày — CHƯA LÀM, bắt buộc trong plan thực thi sau R55:** chọn **trước** đoạn
   dày cố định (video/frame) với căn cứ mật độ **không dựa trên metric**; dùng cache **cùng khóa** đã đủ 40 video và
   đã xác nhận (manifest công bố) cho detector tương ứng; chạy PF500/UKF foreground; ghi thời gian từng run,
   detector/cấu hình/seed, phạm vi tính thời gian (gồm/không gồm nạp thư viện, đọc cache, ghi MOT) và giả định ngoại
   suy; nội dung phải được duyệt trước khi chạy; không xếp hạng, chọn tham số hay đổi giao thức từ phép đo.
3. **Dự toán tạm thời** (hợp đồng `du_toan_tam_thoi`, smoke R54B lần 2 × 220,08): suy luận 3 detector ~3.519 s,
   tracking 36 run ~3.247 s, chấm ≥ 8 s, tổng ~6.774 s; đĩa ~3,13 GB. Chỉ là kịch bản theo giả định (128 frame đầu
   đại diện mật độ — chưa kiểm); **không thay** phép đo đoạn dày ở mục 2.
4. Lượt dài phải tuân giới hạn tác vụ foreground/ngân sách thời gian mà plan thực thi đặt; bộ điều phối tiếp tục theo
   đơn vị (video/run), nên có thể chia nhiều lần gọi.

## 6. Nguyên tắc cache và tính tái lập (ghi để plan thực thi kế thừa)

- **Cache bất biến duy nhất:** mỗi detector có đúng một manifest cache được công bố (`HOAN_TAT.json` ghi sau
  `manifest_cache.json` + `SHA256SUMS`), chỉ khi đủ 40 video đạt kiểm schema/ánh xạ/timeline/unreadable/truncated/hash.
  Manifest đó là đầu vào chuẩn duy nhất cho c25/c10 và mọi tracker (CV/EKF/UKF/PF500 s0/1/2), kể cả khi tiếp tục.
  Không dùng cache legacy, NPZ detector R53 hay cache smoke R54.
- **Sai hash → chặn**, không suy luận lại để thay vào run cũ; lần xác nhận "có thể đã công bố" (manifest có mặt/marker
  hỏng) đóng băng `cache` của detector; tình huống này cần người dùng quyết định.
- **Tái lập tracker / chấm lại ≠ suy luận detector:** `lap-tracker` (cùng cache/seed → so byte MOT) và `cham-lai` (đọc
  đúng bản sao MOT đã băm → so byte metric) không gọi detector. **Không** đặt tiêu chí suy luận detector lại trùng byte.
- **Giới hạn hai phiên smoke R54B (chỉ quan sát):** `yolov8n_*` khác giá trị cache dù cùng số box (ví dụ
  `yolov8n_coco-c25-cv` MOTP lệch khoảng 3e-8); `yolo11n_mixed5` trùng byte trong hai phiên đã kiểm. Chưa biết nguyên
  nhân; **không** suy diễn nguyên nhân, chất lượng hay tính tất định của test40 từ quan sát này.
- **Tùy chọn kiểm biên độ lệch box/conf** giữa hai cache smoke bằng đọc NPZ: ngoài R55, chỉ làm khi người dùng (hoặc
  người được ủy quyền) duyệt cụ thể.
- **173 ảnh D4 vẫn mở:** giữ trong timeline 56.340 ảnh, chấm như frame không GT bằng evaluator hiện hành; không loại ảnh,
  không áp R0/R1, không diễn giải thiếu GT là âm tính đã xác nhận.

## 7. Đã làm / chưa làm / giới hạn

- Đã làm: 3 lệnh chuẩn bị trên đầu vào thật (exit 0); cập nhật mục 0; file này.
- Không làm: cache test40, suy luận, tracking, chấm, `setup_model`/warmup/forward, GPU, đọc NPZ, sửa mã/cấu hình/thiết
  kế R55A, sửa khóa R54, Git ghi. Không tạo `results/phase2/r55_tracking/thuc_thi/`.
- Kiểm `git status`: thay đổi mới của R55B chỉ nằm trong `results/phase2/r55_tracking/chuan_bi/`, mục 0 PLAN và file này.
- Giới hạn: tiền kiểm chỉ băm 256 ảnh mẫu (phần còn lại stat/size theo hash R47 kế thừa); GT không đối chiếu tọa độ box
  với XML (chỉ số đếm qua fingerprint R47); không kiểm pixel tô xám trên ảnh thật ở bước chuẩn bị; dự toán đĩa/thời gian
  là kịch bản; bao phủ import lười dựa vào khóa R54 + kiểm trước/sau đơn vị, chưa thử trên đường thật.
