# R56A — Bổ sung chọn một video cho tracking + khóa mới liên kết khóa R55 cha

Plan R56 (Codex đề xuất ở lời gọi `0017_planner` 21/10 15:12 sau lượt `bytetrack_r55_prepare_20261021_051232`, người dùng
duyệt phương án A). Tài liệu này **bổ sung** thiết kế R55 (`handoff/logs/review55/01_thiet_ke_dieu_phoi.md`, giữ nguyên
byte); không thay thuật toán tracking, metric, adapter, YAML, checkpoint hay giao thức R54. Trạng thái chuẩn: mục 0
`handoff/PLAN_DA_CHOT.md`.

## 1. Phạm vi

- Chỉ sửa 3 file: `src/r55_tracking/dieu_phoi_test40.py`, `src/r55_tracking/test_dieu_phoi_test40.py`,
  `configs/phase2/r55_tracking/hop_dong_dieu_phoi.json`. Helper R54 (`src/r54_tracking/chuan_bi_test40.py`) không đổi.
- Không chạy dữ liệu thật: không cache, tracking, evaluator, đo PF500/UKF; không `setup_model`/forward checkpoint thật,
  không GPU; không tạo `results/phase2/r55_tracking/thuc_thi/`.
- Kiểm khóa cha trước khi sửa: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r55_tracking/dieu_phoi_test40.py
  --buoc kiem-khoa --khoa khoa_20261021_121500_644100_pid22868` (không `--luu`) → exit 0, 376 file, khóa R54 368 file, 0 lệch.

## 2. Chọn video cho tracking

`--buoc thuc-thi --cong-viec tracking --phien <p> --run <r> [<r> ...] [--video <v>]`

- `--video <v>`: `v` phải thuộc manifest (`ctx.videos`); chỉ xử lý **toàn bộ** timeline video đó (frame 1..seqLength, kiểm
  qua `kiem_info`: `n_frames` = `n_buoc_timing` = số frame trong manifest cache, `n_steps` = [1]). Không cắt frame, không
  đổi fps/`n_steps`/seed/cấu hình.
- Bỏ `--video`: mọi video manifest như R55.
- Chỉ **lọc danh sách đơn vị tracking** (`lay_video`). Không đổi `ctx.videos`, manifest hay tập đánh giá.
- Video lạ → `LoiDauVao` exit 2 ở đầu công việc, trước kiểm cache và trước mọi đầu ra tracking (chỉ có nhật ký công việc).
- `--video` chỉ hợp lệ cho `tracking` (tùy chọn) và `lap-tracker` (bắt buộc như R55). Công việc khác (`mo-phien`, `cache`,
  `xac-nhan-cache`, `cham`, `cham-lai`, `hau-kiem`) → exit 2 trước khi mở phiên/nhật ký; không bỏ qua âm thầm.
- Giữ nguyên: kiểm manifest cache đủ **mọi** video và băm lại npz/json/meta **mọi** video của mỗi detector cần dùng trước
  tracking (kể cả video không được chọn) và sau tracking; kiểm môi trường/module trước, kiểm module sau từng đơn vị; ràng
  buộc marker, seed, cách tiếp tục, ngắt; điều kiện đủ video để chấm/chấm lại/hậu kiểm.
- Ràng buộc tracking (`rang_buoc_tracking`) **không** chứa phạm vi lọc → lần tracking hoàn tất từ `--video` được đường đầy
  đủ tái sử dụng khi cùng ràng buộc; chạy tiếp không `--video` chỉ làm phần thiếu.
- Nhật ký `nhat_ky/tracking/lan_*/ket_qua.json`: `pham_vi` {`loc_video`, `video`, `so_video_manifest`} và `don_vi` (mỗi
  run × video: `frame` [1, n] thực xử lý/đã xử lý, `trang_thai` `moi`|`tai_su_dung`, `lan`); stdout in một dòng phạm vi.
  `lap-tracker` ghi thêm `frame` vào `ket_qua_lap.json`.

## 3. Đoạn đo cố định (chưa đo)

- **MVI_40864 frame 1–1515 = toàn video.** Căn cứ: bảng tiền kiểm R55
  `results/phase2/r55_tracking/chuan_bi/tien_kiem_20261021_121436_943200_pid22720/bang_tien_kiem.csv` dòng MVI_40864:
  `so_anh` = `seq_length` = 1515, frame 1–1515, 0 ảnh D4, `so_dong_gt` 44689 → 44.689/1.515 ≈ 29,50 dòng GT/ảnh (mật độ
  GT, không dựa trên metric tracking).
- Ghi trong hợp đồng `doan_day.doan_co_dinh`. Ở đường thật, `tien-kiem` đối chiếu đoạn này với bảng vừa lập (toàn video,
  số dòng GT căn cứ); fixture bỏ qua (video không có trong dữ liệu tổng hợp) và kiểm hàm trên CSV R55 thật (chỉ đọc).
- **Chưa đo.** Phép đo PF500/UKF đoạn dày (P1980cb21) thuộc plan thực thi riêng: cần cache đủ 40 video đã xác nhận của
  detector tương ứng, chạy foreground, ghi thời gian từng run, phạm vi tính thời gian và giả định ngoại suy; không xếp
  hạng/chọn tham số/đổi giao thức từ phép đo.

## 4. Khóa mới liên kết khóa R55 cha

Khóa cha: `results/phase2/r55_tracking/chuan_bi/khoa_20261021_121500_644100_pid22868/`, `khoa.json`
`5a07efb8485466ef71366b94c6ae10051df22dacf606e8b6558eef204cf6e42b`, 376 file (ghi trong hợp đồng `khoa_cha_r55`, cùng
danh sách `duoc_thay` = 3 file trên; mã kiểm danh sách này bằng hằng `DUOC_THAY_KHOA_CHA`).

- `--buoc khoa --tien-kiem <phiên> --tests <phiên>` (đường thật): bắt buộc khóa cha. Kiểm nguyên vẹn khóa cha (chỉ đọc):
  SHA256 `khoa.json` = hợp đồng, `SHA256SUMS` khớp, `buoc` = `R55_LOCK`, cùng chế độ, nối tới cùng khóa R54. Băm lại
  **mọi** file khóa cha: khác hash cha chỉ được ở 3 file `duoc_thay` (`thay`), còn lại phải trùng (`giu`); lệch → từ chối.
  Khóa R54 vẫn kiểm nghiêm ngặt (`kiem_khoa_r54`).
- Khóa mới `buoc` = `R56_LOCK`, gồm: mọi file khóa cha (hash hiện tại), `khoa.json`/`SHA256SUMS`/`thoi_diem_khoa.json`
  của khóa cha, mã/test/hợp đồng hiện hành, thiết kế R55, tài liệu bổ sung này (`bo_sung_r56`), kết quả fixture và tiền
  kiểm mới. Ghi `khoa_cha_r55` (đường dẫn + SHA256 + số file + `duoc_thay`) và `khac_biet_khoa_cha` {`so_giu`, `thay`
  (SHA256 cũ/mới), `moi`}.
- Kiểm khóa thực thi (`kiem-khoa`, `mo-phien`, đầu mỗi công việc): băm **mọi** file theo hash **khóa mới**, không ngoại
  lệ; thêm kiểm liên kết: khóa cha ghi trong khóa = hợp đồng, `khoa.json` cha nguyên vẹn, bảng khác biệt tính lại trùng
  bản ghi. Ngoại lệ 3 file chỉ dùng lúc tạo khóa mới để mô tả chuyển tiếp.
- Đường thật chỉ nhận `R56_LOCK` cho `kiem-khoa`/thực thi; khóa `R55_LOCK` không cha chỉ còn hợp lệ trong fixture (mô
  phỏng khóa cha). Vì vậy `kiem-khoa --khoa khoa_20261021_121500_644100_pid22868` với mã mới sẽ báo lệch — đúng thiết kế,
  khóa cha không bị sửa. Fixture lấy khóa cha từ `<thư mục fixture>/khoa_cha.json` nếu có.

## 5. Bằng chứng fixture

Lệnh foreground: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r55_tracking/test_dieu_phoi_test40.py` →
**exit 0, 28/28 đạt**, 397,6 s, 59 lệnh CLI tiến trình con thật. Phiên:
`results/phase2/r55_tracking/tests/lan_20261021_152952_196200_pid9756/` (`ket_qua_test.json`, `cli_log.json`; dòng in
cuối vẫn ghi nhãn `[fixture R55A]` — chỉ là chuỗi in, `ket_qua_test.json` ghi `buoc` R56A).

SHA256 ghi trong kết quả (bản gửi review, không sửa sau lần chạy):

| File | SHA256 |
|---|---|
| `src/r55_tracking/dieu_phoi_test40.py` | `257f4f028e5a39aec4ea61a8387efbab518631d05021942ce8a3d7d4a973961e` |
| `src/r55_tracking/test_dieu_phoi_test40.py` | `87689fbcce7f4274d8dc152ac88ed1fbbc65fffebc9046d8d8f8ba31ef4e57a6` |
| `configs/phase2/r55_tracking/hop_dong_dieu_phoi.json` | `488238d4fc669ffac3ce98800b7031b08e8233504a84751afe3c29e4239cf95d` |
| `handoff/logs/review55/01_thiet_ke_dieu_phoi.md` (R55, nguyên byte) | `9687aae739813f10ca4be5e010f4abe57b319c4047ead1898b3c65432419a598` |

Tài liệu này được hoàn thiện **sau** lần chạy (SHA lúc chạy `40559c65…18fda` ghi trong kết quả chỉ để truy vết); khóa
R56B sẽ băm bản hiện hành.

- 24 ca R55 (A1–A2, B1, C1–C5, D1, E1–E4, F1–F2, G1–G4, H1–H2, I1–I2, Z1) giữ nguyên và đạt; B1 chỉ thêm lưu tên phiên
  tiền kiểm. Z1 snapshot mở rộng: `handoff/logs/review56`, `results/phase2/r55_tracking/{chuan_bi,tests}` và **đệ quy**
  thư mục khóa R55 cha thật → 0 khác ngoài phiên test.
- **J1 khóa R56 / khóa cha** (khóa cha fixture = bản sao khóa B1, 3 file được thay mang hash R55 thật): khóa cha sai SHA256
  / `SHA256SUMS` cha sai / file cha lệch ngoài 3 file → `--buoc khoa` exit 2, không tạo phiên. Khóa cha hợp lệ → `R56_LOCK`
  `khoa_20261021_153453_305600_pid7220` (fixture), 397 file, `so_giu` 390, `thay` đúng 3 file (SHA cũ = R55 thật, mới = hiện
  tại), `moi` gồm tài liệu bổ sung + `khoa.json`/`SHA256SUMS`/`thoi_diem_khoa.json` khóa cha. `kiem-khoa` khóa mới exit 0;
  khóa cha đổi byte → exit 2; đặc tả khóa cha trỏ khóa khác → exit 2; bản sao khóa mới sửa `khac_biet_khoa_cha` (SUMS
  khớp) → exit 2. Kiểm liên kết khóa R55 cha **thật** (chỉ đọc, `kiem_khoa_cha`): `5a07efb8…6cf42b`, 376 file, 373 giữ,
  thay đúng 3 file được phép, 0 lệch.
- **J2 `--video` qua CLI** (phiên `p2` mở bằng khóa R56 fixture, detector `yolov8n_coco`, run c25 CV/UKF/PF500 s0/s1/s2):
  `--video` với `cache`/`xac-nhan-cache`/`cham`/`cham-lai`/`hau-kiem`/`mo-phien` → exit 2, phiên không đổi byte; cache
  chưa đủ (V1 xong, V2 chưa) → tracking `--video V1` exit 2; video lạ → exit 2; NPZ của video **không chọn** (V2) sai
  hash → tracking `--video V1` exit 2 "khong khop manifest" — cả ba không tạo `tracking/`. Ngắt trước marker (UKF/V1) →
  exit 9; chạy `--video V1` → "moi 5, tai su dung 0", lần dở giữ nguyên byte và không marker, `pham_vi` = {loc_video,
  [V1], 2}, mỗi `don_vi` frame [1, 20]; chỉ có thư mục `tracking/<5 run>/V1`. Gọi lại → "moi 0, tai su dung 5". `cham` →
  exit 2 "chua du 2 video"; `hau-kiem` exit 3 liệt kê 5 run "thieu tracking 1 video". Chạy tiếp không `--video` → "moi 5,
  tai su dung 5" (V2 mới, V1 tái sử dụng). MOT `data/<v>.txt` của 5 run × 2 video **trùng byte** với lần tính độc lập
  `run_tracking.run_video_to_dir` trên cùng NPZ đã xác nhận, cùng YAML/`track_buffer`/seed (gồm UKF và PF500 seed 0/1/2;
  không so timing). `lap-tracker` PF500 s2/V1 trùng byte, ghi `frame` [1, 20].
- **J3 đoạn cố định:** `kiem_doan_co_dinh` trên CSV tiền kiểm R55 thật (chỉ đọc, 40 dòng) → 0 lỗi; 44.689/1.515 =
  29,4977 ≈ 29,50; sửa `seq_length`/`so_dong_gt`/`frame_cuoi` hoặc bỏ dòng → bắt; `tien_kiem.json` fixture ghi `bo_qua`.
- **Z2:** không có `results/phase2/r55_tracking/thuc_thi/`; mọi lệnh CLI đường thật trong fixture đều exit ≠ 0.

## 6. Giới hạn, việc chưa làm

- Fixture 2 video × 20 ảnh tổng hợp, detector giả lập (`MoHinhGia` R54), không YOLO/`setup_model`/GPU; đường thật 40
  video chưa chạy. Nhánh `tien-kiem` đường thật đối chiếu đoạn cố định chưa chạy (chỉ kiểm hàm); sẽ chạy ở R56B.
- Fixture tách khóa cha qua `khoa_cha.json` (không thuộc `ma_r55_sha`); đường thật lấy khóa cha từ hợp đồng — nhánh
  này chỉ được kiểm ở mức hàm trên khóa cha thật (chỉ đọc), `--buoc khoa` đường thật chạy ở R56B.
- Snapshot Z1 chỉ phủ các thư mục liệt kê, không phải toàn đĩa. Lần độc lập J2 gọi cùng `track_video` mà
  `chay_tracker` R54 gọi (đường toàn video hiện có), không phải một cài đặt tracker khác.
- QJTF (đo chi phí kiểm module mỗi tác vụ) chỉ ghi nhận cho plan thực thi; R56 không thêm instrumentation/benchmark.
- Không tiêm module ngoài khóa (P6fb1cad7), không marker nguyên tử (P9857b26c).
- Không kỳ vọng suy luận detector lại trùng byte; cache đã băm và xác nhận đủ 40 video là đầu vào chuẩn duy nhất.
- R56B (tiền kiểm, khóa mới, kiểm khóa trên đầu vào thật) chỉ chạy sau khi Codex approve R56A.
