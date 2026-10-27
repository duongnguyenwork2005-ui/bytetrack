# R59A — Bằng chứng cổng A (đối chứng SORT–ByteTrack test40, chỉ c25)

Ngày: 2026-10-23. Người làm: Claude (`claude-opus-5-5`, effort high). Trạng thái: **gói A bản sửa xong, chờ Codex
review A lại** (review A vòng 1 trả revise, xem §2b). Chưa chạy full, chưa mở phiên B. Quyết định Q-A..Q-F và giao
thức: `01_quyet_dinh_va_giao_thuc.md` (không đổi). Mọi đường dẫn `results/...` dưới đây là tương đối từ gốc repo; thư
mục R59: `results/phase2/r59_sort_test40/`.

**Gói và khóa cần review:** gói `chuan_bi/goi_a_20261023_135911_306800_pid9372` + khóa
`chuan_bi/khoa_20261023_135704_309300_pid25512`. Gói/khóa trước (`goi_a_20261023_130840_575200_pid5080`,
`khoa_20261023_130642_719700_pid18116`) giữ nguyên làm lịch sử, không dùng cho B.

## 1. Lần chạy đầu bị chặn: nguyên nhân môi trường (giữ nguyên bằng chứng)

- Lần đầu (23/10 sáng), fixture chạy 2 lần, cả hai **exit 1** với `OSError: [WinError 4551]` khi nạp
  `torch\lib\shm.dll` lúc import ultralytics, trước mọi test. Nguyên nhân là **Smart App Control của Windows**: nó chặn
  DLL của torch (quy tắc Code Integrity của hệ điều hành), không phải lỗi mã R59 hay helper. Hai lần đó chỉ để lại thư
  mục `tests/lan_*` chứa `_cau_hinh_thu_vien` rỗng, không có `ket_qua_test.json` (theo ghi chép mục 0 lúc đó). Không
  xóa, không sửa các thư mục đó (review A vòng 1 xác nhận chúng vẫn tồn tại).
- Người dùng tắt Smart App Control (~12:00). Phiên điều phối kiểm lại: torch 2.6.0+cu124 import được, CUDA True,
  ultralytics 8.4.141. Từ đó R59A được **làm lại đầy đủ** từ bước (4), mỗi bước vào một phiên mới.

## 2. Các lần sửa (đều trước `chuan-bi` của lần chạy tương ứng; khóa chứa bản sau sửa)

### 2a. Lần làm lại thứ nhất (trước review A vòng 1)

| Lần fixture | Kết quả | Nguyên nhân | Sửa |
|---|---|---|---|
| `tests/lan_20261023_125710_525500_pid17944` | exit 1, 7/13 | (a) `buoc_smoke` import `run_tracking` trước `W.dat_moi_truong` (hàm này đưa `src/` vào `sys.path`) → `ModuleNotFoundError`; 4 test B hỏng dây chuyền. (b) Kỳ vọng K4 trong fixture sai | (a) dời import sau `dat_moi_truong` ở `smoke`, `full`, `lap-tracker`. (b) xem dưới |
| `tests/lan_20261023_125842_798900_pid15304` | exit 1, 8/13 | `cham-nguon`/`dong-goi-a`/`mo-phien` in ID sau tiền tố trên cùng dòng; quy ước bàn giao ID (dòng `KEY=...`) không đọc được | in `CHAM_NGUON_ID=`, `GOI_A_ID=`, `PHIEN_ID=` trên dòng riêng |
| `tests/lan_20261023_125956_551000_pid25124` | exit 0, 13/13 | — | — (bản này đi vào khóa cũ `khoa_20261023_130642_719700_pid18116`) |

K4 (b): fixture cũ kỳ vọng frame 1–3 rỗng, có detection ở frame 4–8 thì xuất `[0,0,0,0,0,1,1,1]`. Kết quả thực tế là
`[0,0,0,0,0,0,1,1]`, và đây là hành vi đúng. Trong `src/sort_tracker.py` (giống `KalmanBoxTracker` gốc), `hit_streak`
bằng 0 khi sinh track và chỉ tăng khi có update ghép. Track sinh ở frame 4 nên đạt `hit_streak = 3 = min_hits` ở frame 7.
Ngoại lệ đầu video (`frame_id <= min_hits`) không áp dụng vì frame 4 > 3. Đã sửa kỳ vọng trong fixture (có chú thích);
**không** sửa mã tracker.

### 2b. Sửa theo review A vòng 1 (revise, 3 finding P2)

Review: `D:/LABOTARY/automation/runs_local/bytetrack_r59_sort_test40_20261023_1053/reports/review_002.json`.

| Finding | Sửa (chỉ trong `src/r59_sort_test40/`, không sửa helper) | Fixture chứng minh |
|---|---|---|
| P2-1: fixture gọi `W.tao_phien(TESTS)` trước khi kiểm junction; helper R54 resolve cả đích lẫn vùng nên không tự chặn | `test_dieu_phoi_sort_test40.py`: `tao_lan_tests(goc)` gọi `R55.kiem_doc_lap(goc, VUNG)` (mọi thành phần đã tồn tại phải `resolve == abspath`, nằm lexical trong vùng R59) **trước** `W.tao_phien`, rồi kiểm lại `<lan>`; `main()` dùng hàm này trước lần ghi đầu tiên | Test "Chan junction truoc ghi": tạo junction thật `tests/<lan>/jx/cong → jx/dich` (`_winapi.CreateJunction`); `tao_lan_tests(cong)` và `tao_lan_tests(cong/tests)` đều `LoiDauVao`; wrapper ở tiến trình con với `R59_FIXTURE=cong/fixture.json` exit 2 ("di qua junction/symlink"); `dich` không có file mới; gỡ điểm nối bằng `os.rmdir` (không đụng `dich`) |
| P2-2: `tim_goi_a` / `Phien.xac_minh` chỉ kiểm marker ngoài gói A (chỉ chứa `goi_a.json`, `kiem_khoa_cuoi.json`); khóa không chứa smoke ⇒ đổi/mất file smoke hoặc chấm nguồn vẫn mở/tiếp tục được B | `dieu_phoi_sort_test40.py`: hàm mới `kiem_goi_a` — marker ngoài hợp lệ, `goi_a.json` gắn đúng `khoa_sha256`, `dat` và không `lech`, `so_file` = số file liệt kê, và **mỗi** file trong `goi_a.json#file` tồn tại, không qua junction (`R55.kiem_doc_lap`), khớp SHA256. Dùng ở `tim_goi_a` (mo-phien) và `Phien.xac_minh` (full, tiếp tục full, chấm, lap-tracker, cham-lai, hậu kiểm, tổng hợp) | Trên dữ liệu tổng hợp: lệch hash `ket_qua_smoke.json`, thiếu `ket_qua_cham_nguon.json`, marker gói A bị cắt nửa → `mo-phien` exit 2, không tạo `thuc_thi/`; sau `mo-phien`: lệch hash smoke / thiếu file chấm nguồn → `full` exit 2, không tạo `tracking/`. Mỗi ca khôi phục nguyên byte và kiểm lại |
| P2-3: K3 tự tính mask `conf > det_thresh` thay vì gọi `SORT.update`; báo cáo nói fixture chứng minh phép cắt `[:300]` nhưng fixture không có ca > 300 detection | K3: thêm phần gọi `SORT.update` thật trên instance cả `sort-cv30` và `sort-ekf30` | Ở frame 1 (`frame_id <= min_hits` nên mọi track vừa sinh được xuất), với 4 box (== 0,25; ngay trên; ngay dưới; 0,2): đúng 1 track, cột chỉ số đầu vào = box ngay trên, điểm xuất == float32 gốc; sau runner cũng 1 track. Test mới "K3 cat [:300]": 480 box (lớp 0 xen kẽ, 1/7 box conf == 0,25) → `det_cache.filter_dets` trả đúng 300 hàng float32 = tham chiếu độc lập `[mask][:300]`, giữ thứ tự; `SORT.update` cả 2 bộ lọc xuất đủ 300 track (chỉ số 0..299) |

Không sửa helper, YAML, evaluator, backend hay nguồn R54–R57. Hai lần fixture lỗi và lần 13/13 cũ giữ nguyên trong `tests/`.
Lần fixture đầu sau sửa `tests/lan_20261023_134453_988900_pid18020` (exit 1, 14/15) giữ nguyên: test cắt `[:300]` chỉ có
219 box hợp lệ (< 300) vì lỗi dựng dữ liệu của chính test; đã tăng 320 → 480 box, không đổi mã wrapper.

## 3. Lệnh và kết quả (Python `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe`, foreground, tuần tự)

Lần chạy hiện hành (sau sửa §2b):

| # | Lệnh (`src/r59_sort_test40/...`) | Exit | Đầu ra |
|---|---|---|---|
| 1 | `test_dieu_phoi_sort_test40.py` | 1 | 14/15, `lan_20261023_134453_988900_pid18020` (xem §2b, giữ nguyên) |
| 2 | `test_dieu_phoi_sort_test40.py` | 0 | **15/15**, `TESTS_ID=lan_20261023_135030_327000_pid26008` |
| 3 | `dieu_phoi_sort_test40.py --buoc chuan-bi` | 0 | `CHUAN_BI_ID=chuan_bi_20261023_135554_118700_pid21600`, 0 lệch, 6 đối chứng đạt, K2 120/120 |
| 4 | `--buoc khoa --chuan-bi chuan_bi_20261023_135554_118700_pid21600 --tests lan_20261023_135030_327000_pid26008` | 0 | `KHOA_ID=khoa_20261023_135704_309300_pid25512`, 1.557 file |
| 5 | `--buoc kiem-khoa --khoa khoa_20261023_135704_309300_pid25512` | 0 | DAT |
| 6 | `--buoc smoke --khoa khoa_20261023_135704_309300_pid25512` | 0 | `SMOKE_ID=smoke_20261023_135748_242700_pid12692` |
| 7 | `--buoc cham-nguon --khoa khoa_20261023_135704_309300_pid25512 --run r54-yolov8n_coco-c25-cv` | 0 | `CHAM_NGUON_ID=cham_nguon_20261023_135811_750100_pid4508`, 3 file metric trùng byte |
| 8 | `--buoc dong-goi-a --khoa khoa_20261023_135704_309300_pid25512 --smoke smoke_20261023_135748_242700_pid12692 --cham-nguon cham_nguon_20261023_135811_750100_pid4508` | 0 | `GOI_A_ID=goi_a_20261023_135911_306800_pid9372`, 463 file, 4.3 + K1–K7 đạt, kiểm khóa cuối DAT |

Lần trước (lịch sử, bị revise): fixture 13/13 `lan_20261023_125956_551000_pid25124` → `chuan_bi_20261023_130527_496800_pid26216`
→ `khoa_20261023_130642_719700_pid18116` → smoke `smoke_20261023_130721_179800_pid3660` → `cham_nguon_20261023_130742_612500_pid11204`
→ `goi_a_20261023_130840_575200_pid5080` (462 file), đều exit 0.

Mỗi thư mục bước có `cli_log.json` (argv, stdout, stderr, exit) và `HOAN_TAT.json` (ràng buộc + SHA256 mọi file).
Mỗi ID ở bước sau được truyền tường minh từ đầu ra của bước trước. Fixture ghi stdout/stderr/exit từng tiến trình con
vào `ket_qua_test.json#tien_trinh_con`.

## 4. Định danh khóa

- Khóa R59: `results/phase2/r59_sort_test40/chuan_bi/khoa_20261023_135704_309300_pid25512/khoa.json`, SHA256
  **`b18fc37632fe94e0b1f5310df1c685f374ab786531800b69f6dced30c3d4057d`**. Khóa nối tới khóa R56
  `ce85628e…35247b`. Nội dung khóa gồm: giao thức, mã/fixture, module nạp thực tế (src + ultralytics + trackeval),
  2 YAML SORT + 2 YAML ByteTrack, cấu hình hiệu lực, nguồn/manifest đã định danh, `chuan_bi.json`, `k2_parity.csv`,
  `ket_qua_test.json`. **Không** gồm mục 0 hay báo cáo tiến độ.
- Giao thức `configs/phase2/r59_sort_test40/giao_thuc.json` SHA256 `06e998d473c2e88afa13d8cff24b683fc9cdc615e3d05c36bae291d8ba1e33c4` (không đổi).
- Mã `dieu_phoi_sort_test40.py` SHA256 `ff805a96072b8c4213b96f673783b9d0dafb175cf6849755652c3eafc93e9d49`; fixture
  `test_dieu_phoi_sort_test40.py` SHA256 `990fa02f6923180fad3823674dc1386cae7daaf6b1bb224a7c9de5a51177b173` (ghi trong
  `ket_qua_test.json#ma_sha256` của lần 15/15). Đây đúng là bản đã chạy fixture 15/15 và `chuan-bi`; bước `khoa` từ chối
  nếu khác.
- Gói A: `results/phase2/r59_sort_test40/chuan_bi/goi_a_20261023_135911_306800_pid9372/goi_a.json` (SHA256
  `d0ddcde25126e316883c93dc206f0113a48d9ab8f011ab6b35b58e331935b740`; bản đồ 4.3/K1–K7, danh sách 463 file + SHA256) và
  `kiem_khoa_cuoi.json` (SHA256 `68b7b3a3a8b16bad3444171a876e4f215baa5b31fedc1f6b2d85c3afb7a50221`). Từ bản này, mở/tiếp
  tục B kiểm lại toàn bộ 463 file đó (§2b, P2-2).

## 5. Bản đồ mục 4.3 và K1–K7 (theo `goi_a.json#ban_do_4_3_K`; tất cả `dat: true`)

| Mục | Bằng chứng chính | Kết quả |
|---|---|---|
| 4.3-1 phiên/khóa R56 nguyên vẹn; 6 lần chấm theo `nguon.json` | `chuan_bi.json#nguon` | `W.kiem_khoa` khóa R56 + `R55.kiem_khoa_r55` (khóa cha) đạt; mỗi đối chứng: summary SHA = `nguon.json`, lần chấm nằm ở `cham/<run>/` của phiên R56, marker hợp lệ, 3 file metric = marker, `combined_metrics.csv` == `combined` marker, 40 MOT = MOT lần tracking (ràng buộc run/detector/c25/0,25/npz/manifest), timeline tracking = seqLength, `mot_sha256_ds` khớp. **6/6 đạt** |
| 4.3-2 cache R56 theo manifest đã công bố | `chuan_bi.json#nguon.cache`, `cache_truoc/sau` | 3 manifest (`78bb4eeb…`, `79197369…`, `ee391365…`), mỗi detector đúng 1 manifest ràng buộc khóa R56; 40 video × NPZ/JSON/meta/marker khớp manifest; `R55.kiem_npz` + kiểm nhẹ đạt; băm cache trước = sau ở chuan-bi và smoke |
| 4.3-3 cùng mã phụ thuộc + backend như khóa R56 | `chuan_bi.json#module_nap/backend/thu_vien/cau_hinh_bytetrack_r56` | Mọi module nạp ở trạng thái `khop_khoa_r56` hoặc `ma_r59`; backend, phiên bản thư viện và cấu hình ByteTrack hiệu lực = khóa R56 |
| 4.3-4 chấm lại ByteTrack nguồn | `kiem_duong_cham/cham_nguon_20261023_135811_750100_pid4508/ket_qua_cham_nguon.json` | Sao nguyên byte 40 MOT `r54-yolov8n_coco-c25-cv` (đã kiểm SHA), chấm lại giữ định danh run: `per_video_metrics.csv`, `combined_metrics.csv`, `raw_fields.json` **trùng byte** nguồn. Không chạy tracker ByteTrack |
| K1 cấu hình + bộ lọc từ instance, `config.py` | `chuan_bi.json#cau_hinh_sort/bo_loc_instance/dem_buoc_ctrv/config_py` | Xem §6 |
| K2 parity đầu vào | `chuan_bi_20261023_135554_118700_pid21600/k2_parity.csv` (120 dòng: hash nội dung runner + tập SORT, lỗi đầu tiên) | **120/120**; xem §7 |
| K3 ngưỡng float32 + cắt | fixture `ket_qua_test.json` (2 test K3) | Conf == 0,25 bị loại (runner `>`; `SORT.update` thật cả 2 bộ lọc chỉ sinh track từ box ngay trên); ngay trên giữ, ngay dưới loại. Nếu không qua runner, ByteTrack đưa == 0,25 vào tầng cao (`>=`), còn SORT loại; ở c25 runner đã che khác biệt này. Cắt `[:300]` sau lọc conf/lớp đúng thứ tự (480 box tổng hợp), SORT nhận đủ 300 |
| K4 vòng đời | fixture | Frame rỗng ở đầu/giữa/cuối không lỗi; giữ track ở `time_since_update` 30, xóa ở lần hụt thứ 31 (cả 2 SORT); ID reset mỗi video qua `run_tracking.track_video`; D4 nằm trong timeline (173) |
| K5 cache bất biến | `cache_truoc == cache_sau` (chuan-bi, smoke) | Đạt |
| K6 khóa mới nối khóa R56 | `kiem_khoa_cuoi.json` | DAT, 1.557 file |
| K7 smoke | `smoke/smoke_20261023_135748_242700_pid12692/ket_qua_smoke.json` | Xem §8 |

## 6. K1 — cấu hình hiệu lực và bộ lọc từ instance

- Hai SORT: lớp `ElapsedSORT`, không có mixin `AssociationLogMixin` (`assoc_log=false`) và không có `kalman_filter`
  (`pf_diag` được truyền `true` như R56 nhưng **không có hiệu lực với SORT**; smoke không tạo `pf_diag/`, `assoc_log/`).
  Instance: `max_age 30`, `iou_threshold 0,3`, `min_hits 3`, `det_thresh 0,25`. Không có khóa `ctrv_`/`cv_` trong cfg.
  SHA256[:12] của YAML khớp R51.
- Bộ lọc: SORT (`tr._new_filter().kf`) và ByteTrack (`tr.kalman_filter`) cùng lớp, cùng toàn bộ tham số instance (Q/R/P0,
  tham số CTRV, `dt = 1,0` bước). Ở cùng trạng thái tham chiếu (initiate → predict → [try_initiate_from_motion] → update),
  hai bên **trùng bit**. Với CV, tham số instance cũng trùng `STrack.shared_kalman` (chỉ so tham số, không so trạng
  thái).
- Bộ đếm dưới `skip` (chuỗi có mặt ở frame 1, mất ở 2–3, có lại ở 4–8): `n_frames` của `try_initiate_from_motion` là
  `[3]` ở cả hai bên; số lần predict EKF là 7/7. ByteTrack đi qua `multi_predict` (7 hàng), SORT gọi predict từng track.
  Đây là khác biệt đường gọi (vòng đời dự kiến), không phải khác tham số.
- Khác vòng đời dự kiến, có test riêng: track chưa xác nhận bị ByteTrack xóa sau 1 frame hụt, SORT giữ (tới
  `max_age`); SORT chỉ xuất khi `hit_streak >= 3` (hoặc ở 3 frame đầu video).
- `config.py`: byte thô `37e4f56247b7be0896c29d74026b863435b2eec5f53c11fc1a8cd073da70d88b` (373 CRLF) **= khóa R56**
  (quy ước `W.sha256_file`, byte thô); chuẩn hóa LF `0e3f8b7584b8f90d5c121cf978d8fd74553195e13f6135b3d9b79362f6ff4ad0`
  **= R51** (`audit_protocol.py`, `bytes.replace(CRLF, LF)`). Hai chuỗi khác nhau ở hồ sơ R58 vì khác quy ước băm, không
  phải nội dung khác. Toàn vẹn được kiểm bằng byte thô; hash LF chỉ dùng để giải thích.

## 7. K2 — parity detection, timeline, D4

- Với từng tổ hợp (detector, video), tham chiếu độc lập được dựng từ NPZ thô: `conf > 0,25` (float64, ngặt) → lớp
  `[2,3,5,7]` → `[:300]`. So với `VideoDetections.frame` của runner: dtype float32, shape và **byte** trùng, thứ tự giữ
  nguyên. Sau `det_cache.to_boxes`, tập SORT nhận (`conf > det_thresh`) phải trùng chỉ số với tầng cao ByteTrack thật
  (`_split_detections`) và tầng thấp phải rỗng; tọa độ SORT phải bằng runner. Mỗi dòng CSV có `sha256_runner`,
  `sha256_sort` và `loi_dau`. Lưu ý: ở K2, "tập SORT nhận" được tính bằng cùng phép so `conf > det_thresh` (float64) như mã
  SORT, với `det_thresh` lấy từ giao thức (= giá trị instance đã kiểm ở K1); K2 không gọi `SORT.update` trên dữ liệu thật; hành vi `SORT.update` tại ngưỡng được kiểm ở fixture K3.
- Kết quả: **120/120** đạt. Mỗi detector có 56.340 ảnh (= tổng seqLength) và 173 D4 (= danh sách
  `detrac_test_loai_D4.csv`); fps 25; `skip`. Ngưỡng ByteTrack cv/ekf là (0,25; 0,1; 0,25).
- Giới hạn quan sát: trên dữ liệu thật **0** detection có conf đúng bằng 0,25 float32, **0** frame bị cắt `[:300]`,
  **0** box suy biến. Biên ngưỡng được kiểm bằng fixture K3 (runner + `SORT.update` thật + tầng ByteTrack); phép cắt
  `[:300]` được kiểm bằng test fixture riêng (480 box tổng hợp qua `det_cache.filter_dets` rồi `SORT.update`). Dữ liệu
  thật không có trường hợp nào chạm tới hai biên này.

## 8. K7 — smoke

`MVI_39031`, `MVI_39051`, frame 1–128, split `DETRAC-smoke128`, cache R56 `yolov8n_coco` (chỉ `W.mo_cache`, không mở
checkpoint, không predict). GT smoke tạo trong thư mục smoke bằng `W.tao_gt_smoke`. Mỗi video có 128 frame/128 bước,
`n_steps [1]`, lớp `ElapsedSORT`; module sau chạy khớp khóa; cache trước = sau.

| Run | HOTA (COMBINED, smoke) | IDSW | box |
|---|---|---|---|
| `r59-yolov8n_coco-c25-sort-cv30` | 0,6389 | 2 | 466 |
| `r59-yolov8n_coco-c25-sort-ekf30` | 0,6229 | 3 | 465 |

Số smoke trùng lần trước (`smoke_20261023_130721_179800_pid3660`) — sửa §2b không chạm đường tracking. Đây chỉ là số kiểm
đường ống trên 2 video × 128 frame, **không** phải kết quả thí nghiệm, không dùng để so sánh.

## 9. Giới hạn và chưa làm

- Chưa có review A cho gói mới. Chưa `mo-phien`, chưa full, chưa có số SORT test40.
- Fixture (không phải `chuan-bi`) kiểm luồng B (mở phiên/full/ngắt/tiếp tục/chấm/lặp/hậu kiểm/tổng hợp, và chặn khi gói
  A không nguyên vẹn) trên dữ liệu tổng hợp; chưa chạy B trên dữ liệu thật.
- Test junction dùng `_winapi.CreateJunction` (junction NTFS, không cần quyền admin); chưa thử symlink thật (cần quyền
  Developer Mode), nhưng `R55.kiem_doc_lap` so `resolve` với `abspath` nên cùng cơ chế chặn.
- Tôi tự kiểm (Tier 1) bằng các lệnh ở §3; không có kiểm độc lập ngoài các lệnh đó.
- Giới hạn diễn giải giữ như `01_quyet_dinh_va_giao_thuc.md` §5: chỉ c25 (không đo lợi ích detection điểm thấp),
  không CI, không suy luận tốc độ (SORT `assoc_log=false`, ByteTrack R56 `assoc_log=true`).
- Chưa commit (người dùng quyết định riêng).
