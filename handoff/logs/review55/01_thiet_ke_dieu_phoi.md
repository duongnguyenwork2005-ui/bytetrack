# R55A — Thiết kế bộ điều phối test40 (tiếp tục theo đơn vị công việc, cache bất biến)

Ngày 2026-10-21. Nhánh `phase2-mixed-traffic`, HEAD lúc lập plan `117d62da027e8ce3e4c9f455d576a924b4d82bbb`; mọi thay đổi
chưa commit được giữ nguyên. **R55 chỉ chuẩn bị**: không cache, suy luận, tracking hay chấm trên dữ liệu thật; không
`setup_model`/warmup/forward checkpoint thật. Mọi lần chạy thật sau R55 cần plan thực thi riêng do người dùng duyệt.

Sản phẩm R55A:

| File | Vai trò |
|---|---|
| `src/r55_tracking/dieu_phoi_test40.py` | bộ điều phối (đường chuẩn bị + đường thực thi) |
| `src/r55_tracking/test_dieu_phoi_test40.py` | fixture CPU, dữ liệu tổng hợp, detector giả lập |
| `configs/phase2/r55_tracking/hop_dong_dieu_phoi.json` | hợp đồng: hash giao thức/khóa R54, 36 run, thứ tự công việc, quy tắc tiếp tục |
| `results/phase2/r55_tracking/tests/lan_*/` | bằng chứng fixture (`ket_qua_test.json`, `cli_log.json`) |

## 1. Căn cứ và lịch sử quyết định

- Giao thức R54 `configs/phase2/r54_tracking/giao_thuc_test40.json` SHA256 `eb1f8f30…715431ad`, khóa R54
  `results/phase2/r54_tracking/chuan_bi/khoa_20261020_211635_887200_pid10816/khoa.json` SHA256 `e2d1b0d5…a418e16b4f`.
  Hợp đồng ghi cả hai hash; bộ điều phối từ chối chạy khi một trong hai lệch.
- Review thực tế lượt `bytetrack_r54_test40_prep_20261020_145514`: `review_003.json` approve R54A; `review_005.json`
  approve smoke foreground R54B; `review_006.json` approve tài liệu Paad1df4f. **Không review nào duyệt full.**
- P1980cb21 (21/10): hoãn phép đo thời gian PF500/UKF trên đoạn dày khỏi R54, yêu cầu đưa vào plan lượt dài.
  Pc5fc63da/P482c0392: từ chối thêm bước vào R54, **không hủy** nội dung chuyển tiếp đó. P7e7ee3c8: từ chối riêng kiểm
  `setup_model` trên CPU trước smoke — không được hiểu là cho phép chạy kiểm đó ở R55 (R55 không làm).
- Giao thức giữ nguyên: `yolov8n_coco`/`yolov8n_mixed5`/`yolo11n_mixed5` (best checkpoint R53), c25/c10, ByteTrack
  CV/EKF/UKF/PF500 seed 0/1/2 → 36 run × 40 video, 56.340 ảnh (gồm 173 ảnh D4 giữ trong timeline), vehicle gộp, tô xám
  `ignored_region`, evaluator/timeline hiện hành. Không SORT/expB/expC/PF100/PF300/K1/K2/ablation/bootstrap/BDD/train/Nano.

## 2. Nguyên tắc: chỉ điều phối, không tính lại

Mọi phép tính gọi hàm đã review của R54 (`src/r54_tracking/chuan_bi_test40.py`, không sửa) và qua đó hàm hiện có:

| Việc | Hàm |
|---|---|
| tô xám + suy luận + ánh xạ lớp + ghi cache | `W.tao_cache` → `det_cache.write_meta(create)`/`build_video_yolo`/`save_video`; `W.khung_mau` băm **từng ảnh** so bảng hash và (đường thật) so `DetracDataset.frames` |
| đọc cache theo điều kiện | `W.mo_cache` → `det_cache.check_compatible`/`sequence_spec_status`/`VideoDetections` (conf `>`, lớp [2,3,5,7], max_det 300, không `allow_truncated`) |
| tracker | `W.chay_tracker` → `run_tracking.track_video` (ghi như `run_video_to_dir`) |
| chấm | `W.cham` → `run_trackeval.build_configs` + `MotChallenge2DBox` + HOTA/CLEAR/Identity + `save_raw_fields` |
| tiền kiểm dữ liệu | `W.tien_kiem` (manifest, checkpoint/names CPU, bảng hash + 256 ảnh mẫu, GT/seqinfo/ignored, D4, cấu hình hiệu lực) |

Phần mới chỉ là: kiểm hợp đồng/khóa, chọn lần thử, marker, manifest cache, xác nhận cache, quản lý phiên, nhật ký.
Không monkeypatch thuật toán. Detector giả lập chỉ được chọn khi biến `R55_FIXTURE` trỏ vào `tests/`.

## 3. Hai đường tách riêng

**Chuẩn bị** (ghi `results/phase2/r55_tracking/chuan_bi/`; dùng ở R55B):
`--buoc tien-kiem` → `--buoc khoa --tien-kiem <phiên> --tests <phiên>` (tên **tường minh**, không tự chọn phiên mới
nhất) → `--buoc kiem-khoa --khoa <phiên> [--luu chuan_bi]` (mặc định chỉ đọc).

- Tiền kiểm R55 = kiểm lại khóa R54 (băm lại toàn bộ file, chỉ đọc) + `W.tien_kiem` + cấu hình hiệu lực/thư viện/
  backend so khóa R54 + vùng ghi + dung lượng đĩa (kích thước file smoke R54 × hệ số ảnh, chỉ `stat`, không đọc NPZ).
  **Ảnh: kế thừa kiểm stat/size theo bảng hash R47 và băm lại 256 ảnh mẫu cố định của R54; không băm lại toàn bộ
  56.340 ảnh ở tiền kiểm** (công việc cache băm từng ảnh ngay trước suy luận).
- Khóa R55 = mọi file của khóa R54 (băm lại, phải trùng — lệch thì không khóa) + mã/test/hợp đồng/thiết kế R55 + kết quả
  test chỉ định (phải `tat_ca_dat` và trùng SHA256 mã/test/hợp đồng hiện tại) + phiên tiền kiểm chỉ định (đầu vào phải
  trùng) + module src/Ultralytics/TrackEval nạp khi import mã thực thi + RECORD thư viện + YAML tracker; ghi
  `khoa_r54` (hash), backend, cấu hình hiệu lực, phiên bản thư viện.
- **Giới hạn import lười:** khóa R55 không khởi tạo predictor; module Ultralytics nạp lười trong `__init__`/
  `setup_model`/`stream_inference` có trong khóa nhờ kế thừa khóa R54 (khởi tạo predictor + rà tĩnh + smoke R54B). Mỗi
  công việc kiểm module nạp **trước** (ngoài khóa hoặc lệch hash → exit 2, không suy luận) và **sau từng đơn vị** (ngoài
  khóa → không ghi marker, exit 4).

**Thực thi** (ghi `results/phase2/r55_tracking/thuc_thi/<phiên>/`): `--buoc thuc-thi --cong-viec <việc>` với tham số
bắt buộc theo việc; **không có mặc định "chạy tất cả"**; import module không chạy gì; đường thật còn bắt buộc
`--duyet <mã plan thực thi đã duyệt>` (R55 không cấp). Mỗi lệnh kiểm lại khóa R55 gắn với phiên (`phien.json`) trước khi làm.

## 4. Thứ tự công việc và đơn vị tiếp tục

`cache → xac-nhan-cache → tracking → cham → hau-kiem` (hợp đồng `thu_tu_cong_viec`), thêm `lap-tracker`, `cham-lai`.

| Việc | Đơn vị | Thư mục lần thử | Ràng buộc trong marker |
|---|---|---|---|
| cache | detector × video | `cache/<det>/<video>/lan_*` | khóa, hợp đồng, giao thức, checkpoint, `spec_sha256`, video, số ảnh, hash danh sách ảnh, cấu hình cache |
| xác nhận cache | detector | `xac_nhan_cache/<det>/lan_*` | khóa, hợp đồng, giao thức, detector; manifest bất biến (một bản **công bố**/detector; công bố = `HOAN_TAT.json` ghi sau `manifest_cache.json` và `SHA256SUMS`) |
| tracking | run × video | `tracking/<run>/<video>/lan_*` | khóa…, run, điều kiện, bộ lọc, seed, SHA YAML, SHA cấu hình hiệu lực, backend, SHA manifest cache, SHA NPZ |
| chấm | run (đủ video) | `cham/<run>/lan_*` | khóa…, hash bộ MOT, hash GT/seqinfo đã khóa, cấu hình evaluator, seq_info |

Quy tắc:
1. Mỗi lần thử là thư mục **mới** (`lan_<thời điểm>_<ns>_<đếm>_pid`, sắp theo tên = thứ tự tạo); tồn tại → từ chối.
2. `HOAN_TAT.json` ghi **cuối cùng**, chứa ràng buộc + SHA256 từng file của lần thử. Chỉ tái sử dụng lần có marker,
   ràng buộc trùng ràng buộc hiện tại và băm lại khớp. Nhiều lần hợp lệ → lấy tên nhỏ nhất.
3. Lần không marker (dở), marker **hỏng** (JSON bị cắt/không đọc được — trạng thái `hong`), marker không khớp hoặc file
   lệch: **giữ nguyên**, ghi vào nhật ký, bỏ qua và tìm tiếp / tạo lần mới; không nối file dở. Lỗi trong lần thử →
   `LOI.json` trong chính thư mục đó, không marker.
3b. Sau mỗi đơn vị và **trước** marker/kết luận thành công: mọi module src/Ultralytics/TrackEval đang nạp phải nằm trong
   khóa **và** băm lại khớp khóa (`kiem_module_sau`); lệch → `LOI.json`, không marker, exit 4.
4. `nhat_ky/<việc>/lan_*/ket_qua.json` ghi tham số, lần thử đã đánh giá, tái sử dụng/mới, có gọi detector không, exit.

## 5. Cache bất biến

- `xac-nhan-cache --detector D` chỉ chạy khi **đủ video của manifest** (thật: 40) có lần cache hợp lệ; kiểm từng
  video: khóa/dtype NPZ, `data` (N,6), `cls_goc` (N,), `n_per_frame` shape (số ảnh,) không âm và tổng = N (vì
  `VideoDetections` dựng offsets từ mảng này), `frame_idx`/`time_s`/unreadable/truncated shape (số ảnh,); `cls` COCO = ánh xạ(`cls_goc`) theo `names` (coco80 giữ ID;
  lop5 ⊆ {1,2,3,5,7}); `frame_idx` = 1..số ảnh bảng hash = seqLength; `time_s = (i-1)·(1/fps)` như
  `DetracDataset.frames`; `orig_shape` 540×960; unreadable = truncated = 0; JSON video khớp; `spec_sha256` đồng nhất
  json/meta/marker; `meta.json` = cấu hình cache giao thức; `check_compatible` c25/c10; `sequence_spec_status` ok; SHA
  NPZ/JSON/meta khớp marker. Đạt → `manifest_cache.json` → `SHA256SUMS` → `HOAN_TAT.json` (ghi cuối = **công bố**).
- Chỉ lần xác nhận có marker hợp lệ mới là manifest đã công bố. Lần ngắt trước `SHA256SUMS`, trước marker hoặc marker bị
  cắt = **chưa công bố**: giữ nguyên, tracking vẫn bị chặn, `xac-nhan-cache` được chạy lại vào thư mục mới từ cache đã
  hoàn tất (không gọi detector). Bản đã công bố (marker đọc được) mà lệch hash/ràng buộc → chặn cả tracking lẫn xác
  nhận lại (exit 2), không tạo bản mới.
- Manifest là **đầu vào chuẩn duy nhất** cho mọi tracker/c25/c10 của detector, kể cả khi tiếp tục. Đã có manifest công
  bố → `cache` và `xac-nhan-cache` của detector đó bị từ chối (không gọi detector, không thay cache). Nhiều bản công bố
  → lỗi. Marker công bố bị hỏng **sau** khi ghi xong (can thiệp ngoài) không phân biệt được với ngắt lúc ghi, nên lần
  xác nhận có `manifest_cache.json` hoặc marker hỏng được coi là **có thể đã công bố** (vòng 3, F1): (a) `cache` đóng
  băng — chỉ tái sử dụng lần hợp lệ; video cần lần mới → exit 2 trước khi nạp model, không gọi detector; (b) lần cache có
  marker đọc được nhưng sai hash → `cache` và `xac-nhan-cache` chặn cả detector (không suy luận lại để thay); (c) xác
  nhận lại chỉ được tạo manifest mới khi trỏ **đúng** lần cache + SHA256 npz/json/meta/marker của mọi bản manifest
  trước đọc được, khác → chặn. Manifest mới có SHA khác nên mọi lần tracking **và chấm** gắn SHA manifest cũ không được
  tái sử dụng/chọn (F6, mục 6). Ràng buộc: nếu sau một lần ngắt thật giữa lúc ghi marker mà cache bị đổi, bộ điều
  phối chặn thay vì tự phục hồi — cần người dùng quyết định.
- `tracking`/`cham` băm lại mọi NPZ/JSON/meta trong manifest của **mọi detector cần dùng trước khi tracking bất kỳ run
  nào**; thiếu manifest hoặc sai hash → exit 2, không tạo lần tracking. Sau tracking băm lại lần nữa.
- Không dùng cache legacy, NPZ detector R53 hay cache smoke R54: đường dẫn cache chỉ lấy từ manifest trong phiên.

## 6. Chấm, chấm lại, lặp tracker, hậu kiểm

- `cham` chỉ khi **mọi run được yêu cầu** đủ video có tracking hợp lệ; sao MOT đã băm vào lần chấm (kiểm hash trước
  sao, sau TrackEval), GT/seqinfo đọc tại chỗ nhưng phải nằm trong khóa và trùng hash trước/sau; metric phải hữu hạn.
  Ràng buộc chấm (`rang_buoc_cham`, vòng 3 F6): cơ sở phiên, run, split, SHA GT, **SHA manifest cache**, danh sách lần
  tracking (`mot_lan_ds`), SHA MOT, cấu hình đánh giá, seq_info. Tái sử dụng (`cham`) và chọn lần chấm (`cham-lai`,
  `hau-kiem` qua `lan_cham_da_chon`) đều so **toàn bộ** ràng buộc qua `chon_lan`.
- `cham-lai --run R`: đọc **đúng bản sao MOT** trong lần chấm hoàn tất (băm lại theo marker), chấm vào thư mục mới, so
  byte 3 file metric; không gọi detector/tracker.
- `lap-tracker --run R --video V`: chạy lại tracker từ **cùng cache/seed** vào thư mục mới, so byte MOT.
- **Không** đặt tiêu chí suy luận detector lại trùng byte. Quan sát hai phiên smoke R54B (chỉ ghi nhận): `yolov8n_*`
  khác giá trị cache dù cùng số box (ví dụ `yolov8n_coco-c25-cv` MOTP lệch ~3e-8), `yolo11n_mixed5` trùng byte trong
  hai phiên đã kiểm; **chưa biết nguyên nhân, không suy rộng sang chất lượng hay tính tất định test40.** Kiểm biên độ
  lệch box/conf bằng đọc NPZ là tùy chọn ngoài R55, cần duyệt riêng.
- `hau-kiem`: băm lại manifest/cache, chọn lại lần tracking/chấm của 36 run, kiểm module sau công việc; đủ → exit 0,
  chưa đủ → exit 3; liệt kê lần không marker và lần marker hỏng (cache, xác nhận cache, tracking, chấm — giữ nguyên).
- **Vùng ghi độc lập:** mọi đích ghi (phiên `chuan_bi`/`thuc_thi`, phiên thực thi, thư mục cha của lần thử, file
  `SHA256SUMS`) phải nằm trong gốc vùng ghi **lexical** (thật: `results/phase2/r55_tracking`; fixture: thư mục chứa
  `fixture.json`) và `resolve(đích) == abspath(đích)` — không thành phần nào là junction/symlink. Kiểm **trước** khi tạo
  phiên/cấu hình thư viện (`kiem_doc_lap`), không dựa vào `W.tao_phien` (helper R54 resolve cả gốc lẫn đích).

## 7. Mã thoát

0 đạt · 1 từ chối ghi đè (phiên/manifest đã có) · 2 đầu vào/khóa/cache/tham số lệch (chặn trước khi làm) · 3 tiền kiểm
không đạt / hậu kiểm chưa đủ · 4 lỗi lúc chạy (TrackEval, module ngoài khóa, lặp không trùng) · 9 ngắt giả lập (fixture).

## 8. Fixture R55A (CPU, dữ liệu tổng hợp)

Dữ liệu và detector giả lập dùng lại `tao_fixture`/`MoHinhGia` của fixture R54 (2 video × 20 ảnh; video 1 có frame
8–10 không GT, video 2 có 3 ảnh D4 18–20), đi qua CLI của bộ điều phối và hàm cache/tracker/TrackEval thật. Ngắt giả lập
(`R55_NGAT`, `os._exit(9)`), đổi backend và chuỗi chuẩn bị chạy bằng tiến trình con thật; các ca khác gọi `main()`
trong tiến trình để tiết kiệm thời gian nạp thư viện. Ca kiểm: import không chạy; thiếu `--duyet`/`--cong-viec`/
`--phien`/tên khóa; env fixture ngoài `tests/`; chuỗi tiền kiểm → khóa (tên tường minh) → kiểm khóa; mở phiên trùng tên
(exit 1) và tên thoát vùng; thiếu video cache (ngắt sau marker) → xác nhận và tracking bị chặn; ngắt trước marker → lần
dở giữ nguyên, tiếp tục chỉ làm video thiếu; manifest bất biến, cache sau manifest không gọi detector; `kiem_npz` bắt
sai timeline/frame/unreadable/ánh xạ/thiếu khóa; NPZ/meta/JSON sai hash → tracking bị chặn trước tracking; tracking ngắt
trước/sau marker; chấm khi thiếu video bị từ chối; đổi backend → exit 2; đổi khóa (phiên gắn khóa khác, file đã khóa bị
sửa) → exit 2; đổi seed/cấu hình/backend/khóa/NPZ trong ràng buộc → không tái sử dụng; 36 run tracking + chấm; chấm lỗi
giả lập → lần lỗi giữ nguyên, chấm lại tạo lần mới; frame không GT/D4 vẫn qua tracker và evaluator; lặp tracker trùng
byte; chấm lại trùng byte, MOT bản sao bị sửa → chặn; hậu kiểm 36/36; đường dẫn thoát vùng; snapshot không ghi ngoài vùng.

Vòng sửa F1–F5 thêm/đổi ca (mục 11): C3 thêm marker cache bị cắt; C4 thêm ngắt xác nhận trước `SHA256SUMS`/trước
marker/marker cắt và `n_per_frame` sai giữ tổng (âm; thiếu phần tử) qua CLI; C5 thêm hai biến thể `n_per_frame`; D1 thêm
manifest đã công bố sai hash; F2 thêm marker chấm bị cắt; G2 chấm lại bỏ qua lần marker cắt; G3 liệt kê lần marker hỏng
và xác nhận chưa công bố; G4 kiểm module sau công việc ở mức hàm; H2 junction.

Giới hạn fixture: không YOLO/`fill_env`/GPU; dữ liệu nhỏ; khóa fixture kế thừa file khóa R54 thật (chỉ đọc) và dùng
một kết quả test **giả** trong vùng fixture; snapshot chỉ phủ thư mục liệt kê; marker cắt là giả lập (ghi nửa đầu byte
rồi `os._exit(9)`), không phải mất điện/lỗi hệ thống tệp thật.

## 9. Việc còn thiếu, chuyển sang plan thực thi sau R55 (bắt buộc)

- **Phép đo thời gian PF500/UKF trên đoạn dày: CHƯA LÀM.** Plan thực thi phải chọn trước đoạn dày cố định (video/frame)
  với căn cứ mật độ **không dựa trên metric**, dùng cache cùng khóa đã đủ 40 video của detector tương ứng, chạy PF500/UKF
  foreground, ghi thời gian từng run, detector/cấu hình/seed, phạm vi tính thời gian và giả định ngoại suy; phải được
  duyệt trước khi chạy; không xếp hạng, chọn tham số hay đổi giao thức từ phép đo.
- **Dự toán tạm thời** (hợp đồng `du_toan_tam_thoi`, từ smoke R54B lần 2, quy đổi tuyến tính ×220,08): suy luận 3
  detector ~3.519 s, tracking 36 run ~3.247 s, chấm ≥ 8 s, tổng ~6.774 s. Là kịch bản theo giả định (128 frame đầu
  đại diện mật độ — chưa kiểm), **không thay** phép đo đoạn dày.
- 173 ảnh D4 vẫn mở (giữ trong timeline, chấm như frame không GT; không diễn giải là âm tính đã xác nhận).
- R55B (sau khi Codex approve R55A): chạy `tien-kiem`, `khoa`, `kiem-khoa` trên đầu vào thật; không cache test40.

## 10. Bằng chứng R55A (mục 0 PLAN đã cập nhật trước khi hoàn thiện mục này)

- Lệnh foreground: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r55_tracking/test_dieu_phoi_test40.py`
  → **exit 0, 20/20 đạt**, 194,2 s, 27 lệnh CLI tiến trình con:
  `results/phase2/r55_tracking/tests/lan_20261021_054108_288700_pid25844/` (`ket_qua_test.json`, `cli_log.json`, toàn bộ
  phiên fixture trong `fx/`).
- SHA256 ghi trong kết quả (bản cuối, không sửa sau lần chạy): `dieu_phoi_test40.py` `74b9cb6df9b8d7804634646ce4647b64b19965696aa8b13f4e40807477fd6d68`,
  `test_dieu_phoi_test40.py` `bdba2f7b09a2c0761967aced4dbadf21f48de7143349d7908cbb39d2ab65666c`, hợp đồng
  `bed7ba74f18f263758ff7830d13ab14af7bf05321d736aead21a789183672f93`. Kết quả cũng ghi hash tài liệu này **trước** khi
  thêm mục 10 (`c2ebfe70…2585b91`); khóa R55B sẽ băm bản hiện hành.
- Lần chạy trước giữ nguyên: `lan_20261021_053138_647000_pid19852` và `lan_20261021_053236_257100_pid22868` (3/20 — lỗi
  tên biến trong `khoa`; assert đường dẫn Windows trong test), `lan_20261021_053307_766200_pid17024` (20/20 nhưng trước
  khi thêm kiểm môi trường cho `lap-tracker`/`cham-lai` và tăng tốc kiểm module sau từng đơn vị — không dùng).
- *Mục 10 là bằng chứng vòng đầu, bị thay bởi mục 11 (Codex `review_001.json` lượt R55: revise F1–F5).*

## 11. Vòng sửa R55A theo Codex review_001 (revise F1–F5)

Nguồn: `D:/LABOTARY/automation/runs_local/bytetrack_r55_prepare_20261021_051232/reports/review_001.json` (review tĩnh,
verdict `revise`). Chỉ sửa trong write_paths R55A; helper R54/protected không đổi.

| Finding | Sửa | Ca hồi quy |
|---|---|---|
| F1 [P1] manifest được coi là công bố chỉ vì file tồn tại | công bố = `HOAN_TAT.json` ghi cuối (sau manifest + `SHA256SUMS`); lần chưa công bố giữ nguyên, xác nhận lại vào thư mục mới không gọi detector; bản công bố lệch hash → chặn | C4 (3 điểm ngắt, tracking bị chặn sau mỗi điểm, xác nhận lại `goi_detector=false`), D1 (manifest công bố sai hash → tracking và xác nhận lại exit 2), G3 |
| F2 [P2] marker hỏng làm vòng chọn dừng | `doc_marker` trả `None` cho JSON cắt; `danh_gia_lan` trạng thái `hong`; xác nhận cache, `lan_cham_da_chon` (chấm lại, hậu kiểm) bỏ qua và tìm tiếp | C3/C4 (marker cache cắt), F2/G2 (marker chấm cắt đứng trước lần hợp lệ), G3 (`lan_marker_hong_giu_nguyen`) |
| F3 [P2] `n_per_frame` chỉ kiểm tổng | kiểm shape (số ảnh,), không âm, tổng = N; thêm shape `cls_goc`, `frame_idx`, `time_s`, unreadable/truncated | C5 (2 biến thể), C4 (qua CLI: xác nhận exit 2, tracking exit 2, không thư mục tracking) |
| F4 [P2] không kiểm module sau chấm/lặp/chấm lại; chỉ kiểm hiện diện | `kiem_module_sau` (hiện diện **và** SHA256) trước marker cache/xác nhận/tracking/chấm, trước kết luận `lap-tracker`/`cham-lai` (ghi `kiem_module_sau` trong kết quả), trong hậu kiểm | G4 mức hàm: khớp khóa; khóa trong bộ nhớ với SHA sai → phát hiện; tập file bằng hàm R54. **Không** có ca tiêm module giả trong một đơn vị: người dùng từ chối P6fb1cad7 |
| F5 [P1] junction ở `chuan_bi`/`thuc_thi` thoát vùng | `kiem_doc_lap` với gốc lexical trước `tao_phien`/`dat_moi_truong`, ở `Phien`, `lan_moi`, `SHA256SUMS` | H2: junction `chuan_bi`/`thuc_thi` (tien-kiem/khoa/kiem-khoa --luu/mo-phien), phiên junction, thư mục cha lần thử junction → exit 2, đích junction không bị ghi |

Ghi chú hiệu năng: kiểm module sau **mỗi** đơn vị làm chậm fixture (lần thử có resolve lại mọi `sys.modules`); bản cuối
nhớ kết quả `resolve` theo chuỗi `__file__` và ánh xạ đường dẫn khóa theo đối tượng khóa, **vẫn băm lại từng file** mỗi
lần. Tập file bằng `W.file_module_da_nap` được kiểm trong G4. Chi phí thêm trên đường thật chưa đo (không chạy thật).

Lần chạy dở giữ nguyên: `lan_20261021_060404_793900_pid9884` — 17/22, exit 1, 631 s; G2–H2 và Z1 lỗi vì mã điều phối
bị sửa **trong lúc** fixture chạy (khóa fixture phát hiện SHA mã đổi — đúng hành vi), không dùng làm bằng chứng.

## 12. Bằng chứng vòng sửa (mục 0 PLAN đã cập nhật trước mục này)

- Lệnh foreground: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r55_tracking/test_dieu_phoi_test40.py`
  → **exit 0, 22/22 đạt**, 263,1 s, 36 lệnh CLI tiến trình con:
  `results/phase2/r55_tracking/tests/lan_20261021_061607_226300_pid7220/` (`ket_qua_test.json` kết quả từng ca,
  `cli_log.json`, phiên fixture trong `fx/`).
- SHA256 ghi trong kết quả (bản cuối, không sửa sau lần chạy): `dieu_phoi_test40.py`
  `16ab6ffce00d80e3188c60d8d16d10bb9de6904f86e40fb430ca807e58c6fe8a`, `test_dieu_phoi_test40.py`
  `7c0313b28008623fea107f96c16214f56fc2a125381453f98a85fe13e06bd2b1`, hợp đồng
  `0c8285713cd9f510e40f3af3dc459e7d2d222a19c2e1c6ce0d82cb6b93459b8c`; tài liệu này `89bdce01…4322467` là bản **trước**
  khi thêm mục 12. Mục 10 (vòng 1, `lan_20261021_054108_288700_pid25844`) là lịch sử.
- `git status --short` sau lần chạy: cùng tập mục chưa commit như đầu lượt (thay đổi R55 chỉ trong
  `configs/phase2/r55_tracking/`, `src/r55_tracking/`, `results/phase2/r55_tracking/`, `handoff/logs/review55/`, mục 0).
- Không suy luận/tracking/chấm dữ liệu thật, không `setup_model` checkpoint thật, không GPU, không Git ghi. Phép đo đoạn
  dày vẫn **chưa làm** (mục 9).
- *Mục 12 là bằng chứng vòng 2, bị thay bởi mục 14 (Codex `review_002.json`: F2–F5 đóng, F1 còn mở, F6 mới).*

## 13. Vòng sửa 3 theo Codex review_002 (F1 còn mở, F6)

Nguồn: `D:/LABOTARY/automation/runs_local/bytetrack_r55_prepare_20261021_051232/reports/review_002.json` (review tĩnh,
verdict `revise`). Chỉ sửa write_paths R55A; helper R54/protected không đổi; không tiêm module (P6fb1cad7 từ chối),
không marker nguyên tử (P9857b26c từ chối). Đính chính vòng 2: khẳng định ở mục 5 cũ rằng lần chấm gắn manifest cũ
không được tái sử dụng là **sai** (ràng buộc chấm khi đó chỉ có SHA MOT), và nhánh `cache` khi marker xác nhận hỏng vẫn
gọi detector thay cache sai hash.

| Finding | Sửa | Ca hồi quy |
|---|---|---|
| F1 [P1] marker xác nhận hỏng/mất luôn coi là chưa công bố → `cache` loại lần sai hash và gọi detector thay cache | `manifest_chua_cong_bo`: lần xác nhận có manifest/marker hỏng = có thể đã công bố → `cache` đóng băng (exit 2 trước khi nạp model); lần cache marker đọc được nhưng sai hash → `cache`/`xac-nhan-cache` chặn; xác nhận lại phải trỏ đúng lần cache + SHA của bản manifest trước | I1 (marker công bố bị cắt + NPZ sai hash: `cache` exit 2 `goi_detector=false` không lần mới, xác nhận exit 2, tracking exit 2; biến thể marker cache hỏng → "đóng băng"; byte cache/xác nhận không đổi); C3/C4 (ngắt trước công bố) vẫn đạt |
| F6 [P2] ràng buộc chấm thiếu SHA manifest | `rang_buoc_cham` thêm `manifest_cache_sha256` + `mot_lan_ds`; `lan_cham_da_chon` so toàn bộ ràng buộc | I2 (manifest mới trỏ đúng lần cache cũ; tracking lần mới MOT trùng byte; `cham` tạo lần mới, không tái sử dụng lần cũ cùng `mot_sha256_ds`; `cham-lai` chọn lần mới; hậu kiểm exit 3 liệt kê đúng 11 run chưa tracking lại, đếm 25 chấm; lần xác nhận marker hỏng giữ nguyên byte) |

Hợp đồng cập nhật mô tả tương ứng (`cau_truc.xac_nhan_cache`, `quy_tac_tiep_tuc` cache F1 vòng 3 và chấm F6); cấu trúc
khóa/36 run/thứ tự công việc không đổi.

## 14. Bằng chứng vòng sửa 3 (mục 0 PLAN đã cập nhật trước mục này)

- Lệnh foreground: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r55_tracking/test_dieu_phoi_test40.py`
  → **exit 0, 24/24 đạt**, 295,1 s, 36 lệnh CLI tiến trình con:
  `results/phase2/r55_tracking/tests/lan_20261021_115928_508900_pid21264/` (`ket_qua_test.json` kết quả từng ca,
  `cli_log.json`, phiên fixture trong `fx/`).
- SHA256 ghi trong kết quả (bản cuối, không sửa mã/test/hợp đồng sau lần chạy): `dieu_phoi_test40.py`
  `b9e7f19941296bee8ff236fd67ee2687313b8d6c68765d52dc0b7b013f23c047`, `test_dieu_phoi_test40.py`
  `9f8a1bcd09e7140a1ea511aea72f0cd3a585485d1d0f99c461bd0a7f6859af05`, hợp đồng
  `626dd24dd9d753dc7c370cbc7cc3a4809f3447b0e5a3e1ab1789db9d7df12c76`; tài liệu này `bbdeaa8c…10bdd1d78` là bản **trước**
  khi thêm mục 13–14 và sửa mục 5–6. Mục 12 (`lan_20261021_061607_226300_pid7220`) và mục 10 là lịch sử, giữ nguyên.
- Không suy luận/tracking/chấm dữ liệu thật, không `setup_model` checkpoint thật, không GPU, không Git ghi. Phép đo đoạn
  dày PF500/UKF vẫn **chưa làm**, bắt buộc chuyển sang plan thực thi sau R55 (mục 9); dự toán smoke chỉ tạm thời.
- Giới hạn: fixture 2 video × 20 ảnh; marker hỏng là cắt giả lập; review/tự kiểm không chạy đường thật 40 video.
