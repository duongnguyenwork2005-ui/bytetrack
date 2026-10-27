# R58A — Hồ sơ thiết kế đối chứng SORT–ByteTrack trên UA-DETRAC test40

> **DỰ THẢO — CHƯA KHÓA — CHƯA CẤP QUYỀN CHẠY.** Hồ sơ này chỉ để người dùng chốt phạm vi. Nó không chọn c25/c10,
> không chọn `det_thresh`, không chốt ma trận run, không cho phép chạy tracker, evaluator, kiểm khóa hay tạo cache.

Ngày: 2026-10-23. Người soạn: Claude (`claude-opus-5-5`, effort high). Reviewer: Codex (một mốc cuối).
Căn cứ quyền: plan R58 (Codex đề xuất sau lượt `bytetrack_r57_bien_thien_20261023_0029`, lời gọi `0005_planner`),
người dùng duyệt; chỉ bước R58A.
Vòng sửa 1 (theo Codex `review_001.json`, revise): sửa mục 1.1 (điều kiện xóa/xuất SORT), 4.3 điều 2 (tái lập
detector), 5 lựa chọn (ii) (diễn giải FP), 7 K4 tách K4a/K4b (detection rỗng ≠ D4); ghi hai ý tưởng bị từ chối ở 6, 9.

Cách làm: **chỉ đọc tĩnh** mã, YAML và metadata nhỏ. Không import, không chạy Python, không chạy test. Mọi khẳng định
"theo đọc mã" là suy luận từ mã nguồn, **chưa kiểm runtime**. Mọi số liệu trích ở đây đều đã có trong nguồn được dẫn;
hồ sơ không tính số mới.

Nguồn đã đọc (gọi tắt ở các mục sau):

| Tắt | Đường dẫn |
|---|---|
| [GT51] | `handoff/logs/review51/01_giao_thuc.md` (mục 2, 4–8) |
| [TH51] | `handoff/logs/review51/03_tong_hop_va_slide.md` (mục 1, 3, 7) |
| [GT54] | `configs/phase2/r54_tracking/giao_thuc_test40.json` |
| [HD55] | `configs/phase2/r55_tracking/hop_dong_dieu_phoi.json` |
| [K56] | `results/phase2/r55_tracking/chuan_bi/khoa_20261021_155510_440500_pid8768/khoa.json` |
| [PH56] | `results/phase2/r55_tracking/thuc_thi/phien_20261021_161832_735800_pid11640/phien.json` |
| [NG56] | `results/phase2/r56_tong_hop/phien_20261021_161832_735800_pid11640/nguon.json` |
| [BC56] | `handoff/logs/review56/03_thuc_thi_tracking_test40.md`; nghiệm thu `D:/LABOTARY/automation/nghiem_thu_test40/lan_013/answer.json` (tóm tắt ở `handoff/PLAN_DA_CHOT.md` mục 0) |
| [SORT] | `src/sort_tracker.py` |
| [RT] | `src/run_tracking.py` |
| [CTRV] | `src/tracker_ctrv.py` |
| [CB54] | `src/r54_tracking/chuan_bi_test40.py` |
| [AL] | `src/assoc_log.py` |
| [DC] | `src/det_cache.py` |
| [CFG] | `src/config.py` |
| [time_units] | `src/time_units.py` (`ElapsedStepsMixin`) |
| YAML | `configs/phase2/trackers/sort_cv_age30.yaml`, `configs/phase2/trackers/sort_ekf_ctrv_age30.yaml`, `configs/bytetrack_ekf_ctrv.yaml` |

## 1. Câu hỏi đối chứng

Câu hỏi (cùng dạng Q2 của R51, [GT51] mục 1 và 6): **với cùng detection đầu vào, cùng lớp và tham số bộ lọc, cùng
evaluator, association + vòng đời kiểu SORT khác ByteTrack bao nhiêu trên test40?** Đối chứng chính dự kiến:
`bt-cv − sort-cv30` và `bt-ekf − sort-ekf30`, tách theo detector (và điều kiện, nếu chọn c10).

Không làm trong câu hỏi này (giữ ranh giới [GT51] mục 1 và plan R58):
- Không ghép với Q1 (so bộ lọc) thành một bảng xếp hạng; không suy hiệu ứng bộ lọc từ đối chứng SORT.
- Không so detector với nhau, không chọn conf hay detector theo kết quả test.
- Không có SORT-UKF/PF, không có SORT nguyên bản, không phân tầng, không BDD/Nano/QJTF.
- Không chuyển quy tắc R0/R1 dev12 sang test40 ([GT54] `quyet_dinh_nguoi_dung.Q3`).

### 1.1 SORT nguyên bản khác SORT kiểu CV/EKF age30

| | SORT nguyên bản (`sort_orig`, không thuộc lượt này) | SORT-kiểu CV/EKF age30 (đề xuất) |
|---|---|---|
| Bộ lọc | `SortKF7` 7-D `[u,v,s,r,u',v',s']`, R/P/Q như mã gốc ([SORT] `SortKF7`, dòng 56–98) | `sort_cv`: `XYAHFilter` bọc `KalmanFilterXYAH` của ultralytics, cùng lớp với ByteTrack-CV ([SORT] dòng 101–118). `sort_ekf_ctrv`: `CTRVFilter` bọc `EKFTrackerCTRV` ([SORT] dòng 121–160) |
| `max_age` | 1 | 30 (`sort_max_age: 30` trong hai YAML) |
| Association / vòng đời | Như cột phải | 1 vòng Hungarian trên IoU, nhận IoU ≥ 0,3; xuất khi `time_since_update < 1 AND (hit_streak ≥ min_hits OR frame_id ≤ min_hits)`, `min_hits = 3` ([SORT] dòng 287); giữ track sau ghép khi `time_since_update ≤ max_age` ([SORT] dòng 290) ([SORT] `associate` dòng 213–229, `SORT.update` dòng 262–293) |

Điều kiện xóa theo đọc mã: mỗi predict tăng `time_since_update` 1 ([SORT] dòng 189), update ghép đặt về 0 (dòng 193);
sau ghép chỉ giữ track có `time_since_update ≤ max_age` (dòng 290). Với `max_age = 1`, track hụt 1 frame còn
`time_since_update = 1` nên **vẫn được giữ** (không xuất box ở frame đó vì điều kiện xuất cần `< 1`); hụt frame liên
tiếp thứ hai thì bị xóa. Với `max_age = 30`: bị xóa ở frame hụt liên tiếp thứ 31. Chú thích đầu module ([SORT] dòng 14,
"mất 1 frame là xóa") diễn đạt không khớp điều kiện dòng 290; hồ sơ theo mã. Chưa kiểm runtime.

Cách gọi đúng (giữ từ [GT51] mục 6, [TH51] mục 3): "association và vòng đời kiểu SORT, dùng bộ lọc của ByteTrack,
`max_age 30`". Không gọi là SORT nguyên bản; không gọi là "chỉ đổi lifecycle".

## 2. Thành phần giữ chung và thành phần khác

### 2.1 Giữ chung (theo thiết kế; kiểm runtime ở lượt sau, mục 7)

| Thành phần | Giá trị dự kiến (= R56) | Căn cứ |
|---|---|---|
| Tập | DETRAC-test, 40 video, 56.340 ảnh, 173 ảnh D4, 56.167 ảnh có `<frame>` | [GT54] `tap_du_lieu` |
| Detector / cache | 3 detector `yolov8n_coco`, `yolov8n_mixed5`, `yolo11n_mixed5`; cache trong phiên R56 (mỗi detector một manifest, mọi tracker dùng chung) | [GT54] `detector`, `cache`; [BC56] mục 2; [CB54] dòng 1334 (`mo_cache(phien / "cache" / detector, …)`) |
| Lọc lúc load | `conf > ngưỡng` (ngặt) → lớp `[2,3,5,7]` → cắt `[:300]` theo thứ tự NMS | [GT54] `tai`; [DC] `filter_dets` dòng 412–425 |
| Điều kiện | c25 = 0,25; c10 = 0,1 | [GT54] `tai.dieu_kien` |
| Thời gian | fps 25, `elapsed_mode skip` ⇒ mỗi frame đọc được là 1 bước (`n_steps = 1`) | [GT54] `tracker`; [RT] `_vong_lap_tracker` dòng 230–247; [CFG] `CTRV_DT = 1.0` dòng 272 |
| Timeline | Tracker chạy trên mọi ảnh đọc được của video, kể cả 173 ảnh D4; evaluator chấm D4 như frame không GT | [GT54] `danh_gia.anh_khong_frame_xml`; [RT] dòng 230–234 |
| Tạo tracker mỗi video | `track_video` → `make_tracker` → `reset()` (SORT: `reset_id` đặt lại đếm ID) | [RT] dòng 140–154, 197; [SORT] dòng 251–256 |
| Evaluator | TrackEval `MotChallenge2DBox`, HOTA/CLEAR/Identity, IoU 0,5, lớp vehicle gộp, GT/seqinfo DETRAC-test hiện hành | [GT54] `danh_gia` |
| Backend | ultralytics 8.4.141, numpy 2.4.6, scipy 1.17.1, lap 0.5.13 | [K56] `backend_tracker.fingerprint` |
| Mã SORT | `src/sort_tracker.py` sha256 `a4ab4b26…` có trong khóa R56 (vai trò `khoa_r54:ma_phu_thuoc`), trùng hash HEAD được nêu ở R51 | [K56] dòng 2387–2391; [GT51] mục 4.2 |

### 2.2 Khác nhau (đây là thứ được so)

Bảng ở [GT51] mục 6 áp dụng nguyên cho test40 vì mã không đổi (cùng hash `sort_tracker.py`); tóm tắt:

| Khía cạnh | SORT-kiểu (`sort_tracker.py`) | ByteTrack (ultralytics 8.4.141) |
|---|---|---|
| Ghép | 1 vòng Hungarian trên −IoU, giữ cặp IoU ≥ 0,3, mọi track; không dùng score | Vòng 1 điểm cao, chi phí fuse score; vòng 2 điểm thấp; vòng 3 unconfirmed ([GT51] mục 6) |
| Sinh / xác nhận | Mọi detection chưa ghép sinh track; xuất khi `hit_streak ≥ min_hits (3)` | Sinh khi score ≥ `new_track_thresh 0,25` ([GT51] mục 6) |
| Track chưa xác nhận | Sống tới `max_age` 30 | Xóa sau 1 frame hụt ([GT51] mục 6) |
| Predict khi mất dấu | Không chỉnh vận tốc chiều cao; mọi track đều predict ([SORT] dòng 271) | Vận tốc chiều cao = 0 khi không Tracked: CV `byte_tracker.py:82` ([GT51] mục 5), CTRV [CTRV] dòng 149 |
| Ngưỡng nhận detection | `conf > det_thresh` (ngặt) bên trong tracker ([SORT] dòng 267), **chồng lên** lọc `conf > conf_min` của runner | Ngưỡng tầng của BYTE trên tập runner đã lọc |
| Buffer | `sort_max_age` (YAML = 30). Override `track_buffer` của runner **không tác động** SORT: lớp chỉ đọc `sort_max_age` ([SORT] dòng 239; YAML ghi chú dòng 10) | `track_buffer` = 30 bước, override từ `track_buffer_sec 1,2` ([CB54] dòng 399–415) |

Ghi chú buffer: cả hai cùng 30 frame về giá trị, nhưng cơ chế khác. Nếu sau này đổi `track_buffer_sec`, SORT sẽ
**không** đổi theo. Ở thiết kế này giá trị khớp nên không phải điểm lệch; lượt chuẩn bị nên kiểm cả `sort_max_age` lẫn
`track_buffer` trong cấu hình hiệu lực.

## 3. Đối chiếu bộ lọc với R56 (và khác biệt R51/R56)

### 3.1 SORT-kiểu so với ByteTrack cùng bộ lọc ở R56

| Mục | sort-cv30 vs bt-cv | sort-ekf30 vs bt-ekf | Trạng thái |
|---|---|---|---|
| Lớp bộ lọc | Cùng `KalmanFilterXYAH` (qua `ul_backend`) ([SORT] dòng 38, 101–118) | Cùng `EKFTrackerCTRV` ([SORT] dòng 35, 129; [CTRV] `get_kalmanfilter` dòng 366–374) | đọc mã |
| Q / R / P0 | Mặc định ultralytics (vị trí 1/20, vận tốc 1/160 × h) ([GT51] mục 5) | `ctrv_kwargs_from_args`: không YAML nào có khóa `ctrv_*` ⇒ cả hai dùng mặc định [CFG]: `std_theta 0,010`, `std_omega 0,005`, `init_std_theta 1,50`, `init_std_omega 0,10`, `heading_std_after_init 0,30`, `min_speed_for_heading 0,5`, `omega_eps 1e-4` ([CFG] dòng 278–316; [CTRV] dòng 331–353) | đọc mã + YAML; giá trị hiệu lực từ instance **chưa kiểm** cho test40 |
| Khởi tạo chuyển động | — | Cả hai: thử `try_initiate_from_motion` ở mỗi lần update cho tới khi thành công, mốc là vị trí **quan sát** trước, `n_frames` = số bước từ quan sát trước. SORT: `_steps_since_obs` đếm số lần predict ([SORT] dòng 142–154); ByteTrack: `frame_id − self.frame_id` ([CTRV] dòng 196–206). Với `skip` (1 predict/frame) hai cách đếm được kỳ vọng bằng nhau | đọc mã, **chưa kiểm runtime** |
| Mốc quan sát đầu | SORT: `z` của detection ([SORT] dòng 138); ByteTrack: `mean[cx, cy]` ngay sau `activate` ([CTRV] dòng 213–218), cũng là quan sát đầu | | đọc mã |
| Predict track chưa xác nhận | SORT predict mọi track; ByteTrack chỉ predict `strack_pool` (khác vòng đời, thuộc "thứ được so") | | theo [GT51] mục 6; dòng ultralytics **chưa trích** trong lượt này |
| Box xuất | SORT: box từ trạng thái bộ lọc sau update ([SORT] `xyxy`, dòng 288); ByteTrack: box từ `mean` của track | | đọc mã SORT; phía ultralytics chưa trích |

### 3.2 Khác biệt cấu hình EKF giữa R51 (dev12) và R56 (test40)

Không được chuyển kết quả Q2 dev12 của R51 sang test40 khi chưa xử lý các khác biệt sau:

| Mục | R51 | R56 | Ảnh hưởng tới SORT test40 |
|---|---|---|---|
| YAML bt-ekf | `configs/bytetrack_ekf_ctrv.yaml` (qua kc48, `results/phase2/kc48_pf_dev/experiment.yaml` dòng 21) | Cùng file, sha256 `2c35f2da…` ([K56] dòng 181–183) | Cùng file; sha256 YAML ở R51 **chưa đối chiếu** trong lượt này |
| `ekf-chinh` (R48D, `std_theta 0,02`, `std_omega 0,01`) | Có trong kc48 nhưng **không** dùng cho Q2 | Không dùng | SORT-EKF YAML không có khóa `ctrv_*` ⇒ ứng với bt-ekf mặc định, **không** ứng với `ekf-chinh` |
| `config.py` (nguồn mặc định CTRV) | sha256 **LF** `0e3f8b75…` (`results/phase2/r51_controlled/protocol/audit_tong_hop.json` → `config_py`) | sha256 `37e4f562…`, size 17.173 ([K56] dòng 2304–2307) | Hai chuỗi khác nhau. **Chưa xác minh** khóa R56 băm byte hay LF, nên chưa biết nội dung có đổi không. Lượt chuẩn bị phải băm cùng quy ước và đối chiếu giá trị `CTRV_*` hiệu lực |
| `assoc_log` / `pf_diag` | `false` / `false` ([GT51] mục 2) | `true` / `true` ([GT54] `tracker`) | Xem 3.3 |
| Override buffer | Không override `track_buffer_sec` ([GT51] mục 2) | `{"track_buffer": 30}` ([K56] `cau_hinh_hieu_luc`) | SORT không đọc `track_buffer` (mục 2.2) |
| Detector / tập / D4 | YOLOv8n COCO, dev12, 572 ảnh không `<frame>`, R0/R1 | 3 detector, test40, 173 D4 chấm như frame không GT, không R1 | Kết quả R51 chỉ là bối cảnh, không phải đối chứng |

### 3.3 `assoc_log` với SORT (điểm cần quyết định)

R56 chạy ByteTrack với `assoc_log: true` ([GT54] `tracker.assoc_log`). Theo đọc mã, `AssociationLogMixin.multi_predict`
đọc `t.state`, `t.end_frame`, `t.is_activated`, `t.tracklet_len` của mỗi track ([AL] dòng 76–83, 120–124); `SORTTrack`
không có các thuộc tính này ([SORT] dòng 169–198). Như vậy bật `assoc_log` cho SORT nhiều khả năng lỗi ngay frame đầu
có track (**suy luận, chưa chạy**). R51 chạy sort-cv30/sort-ekf30 với `assoc_log=false` ([GT51] mục 2). `pf_diag` không
tác động SORT vì `SORT` không có `kalman_filter` ([RT] dòng 200). Mixin được ghi là không đổi kết quả ([AL] dòng 61),
nhưng đây là khác biệt cấu hình phải khai báo, không phải tự quyết.

## 4. Ma trận dự kiến (ĐỀ XUẤT — CHỜ QUYẾT ĐỊNH)

Đối chứng ByteTrack lấy **duy nhất** từ phiên R56 [PH56]; lần chấm nguồn ghi trong [NG56] (đường dẫn `cham/<run>/lan_…`).

### 4.1 Phương án A — chỉ c25: 6 run SORT mới

| Detector | Run SORT mới (tên đề xuất, chưa khóa) | Đối chứng ByteTrack (nguồn R56) |
|---|---|---|
| yolov8n_coco | `…-yolov8n_coco-c25-sort-cv30`, `…-sort-ekf30` | `r54-yolov8n_coco-c25-cv`, `r54-yolov8n_coco-c25-ekf` |
| yolov8n_mixed5 | `…-yolov8n_mixed5-c25-sort-cv30`, `…-sort-ekf30` | `r54-yolov8n_mixed5-c25-cv`, `r54-yolov8n_mixed5-c25-ekf` |
| yolo11n_mixed5 | `…-yolo11n_mixed5-c25-sort-cv30`, `…-sort-ekf30` | `r54-yolo11n_mixed5-c25-cv`, `r54-yolo11n_mixed5-c25-ekf` |

Tổng: 6 run × 40 video. Ở c25, `conf_min = det_thresh = 0,25` nên SORT và ByteTrack nhận cùng tập detection; vòng
ghép điểm thấp của ByteTrack rỗng (như R51, [GT51] mục 2). Đối chứng **không đo** lợi ích ghép detection điểm thấp.

### 4.2 Phương án B — c25 + c10: tổng 12 run SORT mới

Như A, thêm 6 run c10 (3 detector × sort-cv30/sort-ekf30), đối chứng `r54-<detector>-c10-cv` / `-ekf`. Phương án này
**chỉ có nghĩa** khi đã chốt `det_thresh` cho c10 (mục 5).

### 4.3 Điều kiện tái dùng run ByteTrack R56 (CHƯA KIỂM RUNTIME)

Tái dùng 6 hoặc 12 run ByteTrack (cv/ekf) R56 làm đối chứng chỉ hợp lệ khi lượt chuẩn bị chứng minh đủ:
1. Phiên R56 [PH56] và khóa [K56] (`ce85628e…9d35247b`) nguyên vẹn; MOT và lần chấm trỏ bởi [NG56] khớp hash đã ghi.
2. Run SORT đọc **đúng** manifest cache mà run ByteTrack đã dùng (cùng `phien/cache/<detector>`), băm lại khớp; không
   suy luận detector lại. Lý do: **chưa bảo đảm** suy luận detector tái lập byte giữa tiến trình — [HD55] `tai_lap.detector`
   (dòng 106) chỉ ghi ở smoke R54 YOLOv8n khác giá trị cache giữa hai phiên, YOLO11n trùng byte, chưa rõ nguyên nhân và
   không suy rộng; giao thức không đặt tiêu chí tái lập detector.
3. Cùng mã phụ thuộc (`run_tracking`, `det_cache`, `time_units`, `ul_backend`, `tracker_ctrv`, `ekf_ctrv`, `config`,
   `sort_tracker`) và backend như khóa R56, hoặc mọi lệch được liệt kê và người dùng chấp nhận.
4. Cùng evaluator: chấm lại một run ByteTrack nguồn từ bản sao MOT đã băm bằng đường chấm mới phải trùng byte 3 file
   metric với lần chấm R56 (cùng tinh thần `cham-lai`, [HD55]).
Nếu một điều kiện không đạt: ghi lại, dừng nhánh liên quan, người dùng quyết định; không tự chạy lại ByteTrack.

## 5. c10: lựa chọn `det_thresh` và vấn đề lọc kép

Hai YAML SORT đã khóa ghi `det_thresh: 0.25`. SORT lọc `conf > det_thresh` **bên trong** tracker ([SORT] dòng 267),
sau khi runner đã lọc `conf > conf_min` ([DC] dòng 422–425). Hồ sơ **không** chọn và **không** sửa YAML.

| Lựa chọn | Tập detection SORT thực sự nhận ở c10 (theo đọc mã) | Diễn giải và giới hạn |
|---|---|---|
| (i) giữ YAML hiện có, `det_thresh 0,25` | `conf > 0,25` ∩ lớp ∩ prefix `[:300]` của tập `conf > 0,1`. Cache đã sắp giảm dần theo conf ([DC] dòng 31), nên prefix này được kỳ vọng trùng tập c25 ⇒ **lọc kép**: SORT c10 được kỳ vọng cho MOT giống SORT c25 (chưa kiểm; còn phụ thuộc thứ tự khi conf bằng nhau) | 6 run c10 không thêm thông tin về SORT; đối chứng c10 khi đó = "ByteTrack nhận thêm detection 0,1–0,25 cho vòng 2" so với "SORT không nhận". Đây là so khác **tập đầu vào**, không còn "cùng detection" như câu hỏi mục 1 |
| (ii) YAML mới `det_thresh 0,1` (không sửa YAML khóa) | `conf > 0,1` ∩ lớp ∩ `[:300]`, cùng tập ByteTrack nhận | Cùng tập đầu vào (nên số FP **đầu vào** như nhau), khác ở **cách dùng** detection điểm thấp: theo đọc mã SORT ghép và sinh track từ mọi detection như nhau ([SORT] dòng 279–282); ByteTrack chỉ dùng chúng ở vòng ghép 2, không sinh track mới từ chúng ([GT51] mục 6). Có nguy cơ SORT phát sinh thêm track/FP **đầu ra** — chưa đo, không khẳng định. Hiệu số đo tổng thể association + vòng đời (cơ chế điểm, sinh/xác nhận, track chưa xác nhận), **không** tách riêng lợi ích vòng ghép điểm thấp. Cần YAML mới, khóa mới, ghi rõ là biến thể |
| (iii) không chạy c10 (= phương án A) | — | Không có đối chứng c10 |

Ngưỡng biên: so sánh đều là ngặt (`>`) ở runner và SORT; conf được giữ dạng float32 trong cache và qua `to_boxes`, SORT
ép sang float64 ([SORT] dòng 265). Phía ByteTrack, cách so tầng điểm cao/thấp (≥ hay >) của ultralytics **chưa trích dòng**
trong lượt này. Detection có conf đúng bằng ngưỡng (0,25 hoặc 0,1 ở float32) phải được kiểm riêng ở lượt chuẩn bị.

## 6. Điểm chưa xác minh (tổng hợp)

1. Hash `config.py` R51 (LF) và khóa R56 khác chuỗi; chưa biết quy ước băm của khóa R56 (mục 3.2).
2. Giá trị bộ lọc hiệu lực của SORT trên test40 chưa đọc từ instance (R51 đã đọc cho dev12, [GT51] mục 5).
3. SORT c10 với `det_thresh 0,25` trùng SORT c25: chỉ suy từ mã (mục 5). Đề xuất fixture kiểm riêng giả thuyết này
   (P3b9dd663) đã bị người dùng từ chối ở R58A; điểm này vẫn chưa xác minh.
4. `assoc_log` không tương thích SORT: chỉ suy từ mã (mục 3.3).
5. Hai cách đếm `n_frames` (SORT/ByteTrack-EKF) bằng nhau dưới `skip`: chỉ suy từ mã (mục 3.1).
6. Các dòng ultralytics (ngưỡng tầng, predict `strack_pool`, box xuất) chưa trích trong lượt này.
7. Điều kiện tái dùng run R56 (mục 4.3) chưa kiểm.
8. `ElapsedStepsMixin` được viết cho lớp kế thừa `BYTETracker` ([time_units] dòng 179–205); R51 đã chạy SORT qua runner
   ([TH51] mục 1, `moi`), nhưng đường điều phối test40 chưa chạy SORT lần nào.

## 7. Kiểm chứng cần có ở kế hoạch triển khai sau (đặc tả, chưa làm)

| # | Kiểm | Tiêu chí đạt |
|---|---|---|
| K1 | Parity cấu hình | Cấu hình hiệu lực (từ instance, không suy từ YAML) của sort-cv30/sort-ekf30 ghi vào tiền kiểm; tham số bộ lọc trùng bt-cv/bt-ekf R56; `sort_max_age 30`, `sort_iou_threshold 0,3`, `sort_min_hits 3`, `det_thresh` đúng giá trị người dùng chốt; `config.py` băm cùng quy ước với khóa R56 |
| K2 | Parity đầu vào | Với mỗi (detector, điều kiện, video): tập detection SORT nhận sau `det_thresh` được đếm và so với tập ByteTrack nhận; ở c25 phải trùng hoàn toàn; ở c10 báo số chênh theo đúng lựa chọn mục 5 |
| K3 | Ngưỡng confidence biên | Fixture CPU: detection conf đúng bằng 0,25 / 0,1 (float32), ngay trên/dưới; kiểm runner, SORT và tầng ByteTrack xử lý đúng như mô tả |
| K4a | Frame không detection | Fixture: frame không detection (sau lọc) giữa video, ở đầu và cuối video; SORT không lỗi, mọi track tăng `time_since_update`, frame đó không xuất box, track bị xóa đúng điều kiện `time_since_update > max_age` ([SORT] dòng 189, 287, 290) |
| K4b | Ảnh D4 | D4 chỉ là ảnh không có `<frame>` GT, **không** phải frame rỗng: giữ nguyên detection và timeline; tracker xử lý như mọi ảnh, có thể xuất box; box ở D4 được chấm như frame không GT (hành vi evaluator hiện hành; suy ra: không có GT để ghép nên box ở đó tính như FP; không diễn giải là âm tính chắc chắn) ([GT54] `danh_gia.anh_khong_frame_xml`, dòng 93). Tiêu chí: 173 ảnh D4 có mặt trong timeline tracker và MOT/evaluator đúng như R56; **không** đặt tiêu chí buộc D4 không có box. Timeline đủ 56.340 ảnh trên dữ liệu thật |
| K5 | Nguồn cache bất biến | Băm manifest/NPZ cache R56 trước và sau mỗi run SORT; không ghi vào phiên R56; không suy luận detector |
| K6 | Khóa mới | Khóa riêng cho SORT test40 (liên kết khóa R56 cha bằng hash), gồm mã, YAML (hiện có hoặc mới nếu chọn (ii)), giao thức, cấu hình hiệu lực; lệch → exit ≠ 0, không chạy |
| K7 | Smoke | Trên `DETRAC-smoke128` của R54 ([GT54] `smoke`): 2 video × 128 frame, cả 2 SORT, chấm được, assoc_log theo quyết định |
| K8 | Tái lập | Chạy lại ít nhất một (run, video) SORT từ cùng cache phải trùng byte MOT; chấm lại cùng MOT trùng byte metric; chấm lại một run ByteTrack R56 trùng byte (mục 4.3 điều 4) |

## 8. Đầu ra và tiêu chí đối chiếu metric (đề xuất)

- Vùng ghi đề xuất: một namespace mới (ví dụ `results/phase2/r5x_sort_test40/`), không ghi vào phiên R55/R56/R57.
- Mỗi run SORT: MOT 40 video, cấu hình hiệu lực, hash đầu vào/đầu ra, `combined_metrics` và `per_video` của TrackEval.
- Bảng đối chiếu: theo từng detector (× điều kiện nếu có c10) và từng bộ lọc, điểm COMBINED HOTA (chính), AssA, DetA,
  IDF1, IDSW của SORT và ByteTrack nguồn, cùng hiệu số điểm `bt − sort`. MOTA/FP chỉ tham chiếu ([GT51] mục 7).
- Tiêu chí đối chiếu: số ByteTrack trong bảng phải tái tạo đúng metric R56 ([NG56]) theo dung sai làm tròn đã khai báo;
  không so detector, không chọn conf.
- Khoảng tin cậy: **chưa duyệt**. Nếu người dùng muốn CI cho đối chứng SORT, cần một giao thức thống kê riêng được duyệt
  trước khi xem số; hồ sơ này không coi giao thức R57 là đã áp dụng cho SORT.

## 9. Phiếu quyết định (người dùng điền)

| # | Câu hỏi | Lựa chọn | Quyết định |
|---|---|---|---|
| Q-A | Phạm vi điều kiện | A: chỉ c25 (6 run) / B: c25 + c10 (12 run) | |
| Q-B | Nếu B: `det_thresh` SORT ở c10 | (i) 0,25 YAML hiện có (lọc kép) / (ii) YAML mới 0,1 | |
| Q-C | `assoc_log` cho SORT | tắt (như R51; khác R56) / khác (việc mới, cần duyệt riêng; đề xuất sửa mixin P74d51e88 đã bị người dùng từ chối ở R58A) | |
| Q-D | Tái dùng ByteTrack R56 làm đối chứng | đồng ý với điều kiện mục 4.3 / không | |
| Q-E | Khoảng tin cậy | không (chỉ điểm) / có, với giao thức riêng duyệt trước | |
| Q-F | Mục 6 điểm 1 (`config.py`) | kiểm ở lượt chuẩn bị / khác | |

## 10. Ranh giới các lượt tiếp theo

1. **Chốt thiết kế** (lượt riêng, người dùng duyệt): điền phiếu mục 9; nếu cần, cập nhật hồ sơ này thành bản khóa.
   Không viết mã, không chạy.
2. **Chuẩn bị mã / fixture / khóa** (lượt riêng): K1–K4, K6 trên fixture CPU; YAML mới nếu chọn (ii); smoke K7 chỉ khi
   plan lượt đó ghi rõ. Không chạy full.
3. **Duyệt chạy thật** (lượt riêng, người dùng duyệt tường minh): 6 hoặc 12 run SORT, chấm, K5, K8, bảng mục 8.

R58A dừng ở hồ sơ này; không tự sang lượt 1–3.
