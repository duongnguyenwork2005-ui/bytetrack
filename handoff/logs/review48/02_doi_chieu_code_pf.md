# R48A (M1) — Đối chiếu khái niệm PF với mã, sở hữu trạng thái, giả thuyết

Ngày 2026-10-10. Mọi dẫn chiếu `file:dòng` tại commit **`eaf2e877081938420508e696de22dbd2aa0bff97`** (HEAD). Các file
`src/pf_ctrv.py`, `src/tracker_ctrv.py`, `src/ekf_ctrv.py`, `src/run_tracking.py`, `third_party/ultralytics_trackers/`
không sửa so với HEAD (`git status --porcelain` rỗng, khóa `git` trong `results/phase2/kc48_pf_code/so_huu_hat.json`).
Lý thuyết và ký hiệu: `01_ly_thuyet_kf_ekf_pf.md`. Nhãn: [MÃ] đọc mã, [TN] kiểm thực nghiệm bằng script, [QS] số đo
đã có, [LT] lý thuyết, [GT] giả thuyết chưa kiểm. Đã sửa một lần sau review Codex vòng 1 (R48A-F1…F4); chỗ sửa liệt
kê ở mục 7.

**Hai backend.** Pipeline mặc định dùng gói `ultralytics` đã cài (8.4.141, ngoài repo); bản sao trong repo
`third_party/ultralytics_trackers/` dùng cho backend `np`. Dòng của lớp nền dưới đây dẫn theo bản sao trong repo; phần
quyết định kết luận (lời gọi bộ lọc trong `STrack.update` / `re_activate`) đã kiểm lại trên bản đã cài bằng `inspect`
(`so_huu_hat.json`, khóa `moi_truong.STrack_goi_bo_loc`). Test parity hai backend (`src/test_bytetrack_np_parity.py`)
không chạy lại trong R48A.

## 1. Luồng một frame của ByteTrack với PF [MÃ]

`byte_tracker.py` dưới đây là `third_party/ultralytics_trackers/byte_tracker.py`.

1. `BYTETracker.update` (`byte_tracker.py:266-300`) tách detection: điểm ≥ `track_high_thresh` = 0,25 → nhóm cao;
   0,1 < điểm < 0,25 → nhóm thấp (`byte_tracker.py:302-318`; yaml `configs/phase2/trackers/bytetrack_pf_ctrv_n300.yaml:5-6`).
2. Pool = track đã kích hoạt + track LOST (`byte_tracker.py:281-282`); `self.multi_predict(strack_pool)` (dòng 283)
   → `CTRVByteTracker.multi_predict` (`src/tracker_ctrv.py:373-380`) → `PFSTrack.multi_predict`
   (`src/tracker_ctrv.py:291-323`). Track **chưa xác nhận** không nằm trong pool nên **không được predict** trước khi
   ghép (`byte_tracker.py:342-352`) — như nhau cho mọi bộ lọc.
3. Ghép lần 1: detection cao × pool, chi phí `1 − IoU` rồi `fuse_score` thành `1 − IoU·score`
   (`byte_tracker.py:507-512`; `utils/matching.py:62-98, 145-167`), gán tuyến tính với ngưỡng `match_thresh` = 0,8
   (`byte_tracker.py:376`; `utils/matching.py:18-51`). [LT, từ mã] Ngưỡng IoU hiệu lực = 0,2 / score (score 0,9 → IoU ≥
   0,222; score 0,3 → IoU ≥ 0,667). Track đang Tracked → `update`, track LOST → `re_activate`
   (`byte_tracker.py:408-415`).
4. Ghép lần 2: track Tracked còn lại × detection thấp, chỉ IoU, ngưỡng 0,5 (`byte_tracker.py:417-434`); track không
   ghép → `mark_lost` (dòng 436-440).
5. Track chưa xác nhận × detection cao còn lại, ngưỡng 0,7; không ghép thì xóa (`byte_tracker.py:442-468`).
6. Detection cao còn lại có điểm ≥ `new_track_thresh` = 0,25 → `activate` (`byte_tracker.py:470-483`) →
   `PFSTrack.activate` (`src/tracker_ctrv.py:277-285`).
7. Xóa track LOST khi `frame_id − end_frame > max_frames_lost` = `track_buffer` = 30 (`byte_tracker.py:262, 485-490`).

`update` / `re_activate` của CTRV gọi hook khởi tạo hai khung hình trước lớp cha (`src/tracker_ctrv.py:220-233`), rồi
lớp cha gọi `self.kalman_filter.update(mean, cov, z)` **không** kèm confidence (`byte_tracker.py:115-117, 147-149`; bản
8.4.141 đã cài: cùng lời gọi, `co_truyen_confidence: false`). Với PF, `self.kalman_filter` là `PFTrackView` riêng của
track (`src/tracker_ctrv.py:285`) nên lời gọi đi vào `PFTrackView.update` (`src/pf_ctrv.py:338-348`) →
`PFCTRV.update_particles` (`src/pf_ctrv.py:255-292`) trên particle của **chính** track đó.

Khác bài ByteTrack (đã đối chiếu [BT] tr. 5): bài dùng τ = 0,6 và loại ghép khi IoU < 0,2; cấu hình repo dùng ngưỡng
cao 0,25 và ngưỡng IoU phụ thuộc điểm qua `fuse_score`. Giữ track mất 30 frame giống nhau.

## 2. Bảng khái niệm → mã [MÃ]

| Khái niệm | file:hàm:dòng | Mã làm gì |
|:--|:--|:--|
| Proposal (= prior) | `src/pf_ctrv.py:PFCTRV.propagate:207-213`; `PFCTRV.f_batch:159-179`; `PFCTRV.process_noise_std_batch:181-194`; đường pipeline `src/tracker_ctrv.py:PFSTrack.multi_predict:291-323` | X ← f(X) + N(0, Q(h_i)) với h của từng particle; quấn θ; chặn h ≥ 10⁻³ bằng `np.maximum` (phép chặn, không phải cắt cụt — mục 5 ý 7); không đụng trọng số. `multi_predict` gộp particle mọi track vào một lần `f_batch` rồi tách theo offset, nhiễu lấy từ rng **của từng track** (dòng 312-323) |
| Likelihood | `src/pf_ctrv.py:PFCTRV.log_likelihood:215-219`; `PFCTRV.update_particles:261-264` | −½ Σ d²/R_dd trên (c_x, c_y, a, h); R = `_measurement_noise(h_z, conf)` (`src/ekf_ctrv.py:584-600`) với h_z = h quan sát; conf = None trong pipeline |
| Cập nhật + chuẩn hóa trọng số | `src/pf_ctrv.py:PFCTRV.update_particles:282-283`; `_logsumexp:67-71` | log w + ℓ, trừ LSE ⇒ Σ w = 1 **trong từng track** |
| ESS trigger | `src/pf_ctrv.py:PFCTRV.resample_if_needed:240-245` | ESS = 1/Σw² trên trọng số đã nhân likelihood; resample nếu ESS < `resample_ess_ratio`·N; ratio = 0,5 (yaml dòng 14). `PFCTRV.ess:236-238` chỉ dùng cho chẩn đoán (dòng 284) |
| Resampling | `src/pf_ctrv.py:systematic_resample:74-80`; `PFCTRV.resample_if_needed:246-253` | systematic, một u ~ U[0,1) từ rng của track; `X[idx].copy()`; log w = −log N; tăng bộ đếm `n_resample` |
| Roughening | `src/pf_ctrv.py:PFCTRV.resample_if_needed:248-251` | nhiễu `roughening`·std(Q) sau resample; mặc định 0 = tắt (yaml dòng 16) |
| Số hạt | `src/pf_ctrv.py:PFCTRV.__init__:87-95`; `PFCTRV.sample_initial:199-205`; yaml dòng 12 | N cố định cho mọi track của tracker (100 / 300 / 500), ≥ 2; không thích nghi, không chuyển hạt giữa track |
| Ước lượng (`mean` / `map` / ma trận) | `src/pf_ctrv.py:PFCTRV.estimate_state:221-234`; gọi ở `update_particles:285` (trước resample), `src/tracker_ctrv.py:PFSTrack.multi_predict:322` và `src/pf_ctrv.py:PFTrackView.predict:336` (sau predict) | `mean`: trung bình có trọng số, θ trung bình vòng. `map`: particle có trọng số chuẩn hóa lớn nhất (`np.argmax`, dòng 225) — tên tùy chọn, **không** phải MAP của mật độ hậu nghiệm liên tục; khi trọng số đều (sau resample, reseed hay nhánh suy biến) `argmax` trả hàng 0, nên ước lượng sau predict là một particle tùy ý. Ma trận = Σ w̃ (X − ước lượng điểm)(·)ᵀ, phần dư θ quấn: hiệp phương sai khi `mean`, mô-men bậc hai quanh particle được chọn khi `map` (`01` mục 5, heuristic 6). Cả ba yaml đặt `pf_estimate: mean` (dòng 15) |
| Bbox cho association | `src/tracker_ctrv.py:CTRVSTrack.tlwh:158-172`; `byte_tracker.py:STrack.xyxy:172-177`; `utils/matching.py:iou_distance:62-98` | tlwh từ `track.mean[c_x, c_y, a, h]`, tức ước lượng ở dòng trên ⇒ IoU. Đám mây chỉ đi vào association qua **một** bbox |
| Khởi tạo track | `src/tracker_ctrv.py:PFSTrack.activate:277-285`; `src/pf_ctrv.py:PFCTRV.sample_initial:199-205`; `src/ekf_ctrv.py:EKFTrackerCTRV.initiate:458-491` | mean / cov = `initiate(z)` (tất định); rng riêng; N particle ~ N(mean, diag P₀); w đều; gắn `PFTrackView(pf, self)` |
| Khởi tạo hai khung hình | `src/tracker_ctrv.py:CTRVSTrack._init_motion_if_needed:177-206`; `src/pf_ctrv.py:PFTrackView.try_initiate_from_motion:350-366`; `PFCTRV.reseed_motion:294-306` | dùng luật EKF (`src/ekf_ctrv.py:493-543`) quyết định và lấy tốc độ / hướng; lấy lại v, θ của mọi particle; **ω, c_x, c_y, a, h giữ nguyên**; w đều |
| Track mất detection | `src/tracker_ctrv.py:PFSTrack.multi_predict:306-322`; `src/pf_ctrv.py:PFTrackView.predict:323-336`; `byte_tracker.py:436-440, 485-490` | mỗi frame LOST: chép mảng, đặt v_h = 0 cho mọi particle, propagate, **trọng số giữ nguyên**, mean tính lại; không resample khi không có đo; xóa sau 30 frame |
| Ghép lại sau mất | `byte_tracker.py:STrack.re_activate:113-127` → `src/pf_ctrv.py:PFTrackView.update:338-348` | cùng bước update như track đang bám (likelihood + resample); luật suy biến có thể kích hoạt |
| Luật suy biến | `src/pf_ctrv.py:PFCTRV.update_particles:265-281` | min Mahalanobis² (chỉ theo R) > `degenerate_maha2` = 10⁴ (yaml dòng 17) ⇒ đặt lại c_x, c_y, a, h quanh z (ghi **tại chỗ** vào mảng của track), giữ v, θ, ω, w đều |
| Seed / rng | `src/pf_ctrv.py:PFCTRV.set_seed:149-151`, `make_rng:153-154`; `src/tracker_ctrv.py:282, 420-421`; `src/run_tracking.py:198-199` | rng track = `default_rng(SeedSequence([seed, video_salt, track_id]))`, video_salt = 32 bit đầu sha1(tên video) |
| Chẩn đoán | `src/pf_ctrv.py:enable_diag:109-127`, dòng 277-280, 287-291; bộ đếm 102-105, 252, 260, 269; `src/tracker_ctrv.py:pf_stats:423-426` | chỉ ghi; bộ đếm không được đọc trong logic quyết định (grep: chỉ tăng ở 252 / 260 / 269 và đọc ở `pf_stats`) |

## 3. Sở hữu trạng thái

### 3.1 Theo đọc mã [MÃ]

| Thành phần | Chủ sở hữu | Tạo / thay | Chia sẻ? |
|:--|:--|:--|:--|
| `particles` (N, 9) | thuộc tính của từng `PFSTrack` (`src/tracker_ctrv.py:PFSTrack.__init__:273`, `PFSTrack.activate:283`) | predict: mảng mới (`src/tracker_ctrv.py:PFSTrack.multi_predict:318` — `allF[off:off+n] + nhiễu` là phép toán hai ngôi nên tạo mảng mới, không phải view; `src/pf_ctrv.py:PFCTRV.propagate:210` ở đường không gộp); resample: `X[idx].copy()` (`src/pf_ctrv.py:PFCTRV.resample_if_needed:247`); reseed: `X.copy()` (`src/pf_ctrv.py:PFCTRV.reseed_motion:303`). Ghi tại chỗ chỉ trên mảng của chính track: nhánh suy biến (`src/pf_ctrv.py:PFCTRV.update_particles:272-274`), quấn θ / chặn h trên mảng vừa tạo (`src/tracker_ctrv.py:PFSTrack.multi_predict:319-320`) | riêng |
| `logw` (N,) | từng track (`src/tracker_ctrv.py:PFSTrack.__init__:274`, `PFSTrack.activate:284`) | update: `logw + ll` là mảng mới, rồi trừ LSE tại chỗ trên mảng mới (`src/pf_ctrv.py:PFCTRV.update_particles:282-283`); mảng mới khi resample (`PFCTRV.resample_if_needed:253`), suy biến (`PFCTRV.update_particles:275`), reseed (`PFTrackView.try_initiate_from_motion:359`) | riêng |
| `rng` | từng track (`src/tracker_ctrv.py:PFSTrack.__init__:275`, `PFSTrack.activate:282`) | `SeedSequence([seed, video_salt, track_id])` (`src/pf_ctrv.py:PFCTRV.make_rng:153-154`) | riêng |
| `PFTrackView` | một cho mỗi track (`src/tracker_ctrv.py:PFSTrack.activate:285`) | giữ tham chiếu `pf` (chung) và `track` (chính nó) (`src/pf_ctrv.py:PFTrackView.__init__:318-320`) | riêng |
| `PFCTRV` | **một cho mỗi tracker** (`src/tracker_ctrv.py:PFByteTracker.get_kalmanfilter:412-418`; tạo lại ở `byte_tracker.py:BYTETracker.reset:523-530`); truyền cho mọi track khi activate (`byte_tracker.py:BYTETracker._init_new_tracks:482`); `PFSTrack.multi_predict` khẳng định mọi track dùng cùng instance (`src/tracker_ctrv.py:298-302`) | cấu hình (f, Q, R, N, ngưỡng, ước lượng, roughening, seed, video_salt) + bộ đếm / `diag_rows` (`src/pf_ctrv.py:PFCTRV.__init__:87-107`) | **chung có chủ đích** (cấu hình + bộ đếm chẩn đoán) |
| Bộ đệm gộp `allX`, `allF` | biến cục bộ (`src/tracker_ctrv.py:PFSTrack.multi_predict:312-314`) | tạm thời, không lưu | không lưu lại |
| Mức lớp | `BaseTrack._count` (bộ cấp ID, `third_party/ultralytics_trackers/basetrack.py:BaseTrack._count:57`, `BaseTrack.next_id:74-77`, đặt lại `BaseTrack.reset_id:102`); `PFSTrack.n_multi_predict_calls` (`src/tracker_ctrv.py:PFSTrack.multi_predict:303`) | bộ cấp ID dùng chung cả tiến trình; bộ đếm lời gọi | chung (không phải trạng thái ước lượng; ID ảnh hưởng seed của track nếu hai tracker chạy xen kẽ trong một tiến trình — runner chạy tuần tự từng video) |

Hệ quả [MÃ]: trọng số chuẩn hóa **trong từng track** (`src/pf_ctrv.py:PFCTRV.update_particles:283`) và resample chỉ
chọn trong đám mây của chính track (`src/pf_ctrv.py:PFCTRV.resample_if_needed:246-247`); N không đổi theo track ⇒ không
có cơ chế để track này "lấy" particle của track khác.

### 3.2 Kiểm thực nghiệm [TN]

Lệnh: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/kiem_so_huu_hat.py` → **exit 0**, 1,0 s.
Kết quả: `results/phase2/kc48_pf_code/so_huu_hat.json` (`ket_luan`: "DAT…", `loi`: rỗng).

Cách kiểm: `PFByteTracker` thật (`load_tracker_cfg` + `make_tracker` của `src/run_tracking.py`, cấu hình
`bytetrack_pf_ctrv_n100.yaml`, N = 100, seed 0, backend `ultralytics` 8.4.141, numpy 2.4.6) trên cảnh tổng hợp tất định
45 frame, ảnh 960×540: bus lớn; xe máy chạm mép dưới bus (frame 30–33 detection điểm 0,20 → ghép lần 2); xe D mất
detection frame 15–22; xe rẽ trên cung tròn bán kính 125 px; xe máy ở xa; xe sinh ở frame 10 (track chưa xác nhận); xe E
mất detection frame 20–22. Script bọc **quan sát** ở mức instance quanh `tr.multi_predict`, `pf.sample_initial`,
`pf.reseed_motion`, `pf.update_particles`, `pf.resample_if_needed` (gọi đúng hàm gốc, cùng đối số, trả lại cùng giá trị —
cùng kiểu `_PredictTimer` của `src/run_tracking.py:157`), không sửa mã tracker.

| Phép kiểm | Số lần (`dem_su_kien`) | Kết quả |
|:--|--:|:--|
| Cặp mảng giữa hai track khác nhau không dùng chung bộ nhớ (`np.shares_memory`, `is`) — sau mỗi frame | 4.208 cặp mảng, 1.052 cặp track | 0 vi phạm |
| Như trên ngay sau `multi_predict` (pool tối đa 8 track) | 4.044 cặp mảng | 0 vi phạm |
| Mảng tạo ra / trả về khi update, resample, reseed, khởi tạo không dùng chung bộ nhớ với track khác | 7.288 / 3.616 / 86 / 26 cặp | 0 vi phạm |
| Track **khác** không đổi (particles, logw, trạng thái rng so bit) khi update một track | 1.822 | 0 vi phạm |
| Như trên khi reseed / khởi tạo track mới | 43 / 13 | 0 vi phạm |
| Track ngoài pool (chưa xác nhận) không đổi khi `multi_predict` | 2 | 0 vi phạm |
| Mọi particle sau resample lấy từ đám mây của **chính** track (so từng hàng, roughening = 0) | 285 lần resample, 28.500 particle | 0 vi phạm |
| Sau predict: `track.mean` (→ tlwh → IoU) == `estimate_state(particles, logw)` của chính track; particles là mảng mới; logw không đổi | 319 | 0 vi phạm |
| Track LOST (không có đo): predict giữ nguyên trọng số | 33 lần predict LOST | 0 vi phạm |
| Bất biến mỗi track: `PFTrackView` riêng trỏ về chính track, `PFCTRV` chung, shape (N, 9) / (N,), hữu hạn, Σw = 1 (|LSE| < 10⁻⁹); rng và `PFTrackView` khác nhau giữa mọi cặp track | 329 track-frame | 0 vi phạm |
| Thuộc tính `PFCTRV` bị đổi trong lúc chạy | — | chỉ `n_update`, `n_resample` (bộ đếm chẩn đoán) |
| Mức API, nhánh suy biến: track A, B cùng `PFCTRV`, z cách 5.000 px với A | 1 | nhánh kích hoạt; ghi **tại chỗ** vào mảng của A (`X_ra is X_vao` = True); B không đổi (so bit, kể cả rng); không dùng chung bộ nhớ; trọng số A đều |

Độ phủ (`do_phu`, đều đạt): pool ≥ 2 track; update; resample; predict LOST; update khi track đang LOST (re_activate,
1 lần — xe E); track ngoài pool; khởi tạo hai khung hình (8); ghép lần 2 bằng detection điểm thấp (4 lần); nhánh suy
biến mức API.

**Lần chạy đầu** (cùng script, chưa có xe E): exit 2 "THIẾU ĐỘ PHỦ" — không phép kiểm sở hữu nào hỏng (không có dòng
LOI), nhưng nhánh `re_activate` không được chạy vì xe D (mất 8 frame) bị cấp ID mới (3 → 7). Xe E (mất 3 frame) được
thêm **cuối** danh sách detection để ID — và do đó rng — của 5 xe có từ frame 1 giữ nguyên; xe D không đổi. Lần chạy
thứ hai (file JSON hiện tại): D vẫn đổi ID (3 → 8), E giữ ID 6. Hành vi của D ghi ở mục 4 như **quan sát**, không phải
lỗi sở hữu.

Giới hạn của phép kiểm: một cảnh tổng hợp 45 frame, ≤ 8 track trong pool, N = 100, một seed, backend mặc định; không phủ
`ElapsedStepsMixin` với k > 1, các tùy chọn LostAware / lost_recovery / assoc_log, backend `np`, cảnh dày, roughening > 0,
nhánh suy biến trong pipeline (chỉ kiểm ở mức API). Bọc instance không chứng minh tính đúng của thuật toán — chỉ chứng
minh không có chia sẻ trạng thái giữa track trên các đường đã chạy.

### 3.3 Kết luận sở hữu

Đọc mã và kiểm thực nghiệm cùng cho thấy: **mỗi track PF có mảng particle, mảng log-trọng số và rng riêng**; thứ dùng
chung là một `PFCTRV` mang cấu hình + hàm f / Q / R + bộ đếm chẩn đoán (chia sẻ có chủ đích), cùng bộ cấp ID mức lớp.
Không phát hiện chia sẻ nhầm trạng thái mutable ⇒ **không có bằng chứng lỗi ownership**, không đề xuất sửa cấu trúc
shared filter. Không có lỗi nào cần chuyển cho mốc C từ phần kiểm này.

## 4. Giả thuyết về PF kém EKF / mất ID — tách bạch

[QS] nền: PF–CTRV kém rõ rệt EKF trên UA-DETRAC test ở mốc 2 (`handoff/PLAN_DA_CHOT.md` mục 0; `CODEX_HANDOFF.md` mục 5,
8); ESS/N ≈ 0,25 khi đang bám và ~1 % sau mất dấu dài; tăng N 100 → 500 thu hẹp nhưng không xóa khoảng cách
(`handoff/logs/pf_diag_report.log`; `CODEX_HANDOFF.md` mục 5 ý 7).

| # | Giả thuyết | Cơ chế | Bằng chứng hiện có | Dấu hiệu phân biệt / cách kiểm (để R48C chọn) | Trạng thái |
|:--|:--|:--|:--|:--|:--|
| H1 | Cạnh tranh hạt toàn cục | Một ngân sách hạt / chuẩn hóa trọng số chung cho nhiều đối tượng ⇒ đối tượng likelihood cao (hay được nói là "xe lớn") hút hạt | Mã: trọng số chuẩn hóa và resample trong từng track, N cố định (mục 3.1); TN: 28.500 particle sau resample đều lấy từ đám mây của chính track, 0 chia sẻ bộ nhớ (mục 3.2) | — | **Không áp dụng cho cấu trúc mã hiện tại** (mã + TN). Không dùng giải thích "xe lớn hút hạt" |
| H2 | Nhầm association | Bbox ước lượng (`mean`; hoặc particle trọng số lớn nhất nếu đặt `map`) lệch hoặc rung do nhiễu Monte Carlo ⇒ IoU dưới ngưỡng hoặc đổi chỗ giữa hai xe gần nhau. [LT] Box rộng w dịch d theo một trục: IoU = (w − d)/(w + d) ⇒ cùng d px làm box nhỏ mất IoU nhanh hơn | Chưa đo cho PF | Log quyết định ghép (`src/assoc_log.py`) theo lớp / kích thước; tỉ lệ ghép sai khi có xe lớn kề bên; so sánh EKF / PF trên cùng cache | chưa kiểm |
| H3 | Thiên lệch likelihood | R ∝ h² (σ = h/20; a: 0,1 tuyệt đối) có thể không khớp sai số detector thật (lượng tử pixel ở box nhỏ; cắt khung / che khuất ở box lớn); Gauss đuôi mỏng ⇒ một detection lệch làm trọng số sụp. PF tính R tại h quan sát, EKF tại h dự đoán | Chưa đo | Thống kê innovation chuẩn hóa (NIS) theo lớp / kích thước trên dev; so phân bố với χ²₄ | chưa kiểm |
| H4 | Thiếu detection (mất đo) | Khi LOST, đám mây chỉ loang theo Q, trọng số không đổi; lúc có lại detection, likelihood hẹp so với đám mây ⇒ ESS sụp (`01` mục 7.5); bbox mean có thể lệch ⇒ không ghép lại được | [QS] ESS/N 0,007–0,017 sau 11–30 bước mất (log chẩn đoán dev). [TN] xe D (mục dưới) mất ID sau 8 frame | ESS, số tổ tiên, IoU tại lần ghép lại theo n_lost; tỉ lệ ghép lại thành công PF vs EKF | quan sát một phần; nhân quả chưa kiểm |
| H5 | Nghèo hạt nội bộ + trôi thành phần ẩn | Resample gần như mỗi bước (ESS/N ≈ 0,25) với N nhỏ ⇒ ít tổ tiên; v, θ, ω chỉ quan sát gián tiếp qua vị trí, mà vị trí còn có nhiễu cộng σ = h/20 mỗi bước ⇒ chọn lọc yếu, θ / ω trôi theo các lần chọn lọc; prior ω rộng (0,10 rad/bước) không được reseed ở khởi tạo hai khung | [QS] n_unique/N ≈ 0,38 khi đang bám (log chẩn đoán dev). [QS] `src/test_pf_ctrv.py` test 9 (chạy lại R48A): đường thẳng, N = 100 cho θ = −49,3° và v = 3,58 (thật 30°, 5), RMSE 26,50 px; N = 3000 khớp EKF. [TN] xe D (mục dưới) | Mô phỏng biết liên kết: sai số ước lượng θ, ω, v và sai số dự đoán k bước theo N; số tổ tiên khác nhau sau resample; so với EKF | quan sát ở ca tổng hợp; chưa đo hệ thống |
| H6 | Co trung bình do phân tán hướng | [LT] E[v cos θ] = v̄ cos θ̄ · e^{−σ_θ²/2}, σ_θ là độ lệch chuẩn trạng thái của θ (không phải nhiễu quá trình): khi θ phân tán (mất dấu dài), bbox `mean` của PF dời ngắn hơn f(x̂) của EKF | Chưa đo | So độ dời dự đoán `mean` vs `map` vs EKF theo n_lost trong mô phỏng | chưa kiểm |
| H7 | Lệch mô hình CTRV / phối cảnh (yếu tố gây nhiễu chung) | Chuyển động trên ảnh không phải cung tròn với v, ω hằng; ảnh hưởng mọi bộ lọc | — | Mô phỏng có lệch mô hình (M3 yêu cầu) | chưa kiểm |

**Quan sát trên ca tổng hợp — xe D** [TN] (`so_huu_hat.json`, khóa `nhat_ky_xe_mat_do`; một cảnh, một seed, N = 100;
chỉ mô tả, **chưa so với EKF** — so sánh để sau khi khóa giao thức R48C). Chuyển động thật: v = 3 px/bước, θ = 0, ω = 0,
box 50×40. Sau khởi tạo hai khung (frame 2): θ̂ = −0,07, ω̂ = 0,048. Từ frame 3 đến 14, θ̂ tăng đơn điệu 0,02 → 0,99 rad,
ω̂ (frame 2–14) nằm trong 0,048–0,090 rad/bước, v̂ = 2,28 ở frame 14; trong suốt đoạn này IoU bbox–detection vẫn
0,89–0,95 (vị trí bám tốt dù hướng sai ~56°). Sau 8 frame mất đo, c_x của track đi từ 160,7 (frame 14, sau update) đến
168,4 (frame 23, sau predict) trong khi detection ở 189,0; IoU = 0,174 < 0,222 (ngưỡng hiệu lực với score 0,9) ⇒ track
mới (ID 8). Ca này **phù hợp** với chuỗi H5 → H4 (trôi θ / ω khi đang bám → ngoại suy sai khi mất đo), **không** liên
quan H1 (D không ở cạnh xe nào; sở hữu đã kiểm). Xe E (mất 3 frame) giữ ID dù v̂ = 3,1–3,9 ở frame 14 / 19 / 23 (thật 2).
Không suy rộng từ một ca.

**Giả thuyết PF có lợi** (chi tiết `01` mục 8): chỉ kỳ vọng khi hậu nghiệm lệch Gauss rõ (mất đo dài trong khúc cua, hướng
chưa rõ lúc sinh track) và N đủ lớn; bị giới hạn bởi sai mô hình CTRV, N = 100–500 với đề xuất prior, và việc association
chỉ dùng một bbox. Không có cơ sở nói PF luôn thắng EKF khi cua.

## 5. Khác biệt triển khai đáng ghi (không phải lỗi đã xác nhận)

Docstring `src/pf_ctrv.py:4-11` (và `src/time_units.py:60-63`) nói PF khác EKF **duy nhất** ở cách biểu diễn hậu
nghiệm. Câu đó chỉ đúng ở mức mô hình lý tưởng chung (`01` mục 5). Ở mức triển khai, các ý 1, 3, 4, 5, 7, 8 dưới đây
khác nhau giữa PF và EKF (ý 2 và 6 như nhau cho mọi bộ lọc). Vì vậy so sánh EKF / PF trong pipeline đo gộp cả cách
biểu diễn lẫn các heuristic này.

1. R của PF tính tại h **quan sát** (`src/pf_ctrv.py:PFCTRV.update_particles:262-263`), của EKF tại h **dự đoán**
   (`src/ekf_ctrv.py:EKFTrackerCTRV.project:604`).
2. Confidence không được truyền vào bộ lọc ⇒ nhân R theo (1 − conf) không hoạt động ở mọi bộ lọc
   (`byte_tracker.py:STrack.re_activate:115-117`, `STrack.update:147-149`; mục 1).
3. Q của PF tính tại h của từng particle (`src/pf_ctrv.py:PFCTRV.propagate:209`; `src/tracker_ctrv.py:PFSTrack.multi_predict:314`),
   của EKF tại h trung bình (`src/ekf_ctrv.py:EKFTrackerCTRV.predict:562`); vì Q ∝ h² nên E[Q(h)] ≥ Q(E[h]) — PF có
   nhiễu hiệu dụng hơi lớn hơn khi đám mây phân tán theo h [LT].
4. Khởi tạo hai khung của PF chỉ lấy lại v, θ; ω giữ mẫu từ prior N(0, 0,10²) rồi được chọn lọc qua likelihood vị trí
   (`src/pf_ctrv.py:PFCTRV.reseed_motion:294-306`). EKF giữ phương sai ω = 0,10² nhưng không có bước chọn lọc ngẫu
   nhiên. Liên quan H5.
5. Luật suy biến đo Mahalanobis chỉ theo R, không cộng độ phân tán dự đoán (`src/pf_ctrv.py:PFCTRV.update_particles:264-267`)
   ⇒ rất bảo thủ; tỉ lệ kích hoạt trên dev 0.
6. Track chưa xác nhận không được predict trước khi ghép (`byte_tracker.py:BYTETracker._split_tracked:342-352`,
   `BYTETracker.update:281-283`; thiết kế ByteTrack, như nhau cho mọi bộ lọc).
7. Chặn h ≥ 10⁻³ chỉ có ở PF (`src/pf_ctrv.py:PFCTRV.sample_initial:204`, `PFCTRV.propagate:209, 212`,
   `PFCTRV.resample_if_needed:249`, `PFCTRV.update_particles:274`; `src/tracker_ctrv.py:PFSTrack.multi_predict:314, 320`;
   `src/ekf_ctrv.py` không có `1e-3` hay `maximum`). [LT] `np.maximum` dồn khối xác suất vào biên 10⁻³; đó không phải
   lấy mẫu Gauss có điều kiện trên miền hợp lệ, và trọng số không được hiệu chỉnh. Tần suất chạm biên chưa đo.
8. Tùy chọn `pf_estimate: map` (`src/pf_ctrv.py:PFCTRV.estimate_state:224-225`) chọn particle trọng số lớn nhất, không
   phải MAP của mật độ hậu nghiệm. Khi trọng số đều, `argmax` trả hàng 0, và ma trận trả về là mô-men bậc hai quanh
   particle đó (`01` mục 5, heuristic 6). Tại HEAD không cấu hình PF nào dùng `map`: cả ba yaml dòng 15 đặt `mean`, mặc
   định là `mean` (`src/pf_ctrv.py:PFCTRV.__init__:88`). Chưa kiểm metadata của từng run PF cũ.

Không mục nào ở trên có test tái hiện lỗi; theo plan, mốc C chỉ sửa khi có test hỏng-trước / đạt-sau.

**Kỹ thuật PF ứng viên** (chỉ liệt kê để R48C chọn tối đa hai theo giao thức; **chưa chọn**): đã có trong mã — không phải
đóng góp mới — resample theo ngưỡng ESS, roughening (đang tắt), ước lượng `mean` / particle trọng số lớn nhất (`map`),
luật suy biến, N; chưa có — đề xuất
dùng z_k (ví dụ bước update EKF / UKF làm q, nhân đúng tỉ số importance) lúc ghép lại sau mất đo (nhắm H4), bước MCMC /
regularization riêng cho θ, ω (nhắm H5).

## 6. Lệnh đã chạy trong R48A

| Lệnh | Exit | Ghi chú |
|:--|:--|:--|
| `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/test_pf_ctrv.py` | 0 | 26/26 PASS. **Lệch phạm vi**: test 11 (sẵn có) xóa rồi ghi lại `data/interim/det_cache/fixture_pf_seed_test/` và ghi `data/processed/trackers/DETRAC-sample/p2-pftest-s{0,1}/`, cả hai ngoài `write_paths` của R48A — mục 6.1 |
| `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/kiem_so_huu_hat.py` (lần 1, chưa có xe E) | 2 | thiếu độ phủ re_activate; 0 lỗi sở hữu; JSON bị lần 2 ghi đè |
| `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/kiem_so_huu_hat.py` (lần 2) | 0 | DAT; `results/phase2/kc48_pf_code/so_huu_hat.json` |
| `git rev-parse HEAD` | 0 | `eaf2e877081938420508e696de22dbd2aa0bff97` |

### 6.1 Lệch phạm vi đã xảy ra: test cũ ghi ngoài `write_paths` (R48A-F4)

`write_paths` của R48A gồm `handoff/logs/review48/`, `src/kc48/`, `results/phase2/kc48_pf_code/`,
`handoff/PLAN_DA_CHOT.md`. Lệnh `src/test_pf_ctrv.py` có trong allowlist lệnh của kế hoạch, nhưng allowlist lệnh không
mở quyền ghi ngoài `write_paths` của bước. Test 11 sẵn có (`src/test_pf_ctrv.py:test_11_runner_seeds:250-277`) đã ghi
ra ngoài phạm vi đó:
- `shutil.rmtree` rồi tạo lại `data/interim/det_cache/fixture_pf_seed_test/` (dòng 252-264; `CACHE_ROOT` =
  `data/interim/det_cache`, `src/det_cache.py:84`). Hiện có `meta.json`, `synth.npz`, `synth.json`.
- Gọi `src/run_tracking.py ... --tracker-name p2-pftest --seeds 0 1`, rồi chạy lại riêng seed 0 (dòng 265-274). Lệnh
  này ghi `data/processed/trackers/DETRAC-sample/p2-pftest-s{0,1}/`; mỗi thư mục hiện có `run_meta.json`, `summary.csv`,
  `data/synth.txt`, `data_cls/synth.csv`, `timing/synth.csv`, `video_meta/synth.json` (đường ghi:
  `src/run_tracking.py:271-276, 346-347, 371-418`).

Bằng chứng lệnh đã chạy: log thực thi lời gọi Claude đầu tiên của mốc,
`D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\calls\0001_coder\stdout.log`, dòng 162 (lệnh) và
dòng 165 (kết quả, gồm `[PASS] run_tracking PF --seeds 0 1 chay xong` và dòng `p2-pftest-s0 … synth … 40 frame ->
94 bbox`, khoảng trắng rút gọn). Trạng thái
sau (chỉ đọc trong vòng sửa, không chạy lại): `video_meta/synth.json` ghi `finished_at` 2026-10-10T18:29:43 (s1) và
2026-10-10T18:29:48 (s0, lần chạy lại seed 0), `git_commit` `eaf2e87`.

Không xác minh được: (a) các thư mục này có tồn tại trước lệnh hay không, và nội dung trước đó. Cả ba đường dẫn bị git
bỏ qua (`git status --short --ignored` trả `!!`), và không có hash hay bản chụp nào lấy trước khi chạy. (b) Do đó không
chứng minh được nội dung sau trùng nội dung trước. Riêng `video_meta/synth.json` được xóa rồi ghi lại với dấu thời gian
của lần chạy (`src/run_tracking.py:388-400`), nên nếu đã tồn tại từ trước thì chắc chắn đã khác byte. Các đường này không thuộc vùng dữ liệu gốc mà kế hoạch cấm ghi (`data/raw`,
`data/extracted`, `data/yolo_*`, `data/interim/hash_tap_danh_gia`, `models/`) và không phải kết quả thí nghiệm, nhưng
vẫn là ghi ngoài `write_paths`. Theo review: **không** xóa, khôi phục hay chạy lại các fixture này trong R48A.
R48C phải chạy lại `src/test_pf_ctrv.py`, trong khi `write_paths` của R48C cũng không có hai đường này; cần người dùng
quyết định trước khi chạy (ghi ở mục 0 của `handoff/PLAN_DA_CHOT.md`, "Điểm chưa rõ").

## 7. Đính chính sau review Codex vòng 1

Review: `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_001.json` (`revise`). Chỉ sửa
tài liệu; mã, cấu hình, `src/kc48/kiem_so_huu_hat.py` và `results/phase2/kc48_pf_code/so_huu_hat.json` giữ nguyên;
không chạy lại test.
- F1 (đơn vị σ_θ [rad], σ_ω [rad/bước] của gia số mỗi predict): sửa trong `01` mục 0.3, 2, 3 (bảng đơn vị từng thành
  phần Q), 7.7, 8. Trong file này không có chỗ nào ghi đơn vị nhiễu quá trình; các đơn vị rad/bước còn lại (ω̂ của xe D
  ở mục 4, prior ω 0,10 rad/bước ở H5 và mục 5 ý 4) là đơn vị của **trạng thái** ω, đúng. H6 ghi rõ σ_θ là độ lệch
  chuẩn trạng thái.
- F2: mục 5 thêm đoạn đầu (mô hình lý tưởng khác triển khai) và ý 7 (phép chặn h); cột Proposal ở mục 2.
- F3: mục 2 (dòng ước lượng), mục 4 (H2), mục 5 ý 8, danh sách kỹ thuật ứng viên.
- F4: mục 6 và mục 6.1.
