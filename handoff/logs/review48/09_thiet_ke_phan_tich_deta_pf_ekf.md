# R50A / P487415eb — Thiết kế phân tích: vì sao DetA của PF thấp hơn EKF trên dev12 (CHỈ THIẾT KẾ)

Ngày 2026-10-12. Người dùng duyệt P487415eb **chỉ ở mức thiết kế** (quyết định 12/10/2026). Tài liệu này **không** có
kết quả phân tích nào. Lượt này không tính chỉ số, không thêm instrumentation, không chạy tracking hay eval. Phần thực thi
chờ người dùng duyệt riêng, theo từng pha ở mục 7.

Đầu vào đã kiểm kê bằng `src/kc48/kiem_ke_dau_vao_deta.py`, ra `results/phase2/kc48_pf_dev/thiet_ke_deta/kiem_ke.json`.
Script chỉ liệt kê file, kích thước, cột và số dòng; không đọc giá trị để thống kê. Mọi số đếm file / dòng dưới đây lấy
từ file đó.

## 0. Hiện tượng cần giải thích và phạm vi

- Trên dev12 (điều kiện A, `conf_min` 0,25), DetA COMBINED của 8 biến thể PF nằm trong 0,469–0,493; `ekf` 0,521
  (`07_tong_hop_bao_cao_17_10.md` mục 6.3, `results/phase2/kc48_pf_dev/phan_tich/bang_chinh.csv`). Phần lớn khoảng cách
  HOTA nằm ở AssA / IDSW. Câu hỏi ở đây **chỉ** về phần DetA.
- Phạm vi: 12 video dev12, 26 run đã có (`results/phase2/kc48_pf_dev/runs.csv`). Không dùng tập test hay BDD val, không
  chọn tham số.
- Chỉ dùng run đã có. Chỗ nào dữ liệu cũ không đủ thì ghi rõ giới hạn (mục 6) và tách ra pha chạy lại / instrumentation
  (mục 7). Pha đó phải được duyệt riêng.

### 0.1 Cơ chế đầu ra của ByteTrack: DetA bị ảnh hưởng qua kênh nào

Phần này đọc từ mã; chưa kiểm bằng dữ liệu. Pha B / C kiểm lại.

1. Tracker chỉ xuất track **đã kích hoạt và được ghép ở frame hiện tại**: `_format_output` trả về
   `[x.result for x in self.tracked_stracks if x.is_activated]` (ultralytics 8.4.141, `byte_tracker.py:494`; phiên bản
   ghi ở `run_meta.json` → `code.backend.ultralytics` của các run kc48). Track không ghép được ở frame f sẽ chuyển sang
   lost và không có box đầu ra ở f. Mỗi box đầu ra ứng với đúng một detection (gán 1–1). Hệ quả dự kiến:
   số box đầu ra ≤ số detection vào tracker ở mọi frame. Pha A kiểm bất biến này (chỉ số M2).
2. Box đầu ra là **trạng thái sau cập nhật** (posterior mean với PF `pf_estimate: mean`; mean sau update với EKF). Đây
   không phải box detection. Vì vậy độ chính xác định vị phụ thuộc bộ lọc.
3. Detection không ghép được với track nào sẽ thành track mới **chưa xác nhận** nếu score ≥ `new_track_thresh` 0,25.
   Track này không được xuất ở frame sinh ra (trừ frame 1 của video). Nó chỉ được xuất từ lần ghép kế tiếp. Vì vậy mỗi lần
   phải sinh lại track mất ít nhất một frame đầu ra. Đó là DetFN dù detector có thấy xe.
4. **Tầng 2 rỗng trong điều kiện A, theo cấu trúc**: cache được lọc `conf > conf_min = 0,25` (`src/det_cache.py:417-422`).
   Detection thấp điểm của ByteTrack là `0,1 < score < track_high_thresh = 0,25` (`byte_tracker.py:316-317`). Vì vậy tập
   detection của tầng 2 rỗng ở mọi frame. Đây là suy luận từ mã. Pha B kiểm bằng `assoc_log` (cột `tier` của bảng `dets`):
   nếu dự đoán đúng, số dòng `tier = low` phải bằng 0.

Như vậy DetA của PF có thể thấp hơn EKF qua bốn kênh. Các kênh không loại trừ nhau:

| Mã | Giả thuyết | Kênh tác động lên DetA |
|:--|:--|:--|
| HD1 | Dự đoán box (prior) của PF lệch hơn EKF | IoU(prior, detection đúng) thấp → trượt cổng `match_thresh` → track lost, detection sinh track mới chưa xác nhận → DetFN (kênh 0.1-3) |
| HD2 | Ghép tầng 1 / tầng 3 khác nhau (chi phí `1 − IoU·score`, `fuse_score`) | Cùng kênh với HD1, nhưng nguyên nhân nằm ở ma trận chi phí / gán, không ở chất lượng prior |
| HD3 | Ghép tầng 2 (detection thấp điểm) | Theo mục 0.1-4, **không áp dụng cho điều kiện A** nếu Pha B xác nhận tầng 2 rỗng |
| HD4 | Vòng đời track (sinh, xác nhận, lost, xóa) | Sinh lại track nhiều hơn → mỗi lần mất ≥ 1 frame đầu ra; track ngắn chưa xác nhận bị xóa |
| HD5 | Likelihood PF không khớp sai số detector thật (R quá hẹp / sai dạng) | Trọng số suy biến → posterior lệch → box đầu ra kém chính xác (kênh 0.1-2); prior kém ở frame sau (quay về HD1) |
| HD6 | Định vị posterior (box sau update) của PF kém hơn, dù ghép đúng | Cùng số TP ở α thấp, nhưng mất TP ở α cao (DetA trung bình 19 ngưỡng α) |

HD1, HD5 và HD6 liên quan với nhau. Thiết kế dưới đây tách chúng bằng **điều kiện đo** (prior so với posterior; có hoặc
không có ghép), không bằng suy diễn từ HOTA.

## 1. (a) File đầu vào, cấu hình, kiểm tính so sánh được

### 1.1 File đầu vào (đã kiểm kê, `kiem_ke.json`)

| Nhóm | Đường dẫn | Có gì (theo kiểm kê) |
|:--|:--|:--|
| Danh sách run | `results/phase2/kc48_pf_dev/runs.csv` | 26 dòng: `cond, tracker, yaml, set, seed, conf_min, dir, tracker_type, track_buffer_sec` |
| Cấu hình thí nghiệm | `results/phase2/kc48_pf_dev/experiment.yaml`, `configs/phase2/experiments/kc48_pf_dev.yaml` | cache `yolov8n-coco_conf0.05_all`, lớp [2, 3, 5, 7], `max_det` 300, fps 25, `elapsed_mode: skip`, điều kiện A `conf_min` 0,25 |
| Cấu hình tracker | 9 YAML trong `runs.csv` (`configs/bytetrack_ekf_ctrv.yaml`, `configs/phase2/trackers/bytetrack_pf_ctrv_n{100,300,500}.yaml`, `configs/phase2/trackers/kc48/*.yaml`) | đều tồn tại (`kiem_ke.json` → `yaml_tracker`) |
| Đầu ra tracker kc48 | `results/phase2/kc48_pf_dev/_processed/trackers/DETRAC-all/<dir>/` | `data/<video>.txt` (MOT **7 cột**: frame, id, x, y, w, h, score), `data_cls/<video>.csv` (frame, id, cls), `timing/<video>.csv` (frame, n_steps, n_tracks, n_dets, t_*), `video_meta/`, `run_meta.json`. **Không có** `assoc_log/`, `pf_diag/` (`run_meta`: `assoc_log: false`, `pf_diag: false`) |
| TrackEval kc48 | `results/trackeval/DETRAC-all/<dir>/` | `per_video_metrics.csv` (HOTA, DetA, AssA, MOTA, MOTP, FP, FN, IDSW, GT_dets, IDF1, IDP, IDR), `raw_fields.json` (mỗi video: `HOTA_TP/FN/FP` mảng 19 phần tử, AssA, AssRe, AssPr, ID*, CLR_*, IDSW), `combined_metrics.csv`, `eval_meta.json` |
| GT dev12 | `results/phase2/kc48_pf_dev/_processed/DETRAC-all/<video>/gt/gt.txt`, `seqinfo.ini` | 12 video, tổng 116.845 dòng GT |
| Detection thô | `data/interim/det_cache/yolov8n-coco_conf0.05_all/<video>.npz` (+ `.json`, `meta.json`) | mảng `frame_idx, n_per_frame, unreadable, truncated, time_s, data (n, 6), orig_shape`; chưa lọc conf / lớp. Cache dev trùng mốc 2 (`results/phase2/kc48_pf_code/cache_dev.json`) |
| Log ghép + chẩn đoán PF **mốc 2** | `data/processed/trackers/DETRAC-all/p2-expA_detrac_dev-c25-{ekf,cv,ukf,pf100-s0..2,pf300-s0..2,pf500-s0..2}/` | `assoc_log/<video>_{tracks,dets,events}.parquet` đủ 12/12 video ở cả 12 run; `pf_diag/<video>.csv` 12/12 ở 9 run PF. Cùng cache, `conf_min` 0,25, lớp, `max_det`, fps, `elapsed_mode`; commit `a64cb75` (2026-09-16) |
| Phân tích đã có | `results/phase2/kc48_pf_dev/phan_tich/*`, `su_kien/*` | bảng chính, bootstrap, phân tầng, sự kiện (đã nghiệm thu ở R48D) |

Biến thể không có log mốc 2: `pf100-k1`, `pf100-k2`, `pf100-k1k2`, `pf300-k1`, `pf500-k1`, `ekf-chinh`.

### 1.2 Kiểm tính so sánh được (bước K, chạy đầu tiên khi thực thi, trước mọi chỉ số)

| Mã | Kiểm | Cách làm | Hỏng thì |
|:--|:--|:--|:--|
| K1 | Cùng cache | `run_meta.cache_id` của mọi run = `yolov8n-coco_conf0.05_all`; `cache_dev.json` đã xác nhận cache dev trùng mốc 2 | dừng, báo |
| K2 | Cùng video và đủ frame | 12 video của `videos.json` có mặt ở mọi run; `video_meta/<video>.json` có `complete: true`; `n_frames` trùng nhau giữa run | loại video khỏi mọi run (ghép cặp), ghi lý do |
| K3 | Cùng điều kiện | `conf_min` 0,25, `classes`, `max_det` 300, `fps` 25, `elapsed_mode: skip`, khóa ByteTrack (`track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `match_thresh`, `fuse_score`, `track_buffer`) trùng nhau; chỉ được khác khóa `pf_*` / bộ lọc | dừng, báo |
| K4 | Seed | PF: seed 0, 1, 2 có đủ cho mọi biến thể; EKF tất định (1 run). Tổng hợp PF: tính chỉ số **riêng từng seed rồi trung bình** (như `src/kc48/boot_seed.py`, F3 của R48D); báo kèm min–max giữa seed | dừng, báo |
| K5 | Cùng GT, cùng evaluator | GT lấy từ cùng `_processed/DETRAC-all`; số GT_dets theo video trùng nhau giữa run (`per_video_metrics.csv`) | dừng, báo |
| K6 | Mã tại lúc chạy | kc48: `git_commit` eaf2e87 + K1 nằm trong cây làm việc chưa commit (khóa `code` của `run_meta`). Mọi so sánh trong pha A chỉ giữa các run kc48 (cùng đợt R48D) | — |
| A0 | Log mốc 2 có mô tả đúng run kc48 không | So từng file `data/<video>.txt` của `p2-expA_detrac_dev-c25-{ekf,pf100,pf300,pf500}-s*` với run kc48 tương ứng, theo **cùng một quy ước hash tính lại trên chính file** (`05b` mục 6; không so `eval_meta.json` khác đời). Kiểm kê cho thấy tổng số dòng trùng nhau ở cả 10 cặp run (ví dụ `ekf` 91.500 / 91.500, `pf100-s0` 88.574 / 88.574). Đây chỉ là gợi ý, **chưa phải bằng chứng** | cặp nào không trùng byte thì log mốc 2 của cặp đó không dùng được; chuyển biến thể đó sang Pha C |

Ghi chú provenance: `a64cb75` có trước `935a445` (2026-09-20), commit đổi quy ước hash `.txt` (CRLF → LF). Khi
`src/audit_run_provenance.py` chưa hỗ trợ quy ước cũ, cảnh báo "không khớp" của nó với run mốc 2 **không** được dùng để
kết luận run cũ không hợp lệ. A0 phải tính lại hash trên chính file theo cùng quy ước.

## 2. (b) Cách phân biệt các giả thuyết

Nguyên tắc: mỗi giả thuyết có một **phép đo trực tiếp** ở đúng nơi nó tác động. Mức đo dùng được là prior trước ghép,
quyết định ghép, sự kiện vòng đời, posterior sau ghép, hoặc sai số detector độc lập với tracker. Không suy ngược từ HOTA
hay từ file MOT.

| Câu hỏi | Phép đo phân biệt | Dữ liệu |
|:--|:--|:--|
| Khoảng cách DetA là do **thiếu / thừa box** hay do **định vị**? | Đường ΔDetA(α) trên 19 ngưỡng (M1). Ở α = 0,05, IoU gần như không chặn: hiệu ở đó là kênh đếm (HD1/HD2/HD4). Phần hiệu tăng thêm khi α tăng là kênh định vị (HD6/HD5) | `raw_fields.json` (có sẵn) |
| Box có bị **bỏ** khi detector có thấy không? | M2 (box ra / detection vào), M3 (FN có detection) | file MOT + cache + GT (có sẵn) |
| Bỏ do **prior lệch** (HD1) hay do **gán** (HD2)? | M7a: IoU(prior, detection cùng GT) trên track đang bám. M7b: trong các lần trượt, tỉ lệ prior **qua** cổng nhưng vẫn không được gán đúng detection (tranh chấp / gán chéo) | `assoc_log` (box dự đoán, `matched_det`, `stage`). Chỉ có ở mốc 2 cho ekf / pf100 / pf300 / pf500 |
| **Tầng 2** có vai trò không? | M7c: số detection `tier = low` và số ghép `stage = 2` | `assoc_log dets` / `tracks` |
| **Vòng đời** (HD4)? | M6: số lần sinh ID mới trên mỗi track GT, frame mất do độ trễ xác nhận, track ngắn bị xóa | file MOT (sinh / độ dài ID); `assoc_log events` (lost, refind, new, removed) |
| **Định vị posterior** (HD6)? | M4: IoU(box ra, GT) so với IoU(detection, GT) trên **cùng cặp (GT, frame)** mà cả PF và EKF đều có box ghép. M5: độ rung sai số | file MOT + cache + GT (có sẵn) |
| **Likelihood không khớp** (HD5)? | M8a: phần dư chuẩn hóa của detection so với GT theo mô hình R (không cần tracker). M8b: phân bố `maha2_best`, ESS sau cập nhật khi đang bám (n_lost = 0). M8c: NIS / log-likelihood cả EKF lẫn PF ở mọi update | M8a: cache + GT (có sẵn). M8b: `pf_diag` mốc 2 (base PF). M8c: **chưa có**, cần instrumentation |

Đọc kết hợp, viết trước khi tính:
- **HD1 được ủng hộ** nếu: (i) ΔDetA tập trung ở α thấp (M1); (ii) M3 cho thấy PF có nhiều FN-có-detection hơn EKF trên
  cùng (GT, frame); (iii) M7a cho thấy IoU(prior, detection đúng) của PF thấp hơn và tỉ lệ prior dưới cổng cao hơn. HD1
  bị bác bỏ nếu (iii) không khác EKF (CI chứa 0) trong khi (ii) khác.
- **HD2** được ủng hộ nếu (ii) khác, (iii) không khác, và M7b (prior qua cổng nhưng gán sai / không gán) chiếm phần lớn
  các lần trượt.
- **HD3** bị loại cho điều kiện A nếu M7c = 0 ở mọi run có log.
- **HD4** được ủng hộ (như kênh truyền, không như nguyên nhân gốc) nếu số lần sinh lại ID / GT track của PF cao hơn. Muốn
  phân biệt nguyên nhân gốc thì phải đi kèm M7.
- **HD6** được ủng hộ nếu M4 (ΔIoU PF − EKF trên cùng cặp) âm với CI loại 0, và ΔDetA tăng theo α (M1).
- **HD5** được ủng hộ nếu M8a cho thấy phần dư detector vượt xa mô hình R (định nghĩa ở M8a), **và** M8b cho thấy
  `maha2_best` khi đang bám thường lớn / ESS thấp. Chỉ M8a thôi thì chưa đủ: EKF dùng cùng dạng R mà không suy biến.
  Muốn so trực tiếp EKF / PF trên cùng update thì cần M8c.

## 3. (c) Chỉ số: định nghĩa, mẫu số, tổng hợp theo video, bằng chứng

Quy ước chung:
- **Ghép GT ↔ box (tracker hoặc detection) trong một frame**: `occlusion_eval.match_frame` (loại cặp IoU < 0,5 **trước**
  Hungarian), như F1 của R48D. Nếu chỉ số cần ngưỡng khác thì ghi rõ.
- **Theo video**: tỉ số được tính bằng tổng tử / tổng mẫu **trong video**, không lấy trung bình các tỉ số theo frame.
- **Toàn dev12**: (i) COMBINED = tổng tử / tổng mẫu trên 12 video; (ii) hiệu ghép cặp PF − EKF theo video, CI bootstrap
  lấy lại video có hoàn lại, B = 5000, rng seed 0 (`04` mục 5.2). PF: tính riêng từng seed rồi trung bình trong mỗi mẫu
  bootstrap (`src/kc48/boot_seed.py`).
- **Biến thể chính**: `ekf` so với `pf100`, `pf100-k1`, `pf500-k1`. Các biến thể khác chỉ mô tả. Chỉ số cần `assoc_log`
  / `pf_diag` chỉ làm được cho `ekf`, `pf100`, `pf300`, `pf500` (nếu A0 đạt).
- Lớp GT `others` và vùng bỏ qua: xử lý như evaluator hiện hành. Ảnh không có `<frame>` là giới hạn chung, đã ghi ở mục 0
  của PLAN.

| Mã | Định nghĩa | Mẫu số | Tổng hợp theo video | Ủng hộ / bác bỏ | Dữ liệu |
|:--|:--|:--|:--|:--|:--|
| M1 DetRe(α), DetPr(α), DetA(α) | Với mỗi α trong 19 ngưỡng: DetRe = TP/(TP+FN), DetPr = TP/(TP+FP), DetA = TP/(TP+FN+FP) từ `HOTA_TP/FN/FP` | DetRe: số GT det (TP+FN); DetPr: số box tracker (TP+FP); DetA: TP+FN+FP — đều ở ngưỡng α đó | Tổng TP/FN/FP của video theo α → tỉ số; COMBINED cộng trường thô 12 video | ΔDetA(0,05) ≈ ΔDetA trung bình → kênh đếm (HD1/HD2/HD4). ΔDetA tăng rõ theo α, hiệu ở α thấp nhỏ → định vị (HD6/HD5). Tách ΔDetRe / ΔDetPr: thiếu box (FN) hay box thừa (FP). Mô tả, không nhân quả | `raw_fields.json` — có sẵn |
| M2 Tỉ lệ detection có box đầu ra | ρ = Σ_f n_out(f) / Σ_f n_det(f); n_det = `timing.n_dets` (detection sau lọc, `run_tracking.py:250-251`), n_out = số dòng MOT ở frame f. Kèm kiểm bất biến n_out(f) ≤ n_det(f) | Σ n_det của video | Tỉ số tổng trong video | PF có ρ thấp hơn → nhiều detection không được nối vào track đã xác nhận (HD1/HD2/HD4, chưa tách được). Không cho biết tầng nào; không suy ra association từ con số này. Nếu bất biến hỏng thì mục 0.1-1 sai, phải xem lại | `timing/*.csv` + file MOT — có sẵn |
| M3 FN có detection | Cặp (GT g, frame f) có detection ghép với g (IoU ≥ 0,5) nhưng không có box tracker ghép với g. Tách theo vị trí: (a) trong 1 frame ngay trước lần xuất đầu tiên của một ID tracker mới phủ g (độ trễ xác nhận); (b) còn lại | Số cặp (g, f) có detection ghép | Tổng trong video | PF > EKF trên cùng cặp → mất box khi detector có thấy (kênh đếm). Phần (a) lớn → HD4 là kênh truyền chính. Không phân biệt HD1 / HD2 nếu không có M7 | file MOT + cache + GT — có sẵn |
| M4 Định vị trên cặp chung | Trên các cặp (g, f) mà **cả** EKF và PF (cùng seed so ghép cặp) đều có box ghép với g: ΔIoU = IoU(box PF, g) − IoU(box EKF, g). Kèm IoU(det, g) của detection ghép g làm mốc | Số cặp chung | Trung bình ΔIoU trong video (không trọng số theo track) | ΔIoU < 0, CI loại 0 → ủng hộ HD6. ΔIoU ≈ 0 → bác bỏ HD6 là nguồn chính của ΔDetA. IoU(box, g) < IoU(det, g) cho cả hai → bộ lọc làm định vị kém hơn detection (chung, không riêng PF) | file MOT + cache + GT — có sẵn |
| M5 Độ rung sai số (mục 4) | J = ‖e_t − e_{t−1}‖ với e_t = sai số chuẩn hóa của box so với GT | Số cặp frame liên tiếp hợp lệ | Trung vị + trung bình trong video | J_PF > J_EKF trên cùng cặp → posterior PF dao động (HD5/HD6). J ≈ nhau → bác bỏ "PF rung hơn" | file MOT + cache + GT — có sẵn |
| M6 Vòng đời từ đầu ra | Với mỗi GT g: số ID tracker khác nhau ghép với g (≥ 1 frame), số lần ID mới bắt đầu phủ g, độ dài đoạn ID. Với tracker: số ID có tổng độ dài ≤ 3 frame | Số track GT (M6a); số ID tracker (M6b) | Trung bình trên track GT trong video; tỉ lệ trong video | PF sinh lại nhiều hơn → HD4 như kênh truyền (mỗi lần mất ≥ 1 frame, mục 0.1-3). Không suy ra tầng ghép | file MOT + GT — có sẵn |
| M6e Sự kiện vòng đời từ log | Số sự kiện `lost`, `refind`, `new`, `removed`, `removed_unconfirmed` / 1.000 frame; tỉ lệ `new` mà GT ở đó đã có track bám trước đó ≤ 30 frame | Số frame; số `new` | Tỉ lệ trong video | Như M6, nhưng đếm trực tiếp từ log | `assoc_log events` — mốc 2, 4 biến thể |
| M7a Chất lượng prior | Track đang bám (state Tracked, n_lost = 0) ở f, đã ghép GT g ở f−1: IoU(box dự đoán của track tại f, g_f), sai số tâm / h_g. Tỉ lệ prior dưới cổng: IoU(prior, det đúng)·score < 1 − `match_thresh` = 0,2 (chi phí fused > 0,8), với det đúng = detection ghép g_f | Số (track, f) đang bám có g_f và có det ghép g_f | Trung bình / tỉ lệ trong video | PF thấp hơn EKF, CI loại 0 → ủng hộ HD1. Không khác → bác bỏ HD1 | `assoc_log tracks` (box dự đoán) + cache + GT — mốc 2 |
| M7b Gán khi prior qua cổng | Trong các (track, f) ở M7a mà prior qua cổng: tỉ lệ track được gán đúng det của g_f / gán det khác / không gán | Số (track, f) prior qua cổng | Tỉ lệ trong video | Tỉ lệ "gán khác / không gán" của PF cao hơn khi prior vẫn qua cổng → HD2 | `assoc_log tracks/dets` (`matched_det`, `stage`) — mốc 2 |
| M7c Tầng 2 | Số detection `tier = low`; số ghép `stage = 2` | Số detection | Tổng trong video | = 0 ở mọi run → HD3 không áp dụng cho điều kiện A (xác nhận mục 0.1-4). > 0 → mục 0.1-4 sai; phải đo vai trò tầng 2 | `assoc_log` — mốc 2 |
| M8a Phần dư detector so với mô hình R | Cặp (det, g) ghép IoU ≥ 0,5: r_k = (z_det,k − z_g,k)/σ_R,k với z = (cx, cy, a, h), σ_R theo `_measurement_noise(h, conf)` của PF (`src/pf_ctrv.py:303-305`); NIS_det = Σ_k r_k². Báo trung vị NIS_det (so với 3,36 = trung vị χ²₄), tỉ lệ > 13,28 (χ²₄ mức 0,99), theo tầng kích thước | Số cặp (det, g) | Trung vị / tỉ lệ trong video | Trung vị ≫ 3,36 và tỉ lệ > 13,28 ≫ 1 % → R hẹp hơn sai số detector thật (điều kiện cần của HD5). Gần χ²₄ → bác bỏ "R quá hẹp". GT DETRAC cũng có sai số nên NIS_det là cận trên của phần dư detector. Trước khi tính phải đối chiếu R của EKF (chưa kiểm ở lượt này) | cache + GT — có sẵn, không cần tracker |
| M8b Trọng số PF khi bám | Update có `kind` cập nhật thường, n_lost = 0: phân bố `maha2_best` (= −2·max log-likelihood, đã bỏ hằng số, `src/pf_ctrv.py:251-255, 306-307`), `ess_ratio`, tỉ lệ `degenerate` | Số update đủ điều kiện | Trung vị / tỉ lệ trong video, theo N | `maha2_best` khi bám thường ≫ 4 và ESS thấp → likelihood quá nhọn so với độ phân tán hạt (ủng hộ HD5). Nếu tăng N làm giảm rõ thì là thiếu hạt hơn là sai R. Cùng hiện tượng với 05c nhưng đo khi đang bám, không sau mất dấu | `pf_diag` — mốc 2, base PF (không có K1) |
| M8c NIS / log-lik cả hai bộ lọc | NIS của EKF (innovation qua S) và log-likelihood / NIS tương đương của PF trên cùng update | Số update | Trung vị / tỉ lệ trong video | So trực tiếp mức khớp R của hai bộ lọc trên dữ liệu thật | **Không có** — cần instrumentation (Pha D) |

## 4. (d) Độ rung box: tách khỏi chuyển động thật

Không coi mọi thay đổi bbox giữa hai frame là lỗi. Xe chuyển động, đổi tỉ lệ khi lại gần, đi vào vùng cua.

- Với cặp (box, g) ghép ở hai frame liên tiếp f−1, f, cùng ID tracker và cùng GT g: lấy e_f = ((cx − cx_g)/h_g,
  (cy − cy_g)/h_g, (w − w_g)/h_g, (h − h_g)/h_g). Độ rung là J_f = ‖e_f − e_{f−1}‖. Lấy sai phân của **sai số** nên chuyển
  động thật mà GT ghi lại (tịnh tiến, phóng to) bị trừ đi. Chỉ phần box dao động quanh GT còn lại.
- Mốc so sánh trên cùng (g, f−1, f): J_det (cùng công thức cho detection ghép g ở hai frame). J_out / J_det < 1 nghĩa là
  bộ lọc làm mượt; > 1 nghĩa là bộ lọc thêm dao động.
- Sàn nhiễu nhãn: sai phân bậc hai của chính quỹ đạo GT, (b_g,f − 2b_g,f−1 + b_g,f−2)/h_g, báo kèm. GT DETRAC có thể được
  nội suy / làm mượt. Nếu J_out gần sàn này thì không phân biệt được.
- Độ nhạy: thay GT bằng quỹ đạo mô hình CV khớp bình phương tối thiểu trên cửa sổ ±5 frame của GT. Đoạn cua (`loai_max`,
  `03` mục 8) báo riêng vì CV kém ở đó.
- Loại khỏi mẫu: frame có đổi ID, frame đầu sau khi sinh lại, frame GT bị che ≥ 0,5 (theo `occlusion_eval`).
- Diễn giải: J_PF > J_EKF (CI loại 0) trên cùng (g, f) ủng hộ HD5/HD6. Không kết luận nguyên nhân nếu J_det cũng khác giữa
  các tập cặp (tập cặp phải là chung).

## 5. Quy tắc diễn giải (viết trước khi tính)

- Mỗi giả thuyết chỉ được ghi "ủng hộ" khi phép đo trực tiếp của nó ở mục 2 đạt, với CI bootstrap theo video loại 0 trên
  biến thể chính. CI chứa 0 thì ghi "không phân biệt được", không ghi "bác bỏ", trừ khi đã khai báo trước cỡ hiệu tối
  thiểu (chưa khai báo nên không dùng).
- Không suy ra tầng ghép, quyết định gán hay likelihood từ file MOT (7 cột, không có `idx` detection). Không nối box đầu ra
  với detection bằng cách khớp `score`. Kênh đó không được dùng làm bằng chứng association.
- Chẩn đoán trên `pf100`/`pf300`/`pf500` (có log) **không** tự áp sang K1/K2. Kết luận cho K1 cần Pha C.
- 12 video, video là đơn vị lấy mẫu lại, chưa chắc độc lập. Mọi CI là theo video dev12, không suy rộng.
- Số nào sinh ra từ chẩn đoán chỉ để giải thích. Không dùng để chọn tham số hay đổi cấu hình đã khóa.

## 6. (e) Dữ liệu hiện có đủ đến đâu

Theo `kiem_ke.json` → `truong_can`:

| Trường cần | kc48 (26 run) | mốc 2 c25 (12 run) | Hệ quả |
|:--|:--|:--|:--|
| Box đầu ra từng frame | có (MOT 7 cột) | có | M1–M6 làm được |
| `idx` detection đã cập nhật track | **không** (`run_tracking.py` bỏ cột idx) | — (có `matched_det` trong `assoc_log`) | Không biết box ra ứng với detection nào nếu không có `assoc_log` |
| `n_dets` mỗi frame | có (`timing`) | có | M2 |
| `HOTA_TP/FN/FP` × 19 α | có | có | M1 |
| Tổng độ tương đồng TP (LocA) | **không** | không | LocA chính thức cần chạy lại TrackEval (không tracking). M4 thay thế bằng IoU tự tính |
| Box dự đoán (prior) | **không** | có (`tracks.x, y, w, h`) | M7a chỉ cho 4 biến thể base |
| Tầng ghép, `matched_det` | **không** | có | M7b, M7c chỉ cho base |
| `tier`, `reject_reason` của detection | **không** | có | M7c |
| Sự kiện vòng đời | **không** | có | M6e chỉ cho base; M6 (từ MOT) cho mọi biến thể |
| ESS / resample / suy biến / `maha2_best` | **không** | có (9 run PF) | M8b chỉ cho base PF |
| Log-likelihood / NIS mọi update, EKF và PF | **không** | **không** | M8c cần instrumentation |
| GT từng frame | có | (dùng chung) | |
| Detection thô | có | (dùng chung) | M3, M4, M5, M8a |

Giới hạn chính:
1. Với K1/K2 và `ekf-chinh`, **không có** thông tin association từng tầng, prior hay trọng số. Phần tách HD1 / HD2 / HD5 cho
   K1 không làm được từ dữ liệu cũ. Không được thay bằng suy luận từ đầu ra MOT.
2. Log mốc 2 chỉ dùng được nếu A0 xác nhận trùng byte với run kc48 tương ứng. Số dòng trùng chưa đủ.
3. Không có log nào cho phép so mức khớp likelihood của EKF và PF trên cùng update (M8c). M8a + M8b chỉ cho điều kiện cần.
4. GT DETRAC có sai số và có thể được nội suy. M5 và M8a bị chặn dưới bởi nhiễu nhãn.
5. Ảnh không có `<frame>` (572 ảnh dev12) và vùng bỏ qua: xử lý của evaluator chưa kiểm. Mọi biến thể chấm trên cùng GT,
   nhưng mức ảnh hưởng lên hiệu số chưa biết.

## 7. (f) Chi phí và phân pha

Ước tính dựa trên kích thước đã kiểm kê: 26 run × 12 video, khoảng 88–92 nghìn dòng MOT mỗi run, 116.845 dòng GT;
thời gian track R48D là 26 run trong khoảng 10 phút với 5 tiến trình (`07` mục 6.2). Đây là kịch bản theo giả định, không
phải cận trên.

| Pha | Nội dung | Dữ liệu | Ghi gì | Chi phí ước tính | Cần duyệt |
|:--|:--|:--|:--|:--|:--|
| A — chỉ dữ liệu cũ kc48 | K1–K6; M1, M2 (+ bất biến), M3, M4, M5, M6, M8a cho cả 26 run | file MOT, `timing`, `raw_fields`, GT, cache | thư mục mới `results/phase2/kc48_pf_dev/phan_tich_deta/`; script chỉ đọc mới trong `src/kc48/` | Viết + test: 3–4 h. Chạy: < 20 phút CPU (ghép Hungarian từng frame, 26 run) | có (thực thi phân tích) |
| B — log mốc 2 có sẵn | A0 (so byte 10 cặp run × 12 video, vài phút); nếu đạt: M6e, M7a–c, M8b cho `ekf`, `pf100`, `pf300`, `pf500` | `assoc_log` parquet, `pf_diag` của `data/processed/trackers/DETRAC-all/p2-expA_detrac_dev-c25-*` (chỉ đọc) | như A | Viết + test: 2–3 h. Chạy: 10–20 phút (đọc parquet 12 run × 12 video). Lưu ý: fastparquet có thể bị Smart App Control chặn (đọc metadata ở bước kiểm kê đã chạy được) | có |
| C — chạy lại có log (không đổi thuật toán) | Chạy lại `pf100-k1`, `pf500-k1` (3 seed mỗi biến thể), và `ekf` / base nếu A0 hỏng, với `--assoc-log --pf-diag` (log không đổi đầu ra, theo `test_assoc_log.py`); kiểm đầu ra trùng byte với run kc48 cũ; rồi chạy M6e, M7, M8b cho K1 | cache dev | **thư mục run mới**, không ghi đè `p2-kc48_pf_dev-*` | Chạy: khoảng 6 run, ước tính 5–15 phút với 5 tiến trình (log tăng thời gian, chưa đo) + kiểm byte. Phân tích dùng lại script của B | có (tracking mới) |
| D — instrumentation mới | Hook chỉ đọc (mixin bọc ngoài, như `assoc_log`) ghi NIS của EKF và log-likelihood / NIS tương đương của PF ở mọi update (M8c); test đầu ra bit-identical khi bật / tắt; chạy lại `ekf`, `pf100`, `pf100-k1` | cache dev | mã mới + thư mục run mới | Viết + test: 3–5 h. Chạy: 10–20 phút | có (mã mới + tracking mới) |

Thứ tự đề xuất: A → B (A0 trước) → quyết định có cần C / D hay không, dựa trên A và B. A và B không chạy tracking, không
sửa mã có sẵn, không ghi vào thư mục run cũ.

## 8. Truy nguồn

- Hiện tượng: `07_tong_hop_bao_cao_17_10.md` mục 6.3–6.4; `results/phase2/kc48_pf_dev/phan_tich/bang_chinh.csv`.
- Kiểm kê: `src/kc48/kiem_ke_dau_vao_deta.py` →
  `results/phase2/kc48_pf_dev/thiet_ke_deta/kiem_ke.json` (lệnh và kết quả: `10_test11_k1_va_trang_thai_de_xuat.md` mục 2).
- Mã đọc để thiết kế: `src/run_tracking.py:226-282` (ghi MOT 7 cột, `timing.n_dets`); `src/det_cache.py:412-422` (lọc
  `conf > conf_min`); `src/assoc_log.py:1-38` (cột log); `src/pf_ctrv.py:251-255, 293-307` (log-likelihood,
  `maha2_best`); ultralytics 8.4.141 `trackers/byte_tracker.py:313-318, 494`.
- Chẩn đoán ESS mốc 2 (sau mất dấu): `05c_kiem_pf_diag_moc2.md`. Tái tạo EKF trùng byte (2/12 video): `05b_truy_nguyen_lech_ekf.md`.
