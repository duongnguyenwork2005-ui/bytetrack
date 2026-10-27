# R50A / P487415eb — Thiết kế phân tích: vì sao DetA của PF thấp hơn EKF trên dev12 (CHỈ THIẾT KẾ)

Ngày 2026-10-12; sửa hồ sơ F1–F6 ngày 13/10. Người dùng duyệt P487415eb **chỉ ở mức thiết kế** (quyết định 12/10/2026). Tài liệu này **không** có
kết quả phân tích nào. Lượt này không tính chỉ số, không thêm instrumentation, không chạy tracking hay eval. Phần thực thi
chờ người dùng duyệt riêng, theo từng pha ở mục 7.

Đầu vào đã kiểm kê bằng `src/kc48/kiem_ke_dau_vao_deta.py`, ra `results/phase2/kc48_pf_dev/thiet_ke_deta/kiem_ke.json`.
Script chỉ liệt kê file, kích thước, cột và số dòng; không đọc giá trị để thống kê. Mọi số đếm file / dòng dưới đây lấy
từ file đó. Phiên Codex sửa tài liệu chỉ đọc kiểm kê đã có, không chạy lại script.

## 0. Hiện tượng cần giải thích và phạm vi

- Trên dev12 (điều kiện A, `conf_min` 0,25), DetA COMBINED của 8 biến thể PF nằm trong 0,469–0,493; `ekf` 0,521
  (`07_tong_hop_bao_cao_17_10.md` mục 6.3, `results/phase2/kc48_pf_dev/phan_tich/bang_chinh.csv`). Phần lớn khoảng cách
  HOTA nằm ở AssA / IDSW. Câu hỏi ở đây **chỉ** về phần DetA.
- Phạm vi: 12 video dev12, 26 run đã có (`results/phase2/kc48_pf_dev/runs.csv`). Không dùng tập test hay BDD val, không
  chọn tham số.
- Chỉ dùng run đã có. Chỗ nào dữ liệu cũ không đủ thì ghi rõ giới hạn (mục 6) và tách ra pha chạy lại / instrumentation
  (mục 7). Pha đó phải được duyệt riêng.

### 0.1 Cơ chế đầu ra của ByteTrack: DetA bị ảnh hưởng qua kênh nào

Phần này kế thừa đối chiếu mã của bản nháp / handover, chưa kiểm bằng dữ liệu. Pha B / C chỉ được làm sau duyệt.
Các quan hệ dưới đây là cơ chế dự kiến, không phải phân rã nhân quả của DetA.

1. Tracker chỉ xuất track **đã kích hoạt và được ghép ở frame hiện tại**: `_format_output` trả về
   `[x.result for x in self.tracked_stracks if x.is_activated]` (ultralytics 8.4.141, `byte_tracker.py:494`; phiên bản
   ghi ở `run_meta.json` → `code.backend.ultralytics` của các run kc48). Track không ghép được ở frame f sẽ chuyển sang
   lost và không có box đầu ra ở f. Mỗi box đầu ra ứng với đúng một detection (gán 1–1). Hệ quả dự kiến:
   số box đầu ra ≤ số detection vào tracker ở mọi frame. Pha A kiểm bất biến này (chỉ số M2).
2. Box đầu ra là **trạng thái sau cập nhật** (posterior mean với PF `pf_estimate: mean`; mean sau update với EKF). Đây
   không phải box detection. Vì vậy độ chính xác định vị phụ thuộc bộ lọc.
3. Detection không ghép được với track nào sẽ thành track mới **chưa xác nhận** nếu score ≥ `new_track_thresh` 0,25.
   Track này không được xuất ở frame sinh ra (trừ frame 1 của video). Nó chỉ được xuất từ lần ghép kế tiếp. Vì vậy mỗi lần
   sinh track mới có thể mất frame đầu ra. Chỉ thành FN nếu không còn box hợp lệ khác phủ GT theo evaluator;
   ID mới trong MOT không đủ chứng minh sinh lại track hay độ trễ xác nhận.
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
| HD4 | Vòng đời track (sinh, xác nhận, lost, xóa) | Có thể thiếu đầu ra lúc chờ xác nhận; phải kiểm bằng sự kiện log và GT, không đồng nhất đổi ID với sinh lại |
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
| K2 | Cùng video và đủ frame | 12 video của `videos.json` có mặt ở mọi run; `video_meta/<video>.json` có `complete: true`; `n_frames` trùng nhau giữa run | dừng, báo; không tự giảm dev12 hay đổi tập so sánh |
| K3 | Cùng điều kiện | `conf_min` 0,25, `classes`, `max_det` 300, `fps` 25, `elapsed_mode: skip`, khóa ByteTrack (`track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `match_thresh`, `fuse_score`, `track_buffer`) trùng nhau; chỉ được khác khóa `pf_*` / bộ lọc | dừng, báo |
| K4 | Seed | PF: seed 0, 1, 2 có đủ cho mọi biến thể; EKF tất định (1 run). Tổng hợp PF: tính chỉ số **riêng từng seed rồi trung bình** (như `src/kc48/boot_seed.py`, F3 của R48D); báo kèm min–max giữa seed | dừng, báo |
| K5 | Cùng GT, cùng evaluator | GT lấy từ cùng `_processed/DETRAC-all`; số GT_dets theo video trùng nhau giữa run (`per_video_metrics.csv`) | dừng, báo |
| K6 | Mã tại lúc chạy | kc48: `git_commit` eaf2e87 + K1 nằm trong cây làm việc chưa commit (khóa `code` của `run_meta`). Mọi so sánh trong pha A chỉ giữa các run kc48 (cùng đợt R48D) | — |
| A0 | Chỉ kiểm đầu ra MOT mốc 2 và kc48 | So từng file `data/<video>.txt` của `p2-expA_detrac_dev-c25-{ekf,pf100,pf300,pf500}-s*` với run kc48 tương ứng, theo **cùng một quy ước hash tính lại trên chính file** (`05b` mục 6; không so `eval_meta.json` khác đời). Kiểm kê cho thấy tổng số dòng trùng nhau ở cả 10 cặp run (ví dụ `ekf` 91.500 / 91.500, `pf100-s0` 88.574 / 88.574). Đây chỉ là gợi ý, **chưa phải bằng chứng** | trùng hay không trùng đều không chứng minh tương đương nội bộ; ghi kết quả kiểm đầu ra, log chỉ mô tả chính run mốc 2 khi thiếu bằng chứng thực thi liên quan; pha mới chờ duyệt |

Ghi chú provenance: `a64cb75` có trước `935a445` (2026-09-20), commit đổi quy ước hash `.txt` (CRLF → LF). Khi
`src/audit_run_provenance.py` chưa hỗ trợ quy ước cũ, cảnh báo "không khớp" của nó với run mốc 2 **không** được dùng để
kết luận run cũ không hợp lệ. A0 phải ghi rõ hash byte thô hay chuẩn hóa CRLF → LF; chỉ hash byte thô trùng
mới gọi là trùng byte. Đây là kiểm dự kiến, chưa thực hiện.

A0 chỉ kiểm đầu ra. Khi đọc log phải kiểm riêng nguồn, cấu hình hiệu lực, backend / mixin, schema,
khóa `(video, frame, track_id)` và `det_idx` của đúng mảng detection sau lọc. Các kiểm nguồn này cũng **chưa đủ**
chứng minh tương đương thực thi. MOT trùng không chứng minh prior, association, vòng đời hay trọng số giống nhau.
Chưa có bằng chứng tương đương phần thực thi liên quan thì M6e/M7/M8b chỉ mô tả chính run mốc 2,
không giải thích trực tiếp kc48 hay K1/K2; phần thiếu để pha C/D chờ duyệt.
`assoc_log.n_lost` ghi sau predict, trước association (`frame_id - end_frame`, `src/assoc_log.py:78,120–124`).
Với frame tracker tăng một bước và track ghép ở frame liền trước, giá trị là **1**, không phải 0.
M7 chọn `state=Tracked`, `is_activated` và đầu ra cùng ID ở frame liền trước, không lọc `assoc_log.n_lost=0`.
Phải đối chiếu `frame` nguồn với `frame_id` tracker khi có bước nhảy, không thay bằng điều kiện số 1 cho mọi trường hợp.
`pf_diag.n_lost` ở lúc update dùng `max(0, fid - t.frame_id)` với fid lấy từ `_pending_frame_id`
(`src/pf_ctrv.py:390–395`, `PFTrackView.update`). Update thường đã đặt `t.frame_id` thành frame hiện tại trước
khi gọi bộ lọc, nên bằng 0 khi đang bám; thời điểm khác log trước association, không đồng nhất hai cột.
Bước kiểm K1–K6 trong bảng khác tùy chọn proposal K1.

## 2. (b) Cách phân biệt các giả thuyết

Nguyên tắc: mỗi giả thuyết có một **phép đo trực tiếp** ở đúng nơi nó tác động. Mức đo dùng được là prior trước ghép,
quyết định ghép, sự kiện vòng đời, posterior sau ghép, hoặc sai số detector độc lập với tracker. Không suy ngược từ HOTA
hay từ file MOT.

| Câu hỏi | Phép đo phân biệt | Dữ liệu |
|:--|:--|:--|
| Khoảng cách DetA là do **thiếu / thừa box** hay do **định vị**? | Đường ΔDetA(α) trên 19 ngưỡng (M1). Ở α = 0,05 vẫn có cổng IoU và gán HOTA. Đường hiệu chỉ gợi ý độ nhạy theo ngưỡng; dùng cùng M2–M4, không tách chính xác đếm / định vị | `raw_fields.json` (có sẵn) |
| Box có bị **bỏ** khi detector có thấy không? | M2 (box ra / detection vào), M3 (FN có detection) | file MOT + cache + GT (có sẵn) |
| Bỏ do **prior lệch** (HD1) hay do **gán** (HD2)? | M7a: IoU(prior, detection cùng GT) trên track đang bám. M7b: trong các mẫu prior **qua** cổng, tỉ lệ gán đúng / gán khác / không gán detection của GT đó | `assoc_log` (box dự đoán, `matched_det`, `stage`). Chỉ có ở mốc 2 cho ekf / pf100 / pf300 / pf500 |
| **Tầng 2** có vai trò không? | M7c: số detection `tier = low` và số ghép `stage = 2` | `assoc_log dets` / `tracks` |
| **Vòng đời** (HD4)? | M6: ID quan sát được, đoạn đầu ra và khoảng trống; M6e: sự kiện thật từ log | file MOT (ID / độ dài đầu ra, không ghi sinh / xác nhận / xóa); `assoc_log events` (lost, refind, new, removed) |
| **Định vị posterior** (HD6)? | M4: IoU(box ra, GT) so với IoU(detection, GT) trên **cùng cặp (GT, frame)** mà cả PF và EKF đều có box ghép. M5: độ rung sai số | file MOT + cache + GT (có sẵn) |
| **Likelihood không khớp** (HD5)? | M8a: phần dư chuẩn hóa của detection so với GT theo mô hình R (không cần tracker). M8b: phân bố `maha2_best`, ESS sau cập nhật khi đang bám (n_lost = 0). M8c: NIS EKF và predictive log-likelihood EKF/PF trên update chung | M8a: cache + GT (có sẵn). M8b: `pf_diag` mốc 2 (base PF). M8c: **chưa có**, cần instrumentation |

Đọc kết hợp, viết trước khi tính:
- **HD1 có dấu hiệu phù hợp** nếu: (i) khoảng hụt DetA hiện rõ ở α thấp (M1); (ii) M3 cho thấy PF có nhiều FN-có-detection hơn EKF trên
  cùng (GT, frame); (iii) M7a cho thấy IoU(prior, detection đúng) của PF thấp hơn và tỉ lệ prior dưới cổng cao hơn. HD1
  chưa phân biệt được nếu CI của (iii) chứa 0; thiếu ý nghĩa thống kê không chứng minh tương đương.
- **HD2** có dấu hiệu phù hợp nếu M7b ghi gán khác / không gán dù prior qua cổng. CI M7a chứa 0 không loại HD1;
  thiếu các đối thủ / ma trận cùng frame thì chưa tách được nguyên nhân cạnh tranh.
- **HD3** không có kênh tác động quan sát được trên đúng run có log nếu low=0 và stage2=0; không tự mở rộng sang run thiếu log.
- **HD4** chỉ được gợi ý bởi phân mảnh M6; cần M6e nối với GT và M7 để phân biệt mất / sinh / xác nhận.
- **HD6** phù hợp nếu M4 âm với CI loại 0 trên cặp chung; M1 chỉ hỗ trợ mô tả. Cặp chung loại ca thất bại,
  nên không ngoại suy M4 thành toàn bộ khoảng cách DetA.
- **HD5** chỉ có dấu hiệu phù hợp nếu M8a có phần dư lớn cùng M8b có ESS thấp / khoảng cách tới hạt tốt nhất lớn.
  Còn các khả năng prior, số hạt, sai ghép và nhiễu GT. PF lấy h_detection cho R, EKF lấy h_prior;
  confidence hiệu lực phải đối chiếu lời gọi. M8c cần predictive likelihood tương thích, không đồng nhất maha2_best với NIS EKF.

## 3. (c) Chỉ số: định nghĩa, mẫu số, tổng hợp theo video, bằng chứng

Quy ước chung:
- **Ghép GT ↔ box (tracker hoặc detection) trong một frame**: `occlusion_eval.match_frame` (loại cặp IoU < 0,5 **trước**
  Hungarian), như F1 của R48D. Nếu chỉ số cần ngưỡng khác thì ghi rõ.
- **Theo video**: tỉ số được tính bằng tổng tử / tổng mẫu **trong video**, không lấy trung bình các tỉ số theo frame.
- **Toàn dev12**: tỉ số gộp = tổng tử / tổng mẫu; chỉ M1 gọi COMBINED theo evaluator. Trung bình liên tục gộp theo
  số quan sát; trung vị báo từng video và trung vị của các giá trị video (macro), không gọi là trung vị gộp.
  Báo số quan sát / video hợp lệ; mẫu số 0 ghi N/A, không gán 0; thiếu video thì dừng báo, không tự đổi dev12.
  Hiệu ghép cặp PF − EKF theo video, CI bootstrap
  lấy lại video có hoàn lại, B = 5000, rng seed 0 (`04` mục 5.2). PF: tính riêng từng seed rồi trung bình trong mỗi mẫu
  bootstrap (`src/kc48/boot_seed.py`).
- **Biến thể chính**: `ekf` so với `pf100`, `pf100-k1`, `pf500-k1`. Các biến thể khác chỉ mô tả. Chỉ số cần `assoc_log`
  / `pf_diag` chỉ có cho base mốc 2 tương ứng; chưa có bằng chứng tương đương thực thi thì chỉ mô tả mốc 2, dù A0 trùng.
- Câu “không khác / bác bỏ” phải tuân mục 5: CI chứa 0 là chưa phân biệt được. Chỉ chống lại hướng dự đoán khi CI
  hoàn toàn về hướng đối lập; chưa có biên tương đương nên không kết luận tương đương.
- Lớp GT `others` và vùng bỏ qua: xử lý như evaluator hiện hành. Ảnh không có `<frame>` là giới hạn chung, đã ghi ở mục 0
  của PLAN.

| Mã | Định nghĩa | Mẫu số | Tổng hợp theo video | Ủng hộ / bác bỏ | Dữ liệu |
|:--|:--|:--|:--|:--|:--|
| M1 DetRe(α), DetPr(α), DetA(α) | Với mỗi α trong 19 ngưỡng: DetRe = TP/(TP+FN), DetPr = TP/(TP+FP), DetA = TP/(TP+FN+FP) từ `HOTA_TP/FN/FP` | DetRe: số GT det (TP+FN); DetPr: số box tracker (TP+FP); DetA: TP+FN+FP — đều ở ngưỡng α đó | Tổng TP/FN/FP của video theo α → tỉ số; COMBINED cộng trường thô 12 video | Δ=PF−EKF, khoảng hụt D=EKF−PF. D tăng theo α phù hợp độ nhạy định vị; ở α thấp vẫn có sai gán / định vị. DetRe/DetPr mô tả FN/FP, không phân rã nhân quả. Trung bình 19 tỉ số sau khi gộp TP/FN/FP theo α | `raw_fields.json` — có sẵn |
| M2 Tỉ lệ detection có box đầu ra | ρ = Σ_f n_out(f) / Σ_f n_det(f); n_det = `timing.n_dets` (detection sau lọc, `run_tracking.py:250-251`), n_out = số dòng MOT ở frame f. Kèm kiểm bất biến n_out(f) ≤ n_det(f) | Σ n_det của video | Tỉ số tổng trong video | PF có ρ thấp hơn → nhiều detection không được nối vào track đã xác nhận (HD1/HD2/HD4, chưa tách được). Không cho biết tầng nào; không suy ra association từ con số này. Nếu bất biến hỏng thì mục 0.1-1 sai, phải xem lại | `timing/*.csv` + file MOT — có sẵn |
| M3 FN có detection | Cặp (GT g, frame f) có detection ghép với g (IoU ≥ 0,5) nhưng không có box tracker ghép với g. Tách theo vị trí: (a) trong 1 frame ngay trước lần xuất đầu tiên của một ID tracker mới phủ g (proxy khoảng trống trước ID quan sát mới, chưa phải độ trễ xác nhận); (b) còn lại | Số cặp (g, f) có detection ghép | Tổng FN-có-detection / tổng cặp có detection trong video, kèm số đếm (a)/(b) | PF > EKF trên cùng cặp → mất box khi detector có thấy (kênh đếm). Phần (a) lớn chỉ gợi ý HD4; cần log xác nhận. Không phân biệt HD1 / HD2 nếu không có M7 | file MOT + cache + GT — có sẵn |
| M4 Định vị trên cặp chung | Trên các cặp (g, f) mà **cả** EKF và PF (cùng seed so ghép cặp) đều có box ghép với g: ΔIoU = IoU(box PF, g) − IoU(box EKF, g). Kèm IoU(det, g) của detection ghép g làm mốc | Số cặp chung | Trung bình ΔIoU trong video (không trọng số theo track) | ΔIoU < 0, CI loại 0 → ủng hộ HD6 trên cặp chung. CI chứa 0 hoặc hiệu gần 0 → chưa phân biệt được; không bác bỏ lỗi định vị ở các cặp bị loại. IoU(box, g) < IoU(det, g) chỉ mô tả box ra kém mốc detection trên cặp đó; MOT không chứng minh detection này đã cập nhật track | file MOT + cache + GT — có sẵn |
| M5 Độ rung sai số (mục 4) | J = ‖e_t − e_{t−1}‖ với e_t = sai số chuẩn hóa của box so với GT | Số cặp frame liên tiếp hợp lệ | Trung vị + trung bình trong video | J_PF > J_EKF trên cùng cặp → posterior PF dao động (HD5/HD6). CI chứa 0 → chưa phân biệt được; CI hoàn toàn âm chống lại hướng "PF rung hơn" trên cặp chung | file MOT + cache + GT — có sẵn |
| M6 Chỉ báo ID / đoạn từ đầu ra | Với mỗi GT g: số ID tracker khác nhau ghép với g (≥ 1 frame), số lần ID mới bắt đầu phủ g, độ dài đoạn ID. Với tracker: số ID có tổng độ dài ≤ 3 frame | Số GT có ≥1 ghép cho số ID / GT; số đoạn ID liên tiếp cho độ dài đoạn; số ID đầu ra cho M6b | Trung bình số ID trên GT; trung bình / trung vị độ dài trên đoạn; ID ngắn / tổng ID trong video | Nhiều ID / đoạn ngắn gợi ý phân mảnh; cũng có thể do đổi ID hoặc ghép GT khác, không chứng minh sinh lại / xóa / xác nhận. M6a dùng GT có ≥1 ghép, báo riêng GT không ghép; M6b dùng mọi ID đầu ra. Không suy tầng ghép | file MOT + GT — có sẵn |
| M6e Sự kiện vòng đời từ log | Số sự kiện `lost`, `refind`, `new`, `removed`, `removed_unconfirmed` / 1.000 frame; tỉ lệ `new` mà GT ở đó đã có track bám trước đó ≤ 30 frame | Số frame tracker xử lý từ timing (kể cả không có event); số new ghép được GT | Từng event: 1.000 × số event / số frame; new trùng GT: tổng đủ điều kiện / số new ghép GT, báo số new không ghép | Hỗ trợ HD4 trên đúng run có log | `assoc_log events` — mốc 2, 4 biến thể |
| M7a Chất lượng prior | Track đang bám (state Tracked, is_activated, có đầu ra frame liền trước; không lọc assoc_log.n_lost=0) ở f, đã ghép GT g ở f−1: IoU(box dự đoán của track tại f, g_f), sai số tâm / h_g. Tỉ lệ prior dưới cổng: IoU(prior, det đúng)·score < 1 − `match_thresh` = 0,2 (chi phí fused > 0,8), với det đúng = detection ghép g_f | Số (track, f) đang bám có g_f và có det ghép g_f | Trung bình / tỉ lệ trong video | IoU PF thấp hơn, hoặc sai số tâm / tỉ lệ dưới cổng cao hơn EKF với CI loại 0 → phù hợp HD1. CI chứa 0 → chưa phân biệt được HD1 | `assoc_log tracks` (box dự đoán) + cache + GT — mốc 2 |
| M7b Gán khi prior qua cổng | Trong các (track, f) ở M7a mà prior qua cổng: tỉ lệ track được gán đúng det của g_f / gán det khác / không gán | Số (track, f) prior qua cổng | Tỉ lệ trong video | Tỉ lệ "gán khác / không gán" của PF cao hơn khi prior vẫn qua cổng → HD2 | `assoc_log tracks/dets` (`matched_det`, `stage`) — mốc 2 |
| M7c Tầng 2 | n_low từ dets; n_stage2 từ tracks, đếm mỗi ghép một lần | Tỉ lệ low: mọi detection có tier; hiệu suất tầng 2: n_low; số đếm tuyệt đối không cần mẫu số | Tổng mỗi loại theo video, kèm hai tỉ lệ; n_low=0 thì hiệu suất N/A | low=stage2=0 loại kênh tầng 2 trên đúng run có log; có stage2 thì cơ chế mục 0.1-4 cần xem lại | `assoc_log` — mốc 2 |
| M8a Phần dư detector so với mô hình R | Cặp (det, g) ghép IoU ≥ 0,5: r_k = (z_det,k − z_g,k)/σ_R,k với z = (cx, cy, a, h), σ_R theo `_measurement_noise(h, conf)` của PF (`src/pf_ctrv.py:303-305`); D_R = Σ_k r_k². Báo trung vị D_R, tỉ lệ >13,28; χ²₄ (trung vị xấp xỉ 3,36) chỉ là tham chiếu lý tưởng, không kiểm định hiệu chuẩn; dùng tầng kích thước đã khóa | Số cặp (det, g) | Trung vị / tỉ lệ trong video | Phần dư lớn chỉ phù hợp HD5. Ghép IoU≥0,5 gây chọn mẫu; nhiễu GT / tương quan có thể tăng hoặc giảm phần dư, không có cận chắc chắn. R theo h_det / confidence hiệu lực; EKF dùng h_prior nên R có thể khác. Không phải innovation NIS | cache + GT — có sẵn, không cần tracker |
| M8b Trọng số PF khi bám | Trong pf_diag, U: kind=update, n_lost=0; V: U và degenerate=false. Trên V đo `maha2_best` (= −2·max log-likelihood, đã bỏ hằng số, `src/pf_ctrv.py:251-255, 306-307`), `ess_ratio`; degenerate đếm riêng trên U | ESS/maha: số V; degenerate: số U | Trung vị ESS/maha trên V; tổng degenerate / số U theo video và N; V rỗng ghi N/A | maha2_best là cực tiểu qua N hạt, không có luật χ²₄. ESS thấp / maha lớn có thể do prior, N hoặc sai ghép; tăng N không tách riêng nguyên nhân R. Cùng hiện tượng với 05c nhưng đo khi đang bám, không sau mất dấu | `pf_diag` — mốc 2, base PF (không có K1) |
| M8c Predictive likelihood / NIS | Cùng detection và GT: EKF r=z−Hμ⁻, S=HP⁻Hᵀ+R, NIS=rᵀS⁻¹r, log density=log N(z;Hμ⁻,S); PF log predictive density = logsumexp(log w⁻ᵢ + log N(z;Hx⁻ᵢ,R)), gồm hằng chuẩn hóa. Không dùng hạt posterior hoặc maha2_best; K1 phải dùng phân bố trước quan sát | Số update chung đủ prior, R, confidence; báo số loại / tái khởi tạo riêng | Trung vị NIS EKF; trung bình âm log density từng bộ lọc và hiệu ghép cặp theo video | Độ khớp predictive còn phụ thuộc prior, không quy nguyên nhân riêng R; không đồng nhất PF maha với NIS EKF | **Không có** — cần đặc tả / instrumentation được duyệt ở Pha D |

## 4. (d) Độ rung box: tách khỏi chuyển động thật

Không coi mọi thay đổi bbox giữa hai frame là lỗi. Xe chuyển động, đổi tỉ lệ khi lại gần, đi vào vùng cua.

- Với cặp (box, g) ghép ở hai frame liên tiếp f−1, f, cùng ID tracker và cùng GT g: lấy e_f = ((cx − cx_g)/h_g,
  (cy − cy_g)/h_g, (w − w_g)/h_g, (h − h_g)/h_g). Độ rung là J_f = ‖e_f − e_{f−1}‖. Lấy sai phân của **sai số** nên chuyển
  động thật mà GT ghi lại (tịnh tiến, phóng to) bị trừ đi. Phần dư vẫn chịu ảnh hưởng nhiễu GT và chuẩn hóa theo h_g, không phải rung thuần túy.
- Mốc so trên cùng (g, f−1, f) PF/EKF/detection: J_det theo cùng công thức. Báo hiệu J_out−J_det theo video;
  nếu báo tỉ số trung bình J_out / trung bình J_det thì mẫu số 0 ghi N/A. Tỉ số <1 chỉ là sai số thay đổi ít hơn,
  có thể vẫn có bias / trễ; đọc cùng M4, không kết luận tốt hơn chỉ vì mượt.
- Tham chiếu độ cong GT: chuẩn sai phân bậc hai của (cx,cy,w,h) chia h_g,f; mẫu là bộ ba frame GT liên tiếp,
  tổng hợp trung vị / trung bình theo video. Chứa cả gia tốc thật và nhiễu nhãn, **không phải sàn nhiễu**.
  GT có thể nội suy / làm mượt; trừ chuyển động theo GT không loại hoàn toàn sai số nhãn.
- Độ nhạy dự kiến: CV khớp trên cửa sổ ±5 frame đủ 11 frame GT liên tiếp; thiếu cửa sổ ghi loại. Dùng lại định nghĩa J,
  mẫu số cặp frame và tổng hợp M5. Đoạn cua theo `loai_max` đã có (`03` mục 8) báo riêng vì CV kém ở cua;
  không xem sai số này là rung thuần túy. Chỉ thực hiện khi pha phân tích được duyệt.
- Loại khỏi mẫu: frame có đổi ID, frame đầu của đoạn ID quan sát mới, frame GT bị che ≥ 0,5 (theo `occlusion_eval`).
- Diễn giải: J_PF > J_EKF (CI loại 0) trên cùng (g, f) ủng hộ HD5/HD6. Không kết luận nguyên nhân nếu J_det cũng khác giữa
  các tập cặp (tập cặp phải là chung).

## 5. Quy tắc diễn giải (viết trước khi tính)

- Mỗi giả thuyết chỉ được ghi "ủng hộ" khi phép đo trực tiếp của nó ở mục 2 đạt, với CI bootstrap theo video loại 0 trên
  biến thể chính. CI chứa hoặc chạm 0, hay hiệu gần 0, thì ghi "chưa phân biệt được", không ghi "bác bỏ".
  Chưa có biên và phép kiểm tương đương được khai báo nên không kết luận tương đương.
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
2. A0 chỉ kiểm đầu ra; kiểm nguồn / ngữ nghĩa log riêng cũng chưa chứng minh tương đương thực thi.
   Trùng byte MOT không chứng minh nội bộ trùng; thiếu bằng chứng thì chỉ mô tả mốc 2, không chuyển kết luận sang kc48.
3. Không có log so predictive likelihood EKF/PF trên cùng update (M8c). M8a + M8b chỉ gợi ý,
   không là kiểm định hiệu chuẩn hay điều kiện đủ để quy nguyên nhân.
4. GT DETRAC có sai số / có thể nội suy; ảnh hưởng lên M5/M8a không có chiều hay cận chắc chắn.
5. Ảnh không có `<frame>` (572 ảnh dev12) và vùng bỏ qua: xử lý của evaluator chưa kiểm. Mọi biến thể chấm trên cùng GT,
   nhưng mức ảnh hưởng lên hiệu số chưa biết.

## 7. (f) Chi phí và phân pha

Ước tính dựa trên kích thước đã kiểm kê: 26 run × 12 video, khoảng 88–92 nghìn dòng MOT mỗi run, 116.845 dòng GT;
thời gian track R48D là 26 run trong khoảng 10 phút với 5 tiến trình (`07` mục 6.2). Đây là kịch bản theo giả định, không
phải cận trên.

| Pha | Nội dung | Dữ liệu | Ghi gì | Chi phí ước tính | Cần duyệt |
|:--|:--|:--|:--|:--|:--|
| A — chỉ dữ liệu cũ kc48 | K1–K6; M1, M2 (+ bất biến), M3, M4, M5, M6, M8a cho cả 26 run | file MOT, `timing`, `raw_fields`, GT, cache | thư mục mới `results/phase2/kc48_pf_dev/phan_tich_deta/`; script chỉ đọc mới trong `src/kc48/` | Viết + test: 3–4 h. Chạy: < 20 phút CPU (ghép Hungarian từng frame, 26 run) | có (thực thi phân tích) |
| B — log mốc 2 có sẵn | A0: chỉ so đầu ra 10 cặp run × 12 video. Kiểm riêng nguồn / ngữ nghĩa log rồi đo M6e, M7a–c, M8b cho chính run mốc 2 `ekf`, `pf100`, `pf300`, `pf500`; không chuyển diễn giải nội bộ sang kc48 khi chưa có bằng chứng tương đương thực thi | `assoc_log` parquet, `pf_diag` của `data/processed/trackers/DETRAC-all/p2-expA_detrac_dev-c25-*` (chỉ đọc) | như A | Viết + test: 2–3 h. Chạy: 10–20 phút (đọc parquet 12 run × 12 video). Lưu ý: fastparquet có thể bị Smart App Control chặn (đọc metadata ở bước kiểm kê đã chạy được) | có |
| C — chạy lại có log (không đổi thuật toán) | Chạy lại `pf100-k1`, `pf500-k1` (3 seed mỗi biến thể), và `ekf` / base nếu thiếu bằng chứng nội bộ cần thiết, với `--assoc-log --pf-diag` (phải kiểm lại bảo toàn cho backend / cấu hình đó; test cũ không bảo đảm run mới); kiểm đầu ra trùng byte với kc48 cũ; M6e, M7, M8b mô tả chính run mới, chỉ liên hệ nội bộ run cũ khi có bằng chứng thực thi tương đương | cache dev | **thư mục run mới**, không ghi đè `p2-kc48_pf_dev-*` | Chạy: riêng 6 run K1 ước tính 5–15 phút với 5 tiến trình (chưa đo) + kiểm byte. Nếu phải thêm ekf/base thì dự toán này không bao phủ, phải trình lại trước thực thi. Phân tích dùng lại script của B | có (tracking mới) |
| D — instrumentation mới | Hook chỉ đọc ghi NIS EKF và predictive log-likelihood của EKF/PF với prior / R / confidence / detection ID (M8c), không gọi maha2_best là NIS tương đương; test đầu ra bit-identical khi bật / tắt; chạy lại `ekf`, `pf100`, `pf100-k1` | cache dev | mã mới + thư mục run mới | Viết + test: 3–5 h. Chạy: 10–20 phút | có (mã mới + tracking mới) |

Thứ tự trình duyệt: A → B (A0 trước) → người dùng quyết định C / D. A/B cũng **chưa được duyệt thực thi**;
không tự chạy C/D khi A0 hỏng. Các đường ghi ở bảng là dự kiến, không mở rộng ba write_paths hiện tại.
Ước tính chưa đo: A+B viết/test 5–7 h, chạy CPU khoảng 30–40 phút cộng A0; C cần kiểm bảo toàn;
D viết/test 3–5 h cộng thời gian chạy. Chưa có dự toán dung lượng log đáng tin cậy, cần đo sau duyệt;
không hứa cận trên RAM/đĩa. Không cài gói nếu parquet bị chặn, báo thiếu đầu vào. Không mở ngân sách hay dùng GPU.

Tự kiểm phiên tiếp nối 13/10: đã đọc nguyên văn F1–F6 trong `reports/review_001.json` của run Codex cũ,
đối chiếu mã / kiểm kê có sẵn và bản đầu mốc. Bảng sửa và bằng chứng ở `10` mục 5. Không tính M1–M8,
không chạy K/A0, test, kiểm kê, tracking hay eval. R50A vẫn chờ cổng reviewer Codex riêng, chưa approve.

## 8. Truy nguồn

- Hiện tượng: `07_tong_hop_bao_cao_17_10.md` mục 6.3–6.4; `results/phase2/kc48_pf_dev/phan_tich/bang_chinh.csv`.
- Kiểm kê: `src/kc48/kiem_ke_dau_vao_deta.py` →
  `results/phase2/kc48_pf_dev/thiet_ke_deta/kiem_ke.json` (lệnh và kết quả: `10_test11_k1_va_trang_thai_de_xuat.md` mục 2).
- Mã đọc để thiết kế: `src/run_tracking.py:226-282` (ghi MOT 7 cột, `timing.n_dets`); `src/det_cache.py:412-422` (lọc
  `conf > conf_min`); `src/assoc_log.py:1-38` (cột log); `src/pf_ctrv.py:251-255, 293-307` (log-likelihood,
  `maha2_best`); ultralytics 8.4.141 `trackers/byte_tracker.py:313-318, 494`.
- Chẩn đoán ESS mốc 2 (sau mất dấu): `05c_kiem_pf_diag_moc2.md`. Tái tạo EKF trùng byte (2/12 video): `05b_truy_nguyen_lech_ekf.md`.
