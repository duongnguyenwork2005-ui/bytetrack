# R105 — 30 lượt Nano đã chấm; chờ review Codex độc lập

**Mốc triển khai và Tier 1 hoàn tất.** Đã kiểm trực tiếp bằng chứng **8/8 smoke + 30/30 full**, chấm TrackEval và tính lại tốc độ từ CSV. Trên bốn clip UA, stride2 đạt trung vị **25,50 FPS**, tăng ghép cặp **87,98% so v21combo / 76,04% so YOLO11n full**, nhưng FN tăng 101 và khớp lần đầu muộn thêm trung vị 80 ms. Motion/guarded giữ chất lượng gần full hơn về FN, song chậm hơn stride2 và có clip giảm FPS. Chưa có review độc lập và không thay cấu hình luận văn.

## Phạm vi và bằng chứng thực thi

Theo `00_protocol.md`: bốn clip UA 25 fps × v21combo/full/stride2/motion/guarded = 20 lượt; bốn clip BDD 5 fps × v21combo/full = 8 lượt; guarded trên hai clip BDD đầu = 2 lượt. Mỗi lượt đọc đủ 150 JPEG qua tmpfs, KF/c25, queue 4, không render/hiển thị; đảo thứ tự mode theo clip. Smoke trước full có tám lượt trên một clip BDD và một clip UA. Đây là mẫu thăm dò R102 đã có, không dùng để tune hoặc đổi cấu hình chính thức.

`host_evidence/` sao nguyên byte từ bridge trong scratch: `smoke/`, `full/`, `executed_sources/`, log benchmark và `host_exit.json`. Coder không mở SSH mới. Blocker SSH cũ ở `01_blocker.json` là lịch sử; trạng thái bridge đã hoàn tất và không còn chặn phần chấm.

- `full/complete.json`: 30 lượt; SHA archive **146d37e5d18e8e19ba3a5b1a3930407886b722295d7cd4230d7b22b8f96f9c64**.
- Engine v21combo: **a6912393829f8c811d3cc33008f3e4056a626434df327c85aae7beb33af01749**; YOLO11n 50/50: **13f8cd0f9777840052fc671a621dc0473a103eeecce71ce655b54c74a4ba9823**. ONNX khớp hằng R102 trong runner/evaluator và HELLO từng lượt; không build lại engine.
- **5.700 dòng frame** trong 38 lượt khớp frame 1–150, không unreadable/truncated; lịch skip chỉ frame chẵn. BDD không skip; stride2 chính xác 75 inference/150 frame. Cấu hình chạy/command, KF, c25, mapping lớp, ignore-region hash trong mỗi clip, worker exit 0 và backend NumPy đã được đối chiếu.
- Driver R86/R101/R102 trong bản sao và nguồn gốc đều khớp SHA. Đối chiếu MOT với `review102_ridesafe_pf_nano/pipeline_pairs.json` khớp **16/16** lượt full/v21combo; đây là kiểm hồi quy byte, không dùng tốc độ lịch sử làm mẫu ghép cặp mới.
- Boot ID không đổi, EXT4 errors 0; nhiệt telemetry tối đa 50°C. Smoke không đổi swap counters. Full swap-in **2625→2629 trang**, swap-out **63622→63622**, swap dùng **239,734→239,719 MiB**; không tuyên bố hoàn toàn không dùng swap. Nvpmodel báo MAXN nhưng truy vấn EMC có `Permission denied`, governor schedutil và CPU thay đổi xung; không sửa máy.
- Bridge ghi `smoke_exit/full_exit/collection_exit=0`, `active_r105_jobs=[]`. Coder đã kiểm nội dung record này và marker kết thúc trong log; không có kiểm SSH/process mới do coder tự chạy. Mọi subprocess PC của coder đều đã được chờ và kết thúc.

## Nguồn đã chạy và bundle local

Nano chạy schema R105 sớm: runner SHA **94be03b3e12ca8d3da1f9f16db0c136a2a6ee16b92ff333554155ae271f96ef1**. Cổng riêng chỉ nhận runner này, kiểm nguồn snapshot khớp `inputs.json`/`complete.json`, ba driver gốc, package sources, smoke cùng source/engine và archive. Đã kiểm **10 file nguồn duy nhất / 13 tham chiếu**. Mỗi run còn ghi sáu hash nguồn chuẩn hóa LF, nhất quán theo từng package A/B; ba file B có snapshot khớp các hash này.

Bundle local có thêm kiểm driver manifest, package stale và hash toàn bộ package trước/sau. **Bundle local mới hơn chưa được chạy trên Nano**; manifest `deployed=false` của bundle này vẫn đúng. `executed_to_current.diff` giữ khác biệt; cadence và cơ chế patch motion/guarded không đổi. Không sửa JSON raw để thêm field mới. Schema sớm thiếu hash toàn bộ package và thiếu snapshot toàn bộ source baseline; sáu hash trọng yếu/run và hồi quy MOT không thay thế được hạn chế đó. Không chạy lại benchmark chỉ để đổi schema.

## Cách tính và đầu ra tái lập

`eval_benchmark.py` tái dùng staging/first-match R103, GT/metric/cấu hình R102; bổ sung nhóm fallback hai clip. Full làm mốc chất lượng. Trong `quality_full_A_candidate_B`, **A luôn là YOLO11n full**, B là mode đang xét; vì vậy phần `v21combo` có A=full/B=v21combo.

FPS ổn định mỗi run = **125 / (t_xong_frame150 − t_xong_frame25)**, bỏ 25 frame warmup. FPS/p90 được tính lại từ timestamp từng frame; p90 dùng phần tử sắp thứ tự ở `round(0,9×124)`, đúng code Nano, không nội suy. Mức tăng là trung vị **100×(FPS_candidate/FPS_reference−1) theo từng clip**, không lấy tỷ số hai trung vị. FPS và p90 gộp bên dưới là trung vị bốn run; p90 không phải độ trễ end-to-end gồm khởi tạo/ghi file.

- Điểm gộp và theo clip: `host_evidence/full/assessment.json`.
- Toàn bộ 30 lượt, tốc độ, gain từng cặp, inference, scheduler, HOTA/IDF1/IDSW/FN/FP: `per_run_verified.csv`.
- Khớp lần đầu của từng ID mới: `first_match_verified.csv`; cột ID là **chỉ số chuẩn hóa của evaluator**, không phải ID GT gốc. ID xuất hiện ngay frame 1 bị loại theo R103; ID không khớp có ô trống và không tham gia trung vị độ trễ.
- Hash nguồn chấm, GT gốc tám clip, manifest JPEG, artifact và kiểm hồi quy: `verification.json`.

JPEG gốc không nằm trong evidence archive; hash từng ảnh nằm ở các file `*_images.json`. Runner sao một lần vào tmpfs cho cả nhóm mode/clip. Coder kiểm manifest/đầu-cuối/hash liên kết và source runner, không đọc lại pixel gốc trên Nano.

## UA — bốn clip 25 fps

Các dòng chất lượng dưới đây theo thứ tự **HOTA%; IDF1%; IDSW; FN; FP**. Inference liệt kê theo MVI_39031, MVI_40701, MVI_40772, MVI_40855.

- **v21combo:** 13,28 FPS; p90 95,95 ms; chất lượng **64,02; 73,97; 23; 2.700; 1.327**; 150 inference/clip.
- **YOLO11n full:** 14,49 FPS; **+6,73% so v21combo**; p90 85,99 ms; **68,66; 78,53; 49; 1.810; 1.243**; 150 inference/clip.
- **stride2:** 25,50 FPS; **+87,98% so v21combo / +76,04% so full**; p90 83,14 ms; **68,41; 78,37; 23; 1.911; 1.138**; inference **75/75/75/75**, tổng 300.
- **motion:** 16,88 FPS; **+30,65% / +17,23%**; p90 93,80 ms; **68,43; 78,63; 26; 1.872; 1.175**; inference **150/110/112/80**, tổng 452.
- **guarded:** 16,02 FPS; **+19,77% / +7,67%**; p90 95,19 ms; **68,52; 78,66; 28; 1.862; 1.189**; inference **150/126/116/83**, tổng 475.

Stride2 so full giảm HOTA **0,25 điểm phần trăm**, tăng FN **101**, giảm IDSW **26**. Không thể gọi là giữ nguyên chất lượng chỉ từ HOTA gần nhau. Motion/guarded so full tăng FN **62/52**, giảm IDSW **23/21**, nhưng p90 trung vị tăng **7,81/9,21 ms**. Guarded dùng thêm 23 inference so motion, FN giảm 10 nhưng IDSW tăng 2; không có ưu thế đồng thời ở mọi metric.

Có 16 ID GT mới, 14 ID cùng được full và mỗi mode skip khớp ít nhất một lần. Stride2 muộn hơn ở **10/14**, trung vị thêm **80 ms**; motion **5/14**, guarded **4/14**, trung vị đều **0 ms**. Trung vị 0 không có nghĩa mọi ID đều không chậm. Hai ID chưa cùng khớp không được tính độ trễ.

## BDD — bốn clip 5 fps và fallback hai clip

- **v21combo:** 13,70 FPS; p90 81,70 ms; HOTA **29,05%**, IDF1 **38,52%**, IDSW **55**, FN **2.499**, FP **134**.
- **YOLO11n full:** 14,48 FPS; tăng ghép cặp **6,00%**; p90 79,11 ms; HOTA **45,64%**, IDF1 **61,01%**, IDSW **45**, FN **1.669**, FP **93**.
- **Fallback guarded, chỉ hai clip đầu:** scheduler `enabled=false`, **150/150 inference** từng clip, MOT và lớp **trùng byte full 2/2**. Hai mode đều HOTA **52,17%**, IDF1 **64,80%**, IDSW **9**, FN **454**, FP **15**; 15 ID mới cùng khớp, chênh first-match 0. Số này thuộc tập con hai clip, không so trực tiếp với điểm bốn clip ở trên.

Fallback FPS 14,68 vs full cùng hai clip 14,59, chênh ghép cặp +0,64%; vì cùng suy luận đủ khung, coi đây là dao động phép đo, không quy cho cơ chế skip. Không suy rộng stride2 sang BDD 5 fps.

## Vì sao motion/guarded kém hơn stride2

EOS scheduler và inference thực cho thấy hai nguồn chi phí:

1. Motion bật lại inference nên tổng **452**, guarded **475**, cao hơn stride2 **300** trên 600 frame UA. Guarded ưu tiên lost trước motion, vì vậy reason counts không được dùng như các nguyên nhân độc lập để cộng so giữa mode.
2. Scheduler còn resize/so ảnh/mask/kiểm lost: motion **4,68/9,43/6,44/9,87 ms mỗi frame**, guarded **4,91/10,65/6,90/10,30 ms mỗi frame**. Đây là tổng CPU scheduler chia 150, không phải thời gian GPU và không phải chứng minh từng hàm nào chiếm ưu thế.

Ở **MVI_39031**, cả hai dùng đủ 150 inference, MOT/lớp trùng full nhưng FPS giảm **7,03%/6,90% so full**. Ở MVI_40701 guarded chỉ tăng **0,68% so full**. Các ngoại lệ này được giữ cùng kết quả gộp. GPU exec của YOLO11n khi thật sự inference vẫn quanh **41,9 ms** ở các mode; stride2 tăng throughput bằng giảm số inference, không phải làm engine nhanh hơn. `infer_ms` trung vị gồm cả frame skip có thể bằng 0; CSV bổ sung trung vị chỉ trên frame có inference để tránh hiểu sai.

Chưa có phép đo NMS/decode bổ sung trong evidence bridge đã hoàn tất. Phiên coder chỉ có đường nhận evidence, không có kênh SSH đáng tin cậy để tự gửi lệnh bổ sung; không thử lại đọc khóa hoặc bỏ host-key check. Yêu cầu đo bổ sung trong đúng R105 ghi ở `12_bridge_nms_request.md`; chưa có job nào được mở cho yêu cầu này và không báo gain mới từ số R103.

## Tier 1 đã chạy

Môi trường **C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe**, `-B`, chỉ CPU trên PC. Lệnh tổng đã chạy, **exit 0**:

```powershell
& 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe' -B handoff/logs/review105_nano_fps/run_tier1.py handoff/logs/review105_nano_fps/host_evidence handoff/logs/review105_nano_fps/pc_stage_verified
```

`tier1_checks_current.json` ghi từng command, exit và source SHA:

- `test_r105.py`: **exit 0, 20/20**, `cpu_tests_final_verified.log`. Gồm fixture worker thực đã patch, lịch/threshold/ignore/lost/fallback, source isolation, stale package, driver gốc, FPS ghép cặp, oracle TrackEval nhóm hai clip, first-match, legacy schema và từ chối FPS/latency bị sửa sai.
- `audit_benchmark.py <host_evidence> <stage>`: **exit 0**, `audit_final_verified.log`: 38 run, 5.700 frame, tính lại FPS/p90, hash/config/fallback, 16/16 hồi quy MOT, đối chiếu từng ID với aggregate. Đây là tự kiểm của coder, không phải review độc lập.
- Compile **11** file Python local và AST Python 3.6 cho bench/prepare/cadence: **exit 0**, không tạo pycache.
- Kiểm lại bundle hiện có và **7/7** payload/hash với source local: **exit 0**, không build/deploy lại.

Chấm thực cuối đã chạy bằng `eval_benchmark.py <host_evidence/full> <R105/pc_stage_verified>`: **exit 0**, `eval_repo_stage.log`; `assessment.json` là điểm từ phép chấm này. Không chạy detector/GPU trên PC. Fixture cuối dùng `pc_test_scratch/` trong R105. Các lượt chấm đầu đã ghi hai staging vào scratch được nêu trong plan (`r105_score_verified_20261112`, `r105_score_timestamp_20261112`); kiểm cuối chuyển toàn bộ staging/fixture vào write_paths trong repo theo giới hạn ghi của phiên coder. Không xóa staging trước đó.

Giữ log các lỗi tự kiểm đã sửa: `eval_timestamp.log` exit 1 vì harness ban đầu hiểu field hiển thị là null thay vì `{"bat":false}`; `audit_benchmark.log` exit 1 vì đường dẫn evaluator TrackEval nằm ngoài repo, đã sửa ghi đường dẫn tuyệt đối. Đây là lỗi cổng audit trên PC; benchmark Nano không bị chạy lại hoặc sửa raw. Lượt diagnostic in console trước đó kết thúc exit 1 khi đọc log UTF-16 bằng UTF-8; không dùng lượt đó làm kiểm đạt. Các log test cũ 17/19/20 được giữ theo từng mốc, kết quả cuối ở `tier1_checks_current.json`.

## Giới hạn và bàn giao

Một lượt/mode/clip, các cửa sổ R102 đã biết, camera/mật độ hạn chế; không có lặp để ước lượng bất định hoặc CI. Không suy FPS này sang video decode/live display/khởi tạo hoặc sang mọi traffic 25 fps; chỉ trung vị bốn clip stride2 đạt 25,50 FPS, không phải cả bốn đều đạt 25 FPS. Không tune scheduler từ điểm BDD val/UA test.

Mốc coder có benchmark và chấm chất lượng trực tiếp, nhưng **review bắt buộc của phiên Codex khác chưa diễn ra**. Reviewer cần kiểm lại `13_review_brief.md`, nhất là source thực khác bundle mới, frame/timing ghép cặp, giới hạn provenance và đánh đổi FN/first-match. Không tự mở R106/R107. Đã cập nhật mục 0 PLAN trước khi viết báo cáo/brief; không sửa Git, protected code/config, dữ liệu gốc, launcher hay control state.
