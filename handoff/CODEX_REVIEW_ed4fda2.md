# Review delta a8ecaf6 → ed4fda2

Ngày hoàn tất: 2026-09-19. **Kết luận: REQUEST CHANGES.** Có hai lỗi P1 cần sửa trước khi dùng backend NumPy qua runner để sinh kết quả hoặc đo hiệu năng. Review này không đưa ra bằng chứng buộc chạy lại các thí nghiệm UA-DETRAC A/B/C đã thực hiện bằng backend Ultralytics.

Gói đã đọc: `review_delta_latest.zip`, 358.732 byte, SHA-256 `a6b758598795842614d8b2e237218931dc24863c33ac4cad18f92194098f72fc`.

- Base khai báo: `a8ecaf615eb583650d07020c6c0b534e6e2c7d0e`.
- Target khai báo: `ed4fda26b0c2163e5bf224935916fd1c03f78549`.
- Không truy cập hay sửa repository Windows, index, commit hoặc lịch sử Git. Các phép tái hiện chạy trong bản sao phục vụ review.

## 1. P1 — Runner NumPy vẫn bắt buộc import Ultralytics

**Vị trí:** `src/run_tracking.py:88`, lời gọi ở `:302`; `src/run_experiment.py:210`.

`load_tracker_cfg()` có nhánh NumPy, nhưng `resolve_tracker_yaml()` vẫn luôn gọi `ultralytics.utils.checks.check_yaml`. Cả CLI tracking và bước tracking của experiment gọi resolver này để hash YAML.

**Đã tái hiện độc lập:** môi trường không cài torch/Ultralytics, đặt `BYTETRACK_BACKEND=np`, dùng cache tổng hợp hợp lệ gồm bốn frame và YAML có đường dẫn tuyệt đối. CLI `run_tracking.py` thoát với mã 1:

```text
run_tracking.py:302 → resolve_tracker_yaml(...)
run_tracking.py:90  → from ultralytics.utils.checks import check_yaml
ModuleNotFoundError: No module named 'ultralytics'
```

`run_experiment.stage_track()` cũng thất bại tại cùng đường import. Trong khi đó, gọi trực tiếp `track_video()` với CV và EKF đều chạy được, mỗi cấu hình trả bốn dòng, không nạp torch/Ultralytics.

**Cần sửa:** dùng chung một resolver biết backend cho cả đọc cấu hình và hash nguồn YAML. Thêm kiểm thử CLI và `stage_track` trong tiến trình chặn import torch/Ultralytics. Parity ở cấp `track_video()` chưa kiểm chứng được hai đường vào này.

## 2. P1 — Cấu hình NumPy bị ghi thành `{}`, làm resume bỏ qua thay đổi YAML

**Vị trí:** `src/ul_backend.py:85`; `src/run_tracking.py:311`; `src/run_experiment.py:213` và `:223`.

`_Cfg` kế thừa `dict`, lưu tham số trong mapping. Vì vậy `dict(cfg)` có đủ cấu hình nhưng `vars(cfg)` bằng `{}`. Runner dùng `vars(cfg)` để ghi metadata và tính fingerprint đầu vào của bước tracking. Hash YAML có được ghi vào metadata, nhưng không nằm trong fingerprint quyết định tái sử dụng bước này.

**Đã tái hiện độc lập:** để kiểm tra lỗi nằm sau lỗi P1 phía trên, chỉ trong tiến trình review tôi thay resolver bằng hàm trả lại đường dẫn YAML tuyệt đối đã cung cấp. Không sửa file mã nguồn. Sau đó chạy chính `stage_track()`:

| Phép kiểm | Kết quả thực tế |
|---|---|
| Bốn frame, mỗi frame một detection confidence 0,9; `new_track_thresh=0.25` | Sinh bốn dòng tracking |
| Sửa cùng file YAML thành `new_track_thresh=0.95` | Hash YAML thay đổi |
| Chạy lại với resume | `0 video moi, 1 tai su dung` |
| Fingerprint tracking trước/sau | Giống nhau |
| `tracker_cfg` lưu trong metadata | `{}` |
| Output được tái dùng | Vẫn bốn dòng cũ |
| Chạy mới trực tiếp bằng cấu hình 0,95 | Không có dòng tracking |

Đây là lỗi có thể trả kết quả của cấu hình cũ dưới cấu hình mới. Fingerprint của bước evaluation đọc cấu hình đúng không thể sửa lại file tracking đã bị tái sử dụng sai.

**Cần sửa:** có một cách chuyển cấu hình hiệu lực thành mapping dùng thống nhất cho metadata và fingerprint, hoạt động với cả hai backend. Kiểm thử phải sửa nội dung YAML trên cùng đường dẫn, chứng minh tracking chạy lại và output thay đổi; lần tiếp theo không đổi gì mới được tái sử dụng.

## 3. P2 — Bật log association làm backend NumPy lỗi

**Vị trí:** `src/assoc_log.py:27`, `:193`; `src/run_tracking.py` gọi `with_assoc_log()` khi bật log. YAML mới `configs/phase2/experiments/bdd_smoke_val.yaml:42` đặt `assoc_log: true`.

Module log import trực tiếp `TrackState` và `BYTETracker` từ Ultralytics. Ngoài import, phép kiểm `issubclass` còn dùng lớp nền Ultralytics, khác lớp nền được vendor cho NumPy.

**Đã tái hiện:** `make_tracker(cfg, assoc_log=True)` trong tiến trình chặn Ultralytics phát sinh `ImportError`. Lỗi tồn tại độc lập với resolver.

**Cần sửa:** lấy các lớp nền từ backend selector; kiểm thử chạy và ghi log thực sự với NumPy khi không có torch/Ultralytics. So output tracking khi bật/tắt log để bảo đảm việc ghi log không đổi hành vi. Đây là thiếu sót tích hợp backend mới vào module có sẵn.

## 4. P2 — Đường dẫn YAML bị sai có thể âm thầm nạp cấu hình mặc định

**Vị trí:** `src/ul_backend.py:104`, đặc biệt nhánh fallback ở `:118`.

Nếu một đường dẫn tương đối không tồn tại, code bỏ phần thư mục và thử lấy bản vendor theo `p.name`. Đã tái hiện: `review_missing_path/bytetrack.yaml` không tồn tại nhưng vẫn trả cấu hình ByteTrack mặc định, không báo lỗi.

Điều này có thể làm thí nghiệm dùng sai cấu hình khi người chạy viết sai đường dẫn.

**Cần sửa:** chỉ cho phép tra bản vendor đối với tên trần đã hỗ trợ, ví dụ `bytetrack.yaml`. Đường dẫn có thư mục, hoặc đường dẫn tuyệt đối, phải báo lỗi nếu không tồn tại. Dùng cùng quy tắc trong resolver của mục 1.

## 5. P2 — Hai hash vendor trong PROVENANCE không khớp byte được bàn giao

**Vị trí:** `third_party/ultralytics_trackers/PROVENANCE.json`.

| File | SHA-256 trong PROVENANCE, rút gọn | SHA-256 byte trong ZIP, rút gọn |
|---|---|---|
| `byte_tracker.py` | `0e0df805a18c…` | `e4efea1f8187…` |
| `utils/matching.py` | `9adce2b668d2…` | `4d33a6c87845…` |

Hai hash khai báo khớp khi chuyển nội dung LF trong ZIP sang CRLF. Ba file vendor còn lại khớp hash. Đây là sự không nhất quán về quy ước byte/hash, chưa phải bằng chứng thuật toán bị thay đổi.

Tôi đối chiếu độc lập wheel `ultralytics==8.4.141`: cả năm hash upstream khớp; nội dung năm module vendor khớp nguồn sau đúng năm thay đổi import đã khai báo và chuẩn hóa xuống dòng. File `bytetrack.yaml` vendor cũng khớp byte nguồn.

**Cần sửa:** nêu rõ hash của byte Git/ZIP hay hash văn bản đã chuẩn hóa. Sinh và kiểm chứng hash theo đúng quy ước đó. Test hiện tại đối chiếu nội dung bằng `read_text()` nên bỏ qua khác biệt xuống dòng và không xác minh các trường hash trong PROVENANCE.

## 6. P2 — Một số diễn giải vượt bằng chứng vẫn còn trong tài liệu

Các sửa đổi về làm tròn trực tiếp, timer `assoc = update − predict`, manifest và việc lưu mọi lượt đo đã được đưa vào. Tuy nhiên, vẫn còn các câu sau:

- `handoff/MOC2_FINAL_SUMMARY.md:129`: “tăng có ý nghĩa — kể cả ở tầng che hoàn toàn”. Bảng trình bày CI cho chỉ số tổng và tỷ lệ tầng full, chưa cung cấp CI riêng chứng minh khẳng định ở tầng full. Cần dẫn đúng phép kiểm theo tầng, hoặc chỉ báo cáo tỷ lệ tăng từ 11/182 lên 22/182.
- Cùng file `:153`: “chênh lệch không do seed”. Ba seed chỉ cho thấy chênh lệch quan sát được lớn hơn biến thiên giữa ba seed đã thử; không loại trừ mọi ảnh hưởng của seed.
- Cùng file `:188` và `:203`: số đo khi máy chạy việc khác vẫn được dùng để “so tương đối”, và chênh lệch detector được khẳng định là nguyên nhân đảo thứ tự tốc độ. Các lượt chưa được kiểm soát không đủ để xếp hạng hiệu năng chắc chắn hoặc kết luận nguyên nhân. Nên gọi đây là số đo mô tả trong các lượt đã chạy.
- `scripts/nano/README.md:43`: “nhiễu … gấp ~9 lần”, “trên Nano … nặng hơn” biến chênh lệch giữa ít lượt PC thành kết luận về nhiễu và thiết bị chưa đo. Hạ thành rủi ro cần kiểm tra. Cache loại chi phí detector, nhưng tổng thời gian tracker vẫn gồm association và vòng đời; nó không tự cô lập riêng chi phí bộ lọc.
- Giao thức Nano còn ghi A/B/A/B ở mục 2 nhưng yêu cầu cân bằng AB/BA ở mục 4. Đồng bộ thành một lịch đo rõ ràng. Nếu có bảng phụ loại lượt throttling, quy định cách giữ/loại cả cặp để hiệu số ghép cặp vẫn có nghĩa.

Các điểm này cần sửa diễn giải hoặc dẫn bằng chứng còn thiếu, không yêu cầu mở thêm tuning hay chạy lại benchmark để có kết quả thuận lợi hơn.

## Phạm vi đã kiểm chứng và giới hạn

| Nội dung | Trạng thái |
|---|---|
| ZIP, CRC, đường dẫn bất thường/trùng, file riêng bị loại | Không phát hiện lỗi trong gói này |
| Hash payload và mục log dedup trong manifest | Khớp 42/42 mục logic |
| Patch đảo ngược rồi áp lại | Khớp 21 hash preimage được khai báo và 42 hash target |
| Năm module upstream/vendor | Khớp nội dung sau import rewrite; hai hash vendor có vấn đề xuống dòng như mục 5 |
| Core CV/EKF với cache tổng hợp bốn frame, chặn torch/Ultralytics | Chạy được |
| CLI, resume khi YAML đổi, logging, YAML sai đường dẫn | Đã tái hiện các lỗi trên |
| Suite 23/23 và parity trên ba sequence thật | Có trong log bàn giao; không phải kết quả tôi chạy lại độc lập |
| Parity đầy đủ trên cache thật, GPU, Jetson Nano | Chưa chạy độc lập trong môi trường review |

Các file thay đổi dùng để chạy probe khớp target trong manifest. Module phụ thuộc không nằm trong delta được lấy từ các snapshot bàn giao trước có sẵn. Tôi không có checkout đầy đủ, độc lập của `a8ecaf6`; phép round-trip xác minh tính nhất quán patch/preimage/target trong phạm vi gói, không tái xác nhận tuyên bố tái tạo toàn bộ 159 đường dẫn của Claude.

Test fingerprint mới trong `test_run_experiment_guard.py` so hai giá trị `code_fingerprint()['backend']`. Nó có gọi `--dry-run` với NumPy nhưng không kiểm tra kết quả lời gọi đó, và không thực sự đổi backend rồi xác nhận bước tracking/evaluation chạy lại. Cần bổ sung phép kiểm qua runner, thay vì diễn giải phép so hash thành bằng chứng thực thi.

## Việc cần hoàn tất trước smoke BDD

Quy trình mới trong `bdd_smoke_val.yaml` chưa nêu bước xuất GT MOTChallenge. `check_bdd100k.py`, `det_cache.py` và runner được kiểm tra không tự gọi `export_motchallenge()` để chuẩn bị GT; các test fixture lại chuẩn bị GT riêng trước khi chạy evaluator. Cần bổ sung lệnh chuẩn bị GT và kiểm tra đầu vào trên thư mục sạch trước khi coi chuỗi lệnh smoke là hoàn chỉnh. Chú thích “tên riêng ở cả bốn chỗ” cũng cần sửa: `split_name` hiện vẫn là `BDD-val`, dùng chung với đánh giá đầy đủ. Nếu giữ chung GT, phải mô tả rõ và tránh ghi đè seqmap bằng tập con smoke.

File nhãn COCO-video mà người dùng gửi sau đó là đầu vào riêng, không phải `box_track_20` mà adapter hiện tại chờ. Khi tích hợp phải giữ ID theo video, giữ thông tin lớp/ignore/crowd, và ghi rõ các thuộc tính che khuất/timestamp không có trong nguồn. Không dùng nhãn chuyển đổi để khẳng định đã kiểm chứng toàn bộ quy trình nhãn native.

## Yêu cầu bàn giao vòng sửa

1. Sửa hai lỗi P1 và các lỗi tích hợp tại mục 3–5; bổ sung các kiểm thử tái hiện cụ thể nêu trên, gồm cả chạy qua CLI/runner.
2. Chạy lại parity CV/EKF trên cache thật và các suite bị ảnh hưởng. Nếu xác nhận cả suite, báo rõ số thực chạy, lỗi, bỏ qua và đầu vào thiếu.
3. Đồng bộ tài liệu theo mục 6; hoàn thiện bước chuẩn bị GT trước smoke BDD. Không tuning lại UA-DETRAC test.
4. Giữ nguyên kết quả cũ. Nếu đã sinh run bằng NumPy, audit và chạy lại đúng run có nguy cơ tái dùng sai cấu hình; không mặc định mọi kết quả lịch sử đều bị ảnh hưởng.
5. Bàn giao delta nhỏ, log và bằng chứng đóng từng finding. Giữ base review `a8ecaf6` cho vòng sửa này; báo cáo này chưa chấp thuận đổi base sang `ed4fda2`.

Ưu tiên hiện tại là làm đường chạy NumPy đúng từ đầu vào đến lưu kết quả. Chưa có cơ sở tuyên bố backend đã sẵn sàng triển khai hoặc đã đo được hiệu năng trên Nano.
