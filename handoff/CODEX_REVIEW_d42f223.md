# Review delta a8ecaf6 → d42f223

**Kết luận: REQUEST CHANGES — hai lỗi P1 và hai vấn đề P2 còn mở.** Các lỗi resolver, serialize cấu hình trong `stage_track`, logging và hash vendor của vòng trước đã được kiểm chứng là sửa đúng trong phạm vi nêu dưới đây. Các vấn đề mới nằm ở chọn evaluator cho smoke, resume của CLI riêng, provenance backend và timeline khi xuất GT.

Review trên gói `review_delta_latest.zip`, 530.132 byte, SHA-256 `48afee11b8417c98f4589382d03b28eeca737f7b715338d8901bb58193307e4b`.

- Base: `a8ecaf615eb583650d07020c6c0b534e6e2c7d0e`.
- Target: `d42f2236fd5c707a54508d59c54357e3f58d154b`.
- Nhãn thật được đối chiếu: `bdd_box_track_val_cocofmt.json`, SHA-256 `074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d`.
- Review chỉ đọc bản bàn giao và chạy probe trong bản sao. Không sửa source dự án, working tree, index, commit hoặc repository Windows.

## 1. P1 — `BDDSMOKE-val` chọn evaluator thường, bỏ xử lý crowd-ignore

**Vị trí:** `src/run_experiment.py:293`; mặc định của CLI `src/run_trackeval.py` cũng dùng cùng quy tắc tiền tố. YAML smoke mới dùng `split_name: BDDSMOKE-val`.

```python
dataset_cls = "mot_crowd" if split.startswith("BDD-") else "mot"
```

`BDDSMOKE-val` không bắt đầu bằng `BDD-`. Vì vậy đường chạy được bàn giao chọn `mot`, dù phần hướng dẫn trong YAML yêu cầu `mot_crowd`.

**Tái hiện độc lập:** gọi `stage_eval()` với dataset `bdd100k_coco` và split `BDDSMOKE-val`, quan sát đối số truyền tới evaluator: `dataset_class="mot"`. Probe dừng tại lời gọi này để ghi nhận lựa chọn; không thay đổi logic lựa chọn evaluator.

Sau đó chạy TrackEval 1.3.0 thật trên một fixture một frame: một GT hợp lệ được ghép đúng, một detection nằm trong vùng crowd và một FP thông thường. Hai evaluator cho kết quả:

| Evaluator | FP | FN | HOTA |
|---|---:|---:|---:|
| `mot`, lựa chọn hiện tại của runner | 2 | 0 | 0.577350 |
| `mot_crowd` | 1 | 0 | 0.707107 |

Đây là khác biệt hành vi đánh giá, không chỉ khác tên lớp. Các số trên là fixture chứng minh lỗi, không phải kết quả BDD thật.

Đối chiếu chính nhãn thật của ba video trong manifest smoke:

| Video | Số ảnh | Annotation `iscrowd=1` | Annotation `ignore=1` |
|---|---:|---:|---:|
| `b1c66a42-6f7d68ca` | 202 | 132 | 0 |
| `b1c81faa-3df17267` | 202 | 37 | 0 |
| `b1c81faa-c80764c5` | 202 | 147 | 27 |

Hai cột cờ được đếm riêng, không cộng chúng để suy ra số vùng phân biệt. Cả ba video đều có annotation cần chính sách ignore, nên smoke hiện tại chưa chứng minh đường xử lý crowd-ignore đúng. Gói không có output tracking và `eval_meta.json` của smoke để tôi tính lại ảnh hưởng trên ba video thật.

**Cần sửa và đóng finding:** chọn evaluator theo cấu hình hoặc loại dataset, không suy ra từ tên thư mục output. Giữ split smoke riêng. Kiểm thử phải đi qua runner với tên `BDDSMOKE-val`, gồm ba trường hợp: detection trong ignore, detection ghép GT hợp lệ đồng thời chồng vùng ignore, và FP thông thường. Ghi lớp evaluator trong provenance.

Sau sửa, dùng lại cache và tracking smoke đã có nếu đầu vào hợp lệ; chạy lại evaluation và tổng hợp vào thư mục kết quả mới. Giữ số cũ như lịch sử bị thay thế. Không cần chạy detector hay tracking lại chỉ vì lỗi chọn evaluator.

## 2. P1 — CLI `run_tracking.py --skip-existing` vẫn tái dùng output sai cấu hình

**Vị trí:** `src/run_tracking.py:321`, `:329`, `:333`.

Bước `run_experiment.stage_track()` đã có guard đúng. Tuy nhiên, CLI riêng vẫn ghi `run_meta.json` mới trước vòng lặp, rồi bỏ qua video chỉ vì file tracking tồn tại. Không kiểm fingerprint hay trạng thái hoàn tất ở nhánh này.

**Đã tái hiện bằng CLI thật**, backend NumPy, cùng đường dẫn YAML và cùng bốn frame, mỗi frame một detection confidence 0,9:

1. Chạy với `new_track_thresh=0.25`: exit 0, output có bốn dòng.
2. Sửa cùng file YAML thành `new_track_thresh=0.95`.
3. Chạy lại cùng CLI, thêm `--skip-existing`: exit 0, in `[SKIP]`, output vẫn nguyên bốn dòng.
4. `run_meta.json` lúc này lại ghi cấu hình mới `new_track_thresh=0.95`.
5. Gọi tracking mới bằng cấu hình 0,95 trả về không dòng nào.

Như vậy output cũ bị gắn metadata cấu hình mới. Lỗi xảy ra khi dùng cờ `--skip-existing`; không phủ nhận việc guard của `stage_track()` đã được sửa.

**Cần sửa:** thống nhất chính sách resume giữa CLI và experiment runner, hoặc từ chối resume khi không có bằng chứng tương thích. Không ghi đè provenance của output được tái sử dụng bằng cấu hình hiện tại. Kiểm thử đúng kịch bản trên qua CLI; lần không đổi gì mới được skip, đổi YAML hoặc thiếu metadata hoàn tất phải chạy lại hoặc dừng với thông báo rõ ràng.

## 3. P2 — Audit “grep không thấy np ⇒ chưa có run NumPy” không có cơ sở

**Vị trí:** `handoff/BACKEND_NP_NOTES.md:41`, `REVIEW_BRIEF.md`; metadata tại `src/run_experiment.py:211`, `:225`, `:268` và `src/run_tracking.py:321`.

`stage_track()` đưa `code_fingerprint()` vào một đối tượng rồi chỉ lưu hash tổng của đối tượng đó trong `input_fingerprint`. `run_meta.json` không lưu snapshot `code` hoặc tên backend. Metadata từng video cũng không lưu tên backend. CLI riêng tương tự.

**Phản ví dụ đã chạy:** tạo một run mới, biết chắc `ul_backend.backend_name() == "np"`. Tracking hoàn tất. Trong `run_meta.json` của chính run đó:

```text
actual_backend = np
metadata_has_backend = false
metadata_has_code = false
literal '"backend": "np"' found = false
```

Do đó lệnh grep được báo cáo có thể trả 0 ngay cả khi run NumPy tồn tại. Không thể suy ngược backend từ hash tổng.

**Cần sửa:** lưu backend và snapshot thông tin phiên bản/hash mã liên quan một cách tường minh cho run mới; giữ provenance gốc khi resume. Audit run cũ bằng JSON có cấu trúc, đối chiếu `eval_meta.json`, log và các bằng chứng còn lại. Run thiếu bằng chứng phải là `UNKNOWN/LEGACY`, không tự coi là Ultralytics hoặc kết luận không cần audit. Không tạo metadata hồi tố dưới dạng đã được xác minh.

Chỉ chạy lại run cũ khi xác định được rủi ro cụ thể hoặc cần tái lập một kết quả đang sử dụng; không mặc định chạy lại toàn bộ A/B/C.

## 4. P2 — Xuất `seqLength` từ annotation làm mất các frame cuối không có nhãn

**Vị trí:** `src/mot_datasets/bdd100k_coco.py:241`.

```python
seq_len = {v: int(df[df.video == v].frame.max()) for v in videos}
```

`df` chỉ có hàng annotation. Timeline của video phải lấy từ danh sách `images/frame_id`, bao gồm frame không có annotation.

**Tái hiện trên file nhãn thật đã gửi:**

- Video `b23b2649-8349d2a1` có frame ảnh 1–202.
- Annotation cuối ở frame 94.
- Chạy CLI adapter với `--videos b23b2649-8349d2a1 --export-gt --benchmark REVIEWTAIL --no-images` trả exit 0 nhưng ghi `seqLength=94`.

Toàn bộ nguồn có 615 ảnh không có annotation; một video có phần cuối không có annotation như trên. Ba video smoke hiện tại đều có annotation tới frame 202 nên phép kiểm ba video đã không bắt được lỗi này.

Với tracking vẫn chạy đủ ảnh, GT khai báo độ dài 94 có thể làm evaluator từ chối các dòng tracking sau frame 94. Việc cắt tracking theo GT để tránh lỗi cũng sẽ bỏ frame hợp lệ.

**Cần sửa:** lấy độ dài từ timeline ảnh, nhất quán với quy ước `frame_id + 1`; không dùng số dòng annotation hoặc frame GT cuối. Giữ đúng khoảng frame bị bỏ qua và trường hợp cả video không có annotation. Bổ sung regression bằng một fixture có frame cuối không annotation và bằng video thật nêu trên.

Quan sát liên quan, từ đọc mã: CLI hiện chỉ in số ảnh thiếu rồi vẫn trả 0. Khi chạy chế độ có kiểm ảnh, cần trả mã lỗi nếu thiếu ảnh; chế độ `--no-images` phải được ghi rõ là chỉ kiểm nhãn, không chứng minh ảnh đã đủ.

## Các lỗi cũ đã được xác nhận sửa đúng

| Nội dung | Kiểm chứng độc lập |
|---|---|
| Resolver NumPy không gọi Ultralytics | CLI thật exit 0, xuất bốn dòng trong môi trường không cài torch/Ultralytics |
| Serialize cấu hình và resume trong `stage_track()` | Sửa YAML 0,25 → 0,95: output 4 → 0 dòng, fingerprint đổi, một video chạy lại; lần tiếp theo không đổi: một video tái sử dụng |
| Logging NumPy | Chạy trong tiến trình có import blocker hiện đại; ghi ba parquet, output bật/tắt log khớp byte, không nạp torch/Ultralytics |
| YAML sai đường dẫn | Đường dẫn có thư mục không tồn tại phát sinh `FileNotFoundError` |
| Hash vendor | Cả năm hash vendor và upstream khớp theo quy ước LF; nội dung khớp wheel Ultralytics 8.4.141 sau đúng các import rewrite khai báo |
| Unit test adapter COCO-video | Chạy độc lập `test_bdd100k_coco.py`: 8/8 PASS |
| Phân biệt nhãn chuyển đổi và native | Adapter riêng, thông tin thiếu giữ NaN/MISSING; YAML smoke tắt occlusion analysis |

Hash payload trong manifest vẫn là hash byte thực tế: cả 67 mục logic khớp, gồm 64 payload trực tiếp và ba mục dedup. Patch đảo ngược/áp lại thành công, 24 preimage và 67 target hash khớp. Không thấy đường dẫn file riêng trong phạm vi patch/payload được bàn giao.

Quy ước LF hiện sửa hash **module vendor**. `run_fingerprint.code_fingerprint()` vẫn hash byte thô của các module `src`, và hash YAML cũng là byte thô. Vì thế không mở rộng kết luận thành “toàn bộ fingerprint giống nhau giữa Windows và Linux”. Khác môi trường/phần cứng còn có thể cần được phân biệt, nhất là khi báo cáo thời gian chạy. `.gitattributes` hiện dùng quy tắc rộng `* text=auto`, `third_party/** -text`; nếu điều chỉnh tiếp, nên giới hạn đúng phạm vi file cần quản lý và không chuẩn hóa lại file riêng.

## Phạm vi và giới hạn của review

Tôi chạy probe trên Python 3.12.14, dùng NumPy/SciPy hiện có, lap 0.5.13, TrackEval 1.3.0 và dependency parquet trong thư mục riêng cho review. Môi trường không cài torch/Ultralytics; wheel Ultralytics chỉ được đọc để đối chiếu nguồn. Dùng import blocker hỗ trợ `find_spec` khi kiểm các đường chạy NumPy trong tiến trình review.

Các file thay đổi khớp target manifest. Dependency không thay đổi được lấy từ các snapshot bàn giao trước có sẵn. Đây không phải checkout đầy đủ, độc lập của commit base; kiểm tra patch round-trip không tái xác nhận tuyên bố 162/162 đường dẫn của Claude trên máy Windows.

Log `tests_20260919-2259.log` có đủ **25 suite PASS, từng exit code 0**. Đây là log bàn giao. Tôi không chạy lại toàn bộ suite, phép chuyển backend sang Ultralytics, parity trên cache UA-DETRAC thật, detector GPU hoặc Jetson Nano. Không có ảnh/cache/tracking smoke thật trong môi trường review để tính lại HOTA BDD; không suy ra mức ảnh hưởng số học trên dữ liệu thật từ fixture.

Các phép kiểm độc lập nêu trên xác nhận lỗi cụ thể và việc sửa lỗi cũ; không dùng số lượng suite PASS để thay thế kiểm tra đường chạy còn thiếu.

## Phạm vi sửa và chạy lại tiếp theo

1. Sửa chọn evaluator cho `bdd100k_coco`/split tùy đặt tên; kiểm tra crowd-ignore qua runner; chạy lại **evaluation và tổng hợp smoke** từ tracking/cache cũ hợp lệ, lưu riêng kết quả sửa.
2. Sửa guard/provenance của CLI `--skip-existing`; bổ sung kiểm thử CLI đổi YAML và tái sử dụng khi không đổi.
3. Lưu backend tường minh cho run mới; sửa kết luận audit, đánh dấu run cũ chưa xác định; không dựng provenance hồi tố.
4. Sửa timeline xuất GT; kiểm thử video thật `b23b2649-8349d2a1` và fixture frame rỗng/annotation rỗng. Không cần chạy benchmark BDD đầy đủ để đóng finding này.
5. Bàn giao delta nhỏ kèm log và metadata smoke mới: `eval_meta.json`, thông tin backend/evaluator, hash GT/seqmap/nhãn nguồn và lệnh tái hiện. Không cần đóng gói dataset, weights hoặc cache lớn.

**Chưa có bằng chứng cần chạy lại toàn bộ UA-DETRAC A/B/C hoặc GĐ1.** Không mở tuning trên test. Giữ base review `a8ecaf6`; báo cáo này chưa chấp thuận chuyển sang `d42f223`.
