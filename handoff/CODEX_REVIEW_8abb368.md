**CODEX REVIEW — `8abb368` — 2026-09-20**

**Kết luận: REQUEST CHANGES cho mốc hệ thống PC.** Đường chạy trực tiếp đã dùng chung detector, bộ lọc detection và vòng lặp tracker với cache. Đã kiểm chứng độc lập trường hợp bình thường live → cache → replay trên dữ liệu tổng hợp. Còn bốn vấn đề cần sửa trong timing, điều kiện replay, chẩn đoán parity và thứ tự ảnh đầu vào.

Các yêu cầu Nano F1/F3 của review `8bab154` tiếp tục hoãn theo quyết định của người dùng, không dùng chúng làm điều kiện chặn mốc PC. Không có bằng chứng mới buộc chạy lại UA-DETRAC/BDD100K đã chốt.

**1. Snapshot và tính toàn vẹn gói**

| Mục | Kết quả kiểm độc lập |
|---|---|
| Base | `b6234653f616e8059825b59b4da70262e1f98dde` |
| Target | `8abb36864ad65b83682b892c02651e3ab1a665c5` |
| ZIP | 3.005.356 byte, 48 thành viên; `testzip` OK |
| SHA256 ZIP | `91094b08d1c84390c3acfe9e53c864ac264b9321fe79305e7a022684d70c8a18` |
| Payload | 42/42 hash đúng; 4 đường dẫn dedup có nội dung thay thế đúng hash |
| Patch | `git apply --check` và áp thật đều exit 0; 46/46 đường dẫn sau áp khớp target |
| Đường dẫn riêng bị cấm | Không có trong payload hoặc patch |

Base dựng lại không phải checkout Git đầy đủ của máy Windows. README gốc không còn trong thư mục review nên được phục hồi bằng reverse patch rồi đối chiếu SHA256 base; README Nano lấy từ snapshot trước với đúng hash. Đây là giới hạn lưu trữ đầu vào phía reviewer, không phải lỗi gói. Reviewer chỉ thao tác trên bản sao và fixture, không sửa source bàn giao, commit hoặc push.

**2. PC-F1 — [P2] `decode_ms` không đo bước đọc/giải mã ảnh; tổng pipeline thiếu thời gian**

Vị trí: `src/track_video_file.py`, `NguonFrame.__iter__()`, `LiveVideoDetections.iter_frames()` và phần tổng hợp `thoi_gian`.

`NguonFrame` đã thực hiện `cv2.imread()` hoặc `cap.read()` trước khi yield. Trong adapter, `t0` lại đặt bên trong thân `for idx, t_s, img in self.nguon`, tức là sau khi việc đọc hoàn tất. Vì vậy `decode_ms = t1 - t0` chủ yếu đo phần xử lý trước detector, bao gồm mask vùng bỏ qua khi bật; không đo decode.

Đã chạy `track_video_file.main()` thật, backend NumPy và tracker/cache thật, với bộ đọc ảnh giả lập mất 20 ms/frame và detector giả lập. Không có warm-up:

| Đại lượng | Kết quả |
|---|---:|
| Frame xử lý | 3 |
| Độ trễ bộ đọc mỗi frame | 20 ms |
| `decode_ms` được báo | **0,002295 ms/frame** |
| `pipeline_khong_ve_ms` | **1,035670 ms/frame** |
| Wall | **71,257 ms** cho 3 frame |

Đây là kiểm vị trí timer bằng độ trễ kiểm soát, không phải phép đo hiệu năng GPU.

Đối chiếu dữ liệu thật đã đóng gói: số tổng hợp đúng với CSV, nhưng CSV đã thiếu bước decode. Ngay cả khi cộng **toàn bộ 1.470 frame, gồm warm-up**, phần chênh vẫn còn:

| Run không vẽ | Tổng `decode + detect + update` từ mọi dòng CSV | Wall | Chênh |
|---|---:|---:|---:|
| CV | 24,775924 s | 30,931286 s | **6,155362 s** |
| EKF | 25,931545 s | 31,232455 s | **5,300910 s** |

Không quy toàn bộ phần chênh cho decode: còn chi phí vòng lặp, chuyển đổi và hoàn tất. Tuy nhiên, không thể giải thích phần chênh này chỉ bằng warm-up. `15,563 ms/frame` cũng không tương đương `47,5 fps`; đó là hai phạm vi đo khác nhau.

Yêu cầu sửa:

- Đo đọc/giải mã ngay tại nơi gọi `imread/read`; tách mask khỏi decode.
- Đặt timer tổng theo frame bao phủ đúng phạm vi pipeline đã công bố; nếu còn overhead ngoài các stage, ghi riêng thay vì đặt tên tổng của vài stage là toàn pipeline.
- Lưu render/write/show theo frame nếu báo cáo các stage này. Gói hiện chỉ có giá trị trung bình render/write trong metadata, chưa đủ để tự tính lại chúng từ CSV.
- Dùng cùng tập frame sau warm-up khi so các tổng trung bình. Nêu rõ wall hiện đo vòng xử lý và đóng video, chưa bao gồm ghi MOT/CSV/cache sau vòng lặp.
- Thêm test độ trễ đọc kiểm soát. Sau sửa, đo lại riêng bảng PC này bằng output mới, giữ bản cũ để đối chiếu; không cần chạy lại các ma trận đánh giá tracking.

**3. PC-F2 — [P2] Live chấp nhận cấu hình/detection mà replay cache từ chối**

Vị trí: `src/track_video_file.py`, khởi tạo `DetectorSpec`/`LiveVideoDetections`, ghi `truncated`, lưu cache và sinh lệnh replay; đối chiếu `det_cache.check_compatible()`, `VideoDetections`.

Tái sử dụng `filter_dets()` chưa bao gồm các guard của đường đọc cache. Live không kiểm giới hạn confidence/max_det giữa producer và consumer; khi detection chạm trần, chỉ ghi cờ `truncated` rồi tiếp tục thành công. Replay mặc định lại từ chối cache đó.

Đã tái hiện qua `main()` của hai entry point thật với detector/đọc ảnh giả lập và tracker NumPy thật:

| Ca | Live | Replay chính cache vừa lưu |
|---|---|---|
| Thông thường | exit 0 | exit 0; bbox/conf lệch 0, ID khớp |
| `--det-max-det 1 --max-det 1`, mỗi frame có đúng 1 detection | **exit 0**, lưu 3 frame `truncated=True` | **ValueError**: frame chạm trần max_det |
| `--det-conf 0.5 --conf-min 0.1` | **exit 0** | **ValueError**: cache không chứa detection dưới 0,5 |

Lệnh replay được ghi trong metadata không xử lý khác biệt này. Trường hợp chạm trần chỉ chứng minh có nguy cơ bị cắt theo chính sách cache hiện hành, không khẳng định detector thực tế đã bỏ bao nhiêu box.

Yêu cầu sửa:

- Dùng chung chính sách kiểm tương thích producer/consumer trước khi chạy: confidence, giới hạn detection và các điều kiện liên quan.
- Đồng bộ cách xử lý frame chạm trần giữa live và replay. Có thể từ chối rõ ràng theo mặc định, hoặc có tùy chọn chấp nhận được công bố và truyền nhất quán vào metadata/lệnh replay. Không tự tắt strict để làm test PASS.
- Thêm regression live → save-cache → replay cho ba ca trên; kiểm cả mã thoát/trạng thái, số detection và đầu ra. Kiểm thêm `max_det` consumer lớn hơn producer vì cùng một guard còn thiếu.
- Nếu cần cho phép cấu hình live không thể replay theo chế độ strict, phải ghi rõ trạng thái đó thay vì tuyên bố lệnh tái hiện sẵn dùng.

**4. PC-F3 — [P2] Chẩn đoán parity còn khẳng định nguyên nhân và liên kết ID vượt bằng chứng**

Vị trí: `src/nano_parity.py`, `so_phan_hoach()` và nhánh thông báo chẩn đoán trong `so_mot_video()`.

**Phần đã sửa đúng:** chạy độc lập `test_nano_parity.py` đạt **15/15**, exit 0. Các fixture gần/trùng tọa độ của review trước đã được bao phủ. Hai cặp kết quả PC thật cũng vẫn FAIL đúng ở mức đầy đủ vì confidence vượt `1e-6`; không phát hiện việc nâng tolerance hoặc đổi FAIL thành PASS.

**Phần còn sai thứ nhất:** công cụ chỉ nhận hai file tracking nhưng tự in "liên kết ID y hệt → khác biệt nằm ở đầu vào detector, không ở tracker". Nó không có raw detections hay chứng cứ tính toán để xác định nguyên nhân. Với hai dòng cùng bbox/ID, chỉ đổi score từ `0.9` sang `0.90002` trực tiếp trong file kết quả, công cụ cũng quy nguyên nhân cho detector.

Đối chiếu các cặp thật được đóng gói:

| So hai lượt độc lập | Hàng/track mỗi bên | Max bbox | Max confidence | Kết quả |
|---|---:|---:|---:|---|
| CV | 5.842 / 72 | 0,0001220703 px | 0,00000864267 | ID/phân hoạch khớp, **parity đầy đủ FAIL** |
| EKF | 5.845 / 70 | 0,0001220703 px | 0,00000309944 | ID/phân hoạch khớp, **parity đầy đủ FAIL** |

Các con số và việc khớp ID ở hai cặp này được xác nhận độc lập. Nguyên nhân "nhiễu GPU/cuDNN, không phải tracker" chưa được chứng minh chỉ bằng những file đầu ra đó. Log kiểm detection riêng là bằng chứng cho phạm vi phép kiểm riêng ấy, không biến thông báo tổng quát của comparator thành kết luận nhân quả hợp lệ.

**Phần còn sai thứ hai:** khi hình học có nhiều ứng viên, code vẫn chọn Hungarian rồi kết luận phân hoạch đúng/sai. Tái hiện với hai track có bbox/conf giống hệt nhau ở hai frame:

```text
ref:  (frame 1: ID 1,2), (frame 2: ID 1,2)
test: (frame 1: ID 8,9), (frame 2: ID 8,9)
mọi bbox = (10,20,30,40), conf = 0.9
```

Ban đầu báo `phan_hoach_khop=True`, chỉ đánh số lệch. Chỉ đảo thứ tự hai dòng của **frame 2 trong test**, giữ nguyên toàn bộ tập bản ghi, lại báo `phan_hoach_khop=False`, "tracker đã ghép khác". Ở cả hai ca, công cụ đếm 8 box có ứng viên mơ hồ; dữ liệu không đủ để chọn một ánh xạ hình học duy nhất.

Yêu cầu sửa:

- Thông báo điều quan sát được: bbox/conf khác, ID/phân hoạch đạt hoặc không đạt. Nguyên nhân chưa đủ bằng chứng thì ghi unknown; không mặc định quy cho detector.
- Khi chưa giải được sự mơ hồ bằng ràng buộc toàn sequence, trả trạng thái mơ hồ/chưa xác định, không coi lựa chọn tùy ý của Hungarian là bằng chứng đổi liên kết. Không biến unknown thành PASS.
- Thêm regression bảo đảm đảo thứ tự dòng không đổi kết luận; giữ nguyên kiểm strict `(frame,id)` và tolerance đã công bố.
- Cập nhật diễn giải tương ứng trong brief, hướng dẫn và thông báo CLI.

**5. PC-F4 — [P2] Thư mục ảnh có thể bị đọc sai trình tự frame**

Vị trí: `src/track_video_file.py`, `NguonFrame.__init__()` nhánh thư mục ảnh, `paths = sorted(...)` và đánh lại frame từ 1.

Với ba file `1.jpg`, `2.jpg`, `10.jpg`, nguồn hiện đọc **`1.jpg → 10.jpg → 2.jpg`**, gán timestamp tăng đều theo thứ tự này. Đây là sắp xếp chữ, không theo số frame. Luồng tracking sẽ quan sát sai diễn biến chuyển động đối với kiểu đặt tên số phổ biến này.

Đã tái hiện trên thư mục tạm bằng constructor thật; chỉ bộ đọc nội dung ảnh được giả lập.

Yêu cầu sửa: chốt quy ước thứ tự ảnh rõ ràng, dùng manifest hoặc sắp số/natural order cho tên frame phù hợp. Nếu bắt buộc tên zero-padding thì kiểm/giải thích rõ thay vì âm thầm coi mọi tên file là đúng thứ tự. Ghi thứ tự file/frame/time vào manifest đầu vào để tái hiện. Thêm test tên chưa padding, tên đã padding và quy tắc tên mơ hồ.

**6. Các điểm tài liệu/provenance cần chỉnh cùng đợt**

- **Ngưỡng thấp của ByteTrack:** mặc định `det_conf=0.05` nhưng `conf_min=0.25`; bước lọc vẫn loại detection dưới 0,25 trước khi tracker nhận. Fixture ba frame với scores `[0.9,0.15,0.9]` cho `n_dets=[1,0,1]`, output chỉ frame 1/3 ở c25; c10 giữ đủ ba frame. Đây là điều kiện c25 đã có chủ ý trong thí nghiệm, không yêu cầu sửa lịch sử. Tuy nhiên, tài liệu không nên nói đặt detector 0,05 đã đủ bảo đảm vòng ghép confidence thấp hoạt động. Nêu rõ preset ứng dụng và ngưỡng hiệu lực; không tự đổi cấu hình khóa.
- **Ví dụ replay:** lệnh live trong `HE_THONG_PC.md` mục 4 dùng mặc định classes `1 2 3 5 7`, nhưng lệnh replay viết `2 3 5 7`. Sửa để cùng cấu hình thực tế; scene không có bicycle không chứng minh ví dụ tổng quát là đúng.
- **Dùng GT trong chế độ dataset:** adapter DETRAC đọc `gt/ignored_regions.txt`, rồi live mask ảnh. Vì vậy cần mô tả đúng: chế độ file/thư mục không cần GT; chế độ dataset có thể dùng vùng ignore để giữ quy ước benchmark. Không đưa ID/box GT vào association, nhưng cũng không tuyên bố hoàn toàn không đọc GT/annotation cho mọi chế độ.
- **Nguồn code của run:** bốn `run_meta.json` ghi commit `5c7f2fe`; entry point mới được commit sau đó và chưa nằm trong hash 20 module. Các module đã liệt kê khớp source bàn giao khi đối chiếu quy ước xuống dòng, nhưng chưa đủ gắn nội dung entry point lúc chạy với commit ghi trong metadata. Với run mới, ghi riêng hash `track_video_file.py` và tình trạng working tree; có thể làm trong metadata của entry point, không cần mở rộng fingerprint chung gây chạy lại thí nghiệm cũ. Không bổ sung bằng chứng hồi tố cho các run cũ.
- **Log test lỗi đã xóa:** không có log đó trong gói để xác minh nguyên nhân HEAD thay đổi. Giữ log thất bại ở khu vực lịch sử có nhãn rõ trong các lần sau; nếu không khôi phục được bản thật thì ghi nhận thiếu, không tạo lại log như bằng chứng của lần chạy cũ.

**7. Kiểm chứng đã thực hiện và giới hạn**

| Kiểm tra | Nguồn bằng chứng/kết quả |
|---|---|
| `test_nano_parity` | Reviewer chạy độc lập **15/15 PASS**, exit 0 |
| Live → cache → replay bình thường | Reviewer chạy hai `main()` thật, tracker NumPy thật; **3/3 hàng**, bbox/conf lệch 0, ID khớp; detector/đọc ảnh là test doubles |
| Độ trễ đọc, trần detection, ngưỡng không tương thích | Reviewer tái hiện như PC-F1/F2 |
| Parity trên output PC thật | Reviewer đọc đủ hai cặp và đối chiếu như bảng PC-F3 |
| CSV timing/metadata | **4 run × 1.470 = 5.880 frame**; số hàng/track đúng; sai số tái tính 5 giá trị thời gian/run tối đa **4,2e-17** |
| `test_track_video_file` 11/11 và full suite 30/30 | Xác nhận từ log/CSV bàn giao, **không chạy lại độc lập các test dùng YOLO/video thật** |
| Nano | Hoãn, không đo hoặc kiểm lại F1/F3 Nano trong vòng này |

Môi trường review không có OpenCV, torch, Ultralytics, checkpoint và ảnh/video gốc. Do đó không tuyên bố đã chạy lại detector GPU, GUI/codec hoặc full suite trên bản này. Fixture thay riêng I/O/detector giúp kiểm chính sách và timer; không thay thế smoke video thật trên PC tác giả sau khi sửa.

Các file nguồn target vẫn nguyên hash sau kiểm tra. Không sửa source sản phẩm hoặc kết quả gốc trong review.

**8. Phạm vi đợt sửa tiếp theo**

Sửa PC-F1–PC-F4 và các diễn giải liên quan; thêm regression đúng những tình huống đã tái hiện. Chạy smoke live/cache trên PC, sau đó đo lại riêng các lượt PC có timing bị ảnh hưởng và lưu output mới. Bảo toàn số liệu nghiên cứu, GT và lock đã chốt. Không mở tuning, train dài hay setup Nano trong đợt này.

Giữ base review `b623465`; gói kế tiếp chỉ cần delta tích lũy, log cần thiết và dữ liệu nhỏ chứng minh sửa lỗi. Chưa chuyển base hoặc đánh dấu mốc PC đã PASS trước khi đóng các finding này.
