**CODEX REVIEW — `923305d` — 2026-09-20**

**Kết luận: REQUEST CHANGES, còn hai ca nhỏ trong phạm vi PC-F1/PC-F2.** Các bản sửa chính đã hoạt động: timer đọc ảnh đặt đúng vị trí; guard confidence/max_det thống nhất; parity xử lý fixture mơ hồ đúng; ảnh được sắp theo số. Bốn bảng thời gian `v2` được tính lại độc lập và khớp CSV.

Hai ca còn lại: **dừng sớm bằng `q` bỏ mất timing cuối frame**, và **lệnh replay làm tách tên nguồn có dấu cách thành nhiều video**. Không tìm thấy bằng chứng phải chạy lại UA-DETRAC/BDD hoặc bốn lượt `v2` đã hoàn tất. Nano F1/F3 tiếp tục hoãn theo phạm vi người dùng đã chốt.

**1. Snapshot và toàn vẹn**

| Mục | Kết quả kiểm độc lập |
|---|---|
| Base | `b6234653f616e8059825b59b4da70262e1f98dde` |
| Target | `923305df44aa7f13b7d03f6412aabf63daa049cd` |
| ZIP | 6.240.819 byte, 75 thành viên; `testzip` OK |
| SHA256 ZIP | `ec740da01ac2b72ca0ed2c981472294958221072f5ff5485b6f77155e75d94bf` |
| Payload | 69/69 hash đúng; 12 đường dẫn dedup có nội dung thay thế đúng hash |
| Patch | `git apply --check` và áp thật đều exit 0; **81/81** đường dẫn sau áp khớp target |
| File riêng/dữ liệu lớn bị loại | Không có file thuộc các đường dẫn cấm trong thành viên ZIP hoặc đường dẫn đích của patch |
| Sau kiểm thử | 81/81 hash target giữ nguyên; reviewer không sửa source sản phẩm |

Reviewer dựng bản sao từ snapshot base đã lưu. README gốc thiếu trong cây review cũ được phục hồi bằng reverse patch và kiểm toàn bộ hash base; README Nano lấy từ snapshot cũ với đúng hash. Không có checkout Git đầy đủ của máy Windows trong môi trường review. Các thông báo `not a git repository` trong fixture là giới hạn môi trường này, không phải finding của dự án.

So với `8abb368`, source thay đổi trong bốn file: `track_video_file.py`, `nano_parity.py` và hai file test tương ứng. `run_tracking.py` và `det_cache.py` giữ nguyên byte. 20 file kết quả PC vòng trước cũng giữ nguyên byte trong gói tích lũy.

**2. Trạng thái các finding trước**

| Finding | Kết luận vòng này |
|---|---|
| PC-F1 — timer đọc/tổng pipeline | **Đã sửa trên đường chạy hoàn tất.** Còn ca dừng sớm ở mục 3. |
| PC-F2 — guard live/cache | **Đã sửa các ca confidence, max_det, chạm trần và tùy chọn chấp nhận.** Còn lệnh replay với dấu cách ở mục 4. |
| PC-F3 — chẩn đoán parity | **Đóng trong phạm vi đã yêu cầu.** 20/20 test chạy độc lập; fixture trùng hình học trả `mo_ho` ở cả hai thứ tự dòng; không tự quy nguyên nhân cho detector. |
| PC-F4 — thứ tự ảnh | **Đóng trong phạm vi đã yêu cầu.** Kiểm độc lập tên chưa padding/đã padding/cùng tiền tố, ba ca mơ hồ và manifest tường minh. |

**3. [P2] Chốt timing ngay cả khi người dùng bấm `q` — phần còn lại của PC-F1**

Vị trí: `src/track_video_file.py`, `LiveVideoDetections.iter_frames()` quanh dòng 279–315; liên quan `src/run_tracking.py`, `track_video()` quanh dòng 239–240; phần tổng hợp timing dòng 604–635.

Adapter khởi tạo `frame_total_ms=NaN`, rồi chỉ ghi tổng thời gian và sao chép render/write/show **sau `yield`**. Khi callback trả `False`, `track_video()` thực hiện `break`. Generator không được resume theo đường bình thường, nên đoạn chốt timing cuối frame không chạy.

Đã tái hiện qua `track_video_file.main()` thật, tracker NumPy/cache thật. Nguồn có 3 ảnh; bộ đọc giả lập 20 ms/frame; GUI giả lập trả `q` ở frame 2; `--warmup-frames 0`. Detector và GUI là test doubles, không phải smoke màn hình/GPU thật.

| Kết quả | Quan sát |
|---|---|
| Exit và trạng thái | exit 0, `dung_som=true` |
| Tracking/cache | Có đủ 2 frame đã xử lý; replay cache 2 frame **PASS**, bbox/conf lệch 0 |
| Frame 1 | `frame_total_ms=24,616536`; `show_ms=4,083782` |
| Frame 2 | Đã đọc, tracking và hiển thị; **`frame_total_ms=NaN`, `overhead_ms=NaN`**, render/write/show vẫn bằng 0 |
| Metadata | `n_frame_tong_ket=2`, nhưng trung bình `frame_total_ms` chỉ dùng **1** dòng hữu hạn vì `tb()` tự bỏ NaN |
| Wall ngoài frame | **25,339476 ms**, chứa cả thời gian frame cuối chưa được chốt |

Do đó phát biểu "mọi cột dùng cùng tập frame" không còn đúng trong ca này. Kết quả tracking không bị mất; lỗi nằm ở timing và thống kê của lần dừng sớm. Bốn run `v2` đã chạy hết không gặp nhánh này và không cần đo lại chỉ vì finding này.

Yêu cầu sửa gọn:

- Chốt timing và render/write/show của frame đã xử lý trên cả đường chạy tiếp lẫn dừng sớm. Dùng cơ chế hoàn tất được gọi chắc chắn; nếu dựa vào generator `finally`, quản lý việc đóng iterator tường minh thay vì chờ thu gom đối tượng.
- Không chữa bằng cách điền timing thiếu thành 0 hoặc âm thầm bỏ frame cuối khỏi một số cột.
- Regression qua `main()` cho `q` ở frame 1 và frame 2, có warm-up bằng 0. Kiểm số dòng, tính hữu hạn, stage hiển thị cuối frame, cùng tập thống kê và giữ nguyên output/cache đã xử lý.
- Trong ca dừng sớm, `frames_manifest.json` hiện vẫn liệt kê cả 3 ảnh nguồn. Có thể giữ manifest toàn nguồn, nhưng cần phân biệt rõ danh sách nguồn với phần đã xử lý; không gọi toàn bộ danh sách đó là "đã dùng" khi chỉ chạy 2 frame.

**4. [P2] Bảo toàn ranh giới đối số trong lệnh replay — phần còn lại của PC-F2**

Vị trí: `src/track_video_file.py`, `lenh_replay()` quanh dòng 363–379; `src/test_track_video_file.py`, helper `live_replay()` quanh dòng 252–254.

Lệnh replay được dựng bằng nối chuỗi, không quote giá trị. Test hiện dùng `.split()` và chỉ kiểm tên không có dấu cách.

Tái hiện với nguồn ảnh hợp lệ tên **`traffic clip`**:

1. Live exit 0, lưu đúng `traffic clip.npz`, 3 frame/3 hàng tracking.
2. Metadata sinh đoạn lệnh `--videos traffic clip`.
3. Thay hai placeholder `<split>`/`<ten>` như hướng dẫn rồi phân tích lệnh: argparse nhận **hai** video, `traffic` và `clip`.
4. Replay lỗi `FileNotFoundError: Chua co cache cho video traffic: .../traffic.npz`.

Kiểm đối chứng: chỉ truyền `traffic clip` thành **một** phần tử trong danh sách argv, giữ nguyên source/cấu hình/cache. Replay exit 0, **3/3 hàng**, bbox/conf lệch 0, ID khớp. Như vậy cache không hỏng; lỗi nằm ở cách biểu diễn lệnh.

Yêu cầu sửa gọn:

- Ưu tiên lưu `replay_argv` gồm từng đối số, chạy bằng `subprocess.run(argv, shell=False)` hoặc tương đương. Chuỗi để người dùng sao chép phải quote đúng shell được hỗ trợ; không áp quote Bash rồi mặc định đúng cho Windows.
- Bảo toàn tên sequence, đường dẫn YAML và các giá trị khác có dấu cách; không dùng `.split()` để biến chuỗi hiển thị thành argv.
- Thêm regression với tên nguồn và đường dẫn YAML có dấu cách, thực thi replay từ metadata và kiểm parity. Giữ các regression guard mới đã PASS.
- Đường chạy tạo cache hiện có không cần thay đổi để sửa việc truyền đối số này.

**5. Các bản sửa đã kiểm được độc lập**

Fixture I/O/detector kiểm chính sách; tracker NumPy, cache và hai entry point là mã thật:

| Ca | Kết quả |
|---|---|
| Bộ đọc 20 ms/frame, chạy hết | `decode_ms=20,108997`; tổng frame `21,222536` ms; timer bao phủ độ trễ đọc |
| Tổng stage + overhead | Sai số tối đa fixture **3,55e-15 ms** |
| Live → cache → lệnh replay thường | exit 0/0, **3/3 hàng**, bbox/conf lệch 0, ID khớp |
| `det_conf=0.5`, `conf_min=0.1` | Live **exit 2**, không tạo cache |
| Producer max_det 5, consumer 10 | Live **exit 2**, không tạo cache |
| Chạm trần max_det 1, mặc định | Live **exit 3**, không ghi metadata/kết quả tracking/cache |
| Cùng ca với `--allow-truncated` | Live/replay **exit 0/0**, flag có trong lệnh; parity lệch 0 |
| Hai track trùng hình học, đảo thứ tự dòng | Cả hai `mo_ho`, `dat=false`; không kết luận tùy ý là đổi liên kết |
| Chỉ đổi confidence trong file | FAIL đúng tolerance, nguyên nhân **CHƯA XÁC ĐỊNH** |
| Thứ tự số | `1,2,10`; `001,002,010`; `img_7,img_08,img_10` đúng |
| Tên mơ hồ/manifest | Khác tiền tố, không số, trùng số đều báo lỗi; manifest tường minh được tôn trọng |

**6. Bốn bảng thời gian `v2`: khớp dữ liệu bàn giao**

Đã đọc **4 × 1.470 = 5.880 dòng** timing/pipeline; kiểm thứ tự frame, số hàng tracking/track, toàn bộ stage hữu hạn, tập thống kê 1.445 frame sau warm-up và hash entry point. Sai số lớn nhất khi tính lại 10 trung bình mỗi run: **5,55e-17 ms**. Sai số stage + overhead so với tổng từng frame: **4,55e-13 ms**.

| Run | Tổng frame sau warm-up, ms/frame | FPS sau warm-up | FPS vòng lặp gồm warm-up | Wall ngoài các frame |
|---|---:|---:|---:|---:|
| CV | 23,100572 | 43,288971 | 40,303439 | 12,130301 ms |
| CV + video | 29,191549 | 34,256490 | 32,178416 | 21,586402 ms |
| EKF | 24,541552 | 40,747219 | 38,448292 | 11,757198 ms |
| EKF + video | 33,326412 | 30,006231 | 28,610914 | 25,993700 ms |

Hai cột FPS dùng phạm vi khác nhau đúng như metadata ghi. Tổng thời gian 25 frame warm-up lần lượt khoảng **3,081 / 3,479 / 2,759 / 3,196 giây**. Vì thế không có mâu thuẫn số học giữa hai cột FPS của bản `v2`. Mỗi cấu hình một lượt, chỉ dùng làm số mô tả của lượt đo đó.

Đối chiếu hai lượt độc lập có/không ghi video:

| Bộ lọc | Hàng/track mỗi bên | Max bbox | Max confidence | Kết quả |
|---|---:|---:|---:|---|
| CV | 5.842 / 72 | 0,0001220703 px | 1,1920929e-6 | Parity đầy đủ **FAIL**, ID/phân hoạch khớp |
| EKF | 5.845 / 70 | 0,0001220703 px | 6,5565109e-7 | Parity đầy đủ **PASS** |

Các giá trị này xác nhận báo cáo mới; không dùng chúng để quy nguyên nhân sai khác cho detector hay GPU.

**7. Tài liệu/provenance và giới hạn**

- Ngưỡng detector/consumer và ví dụ classes live/replay đã được chỉnh đúng. Bản mới phân biệt việc mask vùng ignore ở chế độ dataset với đưa GT vào association.
- Hash LF của `track_video_file.py` trong cả bốn metadata `v2` khớp source target: `41fa5a3d693027ff75f59021d0abced4e54897670ebb76ac6387870913251d14`. Các run ghi HEAD `8abb368` cùng working tree dirty, không giả vờ đã chạy trên một commit sạch. Đây là bằng chứng tốt hơn vòng trước cho nội dung entry point đã dùng.
- Log stdout thất bại cũ và ghi chú về log chi tiết đã mất được đóng gói. Không đủ dữ liệu để chứng minh độc lập nguyên nhân lỗi cũ; nên giữ mức "giả thuyết được chứng cứ gián tiếp hỗ trợ".
- Câu trong tin nhắn "403 ms giải thích phần lớn khoảng chênh wall cũ" không phù hợp với khoảng chênh 5–6 giây. Bản mới đo decode và tách khởi tạo là đúng, nhưng không nên quy toàn bộ hoặc phần lớn khoảng chênh cũ cho 403 ms. Tương tự, chênh thời gian detector giữa các lượt chưa đủ để khẳng định nguyên nhân là tải máy.
- Docstring đầu `nano_parity.py` còn nói confidence tham gia tập ứng viên, trong khi `_ung_vien_mot_frame()` chỉ dùng hình học và chủ động tách confidence. Sửa câu này để mô tả thống nhất; không cần đổi thuật toán vì ghi chú đó.
- `wall_ghi_s` hiện dừng trước khi ghi `run_meta.json`; tên/mô tả nên phản ánh đúng mốc đo, tránh ghi là đã gồm toàn bộ ghi metadata. Không làm sai các bảng timing mỗi frame đã đối chiếu.

Reviewer chạy trên Python 3.12, không có OpenCV, torch, Ultralytics, checkpoint hoặc video gốc. **`test_nano_parity` 20/20 được chạy độc lập. Full suite 30/30 và `test_track_video_file` 23/23 (gồm fixture và smoke YOLO/video thật) chỉ được xác nhận từ log/CSV bàn giao**, không được mô tả là reviewer đã chạy lại trên GPU. Fixture thay I/O/detector để kiểm timer, chính sách và tham số; không thay thế smoke GUI/codec thật của tác giả.

**8. Phạm vi để đóng vòng này**

Chỉ cần sửa hai ca ở mục 3–4 và các câu tài liệu liên quan; chạy regression dừng sớm, replay tên có dấu cách và smoke ngắn phù hợp. Giữ bốn run `v2`, kết quả nghiên cứu và lock. Không cần mở thêm tuning, train, benchmark dài hoặc setup Nano.

Giữ base review `b623465` cho đến khi hai ca này được chốt. Gói tiếp theo vẫn có thể là delta tích lũy; chỉ cần thêm code/test, log và dữ liệu nhỏ chứng minh hai bản sửa. Sau đó đủ cơ sở xét chốt mốc PC và chuyển sang cách sử dụng/launcher.
