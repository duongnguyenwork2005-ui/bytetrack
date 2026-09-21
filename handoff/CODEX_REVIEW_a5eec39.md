# Review vòng 13 — launcher PC

**Kết luận: PASS WITH LIMITATIONS.** Đóng L-F1, L-F2, L-F3, L-F4 của review `2d1a8df`, trong phạm vi launcher PC và các đường chạy đã kiểm. Không tìm thấy lỗi chặn mới trong phạm vi này. Có thể dùng `a5eec39` làm snapshot PC để bắt đầu chuyển sang Jetson Nano. Kết luận này **không phải nghiệm thu Nano**.

Ngày review: 2026-09-21. Review độc lập trên bản đính kèm; không truy cập trực tiếp repository Windows, không sửa mã sản phẩm, commit, push hoặc index của người dùng.

## 1. Snapshot và tính toàn vẹn gói

| Thuộc tính | Đã kiểm |
| --- | --- |
| Base | `074ceca1ba36777cabc02a54d6699e890b19c51b` |
| Target | `a5eec3963d960fb817b3d83d2b30ec3c12a01795` |
| ZIP | `review_delta_latest.zip`, 1.918.799 byte, 101 thành viên |
| SHA256 | `1c6b133645d1c4543059d94920290640f3d2a18ab80f25b1f233d157e79b185e` |
| Kiểm ZIP | `testzip()` không báo lỗi |
| Payload riêng | 95/95 khớp SHA256 trong manifest |
| Dedup | 7 đường dẫn được khôi phục qua payload trùng nội dung đã khai báo |
| Patch | 102 đường dẫn; `git apply --check` và `git apply` đều exit 0 |
| Cây sau patch | 102/102 đường dẫn trong manifest khớp target; kiểm lại sau các phép thử vẫn khớp |

Không có đường dẫn thuộc nhóm riêng bị cấm trong patch: `.gitignore`, `src/demo/**`, `live.bat`, `outputs/**`; không đưa dataset/weights vào patch. Không xem số lượng đường dẫn này là bằng chứng đã tái kiểm toàn bộ repository ngoài phạm vi gói.

Lưu ý tái dựng base: bản lưu phía reviewer có `src/test_review_delta.py` cũ hơn preimage của gói. Đã khôi phục preimage từ payload target bằng reverse patch, rồi xác minh **đúng SHA256 base** `9cd860bf3a9ba334f477df7301d3fcf0cb2652a44df1bca351e1a114ba7286a3` trước khi áp patch xuôi. Không sửa snapshot base cũ. Nhờ payload đầy đủ ở vòng này, reviewer đã chạy được chính bản test packager 14 phép kiểm hiện tại.

## 2. Môi trường và ranh giới kiểm chứng độc lập

Reviewer chạy Linux, Python 3.12, NumPy/SciPy/Pandas và backend `BYTETRACK_BACKEND=np`; có Tcl/Tk 9.0 nhưng không có display server, OpenCV, PyTorch/Ultralytics hoặc CUDA để chạy detector thật.

Phép kiểm tích hợp dùng **controller thật, subprocess thật, entry point `track_video_file.main()` thật, tracker CV/EKF thật, ghi cache thật và replay CLI thật**. Bộ đọc ảnh, YOLO và giao diện cv2 là test double được khai báo rõ. Kiểm callback launcher dùng phương thức thật của `LauncherApp`, widget/messagebox double và vòng sự kiện Tcl thật. Kiểm thanh tiến độ chạy thủ tục timer `ttk::progressbar` thật của runtime Tcl/Tk, không mở cửa sổ Tk.

Do đó các kết quả dưới đây kiểm được logic điều phối/lưu dữ liệu/replay; không thay thế smoke Windows, GPU, OpenCV GUI hoặc thao tác tay trên PC người dùng.

## 3. Đóng bốn finding

| Finding | Bằng chứng độc lập sau sửa | Kết luận |
| --- | --- | --- |
| **L-F1 — SPACE → Dừng** | Cho entry point thật vào vòng pause tại frame 2 rồi gọi `request_stop()`. Worker tự thoát exit 0, trạng thái `DA_DUNG`, không terminate; khoảng 0,133 s từ yêu cầu dừng đến kết thúc trong fixture. Ghi đủ metadata/data/timing/pipeline, đúng 2 frame; lý do `stop-file`; replay bbox/conf/ID khớp, sai lệch 0. | Đóng |
| **L-F2 — timer thanh tiến độ** | Chuyển khởi tạo indeterminate sang 10/100: có gọi `stop()`, không còn timer autoincrement; bơm vòng sự kiện khi worker giữ nguyên frame vẫn là 10. Cập nhật worker lên 37 thì thanh thành 37. | Đóng |
| **L-F3 — FPS cuối lượt** | Cố tình làm ghi cache cuối lượt chậm khoảng 1,008 s. `progress.fps_vong_lap_tam` bằng chính xác `run_meta.thoi_gian.fps_vong_lap`: khoảng 58,696 fps; `fps_toan_luot` riêng khoảng 3,589 fps. Số minh họa từ fixture, không phải benchmark hệ thống. | Đóng |
| **L-F4 — lỗi nhập số** | Gọi callback `_bat_dau()` thật với `abc` cho cả 9 trường số. Mỗi ca hiện đúng một thông báo chứa tên ô; không tạo worker, không có exception thoát callback. `seed` và `max_frames` trả `CauHinhSai` đúng loại. | Đóng |

### Đóng cửa sổ khi đang SPACE

Kiểm thêm đường `_dong()` → `_cho_dong()` với `_tick()` và controller thật:

- Callback đóng trả về sau khoảng 0,0007 s, không đợi đồng bộ worker.
- Vòng sự kiện tiếp tục xử lý được 21 heartbeat trước khi đóng, khoảng 0,206 s trong fixture.
- Worker tự kết thúc `DA_DUNG`, không terminate, không còn tiến trình chạy; đóng sau khi worker hoàn tất.
- Lưu đúng 2 frame; replay khớp tuyệt đối; lý do `stop-file`.
- Ca SPACE → `q` riêng vẫn lưu đúng 2 frame, replay khớp và giữ lý do `phim q`, không bị đổi thành `stop-file`.

Các thời gian này chỉ chứng minh đặc tính không chặn trong fixture, không phải latency cam kết trên mọi PC hoặc Nano.

## 4. Các phép kiểm độc lập bổ sung

| Phép kiểm | Kết quả |
| --- | --- |
| `python src/test_review_delta.py` trên target tái dựng | **14/14 PASS**, exit 0; bao gồm `--support` |
| Launcher CV kết thúc bình thường | 6 frame, `HOAN_TAT`, exit 0; replay khớp bbox/conf/ID, lệch 0 |
| Launcher EKF kết thúc bình thường | 6 frame, `HOAN_TAT`, exit 0; replay khớp bbox/conf/ID, lệch 0 |
| Dừng giữa lượt không pause | `DA_DUNG`, exit 0; 11 frame trong lượt kiểm này; replay khớp, lệch 0 |
| Bắt đầu lần hai khi worker đang chạy | Bị chặn bằng `DangChay` |
| Cấu hình detection không tương thích | Worker exit 2, launcher báo `LOI` và giữ lý do lỗi |
| Đường dẫn nguồn/checkpoint/YAML có dấu cách | Dùng argv có cấu trúc; replay qua `subprocess` thành công |
| Timing của các fixture thành công | Các cột số hữu hạn, không NaN |

Replay được chạy bằng chính `replay_argv` đã sinh trong metadata, chỉ thay các placeholder được công bố và interpreter của reviewer; không tự viết một lệnh replay khác để tránh lỗi.

## 5. Đối chiếu bằng chứng do Claude bàn giao

### Full suite

Đã đọc log `handoff/logs/review13/tests_20260921-1231.log` và đối chiếu `tests_summary.csv`: **31 suite PASS, exit 0**, log ghi HEAD **`56ce2c1`**. Các bộ liên quan ghi `test_pc_launcher` 17/17, `test_track_video_file` 27/27 và `test_review_delta` 14/14.

Đây là **xác minh log bàn giao**, không phải tuyên bố reviewer tự chạy đủ 31 suite ở `a5eec39`. Target giao nộp chứa thêm tài liệu/log/ảnh sau HEAD của lượt test. Mã packager hiện tại đã được chạy độc lập 14/14 như mục 4.

Hai log thất bại trước đó vẫn tồn tại trong `lich_su_that_bai/`, mỗi log ghi `test_pc_launcher: FAIL (exit 1)`; có ghi chú phân biệt lỗi in cp1252 với race ở helper chờ. Không xóa bỏ lịch sử thất bại và không tính hai lượt đó thành PASS.

### Smoke Windows/GPU

Log smoke công bố đúng phạm vi: YOLO/GPU thật, Tk/cv2 thật; SPACE gửi tự động, nút gọi `invoke()`, đóng qua callback. Log ghi HEAD `2d1a8df` với working tree đã sửa lúc smoke; không gọi đây là lượt chạy trên một checkout sạch của commit target.

Reviewer đã tính lại từ **4 bộ artifact smoke vòng 13**:

| Run | Số frame | FPS cuối bằng metadata | Lý do dừng |
| --- | ---: | --- | --- |
| CV `120531` | 150 | Có | Kết thúc bình thường |
| EKF `120544` | 150 | Có | Kết thúc bình thường |
| EKF `120600` | 148 | Có | `stop-file` |
| CV `120619` | 42 | Có | `stop-file` |

Cả bốn có số dòng pipeline = `n_frame_xu_ly` = số frame đã xử lý trong manifest = frame cuối progress; số liệu timing hữu hạn. Tổng stage + overhead khớp `frame_total` với dư số học lớn nhất dưới `7e-13 ms`.

Các phát biểu về replay GPU chính xác, thao tác Windows và thời gian dừng 1,16/1,32 s được hỗ trợ bởi log bàn giao; không được đo lại bằng GPU tại reviewer. Chưa có smoke phím/chuột thao tác tay, đúng giới hạn Claude đã nêu.

## 6. Bảo toàn kết quả và mã lõi

Đối chiếu byte với snapshot base mà reviewer hiện có:

- `src/run_tracking.py`, `src/det_cache.py`, `src/tracker_ctrv.py`, `src/nano_parity.py`: không đổi.
- 24 file thuộc bốn run `results/track_video/v2_*`: không đổi.
- 28 file `results/phase2` và 52 file `results/trackeval` kế thừa có mặt trong snapshot: không đổi.

Đây là đối chiếu trên **các file thực sự có trong snapshot reviewer**, không phải audit toàn bộ kết quả/GT/cache trên máy Windows. Không có bằng chứng từ vòng này yêu cầu chạy lại UA-DETRAC/BDD hay tuning lại cấu hình nghiên cứu.

## 7. Giới hạn còn mở — không chặn nghiệm thu launcher PC

1. Reviewer không chạy lại Windows/GPU/Tk có display; mức kiểm độc lập và mức đối chiếu log đã tách riêng.
2. Chưa smoke thao tác tay, camera, VFR, bản exe. `fps_vong_lap` có tính thời gian SPACE vì pause nằm trong vòng lặp; không dùng số đó làm benchmark tốc độ tracker thuần.
3. Nano F1/F3 đã hoãn vẫn chưa được nghiệm thu. Backend NumPy chạy trên PC không chứng minh stack Python/dependency/GPU trên Nano tương thích.
4. Chưa có số đo hoặc kết quả parity trên thiết bị Nano thật trong gói này.

## 8. Hành động tiếp theo

**Không cần mở thêm vòng sửa launcher trước khi bắt đầu công việc Nano.** Giữ smoke tay ngắn làm kiểm tra sử dụng khi thuận tiện; nếu phát sinh lỗi thì ghi riêng với cách tái hiện, không phủ nhận các ca đã kiểm.

Đề nghị Claude trên PC:

1. Lưu báo cáo này; ghi nhận launcher PC `a5eec39` là **PASS WITH LIMITATIONS** trong đúng phạm vi trên.
2. Lưu trữ nguyên gói đã review theo kích thước/SHA256 ở mục 1; sau đó có thể chuyển review base sang `a5eec39`. Reviewer xác nhận đã đọc snapshot này; không tự chuyển base tới commit mới hơn chưa review.
3. Chuẩn bị **gói chuyển Nano độc lập**, không dùng delta review làm gói cài đặt. Gói cần mã nguồn/config/vendor và giấy phép, đầu vào ngắn, detection cache và kết quả CV/EKF tham chiếu, manifest hash/commit, hướng dẫn lệnh kiểm chứng; bổ sung artifact detector phù hợp sau khi xác minh stack trên Nano. Không chép môi trường Windows, toàn bộ dataset, file riêng hoặc kết quả lớn không cần thiết.
4. Tách triển khai thành hai mốc: **tracker từ cùng cache → parity CV/EKF**, rồi **detector + tracker end-to-end**. Chỉ benchmark thiết bị sau khi đúng chức năng. Không đổi OS/Python hệ thống hoặc cài toàn bộ requirements PC một cách tự động.

Thông tin thiết bị người dùng đã cung cấp: Nano 4 GB, aarch64, Ubuntu 18.04.6/L4T R32.7.5, Python 3.6.9, `torch 1.10.0` báo CUDA khả dụng; SSH từ Windows đã đăng nhập được. Việc `nvcc`/`trtexec` không được tìm thấy trong PATH chưa đủ để kết luận CUDA/TensorRT chưa cài. Cần kiểm package và vị trí công cụ trước khi chọn cách triển khai.

## 9. Dấu vết review độc lập

Các phép thử được chạy trong thư mục `review_a5eec39/` của workspace reviewer, không trong repository Windows:

- `verify_package.py` → `package_verification.json`.
- `probe_launcher.py` với `worker_probe.py` → `launcher_probes.json` và log worker.
- `probe_gui.py` → `gui_probes.json`.
- `audit_artifacts.py` → `artifact_audit.json`.
- `python src/test_review_delta.py` → stdout 14/14, exit 0.

Scripts này là fixture kiểm tra của reviewer; không được đưa vào mã sản phẩm hoặc xem như thay thế test dự án. Kết quả và phạm vi cần cho quyết định nghiệm thu đã ghi đầy đủ trong báo cáo này.
