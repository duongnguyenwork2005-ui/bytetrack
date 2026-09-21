# Review launcher PC — target `2d1a8df`

Ngày review: 2026-09-21. Nhánh tác giả báo cáo: `phase2-mixed-traffic`.

**Kết luận: REQUEST CHANGES cho mốc launcher.** Có ba lỗi P2 trong điều phối/hiển thị và một lỗi P3 trong thông báo đầu vào. Đường chạy thường và Dừng khi video đang chạy đã qua kiểm chứng độc lập; chưa nghiệm thu launcher vì Dừng lúc đang tạm dừng bằng SPACE có thể làm mất toàn bộ kết quả chưa ghi của lượt đó.

Kết luận này không thu hồi nghiệm thu hệ thống PC tại `074ceca`, không mở lại tuning hoặc đánh giá UA-DETRAC/BDD. Giữ base review ở `074ceca` trong vòng sửa tiếp theo. Nano F1/F3 tiếp tục nằm ngoài phạm vi.

## 1. Gói và phạm vi đã xác minh

| Mục | Kết quả độc lập |
|---|---|
| ZIP | 1.069.119 byte; 54 thành viên; `testzip()` không lỗi |
| SHA256 ZIP | `98973473c355f25d18f1b4454c46c77f06b55e28f3e6c7a22ae2a4aeb7695334` |
| Base | `074ceca1ba36777cabc02a54d6699e890b19c51b` |
| Target | `2d1a8dfd857a7d0d0ff04f1d5dfa16f242c5cf85` |
| Dựng mã | 9 file có preimage khớp hash base; `git apply --check` và `git apply` đều exit 0 |
| Nội dung target | 48 payload + 2 file dedup; 50/50 đường dẫn target khớp hash sau áp patch và vẫn khớp sau review |
| PNG | 5 ảnh trong binary patch tái dựng đúng hash |
| Phạm vi patch | Không có đường dẫn riêng bị cấm; không thay GT, lock hay kết quả thí nghiệm UA-DETRAC/BDD |
| Mã dùng chung | `run_tracking.py`, `det_cache.py`, `tracker_ctrv.py`, `nano_parity.py` không đổi so với base |
| Kết quả PC `v2` có trong cây nền | 24 file đã đối chiếu, không file nào đổi |

Đã đọc `REVIEW_BRIEF.md`, `LAUNCHER_PC.md`, `REVIEW12_CHI_TIET.md`, mã launcher/worker/test, thay đổi đóng gói và log bàn giao. Đã xem ảnh cửa sổ ở trạng thái dừng. Review không sửa mã sản phẩm, không commit/push hay thao tác trên repository Windows.

## 2. Các finding cần xử lý

### L-F1 — P2: Dừng lúc đang SPACE không tới được nhánh stop-file, dẫn đến terminate và mất kết quả chưa ghi

**Vị trí:** `src/track_video_file.py:586–597, 614–625`; timeout trong `src/pc_launcher.py:273–276`.

`on_frame()` gọi `_ve_hien_thi()` trước khi kiểm tra `STOP.flag`. Khi SPACE được bấm, `_ve_hien_thi()` nằm trong vòng `waitKey(50)` và chỉ chờ SPACE hoặc q; vòng này không đọc stop-file. Vì vậy nút Dừng của launcher không có tác dụng khi đang pause. Hết timeout, controller coi worker không phản hồi và terminate. Tracking/cache/timing được ghi sau vòng lặp nên chưa được lưu.

**Tái hiện độc lập:** chạy `LauncherController` và worker subprocess thật, entry point và tracker NumPy thật; giả lập detector, đọc ảnh và các hàm GUI cv2. Tại frame 2, `waitKey(1)` trả SPACE; các lần `waitKey(50)` sau trả 255. Đợi marker xác nhận đã vào vòng pause rồi gọi `request_stop()`. Đặt `timeout_dung=0.3` chỉ để phép kiểm chạy nhanh, không sửa mã sản phẩm.

| Quan sát | Kết quả |
|---|---|
| Trạng thái cuối | `LOI` |
| `da_terminate` | `True` |
| Exit trên môi trường Linux review | `-15` |
| Thời gian từ yêu cầu Dừng đến kết thúc | 0,356 s với timeout fixture 0,3 s |
| `run_meta.json`, `data/`, `pipeline/`, `timing/` | Đều chưa tồn tại |
| Detection cache của lượt | Chưa được tạo |

Đường đóng cửa sổ gọi `_dong()` → `shutdown()` cũng tái hiện kết quả terminate tương tự. Timeout mặc định trong mã là 30 s; review không đợi 30 s để suy ra cùng lỗi điều khiển.

**Sửa tối thiểu:** kiểm yêu cầu Dừng ngay trong vòng pause; thoát qua cùng đường chốt frame/generator/timing và ghi kết quả đã kiểm chứng. Giữ đúng `ly_do_dung="stop-file"`, không gán nhầm thành phím q. Không cần thêm thuật toán tracking hay thay timeout để che lỗi.

**Regression cần có:** đang SPACE → nút Dừng, và đang SPACE → đóng launcher; worker tự kết thúc không terminate, timing frame cuối hữu hạn, manifest chỉ ghi frame thực sự xử lý, cache đủ và replay tương đương. Giữ ca worker thật sự không phản hồi để kiểm fallback timeout.

### L-F2 — P2: thanh tiến trình tự tăng sau khi chuyển sang đếm frame thật

**Vị trí:** `src/pc_launcher.py:519–525`.

Ở trạng thái khởi tạo, `_tick()` gọi `pb.start(80)` để chạy hoạt ảnh. Khi có tổng frame, code chỉ `configure(mode="determinate", maximum=tot, value=n)` mà không `stop()` bộ định thời. Ttk tiếp tục gọi `step()` giữa các lượt poll, nên thanh có thể tăng dù worker không xử lý thêm frame nào. Nhãn frame vẫn đúng nhưng thanh không còn biểu diễn đúng số frame đã xử lý.

**Tái hiện độc lập:** gọi chính `LauncherApp._tick()` với widget double; dùng thủ tục timer `ttk::progressbar` thật của Tcl/Tk 9.0 cài trong môi trường review. Cho trạng thái khởi tạo, sau đó chuyển sang `DANG_CHAY`, `frame=10`, `total=100`. Không phát sinh cập nhật frame mới; chạy vòng sự kiện Tcl sau 100 ms:

- Trước timer: `mode=determinate`, `value=10`.
- Sau timer: `value=11`, trong khi worker vẫn ở frame 10.
- Số lần gọi `stop()` tại bước chuyển: 0; timer vẫn tồn tại.

Đây là kiểm callback và cơ chế timer, không phải smoke cửa sổ Windows thật.

**Sửa tối thiểu:** dừng autoincrement khi chuyển sang determinate, sau đó đặt lại maximum/value từ dữ liệu worker. Lưu ý `stop()` có thể đặt value về 0 nên phải đặt value sau nó.

**Regression:** giữ nguyên frame worker sau khi chuyển mode và bơm event loop; thanh phải đứng yên ở đúng giá trị. Sau đó tăng frame thật và kiểm thanh cập nhật đúng.

### L-F3 — P2: FPS cuối lượt mang nhãn "vòng lặp" nhưng cộng cả ghi kết quả sau vòng lặp

**Vị trí:** `src/track_video_file.py:475–490, 784–788`; hiển thị tại `src/pc_launcher.py:526–529`.

`ghi_tien_trinh()` luôn tính thời gian bằng `now - _tt["t0"]`. Lần gọi cuối nằm sau ghi MOT/CSV/cache/manifest/metadata, nên `progress.json.t_vong_lap_s` và `fps_vong_lap_tam` không cùng phạm vi với `run_meta.thoi_gian.wall_vong_lap_s`/`fps_vong_lap`. GUI tiếp tục gọi số này là FPS vòng lặp, cả khi đã hoàn tất.

**Đã thấy ngay trong gói bàn giao:**

| Run | FPS vòng lặp trong `run_meta` | FPS cuối trong `progress` |
|---|---:|---:|
| Smoke CV, 150 frame | 27,6466 | 27,0649 |
| Smoke EKF, 150 frame | 34,4471 | 33,5814 |
| GUI EKF, dừng tại 153 frame | 28,0618 | 27,4553 |

**Kiểm độ nhạy độc lập:** fixture 4 frame, chèn 1 s trễ chỉ quanh `det_cache.save_video()` sau tracking. Không sửa cách đo vòng lặp:

- Vòng lặp thật: 0,06831 s, `run_meta` = **58,5533 fps**.
- Thời gian ghi cuối lượt: 1,01809 s.
- Progress cuối: 1,11305 s, **3,5937 fps**, vẫn mang nhãn vòng lặp.
- Tracking → cache → replay vẫn khớp hoàn toàn.

**Sửa tối thiểu:** khi phát trạng thái cuối, dùng số frame và `wall_vong_lap` đã chốt. Nếu muốn thêm tốc độ toàn lượt gồm ghi cuối, đặt tên và phạm vi riêng. Không sửa các kết quả lịch sử `v2` để làm chúng khớp một cách hồi tố.

**Regression:** tạo độ trễ sau vòng lặp; FPS vòng lặp cuối trong progress và metadata vẫn bằng nhau, không phụ thuộc độ trễ đó.

### L-F4 — P3: nhập sai số frame/seed không có thông báo lỗi trên GUI

**Vị trí:** `src/pc_launcher.py:159–160, 171–172, 478–486`.

`int(cfg["max_frames"])` và `int(cfg["seed"])` có thể phát `ValueError`, nhưng `_bat_dau()` chỉ bắt `CauHinhSai`, `DangChay` và `OSError`. Nhập `abc` vào một trong hai ô rồi Bắt đầu làm lỗi callback thoát ra ngoài. Người dùng không thấy hộp thoại giải thích, trạng thái vẫn SẴN SÀNG.

**Tái hiện:** gọi chính `_bat_dau()` với controller thật và form/messagebox double. Cả hai ô cho `ValueError` không được bắt, 0 hộp thoại, không tạo worker. Phép kiểm đầu vào hiện có chưa bao phủ hai trường này.

**Sửa:** chuyển lỗi parse/miền giá trị của các trường này thành `CauHinhSai` có tên ô và hướng dẫn rõ. Kiểm qua callback GUI, không chỉ kiểm riêng `build_argv()`.

## 3. Những phần đã đạt kiểm độc lập

Các worker dưới đây là subprocess thật, chạy entry point, vòng lặp và tracker NumPy thật. Detector, đọc ảnh và GUI cv2 là test double được khai báo rõ; không gọi đây là smoke YOLO/GPU.

| Luồng | Kết quả |
|---|---|
| CV chạy hết 6 frame, nguồn/checkpoint có dấu cách | `HOAN_TAT`, exit 0; replay subprocess 6/6 hàng, bbox/conf/ID lệch 0 |
| EKF chạy hết 6 frame, cả đường dẫn YAML có dấu cách | `HOAN_TAT`, exit 0; replay 6/6 hàng, lệch 0 |
| Dừng khi đang chạy bình thường | `DA_DUNG`, exit 0; chốt 11 frame, `ly_do_dung=stop-file`; timing hữu hạn; replay 11/11 hàng lệch 0 |
| Bắt đầu lần hai khi worker còn sống | Bị chặn bằng `DangChay` |
| Cấu hình detector/cache không tương thích | Worker exit 2; launcher báo `LOI`, không báo hoàn tất |
| Lượt có ghi cache chậm | Tracking/replay vẫn đúng; phát hiện lỗi phạm vi FPS như L-F3 |
| Assertion S1 trong test đường dẫn có dấu cách | Đã đổi sang roundtrip theo nền tảng; không còn bắt buộc dấu nháy kép Windows |

Đường đóng cửa sổ hiện gọi `shutdown()` đồng bộ trong callback Tk. Fixture timeout 0,3 s giữ callback khoảng 0,305 s; với timeout mặc định, GUI có thể chờ trên luồng Tk đến 30 s. Khi chỉnh luồng Dừng, nên dùng `after()`/poll để chờ worker mà vẫn cập nhật trạng thái, chỉ destroy sau khi xác nhận tiến trình đã kết thúc. Đây là ghi chú về luồng chờ, không phải yêu cầu thêm một framework GUI.

## 4. Bằng chứng bàn giao và giới hạn xác minh

- `tests_summary.csv` có 31/31 PASS, toàn bộ exit code 0. Log chỉ rõ HEAD **`512712f`**. Đây là **đối chiếu log tác giả**, không phải reviewer đã tự chạy đủ 31 suite.
- `2d1a8df` chỉ bổ sung `--binary` cho packager sau lượt full suite. Kiểm độc lập patch có 5 PNG áp sạch và hash khớp là bằng chứng trực tiếp cho thay đổi này; không cần chạy lại thí nghiệm tracking vì lý do đó.
- Log ở `512712f` có `test_review_delta` 13/13. File test này không được gửi lại trong delta hiện tại; bản lưu trong môi trường reviewer còn là phiên bản cũ 10 phép kiểm. Một lần chạy bản cũ cho 10/10 **không được tính** là xác nhận độc lập 13/13 của target, cũng không phải bằng chứng tác giả làm mất test. Không dùng sai khác số test này làm finding. Nếu cần reviewer chạy đủ test đóng gói mới ở vòng sau, chỉ gửi bổ sung đúng file test hiện hành.
- Log smoke tác giả ghi CV/EKF trên GPU thật, 150 frame, replay 394/394 và 395/395 hàng, lệch 0. Ảnh và output GUI phù hợp lượt dừng 153/1470. Chưa có smoke chuột tay theo chính báo cáo tác giả.
- Môi trường review có NumPy/SciPy/Tk nhưng không có OpenCV, torch, Ultralytics, GPU và display server. Không chạy lại YOLO/GPU hay cửa sổ Tk Windows thật. Các nhận định về callback GUI dựa trên mã thật với widget double; timer progress dùng Tcl/Tk thật như đã nêu.

## 5. Phạm vi sửa tiếp theo

1. Tái hiện rồi sửa L-F1…L-F4 trên đúng target; ưu tiên Dừng khi SPACE vì ảnh hưởng lưu kết quả.
2. Bổ sung regression đúng các tình huống trên; giữ ca hoàn tất, Dừng thường, đường dẫn có dấu cách, worker lỗi và fallback timeout. Khi kiểm đóng cửa sổ, xác nhận không còn worker sau khi GUI đóng.
3. Chạy các suite liên quan. Vì có thể sửa entry point dùng chung, chạy full suite trên trạng thái mã ổn định rồi ghi rõ commit; không commit trong lúc suite đang đo fingerprint.
4. Chạy một smoke ngắn trên Windows cho SPACE → Dừng với lưu cache, rồi replay. Phân biệt thao tác phím tự động với thao tác tay; không cần chạy lại UA-DETRAC/BDD hoặc mở tuning.
5. Cập nhật brief ngắn, ghi đúng phạm vi FPS và trạng thái test. Gửi delta từ **`074ceca`**, giữ file riêng ngoài stage/gói, commit cục bộ không push/merge. Chưa chuyển review base sang `2d1a8df`.

Sau khi các điểm này đóng, bước tiếp theo mới là nghiệm thu launcher và chọn bộ video giao thông thực tế để kiểm hệ thống/demo. Không có finding ở vòng này buộc thay motion model, detector hay kết quả nghiên cứu đã khóa.
