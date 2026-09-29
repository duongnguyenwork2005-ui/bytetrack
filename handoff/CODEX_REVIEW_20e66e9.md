# Review vòng 25 — 20e66e9

**Kết luận: REQUEST CHANGES cho phần giao diện, với hai finding P2 bên dưới. Bằng chứng tracking và các lượt hoàn tất vẫn đạt trong phạm vi đã kiểm.**

Hệ thống đã có launcher và đường hiển thị trực tiếp trên Nano. Tôi không yêu cầu làm lại kiến trúc hoặc chạy lại nghiên cứu UA-DETRAC/BDD. Hai việc cần sửa là đếm frame hiển thị sau khi đóng cửa sổ và quản lý tiến trình hộp chọn file khi đóng launcher.

Tôi review snapshot đính kèm, không SSH vào Nano. Kiểm thử độc lập dùng Linux/Python 3.12.14/OpenCV 4.11.0; các số trên thiết bị được đối chiếu từ bằng chứng bàn giao. Camera nằm ngoài phạm vi.

**1. N25-F1 — P2: frame không còn được trình lên cửa sổ vẫn được tính là đã hiển thị**

Vị trí: `src/nano_hien_thi.py`, `HienThi._ve()` và `_su_kien()`.

Sau sự kiện đóng cửa sổ, `_cua_so_da_dong=True`. `_ve()` bỏ qua `cv2.imshow()` ở trạng thái này, nhưng vẫn thêm timestamp vào `t_hien_thi`, tăng `n_hien_thi` và cập nhật `frame_hien_thi_cuoi`. Frame đang bay hoặc đang chờ vẽ có thể tới sau sự kiện đóng, trước khi stop-file làm toàn luồng dừng.

Tái hiện độc lập bằng chính lớp `HienThi`, renderer thật và stub cho các thao tác cửa sổ HighGUI:

| Thao tác / quan sát | Kết quả trên 20e66e9 |
|:--|:--|
| Hiển thị frame 1, sau đó mô phỏng cửa sổ bị đóng | Gửi yêu cầu dừng `dong_cua_so` |
| Kết quả frame 2 đang bay tới sau đó | Không gọi `imshow()` cho frame 2 |
| Tổng số lần gọi `imshow()` | **1** |
| `n_hien_thi` / `frame_hien_thi_cuoi` | **2 / 2** |
| `fps_hien_thi` | Có giá trị dù chỉ trình một frame lên cửa sổ |

Ảnh hưởng: metadata, tóm tắt launcher và FPS hiển thị của lượt dừng bằng đóng cửa sổ có thể báo thừa. Đây không phải bằng chứng sai MOT hoặc sai FPS xử lý. Các lượt benchmark chạy tới hết, cửa sổ không bị đóng giữa chừng, không bị finding này phủ định.

Yêu cầu sửa: ở chế độ hiển thị thật, chỉ tăng bộ đếm/timestamp đã hiển thị khi thực sự trình frame lên cửa sổ. Phân biệt frame bị bỏ do đóng cửa sổ với frame đã hiển thị và frame vẫn được tracking xử lý. Giữ rõ ngữ nghĩa riêng của chế độ ẩn dùng cho test. Thêm regression có frame đang bay sau khi đóng; xác nhận MOT phần đã xử lý vẫn đầy đủ.

**2. N25-F2 — P2: đóng launcher khi hộp chọn file mở để lại tiến trình chọn file**

Vị trí: `src/nano_launcher.py`, `NanoLauncherApp._chon()`, `_dong()` và `_cho_dong()`.

Nhánh Zenity chạy `subprocess.run()` trong daemon thread, nhưng launcher không giữ handle/PID của tiến trình chọn file. Khi tracker không chạy, `_dong()` gọi `destroy()` ngay. Khi tracker đang chạy, đường kết thúc cũng chỉ đợi tracker. Không có bước đóng/reap Zenity hoặc hủy callback chờ kết quả chọn file.

Tái hiện: gọi chính `_chon("video")` rồi `_dong()`, dùng một tiến trình thật thay Zenity đang chờ người dùng chọn file. Các widget Tk được thay bằng stub vì môi trường review không có màn hình.

```text
launcher_destroyed = true
picker_alive_after_close = true
scheduled_callbacks = 1
```

Script tái hiện tự kết thúc tiến trình thử nghiệm sau khi ghi kết quả. Đây là kiểm vòng đời subprocess qua callback thật, không phải kiểm hộp Zenity/chuột thật trên Nano.

Ảnh hưởng: thao tác sử dụng thông thường “Chọn video/thư mục → đóng launcher” có thể để lại cửa sổ và tiến trình độc lập, trái với mục tiêu thoát sạch. Các test hiện có chỉ kiểm GUI thiếu engine và các đường chạy tracker; chưa bao phủ vòng đời hộp chọn file.

Yêu cầu sửa: quản lý tiến trình chọn file bằng handle cụ thể, đóng/reap khi launcher thoát, hủy các callback chờ và ngăn cập nhật widget sau khi đã đóng. Vẫn giữ giao diện phản hồi trong lúc dừng tracker. Thêm ca đóng launcher khi picker mở, ca hủy chọn file rồi mở lại, và kiểm ngắn với Zenity thật trên Nano.

**3. Script tái hiện bàn giao**

File đi kèm: `CODEX_REPRO_20e66e9_ui.py`.

```bash
python handoff/CODEX_REPRO_20e66e9_ui.py --repo .
```

Trên mã `20e66e9`, script trả **exit 1**, cả hai finding đều tái hiện. Script dùng NumPy/OpenCV, không cần TensorRT/Nano; không sửa repository. Phần picker yêu cầu POSIX. Phạm vi giả lập được ghi rõ trong docstring; bổ sung test GUI thật trên thiết bị khi sửa, không dùng script này thay cho toàn bộ nghiệm thu HDMI.

**4. Phần đã xác minh đạt**

Gói review:

- Base: `f86732b71399f11b1392b54c2558133772d41e26`.
- Target: `20e66e96e45504dafae4b5f70f909cfbee81c20e`.
- ZIP: **9.265.303 byte**, 445 thành viên, `testzip` không lỗi.
- SHA256: `517f092ccb8ddf180b81029e6335a8c30898aaf24cea4507fa610381a89023d1`.
- Payload **439/439** khớp hash và số byte. Patch áp lên base thành công; **507/507** đường dẫn thay đổi khớp target. Không thấy đường dẫn riêng bị cấm hoặc video/engine/weights/NPZ/ZIP khác trong tập thay đổi được đóng gói.

Đối chiếu bằng chứng Nano:

- **14/14** cặp live/replay có MOT và CSV lớp giống từng byte.
- Pipeline CSV của cả 14 lượt có đủ frame từ 1 đến N, không lặp; mọi frame được đếm đều nằm trong cửa sổ xử lý. Tính lại `N/xu_ly_s` khớp FPS ghi trong metadata, sai lệch 0.
- Hai lượt 1.470 frame có MOT và lớp giống byte kết quả vòng 23. Metadata ghi 1.470 frame hiển thị, bỏ 0, không lỗi ghép frame. Tôi không tự chạy lại detector trên Nano.
- Tính lại 8 lượt đối chiếu hiển thị: trung vị FPS CV **9,211724 bật / 9,390481 tắt**, giảm **1,904%**; EKF **9,182451 / 9,377455**, giảm **2,080%**. Đây là mô tả hai lượt mỗi nhóm, không phải kiểm định hoặc xếp hạng CV/EKF.
- **433/433** file trong các manifest sao chép từ Nano khớp hash. Tôi đã xem ảnh chụp launcher đang chạy và hoàn tất; log/ảnh nhất quán với giao diện được mô tả.
- Ảnh PNG xem trước và detection NPZ không nằm trong gói. Vì vậy việc “mẫu ảnh khớp pixel” và “detection các lượt giống nhau” được đối chiếu qua log/JSON/hash; không phải tôi tự tính lại từ các ảnh/cache thật.

Kiểm thử độc lập:

| Kiểm | Kết quả và phạm vi |
|:--|:--|
| `test_nano_stream.py` | **20 PASS, 1 SKIP**; ca detector PyTorch thật thiếu đầu vào trong môi trường review |
| `test_nano_hien_thi.py`, nguyên trạng | **14 PASS, 2 FAIL, 1 SKIP**; hai FAIL ở phép đếm tiến trình, GUI Tk bị SKIP vì không có display |
| Điều tra hai FAIL | `pgrep -f` trong môi trường này trả cả chính lệnh `pgrep`, làm helper báo còn 1 inhibitor dù không có tiến trình inhibitor thật; không quy thành lỗi rò rỉ trên Nano |
| Chạy lại riêng nhóm controller 8a–8f | **6/6 PASS**, chỉ thay helper đếm tiến trình trong harness bằng kiểm executable thực qua `/proc`; không sửa mã dự án |
| Fixture bật/tắt hiển thị | CV/EKF cùng detections cho MOT/lớp giống byte; 40/40 mẫu ảnh mỗi tracker khớp renderer; dừng bằng stop-file đúng trạng thái |

Log full suite bàn giao trên `a73eb1e` ghi **36/36 PASS**. Tôi đã đối chiếu log, không tự chạy lại toàn bộ 36 suite. Không báo gộp thành “mọi test đều PASS trên máy reviewer” vì có các giới hạn môi trường và hai finding mới nêu trên.

**5. Chỉnh nhỏ trong tài liệu và phạm vi sửa tiếp**

Định nghĩa FPS hiển thị trong `NANO_LOCAL_UI.md` và `nano_live.py` đang viết N/(mốc cuối−mốc đầu), trong khi mã `HienThi.dong()` tính **(N−1)/(mốc cuối−mốc đầu)**. Đồng bộ câu chữ theo phép tính thực tế khi sửa N25-F1; không tự thay các số benchmark đã đo.

Giữ base review ở **`f86732b`** đến khi đóng hai finding. Sửa nguồn và thêm regression; kiểm trên Nano một lượt ngắn dừng bằng đóng cửa sổ cùng các thao tác mở/hủy/đóng picker. Chỉ chạy lại các phần bị ảnh hưởng. Chưa có lý do chạy lại UA-DETRAC/BDD, build engine hoặc toàn bộ các lượt 1.470 frame nếu sửa chỉ nằm ở bộ đếm hiển thị và vòng đời picker, với kiểm chứng tracking không đổi.

Sau vòng sửa, bàn giao log thật, trạng thái tiến trình, metadata hiển thị và delta mới. Không cần thêm camera. Không coi việc chưa có người bấm chuột tay là finding riêng; vấn đề cần xử lý ở đây đã có căn cứ từ mã và tái hiện có kiểm soát.

Review không sửa source, index, lịch sử Git hoặc dữ liệu dự án của người dùng.
