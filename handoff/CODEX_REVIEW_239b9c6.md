# Review vòng 26 — 239b9c6

**Kết luận: PASS WITH LIMITATIONS. Đóng N25-F1 và N25-F2; không có finding mới chặn nghiệm thu trong phạm vi hai bản sửa. Có thể đóng mốc giao diện local trên Nano.**

Tôi review gói delta, tái dựng mã nguồn và chạy kiểm thử độc lập trên Linux/Python 3.12.14, NumPy 2.3.5, OpenCV 4.11.0. Tôi không SSH vào Nano và không thao tác trực tiếp trên HDMI; bằng chứng thiết bị được đối chiếu từ các file đã bàn giao. Kết luận này không mở rộng thành nghiệm thu camera, vận hành nhiều giờ hoặc mọi tình huống lỗi hệ thống.

## 1. Gói và phạm vi thay đổi

- Base: `f86732b71399f11b1392b54c2558133772d41e26`.
- Target: `239b9c6b01c74fba3fa5077ffdda7a0f73c77834`.
- ZIP: **11.188.466 byte**, **547 thành viên**, `testzip` không lỗi.
- SHA256: `854b09a303e810d63335e4f140504318e17476d605ea1a6ee2c011bacff4f8f7`.
- **541/541 payload** khớp hash và số byte; 77 đường dẫn được khai báo dùng nội dung trùng. Patch áp được lên cây base giữ từ vòng trước; **618/618 đường dẫn thay đổi** khớp hash target. Các preimage có hash base đều khớp trước khi áp.
- Không thấy file riêng, video, engine, weights, NPZ hoặc ZIP khác trong tập thay đổi được đóng gói. Các NPZ bị loại tiếp tục được khai báo trong `DEPENDENCIES.md`.

So với `20e66e9`, chỉ năm file Python thay đổi: `nano_hien_thi.py`, `nano_launcher.py`, `nano_detect_stream.py`, `nano_live.py` và `test_nano_hien_thi.py`. Hai file stream/live bổ sung hook kiểm thử và sửa định nghĩa metadata; không thay thuật toán tracker hoặc detector.

## 2. Hai finding đã đóng

### N25-F1 — đếm frame sau khi cửa sổ đóng: ĐẠT

`HienThi._ve()` trả về sớm khi đã đóng; đường trình ảnh kiểm cửa sổ trước `imshow`. Bộ đếm/timestamp chỉ cập nhật sau khi trình thành công. Frame không hiển thị được đếm riêng. Chế độ ẩn được ghi rõ là giả lập. FPS dùng `(N−1)/(t_cuối−t_đầu)` và trả `null` khi chưa có hai mốc.

Chạy nguyên script `CODEX_REPRO_20e66e9_ui.py` trên target:

```text
imshow_calls = 1
reported_n_hien_thi = 1
reported_last_frame = 1
fps_hien_thi = null
stop_reason = dong_cua_so
N25-F1: PASS
```

Regression 10/10b/10c/11 cũng PASS độc lập: đóng sau frame, đóng giữa hai frame, công thức FPS và lưu đủ MOT/lớp của phần đã xử lý. Đây là renderer thật với thao tác HighGUI giả lập, cùng một ca chạy qua entry point; không phải phép kiểm màn hình Nano do reviewer thực hiện.

### N25-F2 — picker còn sống khi đóng launcher: ĐẠT

Launcher giữ `Popen`, hủy callback chờ, gửi tín hiệu tới đúng tiến trình do nó tạo, chờ và thu hồi. Picker đã tự thoát thì bỏ kết quả; callback sau destroy được chặn. Đường đóng vẫn chờ tracker dừng có kiểm soát.

Script tái hiện cũ trả:

```text
launcher_destroyed = true
picker_alive_after_close = false
scheduled_callbacks = 1
N25-F2: PASS
```

Giá trị callback bằng 1 thuộc stub cũ: `after()` trả `None` và không có `after_cancel`. Nó không phải bằng chứng còn callback Tk thật. Harness mới có lịch hẹn/hủy kiểm các ca 12a–12d đạt: **0 callback**, không cập nhật widget sau destroy, picker đã thoát; khi đang tracking, worker/detector cũng thoát và lượt mang trạng thái `DA_DUNG`.

Tôi kiểm bổ sung nhánh picker không chịu SIGTERM bằng tiến trình thật trên Linux, gọi đúng callback launcher với widget Tk giả lập. Sau khoảng **2,36 s**, child nhận SIGKILL (exit −9), được thu hồi, stdout đóng, launcher destroy, **0 callback/0 lần ghi widget sau destroy**. Ca này PASS; nó không thay thế phép kiểm Zenity treo trên Nano.

## 3. Kiểm thử độc lập và bằng chứng bàn giao

| Kiểm | Kết quả |
|:--|:--|
| Script Codex nguyên bản, cả hai finding | **exit 0**, cả hai PASS; file trong gói giống byte bản reviewer đã gửi |
| `python -u src/test_nano_hien_thi.py` | **24/24 PASS, 2 SKIP, exit 0** trên máy reviewer |
| Hai SKIP trên máy reviewer | Ca 9 và 12e cần cửa sổ Tk thật; máy review không có display |
| Nhánh SIGTERM không có tác dụng → SIGKILL | PASS trên Linux, tiến trình thật, Tk giả lập |
| Full suite PC do Claude bàn giao | CSV có **36/36 PASS, mọi exit 0**; log ghi HEAD `6dbf40f`. Reviewer không chạy lại full suite |
| Test cuối trên Nano do Claude bàn giao | **23/23 PASS, 1 SKIP**; ca cần cv2 trong env tracker được kiểm bằng Python hệ thống ở phần F1 |

Hai FAIL helper đếm tiến trình ở vòng review trước không còn xuất hiện: helper mới đọc executable/argv trong `/proc`, không tự đếm lệnh tìm kiếm.

Log lần đầu trên Nano vẫn giữ ca 12e FAIL. Bản test cuối thêm kiểm `winfo_exists()` để nhận biết cửa sổ đã hủy, rồi chạy lại đạt. Tôi phân biệt log cũ với `v26_test_lai/B2_ma_da_sua.log`; không bỏ qua thất bại ban đầu hoặc gộp nó thành PASS. Ca 12c trên mã cũ lỗi vì thiếu thuộc tính mới, nên không dùng nó làm bằng chứng tái hiện điều kiện đua; những ca tái hiện gốc và 12a/12d/12e cung cấp bằng chứng riêng.

## 4. Đối chiếu Nano v16

**73/73 file** trong hai manifest sao chép của vòng 26 khớp hash. Sáu hash mã nguồn có trong provenance của mỗi lượt khớp target theo quy ước LF; hai hash rút gọn `nano_hien_thi.py` và `nano_launcher.py` in trong log thiết bị cũng khớp source target. Tôi không tự kiểm toàn bộ gói v16 trên thiết bị.

| Ca | Đối chiếu từ file bàn giao |
|:--|:--|
| D — đóng cửa sổ xem khi TensorRT đang chạy | **181 frame xử lý; 180 hiển thị; 1 không hiện vì đã đóng**, là frame 181; không bỏ preview, không lỗi display |
| D — tracking | MOT **554 hàng** và CSV lớp live/replay **giống từng byte**; cũng giống byte tiền tố 181 frame của lượt CV không hiển thị vòng 25 |
| D — FPS | Tính lại hiển thị `(180−1)/19,143523254` = **9,350420903 FPS**; FPS xử lý **9,231110847** khớp `181/xu_ly_s` |
| E1/E2 — picker thật | Log ghi hủy → mở lại → chọn đúng đường dẫn; khi đóng launcher, SIGTERM đúng PID và thu hồi exit −15 |
| E3 — picker mở khi tracking | Picker được thu hồi, tracker dừng có kiểm soát ở **57 frame**, exit 4; log xác nhận không còn tiến trình/cửa sổ |
| F — desktop, hai lượt EKF | Hai lượt **300/300 frame**, exit 0; hiển thị lần lượt **299 + 1 preview bị bỏ** và **300**; thoát sạch theo log |

Tôi đọc lại pipeline CSV của cả bốn lượt D/E3/F: frame 1…N xuất hiện đúng một lần; mọi frame được đếm nằm trong cửa sổ đo. Công thức FPS xử lý/hiển thị và các bộ đếm khớp metadata. `hien_thi.json` khớp các trường tương ứng trong `run_meta.json`; ba trường bổ sung phía tracker về lệnh/kết quả hiển thị cũng nhất quán.

## 5. Giới hạn và quyết định tiếp theo

- Thao tác HDMI/Zenity thật là tự động hóa theo log; reviewer không trực tiếp bấm chuột hoặc nhìn màn hình thiết bị. Đây là giới hạn bằng chứng, không phải finding mới.
- Nhánh SIGKILL đã được reviewer kiểm bằng tiến trình cố tình bỏ qua SIGTERM trên Linux; chưa có ca Zenity thật treo trên Nano.
- Không có NPZ/engine/ảnh nguồn trong delta để reviewer chạy lại TensorRT. So sánh byte ở đây dùng MOT/lớp đã bàn giao.
- Không suy rộng các lượt ngắn thành độ ổn định nhiều giờ; không mở thêm camera hoặc tính năng giao diện.

**Cho phép ghi nhận đóng N25-F1/N25-F2 và nghiệm thu mốc giao diện Nano với các giới hạn trên.** Claude có thể lưu báo cáo này, lưu trữ đúng gói đã được review và chuyển base review từ `f86732b` sang **`239b9c6`**. Không cần chạy lại UA-DETRAC/BDD, build engine hoặc lặp toàn bộ lượt 1.470 frame cho hai bản sửa này.

Bước tiếp theo giữ đúng plan người dùng đã khóa: chuẩn bị dữ liệu và kế hoạch fine-tune YOLOv8n/YOLO11n, sau đó đánh giá detector, tracker/bộ lọc và cuối cùng tối ưu Nano. Bất kỳ đề xuất mở rộng nào ngoài plan phải hỏi người dùng trước khi đưa vào prompt cho Claude. Review này không tự khởi chạy training hay mở thêm thí nghiệm.

Reviewer không sửa mã nguồn, index, lịch sử Git hoặc môi trường PC/Nano của người dùng.
