# Review vòng 24 — f86732b

**Kết luận: PASS WITH LIMITATIONS. Đóng N23-F1. Không còn finding chặn nghiệm thu trong phạm vi vòng này.**

Tôi đã đọc mã và bằng chứng trong delta, tái dựng snapshot từ base đã có, chạy độc lập bộ test render và nguyên script tái hiện của review trước. Tôi không truy cập Nano trong lần review này; video thật không nằm trong gói. Bằng chứng hai video Nano được đối chiếu qua log, JSON và hash, không phải do tôi tự giải mã lại hai MP4.

**1. Snapshot và tính toàn vẹn**

| Mục | Đã kiểm độc lập |
|:--|:--|
| Base | `c4b39fe3861baac76211b2cb82c62d970b719767` |
| Target | `f86732b71399f11b1392b54c2558133772d41e26` |
| ZIP | 6.308.100 byte, 174 thành viên; `testzip` không lỗi |
| SHA256 ZIP | `71fddf70a64428273498abc508eef4e06fac5d2fef1c2b2b33ea11067fbf809c` |
| Payload | 168/168 khớp hash và số byte trong MANIFEST |
| Patch | Áp thành công lên base; 187/187 đường dẫn do patch thay đổi khớp target |
| Loại trừ | Không thấy đường dẫn riêng bị cấm hoặc payload video/engine/weights/ZIP khác |

Các NPZ bị loại có chủ ý được khai báo trong `DEPENDENCIES.md`; vòng 24 không bổ sung NPZ mới. Không coi việc dựng được snapshot mã là đã có đầy đủ dữ liệu để chạy lại thí nghiệm trên Nano.

Đối chiếu với snapshot vòng 23: thay đổi mã hiện có tập trung ở `src/kiem_video_render.py`, `src/render_mot_video.py`, `src/test_render_mot_video.py`. Mã tracking, cấu hình thí nghiệm và bằng chứng tracking vòng 23 không đổi. Không có căn cứ yêu cầu chạy lại UA-DETRAC/BDD hoặc bốn lượt tracking 1.470 frame vì bản sửa này.

**2. N23-F1 — đã đóng bằng kiểm thử độc lập**

Lỗi cũ loại mất vùng chữ số frame khỏi mặt nạ so sánh khi frame có box. Bản mới dựng nền có bbox/nhãn nhưng chưa có số frame, rồi kiểm số frame riêng trên mọi frame. Kiểm bbox/màu ID và kiểm tên lớp vẫn tách biệt; ngưỡng tỷ lệ 0,5 và mức lỗi 40 không bị nới.

Tôi chạy trên Linux, Python 3.12.14, NumPy 2.3.5, OpenCV 4.11.0; fixture được render và kiểm bằng cùng OpenCV:

```bash
python src/test_render_mot_video.py
python handoff/CODEX_REPRO_1d5ffaa_render_counter.py
```

Lệnh thứ hai dùng nguyên script đã bàn giao ở vòng 23, chạy từ gốc snapshot; không chỉnh fixture để phù hợp bản sửa.

| Phép kiểm | Kết quả độc lập |
|:--|:--|
| Bộ test render | **13/13 PASS**, exit 0 |
| Video đúng của script tái hiện | Exit 0; kiểm đủ 30/30 số frame; không lỗi |
| Video ghi `f999` trên 27 frame có box | Exit 1; bắt **đúng 27 frame** 1–10 và 14–30 |
| Ba frame đúng 11–13 | Không bị báo nhầm |
| Bbox/ID của video sai số frame | Vẫn đạt; chứng minh phép kiểm số frame hoạt động độc lập |
| Script tái hiện tổng thể | **REGRESSION PASS**, exit 0 |

Bộ test cũng bắt được mất số, lệch một frame và số sai chỉ ở frame không có box. Chốt khác phiên bản OpenCV trả exit 2; chế độ chẩn đoán vẫn giữ `dat=false`, không biến một phép kiểm không hợp lệ thành PASS.

**3. Bằng chứng trên Nano và hai video giữ nguyên**

Hash mã trong `handoff/logs/review24/kiem_lai_nano_v11/00_hash_va_phien_ban.txt` khớp chính mã cuối trong snapshot:

- `kiem_video_render.py`: `90b09b9231e28ea9d61299e49596c96c2912bf03d3f0f7b8ccddcfb551271a58`.
- `render_mot_video.py`: `8f52de8de273f9269d82ea333a26a1c433bd4ccfe36e6f0c99f2b54d578bc61a`.

Các bản sao bằng chứng khớp manifest từ Nano: v10 **16/16**, v11 **14/14**, không thiếu hoặc lệch hash. JSON v11 ghi OpenCV kiểm và render đều **3.2.0**.

| Kết quả từ log/JSON v11 | CV | EKF |
|:--|--:|--:|
| Frame giải mã / số frame hiển thị đã kiểm | 300 / 300 | 300 / 300 |
| FPS / kích thước | 25 / 960×540 | 25 / 960×540 |
| Số lỗi số frame / bbox | 0 / 0 | 0 / 0 |
| Số box | 1.272 | 1.270 |
| Nhãn lớp lấy mẫu mỗi 5 frame | 255/255 đạt | 255/255 đạt |
| Kiểm bổ sung mọi nhãn lớp | 1.271/1.272; **exit 1** | 1.269/1.270; **exit 1** |

Ca chưa phân biệt được ở cả hai video là frame 90, ID 30: lớp kỳ vọng `car`, bộ kiểm chọn `bus` với sai số rất gần nhau. Kết quả FAIL của phép kiểm toàn bộ nhãn được giữ nguyên trong JSON/log. Không diễn giải bảng lấy mẫu thành “mọi nhãn đều được máy xác nhận đúng”.

Hash video, MOT và lớp trong hồ sơ v11 khớp hồ sơ vòng 23. Sidecar của lượt render đối chiếu ghi đúng hash video cũ. Điều này hỗ trợ giữ nguyên hai video và kết luận refactor không đổi đầu ra trên thiết bị; tôi chưa tự băm MP4 vì MP4 không có trong delta.

**4. Kiểm thử tổng và giới hạn nghiệm thu**

Log bàn giao `tests_20260928-1820.log` ghi 35 suite PASS trên `974affc`. Log lần trước `tests_20260928-1804.log` vẫn giữ lỗi timeout của test launcher; mã launcher không đổi trong vòng này. Tôi đã đối chiếu các log, **không tự chạy lại toàn bộ 35 suite**. Kết quả tự chạy trong review này là 13/13 test render và script hồi quy nêu trên.

Các giới hạn sau không chặn đóng N23-F1:

- Kiểm số frame dùng tỷ lệ lỗi và tập ứng viên cách một phép sửa chữ số. Fixture `f999` đã bị bắt; chưa có bảo đảm nhận dạng đúng mọi chuỗi sai tùy ý.
- Chốt OpenCV chỉ xác minh được khi sidecar cung cấp phiên bản. Sidecar thiếu/không đọc được vẫn có thể cho `opencv_render=null`; vì vậy cần giữ sidecar cùng video khi dùng làm bằng chứng. Cùng số phiên bản cũng không phải bảo đảm tuyệt đối cho mọi bản build/nền tảng.
- Tên lớp mặc định được kiểm theo mẫu; ca frame 90 nêu trên chưa đạt phép nhận dạng mọi nhãn. Chưa cần đổi codec hoặc video chỉ để đóng lỗi số frame.
- Phạm vi ổn định trước đó là bốn lượt xử lý khoảng 158–163 giây/lượt trong một phiên có nghỉ và replay, không phải chạy liên tục nhiều giờ. Diễn đạt mới về nhiệt CPU/GPU và bộ nhớ đã phù hợp hơn với bằng chứng.
- Camera, hiển thị trong lúc tracking, vận hành nhiều giờ và nguyên nhân hai mức decode vẫn ngoài phạm vi đã nghiệm thu.

**5. Quyết định bàn giao**

Đóng N23-F1; giữ nghiệm thu tracking 1.470 frame của vòng 23 và chấp nhận bổ sung bộ kiểm render sau sửa với giới hạn trên. Có thể lưu bản review này tại `handoff/CODEX_REVIEW_f86732b.md`, lưu trữ đúng ZIP có SHA256 đã nêu, rồi chuyển base review sang **`f86732b`**.

Bước phát triển tiếp theo phù hợp là hiển thị trực tiếp trên HDMI khi tracking, trước tiên với nguồn file đã kiểm chứng và có đối chứng bật/tắt hiển thị. Camera là bước riêng sau đó. Không cần tiếp tục mở rộng bộ nhận dạng chữ của video hậu kỳ để bắt đầu công việc này.

Review này không sửa mã dự án, cấu hình, dữ liệu thí nghiệm, index hoặc lịch sử Git của người dùng.
