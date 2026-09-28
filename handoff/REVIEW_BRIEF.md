# REVIEW_BRIEF — vòng 24: đóng N23-F1, kiểm lại hai video render trên Nano

<!-- BASE_TARGET -->

Base **giữ `c4b39fe`** (review `handoff/CODEX_REVIEW_1d5ffaa.md`: tracking PASS WITH LIMITATIONS, bộ kiểm render REQUEST CHANGES).
Chỉ đổi bộ kiểm render và test. Tracking, runner, luồng trực tiếp, YAML khóa, UA-DETRAC/BDD, benchmark và bốn lượt Nano giữ nguyên,
không chạy lại. Chi tiết: **`handoff/REVIEW24_CHI_TIET.md`**.

## N23-F1

- **Tái hiện** bằng script Codex (không sửa) trên `1d5ffaa`: video `f999` trên 27 frame có box vẫn exit 0, `dat` true. Nguyên nhân
  đúng như review: `ky_vong` và `nen` chung số frame nên mặt nạ loại vùng chữ.
- **Sửa** (`9b1805c`): số frame kiểm **độc lập trên mọi frame**: (a) tỷ lệ lỗi trên điểm chữ `f<i>`, ngưỡng cũ 0,5/40; (b) nhận dạng
  giữa `f<i>`, mọi chuỗi cách một chữ số và bản không số. Giữ phép bbox/màu ID, tên lớp, cú pháp 3.6.
- **Sau sửa**: script Codex exit 0 (video đúng exit 0; video sai exit 1, bắt đúng 27 frame, không báo nhầm 3 frame đúng).
  `test_render_mot_video` 13/13; test mới trên mã cũ chỉ 7/11. Full suite PC trên mã cuối: **35/35**
  (`logs/review24/tests_20260928-1820.log`). Lần chạy trước đó 34/35 vì `test_pc_launcher` phép 9 vượt timeout 3 s (3,23 s); mã launcher
  không đổi, chạy riêng 3 lần đều 17/17. Log lần FAIL được giữ lại.

## Hai video Nano (Python 3.6.9 + OpenCV 3.2.0 hệ thống)

Hash video/MOT/lớp khớp bằng chứng vòng 23. Bộ kiểm mã cuối (gói v11), cùng tham số vòng 23:

| | CV lượt 1 | EKF lượt 2 |
|:--|:--|:--|
| frame / FPS / kích thước | 300 / 25 / 960×540 | 300 / 25 / 960×540 |
| số frame hiển thị | **300/300 đúng** | **300/300 đúng** |
| bbox/màu ID | 1.272 box, 0 lỗi | 1.270 box, 0 lỗi |
| tên lớp (mẫu mỗi 5 frame) | 255/255 | 255/255 |
| exit | 0 | 0 |

Kiểm bổ sung tên lớp trên **mọi box**: 1.271/1.272 và 1.269/1.270. Box còn lại (frame 90, ID 30, `car`) có nét chữ 1 px bị mp4v làm
mờ nên car/bus chỉ chênh 0,3 mức. Render lại từ cùng MOT/lớp cho file **byte-giống**, nên video đúng và được giữ nguyên.

**Phát hiện thêm**: bộ kiểm vẽ lại chữ, mà OpenCV 5.0 (PC) vẽ `putText` khác 3.2 (khung chữ nhật trùng), nên chạy khác phiên bản cho
kết quả vô nghĩa. Đã chốt: OpenCV render (`<video>.json`) khác OpenCV kiểm → exit 2 `KHONG KIEM DUOC` (`4f1fa2f`).

## Đính chính vòng 23 (không đổi số)

Bốn lượt tracking, mỗi lượt khoảng 158–163 giây xử lý, trong phiên khoảng 13 phút 33 giây, có nghỉ và replay giữa lượt; chưa phải
chạy nhiều giờ. Nhiệt: CPU tối đa 38 °C, GPU tối đa 35,5 °C. MemAvailable +3 MB: không thấy tích lũy, chưa loại trừ mọi rò rỉ.
Video là render **hậu kỳ**, không phải hiển thị trực tiếp.

## Giới hạn

Sai từ hai chữ số trở lên chỉ dựa vào phép tỷ lệ. Tên lớp mặc định lấy mẫu. Reviewer cần OpenCV 3.2.0 để kiểm lại video (không nằm
trong delta). Hiển thị trực tiếp và camera chưa làm.
