# Dùng hệ thống tracking tại chỗ trên Jetson Nano (màn hình HDMI + chuột + bàn phím)

Chương trình chạy **hoàn toàn trên Nano**, trong phiên desktop của tài khoản `jetson`; không cần SSH hay PC. Nguồn dữ liệu là
**file video hoặc thư mục ảnh có sẵn** (chưa hỗ trợ camera). Đường xử lý là luồng đã nghiệm thu: đọc frame → TensorRT FP16
(Python 3.6 hệ thống) → tracker CV/EKF (env `nano`, backend NumPy). Giao diện chỉ điều phối, không có thuật toán riêng.

Gói đang triển khai: `~/nano_run/v16/nano_transfer` (dưới đây gọi là `GOI`). Khi có gói mới, thay `v16` bằng thư mục mới
và chạy lại bước 1.

## 1. Tạo biểu tượng (một lần, hoặc sau khi đổi gói)

Mở **Terminal** trên desktop Nano (Ctrl+Alt+T) và chạy:

```bash
bash ~/nano_run/v16/nano_transfer/scripts/nano/08_tao_loi_tat.sh
```

Kết quả: biểu tượng **“Theo vết phương tiện (Nano)”** trên desktop và trong menu ứng dụng (gõ “Theo vết” ở ô tìm kiếm).
Lần đầu bấm đúp, nếu Ubuntu hỏi *Untrusted application launcher*, chọn **Trust and Launch**. Gỡ biểu tượng:
`bash .../08_tao_loi_tat.sh --xoa`. Không cần quyền sudo.

## 2. Mở chương trình

- Bấm đúp biểu tượng trên desktop, hoặc tìm trong menu; hoặc trong Terminal của desktop:
  `bash ~/nano_run/v16/nano_transfer/scripts/nano/07_giao_dien.sh`.
- Dòng đầu cửa sổ cho biết engine TensorRT và Python detector. Chữ đen, có ✔: sẵn sàng. Chữ đỏ: làm theo hướng dẫn ngay trong
  dòng đó (xem mục 6). Nút **Bắt đầu** bị khóa khi cấu hình chưa đúng.
- Engine, Python và thư mục mặc định lấy từ `GOI/configs/nano_ui.json`; muốn đổi riêng cho máy này, tạo
  `~/.config/nano_tracking/nano_ui.json` chỉ chứa khóa cần đổi (ví dụ `{"thu_muc_ket_qua": "/home/jetson/ket_qua"}`).

## 3. Chạy một lượt

1. **Dữ liệu đầu vào**: bấm **Chọn video…** (mp4, avi, mkv, …) hoặc **Chọn thư mục ảnh…** (jpg/png/bmp, tên có số thứ tự;
   nhập **FPS**, ví dụ 25). Dữ liệu có sẵn trên máy:
   - `~/nano_run/data_full/MVI_39031` — 1.470 ảnh, 25 fps;
   - `GOI/data/smoke/MVI_39031_f0001-0300` — 300 ảnh; `GOI/data/smoke/MVI_39031_f0001-0300.mp4` — cùng đoạn dạng video.
   Ô **Số frame tối đa** để trống = chạy hết nguồn.
2. **Bộ lọc**: CV (Kalman vận tốc không đổi) hoặc EKF (CTRV).
3. **Lưu kết quả**: thư mục gốc (mặc định `~/nano_ket_qua`). Mỗi lượt tạo thư mục con mới
   `<tên nguồn>_<cv|ekf>_<ngày-giờ>`, không ghi đè lượt cũ. **Hiển thị trực tiếp** bật sẵn; **Lưu detection cache** để
   replay/kiểm lại (tùy chọn).
4. Bấm **▶ Bắt đầu**. Trạng thái: **ĐANG KHỞI TẠO** (nạp engine: khoảng 10 giây; lần đầu sau khi bật máy có thể khoảng
   1 phút) → **ĐANG CHẠY**. Cửa sổ **“Xem truc tiep - CV - …”** hiện từng frame với khung, số ID và tên lớp; góc trên trái là
   số frame `f<i>`; dòng dưới: bộ lọc, frame, số đối tượng, FPS hiển thị. Thanh tiến trình và dòng trạng thái trong launcher
   ghi frame đã xử lý / tổng và FPS xử lý.

Trong lúc đang chạy, launcher tạm chặn khóa màn hình tự động (Ubuntu khóa sau 5 phút không dùng chuột/bàn phím); hết lượt
hoặc đóng launcher thì trả lại như cũ.

## 4. Dừng

Bốn cách, **cùng một cơ chế dừng có kiểm soát**: hoàn tất frame đang xử lý, ghi đủ kết quả các frame đã xử lý, trạng thái
**ĐÃ DỪNG** (`run_meta.json`: `trang_thai = da_dung`), tiến trình con và cửa sổ xem được đóng.

- nút **■ Dừng** trong launcher;
- bấm vào cửa sổ xem rồi nhấn **q** hoặc **Esc**;
- đóng cửa sổ xem (nút ×);
- đóng launcher (× hoặc **Thoát**) khi đang chạy: launcher dừng lượt, chờ ghi xong rồi mới đóng (dòng trạng thái báo
  “ĐANG DỪNG ĐỂ ĐÓNG…”).

Đóng launcher khi **hộp chọn file đang mở** (kể cả lúc đang chạy): hộp chọn file do launcher mở cũng được đóng và thu hồi theo, không
để lại cửa sổ hay tiến trình nào. Nhật ký cả phiên launcher (kể cả các lần mở/hủy hộp chọn file) được lưu khi đóng vào
`~/.cache/nano_tracking/launcher_phien_*.log`.

Không có chức năng tạm dừng. Sau khi một lượt kết thúc có thể chọn nguồn/bộ lọc khác và bấm **Bắt đầu** lượt mới.

## 5. Kết quả

Bấm **Mở thư mục kết quả**, hoặc mở thư mục lượt trong `~/nano_ket_qua`. Khi kết thúc, launcher hiện tóm tắt: số frame,
số hàng MOT, số ID, FPS xử lý, FPS hiển thị, số frame xem trước bị bỏ, thời gian toàn lượt.

| File | Nội dung |
|:--|:--|
| `data/<tên>.txt` | kết quả tracking dạng MOTChallenge (frame, id, x, y, w, h, conf) |
| `data_cls/<tên>.csv` | lớp phương tiện theo (frame, id) |
| `run_meta.json` | trạng thái (`hoan_tat` / `da_dung` / `loi_*`), số frame, cấu hình hiệu lực, thời gian, thiết bị, lệnh replay |
| `hien_thi.json` | thống kê cửa sổ xem: số frame đã hiện, bị bỏ, FPS hiển thị, lý do dừng từ cửa sổ |
| `progress.json`, `launcher.json`, `launcher_nhat_ky.txt` | trạng thái cuối, thời gian phía launcher, nhật ký đầy đủ của lượt |
| `timing/`, `pipeline/`, `tegrastats.log`, `detector_stderr.log` | thời gian từng frame, tài nguyên, log detector |
| `det_cache/` | detection để replay (nếu chọn lưu cache) |

Ý nghĩa các con số:

- **FPS xử lý** = số frame đã xử lý / thời gian vòng xử lý (đọc frame → TensorRT → tracker), không gồm khởi tạo và ghi
  kết quả. Đây là tốc độ thật trên Nano, **không** phải FPS danh nghĩa của video (25).
- **FPS hiển thị** = (N − 1) / (mốc ảnh hiện cuối − mốc ảnh hiện đầu), N = số ảnh **thực sự đã hiện** lên cửa sổ xem đang mở
  (N ảnh có N − 1 khoảng giữa). Dưới 2 ảnh thì **không có** FPS hiển thị (ghi `null`, tóm tắt ghi “không có FPS hiển thị”).
- **Frame không hiện vì cửa sổ xem đã đóng**: frame mà tracker xử lý xong sau khi bạn đóng cửa sổ xem (thường 1 frame đang dở);
  không tính là đã hiện, nhưng **vẫn có trong kết quả tracking** (`data/`, `data_cls/`).
- **Frame xem trước bị bỏ**: khi cửa sổ xem không kịp vẽ, ảnh xem trước cũ nhất bị bỏ để không làm chậm; **tracking vẫn xử
  lý đủ mọi frame**, không đổi ngưỡng hay bước thời gian.
- **Thời gian toàn lượt**: `run_meta.thoi_gian.tong_s` (trong nano_live) và `launcher.json.tong_s_phia_launcher` (từ lúc
  bấm Bắt đầu). Không có tạm dừng nên không có thời gian tạm dừng trong các chỉ số.

## 6. Lỗi thường gặp

| Hiện tượng | Nguyên nhân / cách khắc phục |
|:--|:--|
| Dòng đỏ “Không thấy engine TensorRT” | Engine chưa build hoặc đường dẫn khác: `cd GOI && bash scripts/nano/01_build_engine.sh` (khoảng 8 phút), hoặc sửa khóa `engine` trong `~/.config/nano_tracking/nano_ui.json` |
| “Python detector … lỗi import” | TensorRT/OpenCV của Python 3.6 hệ thống không nạp được; kiểm `/usr/bin/python3 -c "import tensorrt, cv2"` |
| “Thư mục ảnh cần FPS” / “Ô FPS phải > 0” | Nhập FPS của dữ liệu (UA-DETRAC: 25) |
| “Không ghi được vào thư mục kết quả” | Chọn thư mục trong thư mục nhà, còn chỗ trống |
| LỖI “không mở được màn hình / DISPLAY” | Mở chương trình từ desktop của Nano (biểu tượng hoặc Terminal trên desktop), không từ SSH |
| Bấm biểu tượng không thấy gì | Xem `~/.cache/nano_tracking/giao_dien_*.log` và `launcher_phien_*.log`; chạy `bash GOI/scripts/nano/07_giao_dien.sh --kiem-tra` trong Terminal |
| Màn hình đã khóa | Đăng nhập lại; lượt vẫn chạy phía sau, kết quả không mất |
| Mất điện giữa lượt | Thư mục lượt có `progress.json` ở `dang_chay`/`dang_khoi_tao` và **không có** `run_meta.json`: dữ liệu dở dang, không dùng; chạy lại lượt mới |
| Trạng thái LỖI sau khi chạy | Đọc dòng “Lỗi:” trong launcher và `launcher_nhat_ky.txt`, `detector_stderr.log` trong thư mục lượt |

## 7. Giới hạn

Chưa có camera; một lượt tại một thời điểm; không tạm dừng. FPS phụ thuộc tốc độ giải mã của OpenCV 3.2 (có hai mức
khoảng 11 và 16 ms/frame giữa các lượt), nên chênh FPS nhỏ giữa CV và EKF không dùng để xếp hạng. Cửa sổ xem có thể bỏ
frame xem trước khi máy bận (được đếm trong `hien_thi.json`).

## 8. Tự kiểm nhanh bằng chuột/bàn phím (5 bước, khoảng 5 phút)

1. Bấm đúp **“Theo vết phương tiện (Nano)”** trên desktop. Dòng đầu cửa sổ có ✔ cạnh Python detector.
2. **Chọn thư mục ảnh…** → `~/nano_run/data_full/MVI_39031`, FPS `25`, chọn **EKF**, giữ **Hiển thị trực tiếp**, bấm **▶ Bắt đầu**.
   Cửa sổ “Xem truc tiep - EKF - …” hiện khung, ID, tên lớp và số frame tăng dần.
3. Khoảng 20 giây sau, bấm vào cửa sổ xem rồi nhấn **q** → launcher báo **ĐÃ DỪNG**, cửa sổ xem đóng.
4. **Chọn video…** → `~/nano_run/v16/nano_transfer/data/smoke/MVI_39031_f0001-0300.mp4`, chọn **CV**, **Bắt đầu**, để chạy hết →
   **HOÀN TẤT ✔** và dòng tóm tắt; bấm **Mở thư mục kết quả**.
5. Bấm **Bắt đầu** lần nữa, rồi đóng launcher bằng dấu × khi đang chạy → dòng “ĐANG DỪNG ĐỂ ĐÓNG…”, sau vài giây launcher tự đóng.

Mỗi bước tạo một thư mục trong `~/nano_ket_qua`; `run_meta.json` ở bước 3 và 5 có `trang_thai = da_dung`, bước 4 `hoan_tat`.
