# Khảo sát tốc độ từ đầu vào đến kết quả một video — 29/10/2026

Theo yêu cầu người dùng: kiểm tra công đoạn có thể tối ưu. Đã rà mã tại HEAD
`304ce7ad5c1f30084d96f29bef83bac79a042389` và tính lại raw của R62; chưa sửa mã thực thi,
chưa chạy GPU/benchmark MP4 mới. Mốc 6 vẫn đóng với kết quả dưới ngưỡng, launcher dùng v16.

## 1. Định nghĩa hoàn thành và phạm vi bằng chứng

Luồng hiện tại: khởi tạo nguồn/engine → đọc và giải mã frame → letterbox/chuyển tensor →
H2D/TensorRT/D2H → decode detection/NMS → gửi detection → tracker → MOT/lớp/timing.
Hiển thị là luồng riêng. Detector và tracker đã chồng thời gian qua hai tiến trình.
Xuất MP4 có khung/ID là bước hậu kỳ riêng, không nằm trong `fps_xu_ly`.

- `src/nano_live.py:48` định nghĩa các cửa sổ thời gian; `:948` ghi `render=False`.
- Cần báo cả FPS vòng xử lý và thời gian từ bắt đầu đến khi đầu ra yêu cầu được ghi xong,
  tính cả đóng encoder nếu xuất video. Không quy thời gian startup/render tiết kiệm thành FPS vòng xử lý.
- Số dưới đây từ 16 lượt paired300 **JPG**, mỗi nhóm 4 lượt, bật HDMI, bỏ 25 frame đầu khi
  thống kê công đoạn. Mỗi số công đoạn là trung vị của 4 trung vị từng lượt; không phải phần trăm
  cộng được của tổng thời gian vì các giai đoạn chồng nhau và trung vị không có tính cộng.
- Nguồn: `28_so_cap_300_lan01.json`; `r62_evidence_small.tar.gz` chứa
  `paired300/<run>/live/{pipeline,timing}/MVI_39031.csv` và `run_meta.json`.
  Tar SHA256: `81ccef2b53934b1578feca4c338bbde73fdf490ef9b5e7a2eb7731f0e284e637`.
- Các phép MP4 trước đây là kiểm chức năng; không có so cặp video trên Jetson hiện tại đủ để
  kết luận mức tăng FPS. Không dùng JPG baseline so MP4 ứng viên.

## 2. Số đo tính lại

Các giá trị theo thứ tự CV / EKF, đơn vị ms/frame trừ nơi ghi khác:

- Đọc/giải mã JPG baseline: **11,407 / 11,430**. Không phải số giải mã MP4.
- Tiền xử lý baseline: **13,586 / 13,617**; LUT: **6,326 / 6,272**.
- Toàn lời gọi infer baseline: **74,306 / 74,229**; LUT: **77,964 / 77,851**.
- Bên trong infer baseline: H2D **2,721 / 2,679**, execute **68,979 / 68,962**,
  D2H **1,609 / 1,612**. Đây là thời gian host quanh lệnh đồng bộ, chưa phải CUDA-event profile.
- Phần còn lại của infer: tính trên từng frame `infer_ms-h2d_ms-exec_ms-d2h_ms`,
  rồi lấy trung vị từng lượt/nhóm, baseline **1,036 / 1,008**, ứng viên **1,027 / 1,038**.
  Bao gồm wrapper/copy và overhead đo; chưa quy toàn bộ cho một thao tác.
- Hậu xử lý baseline: **7,192 / 7,223**.
- Truyền detection sang cha: **0,362 / 0,365**; lọc ở cha **0,625 / 0,625**;
  chờ credit ở con **0,0028 / 0,0028**.
- `tracker.update` baseline: **5,595 / 6,449**, trong đó predict **0,784 / 1,353**.
  `t_assoc_ms` trong mã là update trừ predict: còn chứa cập nhật bộ lọc/lifecycle, không phải association thuần.
- Startup baseline: **8,733 / 8,570 s**; ghi kết quả sau vòng lặp **0,130 / 0,130 s**.
  Trong toàn bộ 16 lượt, startup 8,34–11,26 s; total_s 40,55–44,70 s, chưa gồm xuất MP4.
- FPS baseline **9,292 / 9,272**, ứng viên **9,659 / 9,673**. Hiệu theo cặp **3,917% / 4,277%**.
  Nhanh hơn trong phép thử JPG nhưng dưới ngưỡng 5% của R62. Không cộng lợi ích giả định của các tối ưu.

## 3. Các vị trí đáng tối ưu

### A. Copy đầu ra TensorRT và ma trận hậu xử lý — ứng viên mã nhỏ ưu tiên

`src/nano_detector.py:264` cấp `h_out` đúng dtype output; engine đang dùng output float32.
Tuy nhiên `:320` gọi `out.reshape(self.out_shape).astype(np.float32)` tạo một bản sao toàn bộ
1×84×8400, tức **2.822.400 byte/frame**, dù đã float32. Có thể khảo sát trả view hoặc
`astype(copy=False)` cho đường gọi tuần tự. Phải kiểm quyền sở hữu/vòng đời: raw sẽ bị ghi đè
khi infer frame sau; caller lưu raw hoặc pipeline async cần buffer riêng. Chưa có số tiết kiệm Nano.

`decode_nms` tại `:158` đã copy `p[keep0]` đủ 84 cột, rồi `:161` copy tiếp
`cls_scores[keep0]` gồm 80 cột để argmax. Tái dùng phần class trong `p` đã lọc là một giả thuyết
cắt cấp phát cụ thể. Đo riêng max/argmax, chuyển box và NMS trước khi tối ưu sâu vòng lặp NMS.
Giữ thứ tự tie, stable sort, max_det và điều kiện IoU; không mặc định OpenCV NMS giống kết quả.

### B. Tiền xử lý — đã có lợi ích xác minh, chưa đạt ngưỡng triển khai

LUT trong `src/nano_detector.py:54` đến `preprocess` giảm khoảng 7,3 ms toàn tiền xử lý.
Repo hiện đã chứa ứng viên này; baseline phải lấy **bản sao v16 đã kiểm hash**, không lấy HEAD
gọi là baseline. Có thể khảo sát tái dùng buffer với kích thước cố định. Resize/padding/chuyển màu
khác backend có thể khác pixel, cần kiểm tensor và kết quả. Infer của LUT tăng khoảng 3,6 ms
trong các lượt đã đo; chưa có bằng chứng quy nguyên nhân cho DVFS, cache hay contention.

### C. Giải mã video và đọc trước — ưu tiên đo trên MP4 thật

`src/nano_detect_stream.py:102` dùng VideoCapture, `:138` dùng cap.read; decode rồi detector
chạy nối tiếp trong vòng `:381`. Có thể thử decoder phần cứng theo codec hoặc decode frame kế
tiếp trong worker với hàng đợi nhỏ, giữ mọi frame và đúng thứ tự. NVIDIA có nvv4l2decoder và
nvvidconv trên L4T 32.x; phải kiểm plugin/build OpenCV trên máy và chi phí chuyển ảnh về CPU:
[NVIDIA Accelerated GStreamer](https://docs.nvidia.com/jetson/l4t/Tegra%20Linux%20Driver%20Package%20Development%20Guide/accelerated_gstreamer.html).
Hỗ trợ plugin không chứng minh đường đọc hiện tại đã dùng hardware decode.

Decoder/chuyển màu có thể khác pixel và ảnh hưởng detection. Prefetch có thể tăng p90 vì frame
chờ trong queue: giữ mốc đo trước decode (`:136`), không dời mốc để che thời gian chờ.
So cùng codec, độ phân giải, video, engine và chế độ hiển thị/xuất; chưa dự đoán mức tăng FPS.

### D. TensorRT và truyền bộ nhớ — chi phí lớn nhưng ít sửa đơn giản

`src/nano_detector.py:295` thực hiện H2D → execute_v2 → D2H đồng bộ. Engine đã FP16,
batch1, input tĩnh 640×640, nên bật FP16 không phải tối ưu mới. Async trên một stream rồi
đồng bộ ngay không tự làm mất 69 ms compute. Muốn overlap cần đo event, thiết kế buffer và
giữ thứ tự frame. Pinned memory trên Tegra cũ không được CPU cache nên có thể chậm hơn;
không áp dụng máy móc hướng dẫn dành cho GPU rời:
[CUDA for Tegra 10.2](https://docs.nvidia.com/cuda/archive/10.2/pdf/CUDA-for-Tegra-AppNote.pdf),
[TensorRT 8.2.5](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-825/pdf/TensorRT-Developer-Guide.pdf).
Đổi engine, precision, kích thước đầu vào hoặc model cần quyết định thí nghiệm riêng và kiểm chất lượng.

### E. Tracker và giao tiếp — ưu tiên sau detector với CV/EKF

Tracker đã chạy song song với detector; chờ credit gần 0 và pipe chỉ truyền detection,
không truyền toàn ảnh. Tối ưu tracker có thể giảm trễ/CPU nhưng không tự giảm chu kỳ detector.
Đây là kết luận cho CV/EKF và mật độ cảnh R62, không áp dụng tùy ý cho video đông xe.

`src/bytetrack_np.py:98` chỉ đăng ký ByteTrack CV/EKF. UKF/PF/SORT trong repo chưa được
backend NumPy này hỗ trợ; cần tích hợp + kiểm tương đương trước khi đo chúng trên Nano.
Khi có số đo, tách predict, tính cost/IoU, assignment, filter update, lifecycle. PF đã vector hóa
phần lan truyền hạt/batch track (`src/tracker_ctrv.py:293`); cần giữ seed, luồng RNG, thứ tự phép
tính và resampling nếu sửa. Không giảm số hạt hoặc bỏ frame rồi gọi là tối ưu tương đương.

### F. Khởi động, RAM và ghi kết quả — ảnh hưởng thời gian chờ và video dài

Mỗi lượt khởi tạo lại engine/warmup (`src/nano_detect_stream.py:310`) và băm toàn file video
trước khi xử lý (`:112`). Giữ engine sống qua nhiều clip có thể tiết kiệm startup ở clip tiếp theo,
nhưng phải tạo/reset tracker riêng từng video và giữ provenance; không cải thiện cold start đầu tiên.

`src/nano_live.py:197,203,256` giữ raw detection của mọi frame kể cả khi không lưu cache;
MOT/timing cũng tích lũy rồi ghi cuối. Video dài có thể tăng RAM/tạm dừng cuối/swap. Có thể giữ
raw chỉ khi cần và ghi theo khối giới hạn, bảo toàn thứ tự, format và khả năng ghi khi bị ngắt.
Ở bài 300 frame, ghi cuối chỉ khoảng 0,13 s nên không phải ưu tiên tăng FPS hiện tại.

### G. Hiển thị và xuất video có bbox/ID — quan trọng nếu đây là đầu ra cuối

Hiển thị đã có worker riêng; lịch sử R25 tắt hiển thị tăng khoảng 2%, không phải đo mới R62.
Baseline/ứng viên cần cùng chế độ hiển thị khi so tối ưu mã.

`src/render_mot_video.py:90` hiện chỉ nhận thư mục ảnh, đọc lại mỗi ảnh, vẽ và ghi `mp4v`
qua VideoWriter. Lịch sử R23 render 300 ảnh mất 17,3–17,6 s trên thiết bị/lượt cũ; chưa phải
số của Nano hiện tại. Nếu cần MP4 cuối, khảo sát encoder phần cứng và tận dụng frame đã decode
qua queue **không drop**, kiểm encoder flush, đủ frame, thứ tự và chất lượng hình. Queue preview
hiện tại được phép bỏ ảnh cũ (`src/nano_hien_thi.py:113`), không dùng nguyên cơ chế ấy để xuất video.

## 4. Thứ tự đề xuất cho kế hoạch mới

1. Chọn video thực tế và đầu ra cần hoàn thành: MOT/CSV hay cả MP4; đo v16 cùng đầu vào,
   ghi đủ startup, hash/decode, detector, tracker, ghi file và export nếu có.
2. Thử riêng copy float32 dư và copy class lặp trên gói thử; kiểm detection/MOT/lớp trước khi đo tốc độ.
   Sau đó mới đo kết hợp với LUT để biết có vượt 5% hay không; không suy từ microbenchmark.
3. Nếu decode hoặc export chiếm đáng kể, thử thay đường giải mã/ghi video theo điều kiện tương đương.
4. Port UKF/PF/SORT nếu người dùng muốn triển khai chúng, rồi đo riêng từng cấu hình.
   Khi thử nhiều tracker trên cùng video và cùng detector, tận dụng cache detection một lần rồi replay:
   đây là giảm tổng thời gian thí nghiệm, không phải FPS toàn pipeline của một tracker.

Kết luận khảo sát: có các vị trí cụ thể để thử, trước hết copy tensor/hậu xử lý và đường video.
Chưa có benchmark mới xác nhận lợi ích. Giữ kết luận R62 và mọi bằng chứng lỗi cũ;
nếu phép GPU tiếp theo tái diễn kernel panic thì dừng đo để xử lý nguyên nhân.
