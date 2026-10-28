# R62 — kết luận mốc 6 trên Jetson Nano (29/10/2026)

**Kết quả âm theo tiêu chí đã chốt:** ứng viên `cv2.LUT` tăng FPS toàn luồng, nhưng trung vị hiệu theo bốn cặp chỉ **3,917% CV** và **4,277% EKF**, dưới mức 5% bắt buộc cho cả hai. Giữ **v16** làm bản dùng; không thay launcher, engine, detector threshold, thuật toán tracking hay metric. Đây là quyết định theo phép đo ngắn có hiển thị đúng [kế hoạch mốc 6](01_ke_hoach_moc6.md), không dùng phép đo vi mô hay lượt dài có decode khác tầng để nâng kết luận.

## Điều kiện và nguồn gốc

- Jetson mới dùng lại thẻ SD cũ; root `/dev/mmcblk1p1`, L4T 32.7.5, Ubuntu 18.04.6, TensorRT 8.2.1.9, OpenCV 3.2.0. Engine FP16 SHA256 `c32690cdb9f14d5f016212126621ec1fecef6047ebda82fb844530e9b7424a9a`; ZIP v16 SHA256 `302f6af73d97aea009066416aaaa202db81311e60121676a75dacbe204e9cbf1`; 562/562 file manifest đã kiểm.
- Cùng 300 JPG `MVI_39031_f0001-0300`, 25 FPS nguồn, c25, lớp `[1,2,3,5,7]`, `bytetrack.yaml`/EKF đã khóa, engine trên, HDMI 960×540, 25 frame đầu bỏ khỏi thống kê trễ ổn định. Các lượt đo chạy trên cùng boot `f6f8c913-61cf-4b45-af91-ead44ca14f6b` theo thứ tự cân bằng: CV B→C/C→B/B→C/C→B rồi EKF tương tự; nghỉ 20 giây giữa lượt. B là bản sao v16, C là bản sao ứng viên.
- Gói ứng viên ZIP SHA256 `e1e155b0e37316e341343e502e48d329ae363f073cf57dfc83c90941e80d6bd1` được dựng từ v16 đã kiểm manifest; nội dung chỉ khác `src/nano_detector.py` và `PACKAGE_INFO.json`, manifest cập nhật. Mã detector ứng viên SHA256 `6a1165f06ad5f36e48c95a43a1e1c96022c0355058c9669830055dee0e057afe`; `cv2.LUT` chuyển uint8 BGR HWC sang float32 RGB NCHW với bảng 256 giá trị đúng phép chia cũ. PC detector 7/7 đạt, Nano CPU 11 ca tensor trùng byte. Micro chuyển tensor 9,778→2,396 ms, chỉ đo thao tác đó.

## Phép so cặp 300 ảnh

Toàn bộ **16/16 lượt** exit 0, đủ 300 frame, live/replay MOT và lớp trùng byte; detection và MOT/lớp giữa v16/ứng viên trùng byte trong cả tám cặp. Không có frame xem trước bị bỏ ở bất kỳ lượt so cặp nào. Số thô, hash từng file, p90, nhiệt, RAM/swap, provenance và mọi kiểm tra của từng lượt ở [28_so_cap_300_lan01.json](28_so_cap_300_lan01.json); không bỏ lượt hoặc chọn kết quả đẹp.

- CV: hiệu FPS bốn cặp **+3,795%, +4,096%, +4,039%, +3,521%**; trung vị **+3,917%**, khoảng 3,521–4,096%. Trung vị FPS từng nhóm v16 9,292 và ứng viên 9,659. Trung vị thay đổi p90 trễ theo cặp **−3,568%**; tiền xử lý trung vị 13,586→6,326 ms. RAM peak trung vị 2434→2434,5 MB; CPU max 49→49,5 °C; GPU max 46→46 °C.
- EKF: hiệu FPS bốn cặp **+4,436%, +4,119%, +3,962%, +4,477%**; trung vị **+4,277%**, khoảng 3,962–4,477%. Trung vị FPS từng nhóm v16 9,272 và ứng viên 9,673. Trung vị thay đổi p90 trễ theo cặp **−3,877%**; tiền xử lý trung vị 13,617→6,272 ms. RAM peak trung vị 2440→2440 MB; CPU max 51→52 °C; GPU max 48→48 °C.
- Swap peak trong telemetry là 11 MB cho cả 16 lượt; swap không tăng từ trước đến sau ở từng lượt. Nhiệt và xung được ghi riêng trong JSON/raw. Chưa có review chuyên biệt đủ để khẳng định *không throttling*; xung thấp do `schedutil` không tự chứng minh throttling. Điều này không làm thay đổi kết luận âm vì cả hai trung vị FPS đã dưới 5%.

## Lượt dài và giới hạn

Baseline v16 dài **4/4 × 1470** đạt CV 8,275/8,297 và EKF 8,294/8,288 FPS. Ứng viên dài **CV 8,587; EKF 9,576 FPS**, mỗi lượt 1470/1470, live/replay byte đạt. Detection/MOT/lớp của hai lượt ứng viên trùng byte với baseline dài cùng tracker; [29_so_dai_1470_lan01.json](29_so_dai_1470_lan01.json) giữ đủ số và hash. Decode trung vị của lượt EKF ứng viên ~11,42 ms trong khi baseline dài ~23,59 ms, nên **không** lấy hiệu FPS dài làm bằng chứng về sửa mã.

Lượt baseline dài thử đầu `baseline1470/01_cv_on` bị reboot/kernel panic trước khi có kết quả; giữ nguyên file dở và pstore ở [22_dmesg_ramoops.txt](22_dmesg_ramoops.txt), [23_console_ramoops.txt](23_console_ramoops.txt). Chưa xác định căn nguyên panic. Các lượt đạt sau đó cùng boot, không có crash mới. Lượt chẩn đoán đầu lỗi thiếu biến môi trường trước khi chạy GPU; lượt sửa đúng đạt 300/300 và parity byte.

Raw CSV/log/meta của baseline dài, 16 lượt so cặp và ứng viên dài được sao lưu trong [r62_evidence_small.tar.gz](r62_evidence_small.tar.gz), SHA256 `81ccef2b53934b1578feca4c338bbde73fdf490ef9b5e7a2eb7731f0e284e637`, khớp PC/Nano. Tar loại cache NPZ; cache gốc còn trên Nano dưới `~/nano_run/r62_20261028/`.

Shortcut desktop `~/.local/share/applications/nano-tracking.desktop` vẫn trỏ `~/nano_run/v16/nano_transfer/scripts/nano/07_giao_dien.sh`; `src/nano_launcher.py --kiem-tra` từ v16 exit 0, Python detector và TensorRT/OpenCV/NumPy đều OK. Không cần đường quay về vì chưa từng chuyển launcher. Mốc 6 đóng với kết quả âm; một đợt tối ưu khác, nếu có, cần kế hoạch mới.
