# R100 — Jetson Nano, PF500 và dữ liệu xe máy

## TensorRT trên Nano

SSH alias `nano` vào `jetson-desktop`. Engine C1 384×640 FP16 SHA256 `a6912393829f8c811d3cc33008f3e4056a626434df327c85aae7beb33af01749`. TensorRT 8.2.1 chạy `trtexec --loadEngine=… --warmUp=500 --duration=5 --iterations=100`, rồi lặp với `--separateProfileRun --dumpProfile --exportProfile=…`. GPU Compute Time trung vị **41,35 ms**, H2D **0,285 ms**, D2H **0,170 ms** và throughput riêng engine **23,37 truy vấn/s**. Đây là phép đo TensorRT riêng. `baseline.log`, `engine_profile.log` và `layers.json` lưu thời gian và lệnh.

Tổng profile **41,42 ms** trên 256 mục. Mục lớn nhất là convolution đầu **1,58 ms (3,81%)**, kế tiếp là mục convolution ghép của detection head **1,46 ms (3,53%)**. Chín mục tên `Reformatting CopyNode` tổng cộng **1,45 ms**. Thời gian trải qua nhiều lớp. Nếu mục tiêu là giảm đáng kể thời gian GPU, ứng viên tiếp theo có cơ sở là thu nhỏ mô hình theo kênh và đánh giá lại trên cùng kích thước ảnh, ưu tiên recall xe máy. Chưa thử engine mới trong lượt này.

## Tốc độ PF500

`bench_uniform_pf.py` giữ cách ước lượng trung bình và hiệp phương sai, tái sử dụng vector trọng số đã tính chính xác khi `logw` đồng đều. Đối chứng đã có tối ưu `_logsumexp` R98. Lượt thử replay dùng detection cache sẵn, cùng hai clip UA-DETRAC. Đủ 8 lượt × 150 frame, đầu ra MOT và lớp trùng từng clip. Cảnh đông MVI_40855 giảm thời gian update trung vị **93,95 → 90,82 ms (−3,33%)**, cảnh thưa MVI_39031 tăng **4,86 → 5,03 ms (+3,45%)**.

Live từ SD đủ 8 lượt, MOT trùng. FPS đo trên SD chịu nhiễu đọc và giải mã JPEG. Lặp live A/B/B/A riêng trên cảnh đông, dùng chung ảnh trên tmpfs cho cả hai nhánh. Bốn lượt đủ 150 frame, MOT trùng, giải mã JPEG trung vị **10,64 → 10,62 ms**. FPS ổn định **9,43 → 9,58 (+1,54%)**, thời gian update trung vị **103,74 → 102,79 ms (−0,91%)**, p90 trễ **571,31 → 558,78 ms (−2,19%)**. Mỗi số A/B là trung vị hai lượt tương ứng. Kết quả tmpfs chỉ kiểm độ nhạy trên một cảnh đã chọn sau khi thấy nhiễu SD. Giữ ứng viên trong wrapper thử nghiệm vì lợi ích live nhỏ và cảnh thưa chưa cải thiện. Không đổi engine, launcher, cấu hình tracker luận văn. Tệp `pf_uniform_abba.json`, `pf_live_sd.json`, `pf_live_ram.json` chứa đầu ra đo. Cuối lượt không còn job benchmark, lỗi ext4 bằng 0, swap khoảng 5 MB.

## RideSafe-400 — hai pilot xe máy từ camera gắn trên xe

Nguồn [RideSafe-400 của tác giả](https://huggingface.co/datasets/DeepBug/RideSafe-400/tree/main), revision `e2a6637`. Tải byte-range đúng hai cặp MP4/XML từ part1, kiểm CRC từng mục và lưu SHA256. Hai clip khác ngày theo tên nguồn `20211125083549_0060` và `20211109132649_0060`. Mỗi clip giải mã đủ 1.500 frame ở 25 fps, 2560×1440, stride nhãn 5 cho 300 khung chấm ở 5 fps, timestamp lệch dưới 1 ms. Hai clip có **1.799 và 3.367 hộp** cùng **57 và 214 ID theo clip**. Dùng ID track CVAT của lớp `motorcycle`, giữ mọi hộp `outside=0`. Cả hai clip có 0 hộp gắn cờ `occluded=1`. Không phân tầng che khuất từ hai clip này. Dữ liệu và MOT từng lượt ở `tmp/r100_ridesafe_sample` và `tmp/r100_ridesafe_recheck`. Metadata, số đo và hash lưu trong các JSON `ridesafe_*` ở cùng thư mục báo cáo.

Recall detector ghép một-một IoU ≥0,5, gộp 5.166 hộp GT. Ở conf >0,05, YOLOv8n COCO **38,54%**, YOLOv8n mixed5 **13,32%**, YOLO11n mixed5 **21,02%**. Ở conf >0,25, tương ứng **20,79%**, **4,22%**, **6,50%**. COCO vượt hai detector mixed5 ở cả hai clip tại cả hai ngưỡng. Trên clip riêng, recall COCO c05 là **53,59%** và **30,50%**. Hai clip chưa đủ để suy ra hiệu quả trên RideSafe đầy đủ hoặc năm lớp.

Cùng detection COCO c25 và TrackEval, IDSW trên clip thứ nhất của KF/EKF/UKF lần lượt **7/12/10**, PF500 K1K2 motion2 **7/7/6** và R95 **10/10/8** theo seed 0/1/2. Clip thứ hai lần lượt **13/14/13**, PF nền **12/12/12**, R95 **12/12/12**. Tổng IDSW của KF/EKF/UKF là **20/26/23**, PF nền **19/19/18** và R95 **22/22/20**. HOTA KF/PF nền trung bình seed theo clip **23,42/23,45%** và **10,70/11,05%**. Phép chấm oracle GT ở cả hai clip cho HOTA=IDF1=1, IDSW=FP=FN=0. Số lần đổi ID của PF nền gần KF ở hai clip. Nhiều GT thiếu detection ở c25, do đó cần đọc IDSW cùng HOTA, IDF1 và FN. Phép thử chỉ chấm xe máy và giữ các cấu hình đang dùng.

## Dữ liệu tiếp theo

- [DAIR-V2X-Seq SPD](https://github.com/AIR-THU/DAIR-V2X/blob/main/docs/get_started_spd.md) có ảnh tuần tự ven đường, timestamp và nhãn camera trong hệ tọa độ LiDAR. [Mẫu chính thức](https://drive.google.com/file/d/1gjOmGEBMcipvDzu2zOrO9ex_OscUZMYY/view) mở trang Drive được; chưa tải được nội dung qua yêu cầu trực tiếp đã thử. Cần kiểm ID của từng góc camera, hộp 2D và căn ảnh–nhãn trước khi dùng để chấm quỹ đạo rẽ.
- [MEVDT](https://pmc.ncbi.nlm.nih.gov/articles/PMC11720431/) có 63 chuỗi ảnh xám từ camera cố định, nhãn box và ID xe tại 24 Hz. [Kho dữ liệu tác giả](https://doi.org/10.7302/d5k3-9150) trả HTTP 403 từ PC trong lượt thử. Chưa tải và chưa chấm MEVDT. Tập này phù hợp khảo sát chuyển miền camera cố định nếu mở được dữ liệu.
