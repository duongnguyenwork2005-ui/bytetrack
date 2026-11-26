# R104 — giảm dung lượng YOLO11n năm lớp

Chọn checkpoint FP16 **5,47 MB** cho dự án. Đây là bản giảm gần một nửa dung lượng checkpoint và giữ kết quả phát hiện, theo dõi xe máy gần đối chứng nhất trên các mẫu đã kiểm tra. Bản ONNX trọng số INT8 **2,90 MB** dành cho trường hợp ưu tiên dung lượng và chấp nhận chênh lệch tracking đã đo. Các tệp ở `models/r104_yolo_compression/`, kích thước và SHA256 ở `artifacts.json`. MB tính theo 1.000.000 byte.

## Checkpoint và ONNX

- **FP16.** `.pt` **10.708.735 → 5.467.661 byte**, giảm **48,94%**. Mô hình vẫn có 2.590.815 tham số. Mixed5 dev **7.190 ảnh** mAP50 **49,56 → 49,49%**, mức giảm AP50 lớn nhất theo lớp là motorcycle **0,36 điểm**. Nén thêm `.xz` còn **4,78 MB** và giải nén khớp byte checkpoint FP16. Checkpoint FP32 gốc nén `gzip` còn **6,77 MB**, `xz` còn **5,53 MB**, cả hai đã kiểm khôi phục đúng tệp nguồn.
- **INT8 chỉ lượng tử trọng số.** ONNX **10.568.102 → 2.902.719 byte**, giảm **72,53%**. 88 tensor trọng số convolution dùng INT8 đối xứng theo kênh, được giải lượng tử trước phép tích chập. Tệp chạy trực tiếp bằng ONNX Runtime CPU 1.22.1, kích thước vào **1×3×640×640** và ra **1×9×8400**. Dev300 chấm trực tiếp ONNX **48,91 → 48,85% mAP50**. Dev7190 nạp đúng 88 tensor đã giải lượng tử vào PyTorch, mAP50 **49,45%**, mức giảm AP50 từng lớp lớn nhất **0,44 điểm** ở truck. Phép chấm full giữ trọng số đã lượng tử và dùng đường thực thi PyTorch, còn phép chấm dev300 kiểm chính tệp ONNX.
- **INT8 tĩnh và số bit thấp hơn.** Cùng dev300 và đối chứng **48,91% mAP50**, INT8 tĩnh hiệu chuẩn 64 ảnh train, lượng tử trọng số và activation convolution, còn **3,15 MB / 45,21% mAP50**. INT4 chỉ trọng số còn **1,63 MB / 14,83%**. Bản trọng số hỗn hợp 4/8 bit còn **2,11 MB / 47,80%**. Sau pilot QAT trọng số một epoch với 400 ảnh train, bản 4/8 bit đạt **46,90%**, AP50 xe máy **14,83%** so với **23,25%** đối chứng. QAT này kiểm khả năng phục hồi với ngân sách ngắn.

## Pruning và phân rã hạng thấp

- **Pruning rời rạc 30%.** Đặt các trọng số tích chập nhỏ nhất bằng 0. `.pt` FP16 còn **5,47 MB** vì các tensor vẫn có cùng kích thước. `.pt.xz` giảm **4,78 → 3,68 MB**. Full dev **49,59% mAP50**, nhưng recall xe máy trên hai clip RideSafe giảm **32,28 → 30,12%**. Chọn FP16 đầy đủ khi theo dõi xe máy.
- **Pruning kênh 25/50% ở các lớp được chọn.** Tổng tham số giảm **10,62% / 20,23%**. Các nhánh dự đoán và năm lớp còn đủ. Bản cắt 25% fine-tune ba epoch còn **4,89 MB `.pt`**, đạt **46,01% mAP50**. Bản cắt 50% kết hợp một epoch distillation đạt **36,75% mAP50**. Lượt distillation đầu với teacher FP16 gặp loss NaN và được dừng. Lượt đo dùng teacher FP32 hoàn tất một epoch, trên cùng 1.400 ảnh train. Hai bản này có ngân sách huấn luyện khác nhau.
- **Phân rã hạng thấp 19 lớp convolution.** `.pt` FP16 giảm **5,47 → 5,07 MB**, dev **49,16% mAP50**. Engine TensorRT 384×640 giảm **8,09 → 7,87 MB** (**2,78%**), thời gian GPU A/B/B/A **41,62 → 42,62 ms**. Mức giảm dung lượng dưới mốc 10% đặt cho ứng viên engine. Kết quả GPU là benchmark engine riêng.

## Kiểm tra tracking và Jetson Nano

- Hai clip RideSafe khác ngày train, **600 khung gán nhãn và 1.388 hộp xe máy**, ghép detection IoU ≥ 0,5 ở c25. Recall đối chứng / FP16 / INT8 trọng số đã giải lượng tử lần lượt **32,28 / 32,20 / 31,41%**. Recall xe máy nhỏ **26,94 / 26,94 / 26,37%** trên 876 hộp nhỏ. HOTA KF trung bình clip **12,02 / 12,06 / 10,45%**, IDSW cộng hai clip **19 / 18 / 23**. Do đó chọn FP16 cho mục tiêu tracking của dự án dù bản INT8 nhỏ hơn và qua cổng AP nội bộ. Đây là hai clip, nên các chênh lệch IDSW có phạm vi mẫu đó.
- Nano TensorRT **8.2.1.9** báo có FP16 nhanh và không có INT8 nhanh. Engine FP16 dựng lại từ `.pt` FP16 vẫn **8,09 MB**, GPU A/B/B/A **41,56 → 41,60 ms**. Đối chứng và bản dựng lại hoàn thành **16/16 lượt** trên bốn clip BDD và bốn clip UA, mỗi clip 150 JPEG, hiển thị tắt. Chênh lệch FPS ghép cặp **+0,17% BDD / −0,17% UA**. HOTA **45,64 → 45,26% / 68,66 → 68,71%**, IDSW **45 → 44 / 49 → 40**. Điểm tracking có thay đổi nhỏ dù kích thước và tốc độ engine tương đương.
- Khi mục tiêu là chuyển hoặc lưu engine Nano, gói `.engine.xz` của đúng engine R102 giảm **8,09 → 5,73 MB**. Đã xác minh SHA256 sau giải nén. Cấu hình đang cài trên Nano vẫn sử dụng launcher trước vòng này. Các lượt benchmark đã kết thúc và bộ đếm lỗi thẻ là 0.

## Bằng chứng

- `native_dev7190.json`, `native_final_dev7190.json`, `onnx_dev300.json`, `screen_dev300.json`, `qat_dev300.json` và `ridesafe.json` chứa kết quả từng lớp hoặc từng clip.
- `nano_fp16_inputs.json`, `nano_fp16_abba.json`, `nano_fp16_pairs.json`, `nano_fp16_assessment.json`, `nano_lowrank_abba.json` và `nano_lowrank_files.json` chứa đối chiếu engine và pipeline.
- `00_protocol.md` ghi tiêu chí chọn trước khi chấm. Các runner trong cùng thư mục tạo, huấn luyện hoặc chấm ứng viên. Thí nghiệm dùng dev mixed5 để sàng lọc. BDD và UA chỉ dùng hậu kiểm trên cùng mẫu đã đo trước.
