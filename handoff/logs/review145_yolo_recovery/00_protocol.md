# R145 — phục hồi chất lượng sau pruning kênh để chuẩn bị tăng FPS Nano

Người dùng19/11 muốn tăng FPS hơn mốc18,63 bằng xử lý mô hình, chấp nhận
đánh giá trade-off. Nano hiện tắt theo người dùng; chỉ có số đo PC, không
khẳng định FPS Nano. Không đổi launcher hoặc mô hình đang dùng.

R104 đã thử FP16, INT8/INT4/PTQ/QAT ngắn, sparse pruning và channel pruning.
Nano lúc đo báo fast_int8=False; không lặp lại giảm byte trọng số rồi gọi
đó là tăng tốc. Hướng mới: train phục hồi có distillation dài hơn trên tập
train lớn hơn cho cấu trúc channel25 vật lý đã kiểm. Tỷ lệ25% chỉ ở một
số lớp, tổng tham số giảm khoảng10,62%; không gọi toàn mạng giảm25%.

Hai nhánh cùng dữ liệu, seed145,5epoch,640px,batch4,SGD lr0=.002, warmup.5,
mosaic/mixup0,ampFalse,workers0, teacherFP32 anchorYOLO11n50/50 R101:

- control: anchor không cắt, distillation cùng hệ số.
- channel25_KD: channels25_init.pt R104 (chưa fine-tune), distillation.

Dữ liệu: nguyên5000 ảnh mixed5 train đã chọn R91 +200 ảnh RideSafe train
R103 (ngày20220824/20221119). Không lấy BDD val/UA test/HELMET/SVMOT làmtrain.
Monitor dùng đúng300 ảnh mixed5 dev R103; cuối cùng chấm đầy đủ7190ảnhdev.
Giữ năm lớp car/truck/bus/motorcycle/bicycle và đủ ba scalehead. Khóa hash
train/label/image/config/source/weights trước training, kiểm train/dev
không trùng đường dẫn hoặc byte. Không quét/đổi split theo điểm test.

Distillation dùng loss đã thử R104: supervised +.5 confidence-weighted
classificationBCE +.25 DFLKL; chỉteacheranchorscore>.1,teacherFP32nograd.
Hệ số giữ nguyên, không dò theo điểm. Cùng loss cảcontrol vàpruned để tách
tác động của cấu trúc khi có cùng ngân sách training. Best epoch theo
monitor dev300, báo đầy đủ history và không dùng test chọn epoch.

Chấm trước/sau: mAP50,mAP50-95, AP/precision/recall từng5lớp; đối chiếu
anchor gốc vàcontrol cùngtraining. Cổng đề xuất: mAP50 vàmAP50-95 giảm<=.5
điểm phần trăm, AP50 từnglớp giảm<=2điểm. Vẫn lưu/report mọi ứng viên trượt,
không chỉ báo cấu hình thắng. Đây là lựa chọn dev, chưa phải kết quả test.

Xuất cảhai model ONNX FP32 tĩnh1×3×384×640/opset12, giữ rawoutput1×9×5040,
kiểm onnx.checker + CPUORT/PyTorch parity. Ghi params/MAC/dung lượng thực;
không suy FLOPs hoặc checkpoint byte thành FPS. Khi Nano bật: build hai
engineFP16 cùngTRT8.2/cùngshape, warmup/ABBA, rồi đo cùng80đoạn R109,
đồng thời chấm detection/tracking. INT8 chỉ được ưu tiên nếu runtime thực
sự hỗ trợ và số đo cho lợi ích; không hứa tốc độ từ weight-only INT8.

Tối đa3 job bền vững kể cảtrainingGPU, không đo timingPC khi qualityjob
đang chạy. Nếu OOM, dừng và ghi lỗi; có thể giảm batch2 cho **cảhai** nhánh
trước khi có kết quả, không âm thầm thay số epoch/dataset/teacher.

Nguồn: [DepGraph](https://arxiv.org/abs/2301.12900),
[Torch-Pruning tác giả](https://github.com/VainF/Torch-Pruning),
[Ultralytics về dense pruning](https://docs.ultralytics.com/yolov5/tutorials/model-pruning-and-sparsity).
Ở đây dùng cấu trúc slice thủ côngR104 đãkiểm, không gọi là chạy thư viện
DepGraph. TensorRT/Nano khả năng dựa trên bằng chứng thiết bị R104, chưa
được kiểm lại khi Nano đang tắt.
