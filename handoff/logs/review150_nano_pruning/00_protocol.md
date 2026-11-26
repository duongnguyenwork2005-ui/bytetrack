# R150 — tốc độ và chất lượng mô hình sau pruning trên Nano

Người dùng20/11 bật lại Nano và yêu cầu tiếp tục. SSH aliasnano xácnhận
jetson-desktop192.168.1.5,TRT8.2.1.9,L4T32.7.5,swap0/3,96GiBRAM,
SDerrorcount0,MAXN; governorCPU schedutil, GPU76,8–921,6MHz. Không đổi
launcher/hệ điều hành/driver/clock trước phép đối chứng cùngboot.

Phân biệt bốn cấu hình, cùng FP16 TensorRT1x3x384x640, đủ5lớp:

- original: engine YOLO11n50/50 R109 nguyên byte, SHA13f8cd0f...
- anchor_rebuilt: cùng checkpoint gốc nhưng xuất/build theo R145, để
  tách ảnh hưởng graph/export/tactic so với engine cũ.
- control: dense R145, KD5epoch cùng dữ liệu/ngân sách của pruned.
- channel25_KD: slice vật lý một số hiddenchannelsR104, tổng tham số
  giảm10,62%, phục hồi bằng KD5epochR145. Không gọi toànmạng giảm25%.

Giữ nguyên preprocessing/NMS/ByteTrackKF/thresholds/classmap của gói
overlapR108 mà R109 đã đo: detector mỗiảnh,c25,matrixNMS,decodeoverlap,
displayoff,JPEGtmpfs,150ảnh/đoạn,bỏ25ảnhwarmup,FPS5BDD/25UA. Không bỏ
inference để tăng FPS. Code/engine/ONNX/imagehash ghi trước và kiểm sau.

Đầu tiên đo lại engine gốc GPUcompute và trạng thái thiết bị, chưa gọi
đó là end-to-endFPS. Sau R145quality/exportparity: build3engineFP16 mới
trên chính Nano, cùngworkspace512MiB; log parser/layers/tactics, không
épINT8 khi thiết bị không có fastINT8. Kiểm input/outputshape, finite,
TensorRToutput vs CPUORT trên10ảnh cốđịnh trước tracking.

Đo GPU theo thứtự original/anchor/control/pruned/pruned/control/anchor/
original, cùngwarmup/duration/iterations; lưu trtexeclogs. Pipeline pilot
4đoạn (hai mỗi tập chọn trước indices0,10) đủ150ảnh; thành công/shape/
counter/nguồn/parity gốc R109 trước full80 (40BDD+40UA giữmanifestR109).
Đảo cấu hình bắt đầu giữa các video; ghi FPSmean/median/aggregate, p90,
phần trăm tăng ghép cặp theo video, sốclip>=15, HOTA/IDF1/IDSW/FN/FP.
Khởiđộng/boot/swap/SDerror/thermal được kiểm, không chạy build cùng đo.

R145dev7190 AP50/AP50-95/APtừnglớp là cổng dev: giảm<=.5điểm tổng,
<=2điểm AP50 từnglớp so cảanchor/control. Điểm640px không gọi là chất
lượng TensorRT384x640; bổ sung phép chấm detection cùngshape nếu cần.
Ngay cảứng viên trượt chất lượng vẫn có thể đo để báo trade-off theo
yêu cầu người dùng, nhưng không chọn làm mặc định hoặc giấu phần giảm.

Giảm byte/params/MAC/PCtiming không thay thế sốFPS thật. Không áp sốKF
cho EKF/UKF/PF khi chưađo. Không chạy quality trên các tập kịch bản trước
cổng bộlọc BDD/UA của người dùng. Nano80 là phép đo cửa sổ150ảnh, không
gọi là tốc độ trên toàn39.973+56.340ảnh của benchmarktracking đầyđủ.
