# R151 — trade-off độ phân giải, dùng nguyên checkpoint detector

Song song pruningR145 (đangtrain), thử hai input cố định352x608 và320x576
so384x640 trên cùng anchorYOLO11n50/50. Cácshape là bội32, sốpixel giảm
12,92% và25%; đây là giảthuyết tính toán, chưa là mức tăngFPS. Không kéo
méo ảnh: letterboxgiữtỷlệ,pad114. Không bỏframe hoặc sửa tracker.

Tấtcả dùng cùngcheckpointSHA d6aad866..., rawoutput5lớp, ONNXFP32opset12,
TensorRTFP16buildtrênNano, cùngworkspace512MiB. PyTorch/ORTFP32 parity
10ảnhmonitor cốđịnh trước khi build. Ghi hash toànnguồn/checkpoint và7190
ảnh/nhãn dev theo manifestmixed5, không scan/select thêm theo kếtquả.

Chấm đủ7190mixed5dev ở đúngshape, chạyCPU để không chenGPUtraining.
Dùng preprocessing/postprocessing nguyênsource NanoR109 snapshotR150,
c25precision/recall IoU.5 và AP50/AP50-95 từ ngưỡng.001,NMSIoU.7,
maxdet300. AP dùng lớp argmax như Nano, khác Ultralyticsvalmultilabel;
so nội bộ cácshape ở đây, không đánhđồng giátrị với AP640R145.
Matching/AP dùng DetectionValidator/DetMetrics hiện có, không đổiMOTmetric.
Ảnh đã graymaskignore như exporterdev; ghi rõ đây làdev không phảitest.

Cổng dev đềxuất: mAP50/50-95 giảm<=.5điểm %, AP50mỗi5lớp giảm<=2điểm %
so384x640. Giữ lại mọi điểm âm để báotrade-off, không chỉbáocấu hình tốt.
Không dùng điểmBDDval/UA40 đểchọnshape hoặc lấyđiểmtrackingkịchbản.

SauengineparityFP16 (10ảnh, finite/ranges/shape; anchorscores>=.05 có
boxmaxabsdiff<=6px và scoremaxabsdiff<=.02 soFP32), đoGPUABBA rồi pilot
4clip indices0,10 mỗiBDD/UA. Mỗiclip150ảnh/warmup25, cùnggói overlap,
JPEGtmpfs,classes/threshold/ignoremask/fps/trackerKF nhưR109.
Khôngđổi clock/governor/launcher. Nếu tốcđộ đo tốt hơn vàdevđạt, chọn
ứngviên pilotmeanFPS cao nhất trước khi mở phépđo80clip ghépcặp với
original; nếu cảhai trượtquality, vẫnbáo trade-off, khôngpromote mặcđịnh.
TensorRTGPUms, PCinference haypixel không gọi làFPSend-to-endNano.
