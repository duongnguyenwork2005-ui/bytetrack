# Đo tốc độ cả ứngviên không đạt cổng khuyến nghị

Bổsung trước khi chấm AP TensorRT và trước phépđoFPS mới. Người dùng
yêu cầu địnhlượng đánhđổi accuracy/speed, nên cần đo cả điểm trên đường
trade-off, không chỉmôhình được khuyếnnghị. 00_protocol.md giữ nguyên
lịch sử; bổsung này thay điều kiện mở timing, không thay metric/cổngAP.

- Capture7190ảnh vẫn phảihoàn tất: đúngengine/ONNX/hash/shape/lớp,
  finite/ranges, sourcepre/post, đủindices/markers/SHA vàactualAP kiểmđạt.
- AP50/50-95 giảm<=.005, AP50mỗilớp giảm<=.02 sooriginal vẫn làcổng
  **khuyếnnghị chấtlượng**. Môhìnhtrượt vẫnghiFAIL vàkhôngpromote mặcđịnh.
- Sau đủactualAP, đoGPUriêng vàpilot4clip cho cả6engine đãkhóa. Giữcả
  rawscoreparityFAILR151 trongbảng. Không gọi đây làmở R151 đãđóng.
- Full80 chọnứngviênpilotFPS cao nhất trongnhómquaAP vànhanhhơnoriginal.
  Nếu khôngcó, đoứngviênpilotnhanhnhất riêng đểđịnhlượng trade-off, ghi
  rõ chỉthửnghiệm/khôngđạt cổngchấtlượng. Khôngnới cổng,chọnclass hay
  thayGTđểgọiứngviên thànhcông. Cùng80ảnhcửa sổ/nămclass/KF/thresholds,
  khôngskipinference, báo APthật vàHOTA/IDSW/FN/FP cùngFPS.

R145control/pruned đượcđo bằngngân sách vàprotocol riêng khi cómodel;
khôngbỏthửpruning chỉvì shapesmall trượt. Khôngdeploy hoặc sửalauncher/
khóa luận. Các thínghiệm negative cũng làkếtquả cần bàn giao.
