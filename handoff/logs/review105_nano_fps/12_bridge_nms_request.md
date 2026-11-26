# Yêu cầu bridge bổ sung trong R105 — chưa chạy

Benchmark chính 30 lượt đã đủ và đã chấm. Không gửi lại hoặc chạy lại toàn ma trận. Coder chỉ đọc evidence bridge, không mở được kết nối SSH tin cậy và không có công cụ gọi phiên chính; file này là yêu cầu bàn giao, không phải bằng chứng job đã khởi tạo.

Motion/guarded tốn 4,68–10,65 ms scheduler/frame và MVI_39031 không bỏ được inference. Nếu phiên chính còn thời gian/kênh bridge trong ngân sách hiện có, có thể đo **một tối ưu đã nằm trong R105**: bật `R103_EARLY_CUTOFF=1` của package thử R105/R103, đưa cutoff hậu xử lý trước NMS từ 0,05 lên 0,25, giữ filter tracker c25. Không thay source hoặc package đã chạy. Không dùng lại gain R103 để kết luận gain R105.

Phép đo cần cùng engine `13f8cd0f…`, JPEG/tmpfs và KF/c25 hiện có; tắt `R103_CADENCE`, `R103_ROI_Y`, `R105_MODE`. Chạy full và early mới ghép cặp trong output mới, ví dụ `~/nano_run/r105_nano_fps/nms_extra/`, không ghi đè `smoke/` hoặc `full/`. Chọn hai clip BDD đầu và hai clip UA đầu theo cùng manifest đã chốt, tối đa **8 lượt × 150 frame**. Full mới ở đây chỉ dùng làm đối chứng cùng phiên cho thay đổi NMS; không lặp lại ma trận benchmark cũ.

Trước đo, xác nhận code/engine/ONNX không đổi, early flag sẵn có, smoke flag trên đoạn ngắn hoặc cặp đầu. Giữ command, env flags cụ thể, source/package hash trước/sau, pipeline CSV, run_meta, MOT/lớp, swap/boot/nhiệt/thẻ. Kiểm đầy đủ frame/inference và MOT/lớp trùng full; nếu khác phải chấm và báo chất lượng. Tính FPS từ timestamp và gain từng cặp, không lấy tỷ số trung vị; chờ/reap tất cả job, trả exit và danh sách job cuối. Nếu không đủ thời gian/kênh bridge thì giữ mục này **chưa đo**, không mở quyền hay tăng ngân sách.

Phiên coder hiện tại không mở job, không chỉnh cấu hình Nano, không tự chọn tham số mới. Báo cáo chính `10_ket_qua.md` không có kết luận tăng FPS do NMS bổ sung.
