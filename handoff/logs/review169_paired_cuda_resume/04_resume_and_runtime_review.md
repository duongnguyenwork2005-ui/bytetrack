# R169 — kiểm tra tính đối xứng và ranh giới kết luận

Ngày 21/11/2026. Rà soát mã trong lúc huấn luyện và R167 timing đang chạy;
đây là tự rà soát tại cùng phiên, không gọi là đánh giá độc lập của agent khác.
Không sửa bất kỳ nguồn huấn luyện, input contract hay runtime đang khóa.

- `prepare.py` sao chép nguyên byte checkpoint 0–6 của mỗi arm, kiểm EMA,
  optimizer, scaler, epoch và updates của checkpoint resume. Chỉ đổi bốn
  đường dẫn I/O trong train_args. Control cũ 7–9 không thuộc paired study.
- `resume.py` bắt buộc start_epoch=7, epochs=10, seed=164, rect=True,
  batch=4, AMP=False. Dùng cùng helper KD, teacher R101 và hệ số R145 cho
  cả hai arm. Lịch sử được ghép đúng bảy epoch cũ và ba epoch mới; không
  lựa chọn giữa các lượt restart theo AP. Cache nhãn riêng, không dò NPY gốc.
- Checkpoint Ultralytics lưu EMA và optimizer, thiếu raw training model và
  RNG đầy đủ. Resume đối xứng giảm một yếu tố gây nhiễu giữa hai arm, nhưng
  không biến lượt này thành tiếp nối bit-exact. Phần batch thất bại và tail
  control cũ vẫn là chi phí tính toán đã dùng, dù không tham gia lựa chọn.
- CUDA_LAUNCH_BLOCKING=1 áp dụng cho cả hai arm. Nếu lặp lỗi, giữ batch CPU
  cuối và traceback để điều tra; không suy nguyên nhân từ tên operation
  xuất hiện cuối. Event nvlddmkm cùng thời điểm chưa chứng minh root cause.
- `monitor.py` chọn trên đúng 10 checkpoint/arm bằng AP50–95, tie AP50 rồi
  epoch sớm hơn, trên monitor300; sau chọn mới chấm dev7190. Đây là dữ liệu
  phát triển đã sử dụng nhiều vòng, không phải kết quả test mù.
- `export_recovered.py` xuất đúng kiến trúc sliced đã chọn, đủ 5 lớp, 3 scale,
  shape 1×3×384×640 và output 1×9×5040; 10 fixture kiểm ORT/PyTorch trước
  khi chuyển Nano. Không thay checkpoint theo điểm runtime sau đó.
- Runtime R169 dùng cùng package CUDA-pre R162 cho original/control/joint50.
  Không ghép SiLU R167 vào thí nghiệm pruning này. Pipeline này chỉ được
  mở sau recovered_exports PASS và R167 hoàn tất timing/collection.
- Raw parity vẫn giữ ngưỡng 6 px/0,02. Actual TensorRT AP dùng cùng 7.190 ảnh,
  NMS và evaluator đã khóa; macro giảm tối đa 0,5 điểm phần trăm và AP50 mỗi
  lớp tối đa 2 điểm so cả original và matched control. FAIL vẫn giữ nguyên.
- Final FPS chỉ được bắt đầu khi swap <100 MiB, cùng boot/guard đã có.
  Pilot dương mới mở đủ 80 đoạn. Original MOT/CLS phải khớp R109. Không gọi
  GPU-only speedup, số tham số, MAC hay pilot4 là FPS end-to-end full80.

`summarize.py` mới chỉ là bước hậu kiểm sau hoàn tất; nó không nằm trên
đường huấn luyện/đo đang chạy. Nó xác minh lại hash, 10 checkpoint mỗi arm,
3×80 lượt, 36.000 inference và cùng package R162, rồi báo riêng compression,
FPS, AP từng lớp, HOTA, IDSW, FN và FP. Không thêm cổng hay chọn điểm thắng
sau khi xem kết quả. Mặc định, launcher, metric và khóa luận giữ nguyên.
