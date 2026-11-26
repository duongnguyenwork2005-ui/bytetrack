# R105 — bàn giao cho phiên Codex review độc lập

Mốc coder: 30/30 benchmark thực và 8/8 smoke đã kiểm/chấm; Tier 1 hiện tại exit 0, 20/20 test. Chưa có kết luận của reviewer. Chỉ R105, chưa chuyển R106/R107.

Đọc `10_ket_qua.md`, `tier1_checks_current.json`, `verification.json`, rồi xem source và raw được trỏ tới. Source thực thi là `host_evidence/executed_sources/`, **khác bundle local mới hơn** ở `executed_to_current.diff`. Không coi manifest `deployed=false` của bundle mới là bằng chứng Nano chưa chạy; không coi test trên bundle mới là bằng chứng bundle đó đã chạy.

Các điểm cần review trực tiếp:

- `eval_benchmark.py` nhận schema sớm chỉ với runner SHA đã ghim, snapshot/driver/engine/ONNX/archive khớp. Sáu hash nguồn LF mỗi run nhất quán theo A/B nhưng thiếu hash/snapshot toàn bộ baseline package; đánh giá đúng giới hạn này. JSON raw không bị bổ sung field giả.
- `host_evidence/{smoke,full}/evidence.tar.gz`: 38 lượt, 5.700 frame liên tiếp; FPS 125/(t150−t25), p90 phần tử sắp thứ tự, gain là trung vị tỷ số từng cặp. Kiểm config KF/c25/classes/map/regions, số inference, không skip BDD và MOT/lớp guarded/full trùng 2/2.
- Điểm `host_evidence/full/assessment.json` dùng GT/evaluator R102/R103; trường A=full/B=candidate. Regression MOT 16/16 với R102; không dùng tốc độ R102 làm đối chứng R105. Các CSV từng lượt và từng ID mới cho phép đối chiếu quality/first-match và ID không được khớp.
- Đánh đổi: UA stride2 +87,98% FPS so v21combo nhưng FN full→stride2 tăng 101, first-match +80 ms/14 ID; motion/guarded kém về FPS và p90 so full tăng. MVI_39031 vẫn đủ 150 inference, chậm gần 7%; giữ kết quả âm. Không suy rộng lên mọi clip 25 fps.
- Swap-in tăng bốn trang, không tăng swap-out; telemetry có query EMC bị từ chối. Host process-empty là bằng chứng bridge ghi, không phải coder tự SSH kiểm mới. Tất cả subprocess PC đã chờ xong.

Lệnh Tier 1 tái lập được ở `10_ket_qua.md` và `run_tier1.py`; dùng Python đã chỉ định, `-B`, stage `pc_stage_verified/` đã có trong R105. Fixture cuối dùng `pc_test_scratch/` trong R105. Không cần chạy lại Nano; không cài gói, không ghi protected/config/model/data/Git. Nếu cần chấm lại PC thì dùng stage mới trong write_paths R105, vì evaluator từ chối ghi đè stage. Báo cáo ghi rõ hai staging ban đầu đã dùng scratch được nêu trong plan; không xóa chúng. Chưa có phép đo NMS bổ sung; yêu cầu có điều kiện qua bridge ở `12_bridge_nms_request.md`, hiện không có job đang chạy cho việc đó.
