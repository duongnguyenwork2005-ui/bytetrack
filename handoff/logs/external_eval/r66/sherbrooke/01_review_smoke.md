# R66 Sherbrooke — review cổng smoke trước full (02/11/2026)

**Đính chính, 15:45.** Biên bản dưới là kiểm sơ bộ của Codex. Quyết định review đã lưu lúc 15:17 tại `D:/LABOTARY/automation/runs_local/hoi_codex_R66_review_smoke_sherbrooke_20261102/answer.json` là **CAN_SUA_TRUOC_FULL**. Nhận định mở full trước đây bị thu hồi. Codex dừng job đã mở sớm và giữ riêng kết quả cũ. Lượt xác minh B1–B7 phải đạt trước khi mở ma trận chính thức.

- Nguồn tác giả xác nhận đoạn 1.001 frame và 30 fps. SQLite `cars` có 2.015 box/15 ID, gồm ID loại `unknown`; bộ chuyển đổi giữ đủ ID, ánh xạ 0→1, dùng `w=x2−x1`. GT full bỏ đúng 3 box theo tâm ngoài mask; GT smoke không bỏ box. Hash SQLite và mask ghi trong `data/processed/Sherbrooke-cars-f2754-3754-roi-v1/gt_log.json`.
- Mask 800×600 chỉ có {0,1}, 2.012/2.015 tâm GT nằm trong vùng 1. Đã xem overlay frame 436 và 600: đường biên ROI và box hiển thị phù hợp. Cùng phép thử tâm được dùng cho GT và đầu ra tracker; đầu ra thô được giữ riêng.
- Cache ba detector đi qua kiểm fingerprint, provenance, thời gian 1–128 và ánh xạ lớp. Ma trận smoke xong 36/36; hậu kiểm output thô, PF cùng seed trùng byte, tái tổ hợp raw fields khớp COMBINED tới 2,2e-16. Bằng chứng chi tiết: `smoke/00_smoke.md`, `smoke/tong_hop_trang_thai.json`.
- D: còn khoảng 52,5 GB, F: còn khoảng 162 GB trước lượt full; ước tính trong smoke dưới 20 phút. Hiện không thấy tiến trình tải hay GPU đang chạy. Lượt full cần deadline, nhật ký ngân sách và kiểm lại đầu ra.

**Giới hạn phải giữ trong báo cáo:** xe đỏ đỗ/chờ ở góc dưới trái nằm trong ROI nhưng thiếu GT, có thể làm FP tăng; dữ liệu chỉ có một góc máy và 15 ID, không dùng để kết luận tổng quát hay chọn bộ lọc. Điểm cars-ROI HOTA/IoU không so với thước ABSDIST của bài UrbanTracker gốc. Giấy phép mở của ảnh/GT chưa được xác minh, không phân phối lại.
