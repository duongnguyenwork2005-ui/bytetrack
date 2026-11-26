# R107 — chỉ dẫn reviewer phiên riêng

Đọc `00_protocol.md`, `01_method_limit.md`, `10_ket_qua.md`, rồi review trực tiếp plugin/runner/tests và JSON/log. Không coi báo cáo là PASS độc lập. Không cần GPU, ảnh gốc hay quét dữ liệu.

- `pf_postgap.py`: xác minh tiêu chí hai quan sát sau gap, phép dịch Cartesian có trọng số, bảo toàn logw/RNG/box hạt/omega, consume K1 prior một lần, method gốc phục hồi trong finally. Kiểm giới hạn báo mean/cov sau resampling so với baseline trước resampling; không phải ablation thuần vận tốc.
- `inputs.json`: 91 hash nguồn/config/GT/cache/metadata/baseline/HEAD/index. Đối chiếu cấu hình c25, lớp, buffer và seed với R95 smoke. Baseline full R72 dùng raw đã có, không full replay.
- `tier1_checks.json` và log: giữ hai lần unit fixture lỗi; chỉ lần `unit_fixed2` đạt. Hồi quy 10 hàm/21 assertion, test11 CLI chủ động không chạy vì ghi data. Smoke disabled exact array6/6; repeat candidate2/2.
- `dev/tracking/DETRAC-all/candidate-s{0,1,2}`: đủ 12 video mỗi seed, tổng 50.757 frame, event3.398. `dev/evaluation/`: raw/metrics TrackEval ba seed. Không lấy trung bình HOTA video.
- `dev_gate.json`: IDSW 697→673,67, HOTA giảm0,2286 điểm, FN tăng0,379%, seed IDSW687/660/674. Sáu video giảm/sáu tăng, CI delta[−56;+5]; ứng viên bị loại, không mở chuyển miền hay tune thêm.
- `verify_results.py`/`verification.json`: sáu combine trực tiếp HOTA/Identity, output và lineage; LocA dummy chỉ vì raw lịch sử không lưu field ấy, không báo LocA. `diagnosis_verified.json`: tham chiếu KF/EKF/UKF và đếm lại R102, không replay R102.
- Ngoại lệ cần đánh giá minh bạch: unit đầu Ultralytics tự tạo `Ultralytics/settings.json` ở gốc repo ngoài write_paths. Có log/hash; giữ nguyên, không xóa ngoài quyền. Những lượt sau settings nằm trong R107. Không xác nhận “chỉ ghi write_paths”.

Coder đã cập nhật mục0 trước báo cáo, đã chờ mọi job tự mở. Không tự xác nhận mốc review R106 hoặc review R107. Orchestrator thực hiện review Codex phiên riêng; không có sub-agent trong lượt này.
