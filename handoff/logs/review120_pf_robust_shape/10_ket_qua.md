# R120 — loại biến thể làm mềm phép đo kích thước

Giao thức được lưu trước điểm ở `00_protocol.md`. Mã `shape_robust.py` tăng R riêng aspect/height theo innovation chuẩn hóa, ν=4 cố định. Ý tưởng giảm tác động hộp đo bất thường có liên hệ với [robust filtering bằng Student-t](https://doi.org/10.1080/01621459.1989.10478794), nhưng đây là heuristic xấp xỉ, **không phải** cài đặt posterior Student-t của bài báo. Không thay likelihood vị trí, init hai quan sát, Q hoặc association của Cartesian RBPF R118.

Numerical cho hệ số 1 khớp Gaussian, tâm/particle weights không đổi ở bước cập nhật khi chỉ tăng R kích thước, gain chiều cao giảm khi đo lệch, PSD và khôi phục hook đạt. Hai prefix × ba seed: đối chứng R118 6/6 và lặp seed 6/6 đạt. Dev đủ **12 video × ba seed = 36 lượt**, 16.919 ảnh mỗi seed, PF500.

So R118 → ứng viên: IDSW **464,67 → 619 (+33,21%)**, HOTA **58,995 → 55,610% (−3,385 điểm)**, IDF1 **71,883 → 64,613%**, FN **37.450,67 → 44.626,67 (+19,16%)**, FP **12.097 → 18.833,67**. Seed IDSW ứng viên 619/617/621. Trượt cả cổng IDSW/HOTA/FN và tiêu chí seed. Không chạy transfer, không chỉnh ν hoặc ngưỡng để cứu kết quả.

Hậu kiểm 36 marker/MOT, 59 hash đầu vào/nguồn đạt. `dev_summary.json` và `verification.json`. Chưa kiểm sự kiện hình ảnh để quy toàn bộ giảm điểm cho một nguyên nhân: việc giảm gain kích thước có thể làm hộp chậm thích nghi với thay đổi thật, và chiều cao còn ảnh hưởng Q ở các bước sau. Kết quả đủ để **loại** biến thể này, chưa đủ chứng minh mọi cách robust hóa kích thước đều không hiệu quả. Giữ nguyên R118 và backend mặc định.
