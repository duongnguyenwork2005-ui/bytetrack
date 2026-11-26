# R142 — PF kết hợp nhiễu đo thích nghi theo confidence

Hoàn tất ngày 19/11/2026. Đạt mục tiêu quan sát: IDSW trung bình của ứng viên thấp hơn các mốc KF/EKF/UKF đã khóa trên cả BDD val200 và UA-DETRAC test40. Cả ba seed đều đạt trên từng tập; HOTA và số bỏ sót cũng qua các cổng đã định trước. Chưa đổi bộ lọc mặc định, launcher Nano hoặc khóa luận.

## Thay đổi và nguyên nhân cần thử

Nhánh NumPy hiện tại gọi `STrack.update`/`re_activate` mà không truyền confidence vào `RBPFTrackView.update`. Các phép thử trước vì vậy thực sự dùng R cố định; đây là cấu hình đối chứng hợp lệ, không phải các kết quả sai.

R142 giữ nguyên PF R132: Native Cartesian RBPF100, chiều cao dự đoán để tính R, init IID, trung bình posterior, systematic resampling và Lost shape clamp. Không dùng covariance floor R141 hoặc ghép các ứng viên R133–R141. Adapter riêng chuyển score của **detection đang ghép** vào công thức đã có trong `src/ekf_ctrv.py`:

`R(c) = max(1 - c, 0.05) × R0`.

Detection có confidence cao nhận R nhỏ hơn; confidence thấp nhận R lớn hơn. R này đi vào cập nhật Gaussian và likelihood của PF. Q, cache detector, lớp, ngưỡng, association, tuổi track, số hạt và metric giữ nguyên. Không lấy lại score của khung cũ. Đây là áp dụng noise adaptation kiểu NSA, không triển khai toàn bộ StrongSORT hoặc ReID.

Nguồn tham khảo: [StrongSORT](https://arxiv.org/abs/2202.13514), [mã Kalman của tác giả](https://github.com/dyhBUPT/StrongSORT/blob/master/deep_sort/kalman_filter.py). R142 dùng công thức có sẵn trong repo, không dò floor/gain/exponent theo kết quả test.

## Kết quả chất lượng

Cùng cache YOLOv8n COCO c25, năm lớp phương tiện và giao thức chấm đã khóa. PF dùng ba seed 0/1/2. Các số thập phân về IDSW/FN/FP là trung bình ba seed, không phải số sự kiện lẻ trong một lượt.

- **BDD COCO-video val200:** 200 video, 39.973 khung/seed, đủ600 lượt. PF mới **4.788,33 IDSW**; từng seed4785/4785/4795. KF/EKF/UKF lần lượt **6.345/5.548/5.655**. PF giảm **24,53%/13,69%/15,33%** so với ba mốc đó. HOTA **35,4798%**, IDF1 **40,0929%**, FN258.585,33, FP15.603,67.
- **UA-DETRAC test40:** 40 video, 56.340 khung/seed, đủ120 lượt. PF mới **2.362,67 IDSW**; từng seed2347/2364/2377. KF/EKF/UKF lần lượt **2.498/2.525/2.543**. PF giảm **5,42%/6,43%/7,09%**. HOTA **59,2696%**, IDF1 **70,8616%**, FN196.218,67, FP70.503,33.
- **UA train dev12, cổng phát triển:** đủ36 lượt, PF440,67 IDSW (448/436/438), HOTA61,2376%. Không dùng kết quả này thay cho test40.

Đối chiếu trực tiếp với **cùng PF R132 nhưng R cố định** tách riêng tác động của confidence:

- BDD: IDSW **5.464,67 → 4.788,33**, chênh−676,33; HOTA **33,7055% → 35,4798%** (+1,774điểm phần trăm); FN−2.479,33, FP−2.124.
- UA test40: IDSW **2.520 → 2.362,67**, chênh−157,33; HOTA **58,4594% → 59,2696%** (+0,810điểm phần trăm); FN−3.364,33, FP−2.889,67.

## Độ biến thiên và phạm vi suy luận

Bootstrap ghép cặp theo video, 5.000 mẫu, dùng chung chỉ số video giữa các phương án; không coi ba seed là ba dataset độc lập.

- BDD, CI95% của IDSW R142−EKF: **[−939,68; −585]**; so cùng PF R132: **[−825; −528,32]**.
- UA test40, CI95% của IDSW R142−KF: **[−212,67; −58,33]**; so cùng PF R132: **[−219,34; −95,67]**.
- CI IDSW so KF/EKF/UKF trên cả hai tập đều nằm dưới0 trong phép bootstrap này. CI HOTA so cùng PF R132 đều dương. Riêng CI IDF1 so parent trên UA vẫn chứa0; không gọi mọi chỉ số đều cải thiện chắc chắn.

Các tập đã được dự án quan sát nhiều vòng; bootstrap này không sửa được selection bias hoặc thay thế xác nhận trên tập chưa dùng để phát triển. Các baseline KF/EKF/UKF vẫn dùng R cố định. Do đó, kết luận là **cấu hình PF + R theo confidence thắng các cấu hình đối chứng đã khóa về IDSW trong thử nghiệm này**. Chưa chứng minh thuật toán PF thuần tốt hơn khi mọi bộ lọc đều được cấp cùng R(c). Confidence phân lớp cũng chưa được hiệu chỉnh thành độ bất định định vị.

## Chi phí xử lý trên PC

Đo tuần tự42 lượt trên hai prefixBDD200 đã chọn bằng workload50/90 từ trước, seed0, ba lần đảo thứ tự bảy phương án, bỏ16 khung đầu; mỗi lượt184 lần cập nhật được đo. Không chạy job chất lượng/bootstrap đồng thời. Lặp output và đối chiếu prefix với kết quả full đạt.

Mean `tracker.update`: KF **1,055 ms**; EKF **1,281 ms**; UKF **3,173 ms**; PF cổ điển500 **5,634 ms**; Native100 **4,900 ms**; R132 **4,893 ms**; R142 **4,776 ms**.

R142 có chi phí gần parent; không khẳng định chênh lệch nhỏ về timing có ý nghĩa thống kê. PF vẫn tốn CPU hơn KF/EKF/UKF. Đây chỉ là hai prefix trên PC, không phải FPS toàn pipeline hoặc kết quả trên Jetson Nano; không cộng phần trăm với các lượt đo khác.

## Hậu kiểm và tái hiện

`final_verification.json`: **PASS**, `goal_observed_met:true`,756 lượt chất lượng mới,42 lượt timing,756 marker,1.366 hash đầu vào/nguồn. Dùng lại240 lượt đối chứng UA và720 lượt parent ablation đã xác minh hash (120UA +600BDD).

- Numerical360 bước khớp parent khi cấp cùng score; routing current/explicit/None/off, không dùng score cũ, context restoration, PSD và RNG đạt. Smoke có6 parent controls,6 route-off controls và6 cặp lặp; route-off giữ kết quả R132.
- Đủ MOT, lớp, timing, marker và số khung; điểm COMBINED tái hợp từ raw fields khớp. Nguồn/input đã khóa được kiểm lại trước và sau phân tích. Smoke/preflight trước full, không chọn lại seed, video hoặc nới ngưỡng.
- BDD bootstrap và parent ablation dùng seed128; UA bootstrap dùng seed133, mỗi phân tích5.000 mẫu theo giao thức.

Các lệnh đã hoàn tất bằng môi trường `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe`:

1. `test40_job.py summary`, rồi `test40_job.py bootstrap`.
2. `parent_ablation.py`, dùng lại raw R132 đã khóa.
3. `measure.py` sau khi mọi job chất lượng/phân tích kết thúc.
4. `finalize.py`, xác nhận đầy đủ nguồn, kết quả và các cổng trước chốt.

Chứng cứ: `final_summary.json`, `final_verification.json`, `numerical.json`, `bdd/{summary,bootstrap,verification}.json`, `test40/{summary,bootstrap,verification}.json`, `parent_ablation{,_verification}.json`, `serial_timing.json`, `timing_inputs.json` và các log tương ứng.

Đã cập nhật `handoff/PLAN_DA_CHOT.md` mục0 trước báo cáo này. Không còn job chạy. Dừng tìm ứng viên theo mục tiêu đã đạt; các bước riêng nếu được giao tiếp là ablation cùng R(c) cho các bộ lọc và kiểm PF mới trên Nano. Không sửa `src/`, mặc định, slide hoặc khóa luận trong vòng này.
