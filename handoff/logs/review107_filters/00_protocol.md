# R107 — PF: vận tốc từ quan sát sau khi ghép lại

Phạm vi: người dùng giao trực tiếp R107 ngày 12/11/2026. Mục 0 lúc tiếp nhận còn ghi R106 chờ review độc lập; phiên này không xác nhận thay reviewer. Không chạy inference/GPU, không đổi tracker/config chính thức, không BDD val hoặc UA test để chọn. Mọi tệp mới nằm trong R107. Không agent con.

## Căn cứ và giả thuyết (trước chạy)

R102 `pf_association_replay.json`: camera 2 seed0 có 13 IDSW trong đoạn rẽ, 7 track cũ Lost; bằng chứng này không chứng minh nguyên nhân chung. R96 phân nhóm GT-match cho thấy còn IDSW sau ghép lại; R69/R102 và code `CTRVSTrack._init_motion_if_needed` cho thấy vận tốc chỉ khởi tạo một lần. R75/R77/R78–81/R87/R91–95 đã có kết quả âm hoặc chuyển miền không đạt: không lặp scalar roughening, cửa sổ Lost, xác suất chế độ hay sửa association.

Một ứng viên duy nhất: sau một khoảng không được cập nhật >=1 bước, chờ **hai quan sát liên tiếp** (quan sát ghép lại và quan sát bước kế). Sau update PF chuẩn tại quan sát thứ hai, tính vận tốc ảnh từ hai tâm detection thật / dt. Nếu tốc độ đạt ngưỡng khởi tạo hướng sẵn có, dịch đồng đều vận tốc Cartesian của các hạt sao cho trung bình có trọng số bằng vận tốc quan sát. Đổi trở lại speed/heading; không đổi omega, hộp, trọng số, số hạt hay RNG. Tính lại mean/cov từ hạt; K1 prior đã được update chuẩn tiêu thụ và xóa. Không sửa output quá khứ, không nội suy quan sát, không update likelihood lần hai. Đây là heuristic dùng lại hai quan sát có tương quan, **không tuyên bố posterior Bayes chính xác**. Có thể hại nếu ghép sai hoặc tâm detection nhiễu. Chỉ tác động dự đoán sau này, không cứu cặp đã bị từ chối ngay lần xuất hiện đầu.

## Dữ liệu và thứ tự

- Dev: đúng manifest `results/phase2/manifests/detrac_dev.json` (12 chuỗi train), cache `yolov8n-coco_conf0.05_all`, conf>0.25, classes 2/3/5/7, max_det 300, 25 fps, elapsed skip, buffer30. PF500 K1K2 motion2; seed 0/1/2; cfg `configs/phase2/trackers/r72/motion200.yaml` không sửa.
- Baseline: tái dùng raw TrackEval R72 `results/trackeval/DETRAC-all/p2-kc48_r72_pf_dev-A-motion200-s{seed}`. Tái hợp trực tiếp HOTA/IDF1/IDSW/FN/FP; không rerun phần đã có bằng chứng. Smoke plugin tắt trên prefix128 hai chuỗi MVI_20052/20062 phải trùng mảng MOT baseline R95 smoke với cùng seed. Hash nguồn/config/cache/GT và metadata vào `inputs.json` trước dev.
- Unit: phép dịch vận tốc có trọng số không đều, giữ box/omega/logw, covariance hữu hạn/PSD, thời gian gap, bỏ qua quan sát liên tiếp bình thường, dừng/đi chậm, không tiêu thụ RNG; fixture tracker có gap; plugin tắt parity, ứng viên chạy lặp cùng seed trùng MOT.
- Smoke thật: hai chuỗi định trước × prefix128 × ba seed; so plugin tắt/bật, lặp ứng viên seed0. Chỉ kiểm đường chạy/parity/tái lập, không dùng điểm prefix để chọn tham số.
- Dev: một cấu hình × ba seed × 12 chuỗi, chạy CPU tuần tự. TrackEval HOTA/CLEAR/Identity trên GT hiện có, cùng preprocessing; lưu raw đủ tái hợp COMBINED và per-video. Kiểm MOT hữu hạn/box dương/không trùng frame-ID, số frame, hash đầu vào sau chạy. Giữ âm.

## Cổng khóa trước điểm R107

Dev qua nếu đồng thời: mean-seed IDSW <=90% baseline; HOTA >=baseline-0.005 (thang 0..1); FN <=100.5% baseline; **từng seed IDSW không tăng**; số video giảm IDSW > số video tăng (so trung bình seed). IDF1/FP/timing chẩn đoán, không gọi là đo Nano.

Nếu qua dev, chỉ kiểm chuyển miền **RoundaboutHD-prefix3000 bốn camera**, cùng cache mixed5 c25 R73, buffer18, 15 fps, seed0/1/2, baseline motion2 R73; không tune. Cổng chuyển miền tương tự IDSW giảm >=5%, HOTA không giảm >0.005, FN không tăng >0.5%, không seed nào tăng IDSW. Tập này đã xem nhiều lần nên không phải holdout mù/độc lập; không chọn bằng ví dụ camera2 R102. Nếu trượt dev không mở chuyển miền. Không BDD val/UA test.

## Giới hạn, ngân sách và báo cáo

Chỉ một biến thể, không quét tham số/ablation mới. Tổng thời gian trong hạn 7.200 giây của lời gọi; runner nhận deadline, kiểm từng frame và dừng có PARTIAL nếu thiếu thời gian. Không job nền sau kết thúc. Không nhận kết luận PASS từ báo cáo cũ; phân biệt raw tái kiểm và mô tả lịch sử. Cập nhật mục0 PLAN trước `10_ket_qua.md`. Review Codex phiên khác do orchestrator, không tự sinh reviewer. Không kiểm mù vì dev và nguồn chẩn đoán đã quan sát.
