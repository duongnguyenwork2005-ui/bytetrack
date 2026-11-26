# R81 — nhiễu quá trình đối dấu cho PF (04/11/2026)

## Mục tiêu và cách làm

Thử giảm dao động Monte Carlo của tâm dự đoán PF bằng các cặp nhiễu quá trình `+z` và `−z`. Với mỗi hạt, nhiễu vẫn có phân phối biên Gaussian `N(0,Q_i)`; các hạt trong cặp có tương quan âm. Tùy chọn `pf_antithetic_process_noise` mặc định `false`, không thay đổi bộ lọc và cấu hình thí nghiệm đã khóa. Giao thức và cổng trước full: `00_protocol.json`.

PF500-K1K2 motion2, ba seed 0/1/2, UA-DETRAC train dev12, YOLOv8n COCO c25, 25 fps, evaluator và các tham số ByteTrack như R72. Chỉ mở BDD nếu IDSW trung bình UA giảm ít nhất 10% (≤627,3), HOTA ≥56,506% và IDF1 ≥67,861%.

## Kiểm tra thực hiện

- Kiểm mới 4/4: cặp nhiễu đối dấu, mean bằng 0 với Q đồng nhất, phương sai biên xấp xỉ Q, số hạt lẻ và `multi_predict` trùng `predict` từng track khi cùng seed.
- Hồi quy PF hiện hành 26/26. Smoke hai video × sáu run đạt 12/12 lượt; full UA 12 video × ba seed đạt 36/36 lượt và ba TrackEval run.
- Nhánh mặc định R81 có raw TrackEval fields trùng hệt R72 smoke trên 2 video × 3 seed. MOT cũ R72 smoke không còn lưu nên điều kiện trùng byte MOT trong giao thức ban đầu không kiểm được; sửa cách kiểm được ghi rõ trong `00_protocol.json` trước khi chạy full.
- Phân tích kiểm marker từng video, seed, cấu hình, fingerprint cache/GT/evaluator và tái tính COMBINED từ raw fields: `results/phase2/kc48_r81_pf_dev/analysis/`.

## Kết quả UA

| PF motion2 | HOTA | IDF1 | IDSW |
|---|---:|---:|---:|
| Mốc R72 | 57,006% | 68,361% | 697,0 |
| Nhiễu đối dấu R81 | 56,972% | 68,262% | 675,7 |

IDSW giảm 21,3 (3,1%). Bootstrap ghép cặp theo 12 video, 5.000 lần lấy mẫu: chênh IDSW [−45,7; +0,7], qua 0. HOTA giảm 0,034 điểm, IDF1 giảm 0,099 điểm; cả hai khoảng chênh cũng qua 0. Giảm IDSW không đạt cổng 10%, nên **không chạy BDD và không chọn R81**. Tùy chọn giữ mặc định `false` để các cấu hình luận văn còn nguyên hành vi.

## Phân bố khoảng cách PF–EKF

Từ kết quả có sẵn R72/R74 và EKF R70/R61, `src/r81_pf/diagnose_gap.py` tạo `results/phase2/r81_pf_diagnosis/summary.json` cùng hai CSV theo video. UA: PF motion2 cao hơn EKF 225 IDSW; bốn video MVI_40213, MVI_63554, MVI_63561 và MVI_40752 đóng góp khoảng 187,7 lần (83,4% chênh lệch ròng). BDD: PF cao hơn EKF 743,3 IDSW; 141/200 video PF cao hơn, 47 thấp hơn, 12 bằng nhau. Chênh lệch BDD lan rộng hơn UA: 20 video cao nhất chiếm 55,0% chênh lệch ròng.

Chẩn đoán R70 trước đây cho thấy trong các mẫu còn liên kết GT sau 3–8 frame mất đầu ra, sai số tâm dự đoán trung vị của PF500-K1K2 khoảng 0,492 chiều cao GT, EKF 0,253; số mẫu chấm được khác nhau nên không dùng đó làm ước lượng nhân quả. Bản R81 làm nhiễu trung bình ít hơn nhưng không khắc phục chênh IDSW, phù hợp với việc lỗi còn phụ thuộc biểu diễn hậu nghiệm và quá trình ghép, song chưa chứng minh nguyên nhân cụ thể.

## Giới hạn và quyết định

UA là tập dev và BDD val đã được quan sát qua nhiều lượt. Không dùng các thử nghiệm R78–R81 để khẳng định PF tốt ngang EKF trên dữ liệu độc lập. Tránh tiếp tục dò mức roughening hoặc cửa sổ Lost trên hai tập này. Nếu muốn thay cấu trúc PF (ví dụ ước lượng điều kiện vị trí hoặc đề xuất hạt mới), cần khóa thuật toán và cổng đánh giá trước, rồi xác nhận trên dữ liệu chưa dùng để chọn thiết kế.
