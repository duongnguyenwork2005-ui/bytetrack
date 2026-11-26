# R78–R80: PF Lost-track recovery trên UA-DETRAC và BDD

## Câu hỏi và phạm vi

Theo yêu cầu đưa IDSW của PF gần các bộ lọc còn lại, kiểm ba cơ chế Lost đã có trong mã và hai cửa sổ ghép ngắn hơn. PF500-K1K2 motion2, seed 0/1/2, detector và TrackEval giữ theo R72/R74. Chỉ dùng UA-DETRAC dev12 và BDD COCO-video val200. Cấu hình mặc định và các kết quả luận văn đã khóa không bị thay đổi. Giao thức khóa trước: `review78_pf/00_protocol.json`, `review79_pf/00_protocol.json`, `review80_pf/00_protocol.json`.

| Tập / biến thể | HOTA (%) | IDF1 (%) | IDSW |
|---|---:|---:|---:|
| UA PF motion2 mốc | 57,006 | 68,361 | 697,0 |
| UA PF bỏ fuse-score cho Lost | 57,386 | 69,907 | 1.643,7 |
| UA PF IoU Lost ≤3 frame | 56,675 | 68,030 | 525,7 |
| UA PF Mahalanobis Lost ≤3 frame | 57,171 | 68,983 | 639,0 |
| UA EKF tham chiếu | 59,252 | 72,448 | 472 |
| BDD PF motion2 mốc | 32,951 | 37,086 | 6.291,3 |
| BDD PF IoU Lost ≤3 frame (R78) | 33,030 | 37,226 | 6.712,3 |
| BDD PF IoU Lost ≤1 frame (R79) | 32,951 | 37,086 | 6.291,3 |
| BDD PF IoU Lost ≤2 frame (R80) | 33,079 | 37,348 | 6.253,3 |
| BDD EKF tham chiếu | 33,827 | 38,710 | 5.548 |

PF là trung bình số học ba seed; EKF một run. UA IoU≤3 giảm IDSW 171,3 (−24,6%) so với PF mốc; bootstrap ghép cặp theo 12 video, 5.000 mẫu: chênh [−293; −73]. HOTA giảm 0,331 điểm phần trăm và IDF1 giảm 0,332 điểm. BDD IoU≤3 tăng IDSW 421 [198; 661], nên không chuyển được kết quả UA. R79 IoU≤1 không tác động: **600/600 tệp MOT trùng byte** với PF mốc, không chỉ trùng điểm. R80 IoU≤2 giảm IDSW 38 [−192; +129], khoảng đi qua 0; MOTA giảm 0,325 điểm. R80 không đạt cổng IDSW BDD ≤6.102,8 (trong 10% EKF), cũng như UA R78 còn cao hơn EKF 11,4%.

## Kiểm chứng

- `src/test_lost_recovery.py`: 11/11 phép kiểm tra đạt, gồm mặc định tắt trùng byte và fixture Lost tổng hợp.
- R78 UA smoke 18/18 và full 108/108 lượt video, TrackEval chấm đủ 9 run; R78 BDD smoke 9/9 và full 600/600 lượt video, chấm ba seed.
- R79/R80 mỗi lượt BDD smoke 9/9, full 600/600, chấm ba seed.
- Script hậu kiểm kiểm marker từng video, seed, cấu hình tracker, cache/GT qua preflight, và tái tính COMBINED từ trường TrackEval thô. R78 UA còn kiểm fingerprint GT/cache/evaluator với R72. Bảng và khoảng bootstrap: `results/phase2/kc48_r78_pf_dev/analysis/`, `results/phase2/r78_pf_bdd/analysis/`, `results/phase2/r79_pf_bdd/analysis/`, `results/phase2/r80_pf_bdd/analysis/`.

## Quyết định

Không chọn bất kỳ biến thể nào của R78–R80 cho cấu hình luận văn. IoU≤3 có lợi UA nhưng gây thêm IDSW BDD; IoU≤1 không tác động, IoU≤2 chỉ giảm rất ít và không chắc chắn. Dừng dò độ dài cửa sổ Lost trên BDD val. Để tiếp tục cải thiện PF một cách có ý nghĩa, cần tập trung vào sai số dự đoán/hậu nghiệm PF đã chẩn đoán ở R70 và kiểm trên tập xác nhận độc lập; không gọi một biến thể ghép Lost riêng của PF là so sánh thuần túy giữa các bộ lọc.

BDD val và UA dev đã được xem trong nhiều lượt phát triển; mọi so sánh R78–R80 chỉ là thăm dò, không phải bằng chứng tổng quát hóa độc lập. Số lần thử nhiều làm các khoảng bootstrap chỉ mang tính mô tả, không phải xác nhận thống kê sau hiệu chỉnh chọn mô hình.
