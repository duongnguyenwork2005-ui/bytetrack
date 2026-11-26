# R87 — PF ước lượng điều kiện: kết quả

Giao thức khóa trước chạy: `00_protocol.md`. Ứng viên `pf_rb_observed_estimate` chỉ bật trong `configs/phase2/trackers/r87/rb_observed.yaml`, mặc định mã là `false`. Các phép thử hàm analytic, seed và mặc định đều đạt. Smoke baseline ba seed cho tệp MOT trùng byte với R73.

Full SVMOT dùng cùng cache, GT và 9 chuỗi R73, 3 seed, tổng 27 lượt chuỗi. Dữ liệu đọc máy: `result_svmot.json`, `source_svmot_full.json`.

| Chỉ số COMBINED, trung bình 3 seed | R73 PF motion100 | R87 ứng viên | Thay đổi |
|---|---:|---:|---:|
| HOTA | 50,2893% | 50,2985% | +0,0091 điểm |
| IDF1 | 52,5980% | 52,5991% | +0,0011 điểm |
| MOTA | 46,8817% | 46,8627% | −0,0190 điểm |
| IDSW | 436,33 | 436,33 | 0% |

IDSW theo seed: 433→430, 446→447, 430→432. Cổng HOTA không giảm quá 0,5 điểm đạt; cổng giảm ít nhất 10% IDSW **không đạt**. Theo giao thức không chạy RoundaboutHD, không chọn ứng viên, không đổi cấu hình luận văn/Nano. SVMOT đã dùng ở R73, do đó đây là phép thử thăm dò. Việc giảm nhiễu ước lượng hộp hầu như không giải quyết nguồn IDSW của PF.
