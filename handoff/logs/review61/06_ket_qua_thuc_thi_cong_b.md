# R61 — Kết quả cổng B BDD100K val và đánh giá cuối mốc 5

Ngày 27/10/2026. **Cổng B hoàn tất** trong phiên `results/phase2/r61_bdd/thuc_thi/cong_b_20261027_01/`. Đây là kết quả chạy trên dữ liệu BDD thật, không phải fixture CPU của gói chuẩn bị. Phạm vi và quyết định M5-Q1…Q8 ở `handoff/logs/review61/01_quyet_dinh_va_giao_thuc.md`; kế hoạch thực thi thủ công được người dùng ủy quyền trong chat và ghim ở `handoff/logs/review61/05_ke_hoach_thuc_thi_cong_b.json`. Không chạy automation.

## 1. Bằng chứng vận hành và tính toàn vẹn

- File phê duyệt `results/phase2/r61_bdd/phe_duyet/cong_b_20261027_01.json` ghim SHA256 kế hoạch `874876309182c507c68f77505819e1341291e9d9ba973ec1bb4993505e0209a9`, gói A `5c9db320e4965798f50652b823b7a20439410e259752dbf72ebfe990d30fe225`, khóa A `f3a138d7497727a205abfa78237f0ec9a6029b89a13bdd018928865351794d4f`, hợp đồng B `bf2ebe97af6a7d70164dbc6597fdd79cdb0a086d0d34096dc26b6c6b313c05ed`, mã B `297b9361c88dacce3486b6df6b86642c9f5df34f4c1d76906bde099fdc7e757d`. `tien_kiem_b/ket_qua.json`: cả 11 nhóm đạt, gồm PLAN nguyên byte, review A approve, gói/khóa/phụ lục A, payload và ảnh khóa hiện tại.
- Runner `van_hanh/so_crash.json`: 32 lần gọi, 9 exit 0, 23 exit 10 để tiếp tục đúng bước/phiên, **0 crash**, không exit 2/30/40. Lần `0015_tracking` mất 6.960,663 giây theo đồng hồ tường (máy bị gián đoạn lâu); nhật ký vẫn kết thúc exit 10, không ghi crash. Không dùng khoảng này để ước lượng thông lượng.
- `tracking`: 36 run × 200 video = **7.200 marker hợp lệ**. `cham`: **36 run**. `lap/ket_qua_b204a5c1-05981158.json`: **36/36** run lặp trên video định trước có `trung_byte=true`. `cham_lai/ket_qua.json`: **6/6** lần chấm lại có `trung_byte=true`.
- `tong_hop/kiem_chung.json`: một ma trận chỉ số `default_rng(0)` kích thước **5.000 × 200**, lưu ở `tong_hop/chi_so_bootstrap.npy`; 20 khối × 250 mẫu; `tai_tinh_tu_chi_so_luu_trung=true`; **108 CI**. Sai số tái tổ hợp cực đại HOTA và IDF1 là `5.55e-17` (dung sai `1e-9`), IDSW bằng 0. SHA256 bảng chính `6de32d32cfa5a72d9e26c8074421996f6321dc1a84772da46435d2a67463d318`; SHA256 bảng CI `1bfe15c2bf3c6924c123f8e02b39f6cbc37d776557e98c383c79b5d8941873eb`.
- `kiem_khoa/20261027_211257/ket_qua.json`: `dat=true`, `loi=[]`. `hau_kiem.json`: `dat=true`, `loi=[]`, `khoi_lech=[]`, đếm cache **600/600**, tracking **7.200/7.200**, chấm **36/36**. Hậu kiểm tái tính ở `hau_kiem_b/vong_01/` và so byte CSV/kiểm chứng. Không sửa mã tracking, metric, cấu hình, PLAN hoặc gói A trong khi chạy.

## 2. BDD100K val — kết quả mới của R61

Nguồn số: `tong_hop/bang_chinh.csv` (**24 dòng** = 3 detector × 2 điều kiện × 4 bộ lọc; PF500 là trung bình metric COMBINED của 3 seed) và `tong_hop/ci_108.csv`. HOTA là chỉ số chính; IDF1 và IDSW bổ sung. Giá trị HOTA/IDF1 dưới đây là điểm COMBINED trên thang 0–1, không phải trung bình 200 điểm video.

- Điểm HOTA cao nhất trong ma trận là **0,45499** (`yolov8n_mixed5`, c10, EKF), IDF1 **0,55170**, IDSW **11.067**. Trong đúng nhóm này, CV có HOTA 0,45298, UKF 0,45406, PF500 trung bình 0,38988. Hiệu EKF−CV HOTA **+0,201 điểm phần trăm**, CI bootstrap 95% **[+0,081; +0,325] pp**. Hiệu PF500−EKF HOTA **−6,511 pp**, CI **[−7,044; −5,980] pp**.
- Cả 6 nhóm detector × điều kiện đều có HOTA EKF cao hơn CV, CI EKF−CV HOTA không chứa 0. PF500 có HOTA và IDF1 thấp hơn CV/EKF/UKF ở cả 6 nhóm; các CI tương ứng không chứa 0. Với UKF−CV HOTA, cả 6 CI đều chứa 0. Đây là các so sánh **trong từng nhóm** theo giao thức; 108 CI không hiệu chỉnh đa so sánh và không dùng làm CI so detector hoặc c10−c25.
- Các giá trị điểm cho `yolov8n_coco` thấp hơn hai detector mixed5 trong ma trận này, nhưng giao thức không tạo CI so detector. `c10` và `c25` cũng không tách riêng tác động của association điểm thấp; không diễn giải hiệu giữa hai điều kiện như một hiệu ứng nhân quả.

Giao thức BDD dùng COCO-video val, 200 video, 5 fps, buffer 6 bước, `mot_crowd`, lớp [1,2,3,5,7], và giữ frame không annotation trong timeline. Đây là tập val đã từng được quan sát; 615 ảnh không annotation chưa rõ nguyên nhân và được chấm theo chính sách GT hiện có. Không suy ra các ảnh đó là âm tính thật, không đánh giá riêng che khuất hoặc các lớp ít mẫu.

## 3. UA-DETRAC test40 — bằng chứng đã nghiệm thu trước R61

Phần này **không chạy lại** trong cổng B R61. Nguồn là R53/R56/R57/R59 đã nghiệm thu, được kiểm kê trong `handoff/logs/review60/01_ho_so_quyet_dinh_moc5.md`; phân tích biến thiên ở `handoff/logs/review57/01_bien_thien_test40.md`; đối chứng SORT ở `handoff/logs/review59/03_ket_qua_sort_test40.md`. PLAN mục 0 ghi R57A review_002 approve và R59B review_005 approve.

- R56: 3 detector × c25/c10 × CV/EKF/UKF/PF500 ba seed = 36 run ByteTrack trên **40 video test**. R57 ước lượng biến thiên hậu nghiệm bằng bootstrap ghép cặp theo video và theo 39 nhóm; giao thức được cố định **sau khi đã xem điểm test**, nên CI có tính mô tả, không phải tiền đăng ký.
- R59: 6 cặp SORT–ByteTrack **chỉ c25**, 240 đơn vị SORT, 6 chấm, 2 lặp tracker, 2 chấm lại, hậu kiểm và kiểm khóa đạt. HOTA ByteTrack cao hơn SORT ở cả 6 cặp, chênh **+1,380 đến +3,714 pp** theo `review59/03` mục 4; đây là hiệu số điểm, **không có CI** cho đối chứng SORT. Trong thiết lập c25, tầng detection điểm thấp của ByteTrack rỗng nên hiệu số này không đo đóng góp của vòng association điểm thấp. R59 không chạy SORT c10.

Không gộp số BDD và UA-DETRAC hoặc so trực tiếp HOTA giữa hai tập: nguồn nhãn, lớp, fps/buffer, xử lý ignore và tập video khác nhau. Kết luận mốc 5 chỉ áp dụng cho từng tập và giao thức tương ứng. Đầu ra chi tiết và các giới hạn riêng của UA-DETRAC giữ ở báo cáo R57/R59 đã nghiệm thu.

## 4. Trạng thái sau cổng B

Công việc thực thi B và báo cáo cuối trong kế hoạch `05_ke_hoach_thuc_thi_cong_b.json` đã hoàn tất. `handoff/PLAN_DA_CHOT.md` vẫn nguyên byte theo khóa A nên mục 0 còn trạng thái lịch sử; chuyển khóa và cập nhật PLAN là **bước riêng sau B** theo `review61/04_chuan_bi_cong_b.md` mục B2.8. Chưa đưa đầu ra B lên GitHub trong phiên chạy này; Git đang giữ HEAD/trạng thái đã ghim để hậu kiểm.
