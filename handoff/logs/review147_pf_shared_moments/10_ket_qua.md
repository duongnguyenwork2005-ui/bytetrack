# R147 — bảo toàn jointmoment sau systematic resampling với confidence-R chung

Đã chạy đủ36lượt mới UA train dev12×3seed. Giữ PF R142100hạt và chỉ thêm transform R135 giữ mean/cov5D(omega,body), không trộn orderedR146. Controls commonR R144 giữ nguyên vàđối chiếu hash.

PF449/438/443 IDSW, trung bình443,33; KF449, EKF439, UKF439. HOTA61,2316%, FN35.080,67, FP9.959,67. MeanPF>439 và chỉ1/3seed<=439 nên không đạt cổng. HOTA/FNguardrailsđạt. Không mởBDD/UA test40/kịchbản hoặcđổi sốhạt/epsilon/gain/seedđểcứu.

NguồnR135 kiểm hash qua finalproof/inputs. Numerical200mixtures độc lập giữ mean/cov,720scored-control steps qua resample/RNG/PSD, enabledcorrectionthựcsự được kíchhoạt đạt. Smokecontrols6/repeat6/PFparentparity6/candidate repeat6 vàfullraw/MOT/lớp/timing/marker/sourcehash36lượtđạt. Dữliệu đãquan sát nhiềuvòng, chưa đo tốc độ; khôngsửa src/default/Nano/khóa luận. PLAN mục0 cập nhật trước báo cáo.

R148 làaudit độc lập các mốc fullKF/EKF/UKF commonR mà người dùng yêu cầu, khôngtunePF hoặc biến R147 thành nhánh đãđạt. DùngPF R142 cũ nguyên cấu hình chỉ đểđối chiếu; các thấtbại pháttriển này vẫn đượcgiữ.
