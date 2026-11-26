# R168: lấy mẫu phân tầng chưa giảm IDSW

Đủ36lượt,12video UA dev và3seed. GuidedPF R153 dùng100hạt, giữQ/R/init/
ESS/association; chỉ thay innovationnormal bằng randomizedLHS ởpredict và
guidedproposal. SeedIDSW443/442/439, mean441,33 so parent437,67, EKF435,
UKF430 vàKF449. Mean/two-seeds<=430 đều trượt, khôngpromote/full/transfer.

HOTA61,23413%,FN35068,67 so parent61,22110%/35071: hai guard đạt nhưng
không bù việc IDSW cao hơn. Pairedbootstrap12video,5000draw,mean ba seed
cố định: deltaIDSW+3,6667,CI95[-2;+11]. Đây là dev đã xem nhiều vòng,
không chứng minh tổng thể tốt hơn hoặc tệ hơn chắc chắn từ khoảng này.

Numerical/smoke/sourcehash và36MOT/CLS/timing/marker/raw kiểm lại PASS,
final_verification.json vẫn giữ quality_gate_pass=false. Từng innovation có
marginalGaussian và đủstrata trong kiểm số học. Phương sai samplemean toy
thấp hơn IID không bảo đảm giảm IDSW sau chuỗi lọc/ghép cặp phi tuyến.
Không phải fullSQMC và không phải stratifiedresampling củaR140/R146.
Không đo tốc độ PC trong lúc đang train YOLO.

Nguồn sampling: [McKay, Beckman, Conover1979](https://doi.org/10.1080/00401706.1979.10489755).
Không dùng kết quả samplemean toy để quảng bá phần trăm tăng tracking.
