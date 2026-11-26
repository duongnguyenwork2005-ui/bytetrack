# Hậu kiểm toàn BDD200 và UA40

Đã tái dựng CLEAR trên1920output đã có, khớp IDSW/CLR_FN từngvideo và
kiểm source/MOT/GT/hash trước/sau. Không chạy tracker mới hay thay metric.
R148 cùngmeasurementnoise vàR154 headingnoise là hai cấu hìnhGaussian;
PF R142 lấymean3seed. Detector làYOLOv8nCOCOc25, không trộn với NanoYOLO11n.

Trên toànBDD200, KF/EKF/UKF commonR có5319/4854/5028IDSW. Nhưng riêng
15649hộp thuộc đoạn rẽ>=25độ trên ảnh, IDSW là268/277/277 và FN
11946/11981/11975. CI95 pairedvideo cho deltaIDSW soKF là[-7;25,025] và
[-9;27], không chứng minh lợi thế rẽ. Xe máy rẽ chỉ39hộp,36FN,IDSW0mọimode;
không dùng số0 này để kết luận tracking xe máy tốt.

UA40 toànvideo có2319/2359/2372IDSW. Nhóm rẽ gồm2432GT:11/9/9IDSW,
FN800/800/806; CI95 haiGaussian−KF đều[-5;0]. Đây là tín hiệu nhỏ, khoảng
còn chạm0. Nhóm visibility<=.5 gồm69504GT:313/321/331IDSW và
57166/57176/57188FN. Nhóm vừa che trong1giây có672/682/685IDSW. Nhóm
phục hồi sauche có359/361/354; chỉUKF cópointestimate thấp hơnKF ởnhómnày.
Không quy mọiFN cho detector khi chưa kiểmcandidatecoverage riêng.

NhánhGaussian noise-floor cũng không đảo kết luận tổng: BDD rẽ278/275
soKF268; UA che nặng328/343 soKF313. PFmean BDD rẽ278,33,UA rẽ9,
UA che345. Mọi nhóm, mọiseed và haiGaussianversion được giữ trong
summary_v2.json/partial_v2.json; không chọn cấu hình yếu hơn đểPF thắng.

Đây là hậu kiểm trên dữ liệu đã xem nhiều lần. Curve là chuyển hướng điểm
đáy hộp trên ảnh, không chứng minh quỹ đạo thế giới phi tuyến; BDD camera
di chuyển còn cóego-motion. Cờvisibility gốc chỉ dùng choUA. Bootstrap
5000lần theo video,seed166, không hiệu chỉnh nhiều phép thử; PFmean giữ
cố định ba seed nên khoảng đó không mô tả toàn bộ biến thiên MonteCarlo.
Kết quả mẫu lớn không lặp lại mọi lợi thế ở40đoạn150frame trước đây.
Không thay prerequisite full,khóa luận haydefault từ nhóm hậukiểm.

V1 dừng vì bridge dùngkeyFN thayCLR_FN. Giữ run.py/run.log/inputs.json;
v2 sửa đúng têntrường, không sửa CLEAR hoặc outputgốc. Ghi chú01 và
inputs_v2.json lưu đườngdẫn vàhash; summary_v2.json statusPASS nghĩa là
hậu kiểm toànvẹn đạt, không nghĩa làEKF/UKF/PF đã thắng kịchbản.
