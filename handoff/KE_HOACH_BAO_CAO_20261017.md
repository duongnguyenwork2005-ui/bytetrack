# Giao việc cho automation ByteTrack — báo cáo 17/10/2026

Ngày giao: 10/10/2026, múi giờ Asia/Ho_Chi_Minh.

Trạng thái tài liệu: bản giao việc để agent trên máy Windows chuyển thành cấu hình đúng schema của chương trình hiện có. Chưa nạp vào orchestrator, chưa khởi chạy; đây không phải cấu hình CLI có thể chạy trực tiếp.

## Chỉ đạo và quyền thực hiện

Người dùng yêu cầu triển khai kế hoạch bằng chương trình automation đã viết. Ưu tiên làm vững lý thuyết, kiểm tra dữ liệu và thực nghiệm kiểm chứng PF. Hoãn tối ưu Jetson Nano, chuyển TensorRT và đo tốc độ toàn hệ thống trên Nano.

Được triển khai các bước nghiên cứu, audit, sửa lỗi, kiểm thử và thí nghiệm thuộc phạm vi dưới đây bằng chương trình hiện có. Không tự mở rộng mục tiêu, ghi đè dữ liệu gốc/checkpoint/kết quả cũ, sửa quyết định D1–D5 hoặc bỏ cơ chế bảo vệ của runner. Tuân thủ giới hạn tài nguyên đã cấu hình; thiếu ngân sách cho lượt chạy lớn thì ghi rõ phần bị chặn và tiếp tục phần độc lập.

Claude-first trên toàn lượt: Claude thực hiện phần lớn đọc code, nghiên cứu, triển khai, sửa lỗi, kiểm thử, viết báo cáo. Chỉ gọi Codex tại cổng nghiệm thu quan trọng theo chính sách hiện hành; tránh review lặp mỗi thay đổi nhỏ. Claude hết hạn mức thì checkpoint và chờ reset bằng cơ chế sẵn có, không tự chuyển việc sang GPT/OpenAI. Không dựng automation mới, không dùng ChatGPT Tasks thay orchestrator này.

## M0 — Tiếp nhận trạng thái và tạo lượt kế tiếp

1. Đọc AGENTS.md, README, hướng dẫn CLI, schema plan và chính sách ngân sách của chương trình hiện tại. Vị trí được nhắc ở phiên trước: `D:\LABOTARY\automation\auto_review`; phải xác minh trên máy.
2. Kiểm tra trạng thái lượt `D:\LABOTARY\automation\runs\bytetrack_r46_3moc_20261009_025215`, lock, tiến trình sống, HEAD và thay đổi chưa commit trong repo ByteTrack. Không giết tiến trình hoặc xóa lock khi chưa xác minh chủ sở hữu.
3. Đọc báo cáo/log mới nhất để xác định phần đánh giá detector, sửa adapter và R46 đã hoàn thành; không chạy lại chỉ vì bản slide cũ ghi chưa làm. Những số liệu trong bản giao việc chỉ là mốc đối chiếu, không thay thế bằng chứng trên máy.
4. Tạo lượt kế tiếp bằng CLI/schema thật. Bảo toàn lượt cũ. Nếu có lượt đang ghi repo, dùng hàng đợi hoặc cơ chế tuần tự được hỗ trợ; không chạy hai lượt ghi cùng repo.
5. Ghi lại phạm vi, HEAD, config, trạng thái dataset, ngân sách và các mốc dưới đây. Dùng các cổng review/chờ/reset vốn có; không sửa runner để né kiểm soát.

Đầu ra: hồ sơ run mới, trạng thái ban đầu và bảng kế thừa kết quả cũ. Chỉ thông báo đã chạy sau khi có run ID, log khởi động và bước thực thi đầu tiên.

## M1 — Cơ sở lý thuyết và đối chiếu code PF

Ưu tiên hoàn thành trước các thí nghiệm tốn tài nguyên.

- Viết bản giải thích có phương trình, định nghĩa biến và tài liệu gốc về KF–CV, EKF–CTRV, PF–CTRV; mô hình trạng thái, đo, nhiễu Q/R, dt, xử lý góc và giới hạn omega gần 0.
- Phân biệt particle với random seed; SIS, SIR, suy biến trọng số và mất đa dạng sau resampling. Trình bày cập nhật trọng số tổng quát, trường hợp proposal bằng prior, ESS và chuẩn hóa log-weight.
- Đọc code để xác định proposal, likelihood, resampling, ESS trigger, ngân sách hạt, state ownership và cập nhật track mất detection. Ghi đường dẫn/hàm/commit làm bằng chứng.
- Kiểm tra từng track có mảng hạt/trọng số riêng không; phân biệt chia sẻ cấu hình/filter với chia sẻ nhầm trạng thái mutable. Không sửa cấu trúc shared filter nếu chưa xác nhận lỗi ownership.
- Không mặc định xe lớn hút hạt. Phân biệt cạnh tranh hạt toàn cục, nhầm association, thiên lệch likelihood, thiếu detection và nghèo hạt nội bộ.
- Đặt giả thuyết PF có lợi khi phi tuyến mạnh hoặc posterior khó xấp xỉ Gaussian, đồng thời nêu giới hạn do mô hình CTRV, số hạt và association. Không khẳng định PF luôn thắng EKF khi cua.

Nguồn khởi đầu: Arulampalam et al., IEEE TSP 2002, DOI 10.1109/78.978374; ByteTrack, https://arxiv.org/abs/2110.06864. Đọc nguồn gốc, xác minh trích dẫn trước khi đưa vào báo cáo.

Nghiệm thu: mỗi kết luận về triển khai có bằng chứng code; tách rõ lý thuyết, quan sát và giả thuyết chưa kiểm chứng.

## M2 — Phân bố, chất lượng và độ đủ của dữ liệu

- Đối chiếu manifest và log thực tế từng checkpoint: nguồn dataset, split, số video/ảnh/bbox/track theo lớp, cách lấy mẫu frame, cấu hình augmentation đang bật.
- Mốc đối chiếu từ slide lần 2: BDD train 1.261 video/50.487 ảnh; dev 139 video/5.559 ảnh; BDD validation 200 video/39.973 ảnh. Xác minh, không sao chép như kết quả audit mới. Làm rõ tập nào thực sự chỉ có vài chục video.
- Bảng phân bố car/truck/bus/motorcycle/bicycle gồm bbox, track độc lập khi có ID, số video chứa lớp, kích thước bbox tương đối, mật độ, che khuất và điều kiện cảnh khi có nhãn. Trường không có dữ liệu ghi unknown, không suy diễn như ground truth.
- Audit ảnh lỗi/thiếu, nhãn thiếu/sai lớp/bbox ngoài ảnh, bản chuyển nhãn YOLO và trùng/gần trùng xuyên split. Tách theo video; nhiều frame liền nhau không tính là nhiều bối cảnh độc lập.
- Kiểm tra nguồn nhãn BDD bên thứ ba, phiên bản, class mapping và đối chiếu nhãn gốc nếu truy cập được. Nếu không, báo chính xác phạm vi chưa xác thực.
- Kiểm tra trực quan mẫu phân tầng theo lớp, kích thước và điều kiện khó; lưu cách chọn mẫu, số lượng, overlay và loại lỗi. Không kết luận toàn tập sạch từ một mẫu nhỏ.
- Thống kê quỹ đạo cua với tiêu chí định trước; loại quỹ đạo quá ngắn/chậm/nhiễu theo quy tắc công bố. Với BDD, lưu ý ego-motion; gọi là đổi hướng trên ảnh nếu chưa có hiệu chỉnh chuyển động camera/mặt đất.

Đầu ra: CSV thống kê, biểu đồ, mẫu overlay, báo cáo nguồn nhãn và danh sách vấn đề ưu tiên. Không xóa hay tự sửa nhãn gốc.

## M3 — Giao thức kiểm chứng, chốt trước khi xem kết quả

1. Chọn cấu hình detector/cache đã xác thực để giữ cố định trong so sánh bộ lọc. Tuning dùng dev; tập đánh giá cuối không dùng chọn ngưỡng.
2. Thiết kế hai cấp: quỹ đạo tổng hợp có ground truth và phép đo biết liên kết để cô lập bộ lọc; sau đó pipeline ByteTrack trên dữ liệu thực. Quỹ đạo tổng hợp chỉ là kiểm chứng cơ chế, không thay thế benchmark thực.
3. Nhóm chuyển động: thẳng, cua đều, thay đổi tốc độ góc/cua gắt; kết hợp mất đo và nhiễu có kiểm soát. Không chỉ mô phỏng đúng một mô hình thuận lợi cho PF.
4. Nhóm xe nhỏ cạnh/che bởi xe lớn: chọn theo tiêu chí dữ liệu, không chọn theo kết quả PF thắng. Ghi đủ số video, đối tượng, sự kiện và trường hợp bị loại.
5. Chỉ chọn tối đa hai kỹ thuật PF có cơ sở từ M1/M2. Ưu tiên ESS/resampling hoặc regularization/proposal tùy lỗi thực tế; nếu đã có thì không ghi là đóng góp mới. Thay proposal phải cập nhật trọng số đúng tỷ số importance.
6. Nếu sửa association/gating, áp dụng tương đương cho EKF/PF và tách ablation association khỏi ablation filter. Không khóa lớp cứng theo một detection thiếu chắc chắn.
7. Chạy nhiều seed PF, ghép cặp seed nhiễu/đo ở mô phỏng; báo mean và độ biến thiên. Báo số hạt, thời gian và đường đánh đổi tài nguyên. Giữ ngân sách tuning hợp lý giữa các phương pháp.

Cổng nghiệm thu quan trọng: giao thức và correctness của bộ lọc trước thí nghiệm lớn. Chỉ dùng số lượt Codex thực sự cần theo ngân sách; không tự nới hạn mức.

## M4 — Triển khai nhỏ, kiểm chứng và ablation PF

- Thực hiện sửa lỗi có bằng chứng và tối đa hai cải tiến đã chọn, thay đổi từng yếu tố.
- Test có ý nghĩa: trọng số hữu hạn/tổng bằng 1, độc lập state giữa track, resampling hợp lệ, dt/góc, không đo, ổn định số và tái lập seed. Tận dụng test đã có, tránh viết test chỉ lặp implementation.
- Log ESS trước resampling, số tổ tiên/độ đa dạng sau resampling, độ phân tán hạt, innovation, detection được ghép, class/size, thời điểm mất detection và đổi ID. ESS sau reset trọng số không đủ chứng minh hạt đa dạng.
- Minh họa hạt tô màu theo track sở hữu ở tình huống xe nhỏ cạnh xe lớn.
- So sánh EKF, PF hiện tại, PF cải tiến: sai số dự đoán và thất bại ở cấp filter; HOTA/IDF1/IDSW, giữ ID và phân lớp/nhóm cua ở cấp pipeline. Ghi cả kết quả bất lợi và số mẫu ít.
- Bootstrap theo video khi cần CI cho dữ liệu thực; không coi các frame tương quan là mẫu độc lập. Không tái chạy full matrix nếu smoke đã chỉ ra lỗi chưa giải quyết.

## M5 — Kiểm chứng dữ liệu và augmentation có giới hạn

Chỉ tiến hành sau audit và giao thức, trong ngân sách sẵn có. Không để phần này làm chậm báo cáo lý thuyết.

- Dùng một detector trước, cùng khởi tạo và dev cố định. Xem cấu hình đã bật augmentation nào, chọn một thay đổi phù hợp điểm yếu thực tế.
- Nếu cần đường cong lượng dữ liệu, dùng tập con lồng nhau theo video (dự kiến 25/50/100%), có cách lấy mẫu tái lập và kiểm soát mức hiện diện lớp hiếm.
- Ghi rõ số optimizer steps, epoch, số ảnh và compute. Không quy mọi khác biệt cho lượng dữ liệu khi ngân sách tối ưu hóa cũng đổi; chọn giao thức so sánh phù hợp và nêu giới hạn.
- Chỉ augmentation train; không tăng ảnh lặp rồi gọi là tăng cảnh độc lập. Bảo toàn bbox/nhãn, kiểm tra trực quan biến đổi. Nếu thử chuỗi tracking, phải giữ nhất quán thời gian.
- Đánh giá AP từng lớp và đối chiếu miền BDD/UA-DETRAC theo protocol hiện hành; không đổi quy tắc ignore/mapping D1–D5. Không diễn giải domain shift là thiếu dữ liệu trước khi loại lỗi đánh giá.

## M6 — Tổng hợp trước báo cáo

Mục tiêu tiến độ: 10–11/10 tiếp nhận + lý thuyết/audit; 12/10 giao thức; 13–15/10 thí nghiệm; 16/10 chốt nội dung; 17/10 báo cáo. Đây là mục tiêu, không tự tạo lịch chạy hay hứa hoàn tất khi thiếu tài nguyên.

Đầu ra cuối: báo cáo lý thuyết có nguồn, bảng audit dữ liệu, bảng đối chiếu code, protocol, kết quả ablation và giới hạn, hình trường hợp điển hình, đề cương/slide báo cáo. Mỗi số liệu phải truy về run/config/commit/dataset và file kết quả. Không tự bịa số hoặc biến phần chưa chạy thành hoàn thành.

Ưu tiên tối thiểu trước 17/10: lý thuyết vững + audit dữ liệu + chẩn đoán PF + một thí nghiệm kiểm chứng cơ chế. Nếu thiếu thời gian, thu hẹp sweep/huấn luyện trước, không cắt kiểm tra tính đúng.

## Hướng dẫn giao cho agent trên máy Windows

Đọc toàn bộ tài liệu này và triển khai bằng chương trình automation đang có. Tự đọc README/schema/CLI để chuyển thành plan hợp lệ, kiểm tra trạng thái và lock, rồi khởi chạy đúng cơ chế. Người dùng đã yêu cầu chạy phạm vi trên; không xin lại xác nhận chung. Không đoán tên cờ CLI. Nếu thiếu truy cập hoặc bị cổng bảo vệ chặn, báo đúng blocker và hoàn thành các phần độc lập được phép.

Sau khi khởi chạy, trả về: run ID, đường dẫn plan, HEAD, PID hoặc định danh tiến trình do runner cung cấp, trạng thái, mốc đang chạy và đường dẫn log. Báo kết quả theo bằng chứng; không coi việc viết plan là đã khởi chạy.
