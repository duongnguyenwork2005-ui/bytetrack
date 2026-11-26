# Chi tiết đối chứng trước mọi điểm tốc độ sau recovery

Mọi model final, kể cả original và matched control, dùng cùng package CUDApre
R162 đã kiểm bit-exact. Chỉ sửa assertion package của runner R162 từ tập hai
package sang singleton R162; giữ 80 đoạn/150 ảnh/warm25/KF/full inference,
rotation thứ tự, kiểm hash, đối chiếu original MOT/CLS với R109 và guard cũ.

Pilot giữ bốn đoạn R162 (index0,10 trong mỗi BDD/UA). Chỉ student có meanFPS
pilot cao hơn original mới vào full80. Nếu có student đạt, luôn giữ matched
control và original trong full80 để đo đánh đổi kiến trúc. Nếu không, lưu
negative pilot, không mở full. Không bỏ control vì tốc độ control thấp.

Quality gate áp dụng so cả original và matched control: macro AP50/50-95
không mất quá .005, mỗi class AP50 không mất quá .02. Raw parity FAIL hoặc
quality FAIL vẫn giữ nguyên; nếu đo tốc độ để báo trade-off thì ghi rõ là
diagnostic, không gọi PASS về chất lượng hoặc tự thay launcher/default.

Đây là diễn giải triển khai của protocol00, không nới cổng hay đổi dữ liệu.
