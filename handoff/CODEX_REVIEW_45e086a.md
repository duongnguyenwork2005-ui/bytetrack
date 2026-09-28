# Review vòng 20 — `45e086a`

Ngày review: 27/09/2026. Reviewer: Codex.

**Kết luận: PASS WITH LIMITATIONS cho benchmark tracker-from-cache và gói bàn giao hiện tại.** Có thể đóng mốc benchmark CV↔EKF trên Nano. Không có phát hiện buộc chạy lại benchmark này. Công cụ đóng gói vẫn còn lỗi CRLF cần sửa; kết luận PASS không có nghĩa lỗi đó đã được khắc phục, hay hệ thống end-to-end đã được nghiệm thu.

## 1. Phạm vi và tính toàn vẹn

- Gói: `review_delta_latest.zip`, **4.396.467 byte**, 238 thành viên.
- SHA256: `0b42dc93607b9b2240c031e67c4decad19aaa5beaa8911ef0ccb59f9ebdf8c35`.
- Base: `a5eec3963d960fb817b3d83d2b30ec3c12a01795`.
- Target: `45e086ac47d778c463719b9b29398b6d99078542`.
- Kiểm ZIP, 232 payload và các đường dẫn bị cấm: đạt. Áp patch lên snapshot base đã giữ, đối chiếu **272 đường dẫn trong phạm vi patch** với hash target: khớp.
- Một NPZ của vòng 16 bị loại có chủ ý và được khai báo trong `DEPENDENCIES.md`; không dùng file này để tính benchmark. Không tuyên bố đã tái dựng cả repository bao gồm dữ liệu bị loại.
- Cây `src/` không khác snapshot `365dc2c` đã giữ. Không chạy lại full suite; con số **33/33** là bằng chứng từ log lần trước, không phải một lượt test mới của reviewer.
- Tự băm và kiểm số byte của bản sao: **95/95 file gói v5, 5.262.092 byte**, và **3/3 log môi trường, 33.419 byte**, khớp các danh sách nguồn được bàn giao. Đây là đối chiếu bản sao với hồ sơ nguồn, không phải truy cập trực tiếp Nano.

## 2. Benchmark tính lại độc lập

Đọc trực tiếp 12 CSV timing: mỗi lượt đủ 1.470 frame theo thứ tự; bỏ 50 frame đầu, còn **1.420 frame đo/lượt**. Tính trung bình từng lượt rồi lấy hiệu EKF−CV trong từng cặp; không coi các frame là những lần lặp thí nghiệm độc lập.

Điều kiện: `MVI_39031`, cùng detection cache, backend NumPy, 6 cặp gồm 3 AB và 3 BA, 12 lượt hợp lệ, không loại lượt.

| Đại lượng | Kết quả tính lại |
|---|---:|
| Trung vị của 6 trung bình update CV | 4,695 ms/frame |
| Trung vị của 6 trung bình update EKF | 5,468 ms/frame |
| Trung vị 6 hiệu update EKF−CV | **+0,7522697454 ms/frame** |
| Min–max hiệu update | +0,640 đến +0,855 ms/frame |
| Số cặp có hiệu dương | **6/6** |
| Q1–Q3, phương pháp exclusive | [+0,7102014597; +0,8180141194] |
| Q1–Q3, phương pháp inclusive | [+0,7368377327; +0,7939852979] |
| Trung vị hiệu predict | +0,6218531711 ms/frame |
| Trung vị hiệu assoc | +0,1392979144 ms/frame |

Các giá trị update/predict tính lại khớp JSON trong sai số số thực dưới `1e-9`. Trung vị hiệu theo cặp không bắt buộc bằng hiệu hai trung vị; các trung vị thành phần cũng không bắt buộc cộng thành trung vị tổng.

**Kết luận được hỗ trợ:** trên sequence và điều kiện đã đo, cấu hình EKF có chi phí `tracker.update()` cao hơn CV khoảng **0,75 ms/frame, tương ứng khoảng 16% theo cặp**. Chênh lệch lớn nhất trong các phần được đo nằm ở predict. `assoc = update − predict` còn chứa cập nhật bộ lọc và vòng đời track; không gọi toàn bộ phần này là chi phí ghép đối tượng thuần túy.

Đây là thống kê mô tả chi phí của hai cấu hình tracker. Không suy ra FPS end-to-end, HOTA/IDF1 hoặc hiệu năng ở mọi cảnh giao thông.

## 3. Output và phạm vi bằng chứng

- So trực tiếp output **r1/r6 của cả CV và EKF** với `pc-ref` tương ứng: giống hệt sau chuẩn hóa CRLF→LF.
- Danh sách hash `pc-ref` và Mốc 1 trước/sau benchmark giống nhau.
- Output r2–r5 không có đầy đủ trong gói. Kết luận về các lượt này dựa vào log hash do bên thực thi cung cấp; log `09a_hash_output_12_luot.log` in tiền tố 16 ký tự hex. Reviewer không gọi đây là việc tự so toàn bộ output của cả 12 lượt.
- Không chạy trực tiếp Nano, detector hoặc benchmark mới trong review này.

## 4. Các việc còn cần sửa

### R20-P2 — Công cụ đóng gói vẫn làm mất CR trong patch

Vị trí: `src/make_review_delta.py`, hàm `git()` và `make_patch()`.

`git()` dùng `text=True`, nên newline được chuyển đổi khi đọc stdout. `--binary` không tự bảo vệ những hunk văn bản CRLF. Tái hiện độc lập bằng repository tạm, gọi chính `make_patch()` hiện tại:

- Thêm file văn bản 15 byte, hai dòng CRLF, thuộc quy tắc `-text` nhưng không đánh dấu binary.
- Diff Git thô có 2 byte CR; kết quả `make_patch()` còn 0.
- `git apply` **exit 0**, nhưng file tái dựng chỉ còn **13 byte**, hash không khớp target.

Đánh dấu 7 file bằng chứng là binary đã giúp **gói hiện tại** vượt kiểm hash; đó chưa phải bản sửa tổng quát.

Yêu cầu vòng sau: giữ dữ liệu patch dạng **bytes** từ stdout Git đến ZIP; việc phân tích đường dẫn dùng bản xem riêng và không ghi đè bytes gốc. Thêm regression dựng gói thật, áp patch và so byte/hash cho file CRLF thêm mới, sửa đổi, nội dung trộn LF/CRLF và file binary. Tiếp tục kiểm loại file riêng trong patch. Chỉ chạy test đóng gói liên quan nếu không thay mã tracking; không cần chạy lại benchmark vì sửa này.

### R20-P3 — Sửa phạm vi `wall_s` trong tài liệu

`nano_bench_tracker.py::chay_mot_luot()` đo quanh subprocess `[sys.executable, "src/run_tracking.py", ...]`. Lệnh `conda run` khởi động **tiến trình benchmark cha** trước đó.

Do vậy, wall 9,6–10,9 s/lượt bao gồm khởi động Python con, import, đọc cache, tracking và ghi kết quả; **không bao gồm khởi động `conda run` của tiến trình cha**. Sửa hàng wall trong brief và mục 4a/4c của `REVIEW20_CHI_TIET.md`. Wall của hai lệnh warm-up riêng có phạm vi khác. Sai mô tả này không ảnh hưởng số timing update và hiệu +0,752 ms/frame.

### R20-P3 — Diễn đạt giới hạn thống kê chính xác hơn

Thay câu “6 cặp không đủ để nói có ý nghĩa thống kê” bằng: **“Báo cáo này chưa thực hiện kiểm định hoặc xây dựng khoảng tin cậy; IQR chỉ mô tả độ phân tán của 6 hiệu số.”** Chỉ riêng số lượng cặp không quyết định một kiểm định có thể đạt ý nghĩa hay không. Không cần mở thêm kiểm định hậu nghiệm để đóng mốc này.

## 5. Giới hạn giữ nguyên và quyết định tiếp theo

Một sequence, 6 cặp; thiếu xung CPU theo lượt và tegrastats thô; có tác vụ nền. Nhiệt độ quan sát không chứng minh xung ổn định hoặc không throttling. Log khởi động lại và phục hồi journal không xác định nguyên nhân sự cố, cũng không chứng nhận sức khỏe thẻ SD. Hash bản sao khớp hỗ trợ tính toàn vẹn các file đã kiểm.

**Đóng benchmark ở phạm vi trên; giữ nguyên dữ liệu, cấu hình và kết quả.** Sửa công cụ đóng gói cùng hai diễn giải trong một đợt nhỏ, rồi tiếp tục hoàn thiện luồng detector TensorRT → tracker trên Nano và đo thời gian toàn luồng bằng phép đo riêng. Không cộng số detector vòng 16 với benchmark này để công bố FPS.

Có thể dùng `45e086a` làm base review kế tiếp sau khi lưu gói và báo cáo này; tiếp tục khai báo NPZ không được bàn giao. Review không sửa mã dự án, Git config, index, commit, thẻ SD hoặc thiết bị.
