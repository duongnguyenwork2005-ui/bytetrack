**Review vòng 45 — delta `3e94b21` → `d4715aa`**

**Kết luận: REQUEST CHANGES — chưa nghiệm thu bản dự thảo vòng 45 tại `d4715aa`. Kiểm tra toàn vẹn gói đạt.**

Có hai điểm P2 về đặc tả/diễn giải D3–D5 và hai lỗi P3 trong script thống kê. Đây không phải phát hiện regression của thuật toán tracking hay kết quả BDD. **D1–D5 vẫn CHƯA ĐƯỢC NGƯỜI DÙNG DUYỆT.** Review không khóa phương án, không cho phép inference, training hoặc SSH Nano. Giữ review base ở `3e94b21` trong khi xử lý các điểm dưới đây.

**Gói thực sự đã mở và kiểm tra**

| Mục | Kết quả |
|---|---|
| Tệp đầu vào | `review_delta_latest(1).zip` |
| Kích thước | 487.762 byte; 80 thành viên; 1.536.452 byte giải nén |
| SHA256 ZIP | `b1c207dfef8ebadcc7177039aea7a3eb8acb29d14610370a1cca2767ce96cd3d` |
| Base | `3e94b21683b7e63c5852a7ddc04baf513952ddc4` |
| Target | `d4715aa56045c0fd76cddf464adc0a45a86e70d1` |
| Commit do manifest khai báo | `d3b5460`, `3c2958e`, `d4715aa` |
| CRC / đường dẫn ZIP | Đạt; không trùng tên, không đường dẫn thoát thư mục, không symlink |
| Manifest | 74 payload và 1 alias khớp byte/SHA256; JSON và CSV khớp |
| Patch | Đủ 75 đường dẫn; áp ngược/xuôi đều exit 0; 3 preimage đúng hash base, 72 file thêm biến mất khi áp ngược; áp xuôi khớp 75/75 target hash |
| Bằng chứng review `3e94b21` | SHA256SUMS khớp 43/43, kể cả báo cáo trùng nội dung được đóng gói một lần |
| Source fixture lưu trong bằng chứng cũ | 19/19 tệp khớp inventory và hash target trong AUTHOR_MANIFEST |

SHA256 patch: `1aa1473cee8e3a79a099ce6ce0b20529cddc1316a976d43c5f38446a060bd47d`.

Đã đọc đầy đủ brief ở gốc và trong `files/`, `REVIEW45_CHI_TIET.md`, dự thảo `06_du_thao_giao_thuc_detrac.md`, năm script phân tích vòng 45 và toàn bộ log mới. Đã đọc/đối chiếu tất cả CSV/JSON của vòng 45, báo cáo và log review cũ, các harness tái hiện cũ, evaluator và entry point BDD liên quan đến giao thức được đề xuất. Patch của ba tệp sửa được đọc trực tiếp; nội dung các tệp thêm được đối chiếu với payload qua phép áp patch và hash. Toàn bộ 80 tệp được kiểm byte; 27 file Python được parse AST.

Các bản `train_detector.py`, `tiep_tuc_huan_luyen.py` và dependency lịch sử nằm dưới thư mục bằng chứng là snapshot hỗ trợ của review trước. Kiểm hash của chúng không được ghi thành một lượt nghiệm thu lại training. Không chạy lại suite BDD 22/22 hay scorer 13/13 ở vòng này; những con số đó là bằng chứng lịch sử đã đối chiếu.

Các vị trí dưới đây dùng đường dẫn tương đối của repository; trong ZIP, tệp thường nằm dưới `files/` hoặc `logs/`.

**R45-F1 — P2: D3 mô tả lớp N/A “không FP” nhưng scorer được giữ nguyên vẫn đếm FP**

Vị trí: `handoff/REVIEW45_CHI_TIET.md`, dòng 87; dự thảo giao thức, dòng 74 và 105–107. Mã đối chiếu: bản `eval_detector_5lop.py` trong bằng chứng `CODEX_3e94b21/input/src/`, hàm `danh_gia`, dòng 226–251.

Phụ lục nói prediction của lớp N/A “không chấm: không TP, không FP”, đồng thời dự thảo chỉ thêm bộ đọc GT và tái sử dụng scorer/lưu kết quả. Thực tế scorer duyệt cả năm lớp, tính FP và precision ngay cả khi lớp không có GT. Chỉ AP trả NaN/null, recall/F1 trả NaN/null và macro loại lớp đó.

Fixture reviewer dùng scorer nguyên byte: có một GT car, một GT bus, hai detection khớp chúng và một prediction truck ở vị trí khác, conf 0,7. Đầu ra truck là:

```json
{"n_gt": 0, "AP50": null, "AP50_95": null,
 "van_hanh": {"TP": 0, "FP": 1, "FN": 0, "precision": 0.0, "recall": null, "f1": null}}
```

Macro vẫn đúng hai lớp, mAP50 = 1. Đây là mâu thuẫn giữa mô tả đầu ra dự kiến với hàm được chọn, **không phải lỗi tính mAP hiện tại**. Nếu sao chép nguyên entry point BDD, JSON/CSV còn có thể trình bày precision = 0 cho truck, trái với D3 yêu cầu N/A.

Đề nghị sửa trước khi khóa: ghi rõ rằng scorer có thể tạo bộ đếm thô cho năm lớp, nhưng chỉ car/bus là kết quả đánh giá theo D3; các chỉ số lớp ngoài tập chấm phải hiển thị N/A và không được diễn giải FP thô như chất lượng detector. Nếu muốn thực sự loại chúng khỏi mọi bộ đếm đầu ra, phải mô tả bước xử lý/chuẩn hóa báo cáo trong adapter, thay vì nói “chỉ thêm phần đọc GT”. Sau khi D3 được duyệt, fixture đi qua entry point thật phải kiểm cả JSON và CSV. Không cần tự đổi phép ghép/AP của evaluator để sửa lời mô tả này.

**R45-F2 — P2: D5 khẳng định chiều thay đổi recall/AP và gộp quy tắc toolkit vượt quá bằng chứng**

Vị trí: `handoff/REVIEW45_CHI_TIET.md`, dòng 89; đối chiếu chính tài liệu đó dòng 75–79 và dự thảo giao thức dòng 132.

Hàng D5 nói phương án (b)/(c) bỏ khoảng 41 nghìn GT và “recall / AP tăng”. Chỉ đếm GT giao vùng ignore không chứng minh được chiều metric. Cùng scorer của dự án, fixture gồm hai GT car, một GT trong ignore được detect đúng, một GT ngoài ignore bị bỏ sót:

| Quy tắc | GT | TP | FN | Recall | AP50 |
|---|---:|---:|---:|---:|---:|
| Giữ GT trong ignore | 2 | 1 | 1 | 0,5 | 0,5049504950 |
| Chuyển GT đó thành ignore | 1 | 0 | 1 | 0 | 0 |

Đây là phản ví dụ cho mệnh đề tăng chắc chắn, **không phải ước lượng metric UA-DETRAC thật**.

Ngoài ra, 35.248 + 5.587 = **40.835** chỉ là số GT car/bus đạt IoA theo phép đếm `max_don` của tác giả. Chính hồ sơ ghi `parseGT.m` chỉ loại ở hai đầu quỹ đạo trong đường MOT, còn xử lý GT đường DET nằm trong executable chưa đọc được. Vì vậy không thể dùng cùng con số để khẳng định toolkit DET sẽ bỏ toàn bộ nhóm GT này. Script đếm cũng dùng box XML gốc; chưa phải số GT sau cắt khung và loại box nhỏ theo bộ đọc GT dự kiến.

Đề nghị: sửa thành “thay đổi tập GT và cách tính TP/FP/FN; recall/AP có thể tăng hoặc giảm, chưa biết mức tác động khi chưa có prediction”. Tách rõ phương án tự định nghĩa (c), quy tắc MOT đã đọc và phần DET chưa xác minh. Giữ 40.835 dưới nhãn “GT XML đạt ngưỡng trong thống kê này”, không biến thành số GT chắc chắn bị loại bởi toolkit. Không chạy GPU để chứng minh chiều tăng và không chọn D5 theo metric test.

**R45-F3 — P3: phương án giữ 173 ảnh bỏ sót 265 lượt vùng ignore tĩnh**

Vị trí: `handoff/logs/review45/dem_theo_phuong_an.py`, dòng 26–35, 48 và 50–53; đầu ra `phan_tich/dem_theo_phuong_an.json` và log `04_dem_theo_phuong_an.log`.

Script chỉ thêm ảnh có `<frame>` vào danh sách `anh`. Với phương án giữ ảnh thiếu `<frame>`, nó tăng số ảnh và số ảnh không GT, nhưng không cộng `ignored_region` cấp sequence cho các ảnh được giữ thêm. Vì thế tổng vùng ignore được báo giống hệt phương án loại 173 ảnh.

Đối chiếu đầy đủ `anh_khong_frame.csv` với `theo_video.csv`:

| Video | Ảnh giữ thêm | Vùng tĩnh mỗi ảnh | Lượt vùng bị thiếu |
|---|---:|---:|---:|
| MVI_39051 | 69 | 2 | 138 |
| MVI_39211 | 94 | 1 | 94 |
| MVI_39271 | 4 | 4 | 16 |
| MVI_40743 | 1 | 6 | 6 |
| MVI_40763 | 4 | 2 | 8 |
| MVI_40793 | 1 | 3 | 3 |
| Tổng | 173 | — | **265** |

Số đúng của trường `vung_bo_qua_cong_moi_anh`, nếu giữ ảnh và vẫn áp dụng ignore cấp sequence:

| D1 | D2 | Đang báo | Cần báo |
|---|---|---:|---:|
| van bỏ qua | others bỏ qua | 275.724 | **275.989** |
| van bỏ qua | others → truck | 258.809 | **259.074** |
| van → car | others bỏ qua | 237.205 | **237.470** |
| van → car | others → truck | 220.290 | **220.555** |

Fixture CPU gọi nguyên `main()` của script trên sequence giả có hai ảnh, một `<frame>` và hai vùng tĩnh: script báo 2 lượt vùng thay vì 4 khi giữ cả hai ảnh. Đề nghị bổ sung đóng góp vùng tĩnh cho các ảnh thiếu `<frame>`, rồi cập nhật JSON/log. Các phương án loại 173 ảnh, tổng GT và tổng số ảnh không bị lỗi này. Lỗi nằm ở bảng thống kê, chưa phải lỗi của bộ đọc GT UA-DETRAC vì bộ đọc đó chưa được viết.

**R45-F4 — P3: `doc_xml()` làm mất thuộc tính `sequence_attribute`**

Vị trí: `handoff/logs/review45/phan_tich_detrac_test.py`, dòng 64; trường xuất CSV tại dòng 177.

Biểu thức `(root.find("sequence_attribute") or ET.Element("x"))` dùng truthiness của Element. Trên Python 3.12 của reviewer, một Element có thuộc tính nhưng không có phần tử con vẫn là false, nên bị thay bằng Element rỗng. Fixture `<sequence_attribute camera_state="stable" sence_weather="cloudy"/>` trả `attr = {}`. Trong bảng bàn giao, cả 40 hàng đều trống hai cột `camera_state`/`weather`; không thể dùng các ô này để suy ra nguồn XML thiếu metadata.

Đề nghị kiểm `node is not None` rồi lấy `dict(node.attrib)`; tái xuất hai cột trên nguồn XML thật và kèm một fixture cho phần tử tự đóng. Lỗi không làm thay đổi tổng car/bus/van/others hoặc danh sách 173 ảnh.

**Những phần đã đối chiếu đạt và giới hạn tương ứng**

- Cộng 40 hàng `theo_video.csv` khớp 56.167 frame XML, 56.340 ảnh, 173 ảnh thiếu frame, 0 frame thiếu ảnh, 675.774 box và 163 vùng tĩnh. Đã kiểm toàn bộ 173 hàng, sáu đoạn frame và đường dẫn ảnh, không chỉ các hàng mẫu.
- Tổng box theo lớp khớp bảng tổng hợp; phép cộng theo video và quan hệ chênh lệch giữa hai cách tính IoA khớp `vung_bo_qua.json`/log.
- CSV phân loại bằng mắt giữ đúng khóa track/box của CSV mẫu. Chạy lại `ghi_phan_loai_bang_mat.py` trên bản sao CSV tạo đầu ra trùng byte. Tổng `others` là 31 track/16.915 box; nhóm cứu hộ là 7 track/9.171 box. Mẫu van là 48 track/13.940 box; nhóm minibus là 2 track/1.036 box. **Điều này xác minh phép tổng hợp, không xác minh mắt nhìn hoặc taxonomy.**
- Mô tả quy tắc ghép D5(a) khớp scorer đã đóng gói: ưu tiên GT hợp lệ, chỉ xét ignore khi không ghép; max IoA từng vùng, ngưỡng bằng IoU đang xét; GT trong ignore vẫn có thể là TP/FN. Tính phù hợp khoa học của lựa chọn này vẫn là quyết định chưa được duyệt.
- Brief, phụ lục và handoff giữ trạng thái “chưa khóa/chưa chạy/chờ duyệt”. Trong patch không có thay đổi `src/` hoặc config hoạt động; mã core xuất hiện dưới thư mục bằng chứng lịch sử. Không phát hiện hành động vượt phạm vi qua nội dung delta.
- Ghi nhận review `3e94b21` giữ đúng giới hạn của báo cáo cũ; không tự chuyển số test tác giả hoặc reviewer trước thành test của vòng này. `review_base.json` giữ 13 khai báo chưa bàn giao. Giá trị `vong: 44` là vòng của base đã nghiệm thu, không tự nó chứng minh gói nhầm vòng.

**Điểm cần làm rõ khi hoàn thiện dự thảo**

1. Dự thảo mục 7 yêu cầu bảng hash ảnh dựng trước khi khóa, nhưng trình tự đầu tài liệu và phụ lục lại đặt kế hoạch khóa trước bước dựng bảng hash. Thống nhất quy trình: sau khi D1–D5 được duyệt, hoàn tất dữ liệu fingerprint/fixture cần thiết, rồi khóa phiên bản kế hoạch tham chiếu đúng các hash và commit trước inference. Không sửa đè kế hoạch đã khóa sau khi xem kết quả.
2. Hàm `luu_du_doan()` hiện lưu `Path.name`. Hai ảnh `MVI_A/img00001.jpg` và `MVI_B/img00001.jpg` đều thành `img00001.jpg` trong NPZ. `idx` vẫn giữ vị trí, nên đây chưa phải bằng chứng AP sai. Tuy nhiên cần lưu danh sách **có thứ tự** của đường dẫn `video/frame` và đối chiếu nó khi replay; hash của danh sách không thay thế chính danh sách. Đây là lưu ý thiết kế cho adapter chưa viết, không phải regression BDD.
3. Kích thước 960×540 mới được script kiểm ở ảnh đầu/cuối mỗi video. Không gọi đó là kiểm đủ 56.340 ảnh. Dự thảo đã yêu cầu kiểm mọi ảnh trước prediction; cần giữ gate đó.

**Bằng chứng hiện thiếu để kiểm độc lập dữ liệu thật**

ZIP không có ảnh/crop, XML nguồn, manifest video thật, toolkit MATLAB đã được tác giả đọc, checkpoint, NPZ prediction thật hoặc archive ZIP `3e94b21`. Vì vậy reviewer không thể tự đếm lại 675.774 box từ XML, xác minh trực quan MVI_39271 f78, kiểm taxonomy van/others, kiểm hash checkpoint từ byte thật hay xác minh chính xác toolkit đang cài. Lời khai “archive trước khi đổi base” chỉ có log và báo cáo hỗ trợ; không phải thao tác đã quan sát trực tiếp trên máy tác giả.

Nếu muốn nâng mức xác minh D4/D5 ở gói sửa, có thể bổ sung đúng tệp nhỏ cần thiết: manifest video/testlist; bản mã toolkit liên quan kèm hash/version; ảnh minh họa có khóa video/frame và vùng ignore cho nhận định trực quan trọng yếu. Không cần gửi lại toàn repository hoặc checkpoint để xử lý bốn lỗi đã xác nhận.

**Kiểm reviewer thực hiện trong lượt này**

Môi trường Linux, Python 3.12.14, NumPy 2.3.5. `audit_review45.py` hoàn tất exit 0: kiểm toàn vẹn, đối chiếu mọi bảng, tái xuất CSV phân loại và bốn probe nêu trên. Exit 0 nghĩa audit hoàn tất, **không** nghĩa đối tượng review đã PASS. Có thêm một probe minh họa tên ảnh trùng trong NPZ.

Không chạy training, model inference, GPU hoặc SSH. Không import Torch/Ultralytics, không cài dependency, không sửa byte nào trong gói nguồn. Tất cả fixture dùng dữ liệu tổng hợp hoặc bản sao CSV; không tạo bộ đọc GT sản xuất, không áp dụng D1–D5.

Kèm `CODEX_REVIEW_d4715aa_EVIDENCE.zip`: audit script, ZIP đầu vào nguyên byte, output JSON, inventory/hash của cả gói, README và bản báo cáo này. Có thể sửa tài liệu/script phân tích để đóng R45-F1–F4 trước; việc duyệt D1–D5 và cho phép triển khai/chạy UA-DETRAC vẫn là bước riêng của người dùng.
