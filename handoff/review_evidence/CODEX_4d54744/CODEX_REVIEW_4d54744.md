# Review vòng 42 — target `4d54744`

Ngày review: 2026-10-06. Kết luận: **PASS WITH LIMITATIONS** trong phạm vi bằng chứng training được bàn giao. Không phát hiện lỗi chặn buộc sửa runner hoặc chạy lại training. Có ba mục đính chính báo cáo mức thấp, nêu dưới đây.

Hai lượt đã hoàn tất theo các nhật ký được cung cấp: YOLOv8n đủ 15 epoch qua ba lần chạy; YOLO11n đủ 15 epoch trong một lần. Kết luận này không phải nghiệm thu chất lượng trên test, không xác nhận mọi tình huống phục hồi, và không tự phê duyệt bước tiếp theo.

## 1. Gói và phạm vi kiểm độc lập

| Mục | Giá trị |
|:--|:--|
| ZIP | `review_delta_latest.zip` |
| Dung lượng | 741.857 byte |
| SHA256 | `f144a834607324b6a94197612d7851ec0e28137bdf85f4a2b8efc43a0d2b92df` |
| Base | `6c8e37aca2326451f0a46f3a8838b6171fe66b3d` |
| Target | `4d54744e4163d6f2c3ef3e9e0f67299bba00d349` |
| Nội dung | 113 thành viên ZIP; 107 payload, 6 đường dẫn dùng payload trùng nội dung |

Reviewer đã thực hiện:

- Kiểm `testzip`, đường dẫn, hash và kích thước mọi payload, giải quyết đủ 6 khai báo trùng nội dung.
- Áp patch ngược rồi xuôi trên **113 đường dẫn thay đổi**: khớp 113/113 hash target; 8 file có ở base khớp hash base khai báo. Không có đường dẫn riêng bị cấm trong tập này, không có checkpoint/video/NPZ/ZIP.
- Kiểm 43/43 file bằng chứng review `6c8e37a` theo `SHA256SUMS.txt` trong gói. Mã `tiep_tuc_huan_luyen.py` ở base khớp bản nguồn trong bằng chứng trước.
- Đối chiếu AST sau khi bỏ docstring: thay đổi duy nhất của mã dự án ở `src/tiep_tuc_huan_luyen.py` chỉ là lời mô tả; biên dịch cú pháp đạt. R41-N1 đã được đính chính đúng: kiểm nội dung mới và hash file tạm trước `os.replace`, không đọc lại CSV đích trong hàm.
- Tự viết/chạy `audit_training.py`, đạt **128/128 phép đối chiếu hồ sơ**. Đây không phải 128 test của dự án. Kết quả, công thức và CSV 30 epoch nằm trong gói bằng chứng của reviewer.

**Không có checkout độc lập toàn bộ Git base/target.** Phép roundtrip của reviewer chỉ bao phủ đường dẫn thay đổi và các hash được bàn giao; không chuyển tuyên bố kiểm toàn cây của tác giả thành kết quả tự kiểm.

Reviewer không chạy training, inference, test BDD/DETRAC, nạp tensor checkpoint, kiểm lại dataset ảnh/nhãn hay SSH Nano; không sửa mã dự án và không commit/push. Các file đó không có trong delta. 54 file lớn mới và 13 đầu vào cũ được khai báo, nhưng hash khai báo không thay thế việc có byte nguồn để kiểm độc lập.

## 2. Training, resume và chọn checkpoint

| Đối chiếu | YOLOv8n | YOLO11n |
|:--|:--|:--|
| Epoch hoàn tất | 1–15, mỗi epoch đúng một lần | 1–15, mỗi epoch đúng một lần |
| Lần chạy | 1–7; 8–11; 12–15 | 1–15 |
| Exit từng lần | 130, 130, **0** | **0** |
| Checkpoint tốt nhất theo fitness đã ghi | epoch **15** | epoch **13** |
| `final_eval` dev mAP50-95 | 0,2801804207 | 0,2721987133 |
| `final_eval` dev mAP50 | 0,4351549368 | 0,4163501648 |
| Tổng thời gian runner | 103.524,9 s = 28,757 h | 108.820,3 s = 30,228 h |

Nhật ký, `results.csv` và bốn JSON runner khớp về thứ tự epoch, metric và phần epoch của từng lần. Mỗi epoch hoàn tất ghi 6.311 batch và loss hữu hạn. `mosaic_thuc_te`/`mosaic_p` ghi tắt ở epoch 14–15. Bảng validation cuối theo lớp khớp log lọc; đây là validation **dev nội bộ của Ultralytics**, không phải test độc lập hay kết quả từ evaluator 5 lớp của dự án.

Hai lần resume đều nối từ checkpoint đã lưu gần nhất: epoch 7 → 8 và epoch 11 → 12. Hash nguồn được chọn, bản sao nguồn resume, bản sao theo epoch trong nhật ký và manifest file lớn khớp nhau. Cả hai JSON ghi 13/13 kiểm tra trạng thái đạt. Biên nhận CSV ghi không cắt hàng lịch sử. Không có bằng chứng tăng tổng epoch hoặc đổi cấu hình học giữa các lần.

Danh tính cấu hình, dataset, nguồn mã và Ultralytics trainer được ghi nhất quán giữa bốn lần chạy. Tham số chung khớp: 15 epoch, batch 8, workers 2, SGD, seed 0, warmup 1, close_mosaic 2, AMP/cache tắt. Mỗi lần ghi đã nạp đúng 50.487/5.559 ảnh và kiểm đầu vào đạt. Đây là đối chiếu **bản ghi kiểm dữ liệu**, không phải reviewer tự băm lại 56.046 ảnh.

Các hash `best.pt` cần giữ làm định danh cho bước sau:

- YOLOv8n: `5f66776d3202389ca484f3d9e2f107dddb354d21a9f1762d19401d690a4a0d8a`.
- YOLO11n: `d7d1c7e86d03c69d913febe79e4cbb6c2e1bcf9f17f034724af4de2ccfba7923`.

Hash và dung lượng `best`/`last` trong JSON runner, JSON tóm tắt và manifest file lớn khớp. Đủ 30 khai báo checkpoint theo epoch khớp nhật ký. Reviewer **không trực tiếp nạp các file `.pt`**; các cờ nạp lại, đúng 5 lớp và kiểm tensor epoch 11 là bằng chứng do tác giả ghi.

Thời gian tính lại: runner hai model **58,9848 h**; tổng cửa sổ các tiến trình giám sát **59,1735 h**; từ lúc khởi chạy đầu đến kết thúc cuối **68,4717 h**. Các phạm vi này khác nhau. YOLO11n bắt đầu sau YOLOv8n hoàn tất, không chồng hai lượt training.

## 3. Các đính chính không chặn nghiệm thu

### R42-N1 — Ghi đúng nguồn số đo RAM và phạm vi VRAM/RSS

**Mức thấp, chỉ báo cáo.** `handoff/REVIEW42_CHI_TIET.md` mục 5 viết RAM máy trống thấp nhất 81 MiB. Đây là minimum trong tóm tắt bộ giám sát; JSON runner cùng lượt ghi `vram_nvidia_smi.ram_con_trong_min_mib = 6`.

Hai nguồn đều đo `psutil.virtual_memory().available`, nhưng lấy mẫu ở những thời điểm khác nhau. Vì vậy nên ghi riêng: **81 MiB theo bộ giám sát; 6 MiB theo bộ lấy mẫu trong runner**. Không gọi 81 MiB là mức thấp nhất chung của toàn lượt, cũng không suy diễn nguyên nhân NaN từ khác biệt này. CSV tài nguyên thô bị loại khỏi gói nên reviewer chưa đối chiếu được thời điểm mẫu 6 MiB.

Nguồn cụ thể:

- `handoff/logs/review42/ket_qua/yolo11n/giam_sat/yolo11n_lan0/tom_tat.json`, `ram.ram_he_thong_con_trong_mib.min`.
- `handoff/logs/review42/ket_qua/yolo11n/runner/train_bdd5_v1_yolo11n_day_du.json`, `vram_nvidia_smi.ram_con_trong_min_mib`.
- Mã `train_detector.py` trong bằng chứng review trước, lớp đo tài nguyên, xác nhận dùng `available`.

Nên ghi rõ thêm tại bảng tài nguyên: **2.933/3.123 MiB là `nvidia-smi memory.used` của cả GPU**, gồm tiến trình khác; **4.083/4.200 MiB là tổng RSS/working set các tiến trình**, có thể đếm trùng trang bộ nhớ chia sẻ. Số hiện tại giữ nguyên.

### R42-N2 — Phân biệt lúc NaN xuất hiện với lúc dừng

**Mức thấp, chỉ báo cáo.** Mục 4.1 và 7 của phụ lục ghi bỏ epoch 12 ở 73%. Trong `11_dung_luot_nan_yolov8n.log`:

- NaN bắt đầu ở batch **4597/6311 = 72,84%**.
- Dòng tiến độ cuối được ghi trước khi dừng là **4786/6311 = 75,84%**, giao diện in 75%.

Đề nghị viết “NaN từ khoảng 73%; dòng tiến độ cuối trước dừng ở khoảng 76%; toàn bộ phần epoch chưa hoàn tất bị bỏ”. Không khẳng định 4786 là chính xác batch cuối đã thực thi nếu chỉ có dòng log này. Việc resume từ checkpoint epoch 11 và số epoch hoàn tất không bị ảnh hưởng.

### R42-N3 — Năm ô AP trong bảng lịch sử làm tròn lệch

**Mức thấp, chỉ báo cáo.** Làm tròn một lần từ `results.csv` đến ba chữ số sau dấu phẩy:

| Model | Epoch | Chỉ số | CSV gốc | Phụ lục đang ghi | Giá trị đúng |
|:--|--:|:--|--:|--:|--:|
| YOLOv8n | 2 | mAP50-95 | 0,17648 | 0,177 | **0,176** |
| YOLOv8n | 4 | mAP50 | 0,35046 | 0,351 | **0,350** |
| YOLOv8n | 7 | mAP50 | 0,39345 | 0,394 | **0,393** |
| YOLO11n | 1 | mAP50 | 0,22947 | 0,230 | **0,229** |
| YOLO11n | 7 | mAP50 | 0,37348 | 0,374 | **0,373** |

Chỉ sửa bảng trình bày, giữ nguyên CSV/log. Không ảnh hưởng epoch chọn best hoặc kết quả `final_eval` nêu ở mục 2.

## 4. Giới hạn được giữ nguyên

- Mỗi model một cấu hình/seed; YOLOv8n có hai lần resume, không tương đương từng bit với lượt liên tục. Điểm dev dùng chọn best nên không dùng chênh 0,280/0,272 để kết luận model tốt hơn trên test.
- Nguyên nhân NaN chưa xác định. Chứng cứ cho thấy đã dừng và dùng checkpoint epoch 11; không chứng minh cơ chế tự phục hồi NaN của Ultralytics đã chạy hay đã an toàn.
- Khoảng trống snapshot 79,42 và 148,77 phút đúng là của YOLO11n. YOLOv8n không có khoảng trống trên 60 phút giữa các snapshot cùng cửa sổ chạy. Không nên nâng điều này thành chứng minh log RAM thô “liên tục mỗi giây”: CSV thô không được bàn giao, `0 mẫu không đọc được` không tự chứng minh không có khoảng trống giữa mẫu.
- Bằng chứng nguồn nhãn bên thứ ba, chênh native/COCO và các giới hạn đã nghiệm thu trước vẫn còn hiệu lực. Vòng này không xác thực lại nhãn hoặc thay chuẩn đánh giá lịch sử.
- Không kiểm toàn cây Git, không kiểm trọng số từ byte thực và không chạy lại suite/GPU. Không có lý do từ thay đổi docstring để mở lại suite tracking/Nano.

## 5. Hành động trong phạm vi đã chốt

Ghi nhận nghiệm thu bằng chứng training vòng 42 với các giới hạn trên. Đính chính ba mục báo cáo bằng log hiện có; **không cần chạy lại training, đổi checkpoint hay thêm thí nghiệm** để sửa chúng. Guard ghi nối và khóa lượt vẫn là đề xuất chưa triển khai.

Bước đánh giá test trên BDD val/UA-DETRAC, rồi so sánh bộ lọc, thuật toán tracking và tối ưu Nano giữ theo kế hoạch của người dùng. Review này chưa khởi chạy và không tự mở rộng hoặc phê duyệt các bước đó. Khi bước đánh giá được duyệt, dùng checkpoint đã cố định và kiểm hash tại máy local trước khi nạp.

## 6. Bằng chứng reviewer bàn giao

`CODEX_REVIEW_4d54744_EVIDENCE.zip` gồm báo cáo này, `check_package.py`, `audit_training.py`, kết quả JSON/CSV, trích dẫn có số dòng cho ba mục đính chính, hướng dẫn tái chạy và `SHA256SUMS.txt`.

Để tái chạy phép đối chiếu, dùng **đúng ZIP đầu vào có hash ở mục 1**, làm theo `README.md` trong gói bằng chứng. Không cần GPU, dataset, checkpoint hay thay đổi repo. Kết quả kiểm tra là sự nhất quán của hồ sơ được cung cấp, không phải chứng nhận thực nghiệm được tái chạy độc lập.
