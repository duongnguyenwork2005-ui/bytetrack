# Review vòng 44 — `3e94b21`

**Kết luận: PASS WITH LIMITATIONS. Đóng R43-B1 và R43-N1.**

Phạm vi nghiệm thu: bản sửa metadata, support code, đường đọc GT/chấm điểm kiểm bằng fixture, và hồ sơ BDD vòng 43 được giữ nguyên. Không còn lỗi chặn được xác định trong phạm vi này. Đây không phải xác nhận đã chạy lại inference trên dữ liệu thật, kiểm trực tiếp checkpoint hoặc nghiệm thu UA-DETRAC.

## 1. Gói đã đọc

- `review_delta_latest.zip`: **600.666 byte**, 123 thành viên.
- SHA256: `9bb1bbab25134e8615fb0794fd1d6d2968592eab25ac7d4dbc9403727d198e11`.
- Base: `4d54744e4163d6f2c3ef3e9e0f67299bba00d349`.
- Target: `3e94b21683b7e63c5852a7ddc04baf513952ddc4`.
- Review thực hiện trên Linux, CPython 3.12.14, bằng cây dựng từ gói. Không sửa repo tác giả, không train, không chạy GPU hoặc SSH Nano.

## 2. R43-B1 — ĐÓNG: dependency đã đủ cho phần kiểm BDD

17 file support khớp byte/SHA256 khai báo. Cùng với hai file mã mới, chúng cho phép import CLI và chạy nguyên `test_danh_gia_bdd_val.py` trong cây giải nén riêng.

- **22/22 PASS, exit 0** trên Linux/Python 3.12.
- Bộ đọc GT và scorer dùng mô-đun dự án thật; test tự giả lập đầu ra detector theo thiết kế. Không được diễn giải lượt này thành chạy YOLO thật.
- Chín mô-đun được harness theo dõi đều nạp từ cây giải nén; không lấy dependency nội bộ từ repo tác giả. Số chín là phạm vi danh sách theo dõi của harness, không phải đối chiếu số đếm toàn bộ import với tác giả.
- File trong cây không đổi byte sau các lượt kiểm. Hành vi tạo thư mục `models/` rỗng khi import `config.py` là hành vi sẵn có.

Đã xác nhận riêng cách giải thích EOL:

| File | SHA256 blob LF | SHA256 khi chuyển LF → CRLF | Đối chiếu hồ sơ vòng 43 |
|---|---|---|---|
| `bdd_yolo_export.py` | `2fbc231435b21d5e9d03f600de5d049863f095432a17e288128c16cf4beecb03` | `1ca964b0fc81701ac746335fa20ac6e5231322a4cdd11ad47753a2b25fc31493` | Khớp |
| `mot_datasets/bdd100k_coco.py` | `c6c3ad289e8198840cd3ffa1a21a78e00d664bd3c0b1cd7d84fbdf67dabb7705` | `5028ffb98ddff6ba423bef14ff9331fd02f8e40b1055438c88224182914f8363` | Khớp |

`eval_detector_5lop.py` và `detector_classes.py` khớp trực tiếp hash đã ghi khi đánh giá. Không có bằng chứng về thay đổi nội dung mã ở hai file chỉ khác EOL.

## 3. R43-N1 — ĐÓNG: metadata dùng ngưỡng hiệu lực

Đã đọc diff so với `a03f6e1`. `quy_tac_hieu_luc()` tạo bản sao sâu cho mỗi lượt; metadata, scorer và chẩn đoán lấy cùng ngưỡng. Hằng `QUY_TAC` không bị ghi đè. Chốt đối chiếu `van_hanh` trả về từ scorer không đổi phép ghép/AP.

| Kiểm độc lập | Kết quả |
|---|---|
| Script reviewer cũ, chạy nguyên byte trên bản sửa | **exit 0, bug_count=0** |
| Test BDD mới trên bản sửa | **22/22 PASS** |
| Cùng test mới, thay riêng entry point bằng bản `a03f6e1` trong cây tạm | **20/22**, FAIL đúng B5b và B5c |
| Ngưỡng mặc định / 0,5 / gọi 0,4 rồi trở về mặc định | Metadata và scorer khớp; mặc định vẫn 0,25 |

Script reviewer cũ dùng doubles và chỉ kiểm metadata. Nhóm B5 bổ sung bằng chứng qua scorer thật trên prediction fixture. Cả hai phạm vi đều được giữ rõ trong log.

Ba lượt BDD lịch sử dùng ngưỡng 0,25, nên lỗi cũ không kích hoạt. **Không cần chạy lại ba lượt do bản sửa này.**

## 4. Phần evaluator đã kiểm thêm khi có support

Đã đọc các hàm đọc COCO-video, phân vai GT/ignore, ánh xạ lớp, ghép detection, tính AP, P/R và lưu dự đoán. Chạy nhóm E và E7–E9 từ `test_detector_5lop.py`, độc lập với môi trường Windows của tác giả:

- **13/13 PASS, 0 FAIL; 1 SKIP** là E6 cần checkpoint/inference YOLO.
- So AP với **pycocotools 2.0.11 thật**: fixture E1 có độ lệch tối đa **0,0**; E7–E9 khớp AP50, AP50–95 và TP tại IoU 0,5.
- Bao gồm hòa IoU, IoU đúng ngưỡng 0,6/0,7, GT hợp lệ chồng vùng ignore, detection trùng, lớp không có GT, GT theo kích thước ảnh thực và thiếu file nhãn.
- Fixture BDD kiểm ảnh không annotation vẫn nằm trong tập, crowd/ignore, cắt box vào khung và fingerprint đầu vào.

**Không đồng nhất giao thức dự án với benchmark COCO chính thức:** mảng `ign` của dự án dùng chung cho mọi lớp. Test pycocotools mô phỏng chính quy tắc này bằng crowd ở từng lớp. Các kiểm trên chứng minh cài đặt phù hợp với giao thức đã khóa trong các trường hợp fixture; không xác nhận nhãn thật hoặc toàn bộ quy tắc benchmark BDD/COCO. Sai biệt 1 ulp ở ngưỡng 0,9 đã công bố từ trước vẫn là giới hạn, không phát sinh ở vòng này.

Không chạy toàn bộ suite 92 phép có Ultralytics/weights. **92/92 của tác giả vẫn là đối chiếu log**, không phải số test reviewer tự chạy. Môi trường reviewer không cần shim chặn pyarrow; lần thử đầu chỉ thiếu OpenCV, sau đó cài dependency riêng cho review và chạy thành công.

## 5. Toàn vẹn gói và kết quả

- CRC ZIP đạt, không trùng tên/đường dẫn thoát thư mục/symlink.
- **100 payload + 16 alias** khớp MANIFEST; **17 support** khớp hash/byte khai báo, trong đó hash base và target được manifest khai báo bằng nhau.
- Reverse/apply patch: **116/116 đường dẫn thay đổi** khớp target.
- **6/6 preimage** còn đối chiếu được với snapshot base đã review trước và đều trùng từng byte.
- Trong 61 đường dẫn của delta vòng 43, **57 không đổi**. Bốn file thay đổi là entry point, test BDD, brief và `CODEX_HANDOFF.md`. Các file kết quả BDD vòng 43 có trong gói giữ nguyên.
- Kết quả kiểm số học **180/180 từ review trước** tiếp tục áp dụng cho các bảng/JSON không đổi; không ghi đó thành một lượt tái tính AP mới.

Không thấy file dữ liệu/model bị cấm trong gói. Reviewer không có checkout Git đầy đủ, nên không tự xác nhận con số **4.697 file toàn cây**; đó là bằng chứng của tác giả. Cũng không thể tự xác minh hash checkpoint và chín file lớn bị loại chỉ từ bảng kê của chúng.

## 6. Kết quả BDD được giữ và giới hạn diễn giải

| Model | mAP50 | mAP50–95 |
|---|---:|---:|
| YOLOv8n COCO | 0,2439417100 | 0,1570728458 |
| YOLOv8n fine-tune | 0,3835045640 | 0,2506992858 |
| YOLO11n fine-tune | 0,3757858772 | 0,2467297056 |

Các bảng nhất quán với kết quả được bàn giao và chưa bị sửa. Reviewer **không có ảnh, GT thật, checkpoint hoặc NPZ** để lặp lại ba lượt hay tính lại AP thực tế. Việc tác giả tính lại NPZ bằng chính evaluator là bằng chứng tái lập của đầu ra, không phải một cài đặt AP độc lập trên dữ liệu thật.

Giữ các giới hạn đã công bố: mỗi model một lượt training, v8n có resume; BDD val từng được xem; khác biệt 183 lớp/50 ngoại lệ ignore giữa native và COCO chưa giải thích; P/R ở cùng confidence phụ thuộc hiệu chỉnh điểm số. Chênh khoảng 0,397 điểm mAP50–95 giữa hai model fine-tune chưa đủ để khẳng định model nào tốt hơn tổng quát.

## 7. Bàn giao và bước tiếp theo

Có thể ghi nhận **PASS WITH LIMITATIONS cho `3e94b21`**, lưu gói đã review đúng hash trên và chuyển review base sang target này theo quy ước dự án. Không cần sửa thêm hoặc chạy lại GPU để đóng R43-B1/N1.

**UA-DETRAC D1–D5 vẫn chưa được người dùng quyết định.** Review này không duyệt ánh xạ, không cho phép chọn checkpoint/tuning theo test, không mở giai đoạn so sánh tracker/bộ lọc hay tối ưu Nano. Bước tiếp theo là chốt phạm vi ánh xạ UA-DETRAC trước khi khóa giao thức và chạy đánh giá đó.

Kèm `CODEX_REVIEW_3e94b21_EVIDENCE.zip`: script/harness reviewer, log, kết quả JSON, source fixture cần tái lập, MANIFEST/hash và README. Không có dữ liệu thật, weights hoặc GPU workload.
