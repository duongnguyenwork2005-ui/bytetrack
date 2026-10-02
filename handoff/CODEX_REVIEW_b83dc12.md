# Review vòng 38 — `b83dc12`

**Kết luận: PASS WITH LIMITATIONS** cho việc xuất dataset và bàn giao bằng chứng trong phạm vi kiểm được. Không phát hiện lỗi chặn; không yêu cầu xuất lại dataset hay chạy thêm pilot. Đây **không phải phê duyệt training dài**, bổ sung resume hoặc thay đổi kế hoạch.

- Base: `0c0a07a3995ace893610c3d68603a6e3cf949f61`.
- Target: `b83dc1295cad821b55c60ae15855331b870fa98f`.
- ZIP: **816.591 byte**, SHA256 `f4efa4782d62bf766d86d3be1e1fb0d2aeade0a1eee5d323ab5472d20c22be7c`.
- Chỉ review bản đã tải lên; không truy cập ổ D: trên PC, không SSH Nano, không chạy training/inference và không sửa mã dự án.

## 1. Những gì reviewer tự kiểm

| Kiểm | Kết quả |
|:--|:--|
| ZIP, kích thước, SHA256, đường dẫn an toàn, testzip | Khớp; 121 thành viên |
| Payload, số byte và hash | **115/115** khớp; 3 file trùng nội dung khôi phục đúng bằng alias |
| Áp ngược rồi áp xuôi patch | **118/118 đường dẫn** khớp target từng byte |
| Sáu file base bị sửa | Hash preimage khớp manifest; **6/6** còn đối chiếu được với file từ các vòng review trước |
| Báo cáo/bằng chứng `0c0a07a` được lưu lại | **91/91** file khớp `SHA256SUMS.txt`; báo cáo khớp bản reviewer đã phát hành |
| Phạm vi patch | Chỉ `handoff/` và `CODEX_HANDOFF.md`; mã trong thư mục bằng chứng là bản sao lịch sử, không phải sửa `src/` đang chạy |
| Đầu vào không bàn giao | Giữ nguyên **13** khai báo NPZ |
| Kiểm số liệu/hash/log bằng script reviewer | **188/188 điều kiện đạt**, không phải 188 unit test của dự án |

Không có ảnh, nhãn dataset, checkpoint, engine, video hoặc file riêng thuộc diện loại trừ trong các payload/đường dẫn patch đã kiểm. Reviewer **không có toàn bộ cây Git** để xác nhận riêng tuyên bố 4.084 file của tác giả. Việc 118 đường dẫn roundtrip khớp không thay thế phép kiểm đó.

## 2. Bằng chứng xuất dataset nhất quán

Reviewer đã đối chiếu log xuất, log kiểm hợp đồng riêng, JSON của công cụ kiểm bổ sung và hai dry-run. Bốn bản ghi hợp đồng cùng nhận diện dataset đầy đủ v2, không phải tập smoke, cùng số lượng và bốn hash:

| Tệp | SHA256 được các log ghi nhận |
|:--|:--|
| `data.yaml` | `1280e57bcc5c6927a3a1143f59b3ffaf2a05fc9d16b63507ab4f81be2eaeecd1` |
| `tom_tat.json` | `9eae5eb3ee0a70a13dee0dca7afd682044b480d55690257f2ac4cbe574749010` |
| `manifest/anh.csv` | `1ee6174f8be328f4f62ea10e3a9bb0502fd18c4e55f9b9aa0d2512cd565f6069` |
| `video.json` | `05140872cc35b4399700960139ce287810640985e2ee20f9700e848826a09750` |

File split từ vòng đã review trước được băm lại: `b8a2848e3da053a2b1b79e7fa6cfad812540315af6e48b12b4174a40efc06cdb`. Tính lại hash danh sách video cho đúng `a59a8d0672f30d6e9ff90d4347965fa003b299bb79995c0883796712446ca723`; 1.261/139 video là hai tập không giao nhau. Hash cấu hình và nguồn nhãn trong log trước xuất cũng khớp bản khóa cũ.

Các đẳng thức/tổng thống kê tính lại được:

- 50.487 + 5.559 = **56.046 ảnh**.
- 56.046 + 221.729 + 304 = **278.079 frame nguồn**.
- Tổng box năm lớp hai split = **475.234**.
- File nội dung được hợp đồng ghi nhận đã băm: 56.046 ảnh + 56.046 nhãn + 18.600 ignore = **130.692**; ít hơn tổng 130.699 file toàn thư mục 7 file; gói không có inventory đầy đủ để reviewer tự liệt kê 7 file còn lại. Không coi 130.699 là số ảnh.
- Hai mốc 23:35:25–23:48:22 tương ứng **777 giây**, khớp log thời gian xuất.

| Thống kê từ log, khớp giữa exporter và công cụ kiểm bổ sung | Train-fit | Dev |
|:--|--:|--:|
| Video / ảnh | 1.261 / 50.487 | 139 / 5.559 |
| Ảnh âm tính | 686 | 95 |
| Ảnh có ignore | 17.040 | 1.560 |
| Box motorcycle / ảnh / video có lớp | 2.235 / 1.822 / 230 | 226 / 205 / 24 |
| Box bicycle / ảnh / video có lớp | 4.747 / 3.598 / 397 | 407 / 334 / 44 |

Dev đầy đủ có đủ năm lớp theo các bản ghi; giới hạn không có motorcycle/bicycle của dev smoke không còn áp dụng cho tập đầy đủ. Số video có lớp sau lấy mẫu nhỏ hơn số video có lớp trong toàn nguồn không đồng nghĩa đổi split. Không dùng các số này để chọn lại dev.

## 3. Hai ghi chú không chặn

### R38-N1 — 789 bước tối ưu/epoch là ước tính tĩnh

`src/train_detector.py::suy_ra()` hiện tính `ceil(ceil(n_train / batch) / accumulate)` với `accumulate = round(nbs / batch)`. Với 50.487 ảnh, batch 8 và nbs 64, kết quả đúng của **công thức đó** là 6.311 batch và 789 bước tối ưu. Công thức không mô tả thay đổi accumulate trong warmup và ranh giới cập nhật giữa các epoch.

Reviewer chạy riêng đúng hàm này, không import Torch/train: trên cấu hình pilot đã bàn giao, hàm dự báo **6 bước/epoch**, trong khi log cập nhật thật vòng 37 ghi **12 và 5 bước** cho hai epoch. Vì vậy không được gọi 789 là số bước thực chạy mỗi epoch của lượt dài.

**Đề nghị đính chính khi cập nhật tài liệu kế tiếp:** “6.311 batch/epoch; khoảng 789 bước tối ưu theo accumulate cố định 8, chưa tính warmup và ranh giới cập nhật”. Chưa có số đo thật trên dataset đầy đủ. 94.665 là số batch dự kiến **nếu chạy đủ 15 epoch**; cấu hình đang có `patience=5` nên không phải cam kết chạy đủ. Không cần sửa lịch học hoặc chạy thêm pilot để đính chính câu chữ này.

### R38-N2 — phạm vi “kiểm độc lập” danh sách frame

`05_kiem_doc_lap.py` mở manifest/nhãn đầu ra rồi đối chiếu lại, nhưng tập frame kỳ vọng được tính bằng **chính `bdd_yolo_export.chon_frame()`**. Đây là kiểm nhất quán đầu ra với quy tắc exporter và có bước đọc/đếm nhãn riêng; không phải triển khai độc lập thứ hai để phát hiện mọi lỗi của hàm chọn frame.

Script và phụ lục đã nêu cách gọi đó. Khi tóm tắt, nên giữ rõ phạm vi, tránh mở rộng thành “đã xác minh độc lập toàn bộ tính đúng đắn lấy mẫu”. Không thấy bằng chứng chọn sai frame trong vòng này; không yêu cầu đổi quy tắc hoặc xuất lại.

## 4. Tài nguyên và chuẩn bị lượt dài

- Tính lại từ **781 mẫu CSV**: RAM toàn máy còn trống thấp nhất **449 MiB**, trung vị **727 MiB**, cao nhất **1.274 MiB**, khớp JSON. Không sử dụng RSS tiến trình xuất vì tác giả đã xác nhận monitor bám nhầm shell. Reviewer không có snapshot tiến trình riêng để xác minh lại PID exporter.
- Hai log dry-run đều exit 0, cùng dataset đầy đủ, cùng cấu hình huấn luyện ngoài model/tên lượt, và ghi rõ không train/không tạo thư mục lượt. Nhánh dry-run trong mã runner đã review trở về trước nhánh huấn luyện.
- Số ảnh trùng đầu vào dự toán đã được review trước; **56,64 h / 73,95 h** vẫn là hai kịch bản theo giả định. Review này không tạo số đo tốc độ mới hay bảo đảm thời gian/bộ nhớ.
- Không coi nghiệm thu dataset là phê duyệt 15 epoch, ngân sách, thời điểm bắt đầu, bổ sung resume hay sửa monitor. Các quyết định đó thuộc người dùng. Không tự đóng ứng dụng, SSH Nano hoặc mở thí nghiệm khác.

## 5. Giới hạn nghiệm thu

1. Gói không có 8,03 GB dataset, toàn bộ `anh.csv`, nhãn YOLO hay ảnh nguồn. Reviewer **không tự băm 56.046 ảnh, mở 475.234 box hoặc so từng frame với native JSON**. Các số đó là kết quả kiểm do tác giả chạy trên PC, được đối chiếu về tính nhất quán trong gói này.
2. Việc không trùng nội dung train/dev/BDD val là kết quả tác giả báo; reviewer không có các ảnh để tự lặp lại. UA-DETRAC test chỉ được kiểm **tên video**, không được mô tả thành kiểm nội dung.
3. Nhãn train vẫn là snapshot bên thứ ba đã ghim và được chấp thuận có điều kiện; chưa xác thực bằng archive gốc. Khác biệt 183 lớp/50 ngoại lệ ignore giữa native val và COCO val lịch sử chưa được giải quyết bởi vòng xuất này.
4. Không chạy lại exporter, full suite, training hoặc inference. Không có thay đổi runtime tracking/Nano trong delta để yêu cầu chạy lại các thí nghiệm đó.

Có thể ghi nhận review target `b83dc12` là **PASS WITH LIMITATIONS** và lưu trữ đúng ZIP đã kiểm. Hai ghi chú R38-N1/N2 có thể gộp vào lần cập nhật tài liệu tiếp theo; không cần tạo một vòng sửa riêng hay xuất lại dataset chỉ để đóng chúng.

## 6. Bằng chứng đi kèm

`CODEX_REVIEW_b83dc12_EVIDENCE.zip` chứa báo cáo, script reviewer, kết quả kiểm, log/tài liệu liên quan, split/dự toán đã review trước và `SHA256SUMS.txt`.

Lệnh tái lập tại gốc gói bằng chứng (Python chuẩn, Git; không cần GPU):

```text
python check_package.py --zip <duong-dan-ZIP-vong38> --out evidence_lap_lai/package.json
python check_round38.py --tree tree --prior prior --out evidence_lap_lai/receipts.json
```

Lệnh đầu cần đúng ZIP 816.591 byte đã ghi ở đầu báo cáo; nó chỉ dựng cây tối thiểu chứa các đường dẫn thay đổi, không checkout toàn repo. Lệnh thứ hai chỉ kiểm các log/receipt có trong gói, không thay cho kiểm dataset trên PC.
