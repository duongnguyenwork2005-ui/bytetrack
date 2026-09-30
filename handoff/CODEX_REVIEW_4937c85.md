**Review vòng 27 — `4937c85`: REQUEST CHANGES**

Ngày review: 29/09/2026. Base: `239b9c6`. Phạm vi: chuẩn bị BDD100K MOT train, evaluator 5 lớp và runner fine-tune; chưa nghiệm thu training trên dữ liệu thật.

Có **3 finding cần sửa trước training dài**. Chúng được tái hiện trên bản mã trong delta, bằng dữ liệu tổng hợp trong thư mục tạm. Không có bằng chứng rằng một lượt huấn luyện thật đã bị ảnh hưởng: theo bàn giao, MOT train chưa có và mới smoke trên fixture. Nghiệm thu Nano v16 và các kết quả tracking cũ không bị mở lại bởi review này.

**R27-F1 — P1: chốt của runner vẫn nhận train trùng dev hoặc thiếu file nhãn**

Vị trí: `src/train_detector.py:98`, `src/train_detector.py:108`, lời gọi tại `src/train_detector.py:349`.

`train_detector.kiem_dataset()` chỉ kiểm `val == images/dev`, hash của file CSV manifest và số mục trong hai thư mục cố định. Nó không kiểm `train`, không đối chiếu danh sách ảnh/nhãn thực tế và không gọi bộ kiểm dataset của exporter. Trong khi đó, YAML nguyên văn được truyền cho Ultralytics.

Tái hiện trên dataset do chính exporter tạo: 77 ảnh train, 48 ảnh dev.

| Biến đổi duy nhất sau khi xuất | Chốt trước training | Bộ kiểm riêng của exporter |
|:--|:--|:--|
| Đổi `train: images/train` thành `train: images/dev` | Chấp nhận; vẫn báo 77 ảnh train | Từ chối đúng |
| Xóa một file nhãn của ảnh train | Chấp nhận | Từ chối đúng |

Ở ca đầu, đường train và val thực tế cùng là `images/dev`. Runner có thể đưa dev vào training và ghi số lượng đầu vào sai. Ở ca thứ hai, chốt trước training không bảo vệ chính quy tắc “thiếu nhãn không được coi là ảnh âm tính”. Không chạy YOLO trong phép tái hiện này; bằng chứng là hàm kiểm đầu vào thật mà `main()` gọi trước khi nạp checkpoint.

Yêu cầu sửa: dùng một bộ kiểm chung trên đúng các đường dẫn đã resolve trong YAML, xác minh train/dev đúng hợp đồng đã khóa, danh sách ảnh/nhãn khớp manifest và không trùng split. Gọi bộ kiểm trước khi nạp model hoặc tạo thư mục lượt. Đầu vào hỏng phải dừng rõ ràng. Regression cần đi qua entry point runner và chứng minh không gọi training; không chỉ gọi bộ kiểm riêng của exporter.

**R27-F2 — P1: danh tính dataset không gắn với nội dung ảnh, tọa độ nhãn và vùng ignore đang được sử dụng**

Vị trí: `src/bdd_yolo_export.py:700`, `src/bdd_yolo_export.py:786`, `src/train_detector.py:105`, `src/eval_detector_5lop.py:302`.

Manifest lưu hash ảnh, nhưng `bdd_yolo_export.kiem_dataset()` không so hash đó với ảnh trên đĩa. Nhãn chỉ được kiểm định dạng, tọa độ hợp lệ và số box mỗi lớp; không có hash nội dung nhãn. File ignore không được xác minh. Runner chỉ băm lại CSV; evaluator chép hash khai báo từ `tom_tat.json`.

| Tái hiện độc lập | Kết quả hiện tại |
|:--|:--|
| Lật ngang một ảnh rồi lưu JPEG cùng kích thước | Hash ảnh thực tế khác manifest; cả bộ kiểm exporter và chốt runner vẫn nhận |
| Đổi tọa độ một box, giữ nguyên lớp và số box | Cả hai bộ kiểm vẫn nhận; `manifest_sha256` không đổi |
| Xóa file ignore có 3 vùng | Bộ kiểm vẫn báo `loi: 0`; evaluator đọc thành 0 vùng |

Như vậy ảnh đầu vào hoặc GT đánh giá có thể thay đổi nhưng vẫn mang cùng danh tính dataset. Việc chỉ nối runner vào bộ kiểm hiện tại ở F1 chưa sửa được F2.

Yêu cầu sửa: khóa nội dung ảnh, file nhãn và thông tin ignore trong manifest; phân biệt rõ file rỗng hợp lệ với file bị thiếu. Xác minh các nội dung thực tế tại bước kiểm và trước train/eval, sau đó ghi danh tính đã kiểm vào kết quả. Metadata cũ chưa đủ bằng chứng phải được báo rõ, không tự bổ sung hash để coi như đã được xác minh từ lúc xuất. Thêm các regression ở bảng trên; chúng phải thất bại trên mã cũ.

**R27-F3 — P2: evaluator chưa tương đương COCO trong ca hai GT có cùng IoU**

Vị trí: `src/eval_detector_5lop.py:151`.

`np.argmax(cand)` chọn GT đầu tiên khi hòa. Bộ ghép của pycocotools chọn GT sau trong thứ tự duyệt khi IoU bằng nhau. Khác biệt này có thể đổi TP/FP của detection kế tiếp, nên không chỉ là khác ID của cặp ghép.

Fixture: cùng lớp car, hai GT `[0,0,10,10]`, `[5,0,15,10]`; detection confidence 0,9 là `[0,0,15,10]`, detection confidence 0,8 là `[0,0,10,10]`. Detection đầu có IoU bằng 2/3 với cả hai GT. Không có crowd/ignore; chỉ có hai detection nên không bị ảnh hưởng bởi giới hạn `maxDets`.

| Chỉ số | Evaluator hiện tại | pycocotools 2.0.11 |
|:--|--:|--:|
| TP tại IoU 0,50 | 1 | 2 |
| AP50 | 0,5049504950 | 1,0000000000 |
| AP50–95 | 0,3534653465 | 0,5514851485 |

Phép kiểm ngẫu nhiên E1 hiện có vẫn PASS với sai lệch 0; nó chưa chứa ca hòa này. Yêu cầu sửa quy tắc ghép để khớp COCO đã chọn và thêm fixture hòa IoU, đồng thời giữ các kiểm GT hợp lệ chồng vùng ignore. Không đổi công thức AP hoặc ngưỡng để che sai khác.

**Kiểm gói và bằng chứng đã chạy**

- ZIP: **413.410 byte**, 77 thành viên, `testzip` không có lỗi.
- SHA256: `9df0135e2a33f855deab715757646c66e8eaf963ab0d78552be6d6272775f361`.
- Base đầy đủ: `239b9c6b01c74fba3fa5077ffdda7a0f73c77834`.
- Target đầy đủ: `4937c859a48dcd1954a203b3c17746eea527a6b7`.
- 71 payload và 2 mục dedup khớp manifest; patch áp thành công và **73/73 đường dẫn** khớp hash target. Các đường dẫn đó vẫn nguyên vẹn sau kiểm thử. Không thấy đường dẫn riêng hoặc dữ liệu lớn bị cấm trong patch.
- Bản base đang giữ ở phía reviewer thiếu `BDD100K_INVENTORY.md`, nên phục hồi preimage từ payload và reverse patch, rồi kiểm đúng hash base. `prepare_mixed_yolo_dataset.py` ở bản giữ có CRLF; chuẩn hóa LF cho ra đúng hash base. Chỉ làm trong cây review mới, không sửa snapshot cũ. Đây là giới hạn tái dựng phía reviewer, không phải lỗi gói.
- Chạy độc lập **37/37** phép hiện có thuộc nhóm B/C/D và E1–E5 của `test_detector_5lop`; E1 đối chiếu pycocotools thật, không bị bỏ qua. Bao gồm xuất Scalabel/COCO-video, các chốt nguồn thiếu/trùng tập đánh giá và AP ngẫu nhiên.
- Script đi kèm tái hiện **6 ca lỗi** thuộc F1–F3 bằng các hàm thật. Exit 1 của script có nghĩa là lỗi vẫn tái hiện, không phải lỗi hạ tầng.
- Đối chiếu log bàn giao: 52/52; hai smoke cuối cùng đều 1 epoch, batch 8, SGD, AMP false, worker 2; loss hữu hạn và nạp lại checkpoint đúng 5 lớp. Đây là bằng chứng do Claude cung cấp, không phải reviewer tự chạy GPU.

Reviewer chạy Python 3.12 trên Linux/CPU, dùng OpenCV 4.11 và pycocotools 2.0.11. Không có torch, Ultralytics, checkpoint hoặc BDD MOT train ở môi trường này; không chạy toàn suite, GPU smoke hoặc training dài. Không truy cập/sửa repo Windows, index, commit, Nano hay dữ liệu nghiên cứu.

**Các điểm đã làm rõ và giới hạn phương pháp**

- **902/915 motorcycle không mâu thuẫn**: inventory ghi 902 box không crowd/ignore và 13 box crowd, tổng 915. Hàm `kiem_ke_bdd100k.thong_ke_cocovid()` đếm đúng theo định nghĩa này. Chỉ đối chiếu mã và inventory đóng gói; không đếm lại file annotation thật tại đây.
- Chia theo video đã được kiểm. Kiểm trùng SHA với BDD val cũng đã được thử. Tuy nhiên exporter chưa so SHA giữa train-fit và dev: fixture sao chép nội dung ảnh sang video khác ở split kia vẫn xuất được. Vì vậy chưa được diễn đạt là “đã chứng minh không trùng nội dung ở mọi split”; chưa có dữ liệu train thật để kết luận nó có trùng hay không. Không yêu cầu thêm dataset hay thay cách chia trong vòng sửa này.
- Quy tắc coi frame rỗng nằm giữa frame có nhãn đầu/cuối là ảnh âm tính đang được khai báo rõ, nhưng thử trên fixture không chứng minh tính đầy đủ nhãn của nguồn thật. Việc kiểm kê nguồn và smoke trên MOT train thật vẫn cần làm khi tải đủ.
- Batch 8 có số đo fixture hỗ trợ việc dùng để smoke tiếp. Câu “batch 16 tràn VRAM” nên phân biệt lỗi OOM với áp lực bộ nhớ/giả thuyết paging: các lượt batch 16 trong bảng vẫn hoàn tất, và chỉ số đã lưu chưa trực tiếp đo paging. Không dùng chênh tốc độ fixture để kết luận chất lượng model hoặc thời lượng training thật.
- Hai tập BDD val và UA-DETRAC test đã được xem ở các vòng trước; tiếp tục giữ đúng mô tả trong `PLAN_DA_CHOT.md`, không gọi là test chưa từng quan sát.

**Bàn giao để tái hiện và đóng review**

Gửi Claude `CODEX_REVIEW_4937c85.md` cùng `REPRO_REVIEW_4937c85.py`; JSON đi kèm chứa kết quả gốc của các probe. Script chỉ tạo fixture và sửa dữ liệu trong thư mục tạm, không train hoặc sửa mã dự án.

```text
python REPRO_REVIEW_4937c85.py D:\LABOTARY\bytetrack --out REPRO_REVIEW_4937c85.json
```

Script cần numpy, Pillow, OpenCV, PyYAML, pandas và pycocotools. Trên `4937c85`, kết quả mong đợi là exit 1 với sáu cờ `bug_reproduced: true`. Khi sửa, chuyển các ca đó thành regression chứng minh đầu vào sai bị chặn và ca AP khớp tham chiếu; giữ ca dataset hợp lệ tiếp tục chạy.

Giữ base review **`239b9c6`** cho đến khi bản sửa được nghiệm thu. Vòng tiếp theo chỉ sửa ba finding, kiểm thử liên quan, cập nhật tài liệu/bằng chứng và dựng delta. Chưa duyệt 15 epoch hoặc ngân sách 27/32 giờ; chưa thêm detection100K, model, thuật toán hay tối ưu Nano. Khi có dữ liệu thật, hoàn tất kiểm nguồn → split khóa → xuất → smoke thực để báo lại chi phí trước quyết định training dài.
