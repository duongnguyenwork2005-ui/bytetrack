# Review vòng 43 — `a03f6e1`

**Kết luận: CHƯA NGHIỆM THU ĐẦY ĐỦ — cần bổ sung mã phụ thuộc để hoàn tất review.**

Phần gói delta và số học của các bảng kết quả đã kiểm đạt. Có một lỗi metadata ở tham số CLI không mặc định; không kích hoạt trong ba lượt BDD đã bàn giao. Chưa có bằng chứng từ lần review này buộc chạy lại inference hoặc training.

## 1. Đối tượng và phạm vi

- Gói: `review_delta_latest.zip`, **319.428 byte**.
- SHA256: `f93a35e659ff59f4b287965c2fdb1b062a28451b96d63d5e1a3203ed3a10c9c0`.
- Base: `4d54744e4163d6f2c3ef3e9e0f67299bba00d349`.
- Target: `a03f6e1a0e2612a8879f4ad59c21f18cecdcca4e`.
- Đã đọc entry point BDD mới, test mới, kế hoạch, script kiểm checkpoint/tổng hợp, brief, phụ lục và kết quả kèm theo.
- Không sửa repo dự án, train, chạy inference GPU, truy cập Nano hoặc duyệt ánh xạ UA-DETRAC.

## 2. R43-B1 — Thiếu dependency để reviewer chạy test và kiểm đường chấm thật

**Đây là giới hạn bàn giao/môi trường review, chưa phải kết luận lỗi thuật toán.** Delta có hai file `src/` mới, nhưng không có mã các hàm mà chúng gọi. Các mô-đun này cũng không nằm trong snapshot vòng 42 hiện còn ở môi trường review. Bằng chứng cũ đã tìm được không chứa chúng.

Chạy nguyên `src/test_danh_gia_bdd_val.py` dừng tại:

```text
ModuleNotFoundError: No module named 'bdd_yolo_export'
```

Vì vậy **19/19 và 92/92 hiện chỉ được đối chiếu từ log tác giả**, không phải kết quả tôi chạy độc lập. Không thể dùng số 180 phép đối chiếu số học bên dưới thay cho kiểm thử evaluator.

Đề nghị gửi support code **đúng target và nguyên byte**, gồm ít nhất:

| File | SHA256 được ghi trong hồ sơ chạy |
|---|---|
| `src/eval_detector_5lop.py` | `30a1fd5a63641f1cebc22e4e7ef7e4600a48a45ed6f7d533bf319af0fa156f7f` |
| `src/bdd_yolo_export.py` | `1ca964b0fc81701ac746335fa20ac6e5231322a4cdd11ad47753a2b25fc31493` |
| `src/detector_classes.py` | `7bba3439987377cb2f161eb31e2cf0afafae3e814f165279087b7a3780699c3e` |
| `src/mot_datasets/bdd100k_coco.py` | `5028ffb98ddff6ba423bef14ff9331fd02f8e40b1055438c88224182914f8363` |
| `src/config.py` | Ghi hash từ target khi đóng gói |

Kèm các mô-đun nội bộ được chúng import, `__init__.py` cần thiết và khai báo dependency Python để test fixture chạy được trong thư mục giải nén sạch. Có thể dùng chức năng `--support` của packager; giữ riêng danh sách support với danh sách file thay đổi. Nếu gửi thêm `test_detector_5lop.py`, gửi cả dependency fixture của nó.

**Không cần gửi dataset, checkpoint hay NPZ lớn để mở khóa bước kiểm mã/fixture này.** Không cần đổi base review hoặc chạy lại thí nghiệm. Việc thiếu support không phủ nhận kết quả test tác giả đã chạy trên repo đầy đủ.

## 3. R43-N1 — Metadata không phản ánh `--conf-van-hanh` hiệu lực

**Mức độ: P2, lỗi provenance của nhánh CLI không mặc định; không làm sai ba kết quả BDD hiện có.**

Trong `src/danh_gia_detector_bdd_val.py`:

- Dòng 173 nhận `--conf-van-hanh`.
- Dòng 204 ghi nguyên hằng `QUY_TAC` vào `ket_qua.json`.
- Dòng 216 lại truyền giá trị CLI cho `danh_gia()`.

Tái hiện bằng entry point nguyên byte, với dependency doubles được khai báo rõ:

| CLI | Ngưỡng truyền cho scorer | `quy_tac.van_hanh.conf` | Exit |
|---|---:|---:|---:|
| `--conf-van-hanh 0.25` | 0,25 | 0,25 | 0 |
| `--conf-van-hanh 0.5` | 0,5 | **0,25** | 0 |

Phép kiểm này chỉ chứng minh đường truyền tham số và ghi metadata của file mới; **không dùng scorer giả để tuyên bố AP đúng**. File `REPRO_REVIEW_a03f6e1.py` ghi đầy đủ phạm vi, kết quả `bug_count=1`.

Sửa đề nghị: tạo bản cấu hình hiệu lực riêng cho lượt chạy, ghi ngưỡng thực vào đó, dùng cùng giá trị cho scorer/chẩn đoán/metadata. Không sửa hằng dùng chung theo cách làm rò cấu hình giữa các lần gọi. Thêm một test qua CLI với ngưỡng không mặc định và đối chiếu metadata với tham số scorer.

Ba JSON bàn giao đều có ngưỡng hiệu lực và quy tắc chung **0,25**, nên lỗi này **không yêu cầu chạy lại ba lượt**. Không chỉnh ngưỡng đánh giá hoặc sửa số liệu lịch sử.

## 4. Phần đã kiểm độc lập

### Gói và patch

- ZIP CRC đạt, 66 thành viên, không trùng tên/đường dẫn thoát thư mục/symlink.
- **60 payload + 1 bản trùng nội dung** khớp byte và SHA256 của MANIFEST.
- Reverse rồi apply lại patch: **61/61 đường dẫn thay đổi** khớp target; 6 preimage khớp hash base.
- Cả **6/6 preimage** còn có bản đối chiếu từ snapshot vòng 42 trước đó và đều trùng từng byte.
- Không thấy đường dẫn riêng hoặc file dữ liệu/model bị cấm trong phần kiểm gói.

Đây là kiểm phạm vi đường dẫn thay đổi. **Tôi không có checkout Git đầy đủ để xác nhận độc lập tuyên bố “4.642 file toàn cây”.**

### Số học và đối chiếu báo cáo

`audit_results.py`: **180/180 phép đối chiếu đạt**, gồm P/R/F1 tính từ TP/FP/FN, trung bình AP/P/R theo lớp, chênh AP theo lớp, bảng CSV với JSON, số ảnh, số đếm nhãn, fingerprint được báo cáo, exit code và mã entry point với hash trước chạy.

| Model | mAP50 | mAP50–95 | Δ mAP50–95 so baseline |
|---|---:|---:|---:|
| YOLOv8n COCO | 0,2439417100 | 0,1570728458 | — |
| YOLOv8n fine-tune | 0,3835045640 | 0,2506992858 | +9,362644 điểm |
| YOLO11n fine-tune | 0,3757858772 | 0,2467297056 | +8,965686 điểm |

Các giá trị này **nhất quán nội bộ**. AP theo lớp vẫn là đầu vào lấy từ kết quả tác giả, không được reviewer tính lại từ dự đoán thật. Tổng GT 5 lớp là 382.903; cộng 20.848 vùng bỏ qua và 61.761 annotation lớp khác bằng 465.512, đúng hồ sơ.

Tôi xác nhận ba log kết quả ghi 39.973 ảnh, exit 0, cùng GT/quy tắc và checkpoint hash đã khóa. Đây là đối chiếu hồ sơ, không thay cho đọc 39.973 ảnh hay băm trực tiếp checkpoint.

## 5. Giới hạn và bước tiếp theo

- Không có ảnh, GT JSON thật, weights hoặc NPZ dự đoán trong gói. Chưa tự xác minh pixel/nhãn/tensor hay tính lại AP thực tế.
- Script tổng hợp của tác giả gọi lại chính `danh_gia()` khi đọc NPZ: chứng minh tính tái lập của đầu ra theo evaluator đó, không phải đối chứng AP bằng một cài đặt độc lập.
- Nhánh GT mới bỏ thông tin lớp khỏi mảng `ign`; cần giữ cách diễn đạt rõ đây là quy tắc vùng bỏ qua của dự án, không mặc nhiên đồng nhất toàn bộ giao thức với benchmark COCO/BDD chính thức. Chưa kết luận thêm về hàm ghép khi mã phụ thuộc chưa có.
- Mỗi model một lượt training; chưa có CI/kiểm định. Chênh 0,397 điểm mAP50–95 giữa hai model fine-tune chưa phải căn cứ chọn model tổng quát.
- BDD val đã được xem từ trước; khác biệt nhãn native/COCO và khác biệt ảnh dev tô xám đã được tác giả công bố, vẫn giữ nguyên.
- UA-DETRAC D1–D5 **chưa được người dùng chốt**. Review này không phê duyệt một ánh xạ nào.

**Việc tiếp theo:** bổ sung support code để hoàn tất kiểm fixture và evaluator; sửa/test metadata R43-N1 trong phạm vi nhỏ. Giữ nguyên ba kết quả BDD và base `4d54744` cho đến khi có kết luận review đầy đủ. Chưa mở training mới, so sánh tracker/bộ lọc hay tối ưu Nano.

## 6. Bằng chứng kèm theo

`CODEX_REVIEW_a03f6e1_EVIDENCE.zip` chứa script kiểm gói, script đối chiếu số học, script tái hiện metadata, kết quả JSON, log lần thử chạy test bị chặn, so preimage và các đầu vào nhỏ cần tái lập. Không chứa dataset/weights.

Ghi nhận lỗi của reviewer: lần đầu script đối chiếu giả định `ctrl_c` là boolean; dữ liệu thật dùng danh sách sự kiện `[]`. Ba phép kiểm sai do giả định này đã được sửa theo schema, không thay đổi dữ liệu nguồn. Kết quả lần đầu được giữ kèm ghi chú.
