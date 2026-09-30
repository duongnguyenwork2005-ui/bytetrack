# Nghiệm thu vòng 29 — `bcb92f8`

**Kết luận: PASS WITH LIMITATIONS. Đóng R28-F1. Không phát hiện lỗi chặn mới trong phạm vi review.**

Ngày: 2026-09-30. Review giới hạn ở chốt cache ảnh `.npy`, regression liên quan và tính toàn vẹn delta. Không sửa mã sản phẩm, không training dài, không SSH hoặc thay đổi Nano.

Base của gói: `239b9c6b01c74fba3fa5077ffdda7a0f73c77834`.
Target đã kiểm: `bcb92f86824812ceea3dee56dabe88ef802b1a25`.

## 1. Script Codex gốc đã được chạy độc lập

Reviewer đã lấy lại **nguyên file** `REPRO_REVIEW_07615ba.py` đã phát hành và chạy trên cây mã target tái dựng từ delta:

```bash
python REPRO_REVIEW_07615ba.py <cay-ma-bcb92f8> --out repro_codex_original.json
```

Kết quả: **exit 0, `bug_count=0`**.

- `.npy` sai cạnh ảnh train và dev đều bị chốt dataset từ chối; thông báo có split, đường dẫn và lý do.
- Dataset sạch vẫn được chấp nhận; sau khi bỏ file cache của fixture, bộ nạp thật đọc đúng pixel JPEG.
- Các đối chứng entry point trong script vẫn đúng: dataset hợp lệ tới ranh giới training; YAML train trỏ dev và thiếu nhãn có box/nhãn rỗng bị chặn trước model/train. Ranh giới model của các đối chứng này dùng spy, được ghi rõ trong script.
- Script không sửa dữ liệu nghiên cứu hoặc nạp checkpoint thật; fixture nằm trong thư mục tạm.

**Lưu ý đọc JSON:** trường `target: "07615ba"` được ghi cố định trong script review cũ. Reviewer giữ nguyên script nên không sửa trường này. Tham số lệnh thực tế trỏ tới `review_bcb92f8/tree`; `execution_manifest.json` ghi target đúng và hash script/source để đối chiếu.

Việc tác giả ghi `NOT_RUN` ở máy của mình vẫn là lịch sử đúng. Bằng chứng chạy độc lập này đóng phần còn thiếu; không cần đổi log NOT_RUN cũ thành PASS hồi tố.

## 2. Mã sửa và các kiểm tra bổ sung

Đã đọc thay đổi tại `bdd_yolo_export.py`, `train_detector.py` và `test_detector_5lop.py`:

- Hợp đồng dataset kiểm đúng đường dẫn `image.with_suffix('.npy')` mà Ultralytics dùng thay ảnh nguồn, cho cả train/dev.
- Không tự xóa file; `.npy` không liên quan tới ảnh của dataset không bị cấm.
- Runner chặn cấu hình cache ảnh không được hỗ trợ; callback kiểm lại `ds.cache` và `ds.npy_files` trước vòng train.
- Cách ghi SKIP đã được sửa; không còn cộng dòng bỏ qua weights vào PASS.

**Suite chạy độc lập:** `test_detector_5lop.py` trả exit 0, **76/76 phép thực thi PASS**, 6 mục/nhóm SKIP do gói không mang checkpoint. Các mục đó là hai A7, E6, F, G và J3/J4/J6. Không diễn giải lượt này thành 92/92.

Trong các phép đã chạy:

- J0 dùng `YOLODataset` thật, đối chiếu pixel sạch **125/125 ảnh** train/dev.
- J1/J2 chứng minh cache `.npy` sai bị chốt file/runner từ chối.
- J5a/J5b dùng dataset thật: `.npy` xuất hiện sau bước kiểm đầu bị callback chặn.
- Regression cache nhãn `.cache`, hash ảnh/nhãn/ignore và evaluator COCO vẫn PASS.

**Kiểm độc lập bổ sung qua subprocess/CLI thật, không spy: 7/7 PASS.** Không cần checkpoint vì chỉ kiểm các nhánh phải dừng trước model:

| Ca | Kết quả |
|:--|:--|
| Dataset sạch | Chốt dataset qua; dry-run tới đúng bước báo thiếu checkpoint được cố ý để trống |
| `.npy` không liên quan | Dataset vẫn được chấp nhận |
| `.npy` cạnh ảnh train | Exit 2 vì cache ảnh; không tạo output, file `.npy` được giữ nguyên |
| `.npy` cạnh ảnh dev | Exit 2 vì cache ảnh; không tạo output, file `.npy` được giữ nguyên |
| `cache: true` | Exit 2 tại chốt cấu hình, không output |
| `cache: ram` | Exit 2 tại chốt cấu hình, không output |
| `cache: disk` | Exit 2 tại chốt cấu hình, không output |

Ở hai ca `.npy` cạnh ảnh, entry point chuẩn bị đánh giá cũng bị hợp đồng dùng chung chặn. Thông báo lỗi được đối chiếu để không nhầm lỗi thiếu checkpoint thành bằng chứng chốt cache hoạt động.

## 3. Gói và provenance

- ZIP nhận được: **717.372 byte**.
- SHA256: `5a9319763373e56b6073fc94b9aa572a962606a7bab3bf2d333e73f35a9128ba`.
- `testzip` sạch; 128 payload + 21 mục dedup khớp hash/số byte.
- 9 preimage base đúng hash khai báo; `git apply --check` và áp patch đều exit 0.
- **149/149 đường dẫn thay đổi** khớp target sau tái dựng. Các kiểm đường dẫn cấm trong verifier đều qua.
- Một tài liệu base được phục hồi bằng reverse patch, một source base chuẩn hóa CRLF thành LF; chỉ dùng sau khi khớp đúng hash preimage khai báo. Không chỉnh source target để làm test qua.
- Không khẳng định đã đối chiếu toàn bộ 3.387 file với Git trên PC tác giả. Phạm vi xác minh độc lập là delta và các preimage liên quan; những NPZ lớn đã loại không cần cho review này.

## 4. Giới hạn và bằng chứng của tác giả

Reviewer chạy Linux CPU, Python 3.12.14, Ultralytics 8.4.141, torch 2.6.0+cpu, NumPy 2.3.5, OpenCV 4.11.0, pycocotools 2.0.11. Không có checkpoint/GPU/dataset BDD MOT train thật trong gói.

Đã đối chiếu log và JSON bàn giao: tác giả ghi nhận `92/92`, hai ca runner thật chặn `.npy` và smoke YOLOv8n 1 epoch trên fixture GPU. JSON smoke ghi loss hữu hạn, không lỗi, **719/90 ảnh**, `cache_anh: null`, `npy_canh_anh: 0`. Đây là **bằng chứng bàn giao được đối chiếu**, không phải reviewer tự chạy GPU.

Chốt vẫn dựa vào dữ liệu được giữ nguyên trong khi chạy; không băm lại từng pixel mỗi batch. Không coi cơ chế này là bảo vệ khỏi tiến trình khác cố tình thay dữ liệu sau kiểm. Không thấy lý do mở thêm thay đổi ngoài R28-F1 hoặc chạy lại thí nghiệm tracking/Nano đã nghiệm thu.

Các log tạm ở lượt kiểm trước bị mất khi workspace được làm mới. Reviewer đã khôi phục lại đúng gói và script đã lưu, rồi chạy lại các kiểm trên; log trong bộ bằng chứng là output thật của lượt hoàn tất này.

## 5. Bước tiếp theo trong kế hoạch đã chốt

- Có thể ghi nhận nghiệm thu R28-F1 và dùng **`bcb92f8` làm base review tiếp theo**, sau khi lưu trữ gói đã được review cùng hash.
- Bước tiếp vẫn là hoàn tất dữ liệu **BDD100K MOT train**, kiểm nguồn, khóa split theo video và smoke trên dữ liệu thật theo kế hoạch hiện có.
- Kết luận này không duyệt training dài, số epoch/ngân sách, chuyển cloud, thêm dataset/model hoặc thay đổi Nano v16. Các quyết định đó giữ nguyên trạng thái chờ người dùng.

File bằng chứng kèm: `CODEX_REVIEW_bcb92f8_EVIDENCE.zip`, gồm script gốc không đổi, kết quả/log, script CLI bổ sung, manifest thực thi và kết quả kiểm gói. Không chứa dataset, weights hoặc video.
