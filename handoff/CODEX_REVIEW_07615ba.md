# Review vòng 28 — `07615ba`

**Kết luận: REQUEST CHANGES — một finding còn cần sửa trước khi nghiệm thu chốt đầu vào training.** Sáu ca tái hiện vòng 27 đều đã hết lỗi. Finding mới nằm trong cùng phạm vi bảo đảm dữ liệu thực nạp đúng manifest; không yêu cầu thêm model, dataset, thí nghiệm hay chức năng Nano.

Ngày review: 2026-09-30. Base của delta: `239b9c6b01c74fba3fa5077ffdda7a0f73c77834`. Target: `07615baa32c1ff4d177a0c78aaad96ffd8324a1e`.

## R28-F1 — P2: ảnh cache `.npy` cũ lọt qua cả chốt manifest và callback trước training

**Điều kiện xảy ra:** bên cạnh một ảnh đã xuất có file cùng tên, đuôi `.npy`, chứa pixel khác JPEG. Cấu hình hiện tại `cache: false` không ngăn Ultralytics 8.4.141 đọc file này.

Vị trí cần xem:

- `src/train_detector.py:153`, hàm `kiem_dau_vao_da_nap`: kiểm `ds.labels` gồm đường dẫn, lớp và bbox; chưa xác minh nguồn pixel mà dataset sẽ đọc.
- `src/bdd_yolo_export.py:795`, hàm `liet_ke_anh_nhu_ultralytics`: liệt kê các định dạng ảnh, nên `.npy` không thuộc tập file được kiểm.
- Ultralytics 8.4.141, `data/base.py:249`: `load_image()` dùng `np.load(fn)` khi file `.npy` tồn tại, kể cả khi `self.cache is None` do truyền `cache=False`.

**Tái hiện độc lập bằng bộ nạp thật**, không chỉ suy luận từ mã:

1. Tạo và xuất fixture bằng chính exporter của dự án: 77 ảnh train, 48 ảnh dev, manifest v2 hợp lệ.
2. Giữ nguyên JPEG, nhãn, ignore, YAML và toàn bộ hash; thêm một `.npy` cùng kích thước, 3 kênh, chứa ảnh đen.
3. Gọi `train_detector.kiem_dataset()`; chốt vẫn chấp nhận.
4. Dựng `YOLODataset(..., cache=False, augment=False)` thật cho hai split; gọi callback thật `kiem_dau_vao_da_nap()`. Callback vẫn báo không thiếu/thừa ảnh và không lệch nhãn.
5. `YOLODataset.get_image_and_label()` thực sự trả pixel đen từ `.npy`.

| Ca | Trung bình pixel JPEG | Pixel được nạp | Hash JPEG đúng manifest | Chốt file / callback |
|:--|--:|--:|:--:|:--:|
| Ảnh train | 90,403125 | 0 | Có | Đều cho qua |
| Ảnh dev | 89,689745 | 0 | Có | Đều cho qua |

Ở cả hai ca, bỏ `.npy` rồi dựng lại dataset thì pixel nạp khớp hoàn toàn với JPEG. Không thay JPEG hay nhãn trong phép kiểm này. `SimpleNamespace` chỉ làm đối tượng chứa hai dataset cho callback; việc xuất, kiểm, đọc nhãn và đọc pixel đều dùng mã thật. Không chạy optimizer/training hoặc nạp checkpoint trong các ca này.

**Ảnh hưởng:** khi có cache ảnh cũ, model có thể học hoặc được validation trên pixel khác dữ liệu đã khóa, trong khi metadata vẫn ghi dataset đã xác minh. Chốt cache nhãn `.cache` mới sửa hoạt động đúng, nhưng chưa bao phủ cache ảnh `.npy`.

**Giới hạn kết luận:** chưa có bằng chứng các smoke GPU đã bàn giao gặp tình huống này. Dataset xuất mới trong thư mục sạch và chưa có `.npy` không kích hoạt lỗi. Không yêu cầu chạy lại kết quả UA-DETRAC/BDD hay Nano đã nghiệm thu.

**Sửa tối thiểu đề nghị:** với chính sách `cache: false` hiện tại, từ chối rõ ràng các file `.npy` mà bộ nạp có thể dùng thay ảnh đã khóa, trước khi nạp model hoặc tạo lượt train. Hoặc kiểm chứng/cache theo một cơ chế bảo đảm pixel đúng ảnh nguồn. Chỉ đặt lại `cache=False` là chưa đủ. Không tự xóa ảnh nguồn hoặc sửa manifest để hợp thức hóa cache cũ.

Regression cần giữ trong phạm vi finding:

- `.npy` sai ở train và dev: bị chặn trước training, hoặc bộ nạp thực sự dùng đúng pixel đã xác minh.
- Dataset sạch: tiếp tục qua kiểm và nạp đúng ảnh.
- Các ca cache nhãn cũ, nhãn rỗng hợp lệ, nhãn thiếu và YAML trỏ train vào dev vẫn giữ hành vi đã sửa.
- Kiểm qua bộ nạp thật; test chỉ đếm ảnh/nhãn không đủ.

## Những điểm đã xác minh đạt

**R27-F1:** script vòng 27 không còn tái hiện lỗi train trỏ dev và thiếu nhãn. Kiểm bổ sung qua `train_detector.main()` với spy ở ranh giới model: dữ liệu hợp lệ tới lệnh train; ba ca sai (train trỏ dev, thiếu nhãn có box, thiếu nhãn rỗng) đều exit 2, không gọi model/train và không tạo kết quả. Checkpoint trong phép này là file giả có hash tương ứng, model được giả lập rõ ràng; không coi đây là smoke training.

**R27-F2:** các ca đổi JPEG, tọa độ nhãn và xóa ignore đều bị chặn. Test với `YOLODataset` thật xác nhận cache nhãn cũ bị phát hiện. Việc bảo đảm pixel còn hở như R28-F1.

**R27-F3:** ca hòa IoU khớp pycocotools: AP50 = 1,0; AP50–95 = 0,5514851485, sai khác số học khoảng `1e-16`. Ba test mới hòa IoU, IoU tại ngưỡng và GT chồng ignore đều qua. Giới hạn 1 ulp tại ngưỡng 0,9 đã được tác giả công bố; không mở thành finding mới trong vòng này.

## Phạm vi chạy độc lập và đối chiếu log

Môi trường reviewer: Linux, Python 3.12.14, Ultralytics 8.4.141, torch 2.6.0+cpu, torchvision 0.21.0+cpu, NumPy 2.3.5, OpenCV 4.11.0, pycocotools 2.0.11.

| Kiểm | Kết quả |
|:--|:--|
| Script Codex vòng 27 nguyên bản chạy trên target mới | Exit 0; cả 6 cờ lỗi = false |
| `test_detector_5lop.py` nguyên bản | Exit 0; các ca thực thi đều qua |
| Bốn ca entry point trong script review mới | Đúng kỳ vọng; ranh giới model dùng spy |
| Hai ca cache ảnh `.npy` bằng loader thật | Đều tái hiện lỗi; script review mới exit 1 |

Không diễn giải lượt test ở reviewer là `83/83`: thiếu checkpoint trong gói nên A7/E6 và các ca cần weights không chạy. Script in `71/71` kèm một nhóm SKIP; trong 71 có một dòng `[PASS] F1..F5 bo qua` chỉ là thông báo bỏ qua, nên có **70 phép kiểm thực thi đạt** ở lượt này. Nhóm G được kiểm riêng bằng spy như trên. Đây là ghi rõ phạm vi kiểm, không phải một finding bổ sung.

Đã đọc log và JSON bàn giao xác nhận tác giả ghi nhận `83/83`, hai suite BDD `42/42`, `11/11`, và smoke GPU 1 epoch của cả hai model với 719/90 ảnh, loss hữu hạn, không lệch nhãn nạp. Đây là **đối chiếu bằng chứng bàn giao**, không phải reviewer tự chạy GPU. Không chạy full suite, training dài hoặc SSH tới Nano.

## Tính toàn vẹn gói

- ZIP: 628.509 byte, SHA256 `f150ea8e867435240eb64aebc5286cf730f940963df88aeab789461336bb6396`; `testzip` sạch.
- 109 payload và 14 mục dedup được xác minh; 9 preimage base đúng hash khai báo.
- `git apply --check` và áp patch đều exit 0; 123/123 đường dẫn thay đổi khớp hash target.
- Một tài liệu base không còn trong bản lưu được phục hồi bằng target + reverse patch; một source base được chuẩn hóa CRLF thành LF. Cả hai chỉ dùng sau khi hash khớp chính xác preimage khai báo. Không sửa source target để chạy test.
- Script tái hiện vòng 27 bàn giao khớp nguyên byte bản Codex đã phát hành.

## File và lệnh gửi lại Claude

`REPRO_REVIEW_07615ba.py` tạo fixture tạm và tự dọn; không sửa dữ liệu dự án. Chạy trong môi trường của dự án đã có Ultralytics:

```bash
python handoff/REPRO_REVIEW_07615ba.py . --out handoff/logs/review29/repro_codex.json
```

Trên `07615ba`: exit 1, `bug_count=2`. Sau sửa: cần exit 0 với đối chứng dataset sạch vẫn đạt. Nếu thiếu thư viện hoặc lỗi dựng fixture, đó là lỗi môi trường, không phải bằng chứng đã sửa.

Gói `CODEX_REVIEW_07615ba_EVIDENCE.zip` kèm báo cáo, script, JSON, log chạy độc lập, kết quả kiểm gói và kết quả chạy lại script vòng 27. Giữ base review `239b9c6` trong lúc finding chưa đóng. Chỉ sửa R28-F1 và kiểm phần bị ảnh hưởng; chưa duyệt ngân sách training dài, chưa chuyển cloud, không thay plan hay Nano v16.
