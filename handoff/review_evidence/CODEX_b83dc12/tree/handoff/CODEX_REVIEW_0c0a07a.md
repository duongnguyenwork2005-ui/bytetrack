# Review vòng 36–37 — target `0c0a07a`

**Kết luận: PASS WITH LIMITATIONS cho đính chính vòng 36 và pilot chẩn đoán vòng 37. Không phát hiện lỗi chặn nghiệm thu trong phạm vi đã kiểm. Không cần chạy thêm pilot để đóng vòng này.**

Đóng **R35-F1 và R35-F2 về yêu cầu đính chính báo cáo**. Câu hỏi nghiên cứu về ảnh hưởng riêng của warmup/mosaic vẫn chưa được xác định; câu hỏi đó không phải một finding bắt buộc phải mở thêm ablation để sửa.

Nghiệm thu này **không phải phê duyệt xuất dataset đầy đủ hoặc training dài**. Các quyết định đó thuộc người dùng.

## 1. Gói và phạm vi kiểm độc lập

| Mục | Giá trị |
|:--|:--|
| Base | `2351c677c9a330050ebd152f10f46ed3922a431f` |
| Target | `0c0a07a3995ace893610c3d68603a6e3cf949f61` |
| ZIP | `review_delta_latest.zip`, **967.546 B**, 136 thành viên |
| SHA256 ZIP | `a2985214d9e2e25f2abd1f9578b4517582fc6684ad79418b471d99a5e9e4928d` |
| Payload | **130/130** khớp số byte và SHA256; 2 file bỏ trùng nội dung được khôi phục đúng |
| Patch | Áp ngược/áp xuôi: **132/132 đường dẫn thay đổi** khớp target |
| Base preimage | 9 file sửa khớp hash base khai báo; **7 file** còn có bản từ review trước và trùng từng byte. Hai file mã base được dựng bằng patch ngược, không có checkout Git base độc lập |
| Bằng chứng review trước | `CODEX_2351c67/SHA256SUMS.txt`: **69/69** khớp |
| Dữ liệu bị loại | Không có dataset, weights, video, engine, NPZ hoặc file riêng bị cấm; 13 khai báo NPZ lịch sử được giữ |

Đã đọc `train_detector.py`, `pilot_chan_doan.py`, test mới, cấu hình pilot, script khóa/tóm tắt, phụ lục và log. Đã chạy phép kiểm độc lập trên Linux/Python 3.12.14/NumPy 2.3.5:

- **11/11** kiểm cô lập trên chính hàm của runner: kế thừa cấu hình, hash LF/CRLF, chặn thay đổi ngoài epochs, và lỗi `final_eval` trước/sau sửa. Fixture thay đồng hồ và nhánh `torch.cuda.is_available`; không giả lập một lượt training.
- **468 điều kiện đối chiếu số liệu/receipt đạt, 0 lệch**: tham số, nguồn dữ liệu/checkpoint, batch, warmup/LR/momentum, số bước cập nhật, mốc giờ, thống kê tài nguyên và bảng tóm tắt. Đây là kiểm tính nhất quán của bằng chứng, **không phải 468 unit test huấn luyện**.
- Chạy lại script tóm tắt trên chính JSON/CSV đã giao: kết quả cho cả hai model trùng JSON bàn giao.
- Chạy lại công cụ so khóa: **46/46 mục khớp** cho mỗi model.

**Giới hạn reviewer:** không có PyTorch/Ultralytics, GPU, ảnh smoke hay checkpoint `.pt` trong môi trường này. Không lặp lại training/inference; không tự tính lại AP từ dự đoán/GT; không kiểm lại trực tiếp trọng số trong checkpoint. Không xác minh toàn bộ cây Git hay trực tiếp trạng thái PC/Nano. Những con số **34/34 pilot, 92/92 detector** và A/B CPU bit-identical là đối chiếu log do tác giả bàn giao, không phải reviewer tự chạy các suite đó.

## 2. Vòng 36: đóng hai điểm diễn giải

**R35-F1 — ĐÓNG về báo cáo.** Tài liệu đã phân biệt checkpoint COCO 80 lớp với model 5 lớp sau chuyển cấu trúc/nạp trọng số. Đã rút kết luận không có bằng chứng rằng thiếu warmup gây giảm confidence, và không còn dùng AP car ở conf 1e-6 để chứng minh toàn bộ pipeline đúng. Kết quả lịch sử và ngưỡng đánh giá chính được giữ.

**R35-F2 — ĐÓNG.** Đối chiếu lại JSON: **162 giá trị số và cấu trúc không đổi**, 4 chuỗi mô tả được sửa. Mức 56,64 h / 73,95 h vẫn là hai kịch bản theo giả định, không phải thời hạn tối đa hoặc bảo đảm bộ nhớ. Không yêu cầu chạy lại smoke để sửa câu chữ.

## 3. Pilot: điều gì được bằng chứng hỗ trợ

### Đầu vào và cấu hình

- Cả hai lượt dùng cùng **334 ảnh train / 84 ảnh dev**, hash dataset/manifest/split trùng bằng chứng vòng 35.
- Khởi tạo từ checkpoint pretrained gốc của từng model, không resume checkpoint smoke.
- Tham số khai báo và tham số trainer thực nhận trùng; so với smoke, tham số được đổi là `epochs: 1 → 2`. Batch 8, seed 0, AMP false, workers 2, cache false, optimizer SGD giữ nguyên.
- `close_mosaic=2` làm **mosaic tắt cả hai epoch pilot**, khác hiệu lực smoke 1 epoch. Khác biệt này đã được ghi trước khi chạy và xác nhận trong log; không được gọi pilot là đối chứng chỉ thay warmup.
- File khóa tồn tại trước lượt train theo chuỗi commit/mốc giờ được bàn giao. Hai lượt chạy ghi HEAD `6f2ca14`, tuần tự v8n rồi 11n. Không có bằng chứng trong gói cho việc chạy thêm một lượt pilot trên dataset thật.

### Đo trước học và diễn biến

Callback trước học đặt ở `on_train_start`, sau trainer dựng model/optimizer/EMA. Receipt ghi optimizer chưa có state, EMA updates = 0, không gradient, model bằng EMA, đúng thứ tự 5 lớp. Mã đo tạo bản sao model, không gọi inference trực tiếp trên model đang train.

| Model | Detection ≥ 0,001 trước học | Sau epoch 1 — model hiện tại | Sau epoch 2 — model hiện tại | Trung vị điểm max/ảnh trước học → epoch 2 |
|:--|--:|--:|--:|:--|
| YOLOv8n | 0 | 631 | 7.496 | 0,000185 → 0,1091 |
| YOLO11n | 0 | 18 | 3.798 | 0,000278 → 0,05535 |

AP50 car trong receipt của model hiện tại tăng lần lượt **0 → 0,3753 → 0,5101** và **0 → 0,00495 → 0,4131**. Model hiện tại, EMA trong bộ nhớ và checkpoint nạp lại được ghi riêng; không gộp các nguồn này.

Kết luận phù hợp: **trong pilot, confidence thấp đã tồn tại trước bước tối ưu đầu tiên, sau đó đầu ra cải thiện trong hai epoch**. Điều này loại bỏ cách giải thích rằng mọi mức confidence thấp quan sát được nhất thiết do bước học gây ra. Nó chưa xác định trạng thái trước học của chính lượt smoke vòng 35, cũng chưa tách tác động warmup, mosaic hoặc từng thành phần khởi tạo.

Receipt nạp trọng số khớp log `Transferred`: v8n 319 tensor chép nguyên + 3 bias remap + 33 tensor mới; 11n 448 + 3 + 48. Đây là đối chiếu log/mã, chưa phải reviewer mở `.pt` và so tensor.

### Warmup và cập nhật thực tế

Tính lại trực tiếp từ CSV, không chỉ đọc cờ tổng kết:

- Mỗi model: **84 batch = 42 × 2 epoch**; chỉ số liên tiếp, loss tổng và từng thành phần hữu hạn.
- Warmup **42 batch**. LR lệch tối đa so công thức độc lập **1,39e-17**, momentum không lệch.
- Epoch 2 LR cả ba nhóm = **0,00505**.
- **17 lần cập nhật/model = 12 + 5**. Các batch cập nhật `[0,1,2,4,6,8,11,14,18,23,29,36,44,52,60,68,76]` khớp cả mô phỏng accumulation và tăng `ema.updates` trong CSV. AMP tắt trong cấu hình này.
- Có đủ 7 phép đo/model. Năm phép đo trong training và ba callback ghi không còn khác trạng thái sau khôi phục; log có ghi RNG CUDA bị tiêu thụ rồi khôi phục.

Mã snapshot bao gồm trọng số/buffer, gradient, mode, optimizer/scheduler/scaler, một số biến trainer và RNG/cờ. Log A/B CPU hỗ trợ tính không gây thay đổi trên fixture đã thử. Không mở rộng thành bảo đảm mọi cấu hình CUDA, worker hay augmentation đều bất biến.

## 4. Lỗi đo `final_eval` và tài nguyên

Đã tái hiện độc lập bằng callback fixture: base ghi đè metric epoch cuối thành metric `final_eval` và kéo `val_s` từ 2 s lên 52 s. Target giữ metric epoch cuối và 2 s, đồng thời ghi riêng kết quả cuối. Đây là sửa lỗi ghi nhận; không phải thay thuật toán training.

Các mốc giờ pilot cho thấy `train_s` đã loại thời gian đo chẩn đoán. `val_s` của runner là khoảng từ hết train đến `on_fit_epoch_end`, **có cả phần lưu/checkpoint**, không phải timer thuần validation. Mốc `on_val_start/end` cung cấp thời gian validation riêng. Dự toán vòng 35 chưa được tính lại từ pilot; phụ lục đã nói rõ điều này.

Tính lại từ mẫu tài nguyên thô:

| Model | Train epoch 1 / 2 | VRAM toàn GPU cao nhất | RAM available thấp nhất — bộ đo ngoài | RSS runner + con cao nhất |
|:--|:--|--:|--:|--:|
| YOLOv8n | 50,09 / 38,97 s | 2.655 MiB | **19 MiB** | 3.084 MiB |
| YOLO11n | 58,01 / 46,35 s | 2.892 MiB | **7 MiB** | 3.064 MiB |

Bộ đo trong runner thấy available thấp nhất 59/35 MiB; khác thời điểm lấy mẫu nên không mâu thuẫn với 19/7 MiB ở bộ đo ngoài. Chu kỳ khoảng 1 giây có thể bỏ lỡ đỉnh. RSS cộng tiến trình có thể đếm trùng trang dùng chung. Không suy ra OOM hoặc paging chỉ từ các giá trị này.

**RAM là điểm cần xử lý trước lượt dài.** Các lượt nhỏ đã hoàn tất, nhưng số đo không chứng minh dataset đầy đủ sẽ chạy ổn với cùng lượng ứng dụng nền. Giải phóng RAM bằng cách người dùng đóng ứng dụng không cần thiết rồi ghi lại RAM trước khi train; không cần thêm pilot để kết luận lại điều đã đo.

## 5. Giới hạn và bước tiếp

- Dev smoke chỉ có GT car/truck/bus = 542/6/4; motorcycle/bicycle = 0. Không dùng AP này để xếp hạng hai model hoặc kết luận về chất lượng 5 lớp.
- Hai epoch trên 334 ảnh không chứng minh hội tụ, khả năng tổng quát hóa, chất lượng tracking hoặc độ ổn định nhiều giờ. Không yêu cầu thêm ablation warmup/mosaic trong phạm vi hiện tại.
- Nhãn train vẫn là snapshot bên thứ ba đã được người dùng chấp nhận có điều kiện; review này không xác thực lại nguồn phát hành chính thức.
- Kế hoạch hiện tại cho phép giữ hai model và chuẩn bị bước xuất dataset đầy đủ. **Stride 5, khoảng 8,5 GB xuất dữ liệu, ngân sách 15 epoch/model và hai kịch bản thời gian phải được người dùng duyệt riêng.**
- Nếu người dùng duyệt bước tiếp, ghi nhận review này, lưu trữ đúng delta/hash rồi mới chuyển base sang `0c0a07a`. Không sửa lại số liệu pilot, không chạy lại 2 epoch chỉ để bàn giao tài liệu.

## 6. Tái lập kiểm của reviewer

ZIP bằng chứng đi kèm có script, phần mã/receipt cần thiết, log kiểm và `SHA256SUMS.txt`. Trong thư mục đã giải nén:

```text
python check_package.py --zip <duong-dan-review_delta_latest.zip> --out evidence/package_check_rerun.json
python check_round37.py --tree tree --out evidence/independent_rerun.json
python check_runner_isolated.py --tree tree --base base_preimages --out evidence/runner_isolated_rerun.json
```

Hai script kiểm dữ liệu/hàm cô lập dùng Python + NumPy + PyYAML; script kiểm gói dùng thư viện chuẩn + Git. Chúng không nạp model, không train, không inference và không SSH Nano. Không cần gửi dataset hoặc weights để lặp lại các kiểm này.
