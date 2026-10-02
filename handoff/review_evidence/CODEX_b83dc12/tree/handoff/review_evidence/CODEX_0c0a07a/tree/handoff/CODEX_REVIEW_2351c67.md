# Review vòng 35 — `2351c67`

Ngày review: 2026-10-02. **PASS WITH LIMITATIONS cho khóa split, xuất tập con và smoke 1 epoch.** Không phát hiện lỗi chặn các bước này trong bằng chứng được bàn giao. Có **hai điểm diễn giải cần sửa trước khi dùng báo cáo để chốt ngân sách training dài**, nêu dưới đây.

Kết luận này không duyệt pilot thêm epoch, xuất dataset đầy đủ hoặc training dài. Reviewer không chạy training, không thay repository Windows và không thao tác Nano.

## Phạm vi và tính toàn vẹn

- Base: `ccd801fb7c517edd3c61b412d015a8ae4d7faba6`.
- Target: `2351c677c9a330050ebd152f10f46ed3922a431f`.
- ZIP: **378.096 byte**, SHA256 `76e7d2addaad29fdc3a42e91fb0a7fd2851864dcf47fc233ea52dbd688b68c1d`.
- ZIP sạch CRC; **74 payload + 1 alias** khớp hash và số byte.
- **6 preimage** khớp snapshot đã lưu; áp patch nguyên byte tái tạo đúng **75/75 đường dẫn thay đổi**. Bằng chứng review `ccd801f` mang sang khớp **15/15**.
- Không có thay đổi `src/` trong delta. `configs/` thêm split và quy tắc giữ byte.
- Không xác minh độc lập toàn cây 3.849 file Git; con số đó thuộc kiểm chứng trên máy của Claude.

## 1. Split và tập con smoke

Reviewer tính lại từ split và manifest nhãn đã nghiệm thu:

| Kiểm | Kết quả |
|:--|:--|
| train-fit / dev | **1.261 / 139 video**, không trùng; không tên lặp trong mỗi phần |
| Hợp hai phần | Đúng đủ **1.400 video** trong manifest native train đã kiểm |
| Trùng tên với 200 video native val | **0** |
| Fingerprint nguồn nhãn | Khớp `d20244932880e4d4bce842417f9eaa2cc24787a1fdaa72e64ad2faaafa6fd221` |
| Hash danh sách split | Khớp `a59a8d0672f30d6e9ff90d4347965fa003b299bb79995c0883796712446ca723` |
| Chọn smoke bằng RNG seed 0 | Tái lập đúng **16 video train + 4 video dev** đã ghi |
| Tổng các tầng và cách làm tròn | Khớp 1.400 video và 139 dev |

File split target giữ **1.499 CRLF**, SHA256 `b8a2848e3da053a2b1b79e7fa6cfad812540315af6e48b12b4174a40efc06cdb`. Chuẩn hóa LF cho đúng hash blob ban đầu `bc9c85d1e71891ec3247b602c9bb5e54b2c047f9e7fe9405a8baa8c6b8ce429a`. Các hash split trong log xuất, kiểm dataset và hai runner đều khớp target. Việc sửa xuống dòng không thay danh sách video.

Thứ tự trong bằng chứng nhất quán: split được tạo/khóa trước xuất dataset; smoke lỗi lần đầu chưa vào epoch; hai lượt thành công dùng split đó. Reviewer không có lịch sử Git đầy đủ để độc lập xác thực thời điểm commit.

Log xuất/kiểm và runner ghi cùng **334 ảnh train, 84 ảnh dev**, cùng hash dataset, manifest và danh sách video. Các phép kiểm ảnh trùng nội dung, pixel/nhãn thực nạp và 0 `.npy` là bằng chứng từ Windows; reviewer không có dataset để chạy lại. Không tái tính thành viên từng tầng từ 1.400 JSON hoặc số box từ ảnh/nhãn nguồn trong vòng này. Đối chiếu UA-DETRAC chỉ theo tên video, đúng phạm vi đã ghi.

## 2. Smoke và công cụ log

- Chạy nguyên `test_do_tai_nguyen_ngoai.py`: **12/12 PASS** trên Linux/Python 3.12. Môi trường E1 tại đây là UTF-8, không phải cp1252 Windows; tái hiện lỗi Windows của mã cũ được đối chiếu qua log bàn giao, không gọi là reviewer tự tái hiện errno 22.
- Hai JSON runner và log `.meta` ghi exit 0, đúng **1 epoch, batch 8, ảnh 640, SGD, seed 0, cache false**, không resume.
- Hai model dùng cùng fingerprint dataset; kiểm thực nạp ghi 334/84 ảnh, 0 thiếu/thừa/nhãn khác/`.npy`.
- Loss hữu hạn; checkpoint được runner báo nạp lại đúng năm lớp và không lệch tham số. Reviewer không có GPU/checkpoint để lặp lại training hoặc kiểm byte checkpoint.
- Dev smoke có GT cho car/truck/bus, không có motorcycle/bicycle. Vì vậy mAP là trung bình **ba lớp có GT**, không chứng minh chất lượng đủ năm lớp.

## 3. R35-F1 — diễn giải nguyên nhân giảm confidence vượt bằng chứng

**Mức: P2, cần đính chính báo cáo; không làm mất hiệu lực smoke đã chạy.**

Vị trí: `13_chan_doan_khong_detection.py` (`CKPT` và vòng dự đoán), `REVIEW35_CHI_TIET.md` mục 7/10, `DETECTOR_TRAIN_PLAN.md` mục 8, và câu kết luận trong brief.

Script chẩn đoán so **checkpoint COCO 80 lớp** với **checkpoint smoke 5 lớp**. Nó không lưu hay đo model 5 lớp sau khi khởi tạo/nạp trọng số nhưng **trước optimizer step đầu tiên**. Log còn ghi rõ:

- `Overriding model.yaml nc=80 with nc=5`;
- YOLOv8n nạp `322/355` items; YOLO11n nạp `451/499` items.

Do đó, chênh confidence giữa hai checkpoint chưa tách ảnh hưởng chuyển model sang năm lớp khỏi ảnh hưởng các bước học. Câu “smoke 1 epoch không warmup làm điểm lớp giảm mạnh” chưa có bằng chứng nhân quả. Các tham số và snapshot LR bàn giao cũng không thay thế log đầy đủ warmup/optimizer thực chạy; không có source/hash Ultralytics 8.4.141 kèm theo để kiểm riêng nhánh này.

Hạ conf xuống `1e-6` cho car AP50 khoảng 0,233 là bằng chứng có dự đoán car phù hợp ở điểm thấp. Nó **không chứng minh toàn bộ đường ống, mọi lớp và tọa độ đều không có lỗi**, như câu “không phải lỗi đường ống” đang khẳng định.

**Sửa tối thiểu:** ghi “checkpoint smoke có confidence thấp hơn checkpoint COCO 80 lớp; nguyên nhân chưa xác định. Warmup và trạng thái khởi tạo model 5 lớp là các yếu tố cần phân biệt.” Giới hạn kết luận AP thấp ngưỡng ở đúng lớp car và dữ liệu đã quan sát. Không đổi ngưỡng đánh giá chính để làm AP đẹp hơn.

Không cần chạy thêm epoch để đính chính câu chữ. Nếu sau này được duyệt chẩn đoán thêm, nên đo model 5 lớp trước bước học và lưu số warmup iteration/LR thực tế; chỉ tăng epoch không tự tạo phép đối chứng nguyên nhân sạch.

## 4. R35-F2 — “cận trên” thời gian/VRAM chưa được bảo đảm

**Mức: P2 cho cách dùng dự toán; không phải sai phép tính.**

Vị trí: `15_du_toan_training.py` (`GIA_DINH`), `DETECTOR_TRAIN_PLAN.md` mục 5a, brief và bảng RAM/VRAM trong phụ lục.

Reviewer chạy lại script và tính lại độc lập công thức; bốn phương án khớp hoàn toàn ở độ chính xác báo cáo:

| Bước frame / epoch | Tổng hai model | Kịch bản chậm hơn |
|:--|--:|--:|
| 5 / 15 | **56,64 giờ** | **73,95 giờ** |
| 5 / 30 | 113,08 giờ | 147,61 giờ |
| 10 / 15 | 29,10 giờ | 37,99 giờ |
| 10 / 30 | 58,08 giờ | 75,81 giờ |

Nhưng mức cao dựa vào giả định thêm 5 ms/ảnh validation và hệ số 1,3; VRAM đầy đủ dựa vào cộng 0–300 MiB từ smoke. Các giả định được khai báo, song chưa có đo dài hoặc batch nhiều box trên tập đầy đủ để chứng minh chúng bao hết trường hợp xấu. Vì thế “cận trên khuyến nghị” và “VRAM ≤ khoảng 3,1 GB” dễ bị hiểu thành bảo đảm thời gian hoặc không OOM.

**Sửa tối thiểu:** gọi đây là **kịch bản dự toán với các giả định nêu rõ**; bỏ nghĩa bảo đảm của “cận trên”/dấu `≤`. Nêu thời gian và bộ nhớ có thể vượt kịch bản. Không tự bổ sung benchmark hoặc training để sửa tài liệu.

Phần RAM thấp có bằng chứng dùng `psutil.virtual_memory().available`, không chỉ là số trang “free”. Tuy nhiên log đã tổng hợp, không có chuỗi mẫu thô để reviewer tái tính đỉnh từng thời điểm. RSS cộng các tiến trình có thể đếm trùng trang chia sẻ, như báo cáo đã ghi.

## Kết luận và bước tiếp theo

Nghiệm thu có giới hạn **mốc split + smoke**, giữ nguyên split, dataset smoke, checkpoint và log thất bại/thành công. Hai điểm trên cần sửa cách diễn giải; **không có yêu cầu chạy lại smoke hay full suite chỉ để sửa tài liệu**.

Nhãn vẫn là snapshot bên thứ ba được chấp thuận có điều kiện; 183 lệch lớp/50 ngoại lệ ignore ở native val so với COCO lịch sử chưa được giải quyết và không thuộc vòng này.

Training dài, xuất dataset đầy đủ, số epoch/bước frame/ngân sách và pilot thêm epoch vẫn cần người dùng chốt. Không coi việc gửi delta hoặc yêu cầu “tiếp tục review” là đồng ý chạy pilot. Giữ nguyên Nano và các thí nghiệm tracking cũ.

Bằng chứng: `CODEX_REVIEW_2351c67_EVIDENCE.zip` gồm script kiểm, log, JSON kết quả và dữ liệu cần cho các phép đối chiếu nhỏ; không chứa ảnh, weights hoặc video.
