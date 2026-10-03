# Review vòng 39 — `1454b3d`

**Kết luận: REQUEST CHANGES — cần sửa R39-F1 trước khi nghiệm thu resume.**

Resume trên nhật ký nguyên vẹn có bằng chứng CPU/GPU; bộ giám sát RAM đã có kiểm chứng tốt hơn. Tuy nhiên, đường xử lý nhật ký bị ghi dở chưa an toàn: bước đọc chấp nhận bỏ qua dòng cuối hỏng, nhưng bước ghi tiếp để nguyên dòng đó. Sau khi nối sự kiện mới, nhật ký không đọc lại được và lần resume tiếp theo bị chặn.

Đây là một finding, có ba biến thể tái hiện. Không có bằng chứng rằng các lượt CPU/GPU đã bàn giao gặp lỗi này; không rút lại kết quả của các lượt đó. **Chưa bắt đầu training dài; giữ base review `b83dc12` trong lúc sửa.**

## 1. Gói và phạm vi kiểm

| Mục | Giá trị |
|:--|:--|
| Ngày review | 2026-10-03 |
| Base | `b83dc1295cad821b55c60ae15855331b870fa98f` |
| Target | `1454b3df4d03edb2536846bec3ec69545c52878b` |
| ZIP | `review_delta_latest.zip`, 1.607.478 byte |
| SHA256 | `b7674449fca6a7b6b0ee42d6c245601dd15b4e594ba8674a6e520c7c50d8b397` |
| Thành viên ZIP | 209; `testzip` đạt |
| Payload / alias trùng nội dung | 203 / 11, đều khớp số byte và SHA256 |
| Áp patch và đối chiếu | 214/214 đường dẫn thay đổi khớp target |
| Preimage so với snapshot trước đã có | 8/8 khớp từng byte |
| Bằng chứng review `b83dc12` được chép vào gói | 137/137 file khớp `SHA256SUMS.txt`; báo cáo khớp SHA256 đã nghiệm thu |

Không thấy đường dẫn riêng bị cấm, dataset, weights, video hay NPZ trong payload được kiểm. Tôi kiểm phần cây thay đổi và các preimage có sẵn, **không có một checkout Git đầy đủ độc lập để xác nhận toàn bộ cây target**.

Đã đọc mã runner, resume, monitor, test và phụ lục vòng 39; đối chiếu log CPU/GPU, CSV theo batch, nhật ký và CSV RAM. Không sửa mã dự án, không commit, không SSH Nano, không train hoặc chạy inference.

## 2. R39-F1 — P1: nối tiếp nhật ký sau dòng cuối ghi dở làm mất khả năng resume tiếp

**Vị trí:** `src/tiep_tuc_huan_luyen.py`, `ghi_nhat_ky()` dòng 147–155; `doc_nhat_ky()` dòng 158–175. Đường sử dụng liên quan: `chuan_bi_tiep_tuc()` dòng 321 trở đi, `sao_luu_truoc_tiep_tuc()` dòng 405–444, `GhiLuotDayDu.sau_chuan_bi()` dòng 493–499.

**Điều kiện kích hoạt:** tiến trình bị ngắt trong lúc ghi dòng nhật ký cuối; vẫn còn ít nhất một epoch/checkpoint hợp lệ trước đó. Đây chính là tình huống hàm đọc hiện tuyên bố hỗ trợ bằng việc bỏ qua dòng cuối ghi dở.

Trình tự trong mã:

1. `doc_nhat_ky()` trả lại các sự kiện hợp lệ và cảnh báo, bỏ qua dòng cuối không parse được. Nó không sửa file.
2. `chuan_bi_tiep_tuc()` giữ cảnh báo trong kế hoạch nhưng không xử lý phần dở. Bước sao lưu lưu checkpoint và CSV, không sao lưu/sửa nhật ký.
3. Callback bắt đầu lượt resume gọi `ghi_nhat_ky()`, mở file bằng chế độ append và ghi ngay JSON mới.
4. Nếu đuôi cũ không có newline, hai đoạn JSON dính vào nhau. Nếu đuôi hỏng có newline, dòng hỏng trở thành dòng giữa. Khi ghi thêm sự kiện epoch, lần đọc kế tiếp ném `LoiTiepTuc: dong 3 hong o GIUA nhat ky`.

Ngay cả một dòng JSON cuối **hợp lệ nhưng thiếu newline** cũng bị nối dính và trở thành dòng hỏng. Không cần checkpoint hỏng để kích hoạt lỗi.

### Tái hiện độc lập

`REPRO_REVIEW_1454b3d.py` dùng trực tiếp hai hàm đọc/ghi của target trong thư mục tạm. Không dùng torch, model, dataset hoặc GPU; không giả lập hai hàm này.

| Đuôi nhật ký trước khi ghi tiếp | Đọc trước | Đọc sau khi ghi `bat_dau` và `epoch_da_luu` mới |
|:--|:--|:--|
| Sạch, kết thúc bằng newline | 2 sự kiện | **ĐẠT:** 4 sự kiện |
| JSON ghi dở, không newline | 2 sự kiện + cảnh báo | **LỖI:** dòng 3 hỏng ở giữa |
| JSON ghi dở, có newline | 2 sự kiện + cảnh báo | **LỖI:** dòng 3 hỏng ở giữa |
| JSON cuối hợp lệ, không newline | 3 sự kiện, không cảnh báo | **LỖI:** dòng 3 hỏng ở giữa |

Kết quả script: **exit 1, `bug_count=3`**. Hai sự kiện hợp lệ ban đầu vẫn nguyên byte; lỗi phát sinh ở ranh giới nối thêm. Script này là phép tái hiện phần nhật ký, không phải phép train/resume toàn bộ Ultralytics. Đường gọi thật đã được kiểm qua mã như mô tả trên.

Test U1a hiện chỉ kiểm đọc bỏ qua đuôi hỏng. U1b kiểm từ chối lỗi ở giữa. Chưa có phép kiểm chuỗi **đọc chấp nhận → resume ghi tiếp → đọc lại thành công**, nên hai test cũ không bắt được lỗi này.

**Ảnh hưởng:** một lượt resume có thể bắt đầu và tiếp tục tính toán, nhưng lịch sử dùng để xác minh checkpoint/quỹ đạo bị hỏng; lần ngắt tiếp theo có thể không tiếp tục được nữa nếu không can thiệp thủ công. Điều này đi ngược mục đích phục hồi sau gián đoạn của thay đổi vòng 39. Không khẳng định checkpoint đã bị hỏng hoặc số liệu các lượt đã bàn giao bị sai.

### Yêu cầu sửa và kiểm đủ để đóng finding

- Giữ `--dry-run --tiep-tuc` chỉ đọc.
- Trong đường resume thật, trước lần append đầu tiên, bảo toàn bản gốc của nhật ký và hash; xử lý đuôi ghi dở bằng cách không làm mất các sự kiện hợp lệ. File đang dùng sau xử lý phải có ranh giới newline hợp lệ. Có thể đặt việc sửa ở bước chuẩn bị ghi, không bắt buộc đặt vào hàm append.
- JSON cuối hợp lệ nhưng thiếu newline phải được giữ nguyên sự kiện. Dòng hỏng ở giữa vẫn phải bị từ chối; không bỏ qua lỗi tùy tiện.
- Thêm regression qua đường resume thật: đuôi ghi dở có/không newline; JSON hợp lệ thiếu newline; nhật ký sạch; lỗi giữa file. Sau resume và ghi thêm epoch, phải đọc lại được, lịch sử hợp lệ còn đủ và bản sao trước sửa còn nguyên. Kiểm cả dry-run không ghi.
- Dùng fixture nhỏ/kiểm kỹ thuật hiện có. **Không cần chạy dataset đầy đủ hoặc training dài để đóng lỗi này.** Nếu sửa ở đường chuẩn bị riêng, dùng test tích hợp làm tiêu chí nghiệm thu; script hai hàm đi kèm chỉ là bằng chứng của mã hiện tại.

## 3. Những phần đã kiểm đạt

### Bộ giám sát

Tôi chạy các test được cung cấp M1/M2/M3/M7/M8/M9/M10/M11, không sửa nội dung test, trên Linux/Python 3.12 với psutil 7.2.2: **19/19 phép đạt, exit 0**. Bao phủ PID tiến trình thật, worker xuất hiện/kết thúc, exit 3, log Unicode/dữ liệu lỗi, gắn PID, giữ thư mục cũ, truy vấn trạng thái và tính lại CSV.

M4/M5/M6 về mã thoát Windows và Ctrl+Break **không chạy lại trong môi trường này**. Log Windows của tác giả ghi 23/23, được đối chiếu riêng; không gọi đó là 23 phép do reviewer tự chạy.

Lần thử đầu trong sandbox của reviewer không dùng được: psutil báo `NoSuchProcess` ngay khi truy vấn PID của chính tiến trình đang chạy. Chạy lại cùng test ngoài sandbox đạt 19/19. Log lần đầu được giữ để minh bạch; không quy lỗi môi trường đó thành finding của dự án.

### Đối chiếu độc lập bằng chứng GPU và RAM

`check_receipts.py`: **111/111 điều kiện đạt**, gồm hash cấu hình và 20 file bằng chứng, PID/exit code, số mẫu và thống kê RAM, chuỗi epoch, batch, LR/momentum, nguồn checkpoint và sự nhất quán của 13 kết quả kiểm khôi phục giữa các bản ghi.

| Lượt | Mẫu RAM | Exit | RSS chính max (MiB) | Tổng RSS max (MiB) | RAM available min (MiB) |
|:--|--:|--:|--:|--:|--:|
| v8n liên tục | 164 | 0 | 1.169,2 | 2.945,0 | **17** |
| v8n bị ngắt | 64 | 15 | 1.277,0 | 3.085,5 | 45 |
| v8n resume | 124 | 0 | 1.262,7 | 3.393,5 | 101 |
| 11n liên tục | 188 | 0 | 1.195,0 | 3.533,1 | 163 |
| 11n bị ngắt | 77 | 15 | 1.123,3 | 3.235,9 | 263 |
| 11n resume | 137 | 0 | 1.393,1 | 3.864,8 | 33 |

PID trong từng CSV khớp PID monitor và PID runner tự ghi trong sự kiện bắt đầu. Tổng RSS có thể đếm trùng trang chia sẻ; available là số toàn máy. Không diễn giải các số này thành lượng RAM duy nhất do training chiếm hoặc bằng chứng OOM.

Với mỗi model: liên tục và ngắt/resume đều có đủ 126 hàng batch; resume ghi epoch 2–3, không tăng tổng 3 epoch của phép kiểm. LR, momentum và batch sau resume khớp lượt liên tục; log mosaic `p` lần lượt 1/0/0. Loss và hash thứ tự batch khác ở 84/84 batch sau resume, phù hợp giới hạn đã công bố.

**Giới hạn xác minh:** 13 phép kiểm optimizer/EMA được đối chiếu mã và các receipt nhất quán; reviewer không có checkpoint và môi trường torch/Ultralytics để tự nạp lại tensor hay lặp training CPU/GPU. Đây không phải xác minh tensor độc lập.

Log tác giả được đối chiếu: CPU lần cuối 68/68; detector 92/92; pilot 34/34. Các lần CPU trước có lịch sử riêng: lần 1 lỗi test, lần 2 57/59, lần 3 60/60, lần 4 64/64. Không gộp chúng thành một lần full suite mới.

## 4. Giới hạn và bước tiếp theo

- Chấp nhận ranh giới resume sau epoch đã lưu; không đòi kết quả trùng từng bit với lượt liên tục. EMA/FP16, RNG, augmentation và gradient tích lũy dở đã được nêu đúng là giới hạn của đường khôi phục đang dùng.
- **17 MiB available là quan sát có thật.** Nó không chứng minh lượt dài sẽ OOM, cũng không chứng minh đủ an toàn cho 26–30 giờ. Việc đóng ứng dụng hoặc đổi worker cần quyết định riêng trước khi chạy; review này không tự đổi cấu hình.
- Ctrl+C thật khi train với DataLoader chưa được kiểm; bài spy và Ctrl+Break với chương trình giả không thay thế điều đó. Giữ nguyên giới hạn, không gọi là đã kiểm.
- Trường mosaic cũ của pilot vòng 37 đã được công khai là không đáng tin. Vòng 39 có phép đọc transform riêng. Không cần mở thêm nghiên cứu/pilot để sửa R39-F1.
- Việc ghi sớm bốn tài liệu trong lúc kiểm GPU là vi phạm quy tắc làm việc đã được báo cáo. Chưa thấy bằng chứng nó làm thay đổi tính toán GPU; không biến suy đoán thành kết luận về kết quả.

**Việc cần làm tiếp:** sửa riêng R39-F1, kiểm hồi quy ngắn, cập nhật brief/phụ lục và gửi delta mới từ `b83dc12`. Sau khi review đạt mới chốt quyết định khởi chạy; số epoch, ngân sách và thời điểm training dài vẫn chưa được duyệt trong vòng này.

## 5. Gói bằng chứng reviewer

`CODEX_REVIEW_1454b3d_EVIDENCE.zip` gồm báo cáo này, script tái hiện, mã nguồn target liên quan, checker gói/CSV, log reviewer, bằng chứng vòng 39 cần để tính lại và `SHA256SUMS.txt`. Không chứa dataset, checkpoint, weights hay môi trường Python cài thêm.

Sau khi giải nén, có thể tái hiện lỗi nhật ký không cần torch:

```text
python REPRO_REVIEW_1454b3d.py --repo tree --out repro_local.json
```

Muốn đối chiếu với repo Windows thật, thay `tree` bằng `D:/LABOTARY/bytetrack`. Script chỉ ghi trong thư mục tạm và file JSON đầu ra, không sửa repo.
