**CODEX REVIEW — `8bab154` — 2026-09-20**

**Kết luận: REQUEST CHANGES.** Có ba lỗi cần sửa trong công cụ chuẩn bị Nano trước khi dùng chúng làm điều kiện kiểm đúng đắn hoặc báo cáo benchmark. Không tìm thấy thay đổi thuật toán tracking hay số liệu UA-DETRAC/BDD100K trong delta này; không yêu cầu chạy lại các thí nghiệm đã chốt.

Review độc lập trên bản sao trong môi trường Linux/Python 3.12. Không truy cập máy Windows hoặc Jetson Nano của tác giả, không sửa source bàn giao, không commit/push.

**Snapshot và gói bàn giao**

| Mục | Đối chiếu |
|---|---|
| Base | `b6234653f616e8059825b59b4da70262e1f98dde` |
| Target | `8bab1547e87aa7a92944b6f88e6418dba2e6c429` |
| ZIP | 185.666 byte; 21 thành viên |
| SHA256 ZIP | `2d170edaa923dd823581374a125cf60acb6f43a2b1742f4d02175eea5bc8f07b` |
| Payload | 15/15 hash khớp |
| Dedup | `tests_latest.log` khớp bản theo ngày; thay đổi vẫn nằm trong patch |
| Patch | `git apply --check` và áp thật đều exit 0; 16/16 đường dẫn sau áp khớp hash target |
| Đường dẫn riêng bị cấm | Không có trong payload hay patch |

Một README còn thiếu trong thư mục review base cục bộ được lấy lại từ snapshot trước, đối chiếu **đúng SHA256 base** trong manifest trước khi áp patch. Đây là phục hồi đầu vào của reviewer, không phải lỗi gói. Không coi thư mục dựng lại là checkout đầy đủ của repository Windows.

**F1 — [P1] Benchmark báo thành công khi không đủ cặp đo theo giao thức**

Vị trí: `src/nano_bench_tracker.py`, phần tổng hợp `hieu_so_theo_cap`, tạo `tong_hop` và `return 0 if hieu else 1` (dòng 260–302 ở target).

CLI kiểm `--pairs` phải chẵn và ít nhất 5, nhưng điều kiện thành công cuối cùng chỉ cần **một** cặp hoàn chỉnh. Các cặp thất bại bị bỏ khỏi tổng hợp mà không làm mã thoát thất bại. Khi còn dưới 4 cặp, mã còn ghi min/max vào trường tên `iqr`.

Đã tái hiện qua CLI/orchestration benchmark thật với tiến trình tracker con dạng fixture: cặp 1 ghi CSV timing hợp lệ; cặp 2–6 trả exit 1. Đây là mô phỏng lỗi tracker, không phải phép đo phần cứng.

| Yêu cầu/kết quả | Quan sát |
|---|---|
| Lịch yêu cầu | 6 cặp: 3 AB + 3 BA |
| Lượt thực hiện | 12 |
| Lượt thất bại | 10 |
| Cặp được tổng hợp | 1, chỉ chiều AB |
| Exit của benchmark | **0** |
| JSON | `n_cap=1`, trung vị `0.5`, `iqr=[0.5, 0.5]` |

Chương trình có in cảnh báo thiếu cặp, nhưng vẫn xuất thống kê và mã thành công cho một lần chạy không đạt giao thức đã công bố. Điều này có thể khiến bước tự động tiếp theo tiếp nhận benchmark chưa đủ dữ liệu.

Yêu cầu sửa:

- Lưu mọi lượt và lỗi như hiện tại; đánh dấu lần chạy chưa hoàn chỉnh, trả exit khác 0 nếu không đạt số cặp/cân bằng đã khóa. Với giao thức hiện tại, phải đủ 6 cặp, 3 AB và 3 BA.
- Nếu muốn cho phép loại cặp, định nghĩa tiêu chí và số lượng tối thiểu trước khi đo; không tự hạ tiêu chuẩn sau khi có lỗi.
- Không gọi min/max là IQR. Nếu số mẫu chưa đáp ứng quy tắc công bố, ghi rõ chưa đủ mẫu, giữ thống kê mô tả riêng.
- Thêm regression qua CLI: đủ 6 cặp; chỉ một cặp thành công; thiếu một nửa cặp. Kiểm cả exit, trạng thái JSON, số lượt lưu và điều kiện được tổng hợp.

**F2 — [P2] Ghép hàng bằng thứ tự `(frame, x, y)` làm parity báo sai**

Vị trí: `src/nano_parity.py`, `so_mot_video()`, hai lệnh `np.lexsort` và phép so theo chỉ số hàng (dòng 84–103).

Tọa độ đang được phép lệch tới `1e-3 px`, nên không thể dùng thứ tự tuyệt đối của tọa độ để khẳng định hai hàng tương ứng. Hai box có x gần nhau có thể đổi thứ tự sau sai số nhỏ; hai box trùng x/y cũng phụ thuộc thứ tự dòng trong file. Mã hiện tại ghép nhầm hàng rồi tính cả bbox và phân hoạch ID trên phép ghép sai đó.

Fixture đã chạy:

```text
# ref: frame,id,x,y,w,h,conf
1,1,20,100,50,40,0.9
1,2,40,500,60,50,0.8
2,1,100.0,100,50,40,0.9
2,2,100.0001,500,60,50,0.8

# test: giữ nguyên hai dòng frame 1; frame 2 chỉ đổi x như sau
2,1,100.0002,100,50,40,0.9
2,2,99.9999,500,60,50,0.8
```

| Đại lượng | Theo hàng tương ứng `(frame, id)` | Công cụ đang báo |
|---|---:|---:|
| Lệch bbox tối đa | `0.0002 px` | **`400 px`** |
| Lệch confidence | `0` | `0.1` |
| Giá trị ID/liên kết | Giữ nguyên | ID khác, phân hoạch khác |
| CLI | Phải đạt tolerance đã công bố | **exit 1** |

Ca thứ hai: cùng frame, hai box cùng `(x,y)` nhưng khác `(w,h)`; file test chỉ đảo thứ tự hai dòng, nội dung không đổi. Công cụ vẫn FAIL, báo bbox lệch `10 px`.

Yêu cầu sửa:

- Với kiểm ID thô, đối chiếu bằng khóa `(frame, id)` hợp lệ rồi so bbox/conf; kiểm đủ tập khóa và tránh chấp nhận khóa trùng.
- Với chẩn đoán phân hoạch khi đánh số ID khác nhau, cần phép ghép hình học một-một theo tolerance và song ánh ID cố định trên toàn sequence. Không ánh xạ lại ID độc lập từng frame. Nếu hình học không phân biệt được ứng viên, báo rõ sự mơ hồ.
- Giữ nguyên tolerance đã công bố; không nới tolerance để che lỗi ghép hàng.
- Thêm regression hai ca trên, đổi số ID làm đảo thứ tự ID, và ca thật sự đổi liên kết ở một frame. Ca đổi liên kết phải tiếp tục FAIL.

**F3 — [P2] Không lưu bằng chứng CPU/clocks/RAM cần cho benchmark Nano**

Vị trí: `src/nano_bench_tracker.py`, `trang_thai_thiet_bi()`, `MauTegrastats.dung()/tom_tat()` và `chay_mot_luot()` (dòng 71–128, 165–176).

`MauTegrastats` nhận dòng thô trong `self.mau`, nhưng chỉ đưa bản tóm tắt nhiệt độ/GPU frequency vào JSON. Dòng thô không được lưu. CPU frequency, RAM/swap bị bỏ; trạng thái thiết bị chỉ giữ `nvpmodel`, governor và nhiệt độ, không ghi trạng thái `jetson_clocks` như giao thức yêu cầu.

Đã đưa hai dòng fixture vào đúng parser hiện tại:

```text
RAM 300/3964MB (lfb 600x4MB) SWAP 0/1982MB CPU [25%@1479,off,off,off] EMC_FREQ 2%@1600 GR3D_FREQ 0%@76 PLL@35C CPU@40C PMIC@50C GPU@39C
RAM 3500/3964MB (lfb 1x4MB) SWAP 900/1982MB CPU [25%@102,off,off,off] EMC_FREQ 2%@1600 GR3D_FREQ 0%@76 PLL@35C CPU@40C PMIC@50C GPU@39C
```

Hai đầu vào khác rõ CPU frequency và mức dùng RAM/swap nhưng cho **cùng bản tóm tắt được giữ lại**: `n_mau=1`, nhiệt tối đa 40°C, trung bình 39,5°C, GPU 76 MHz. Với tracker NumPy đo trên CPU, báo cáo này không giữ được dữ liệu cần để kiểm tra điều kiện chạy sau khi đo.

Yêu cầu sửa:

- Lưu nguyên mẫu `tegrastats` với mốc thời gian theo từng lượt; JSON trỏ tới log đó. Bảo toàn dữ liệu ngay cả khi tracker lỗi.
- Giữ CPU frequency và RAM/swap; lưu trạng thái clocks/power thực tế đọc được. Trường không đọc được phải là thiếu/unknown, không suy ra là không throttling.
- Kết thúc tiến trình/thread lấy mẫu và hoàn tất ghi log trước khi chốt báo cáo lượt đo.
- Thêm test bằng dòng mẫu và tiến trình giả, kiểm **artifact đã lưu**, không chỉ hàm parser. Có thể làm các test này trên PC; việc kiểm định dạng/output thật vẫn cần Nano.

**Các chỗ tài liệu cần sửa cùng đợt, không mở thêm thí nghiệm**

- `handoff/NANO_STATUS.md` mục 5: lệnh `python3 -c "import sys, ul_backend; ..."` không thêm `src` vào đường import. Chạy nguyên lệnh từ gốc repo trong môi trường sạch cho `ModuleNotFoundError: No module named 'ul_backend'`. Thêm `PYTHONPATH=src` hoặc `sys.path.insert(0, 'src')`; kiểm lệnh thực tế từ thư mục đã hướng dẫn. Bổ sung lệnh parity EKF tương ứng với điều kiện phải đạt cả CV/EKF.
- Thu hẹp kết luận mạng thành “chưa truy cập được Nano bằng các cách đã thử”. DNS/ARP/serial hiện có không chứng minh mọi đường kết nối đều không tồn tại.
- Trong brief/Nano status, đổi “hash sẽ lệch” thành điều kiện cụ thể về byte xuống dòng. Tách đối chiếu nội dung source khỏi môi trường/backend: fingerprint tổng thể khác giữa hai máy có thể là hành vi đúng. Không bỏ thông tin môi trường để ép fingerprint khớp.
- Đồng bộ giao thức 4c: tài liệu gọi số chính thức là chạy liên tục, nhưng lệnh mẫu có nghỉ 30 giây giữa các lượt; tài liệu cũng nói đo 450 frame sau warm-up trong khi video mẫu dài 202 frame và công cụ đo toàn phần còn lại. Chốt rõ đây là benchmark ghép cặp có nghỉ hay phép đo chạy liên tục, số frame đo và thời gian nghỉ; ghi chúng vào kết quả. Không đổi tiêu chí sau khi xem số Nano.
- IQR là khoảng tứ phân vị của các hiệu số, không phải CI. Việc IQR chứa 0 không tự nó là kiểm định ý nghĩa thống kê.

**Những gì đã kiểm chứng độc lập**

| Kiểm tra | Kết quả |
|---|---|
| `python src/test_nano_parity.py` | **10/10 PASS**, exit 0 |
| Hai fixture parity bổ sung | Tái hiện F2; CLI trường hợp nhiễu nhỏ exit 1 |
| Benchmark thật → tracker NumPy thật, CV/EKF, cache tổng hợp 12 frame | **12/12 lượt exit 0**, đủ 6 cặp, mỗi lượt đo đúng 10 frame sau warm-up 2 |
| Benchmark CLI với tracker con giả lập lỗi | Tái hiện F1: 10 lượt lỗi nhưng benchmark exit 0 |
| Parser telemetry với hai dòng mẫu | Tái hiện F3: dữ liệu CPU/RAM/swap bị mất |
| Full suite tác giả bàn giao | Log/CSV ghi **29/29 PASS**, exit 0; **không chạy lại độc lập toàn suite** |

Smoke tracker thật chạy với `BYTETRACK_BACKEND=np`, `lap` đã có trong môi trường review. Đây chỉ là kiểm công cụ trên máy review với cache tổng hợp, **không phải benchmark CV/EKF hoặc số Nano**. Không diễn giải các giá trị latency của smoke thành kết quả nghiên cứu.

Test hiện có PASS không phủ định F1–F3: ca gần/trùng tọa độ không có trong `test_nano_parity`; chưa có suite riêng cho orchestration và lưu telemetry của benchmark.

**Phạm vi và bước tiếp theo**

Không có thiết bị Nano trong phiên review; chưa kiểm JetPack/Python/aarch64, output `tegrastats` thật, clocks, detector TensorRT hoặc parity PC↔Nano. Không có ảnh/cache BDD thật tại đây để tái đo phần cứng. Các giới hạn này không ngăn tái hiện và sửa ba lỗi phần mềm đã nêu.

Đề nghị Claude sửa F1–F3 và các lệnh/tài liệu liên quan; chạy regression mới cùng smoke nhỏ, giữ nguyên kết quả/GT/lock UA-DETRAC và BDD. Chưa cần chạy lại các thí nghiệm cũ. Giữ base review `b623465` trong đợt sửa này để gói kế tiếp chứa trọn thay đổi đang được review; chưa chuyển base chỉ vì đã đọc một snapshot còn REQUEST CHANGES.

Khi có Nano: thu thập stack → xác minh backend NumPy → parity CV/EKF bằng công cụ đã sửa → đo theo giao thức đã chốt. Chỉ báo cáo số Nano khi có log đo thật.
