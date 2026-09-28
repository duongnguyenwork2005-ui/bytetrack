# Review vòng 21 — `9f210e2`

Ngày: 28/09/2026. Reviewer: Codex.

**Kết luận: REQUEST CHANGES — còn ba lỗi mức P2 trong entry point/đo thời gian.** Ba điểm R20-P2/P3 đã được sửa. Có bằng chứng cho luồng mặc định TensorRT → tracker chạy thật trên Nano; các phát hiện dưới đây không làm mất giá trị smoke đó hoặc benchmark vòng 20. Chưa đóng nghiệm thu toàn bộ vòng 21.

## 1. Những gì đã kiểm độc lập

- ZIP **2.650.980 byte**, 194 thành viên; SHA256 `0af0e399389209488778908e10f78e32bf382b95076996681469e4200ed79dbc` khớp. Base `45e086ac47d778c463719b9b29398b6d99078542`, target `9f210e2e097f4a6ed9ebeb2f8fdd4b11ba6062fe`.
- **188/188 payload** khớp hash. Áp patch lên snapshot base đã giữ: **202/202 đường dẫn** khớp target. Không phát hiện file riêng, weights, engine, video hoặc NPZ trong payload/patch.
- Bản sao Nano: **152/152 file có mặt** khớp SHA256 và số byte trong danh sách nguồn; 4 NPZ mới không có, đúng khai báo. NPZ vòng 16 tiếp tục được khai báo riêng. Không tuyên bố đã kiểm độc lập toàn bộ 2.262 file của repository gốc.
- `test_review_delta.py`: **19/19 PASS** do reviewer tự chạy. Đóng R20-P2; patch giữ bytes, các ca thêm/sửa/xóa CRLF, CR lẻ, binary và phát hiện patch hỏng đều đạt. Hai diễn giải R20-P3 đã sửa trong tài liệu.
- `test_nano_stream.py`: **14/14 PASS, 1 SKIP** do reviewer tự chạy, gồm SIGINT, stop-file, con bị kill, mất frame, ảnh hỏng, cửa sổ credit và live→cache→replay với detector mô phỏng. SKIP là detector PyTorch thật do thiếu model/ảnh gốc.
- Môi trường review: Linux, Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, lap 0.5.13, OpenCV headless 4.11.0. Không chạy TensorRT hoặc truy cập Nano trong review này.
- Log full suite bên thực thi có **34 suite PASS** trên `42dff0c`; đây là đối chiếu log, không phải full suite mới của reviewer.
- `run_tracking`, `det_cache`, tracker CTRV, EKF/UKF và backend selector không đổi so với base. **36/36 hash mã** được ghi trong metadata sáu lượt Nano khớp sáu module tương ứng của target sau chuẩn hóa LF.

## 2. N21-F1 — P2: Timer toàn luồng bắt đầu sau khi detector đã xử lý frame

Vị trí: `src/nano_live.py` phần nhận HELLO, chuẩn bị tracker và đặt `t_vl` (đặc biệt dòng 540–571); `src/nano_detect_stream.py` phần gửi HELLO rồi bắt đầu vòng đọc frame (dòng 301–308).

Con bắt đầu đọc/infer ngay sau HELLO. Cha còn đọc YAML, dựng cấu hình và bật telemetry rồi mới đặt `t_vl`. Vì vậy một phần xử lý những frame đầu đã xảy ra **trước cửa sổ đo**, nhưng tử số FPS vẫn tính đủ các frame đó. Docstring nói timer bắt đầu từ nhận HELLO cũng chưa đúng với mã.

**Tái hiện qua entry point thật:** fixture 8 ảnh, detector mô phỏng, chèn trễ 1 giây vào `load_tracker_cfg()` sau HELLO. Chương trình exit 0 và xử lý đủ 8 frame, nhưng:

| Đại lượng | Kết quả |
|---|---:|
| `wall_vong_lap_s` báo cáo | 0,011898 s |
| Từ bắt đầu đọc frame 1 đến xong tracker frame 8 | 1,057024 s |
| FPS báo cáo | 672,37 |
| FPS theo khoảng thời gian bao đủ 8 frame trên | 7,57 |

Đây là kiểm thử độ nhạy có chèn trễ, **không phải benchmark hiệu năng**. Nó cho thấy lỗi đo phụ thuộc thời gian chuẩn bị ở cha.

**Ảnh hưởng trên dữ liệu Nano thật:** mọi lượt đều có `wall_vong_lap_s` nhỏ hơn khoảng từ `t_bat_dau_con` đầu tiên đến `t_xong_cha` cuối cùng:

| Lượt | Thiếu tối thiểu (ms) | FPS đang báo | FPS từ đọc frame 1 đến xong tracker frame 300 |
|---|---:|---:|---:|
| CV r1 | 67,929 | 9,1148 | 9,0960 |
| EKF r1 | 74,450 | 9,3737 | 9,3520 |
| EKF r2 | 90,393 | 9,4425 | 9,4157 |
| CV r2 | 71,732 | 9,3644 | 9,3435 |
| CV r3 | 71,816 | 9,2903 | 9,2697 |
| EKF r3 | 81,290 | 9,2862 | 9,2629 |

Cột cuối là **một phạm vi đo được xác định rõ từ timestamp sẵn có**, không tự thay thế toàn bộ định nghĩa `wall_vong_lap` cũ: nó chưa gồm ghi kết quả cuối lượt. Mức lệch FPS hiện tại khoảng 0,21–0,28%; kết luận tốc độ xấp xỉ **9,3 FPS** vẫn được hỗ trợ. `fps_on_dinh` dùng mốc hoàn thành sau warm-up không mắc lỗi mất đoạn đầu này; reviewer tính lại đều khớp.

**Sửa:** chốt ranh giới đo và đồng bộ bắt đầu xử lý (chẳng hạn HELLO → cha sẵn sàng → START), hoặc đo bằng các mốc monotonic bao đủ công việc được đếm. Ghi rõ khởi tạo, xử lý frame, drain và ghi kết quả. Thêm regression cố ý làm chậm chuẩn bị ở cha để không thể đếm frame nằm ngoài cửa sổ. Giữ số cũ và ghi đính chính; không sửa timestamp gốc.

## 3. N21-F2 — P2: `--cache-root` không được giữ trong lệnh replay

Vị trí: `src/nano_live.py` dòng 473, 639–643 và 646–656.

Live ghi cache đúng thư mục tùy chọn, nhưng `replay_argv` chỉ giữ cache ID. `run_tracking.py` lại tìm ID đó dưới `det_cache.CACHE_ROOT` mặc định.

**Tái hiện:** chạy live với `--save-cache review_custom --cache-root <thu-muc-tam>/custom_root` → **exit 0**, NPZ tồn tại tại thư mục đã chỉ định. Chạy nguyên `replay_argv` trong metadata, chỉ thay interpreter/split/tên output → **exit 1**, `FileNotFoundError` vì tìm `data/interim/det_cache/review_custom/meta.json`.

Lỗi có ý nghĩa khi chuyển cache sang ổ lưu trữ khác trên Nano. Các phép parity mặc định hiện tại không đi qua nhánh này.

**Sửa:** lệnh replay phải giữ vị trí cache thực tế, qua đường dẫn cache tuyệt đối hoặc cơ chế root tương thích với runner. Regression phải chạy nguyên lệnh đã sinh, không tự sửa đường dẫn trong helper test; thêm đường dẫn có dấu cách, rồi kiểm parity toàn sequence.

## 4. N21-F3 — P2: YAML không tồn tại vẫn khởi động detector và thoát ngoài hợp đồng lỗi

Vị trí: `src/nano_live.py` kiểm đầu vào dòng 460–484, spawn dòng 525, nạp YAML dòng 549; thiếu `try/finally` bao vòng đời tài nguyên sau spawn.

**Tái hiện:** dùng ảnh/cache mô phỏng hợp lệ và `--tracker-yaml <duong-dan-khong-ton-tai>.yaml`.

- Tiến trình detector đã được khởi động (`detector.pid` có mặt).
- Cha ném `FileNotFoundError` ra ngoài, **exit 1**, không có `run_meta.json`.
- Không đi qua `con.ket_thuc()` của ứng dụng.

Đây là lỗi đầu vào có thể phát hiện trước khi nạp engine, nhưng chưa trả mã 2 theo hợp đồng đã công bố. Reviewer không khẳng định tiến trình con tồn tại mãi sau khi cha thoát; vấn đề đã chứng minh là ứng dụng bỏ qua đường xử lý lỗi và dọn tài nguyên có kiểm soát.

**Sửa:** kiểm đường dẫn/YAML/override và các tham số có thể kiểm trước spawn. Bao phần sau spawn trong `try/finally` để luôn dừng/reap con và telemetry. Thêm regression YAML thiếu/hỏng, kiểm mã thoát 2 và không khởi động detector đối với lỗi đã phát hiện được trước. Với lỗi phát sinh trong vòng lặp, giữ đúng hợp đồng lưu phần đã xử lý.

## 5. Cách đọc bằng chứng Nano và phạm vi nghiệm thu

Reviewer đọc lại sáu CSV pipeline và timing, đủ **300 frame/lượt**, 275 frame sau warm-up, thứ tự đúng; tính lại median latency/update và `fps_on_dinh` đều khớp metadata. Cửa sổ credit ghi nhận không vượt K=4. Output MOT của **6/6 lượt benchmark** giống output live của lượt parity cùng cấu hình.

Log và JSON trên Nano ghi **10/10 parity/ca biên đạt**, gồm live↔replay CV/EKF. Tuy nhiên gói không kèm MOT của nhánh replay và không có 4 cache NPZ, nên reviewer **chưa tự chạy lại replay trên detections TensorRT gốc hoặc so trực tiếp hai file live/replay đó**. Điều đã tự chạy là live↔replay với fixture; điều từ Nano là bằng chứng thực thi do bên thực hiện bàn giao. Không trộn hai mức bằng chứng.

Số khoảng 9,3 FPS và latency khoảng 115 ms không mâu thuẫn: hai tiến trình có thể xử lý chồng lấp. Timing ủng hộ detector là phần chi phối thông lượng trong cấu hình này. Không kết luận EKF nhanh hơn CV từ ba lượt mỗi cấu hình, cũng không suy ra FPS camera/render hoặc cảnh khác.

## 6. Đề nghị vòng sửa kế tiếp

1. Đóng ba finding trên bằng fixture tái hiện trước/sau. Giữ nguyên mã lõi tracking và các kết quả/lock nghiên cứu nếu không có bằng chứng cần đổi.
2. Chạy các test liên quan; smoke lại CV/EKF với cùng detections để xác nhận sửa điều phối không đổi tracking.
3. Sau khi sửa ranh giới timer, đo lại bộ sáu lượt ngắn trên Nano nếu thiết bị đủ điều kiện, giữ lượt vòng 21 làm lịch sử. Không cần chạy lại benchmark tracker vòng 20 hay thí nghiệm UA-DETRAC/BDD.
4. Để review parity Nano độc lập mà không gửi dataset, bổ sung đúng hai MOT replay CV/EKF tương ứng (và nhánh dừng nếu cần đối chiếu), kèm hash. NPZ có thể tiếp tục khai báo là đầu vào ngoài gói.
5. Bàn giao delta từ base `45e086a`, brief ngắn và log thật; không tuyên bố toàn bộ vòng 21 PASS trước khi đóng các finding.

Review chỉ tạo bản sao để kiểm và báo cáo; không sửa source dự án, cấu hình Git, commit hoặc thiết bị của người dùng.
