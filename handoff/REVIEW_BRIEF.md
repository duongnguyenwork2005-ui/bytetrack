# REVIEW_BRIEF — vòng 8: chốt review `b623465` + chuẩn bị đánh giá Jetson Nano

<!-- BASE_TARGET -->

Review `b623465`: **PASS WITH LIMITATIONS**, ba finding P1/P2/P3 đã đóng
(`handoff/CODEX_REVIEW_b623465.md`). Base delta chuyển sang `b623465` — ghi nhận snapshot đã đọc.
Đợt này **không** chạy lại thí nghiệm; metric, bootstrap, GT và bản khóa máy đọc giữ nguyên.
Kiểm kê Nano đầy đủ + lệnh tiếp tục: **`handoff/NANO_STATUS.md`**.

## 1. Đã chốt gì từ review

Hàng mục 6 của `CODEX_HANDOFF.md`: quy mô → **6 cấu hình/seed × 2 điều kiện = 12 run tổng cộng,
mỗi run 200 video**; SD → ghi rõ **`0,00005–0,00095` là SD HOTA** của PF giữa ba seed.

Hai chỗ mô tả bằng chứng Codex chỉ đúng, đã sửa:

- **Tracking**: "2.400 file 0 lệch" → **"0 lệch theo quy ước chuẩn hóa LF"** (brief, phụ lục,
  handoff). CRLF và LF coi là cùng nội dung, **không** phải so byte thô.
- **Phạm vi ổn định hash**: docstring `run_fingerprint.py` từng nói dấu vân tay "không đổi giữa
  Windows và Nano" — sai phạm vi, chỉ `.txt`/`.ini`/`.csv` được chuẩn hóa. Đã kiểm trên đĩa:
  `run_fingerprint.py` **281 CRLF / 0 LF**, `bytetrack_np.py` **0 CRLF / 125 LF** ⇒ trường `code`
  **sẽ lệch giữa hai máy dù cùng commit**. Đã sửa docstring và ghi thành điều kiện phải xử lý
  trước khi so provenance giữa hai máy.

## 2. Nano: KHÔNG tiếp cận được — bằng chứng, không suy đoán

`handoff/logs/review8/thu_ket_noi_nano.log`: `~/.ssh/config` và `known_hosts` **không tồn tại**;
4 tên (`nano.local`, `jetson.local`, `nano`, `tegra-ubuntu.local`) **đều không phân giải được**;
ARP **không có OUI NVIDIA `00-04-4b`**; **không** cổng serial; thiết bị PnP khớp
`NVIDIA/Jetson/Tegra/APX` chỉ có **GTX 1650 của chính PC này**.

⇒ Không thu thập được model, RAM, kiến trúc CPU, JetPack/L4T, numpy/scipy/lap, CUDA/TensorRT,
dung lượng, nguồn, clocks hay nhiệt độ. **Không số đo Nano nào được tạo ra.**
**Cần bổ sung**: IP/hostname + thiết bị bật cùng LAN; tài khoản và cách xác thực; hoặc cáp USB +
chế độ recovery.

## 3. Đã hoàn thiện để chạy ngay khi có thiết bị

- **`src/nano_parity.py`** — so tracking Nano ↔ PC, **tolerance công bố trước**: bbox `≤1e-3` px,
  conf `≤1e-6`, **liên kết ID không có tolerance**. ID kiểm hai mức (giá trị thô **và** phân hoạch
  track) để tách "đánh số lệch" khỏi "ghép khác hẳn". `test_nano_parity` **10/10**; kiểm ngược
  trên dữ liệu thật: `cv` vs chính nó → 2/2 ĐẠT, `cv` vs `ekf` → 0/2 exit 1.
- **`src/nano_bench_tracker.py`** — giao thức 4c: AB/BA **cân bằng**, mặc định **6 cặp** (≥5, chia
  đều), warm-up 50 frame/video, lấy mẫu `tegrastats`, ghi nvpmodel/nhiệt độ trước–sau **từng
  lượt**, **lưu mọi lượt**, báo **trung vị + IQR của hiệu số theo cặp**; cặp thiếu một nửa không
  dùng; tự cảnh báo khi máy không phải Jetson.

Tự kiểm trên PC (kiểm **công cụ**, không phải kết quả): 6 cặp trọn, warm-up đúng 152/202 frame.
Hiệu số `ekf − cv` trung vị `+0,28 ms` nhưng **IQR `[-0,0002; +0,483]` chứa 0** — đúng lý do giao
thức phải ghép cặp và cân bằng thứ tự.

## 4. Còn bị chặn

Mọi số đo Nano (**không tiếp cận được thiết bị**). Python ≥ 3.7 và `scipy`/`lap`/**`pandas`** cho
aarch64 chưa xác minh — `run_tracking.py` có import `pandas` nên đó là phụ thuộc bắt buộc. Đường
detector trên Nano chưa nối vào backend NumPy. Ổn định hash `.py` giữa hai máy: **hoãn có chủ ý**
— sửa `TEXT_SUFFIXES` bây giờ sẽ làm 12 `eval_meta.json` vừa review thành cũ mà chưa đổi lấy thông
tin gì. Thí nghiệm 2 chờ split train/dev. Che khuất và metric xe máy trên BDD vẫn không làm được
(nguồn thiếu `occluded`; GT gộp 5 lớp xe).

## 5. Trạng thái

Test liên quan: `test_nano_parity` **10/10**, `test_eval_guard` **11/11**, `test_report_stats`
**8/8**, `test_evaluator_choice` **7/7**, tất cả exit 0. Full suite ở `handoff/logs/review8/`.
