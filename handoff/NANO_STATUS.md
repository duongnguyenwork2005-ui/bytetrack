# Trạng thái triển khai Jetson Nano

Cập nhật 2026-09-29 (mục 4k: vòng 25 — dùng tại chỗ trên Nano: launcher HDMI + hiển thị trực tiếp bbox/ID/lớp, hướng dẫn
`handoff/NANO_LOCAL_UI.md`; 4j: vòng 24 — đóng N23-F1 bộ kiểm số frame, kiểm lại 2 video render trên Nano, chốt phiên bản OpenCV của bộ kiểm; 4i: vòng 23 — chạy đủ 1.470 frame, 4 lượt, render video trên Nano; 4h: vòng 22 — sửa N21-F1/F2/F3, parity + đo lại trên Nano; 4g: vòng 21 — luồng trực tiếp video → TensorRT → tracker chạy thật, parity, FPS toàn luồng; 4f: vòng 18–20 — Mốc 1 v5, sự cố thẻ SD, dựng lại, benchmark CV↔EKF, **đóng benchmark**: sao lưu + tính lại từ dữ liệu thô; 4e: vòng 17 — preflight; 4d: vòng 16). Tài liệu này chỉ ghi **kết quả lệnh thật**. Mọi ô "chưa" đều kèm lý do quan
sát được, không suy ra từ việc thiếu file cấu hình.

## 1. Thiết bị

**2026-09-21 (vòng 14)** — người dùng cung cấp: Jetson Nano Dev Kit 4 GB, aarch64, Ubuntu 18.04.6 / L4T R32.7.5, Python hệ thống
**3.6.9**, numpy 1.19.5, scipy 0.19.1, pandas 0.22.0, **chưa có `lap`**, torch 1.10.0 (`cuda.is_available()=True`), ~26 GB trống;
SSH `jetson@10.11.53.239` đã đăng nhập được từ Windows (mật khẩu). `nvcc`/`trtexec` không thấy trong PATH — **chưa đủ** để kết luận
CUDA/TensorRT chưa cài (kiểm `dpkg -l | grep -i tensorrt`, `/usr/src/tensorrt/bin/trtexec`, `/usr/local/cuda*/bin/nvcc`).
Thử từ phiên Claude (`handoff/logs/review14/thu_ket_noi_nano.log`): ping unreachable, SSH BatchMode timeout, không có khóa SSH ⇒
**không tới được lúc đó**; chưa kiểm được gì trên máy. Yêu cầu của mã: **Python ≥ 3.8** (vermin) ⇒ cần môi trường riêng
(python3.8 apt + venv hoặc miniforge), giữ nguyên python3 hệ thống/torch — xem `scripts/nano/NANO_START_HERE.md` mục 1.

### 1a. Lần thử trước (2026-09-20): KHÔNG tiếp cận được

Bằng chứng: `handoff/logs/review8/thu_ket_noi_nano.log` (chạy 2026-09-20T08:02:47+07:00).

| Kiểm tra | Kết quả |
|:--|:--|
| `~/.ssh/config` | **không tồn tại** |
| `~/.ssh/known_hosts` | **không tồn tại** |
| Biến môi trường khớp `nano`/`jetson` | không có |
| `Resolve-DnsName nano.local` / `jetson.local` / `nano` / `tegra-ubuntu.local` | **cả 4 đều không phân giải được** |
| Bảng ARP, OUI NVIDIA `00-04-4b` | **không có mục nào** |
| Cổng serial (`Win32_SerialPort`) | không có |
| Thiết bị PnP khớp `NVIDIA/Jetson/Tegra/APX` | chỉ có **GTX 1650 của chính PC này** + audio của nó; **không** có Jetson/Tegra/APX |

⇒ Không có đường nào tới thiết bị từ máy này: không tên nào phân giải được, và không thiết bị
NVIDIA nào khác GPU của PC xuất hiện trong ARP/USB/serial. **Không thu thập được** model thiết bị,
RAM, kiến trúc CPU, JetPack/L4T, phiên bản numpy/scipy/lap, CUDA/TensorRT, dung lượng trống,
chế độ nguồn, clocks hay nhiệt độ.

**Cần người dùng cung cấp để tiếp tục**:

1. IP hoặc hostname của Nano, thiết bị đang bật và **cùng mạng LAN** với PC này;
2. tài khoản đăng nhập và cách xác thực (khóa SSH hoặc mật khẩu);
3. hoặc cáp USB + Nano ở chế độ recovery (APX) nếu muốn đi đường serial/flash.

## 2. Đã chạy thật trên PC / mới chỉ chuẩn bị

| Hạng mục | Trạng thái | Bằng chứng |
|:--|:--|:--|
| Backend ByteTrack không phụ thuộc torch | ✅ **chạy thật trên PC** | `test_bytetrack_np_parity` 10/10, `test_runner_backend_np` 14/14 — CLI và `stage_track` chạy exit 0 trong tiến trình **chặn** `torch`/`ultralytics` |
| Bản vendor 5 module tracker | ✅ có, AGPL-3.0, `PROVENANCE.json` ghi sha256 | `third_party/ultralytics_trackers/` |
| Export ONNX (opset 12, static) | ✅ trên PC | `models/export/*.onnx`, `results/phase2/export/*.json` |
| Đối chiếu op với bảng TensorRT 8.2 | ✅ **trên giấy** — `yolov8n`, `yolo11n` qua; `yolo26n` bị loại (op `Mod`) | `scripts/nano/trt82_ops.txt` |
| Script kiểm stack / build engine / benchmark engine | ✅ đã viết, **chưa chạy trên thiết bị** | `scripts/nano/0{0,1,2}_*.sh` |
| Giao thức đo CV ↔ EKF | ✅ chốt trước | `scripts/nano/README.md` mục 4c |
| **Bộ so kết quả Nano ↔ PC** | ✅ **mới, đã kiểm** | `src/nano_parity.py`, `src/test_nano_parity.py` **10/10** |
| **Khung đo AB/BA** | ✅ **mới, đã tự kiểm trên PC** | `src/nano_bench_tracker.py` |
| Build engine TensorRT trên Nano | ✅ **đã build** (485 s, FP16, `trtexec` 68,1 ms) | `handoff/REVIEW16_CHI_TIET.md` mục 3 |
| Đường detector trên Nano nối vào backend NumPy | ✅ **đã chạy TRT thật** (Py 3.6 → det cache → tracker Py 3.8) | `handoff/REVIEW16_CHI_TIET.md` mục 4 |
| Số đo tốc độ trên Nano | ✅ **có** (vòng 16, một lượt): detector 107,6 ms/frame, infer 73,3 ms; tracker replay ~148–244 fps | `handoff/REVIEW16_CHI_TIET.md` mục 4 |

## 3. Hai công cụ mới

### `src/nano_parity.py` — kiểm đúng đắn trước khi đo

So kết quả tracking của Nano với bản tham chiếu PC. **Tolerance công bố trước khi so**, in ra đầu
mỗi lần chạy và lưu vào JSON:

| Đại lượng | Tolerance | Lý do |
|:--|:--|:--|
| bbox (x, y, w, h) | `\|Δ\| ≤ 1e-3` pixel | x86-64 và aarch64 có thể dùng BLAS/libm khác ⇒ lệch chữ số cuối |
| confidence | `\|Δ\| ≤ 1e-6` | như trên |
| **liên kết ID** | **không có tolerance** | association là quyết định **rời rạc**; một khác biệt nghĩa là tracker đã ghép khác, không phải sai số |

Liên kết ID kiểm **hai mức**: (a) giá trị ID thô phải bằng nhau; (b) **phân hoạch** track phải
tương đương. Tách hai mức để phân biệt "chỉ đánh số lệch" với "ghép track khác hẳn" — hai loại lỗi
rất khác nhau. Khóa sắp xếp cố ý **không chứa `id`**, nếu không phép kiểm (b) sẽ mất nghĩa.

```bash
python src/nano_parity.py --ref <thư mục PC> --test <thư mục Nano> --json ket_qua.json
```

Đã kiểm ngược trên dữ liệu thật: so `c25-cv` với chính nó → 2/2 DAT; so `c25-cv` với `c25-ekf` →
0/2, exit 1.

### `src/nano_bench_tracker.py` — đo theo giao thức 4c

Thực thi đúng giao thức đã chốt: lịch **AB/BA cân bằng**, mặc định **6 cặp** (3 mỗi chiều, thỏa
yêu cầu ≥ 5 và chia đều), warm-up bỏ 50 frame đầu **mỗi video**, lấy mẫu `tegrastats` song song,
ghi `nvpmodel`/governor/nhiệt độ **trước và sau từng lượt**, **lưu mọi lượt** (không loại lượt nào
khỏi dữ liệu thô), và báo **trung vị + IQR của hiệu số theo cặp**, không phải hiệu hai trung bình.
Cặp thiếu một nửa **không** được dùng.

Công cụ tự phát hiện máy không phải Jetson (không có `/etc/nv_tegra_release`) và in cảnh báo rằng
kết quả **không được báo cáo là số Nano**.

**Tự kiểm trên PC** (chỉ để xác minh công cụ, **không phải kết quả nghiên cứu**): 6 cặp chạy trọn,
warm-up áp đúng 152/202 frame. Hiệu số `ekf − cv` có trung vị `+0,28 ms` nhưng **IQR `[-0,0002;
+0,483]` chứa 0** và min/max `−0,396 / +0,500`. Điều này cho thấy đúng lý do giao thức cần ghép
cặp và cân bằng thứ tự: trên một máy đang chạy việc khác, nhiễu giữa các lượt **ngang với hiệu ứng
cần đo**. Con số này **không** được dùng làm kết luận về CV/EKF.

## 4. Điều kiện còn thiếu (ngoài thiết bị)

| Việc | Trạng thái | Ghi chú |
|:--|:--|:--|
| Python ≥ 3.7 trên Nano | ❓ chưa xác minh | Bản vendor dùng `from __future__ import annotations`; JetPack 4.6 mặc định **Python 3.6** ⇒ cần miniforge aarch64 hoặc tương đương |
| `numpy` / `scipy` / `lap` cho aarch64 | ❓ chưa xác minh | `lap` phải build được; PC dùng 2.4.6 / 1.17.1 / 0.5.13 |
| `pandas` cho aarch64 | ❓ chưa xác minh | `src/run_tracking.py` **có import `pandas`** để ghi `timing/*.csv` ⇒ đây là phụ thuộc bắt buộc của đường tracker-từ-cache, phải kiểm cùng scipy/lap |
| `pyyaml` cho aarch64 | ❓ chưa xác minh | `src/ul_backend.py` nạp YAML tracker bằng `yaml.safe_load` ⇒ bắt buộc cho backend NumPy (đã kiểm: gói chuyển chạy replay trong tiến trình **chặn** torch/ultralytics chỉ cần numpy/scipy/lap/pandas/pyyaml) |
| **Phạm vi ổn định hash giữa máy** | ⚠️ **đã biết, chưa xử lý** | `run_fingerprint.TEXT_SUFFIXES` chỉ chuẩn hóa `.txt`/`.ini`/`.csv`. Module `.py` hash theo **byte thô**, mà cây nguồn xuống dòng **lẫn lộn** — đã kiểm: `src/run_fingerprint.py` **281 CRLF / 0 LF**, `src/bytetrack_np.py` **0 CRLF / 125 LF**. ⇒ trường `code` **sẽ lệch giữa Windows và Nano dù cùng commit** |

### Về phạm vi ổn định hash

Đây **không** phải lỗi tính toán — metric không bị ảnh hưởng. Hệ quả thực tế: khi chạy trên Nano,
mọi evaluation sẽ bị coi là "code đã đổi" và chạy lại, và provenance giữa hai máy **không so trực
tiếp được**.

Cách xử lý khi bắt đầu chạy trên thiết bị: thêm `.py` (và `.json`/`.yaml` nếu cần) vào
`TEXT_SUFFIXES`, **hoặc** ép `.py` về LF trong `.gitattributes`. **Chưa làm trong đợt này** vì đổi
`TEXT_SUFFIXES` sẽ làm 12 `eval_meta.json` vừa được review (`b623465`) thành cũ, trong khi chưa
chạy gì trên thiết bị — đổi lúc đó không mang lại thông tin nào.

## 4k. Vòng 25 (2026-09-29): dùng tại chỗ bằng HDMI + chuột + bàn phím — chi tiết `handoff/REVIEW25_CHI_TIET.md`

- Gói **v14** `315407c5…` ở `~/nano_run/v14` (đang dùng; v12/v13 là bản thử trong vòng). Biểu tượng desktop + menu “Theo vết phương
  tiện (Nano)” trỏ tới `~/nano_run/v14/nano_transfer/scripts/nano/07_giao_dien.sh`; hướng dẫn `handoff/NANO_LOCAL_UI.md`.
- Launcher Tk (env `nano`) điều phối `nano_live.py`; cửa sổ xem trực tiếp do tiến trình detector (Python 3.6, OpenCV GTK) vẽ, ghép theo
  `frame_idx` qua lệnh `R` (giao thức 3). Dừng: nút, `q`/`Esc`, đóng cửa sổ xem, đóng launcher → dừng có kiểm soát (mã 4).
- Trên thiết bị: test 17/17 + 20/20; TensorRT bật/tắt hiển thị MOT giống byte; 1.470 ảnh CV/EKF có hiển thị trên HDMI: hiện 1470/1470,
  bỏ 0, MOT giống byte vòng 23; FPS xử lý giảm ≈ 2 % khi bật hiển thị (300 ảnh, ABBA); phím q, đóng cửa sổ, mở từ biểu tượng đều đúng.
- Màn hình Nano tự khóa sau 5 phút rảnh (Unity); launcher chặn khóa khi rảnh chỉ trong lúc chạy. Camera: chưa.

## 4j. Vòng 24 (2026-09-28): đóng N23-F1, kiểm lại hai video render trên Nano — chi tiết `handoff/REVIEW24_CHI_TIET.md`

- `src/kiem_video_render.py` sửa: số frame hiển thị kiểm **độc lập trên mọi frame** (tỷ lệ lỗi trên điểm chữ, ngưỡng cũ + nhận dạng
  ứng viên cách một chữ số); chốt **OpenCV render (`<video>.json`) == OpenCV kiểm**, khác → exit 2 `KHONG KIEM DUOC`. Script Codex
  exit 0; `test_render_mot_video` 13/13.
- Gói **v10** `a3bb5841…` (`9b1805c`) và **v11** `6f80806d…` (mã cuối) ở `~/nano_run/v10`, `~/nano_run/v11`; so với v9 chỉ khác 3 file
  render/kiểm. Không chạy lại tracking, không build engine, không đổi môi trường hệ thống.
- Hai video vòng 23 (hash video/MOT/lớp khớp), `/usr/bin/python3` 3.6.9 + OpenCV 3.2.0: CV và EKF đều 300 frame, 25 fps, 960×540;
  **số frame 300/300 đúng**; bbox/ID 0 lỗi (1.272 / 1.270 box); tên lớp mẫu mỗi 5 frame 255/255; exit 0. Bổ sung kiểm mọi box:
  1.271/1.272 và 1.269/1.270 — frame 90 ID 30 (`car`) không phân định được do nén mp4v. Render lại từ cùng MOT/lớp **byte-giống** →
  giữ nguyên video.
- Bộ kiểm vẽ lại chữ nên chỉ hợp lệ với cùng OpenCV đã render: OpenCV 5.0 trên PC vẽ `putText` khác 3.2 (khung chữ nhật trùng) →
  chạy trên PC nay trả exit 2.

## 4i. Vòng 23 (2026-09-28): đầy đủ 1.470 frame + render trên Nano — chi tiết `handoff/REVIEW23_CHI_TIET.md`

- 1.470 JPG gốc `MVI_39031` ở `~/nano_run/data_full/MVI_39031` (SHA256 1470/1470 khớp PC). Mã chạy gói **v8** (== `c4b39fe`), engine
  `~/nano_run/engine/yolov8n_fp16.engine` (`c32690cd…`). Driver `handoff/logs/review23/chay_dai_1470.sh` (chạy từ gốc v8).
- 4 lượt CV → EKF → EKF → CV: exit 0, hoàn tất, frame 1–1470 đúng một lần, MOT live == replay từng byte, parity ĐẠT, con thoát 0,
  không thấy tích lũy bộ nhớ giữa các lượt (MemAvailable +3 MB; chưa loại trừ mọi rò rỉ), không khởi động lại. Phạm vi: bốn lượt,
  mỗi lượt khoảng 158–163 giây xử lý, trong phiên khoảng 13 phút 33 giây, có nghỉ và replay giữa lượt; nhiệt CPU tối đa 38 °C, GPU
  tối đa 35,5 °C (diễn đạt đính chính ở vòng 24). FPS toàn luồng CV 9,22 (9,14–9,31), EKF 9,05 (9,04–9,05); decode hai mức
  (~11,5/16,3 ms) rơi vào hai lượt EKF → không xếp hạng.
- Render **hậu kỳ** trên Nano: `/usr/bin/python3 src/render_mot_video.py` (gói **v9** `ea1fb21f…`, `~/nano_run/v9`) + kiểm
  `src/kiem_video_render.py` ĐẠT (300/300 frame, 25 fps, bbox/ID/lớp đúng) — *vòng 24: bộ kiểm này chưa kiểm số frame trên frame
  có box (N23-F1); kiểm lại ở 4j*. Hiển thị trực tiếp khi tracking và camera: chưa làm
  (màn hình HDMI đã kết nối).

## 4h. Vòng 22 (2026-09-28): sửa N21-F1/F2/F3 (review `9f210e2`) — chi tiết `handoff/REVIEW22_CHI_TIET.md`

- **Giao thức 2**: con gửi HELLO rồi **chờ lệnh BẮT ĐẦU**; cha chuẩn bị xong, bật timer rồi mới gửi → mọi frame được đếm nằm trong cửa
  sổ đo. Thời gian mới: `khoi_tao_s`, `chuan_bi_cha_s`, `xu_ly_s`/`fps_xu_ly`, `ket_thuc_s`, `ghi_s` (thay `wall_vong_lap_s`/`fps_vong_lap`).
- Replay cache ở `--cache-root` riêng: `src/nano_replay.py` (lệnh trong `replay_argv`). Cấu hình sai (YAML, `--set`, tham số) → mã 2,
  không khởi động detector; lỗi sau spawn luôn dừng/reap con.
- Gói **v8** `f4da9f08…` ở `~/nano_run/v8`, engine dùng lại `~/nano_run/engine/yolov8n_fp16.engine`. Test thiết bị 20/20, 9/9, 20/20;
  parity 10/10 (TRT live == replay byte, MOT replay bàn giao). Đo lại: **CV 9,13 (9,07–9,19), EKF 9,36 (9,20–9,42) FPS**; decode có hai
  mức ~11,3/16,3 ms làm lệch giữa các lượt → không xếp hạng CV/EKF. Số vòng 21 giữ, có đính chính (lệch 0,21–0,28 %).

## 4g. Vòng 21 (2026-09-28): luồng TRỰC TIẾP video → TensorRT → tracker chạy thật trên Nano — chi tiết `handoff/REVIEW21_CHI_TIET.md`

- Hai tiến trình: con `src/nano_detect_stream.py` (Python 3.6 hệ thống, TensorRT FP16, cudart) gửi **từng frame** qua pipe có cửa sổ
  credit K = 4 cho cha `src/nano_live.py` (env `nano` 3.8, backend np, `run_tracking.track_video` dùng chung với replay). Khởi động
  `scripts/nano/04_live_nano.sh cv|ekf <out>`; dừng `touch <out>/STOP` hoặc Ctrl+C (mã 4). Mã 0 hoàn tất, 2 lỗi trước vòng lặp,
  3 lỗi giữa luồng (ghi phần đã xử lý, không "hoàn tất").
- Gói **v7** `6dc4100d…` ở `~/nano_run/v7` (v5/v6 giữ nguyên); engine `~/nano_run/engine/yolov8n_fp16.engine` (build trên máy 483 s,
  `c32690cd…`). Gói v6 lộ hai lỗi trên thiết bị (OpenCV 3.2 **abort cả tiến trình** với BMP hỏng; test giết con trước HELLO) → sửa ở v7.
- Test thiết bị: `test_nano_parity` 20/20, `test_nano_run_cache` 9/9, `test_nano_stream` 14/14 (con 3.6 ↔ cha 3.8, kể cả SIGINT).
- Parity **10/10**: giả lập trên 300 JPG thật == `pc-ref` smoke; **TensorRT thật CV/EKF: live == replay từ cache, byte giống hệt**;
  biên (ảnh đen, mp4 300/300, stop-file, SIGINT, `kill -9` con, engine sai) đúng mã thoát. Detection TRT lượt CV == lượt EKF (bit).
- FPS toàn luồng (điều kiện công bố trước, 6 lượt, 300 JPG, không render, không `--regions`): **CV 9,29 (9,11–9,36), EKF 9,37
  (9,29–9,44)**; trễ/frame trung vị ~114–119 ms. Detector ~106 ms/frame (infer ~74) quyết định thông lượng; tracker chạy song song.
- **Mất nguồn** khoảng 11:52 (tắt không sạch, ext4 phục hồi journal, `errors_count` 0); kết quả dùng để kết luận đều tạo sau đó.
  Nguồn điện là rủi ro cho các lần đo dài.

## 4f. Vòng 18–20 (2026-09-22 → 24): Mốc 1 v5 ĐẠT; mất dữ liệu do đổi thẻ SD; dựng lại; benchmark tracker CV↔EKF từ cache

- **Vòng 18** (22/9): gói v5 (`ae5fd57b…`) → Mốc 1 **ĐẠT** trên Nano (`handoff/logs/review18/`).
- **Vòng 19** (22–23/9): benchmark lần 1 **thất bại, 0/6 cặp**. Lệnh thiếu `BYTETRACK_BACKEND=np` nên backend mặc định là ultralytics và
  không có gói ultralytics (lỗi gọi, **không** phải lỗi code); warm-up `--pairs 2` bị công cụ từ chối (tối thiểu 5 cặp). Sau đó máy
  hỏng boot vì `apt upgrade` (DTB trên eMMC). Máy boot lại bằng **thẻ 1**: bản sao cũ, trạng thái đến 21/9 16:16, fsck có nhiều lỗi.
  Toàn bộ dữ liệu 22/9 (`~/nano_run`, `~/miniforge3`) **không có trên thẻ này**; nhiều khả năng còn trên thẻ 2
  (`handoff/logs/review19/08_xac_dinh_the_sd.log`).
- **Vòng 20** (24/9, IP `10.11.56.132`, vẫn **thẻ 1**): dựng lại Miniforge user-space + env `nano`, ghim **đúng phiên bản cũ**
  (Py 3.8.20, numpy 1.24.4, pandas 2.0.3, pyyaml 6.0.2, lap 0.5.13), không sudo; gói v5 → `~/nano_run/v5`, manifest OK;
  **Mốc 1 lại ĐẠT**: 4/4 parity, `max|Δ| = 0`, 4 output byte-identical với `pc-ref` (chuẩn hóa LF).
- **Benchmark tracker-from-cache** (`handoff/logs/review20/bench_lan2/`): cache `yolov8n-coco_conf0.05_all`, `MVI_39031` 1470 frame,
  điều kiện c25 như Mốc 1; `nano_bench_tracker.py --pairs 6` (3 AB + 3 BA), bỏ 50 frame warm-up mỗi lượt, warm-up riêng 1 CV + 1 EKF
  vào split khác; MAXN, governor schedutil, nhiệt tegrastats 40–44 °C, swap 0; 12/12 lượt exit 0, không loại lượt nào.
  `update` (tổng `tracker.update()`, ms/frame): CV trung vị **4,695** (4,657–4,774), EKF **5,468** (5,414–5,512).
  Hiệu **EKF − CV theo cặp**: trung vị **+0,752 ms/frame**, IQR **[+0,710; +0,818]** (theo công cụ; theo `quantiles inclusive`:
  [+0,737; +0,794]; IQR chỉ mô tả độ phân tán của 6 hiệu số, **không** phải khoảng tin cậy), 6/6 cặp dương (+0,640 … +0,855)
  ≈ **+16 %** chi phí tracker. Phần `predict` chiếm +0,622 ms (6/6 cặp),
  `assoc`/cập nhật +0,139 ms. Parity đại diện `bench-{cv,ekf}-r{1,6}` so với `NANO-full/pc-ref` ĐẠT lệch 0; `pc-ref-*` và `nano-*` của
  Mốc 1 không đổi. **Phạm vi**: chỉ tracker từ cache, không phải end-to-end; công cụ không lưu tần số CPU theo lượt; lúc đo có
  `update-manager`/`packagekitd`/`gnome-software` chạy nền.
- **Chưa dựng lại**: thư mục v3/v4 (bằng chứng đã có trên PC), engine TensorRT và Mốc 2 (bằng chứng vòng 16 nằm trên PC).
- **Đóng benchmark** (24/9, chỉ đọc trên Nano, không đo mới; chi tiết `handoff/REVIEW20_CHI_TIET.md`):
  - **Sao lưu** về `handoff/logs/review20/sao_luu_nano/`: 95 file gói v5 (Mốc 1, 12 lượt bench + warm-up, `pc-ref`, manifest)
    và 3 log dựng env, kèm danh sách env. SHA256 và số byte khớp nguồn 95/95 và 3/3 (`08_kiem_sao_luu.log`).
  - **Tính lại từ timing CSV thô** (`09_tinh_lai_tu_raw.log`, script độc lập `tinh_lai_bench.py`): 1470 → 1420 frame/lượt, ghép
    đúng 6 cặp AB/BA, mọi số khớp JSON của công cụ (lệch 0). Output `data/` của **cả 12 lượt** + warm-up + Mốc 1 cùng hash với
    `pc-ref` (LF) theo cấu hình (`09a_hash_output_12_luot.log`).
  - **Thẻ** (thẻ 1, `mmcblk1p1` rw, 27G/58G, ext4 `errors_count` 0, dmesg không có lỗi I/O). Sau benchmark Nano khởi động lại
    **hai lần**: 10:13:40 tắt sạch; ≈10:36 tắt **không sạch**, ext4 phục hồi journal (18 orphan inode), vẫn `errors_count` 0.
    Nguyên nhân không có trong log (journal không lưu lâu dài). Băm lại nguồn sau đó: 95/95 và 3/3 không đổi (`08b_*.log`).
    Ngày 24/9 không có thay đổi gói nào (dpkg/apt 0).
  - Không có log tegrastats thô (công cụ chỉ lưu tóm tắt nhiệt).

## 4e. Vòng 17 (2026-09-22): preflight theo review `20207ec` — gói v5 `ae5fd57b…` (chưa chuyển sang Nano)

`src/nano_preflight.py` là hợp đồng đầu vào **dùng chung** cho `nano_env_check.py` (Python 3.6 chạy được) và `nano_run_cache.py`:
đúng 2 cache (meta/NPZ/sidecar parse + schema + timeline), đủ 4 `pc-ref`, YAML, mã; env_check nonzero hoặc thiếu/hỏng → `exit 2`
trước khi tracker chạy, không tạo `nano-*`. `test_nano_run_cache` 9/9, full suite 33/33, verify gói v5 hai stack ĐẠT với 4 negative
case. Chi tiết `handoff/REVIEW17_CHI_TIET.md`.

## 4d. Vòng 16 (2026-09-22): **ĐÃ CHẠY THẬT TRÊN NANO** — chi tiết `handoff/REVIEW16_CHI_TIET.md`

| Hạng mục | Kết quả thật |
|:--|:--|
| Thiết bị | Jetson Nano 4 GB, JetPack 4.6.5 / L4T 32.7.5, TensorRT 8.2.1.9, CUDA 10.2, Py 3.6.9 hệ thống; SSH khóa riêng `10.11.59.212` |
| Môi trường tracker | Miniforge user-space → env `nano` Py 3.8.20, numpy 1.24.4, pandas 2.0.3, pyyaml 6.0.2, lap 0.5.13 (Python 3.6/torch hệ thống nguyên vẹn) |
| Mốc 1 tracker-from-cache | **ĐẠT**: smoke 545/545 ×2, full 5842/5845 parity ĐẠT lệch 0; `pc-ref` bất biến; byte-identical sau chuẩn hóa LF |
| Engine TensorRT | build trên Nano 485 s; `trtexec` GPU compute 68,1 ms mean (14,7 qps) |
| Detector TRT + tracker (mốc 2) | chạy trọn (gói v4 `3a592ea0…`, sửa `0423616`: cudart + LD_PRELOAD): 107,6 ms/frame detector (decode 15,9 / pre 13,1 / infer 73,3 / post 5,2), det ghép 579/581 IoU 0,988 vs cache PC; tracker CV 548 / EKF 547 hàng |
| Số đo tốc độ Nano | **có** (một lượt, MAXN, chưa `jetson_clocks`, phạm vi ghi rõ); chưa tối ưu |

## 4c. Vòng 15 (2026-09-22): script thiết bị + detector TensorRT — gói v3 `0b190b3`

Thiết bị vẫn **không tới được** (`handoff/logs/review15/thu_ket_noi_nano.log`, `quet_mang.log`: ping/SSH timeout; host `.179` duy nhất mở SSH
có host key khác Nano). **Chưa chạy gì trên Nano.** Đã làm trên PC: `src/nano_run_cache.py` (chỉ ghi `nano-*`, kiểm đầu vào, hash `pc-ref`
trước/sau — hai finding review `7a2feb6`); `src/nano_detector.py` (Python 3.6 hệ thống: TensorRT + tiền/hậu xử lý NumPy → det cache đúng
định dạng PC; tracker ở venv ≥ 3.8 replay — hai tiến trình vì binding TRT chỉ có cho Py 3.6); `test_nano_detector` 6/6 (letterbox pixel-identical,
NMS lệch 0 với ultralytics trên cùng đầu ra thô); `nano_det_compare.py`, `render_mot_video.py`, `scripts/nano/03_e2e_nano.sh`; baseline e2e PC
(torch FP32 + letterbox 640×640: det ghép 99,7 %, IoU 0,996; tracking không parity với `pc-ref` — dự kiến). Gói v3 `nano_transfer_latest.zip`
63.792.494 B sha256 `4dc0530a…`, verify 2 stack ĐẠT. Chi tiết `handoff/REVIEW15_CHI_TIET.md`; khối lệnh mục 5.

## 4b. Gói chuyển sang Nano (2026-09-21, v2 `10d26fa`)

`handoff/nano_transfer_latest.zip` 63.764.728 B sha256 `02518777…`: thêm so với v1 — cache **cắt đúng đoạn** frame 1–300
(`nano_smoke_MVI_39031_f0001-0300`, giữ frame rỗng/`time_s`, `meta.json` nguyên văn) + tham chiếu `NANO-smoke`, cache gốc + tham chiếu
`NANO-full`, `models/yolov8n.pt`, `requirements_nano.txt` (numpy/lap/pandas/pyyaml), `src/nano_env_check.py`; kiểm sạch trên 2 stack
(PC; numpy 1.24/lap 0.5.12/không scipy) đều parity ĐẠT + bit-identical. Chi tiết `handoff/REVIEW14_CHI_TIET.md`.

### 4b-cũ. Gói v1 (2026-09-21 sáng)

`handoff/nano_transfer.zip` (dựng bằng `src/make_nano_transfer.py build`, kiểm bằng `verify`): mã + cấu hình + vendor + `scripts/nano/`, ONNX `yolov8n_640_op12.onnx` (đối chiếu trọng số với bản re-export từ `models/yolov8n.pt`: 127/127 initializer, 262/262 node giống hệt, chỉ khác metadata `date`), det cache `MVI_39031` (1470 frame), tham chiếu PC `pc-ref-{cv,ekf}` từ chính cache đó (backend NumPy; y hệt backend ultralytics), đoạn smoke frame 1–300 (JPG y nguyên + mp4 lossy), `NANO_START_HERE.md`, manifest SHA256. Kiểm gói trong thư mục sạch trên PC (`handoff/logs/nano_transfer/kiem_goi_sach.log`): manifest 516/516, import backend np chặn torch/ultralytics, `test_nano_parity` 20/20, replay CV/EKF + parity ĐẠT lệch 0 — **chỉ xác minh gói đủ**, không chứng minh tương thích Nano. Phần **chưa có mã**: suy luận TensorRT + NMS NumPy + adapter nối vào `run_tracking.track_video` (`NANO_START_HERE.md` mục 5).

## 5. Lệnh tiếp tục khi có thiết bị

```bash
# --- TRÊN NANO: thu thập trạng thái TRƯỚC mọi phép đo ---
bash scripts/nano/00_check_stack.sh            # JetPack/L4T, CUDA, TensorRT, RAM, nvpmodel
uname -m && cat /etc/nv_tegra_release
python3 -VV && python3 -c "import sys; print(sys.version_info >= (3,7))"
python3 -c "import numpy, scipy, lap, pandas; print(numpy.__version__, scipy.__version__, lap.__version__, pandas.__version__)"
free -h && swapon --show && nvpmodel -q
cat /sys/devices/virtual/thermal/thermal_zone*/temp

# --- Backend NumPy phải import được mà KHÔNG kéo torch/Ultralytics ---
BYTETRACK_BACKEND=np python3 -c "import sys, ul_backend; print(ul_backend.fingerprint()); \
  print('torch da nap?', 'torch' in sys.modules, '| ultralytics da nap?', 'ultralytics' in sys.modules)"

# --- ĐÚNG ĐẮN trước, TỐC ĐỘ sau ---
BYTETRACK_BACKEND=np python3 src/run_tracking.py \
    --cache-id yolov8n-coco_conf0.05_all_bddcocovid --split-name BDDCOCOVID-val \
    --tracker-yaml bytetrack.yaml --tracker-name nano-cv \
    --videos b1c66a42-6f7d68ca --conf-min 0.25 --classes 1 2 3 5 7 \
    --fps 5 --track-buffer-sec 1.2 --elapsed-mode steps

python3 src/nano_parity.py \
    --ref  data/processed/trackers/BDDCOCOVID-val/p2-bdd_transfer_cocovid_val-c25-cv \
    --test data/processed/trackers/BDDCOCOVID-val/nano-cv \
    --videos b1c66a42-6f7d68ca --json results/phase2/profile/nano_parity_cv.json
# exit != 0  =>  DỪNG. Không đo tốc độ khi kết quả tracking còn khác.

# --- Chỉ chạy khi parity ĐẠT cho cả CV và EKF ---
BYTETRACK_BACKEND=np python3 src/nano_bench_tracker.py \
    --cache-id yolov8n-coco_conf0.05_all_bddcocovid --split-name BDDCOCOVID-val \
    --videos b1c66a42-6f7d68ca b1c81faa-3df17267 b1c81faa-c80764c5 \
    --a cv:bytetrack.yaml --b ekf:configs/bytetrack_ekf_ctrv.yaml \
    --conf-min 0.25 --classes 1 2 3 5 7 --fps 5 --track-buffer-sec 1.2 \
    --pairs 6 --warmup-frames 50 --nghi-giay 30 \
    --out results/phase2/profile/nano_bench
```

### Đầu vào phải chép sang Nano

| Thứ | Vì sao |
|:--|:--|
| `src/`, `configs/`, `third_party/` | mã và cấu hình đã khóa |
| `data/interim/det_cache/<cache_id>/{meta.json, <video>.npz, <video>.json}` cho các sequence đã chọn | **cùng detection** với PC — đây là điều kiện để hiệu số có nghĩa |
| `data/processed/BDDCOCOVID-val/<video>/{gt/gt.txt, seqinfo.ini}` | chỉ cần nếu chạy đánh giá trên Nano; **không cần** cho riêng phép đo tracker |
| Bản tham chiếu PC: `data/processed/trackers/BDDCOCOVID-val/p2-bdd_transfer_cocovid_val-c25-{cv,ekf}/data/*.txt` | để `nano_parity.py` so |

**Không** cần chép ảnh JPG, weights hay toàn bộ cache 200 video — phép đo tracker chỉ cần cache của
các sequence được chọn.
