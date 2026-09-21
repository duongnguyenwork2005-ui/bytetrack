# Trạng thái triển khai Jetson Nano

Cập nhật 2026-09-21 (mục 1: thông tin thiết bị; 4b: gói chuyển v2). Tài liệu này chỉ ghi **kết quả lệnh thật**. Mọi ô "chưa" đều kèm lý do quan
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
| Build engine TensorRT trên Nano | ❌ chưa chạy (không có thiết bị) | — |
| Đường detector trên Nano nối vào backend NumPy | ❌ **chưa làm** | — |
| Số đo tốc độ trên Nano | ❌ **chưa có số nào**; mọi FPS trong repo là PC (GTX 1650) | — |

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
