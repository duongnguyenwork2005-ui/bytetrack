# NANO_START_HERE — gói chuyển sang Jetson Nano (mốc 1: tracker từ cùng cache; mốc 2: detector TensorRT + tracker)

Gói này **độc lập** với repository PC và với mọi gói review: giải nén ở bất kỳ đâu, `cd` vào thư mục gói
(`nano_transfer/`) và chạy các lệnh dưới đây theo **đúng thứ tự**, **đường dẫn tương đối từ thư mục gói**.
Mỗi bước phải đạt mới sang bước sau. Nguồn gói (commit code / commit đóng gói, phiên bản môi trường PC, hash
đầu vào): `PACKAGE_INFO.json`; SHA256 **từng file**: `MANIFEST_SHA256.txt`.

> Gói **chưa được chạy trên Nano**. Kiểm trên PC (repo PC: `handoff/logs/nano_transfer/kiem_goi_sach*.log`) chỉ
> xác nhận gói **đủ file và đủ phụ thuộc Python** để chạy đường replay/parity trong một thư mục sạch; nó **không**
> chứng minh tương thích Nano (aarch64/JetPack). Không giả định phiên bản gì — bước 1 đo thật bằng
> `src/nano_env_check.py`.

## 0. Cấu trúc gói

| Thư mục / file | Nội dung | Bước |
|:--|:--|:--|
| `NANO_START_HERE.md`, `PACKAGE_INFO.json`, `MANIFEST_SHA256.txt`, `requirements_nano.txt` | tài liệu; nguồn/hash; phụ thuộc **tối thiểu** (từ chuỗi import thật, không phải requirements PC) | 0–1 |
| `src/` (không `src/demo/**`), `src/mot_datasets/` | mã tracker, backend NumPy (`ul_backend.py`, `bytetrack_np.py`, `tracker_ctrv.py`, `ekf_ctrv.py`…), `run_tracking.py` (replay cache), **`nano_run_cache.py`** (script chạy trên thiết bị: preflight, hash tham chiếu, chỉ ghi `nano-*`), **`nano_preflight.py`** (hợp đồng đầu vào **dùng chung**, cú pháp 3.6), `nano_env_check.py`, `nano_parity.py` + `test_nano_parity.py`, `test_nano_run_cache.py`, **`nano_detector.py`** (Python 3.6: TensorRT + tiền/hậu xử lý NumPy → det cache), `nano_det_compare.py`, `render_mot_video.py` | 1–6 |
| `configs/bytetrack_ekf_ctrv.yaml`, `bytetrack_ukf_ctrv.yaml` (+ YAML thí nghiệm, không cần) | cấu hình bộ lọc đã khóa; CV dùng `bytetrack.yaml` của bản vendor | 3 |
| `third_party/ultralytics_trackers/` | bản **vendor** vòng lặp ByteTrack ultralytics 8.4.141 (**AGPL-3.0**, `LICENSE`, `PROVENANCE.json` sha256) — không cần torch/ultralytics | 2–3 |
| `scripts/nano/` | `00_check_stack.sh`, `01_build_engine.sh`, `02_benchmark_engine.sh`, **`03_e2e_nano.sh`** (chuỗi mốc 2), `trt82_ops.txt`, `README.md` | 1, 6 |
| `data/interim/det_cache/nano_smoke_MVI_39031_f0001-0300/` | **cache đúng đoạn smoke** (frame 1–300, cắt *tiền tố* từ cache PC; **giữ frame rỗng** `n_per_frame=0` và `time_s`; `meta.json` copy nguyên văn = cùng DetectorSpec) | 3–4 |
| `data/interim/det_cache/yolov8n-coco_conf0.05_all/` | cache PC **toàn sequence** `MVI_39031` (1470 frame) — parity mở rộng | 3–4 |
| `data/processed/trackers/NANO-smoke/pc-ref-{cv,ekf}/` | **tham chiếu PC** từ cache đoạn smoke (backend NumPy, torch/ultralytics **bị chặn** khi tạo): `data/MVI_39031.txt`, `data_cls/`, `timing/`, `run_meta.json` (cấu hình hiệu lực, fingerprint) | 4 |
| `data/processed/trackers/NANO-full/pc-ref-{cv,ekf}/` | tham chiếu PC từ cache toàn sequence | 4 |
| `reference/` | `pc_ref_sha256.txt` (hash **bất biến** của 4 thư mục `pc-ref-*`, kiểm trước/sau khi chạy), `lenh_tai_hien.sh` (**chỉ để tái tạo tham chiếu trên PC**, không chạy trên Nano), `parity_pc_np_vs_ul_*.json` | 3–4 |
| `data/smoke/MVI_39031_f0001-0300/` (300 `.jpg`), `MVI_39031_f0001-0300.mp4`, `MVI_39031_smoke.json`, `MVI_39031/{seqinfo.ini, ignored_regions.txt}` | **đoạn dữ liệu**: `MVI_39031` UA-DETRAC test, frame 1–300, 25 fps, 960×540, 12,0 s; JPG y nguyên byte; mp4 nén lại (**lossy**, chỉ để tiện `--source`); vùng bỏ qua đã che khi dựng cache | 6 |
| `models/yolov8n.pt`, `models/export/yolov8n_640_op12.onnx`, `.export.json` | checkpoint đang dùng (sha256 khớp `weights_sha256` của cache) và ONNX export từ **chính** nó (đối chiếu trọng số 127/127 initializer, 262/262 node với bản re-export) | 6 |

**Không có (cố ý)**: môi trường Conda/venv Windows, `.git`, dataset đầy đủ, video dài, lịch sử kết quả, engine
TensorRT (build **trên Nano**), GT, file riêng.

## 1. Môi trường trên Nano

Thiết bị đã biết (do người dùng cung cấp, chưa kiểm lại từ PC): Jetson Nano 4 GB aarch64, Ubuntu 18.04.6 /
L4T R32.7.5, **Python hệ thống 3.6.9**, numpy 1.19.5, scipy 0.19.1, pandas 0.22.0, **chưa có `lap`**, torch
1.10.0 CUDA. **Giữ nguyên** Python hệ thống và torch.

Yêu cầu **thực tế của mã** (không phải giả định): chuỗi import tracker-từ-cache cần **Python ≥ 3.8** (kiểm tĩnh
`vermin`: 21 file `from __future__ import annotations` → 3.7, `dataclasses` → 3.7, `importlib.metadata` → 3.8) và
các gói `numpy`, `lap` (`lap.lapjv` — bắt buộc, quyết định ghép cặp), `pandas`, `pyyaml`; **không** cần scipy
(chỉ dùng khi `use_lap=False`), **không** cần torch/ultralytics. Đã kiểm trên PC với stack cũ hơn (numpy 1.24.4,
pandas 1.5.3, lap 0.5.12, không scipy): parity ĐẠT, bit-identical.

⇒ Python 3.6.9 **không chạy được** mã này; tạo **một môi trường riêng ≥ 3.8**, ví dụ (chọn một, ghi lại cách đã làm):

```bash
# (a) python3.8 của Ubuntu 18.04 (universe) + venv -- không đụng python3 hệ thống
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev build-essential
python3.8 -m venv ~/nano_env && source ~/nano_env/bin/activate
pip install --upgrade pip && pip install -r requirements_nano.txt      # lap có thể build từ nguồn (cần gcc + numpy)
# (b) hoặc miniforge aarch64 (conda-forge): conda create -n nano python=3.8 numpy pandas pyyaml && pip install lap
```

Rồi **đo thật**:

```bash
cd <thư mục gói>
python src/nano_env_check.py --json reference/env_check_nano.json     # Python>=3.8, numpy/lap/pandas/yaml, backend np không kéo torch, cache đọc được
bash scripts/nano/00_check_stack.sh                                   # JetPack/CUDA/TensorRT/RAM/nvpmodel (cho bước 6)
```

`nano_env_check.py` viết bằng cú pháp 3.6 nên chạy được cả trên Python hệ thống để **báo rõ** thiếu gì; **exit 0 chỉ khi**
Python ≥ 3.8, numpy/lap/pandas/yaml import được, backend np không kéo torch/ultralytics, **và** hợp đồng đầu vào
(`src/nano_preflight.py`, dùng chung với `nano_run_cache.py`) đạt: **đúng hai** cache `nano_smoke_MVI_39031_f0001-0300` +
`yolov8n-coco_conf0.05_all` (mỗi cache: `meta.json` parse được; `MVI_39031.npz` đọc được, đủ 7 mảng, Σ`n_per_frame` = số box,
`frame_idx`/`time_s` tăng ngặt; `MVI_39031.json` parse được và khớp `video`/`n_frames`/`n_boxes`), **đủ bốn** file
`NANO-{smoke,full}/pc-ref-{cv,ekf}/data/MVI_39031.txt` đọc được (≥ 1 dòng MOTChallenge hợp lệ), YAML, mã. "Có thư mục"
**không** đủ. Thiếu gói nào → ghi lại nguyên văn lỗi, **không** đoán.

## 2. Import backend NumPy (không kéo torch/ultralytics)

```bash
BYTETRACK_BACKEND=np python -c "import sys; sys.path.insert(0,'src'); import ul_backend; print(ul_backend.fingerprint()); print('torch?', 'torch' in sys.modules, '| ultralytics?', 'ultralytics' in sys.modules)"
python src/test_nano_parity.py        # tự tạo fixture; mong 20/20 PASS
```

## 3–4. Replay cache + parity — **một lệnh, chỉ ghi `nano-*`, không đụng `pc-ref-*`**

```bash
python src/nano_run_cache.py                 # mặc định --tag nano; --chi-smoke nếu chỉ muốn 300 frame
```

Script làm đúng thứ tự: (0) **preflight** = cùng hợp đồng `nano_preflight.py` với `nano_env_check.py` (2 cache: meta parse,
npz đủ mảng/đếm khớp/timeline, **sidecar parse được + đúng schema**; **đủ 4 tham chiếu PC** đọc được; YAML; mã) **và**
`nano_env_check.py` phải exit 0 (nonzero = lỗi môi trường, tail ghi vào log/JSON) **và** backend np import được với
torch/ultralytics **bị chặn** — thiếu/hỏng bất kỳ thứ gì → **exit 2, không gọi `run_tracking.py`, không tạo `<tag>-*`**; (1) hash SHA256 toàn bộ cây `pc-ref-*` và đối chiếu
`reference/pc_ref_sha256.txt` do PC đóng gói; (2) smoke: CV rồi EKF từ `nano_smoke_MVI_39031_f0001-0300` → `nano_parity`
với `pc-ref` + kiểm bit-identical; smoke không đạt → **dừng**, không chạy full; (3) full 1470 frame tương tự;
(4) hash lại `pc-ref-*` → phải y hệt. Kết quả: `data/processed/trackers/NANO-{smoke,full}/nano-{cv,ekf}/`,
`reference/parity_nano_{smoke,full}_{cv,ekf}.json`, `reference/nano_run_<thời điểm>.{json,log}`; exit 0 = **ĐẠT**.

Lệnh tương đương chạy tay (cùng tham số với tham chiếu; cấu hình = điều kiện **c25**: ngưỡng 0,25, lớp COCO 1 2 3 5 7,
max_det 300, 25 fps, buffer 1,2 s = 30 bước, `elapsed steps`):

```bash
BYTETRACK_BACKEND=np python src/run_tracking.py --cache-id nano_smoke_MVI_39031_f0001-0300 --split-name NANO-smoke \n    --tracker-yaml bytetrack.yaml --tracker-name nano-cv --videos MVI_39031 \n    --conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --fps 25 --track-buffer-sec 1.2 --elapsed-mode steps
python src/nano_parity.py --ref data/processed/trackers/NANO-smoke/pc-ref-cv --test data/processed/trackers/NANO-smoke/nano-cv --json reference/parity_nano_cv.json
```

Tiêu chí (công bố trước, in ở đầu mỗi lần chạy): bbox `|Δ| ≤ 1e-3` px, conf `|Δ| ≤ 1e-6`, **liên kết ID tương đương bằng
một song ánh cố định trên toàn sequence** (hai mức: ID thô và phân hoạch); ghép **mơ hồ** → báo mơ hồ, **không** PASS.
Trên PC, np ↔ ultralytics **bit-identical**. Không đạt ⇒ **DỪNG**: giữ output, gửi `parity_nano_*.json`, `run_meta.json`,
`env_check_nano.json`, `nano_run_*.log`; xác định tầng khác biệt (dữ liệu → cấu hình → dependency (`lap`/numpy) → số học
→ liên kết ID). **Không** nới tolerance, **không** sinh lại tham chiếu PC trên Nano.

## 5. Sau khi parity đạt — bàn giao mốc 1

Gửi lại: `reference/env_check_nano.json`, output `data/processed/trackers/NANO-*/nano-*` (+ `run_meta.json`),
`reference/parity_nano_*.json`, nguyên văn các lệnh đã chạy và cách tạo môi trường (a/b). Chưa có số tốc độ nào ở mốc này.

## 6. Mốc 2 — detector TensorRT + tracker end-to-end trên Nano

Kiến trúc trên thiết bị (vì binding Python của TensorRT trên JetPack 4.6 chỉ có cho **Python 3.6**, còn tracker cần
**≥ 3.8**): **detector chạy ở Python 3.6 hệ thống** (`src/nano_detector.py`: tensorrt + numpy + cv2; bộ đệm GPU bằng
torch 1.10 có sẵn, hoặc pycuda) và ghi **det cache** đúng định dạng PC → **tracker chạy ở venv ≥ 3.8** (`run_tracking.py`)
replay cache đó — cùng cơ chế với PC, không có logic tracker riêng cho Nano. Hợp đồng đã kiểm trên PC
(`src/test_nano_detector.py` 6/6): letterbox 640×640 **pixel-identical** ultralytics; decode + NMS + scale **giống hệt**
`nms.non_max_suppression` + `ops.scale_boxes` trên cùng đầu ra thô; cache ghi ra `open_cache` đọc được và replay được.

```bash
bash scripts/nano/00_check_stack.sh                              # TensorRT/trtexec ở đâu, CUDA, nvpmodel
bash scripts/nano/03_e2e_nano.sh ~/nano_env/bin/python yolov8n_fp16.engine nano-trt
```

`03_e2e_nano.sh` làm: (1) `01_build_engine.sh` — build engine FP16 **trên Nano** từ `models/export/yolov8n_640_op12.onnx`
(engine tạo trên PC **không** dùng được), `02_benchmark_engine.sh` (latency GPU detector, chưa phải pipeline);
(2) `python3 src/nano_detector.py --engine … --images data/smoke/MVI_39031_f0001-0300 --fps 25 --regions
data/smoke/MVI_39031/ignored_regions.txt --cache-id nano_trt_MVI_39031_f0001-0300 --timing-json …` (che vùng bỏ qua như
cache PC; ghi timing decode/pre/infer/post từng frame, bỏ 5 frame warm-up); (3) `nano_det_compare.py` cache TRT ↔ cache PC
ở mức vào tracker (c25) và mức thô — **mô tả** sai khác detector (FP16 + letterbox 640×640 vs .pt rect 640×384): trên PC,
torch FP32 + letterbox 640×640 cho 100 % ghép, IoU 0,9975, |Δconf| 0,003 với 40 frame; (4) tracker CV/EKF từ cache TRT
(`NANO-e2e/nano-trt-{cv,ekf}`) + `nano_parity` với `pc-ref` **chỉ để mô tả** (khác detector ⇒ không kỳ vọng parity);
(5) video minh họa `results_nano/nano-trt/nano-trt-cv.mp4` (cần opencv trong venv; không có thì render trên PC từ MOT txt).

Số đo được phép công bố: `trtexec` (GPU detector), `detector_timing.json` (decode/pre/infer/post, Python 3.6), `run_meta`
tracker (`timing/`), kèm `nvpmodel`/`tegrastats`/nhiệt độ — ghi rõ phạm vi từng số; **không** cộng gộp thành "FPS
end-to-end" khi hai tiến trình chạy tuần tự qua file.

## 6b. Luồng TRỰC TIẾP (vòng 21) — video/ảnh → TensorRT → tracker, xử lý **từng frame**

Hai tiến trình (TensorRT chỉ có binding Python 3.6, tracker cần ≥ 3.8): con `src/nano_detect_stream.py` (Python 3.6 hệ
thống, `LD_PRELOAD` libgomp) đọc frame → TensorRT FP16 → hậu xử lý NumPy → gửi detection của frame đó qua pipe
(`src/nano_stream_proto.py`); cha `src/nano_live.py` (env ≥ 3.8, `BYTETRACK_BACKEND=np`) lọc (`det_cache.filter_dets`) →
`run_tracking.track_video` (cùng vòng lặp với replay cache) → MOT + timing + pipeline + `run_meta.json`. Cửa sổ credit
`--hang-doi K` (mặc định 4): tối đa K frame đã gửi chưa xử lý xong, chế độ offline **không bỏ frame**.

```bash
PYV=~/miniforge3/envs/nano/bin/python          # gọi TRỰC TIẾP, không `conda run`
# 0. engine (một lần, ~8 phút): bash scripts/nano/01_build_engine.sh models/export/yolov8n_640_op12.onnx yolov8n_fp16.engine --fp16
# KHỞI ĐỘNG (một lượt):
bash scripts/nano/04_live_nano.sh cv  results_nano/live/thu-cv                     # 300 JPG smoke, EKF: thay cv -> ekf
bash scripts/nano/04_live_nano.sh ekf results_nano/live/mp4-ekf --source data/smoke/MVI_39031_f0001-0300.mp4
# DỪNG có kiểm soát (từ terminal khác, hoặc Ctrl+C ở terminal đang chạy):
touch results_nano/live/thu-cv/STOP
# KIỂM CHỨNG trước khi đo (A gia lập == pc-ref; B TRT live == replay; C mô tả; D biên):
bash scripts/nano/05_parity_live.sh yolov8n_fp16.engine results_nano/live_v6/parity
# ĐO toàn luồng (chỉ sau khi 05 ĐẠT; điều kiện phải được công bố trước trong file):
bash scripts/nano/06_bench_live.sh reference/dieu_kien_do_toan_luong.md yolov8n_fp16.engine results_nano/live_v6/bench
```

Mã thoát `nano_live.py`: **0** hoàn tất · **2** lỗi trước vòng lặp (không có kết quả) · **3** lỗi giữa luồng (con chết,
EOF không EOS, mất frame, ảnh hỏng, chạm trần `det_max_det`, thiếu frame so với khai báo) — kết quả ghi đủ phần đã xử
lý nhưng **không** "hoàn tất" · **4** dừng theo yêu cầu. Không cần GT; `--regions` (vùng bỏ qua từ nhãn) chỉ dùng cho
chế độ đối chiếu và được ghi trong metadata. Phạm vi từng timer: docstring `src/nano_live.py`. Không render trong mốc này
(video minh họa: `render_mot_video.py` sau khi chạy).

## 7. Đã làm / chưa làm

| Hạng mục | Trạng thái |
|:--|:--|
| Backend ByteTrack NumPy (không torch) | ✅ chạy thật trên PC, bit-identical với backend ultralytics (`reference/parity_pc_np_vs_ul_*.json`) |
| Cache đoạn smoke + cache toàn sequence + tham chiếu CV/EKF | ✅ trong gói |
| Bộ so parity + test, công cụ kiểm môi trường, script chạy thiết bị | ✅ `nano_parity.py` 20/20; `nano_env_check.py`; `nano_run_cache.py` (kiểm trên PC trong gói sạch) |
| Detector: tiền/hậu xử lý NumPy, ghi cache, so detection, render | ✅ `test_nano_detector.py` 6/6 trên PC (suy luận thô bằng torch FP32 thay TensorRT) |
| Yêu cầu Python/dependency của mã | ✅ đo tĩnh (3.8; numpy/lap/pandas/pyyaml) + kiểm stack cũ trên PC |
| Chạy trên Nano (import / parity) | ❌ chưa — thiết bị không tới được từ PC lúc dựng gói |
| Engine TensorRT, suy luận TRT thật, e2e và số đo trên Nano | ❌ chưa — cần thiết bị (`03_e2e_nano.sh`) |
