# REVIEW_BRIEF — vòng 14: nghiệm thu launcher PC (`a5eec39`) + gói chuyển Jetson Nano độc lập

<!-- BASE_TARGET -->

Review `a5eec39`: **PASS WITH LIMITATIONS** cho launcher PC (đóng L-F1…L-F4). Gói khớp **chính xác** 1.918.799 byte /
`1c6b1336…` → lưu `handoff/archive/review_delta_a5eec39.zip`; **base review → `a5eec39`** (vòng 14). Nano F1/F3 và chạy trên
thiết bị **chưa** nghiệm thu. Không tuning/train; tracker, cấu hình khóa, GT, kết quả UA-DETRAC/BDD, 4 run `v2` **không đổi**.
Chi tiết: **`handoff/REVIEW14_CHI_TIET.md`**; hướng dẫn: **`NANO_START_HERE.md`** trong gói.

## Thiết bị

Nano 4 GB aarch64, Ubuntu 18.04.6 / L4T R32.7.5, **Python hệ thống 3.6.9**, numpy 1.19.5, pandas 0.22.0, **chưa có `lap`**,
torch 1.10.0 CUDA. Từ phiên này: ping unreachable, SSH timeout, không khóa SSH ⇒ **không tới được** → làm trên PC, kèm khối
lệnh (phụ lục mục 5).

## Yêu cầu thực tế của mã (không giả định)

- Chuỗi import thật của tracker-từ-cache (chặn torch/ultralytics): **numpy, lap, pandas, pyyaml**; scipy **không** được nạp.
- `vermin` tĩnh: **Python ≥ 3.8** (21 file `from __future__ import annotations`, `dataclasses`, `importlib.metadata`) ⇒
  3.6.9 hệ thống không chạy được; sửa cho 3.6 = đổi 21+ file → **không làm**; dùng **môi trường riêng ≥ 3.8** (python3.8 apt
  + venv hoặc miniforge), giữ nguyên python3 hệ thống và torch.
- Kiểm trên PC với stack cũ (numpy 1.24.4, pandas 1.5.3, lap 0.5.12, **không scipy**): replay + parity ĐẠT, **bit-identical**.

## Gói `handoff/nano_transfer_latest.zip` (`src/make_nano_transfer.py build`, `10d26fa`)

**63.764.728 byte**, sha256 `025187770c0876ad849ac32c27ba94b6aff934742f3592e80cf79ed8a99a8fd5`, 537 thành viên. Gồm: `src/`
(không `demo`), `configs/`, vendor + `LICENSE`, `scripts/nano/`, `nano_env_check.py` (cú pháp 3.6), `requirements_nano.txt`;
đoạn `MVI_39031` frame 1–300 @25 fps (300 JPG y nguyên + mp4 lossy); cache **cắt đúng đoạn** (`meta.json` nguyên văn, giữ
frame rỗng/`time_s`) + cache gốc 1470 frame (34 frame rỗng giữ nguyên); tham chiếu PC `NANO-smoke`/`NANO-full`
`pc-ref-{cv,ekf}` (np, chặn torch; == ultralytics bit-identical); `yolov8n.pt` + ONNX (đối chiếu trọng số) + hash;
`PACKAGE_INFO.json` (commit đóng gói / commit code), `MANIFEST_SHA256.txt`. Không: env Windows, `.git`, dataset, engine, GT, file riêng.

## Kiểm gói (giải nén thư mục tạm ngoài repo, cwd = gốc gói, đường dẫn tương đối)

| Bước | PC env | Stack cũ |
|:--|:--|:--|
| manifest 536/536; `nano_env_check` ĐẠT; import np `torch=false` | ✅ | ✅ |
| `test_nano_parity` | 20/20 | 20/20 |
| replay CV/EKF → parity vs `pc-ref` (smoke 545 hàng; full 5842/5845) | ĐẠT, **bit-identical** ×4 | ĐẠT, **bit-identical** ×4 |
| ONNX đọc được; 300 JPG + 300 frame mp4 | ✅ | (env PC) |

Chỉ chứng minh gói **đủ**, không chứng minh tương thích Nano. Tiêu chí parity giữ nguyên (bbox 1e-3, conf 1e-6, song ánh
ID toàn sequence, mơ hồ không PASS).

## Trạng thái Nano và commit

Nano: **chưa chạy** (chưa import, chưa parity). Commit (cục bộ): `3c20fbe` review + base; `10d26fa` builder v2 +
`nano_env_check` + START_HERE; sau đó log/tài liệu. Không chạy full suite (mã lõi không đổi). Delta từ `a5eec39` không chứa
gói chuyển (untracked/ignored).

**Bước tiếp theo**: chạy khối lệnh phụ lục mục 5 trên Nano, gửi `env_check_nano.json` + `parity_nano_*.json`; đạt mới sang
detector TensorRT (build engine trên thiết bị).
