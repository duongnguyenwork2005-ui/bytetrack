# R63 G1(a0)+(a)+(b) — engine cũ vs engine build lại trên board hiện tại (31/10/2026)

Boot `89a0f445-6290-427d-9305-08ed1adaa30a` (reboot ấm, thẻ HS 50 MHz), Chromium đã đóng, MAXN. Một tác vụ Nano
một lúc. Bằng chứng nhỏ kéo về `a0_hai_engine/` (tom_tat, moi_truong, frame.csv, det_pass0.npz, giam_sat, trtexec).

## Build (G1 a0)

`cong_cu_v4/build_engine_nano.sh` trên ONNX v16 `a66a5e03…` (cùng cờ v16: `--fp16 --workspace=1024 --verbose`):
PASSED sau 419 s, engine **`9fae6b43…`** (12.630.168 byte), `engine_v16_board_hien_tai_01/`; giám sát không vi phạm
(CPU/GPU max 50/48,5 °C, swap +44 MB < 50). Engine cũ `c32690cd…` (build board R62) giữ nguyên.

## trtexec A–E (`cong_cu_v4/g1_a0_hai_engine.sh`, cũ trước rồi mới, nghỉ 20–30 s; mọi lượt hợp lệ, không vi phạm)

| GPU Compute median (ms) | A mặc định | B noDataTransfers | C spinWait | D CUDA graph | E profile |
|:--|--:|--:|--:|--:|--:|
| engine cũ `c32690cd` | 68,46 | 68,45 | 67,82 | 68,28 | 68,41 |
| engine mới `9fae6b43` | 67,94 | 67,93 | 67,27 | 67,72 | 67,92 |

Lượt A: latency median 69,22 (cũ) / 68,71 (mới); H2D 0,48; D2H 0,28; enqueue ~9 ms (bất đồng bộ).

## bench 300 ảnh MVI_39031, 4 pass xen kẽ đồng bộ/CUDA event, warmup 25 (mới trước, cũ sau)

| median (ms) | exec host đồng bộ | GPU exec (event) | GPU H2D+exec+D2H | host infer | decode ảnh | letterbox+pre | decode_nms | copy f32 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| mới | 68,16 | 68,11 | 72,01 | 72,33 | 11,26 | 13,31 | 6,76 | 0,98 |
| cũ | 68,66 | 68,59 | 72,58 | 72,91 | 11,21 | 13,38 | 6,80 | 0,98 |

Parity đồng bộ/event: 300/300 frame trùng byte (cả hai engine).

## Kết luận G1 (đo được)

- **~69 ms là thời gian GPU thực thi engine**, không phải phí gọi: exec host đồng bộ 68,16 vs GPU event 68,11 ms
  (chênh 0,05 ms); trtexec GPU Compute 67,9 ms. H1 đúng ⇒ engine là nút thắt (~68 ms trong ~114 ms/frame của lượt
  ổn định) ⇒ G3 (giảm kích thước đầu vào) có cơ sở.
- Engine build lại trên board hiện tại nhanh hơn engine cũ **~0,5 ms (−0,7 %)** — nhỏ; số R62 vẫn nối được.
- Parity detection hai engine (300 frame, `occlusion_eval.match_frame`, IoU ≥ 0,5; `parity_hai_engine.json`):
  conf ≥ 0,25: 1.516/1.519 ghép (99,8 % hai chiều), IoU trung vị 1,0 (min 0,84), cùng lớp 100 %, |Δconf| trung vị
  0,0017; conf ≥ 0,05: 2.918 ghép / 2.938 cũ / 2.940 mới (99,3 %). Không trùng byte (đúng kỳ vọng FP16 tactic).
- Engine mới `9fae6b43…` là baseline "trước" của R63.
