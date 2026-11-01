# R63 v17-C — v16 + engine chữ nhật (31/10/2026)

Gốc: `src_v16_goc/` = hai file trích nguyên từ ZIP v16 `handoff/nano_transfer_latest.zip` (`302f6af7…`;
`nano_detector.py` `43687a07…`, `nano_detect_stream.py` `5289f3c7…`, khớp `MANIFEST_SHA256.txt` của v16). **Không**
dựa trên `src/` của HEAD (HEAD có LUT R62). Bản sửa: `src/`, diff: `v17_c.diff`.

Sửa (theo review GPT DUYET_CO_SUA, mục `src/nano_detector.py`/`nano_detect_stream.py`):
- `TRTEngine.hw` = (H, W) từ binding input tĩnh 1×3×H×W (lỗi nếu không phải dạng này).
- `TRTEngine.__call__` từ chối tensor sai shape **trước** `cudaMemcpy` (trước đây letterbox 640×640 vào engine
  384×640 sẽ ghi tràn buffer GPU).
- `imgsz_hieu_luc(imgsz, engine.hw)`: engine vuông `imgsz`×`imgsz` → trả `imgsz` (int) ⇒ hành vi và metadata v16 giữ
  nguyên từng byte; engine khác → (H, W). Dùng ở `nano_detector.main` và `nano_detect_stream` (HELLO/meta cache:
  chuỗi `letterbox` theo (H,W) thật, thêm `letterbox_hw` chỉ khi không vuông). `nano_live.py` không đổi (lấy
  `letterbox` từ HELLO).

Kiểm trên PC (env `khoaluan-mot`, 31/10):
- Test v16 `src/test_nano_detector.py` (bản trong ZIP v16) chạy với `nano_detector.py` đã sửa: **6/6 PASS**
  (letterbox 640 pixel-identical, decode/NMS/scale trùng ultralytics, pipeline → cache → replay → so cache PC).
- `test_v17_c.py`: **5/5 PASS** — letterbox (384,640) pixel-identical `LetterBox((384,640), auto=False)` (đệm 12+12
  hàng) **và trùng `LetterBox((640,640), auto=True, stride=32)` của đường PC `predict(imgsz=640)`** ⇒ C1 trên Nano nhận
  đúng ảnh vào như mọi đánh giá PC R53–R61; NanoDetector (384,640) FP32 == `model.predict(imgsz=640)` trên 5 frame
  (bbox |d| ≤ 6,1e-5, conf |d| ≤ 1,3e-6); từ chối tensor 640×640 vào engine 384×640.

Chưa: kiểm trên Nano (11 ca CPU như R62, đường 640 trùng byte v16, pipeline C1), so cặp tốc độ, chất lượng G4.
