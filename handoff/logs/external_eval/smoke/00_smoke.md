# R64 smoke 128 frame — SVMOT `seq_22_morning`, RoundaboutHD `imagesc001` (02/11/2026)

Giao thức: GPT `hoi_codex_R64_dataset_moi_20261101/answer.json` (Q1–Q5). Mã `src/r64_ngoai/danh_gia_ngoai.py`
(test `test_danh_gia_ngoai.py` 5/5), điều phối `src/r64_ngoai/chay.sh`. Log `01_svmot_smoke.log`, `02_roundabout_smoke.log`.

## Kiểm kỹ thuật (Q5)

1. Checkpoint sha256 = R54 cho 3 detector (kiểm trong `buoc_cache`); imgsz 640 rect mặc định, conf 0,05, iou 0,7,
   max_det 3000, FP32, batch 1, không mask; tải: conf > 0,25 / > 0,10, lớp 1 2 3 5 7, max 300.
2. Namespace mới: split `SVMOTSYS-test`, `RoundaboutHD-prefix3000` (+ `-smoke128`), cache `r64[smoke]_<ds>_<det>_c0.05`,
   run `r64[smoke]-<det>-<cond>-<loc>[-sN]`; chẩn đoán cũ (`SVMOT-test`, `UTUAV-pilot*`) giữ nguyên. seqinfo mới ghi
   `name` = tên thư mục (bản gốc SVMOT lệch 4/9). SVMOT: 9 chuỗi = 1.200 + 3×900 + 5×870 = **8.250 frame** (GPT đúng;
   con số 8.150 trong câu hỏi là nhầm); số ảnh = seqLength mọi chuỗi.
3. Converter: GT SVMOT mới = nguồn (7 cột đầu + visibility trùng; lớp gộp 1) = bản `SVMOT-test` của GPT trên cả 9 chuỗi.
   RoundaboutHD `SCT_GT` xyxy → xywh (kiểm x2 > x1, y2 > y1 mọi dòng); vẽ GT lên frame 1/2537/3000 cam4 và 1/435/870
   SVMOT seq_26: box khớp xe (ảnh trong scratchpad phiên). Không NaN, w/h > 0, frame trong 1..N.
   **Lỗi nguồn RoundaboutHD:** cùng (frame, id) cho hai box khác nhau — toàn nguồn cam1/cam3/cam4 có **2/10/14 cặp
   (4/20/28 dòng)**; trong frame ≤ 3000 chỉ cam4 **8 cặp = 16 dòng, đều ID 105, frame 2536–2543** (bản trước ghi nhầm
   "ID 314" và nhầm cặp với dòng — đính chính theo GPT). Chính sách tạm lớp 8 **bị GPT bác** (TrackEval kiểm ID duy nhất
   trước khi lọc distractor → lỗi tại imagesc004 frame 2536 ID 105). **Áp dụng B1 (02/11):** bỏ cả 16 dòng, không bỏ cả
   track, không đổi ID, không tạo ignore; cam4 còn 19.528 dòng; danh sách dòng nguồn bị bỏ trong
   `data/processed/RoundaboutHD-prefix3000/gt_log.json` (`dong_bo_vi_trung_frame_id`); GT cũ giữ ở
   `data/processed/_r64_bang_chung/RoundaboutHD-prefix3000_gt_v1_distractor8/`. Kiểm qua TrackEval: tracker oracle (GT
   làm dự đoán) cam4 → HOTA 1,0, không lỗi (`results/trackeval/RoundaboutHD-prefix3000/r64kiem-gt-oracle-cam4`). Smoke
   (frame ≤ 128) không chứa dòng trùng nên kết quả smoke không đổi.
   Lớp nguồn SVMOT: cột 8 nhận 0..6 (0-based; inventory gọi `classes_1_based` nhưng có 0) — không dùng để ánh xạ, chỉ gộp.
4. mixed5 → COCO qua `detector_classes` (bảng R61), `cls_goc` lưu; kiểm box/conf/thứ tự không đổi; khóa cache theo tên
   chuỗi/camera thật (không dùng `track_video_file --save-cache`). RoundaboutHD đọc thẳng `video.mp4` (cv2), không ghi lại
   JPG; dừng nếu decode thiếu frame hoặc sai kích thước.
5. Timeline cache: frame 1..N liên tục, time_s = (i−1)/fps, seqLength GT = số frame cache; `run_meta`: buffer 36 (SVMOT)
   / 18 (RoundaboutHD), elapsed steps, fps đúng. Cache có `.xong.json` (sha256 npz/json/meta).
6. Smoke đủ 3 detector × c25/c10 × CV/EKF/UKF/PF500 s0/1/2 = **36 run mỗi tập, 0 thiếu**; tái tổ hợp `raw_fields` =
   COMBINED (HOTA/IDF1/MOTA/IDSW, |Δ| ≤ 1e-9) cả 72 run. **Lặp PF cùng seed:** yolov8n_coco c25 PF500 s0 chạy lại tên mới
   → `data/seq_22_morning.txt` **trùng byte**; s1 khác s0; chấm lại → `combined_metrics` trùng.
7. Không unreadable/truncated; tối đa 73 box/frame (≪ 3000).

## Sửa theo review GPT 02/11 (`hoi_codex_R64_review_smoke_20261102/answer.json`: B2 KHONG_DAT_Q5, B3 cho phép có điều kiện)

- `danh_gia_ngoai.py` viết lại cổng: `kiem_don_vi_cache` (marker đọc được, đúng detector/video/dataset/split, đủ
  artifact, sha256 khớp) + `kiem_noi_dung_cache` (meta/spec, mapping tính lại từ `names`, frame 1..N, time_s=(i−1)/fps,
  tổng n_per_frame, orig_shape, hữu hạn, conf, box, cls = ánh xạ(cls_goc), unreadable/truncated = 0) dùng chung trước
  skip cache và trước tracking; thư mục có artifact thiếu meta / đơn vị dở (artifact không marker) → dừng, không ghi đè;
  marker ghi nguyên tử SAU hậu kiểm nội dung. Provenance mới: manifest nguồn (sha256 từng ảnh gộp / sha256 mp4, phạm vi
  frame, fps), hash mã wrapper + helper, GT lúc sinh tách khỏi GT chấm. Đơn vị đã sinh bằng mã cũ (SVMOT yolov8n_coco
  seq_22–25, RoundaboutHD yolov8n_coco cam1–2, 6 cache smoke): **biên bản xác minh bổ sung** riêng
  `<video>.xac_minh_bo_sung.json` (thời điểm, cổng ĐẠT, nguồn + hash mã tại thời điểm xác minh), không sửa metadata cũ
  (log `../03_xac_minh_bo_sung.log`, 12/12 ĐẠT).
- Tracking: cổng trước (cache + GT chấm không trùng); hậu kiểm (`run_meta` buffer/fps/elapsed, `max_time_lost` thực của
  tracker dựng lại = 36/18, timing đủ 1..N và n_steps = 1, output hữu hạn, không trùng (frame,id)); log subprocess đầy đủ
  `<ds>/track_logs/`; `cong_r64.json` cạnh kết quả chấm. Ngân sách: `--het-han` + dự toán thời gian, dung lượng > dự toán
  + 5 GiB; hết ngân sách → mã 10 (PARTIAL), chuỗi dừng. Tổng hợp: COMBINED/tung chuỗi/theo scene đều tái tổ hợp
  raw_fields; thiếu run → `PARTIAL`.
- Test `src/r64_ngoai/test_danh_gia_ngoai.py` **13/13** (thêm: marker thiếu/hỏng, hash sai, đơn vị dở, timeline sai, lớp
  sai, NaN, mapping sai, meta lệch, MP4 thiếu frame/sai kích thước + gọi `detect_frame`).
- Lặp PF500 cùng seed + chấm lại trên **RoundaboutHD** smoke (yolov8n_mixed5 c25 s0): `data/imagesc001.txt` **trùng
  byte**, s1 khác s0, `combined_metrics`/`per_video_metrics` chấm lại trùng. **Bằng chứng kiểm lại được** (bổ sung theo
  review cuối GPT): `04_lap_pf_cham_lai.json` — cho cả SVMOT và RoundaboutHD: tên run gốc/lặp, seed, sha256 MOT hai phía
  + seed 1, so metric, mtime file (MOT lặp SVMOT 01:50:48, RoundaboutHD 02:07:18). Artifact:
  `data/processed/trackers/<split>-smoke128/r64smoke-lap-*`, `results/trackeval/<split>-smoke128/r64smoke-lap-*`.
- **Review cuối GPT 02/11 (DAT_CO_SUA):** thêm kiểm fingerprint `spec_sha256` từ GEN_FIELDS; marker legacy chỉ được nhận
  khi có biên bản xác minh gắn đúng hash marker; marker mới phải có provenance (nguon, ma_sha256, split, gt lúc sinh);
  `kiem-run` (xong / chưa / dở) thay kiểm `ls` lỗi trong `chay.sh` (đòi đồng thời run thường và -s0); tiền kiểm ngân
  sách bắt buộc `--het-han` (trừ smoke), dự phòng 30 phút, ghi `ngan_sach.jsonl`. Test **15/15**.
- Converter đầu/giữa/cuối + một cặp liên tiếp cho **mọi** chuỗi/camera: `../kiem_converter/` (13 ảnh + JSON số GT/frame).
- **Sai lệch Q4 ghi nhận:** cache RoundaboutHD đầy đủ đã được mở song song lúc 01:51 trước khi xong SVMOT (cam1–2 xong);
  đã dừng sau review, không mở thêm đơn vị RoundaboutHD trước khi xong SVMOT.

## Số smoke (mô tả, không chọn/chỉnh gì theo đây)

| Tập | Detector | c25 HOTA CV / EKF / UKF / PF500 (s0–s2) | MOTA CV |
|:--|:--|:--|:--|
| SVMOT seq_22 | yolov8n_coco | 75,24 / 75,26 / 75,30 / 75,34–76,08 | 72,0 |
| | yolov8n_mixed5 | 64,63 / 65,44 / 65,40 / 64,43–64,71 | 22,1 |
| | yolo11n_mixed5 | 65,38 / 65,40 / 65,41 / 65,35–65,53 | 21,5 |
| RoundaboutHD cam1 | yolov8n_coco | 34,63 / 34,54 / 34,54 / 18,71–21,83 | −145,6 |
| | yolov8n_mixed5 | 65,78 / 65,90 / 65,90 / 45,30–52,64 | 69,9 |
| | yolo11n_mixed5 | 46,52 / 46,43 / 46,43 / 31,66–33,92 | 3,4 |

Quan sát (không suy nguyên nhân chắc chắn): RoundaboutHD 128 frame đầu chỉ có 239 box GT (5 ID) trong khi cảnh có nhiều
xe đỗ ngoài SCT_GT ⇒ MOTA âm lớn với yolov8n_coco là đúng chính sách "xe đỗ chịu FP" đã chốt; PF500 thấp hơn rõ trên
RoundaboutHD smoke. Thời gian: cache SVMOT 6–8 fps (ảnh trên ổ rời F), RoundaboutHD 11–13 fps (decode 4K); tracking
CV/EKF/UKF ~7–9 s, PF500 ~8–15 s mỗi run 128 frame.
