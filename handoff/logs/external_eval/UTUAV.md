# R64 — UTUAV: CHƯA ĐÁNH GIÁ ĐỦ MA TRẬN (02/11/2026)

**Trạng thái:** không chạy benchmark hệ thống đêm 01–02/11 theo quyết định GPT (Q3,
`D:/LABOTARY/automation/runs_local/hoi_codex_R64_dataset_moi_20261101/answer.json`): chỉ trích dẫn pilot. Lý do là
**độ sẵn sàng giao thức và ngân sách**, không phải vì điểm thấp.

## Đã có (pilot GPT 01/11, `handoff/logs/external_mot_prep/02_pilot_utuav.md`)

- Dữ liệu: ảnh 4K chụp từ drone, split Sequential_Gap300; đoạn liên tục dùng được cho tracking là validation B
  (738 frame, 26341–27078) và C (751 frame, 20271–21021). Split evaluation: **B có 975 frame liên tục 27379–28353**,
  còn C rời rạc (990 frame trải 6707–22307, đoạn liên tục dài nhất 4 frame) — `images_inventory.json` (đính chính theo
  review cuối GPT 02/11; bản trước ghi nhầm cả evaluation đều rời rạc). Giao thức dự kiến vẫn chỉ dùng validation B/C;
  không tự mở evaluation.
  Trường `occluded` toàn 0 (placeholder). **FPS nguồn chưa xác minh** (pilot giả định 30).
- Smoke detector: chỉ thử **YOLO11n mixed5** trên vài frame (C 10 frame và B 3 frame ở 640, C 3 frame ở 1280) →
  **0 GT động khớp IoU 0,5**. Detector cho ra box lớn hơn nhiều so với xe nhỏ nhìn từ trên cao, và không có
  motorcycle nào trên các frame đã thử.
- Chẩn đoán riêng bằng detection tổng hợp từ GT (không phải kết quả YOLO): KF/CV nhỉnh hơn EKF/UKF rất ít, còn PF100 kém
  hơn rõ.

## Diễn giải đúng phạm vi

Có thể viết: "pilot YOLO11n mixed5 cho thấy khó chuyển miền sang ảnh UAV trên các frame đã thử". **Không** được viết
"toàn hệ thống thất bại trên UTUAV": mới thử một checkpoint trên vài frame, chưa chạy ma trận 3 detector × 4 bộ lọc,
và FPS chưa có nguồn.

## Lượt sau (đã chốt trước, chưa chạy)

1. Xác minh FPS nguồn trước khi khóa benchmark; không dùng FPS giả định cho kết quả chính thức.
2. Dùng toàn bộ B validation 26341–27078 và C 20271–21021. Nhãn 6 lớp: motorbike/car/bus → class 1, ba lớp `_stat` →
   distractor class 8. Không dùng split evaluation (C rời rạc; B liên tục nhưng ngoài giao thức đã chốt), không thay
   bằng split training.
3. Ma trận như SVMOT/RoundaboutHD: 3 detector R54 × CV/EKF/UKF/PF500 s0/1/2 × c25/c10; wrapper
   `src/r64_ngoai/danh_gia_ngoai.py` cần thêm adapter UTUAV (ảnh 4K, XML/MOT) theo cùng các cổng đã kiểm.
