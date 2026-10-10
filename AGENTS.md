# AGENTS.md — lộ trình đọc cho agent (Codex, Claude Code)

Repo luận văn ByteTrack + KF/EKF/UKF/PF, nhánh làm việc `phase2-mixed-traffic`. File này chỉ chỉ đường; **không** ghi
trạng thái ở đây (trạng thái đổi theo vòng, file này được orchestrator bảo vệ).

## Đọc theo thứ tự, dừng khi đủ cho nhiệm vụ

1. `handoff/PLAN_DA_CHOT.md` **mục 0** — trạng thái chuẩn duy nhất: mục tiêu, quyết định đã chốt, kết quả đã xác
   minh (kèm đường dẫn), việc dở, bước tiếp, điểm chưa rõ. Phần còn lại của file: phạm vi và quy tắc đã chốt.
2. Kế hoạch đang chạy, nếu có: `plan.json` của lượt orchestrator được giao (nhiệm vụ, `write_paths`, tiêu chí).
3. Chỉ những file mà mục 0 hoặc kế hoạch trỏ tới cho nhiệm vụ hiện tại.

**Chỉ mở tài liệu lịch sử khi nhiệm vụ cần hoặc khi có mâu thuẫn**: `CODEX_HANDOFF.md` (lớn, tìm theo mục),
`handoff/REVIEW_BRIEF.md`, `handoff/REVIEW*_CHI_TIET.md`, `handoff/CODEX_REVIEW_*.md`, `handoff/logs/`,
`PHASE2_REPORT.md`, `README.md`. Bảng "file nào thay file nào" và vị trí lưu trữ: `handoff/LUU_TRU.md`.

## Không quét

Dữ liệu và đầu ra lớn: `data/raw/`, `data/extracted/`, `data/yolo_*`, `handoff/archive/`, `results/track_video/`,
video/NPZ/zip. Dùng manifest, CSV tổng hợp hoặc báo cáo trỏ tới chúng. Vùng lưu trữ ngoài repo
`D:\LABOTARY\_luu_tru\` chỉ mở khi cần khôi phục.

## Quy tắc giữ nguyên

- Phạm vi và quy tắc "hỏi trước" trong `handoff/PLAN_DA_CHOT.md`; mọi việc ngoài kế hoạch đã duyệt phải hỏi người dùng.
- Không đổi thuật toán tracking, metric hay cấu hình thí nghiệm đã khóa nếu kế hoạch không yêu cầu.
- Hết vòng việc: cập nhật `handoff/PLAN_DA_CHOT.md` mục 0 trước, rồi mới viết brief/handoff.
- Môi trường Python: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe`.
