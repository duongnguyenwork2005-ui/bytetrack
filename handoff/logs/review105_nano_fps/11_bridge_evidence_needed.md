# Bằng chứng bridge đã nhận và kiểm

Đã nhận `scratch/r105_nano_host/full/`, `smoke/`, `executed_sources/`, log kết thúc và `host_exit.json`; đã sao vào `host_evidence/` trong R105. Không còn thiếu source để chấm schema sớm.

- Sáu source ở gốc và bốn source `package/` khớp hash thực thi: 10 file duy nhất, 13 tham chiếu trong hai manifest do ba driver được nhắc lại.
- Full 30/30, smoke 8/8, archive/hash/5.700 frame/cấu hình/FPS/p90/fallback đã tự kiểm trực tiếp trên PC, exit 0: `verification.json`, `tier1_checks_current.json`.
- `host_exit.json` ghi exit 0, guard cuối và job R105 rỗng. Đây là record do bridge thu thập, không phải probe SSH mới từ coder.
- Giữ JSON/archive nguyên byte. Không thêm giả field package hash vào schema cũ. Giới hạn: sáu hash source LF trọng yếu mỗi run nhưng chưa có hash/snapshot toàn bộ baseline package. Khác biệt source đã chạy và bundle local: `executed_to_current.diff`.

Không cần chạy lại 38 lượt đã có bằng chứng hợp lệ. Phép đo NMS bổ sung, nếu phiên chính còn kênh bridge và thời gian trong R105, được mô tả riêng ở `12_bridge_nms_request.md`; hiện chưa mở job. Review Codex độc lập vẫn cần thực hiện trước bước sau.
