# Snapshot sao lưu ngày 26/11/2026

Người dùng yêu cầu push để phòng xóa nhầm. Đây là snapshot công việc hiện
tại, không phải tuyên bố mọi nhánh nghiên cứu đã đạt yêu cầu hoặc source
đã được chỉnh theo mục lục mới.

## Cách tổ chức

- Nhánh `phase2-mixed-traffic`: source, cấu hình, source LaTeX trong
  `khoa_luan_prism/`, PDF/slide và các báo cáo/summary.
- Tag `backup-20261126`: mốc khôi phục bất biến của lần sao lưu này.
- GitHub Release **dạng nháp**: các gói payload theo manifest, gồm kết quả
  thô, cache/bằng chứng thực nghiệm và model. Chỉ coi phần này hoàn tất
  khi manifest và asset đã được kiểm trên GitHub.
- `git_snapshot_manifest.tsv`: SHA256 của các tệp working-tree được chọn
  vào snapshot; không gồm chính manifest để tránh tự tham chiếu.

## Phạm vi chưa bao gồm

Dataset gốc/giải nén/xuất YOLO, `data/`, `tmp/`,
`results/track_video/`, `handoff/archive/`, các ổ/thư mục ngoài repo,
tệp local bị Git ignore khác và filesystem của Nano không được mặc nhiên
sao lưu. Model local trong `models/` được kiểm riêng và thêm vào payload
nháp. Khôi phục thí nghiệm vẫn cần dữ liệu đầu vào theo manifest nguồn.
Release nháp không nên được publish trước khi kiểm quyền phân phối dữ liệu.

Không thay đổi thuật toán, cấu hình thí nghiệm hay nội dung sourceLaTeX
trong thao tác sao lưu. Git attributes chỉ bảo toàn byte của các vùng mới,
để checksum trong bằng chứng nghiên cứu không bị đổi do CRLF/LF.
