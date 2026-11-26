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
  vào tag nguồn `backup-20261126-source` (commit5591695); không gồm chính
  manifest để tránh tự tham chiếu. Các helper/metadata đóng mốc được bổ sung
  ở các commit sau, không sửa nội dung nghiên cứu hoặc quyển.

## Tình trạng

**Đã hoàn tất.** Source20940tệp đã đối chiếu byte và push tới GitHub.
Payload160391tệp, gồm145model local bịignore bổ sung, đã sao lưu bằng30ZIP
và các mảnh48MiB. Tổng204asset/9.510.266.459byte trên Release nháp459160982
đã khớp SHA256 phía GitHub;31archive ghép lại đúng hash gốc.
`BACKUP_STATUS.json` ghi COMPLETE; `final_verification.json` giữ danh sách
asset/hash và phạm vi. Tag `backup-20261126` là mốc đóng sao lưu.

Engine/ONNX/plugin15tệp được chọn trên Nano đã sao lưu riêng và đã kiểm
khôi phục bằng tải thật từ GitHub, không chỉ đọc lại bản local. Đây không
phải toàn bộ filesystem Nano. Release nháp cần tài khoản có quyền truy cập;
giữ ở dạng nháp để tránh phân phối công khai dữ liệu thô.

## Khôi phục sau khi bản sao lưu hoàn tất

Clone đúng nhánh làm việc để lấy source và helper mới nhất:

```powershell
git clone --branch phase2-mixed-traffic https://github.com/duongnguyenwork2005-ui/bytetrack.git bytetrack-restore
cd bytetrack-restore
```

Dùng Python3.11 trở lên và Git đã đăng nhập GitHub qua credential manager.
Chọn thư mục trống có đủ dung lượng, ví dụ `D:/backup-assets`:

```powershell
python handoff/backups/20261126/download_assets.py --release-id 459160982 --dest D:/backup-assets
python handoff/backups/20261126/assemble_transport.py --assets D:/backup-assets
python handoff/backups/20261126/restore_payload.py --assets D:/backup-assets --dest .
```

Downloader kiểm digest từ GitHub; assembler kiểm từng mảnh và ZIP hoàn chỉnh;
restorer kiểm từng tệp và từ chối ghi đè tệp đang có nội dung khác. Nó không
xóa tệp có sẵn. `--names` của downloader cho phép chọn riêng một số asset.

Gói Nano ghép riêng bằng:

```powershell
python handoff/backups/20261126/assemble_transport.py --assets D:/backup-assets --manifest nano-transport-manifest.json.gz
```

`nano-engines-20261126.tar.gz` chứa các đường dẫn tương đối với
`/home/jetson/nano_run/`, kèm bindings.json. Chỉ khôi phục vào thư mục đã
chọn sau khi kiểm manifest; engine gắn với Nano/CUDA10.2/TensorRT8.2.1.9
đã dùng đo kết quả, không phải binary dùng tùy ý trên mọi thiết bị.

## Phạm vi chưa bao gồm

Dataset gốc/giải nén/xuất YOLO, `data/`, `tmp/`,
`results/track_video/`, `handoff/archive/`, các ổ/thư mục ngoài repo,
tệp local bị Git ignore khác và toàn bộ filesystem của Nano không được mặc
nhiên sao lưu. Model local trong `models/` và15engine/ONNX/plugin được
whitelist trên Nano được kiểm riêng và thêm vào payload nháp. Khôi phục
thí nghiệm vẫn cần dữ liệu đầu vào theo manifest nguồn.
Release nháp không nên được publish trước khi kiểm quyền phân phối dữ liệu.

Không thay đổi thuật toán, cấu hình thí nghiệm hay nội dung sourceLaTeX
trong thao tác sao lưu. Git attributes chỉ bảo toàn byte của các vùng mới,
để checksum trong bằng chứng nghiên cứu không bị đổi do CRLF/LF.
