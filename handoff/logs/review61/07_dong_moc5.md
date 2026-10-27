# R61 — Đóng mốc 5 và chuyển trạng thái PLAN (27/10/2026)

Người dùng yêu cầu hoàn tất các việc đóng mốc 5 để ngày 28/10 có thể bắt đầu mốc 6. Lượt này làm thủ công trong cửa sổ chat hiện tại, không chạy automation, không chạy lại tracking/evaluator, không sửa thuật toán hay cấu hình đã khóa.

## 1. Trước khi chuyển trạng thái

- `handoff/PLAN_DA_CHOT.md` SHA256 `2b2474aabd3215deb3973d5fa342457d09db422df5b537cbe238d3fe6d93e12a`, trùng ảnh PLAN mà khóa A và tiền/hậu kiểm B đã dùng. HEAD cục bộ tại thời điểm chạy B: `117d62da027e8ce3e4c9f455d576a924b4d82bbb`; snapshot GitHub `4f8131bb0c88b2352a69c3ca266db38b32bfb359` chứa đúng byte PLAN cũ và mã R61 đã chuẩn bị.
- Gói A SHA256 `5c9db320e4965798f50652b823b7a20439410e259752dbf72ebfe990d30fe225`; khóa A SHA256 `f3a138d7497727a205abfa78237f0ec9a6029b89a13bdd018928865351794d4f`. Băm lại trực tiếp hai file sau B, khớp giá trị ghim. Review cuối A `review_009.json` verdict approve; review cuối gói chuẩn bị B `review_009.json` verdict approve ở lượt tương ứng.
- Phiên B `results/phase2/r61_bdd/thuc_thi/cong_b_20261027_01/` đã xong cả 9 bước; `van_hanh/so_crash.json` có 32 lần gọi (9 exit 0, 23 exit 10 tiếp tục, không exit chặn, crash 0). `kiem_khoa/20261027_211257/ket_qua.json` và `hau_kiem.json` đều `dat=true`; hậu kiểm ghi cache 600, tracking 7.200, chấm 36, không lỗi hoặc khối lệch. Báo cáo kết quả: `06_ket_qua_thuc_thi_cong_b.md`.

## 2. Đối chiếu lại kết quả B, độc lập với cờ `dat` của runner

Đọc lại trực tiếp các file JSON/CSV đã lưu bằng công cụ PowerShell; không lấy riêng kết luận `dat=true` làm bằng chứng duy nhất:

1. SHA256 file phê duyệt B `49fa6c7ff81b761fab12ab0c9d8c73ae6eb4ebf2285900a273fbb246a9b17fc0` khớp tiền kiểm. Hash hai CSV tổng hợp tự băm lại khớp `tong_hop/kiem_chung.json`: `bang_chinh.csv` = `6de32d32cfa5a72d9e26c8074421996f6321dc1a84772da46435d2a67463d318`; `ci_108.csv` = `1bfe15c2bf3c6924c123f8e02b39f6cbc37d776557e98c383c79b5d8941873eb`.
2. Bảng chính có 24 nhóm gộp, bảng CI có 108 dòng, metadata bootstrap ghi `default_rng(0)`, 5.000 mẫu × 200 video, tái tính từ ma trận chỉ số lưu đạt. Với **từng** dòng CI, tự tìm hai hàng nguồn theo detector/điều kiện/bộ lọc, tính lại hiệu có hướng (HOTA/IDF1 ×100 pp, IDSW số đếm) và kiểm `ci_chua_0` từ hai cận. **0/108** dòng lệch quá `1e-8` hoặc sai cờ chứa 0.
3. `lap/ket_qua_b204a5c1-05981158.json`: 36 phần tử, tất cả `trung_byte=true`. `cham_lai/ket_qua.json`: 6 phần tử, tất cả `trung_byte=true`. Nhật ký runner không có exit ngoài 0/10, bộ đếm crash bằng 0, không có `van_hanh/dung.json`.

Đây là **kiểm tra độc lập với kết luận của runner nhưng do cùng Codex thực hiện**, không phải review bởi tác nhân thứ hai. Không chạy lại tracker, evaluator hay bootstrap, và không băm lại từng MOT/tracking marker; hậu kiểm B của runner là bằng chứng cho toàn bộ 7.200 đơn vị. Kết luận kiểm tra: **không phát hiện sai lệch trong phạm vi đối chiếu**. Review A và review gói chuẩn bị B là nghiệm thu trước thực thi; không gọi chúng là review độc lập của kết quả B thật.

## 3. Chuyển khóa và ranh giới bằng chứng

Sau khi B hậu kiểm exit 0, cập nhật mục 0 và hàng mốc 5 trong PLAN. SHA256 PLAN sau cập nhật: `303cd5bef84fcc6e9374dd325d01198988816a5b0aead4a9f297016576122ad3`. Gói A, khóa A, phê duyệt B và đầu ra B giữ nguyên. PLAN mới thể hiện trạng thái kế tiếp; vì PLAN là thành phần của ảnh khóa A, không dùng PLAN mới để chạy lại lệnh B cũ. Muốn kiểm lịch sử phải đối chiếu ảnh PLAN cũ ở snapshot GitHub `4f8131b` cùng bằng chứng đã lưu, không sửa hoặc giả làm cho khóa cũ khớp PLAN mới.

Mốc 5 đã hoàn tất thực thi và báo cáo. Mốc 6 (tối ưu Nano) chưa bắt đầu; theo quyết định của người dùng, để ngày 28/10 bắt đầu bằng một phạm vi/kế hoạch riêng. Không đưa dữ liệu gốc, video, cache NPZ hay toàn bộ 7.200 kết quả tracking lên GitHub; chỉ snapshot mã/hồ sơ và các bảng, biên bản tổng hợp nhỏ phục vụ xem xét.
