# REVIEW_BRIEF — sau nghiệm thu `e4a0311`: kế hoạch R46 (ghi nhận, đính chính R46-N1 / N2, bảng D1–D5 CHƯA DUYỆT)

<!-- BASE_TARGET -->

**Trạng thái** (2026-10-09, sau mốc C, mốc cuối của kế hoạch R46):
- Vòng 46 (`e4a0311`): Codex **PASS WITH LIMITATIONS**, đóng R45-F1…F4 trong phạm vi công cụ phân tích và hồ sơ dự thảo.
  R45-F1 đóng ở mức đặc tả: bước chuẩn hoá báo cáo theo D3 **chưa viết**, nên chưa được nghiệm thu.
- Hai ghi chú không chặn R46-N1 / R46-N2: **đã đính chính câu chữ** (mốc B).
- **D1–D5 CHƯA DUYỆT.** Dự thảo giao thức chưa khóa; adapter UA-DETRAC chưa viết; chưa có kết quả detector UA-DETRAC.
- HEAD = `e4a0311`. Thay đổi của kế hoạch R46 chỉ là tài liệu, **chưa commit**, chưa đóng gói delta.
- Base review **vẫn `3e94b21`** (`review_base.json` không đổi).

> **Cập nhật 2026-10-09** (sau mốc C; quyết định người dùng `P712983ae`): người dùng **đã chốt D1–D5**. Năm lựa chọn ghi
> nguyên văn ở `REVIEW46_CHI_TIET.md` mục 11.4, kèm ngày (2026-10-09, Asia/Ho_Chi_Minh) và nguồn (phản hồi trực tiếp của
> chủ tài khoản trong chuỗi email QK9N); mục 11.1–11.3 không đổi. Tóm tắt, căn cứ là nguyên văn ở 11.4: box `van` và
> `others` thành vùng bỏ qua cho mọi lớp; chỉ báo AP car và bus, “mAP (car, bus)”, truck / motorcycle / bicycle N/A; loại
> 173 ảnh không có `<frame>`, lưu danh sách; giữ quy tắc bỏ qua đang dùng trên BDD, không tô xám. Các chỗ "CHƯA DUYỆT" /
> "để trống" khác trong brief này là trạng thái sau mốc C, giữ nguyên. Chốt D1–D5 chưa cho phép chạy: chưa fingerprint,
> adapter, khóa kế hoạch hay đánh giá; mỗi bước cần người dùng duyệt riêng. Chỉ sửa tài liệu, chưa commit.

## Kế hoạch R46 đã làm (chỉ tài liệu)

| Mốc | Nội dung | Chi tiết |
|:--|:--|:--|
| A | Đối chiếu review `e4a0311` và bằng chứng đã giải nén (`SHA256SUMS` 29/29 theo báo cáo giải nén) với repo: đúng target, base, kết luận; ghi nhận PASS WITH LIMITATIONS | `REVIEW46_CHI_TIET.md` mục 9 |
| B | Đính chính R46-N1 ở ba vị trí review nêu; R46-N2 ở câu duy nhất, `REVIEW45_CHI_TIET.md` **mục 2** (review ghi mục 3). Không sửa scorer / công cụ / log, không chạy lại | `REVIEW46_CHI_TIET.md` mục 10 |
| C | Bảng quyết định D1–D5 cho người dùng: phương án, hệ quả, số đếm 8 tổ hợp D1 × D2 × D4, phiếu chọn để trống. **Không phương án nào được chọn**; không có số mới. Brief này | `REVIEW46_CHI_TIET.md` mục 11 |

Không chạy Python, GPU, train, inference, SSH. Không sửa `src/`, scorer, công cụ, log / CSV / JSON, dataset, checkpoint,
kết quả. Không stage / commit.

## Bước tiếp theo: người dùng chốt D1–D5

- Chọn theo phiếu ở `REVIEW46_CHI_TIET.md` mục 11.4, mẫu `D1=a|b; D2=a|b; D3=a|b; D4=a|b; D5=a|b|c`. Đề xuất vòng 45 của
  tác giả chỉ là đề xuất, không có hiệu lực.
- Chốt D1–D5 **chưa** cho phép chạy. Sau đó từng bước vẫn cần người dùng duyệt, theo trình tự ở đầu dự thảo
  `logs/review45/06_du_thao_giao_thuc_detrac.md`: fingerprint dữ liệu → adapter + fixture → khóa kế hoạch mới (commit trước
  suy luận) → ba lượt đánh giá trên PC.
- D5 = (b) cần sửa evaluator: ngoài kế hoạch hiện tại, phải có kế hoạch riêng.

## Chưa làm (ngoài mốc A–C; chờ người dùng)

- Lưu review `e4a0311` và bằng chứng vào repo, archive gói đã review, chuyển base → `e4a0311`.
- Ghi chú N2 ở hàng 41 `CODEX_HANDOFF.md`; trạng thái vòng 44–46 trong `handoff/PLAN_DA_CHOT.md`.

## Nên xem

- `git diff` so với `e4a0311` của: `CODEX_HANDOFF.md` (chỉ hàng 42), `handoff/REVIEW_BRIEF.md`,
  `handoff/REVIEW45_CHI_TIET.md`, `handoff/REVIEW46_CHI_TIET.md`, `handoff/logs/review45/06_du_thao_giao_thuc_detrac.md`
  (chỉ có thay đổi của mốc B).
- `REVIEW46_CHI_TIET.md` mục 9–11. Đối chiếu số ở mục 11 với `REVIEW45_CHI_TIET.md` mục 2–3, mục 3 của
  `REVIEW46_CHI_TIET.md` và `logs/review46/phan_tich/dem_theo_phuong_an.json`.

## Giới hạn

- Mục 11 chỉ tổng hợp hồ sơ vòng 45–46 đã nghiệm thu; không đọc lại XML / ảnh, không chạy công cụ.
- Phân loại `van` / `others` bằng mắt là chủ quan, trên mẫu; reviewer chưa xác minh (không có ảnh).
- Chưa có dự đoán UA-DETRAC: hệ quả ở mục 11 là phía GT / quy tắc chấm, không phải metric.
- Chưa có gói delta cho các thay đổi này; commit và đóng gói do người dùng quyết định.

---

## Phụ lục — brief vòng 46 nguyên văn (gói `e4a0311`, đã nghiệm thu PASS WITH LIMITATIONS)

Tiêu đề gốc: "REVIEW_BRIEF — vòng 46: xử lý review `d4715aa` (R45-F1…F4); UA-DETRAC chưa triển khai". Phần dưới giữ
nguyên văn, kể cả hai khối cập nhật mốc A / B; chỉ hạ cấp tiêu đề, và dấu điền base / target đã chuyển lên đầu file.

Chi tiết: **`handoff/REVIEW46_CHI_TIET.md`**. Base **giữ `3e94b21`**. D1–D5 **chưa duyệt**. Không GPU, không khóa
giao thức.

> **Cập nhật 2026-10-09** (kế hoạch R46, mốc A): Codex nghiệm thu `e4a0311` — **PASS WITH LIMITATIONS**, đóng
> R45-F1…F4 trong phạm vi công cụ phân tích và hồ sơ dự thảo (R45-F1 ở mức đặc tả; chuẩn hoá D3 chưa viết). Hai ghi
> chú không chặn R46-N1 / R46-N2 **chưa** đính chính. Base vẫn `3e94b21`. **D1–D5 vẫn chưa duyệt.** Brief cho vòng kế
> tiếp soạn ở mốc C. Chi tiết `REVIEW46_CHI_TIET.md` mục 9.

> **Cập nhật mốc B** (2026-10-09): đã đính chính câu chữ R46-N1 (dự thảo mục 6, `REVIEW46_CHI_TIET.md` phần R45-F1,
> hàng D3 của `REVIEW45_CHI_TIET.md`) và R46-N2 (câu duy nhất ở `REVIEW45_CHI_TIET.md` **mục 2**; review ghi mục 3).
> Không sửa scorer / công cụ / log, không chạy lại; hàng 41 `CODEX_HANDOFF.md` và `PLAN_DA_CHOT.md` giữ như `e4a0311`.
> Base vẫn `3e94b21`. **D1–D5 vẫn chưa duyệt.** Chi tiết `REVIEW46_CHI_TIET.md` mục 10.

### Tái hiện trước sửa

Chạy nguyên byte `audit_review45.py` trên gói `d4715aa`. Lần 1 FAIL do môi trường (cp1252 trên Windows). Lần 2 với
`-X utf8`: exit 0, tái hiện **4/4**, đầu ra trùng kết quả reviewer.

### Sửa bằng mã (công cụ audit; bản vòng 45 giữ nguyên làm mã cũ)

- **R45-F4**: `doc_xml()` dùng `node is not None`; thiếu thật thì ghi rõ, không để trống. Trên XML thật cả 40/40 video có
  đủ camera / thời tiết (ô trống ở vòng 45 là do parser).
- **R45-F3**: cộng `ignored_region` cấp sequence cho 173 ảnh khi giữ: **+265 lượt vùng áp dụng trên ảnh** (không phải
  vùng hình học). Bốn phương án giữ thành 275.989 / 259.074 / 237.470 / 220.555.
- **Test** `test_cong_cu_detrac.py`: công cụ cũ **2/8** (FAIL đúng lỗi), công cụ sửa **8/8**.

### Sửa diễn giải

- **R45-F1**: scorer **vẫn** đếm TP / FP thô và precision cho lớp không có GT; chỉ AP / recall / F1 = null. Đã sửa
  REVIEW45, dự thảo, REVIEW43, và thêm vào phụ lục mục 10 của kế hoạch vòng 43. Hiển thị N/A cần bước chuẩn hoá báo cáo,
  **phải triển khai sau khi D3 được duyệt**.
- **R45-F2**: rút kết luận "recall / AP tăng". Phản ví dụ AP50 0,505 → 0; chiều trên dữ liệu thật chưa biết. Toolkit
  chỉ theo phần mã đã đọc (bản trích + sha256 trong gói); `AP_DET_EVAL.exe` chưa xác minh. 40.835 chỉ là GT XML đạt
  ngưỡng trong thống kê này.
- **Làm rõ 1–3**: fingerprint → adapter → khóa; adapter lưu danh sách `video/frame` có thứ tự; kiểm w/h mọi ảnh.

### Còn phụ thuộc người dùng

Chọn D1–D5. Chuẩn hoá D3, adapter, bảng hash ảnh làm sau khi duyệt.

**Nên xem**: `REVIEW46_CHI_TIET.md` mục 3–4, `logs/review46/01`–`03`, `05`, `test_cong_cu_detrac.py`, diff của
`logs/review45/06_du_thao_giao_thuc_detrac.md`.
