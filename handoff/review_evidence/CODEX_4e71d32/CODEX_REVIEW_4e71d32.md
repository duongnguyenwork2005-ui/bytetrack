# Review 4e71d32 — nghiệm thu R32-F1 và R32-F2

**Kết luận: PASS WITH LIMITATIONS. Đóng R32-F1 và R32-F2.** Không có lỗi chặn mới trong phạm vi hai finding và bộ nhãn đã ghim. Kết luận này nghiệm thu bản sửa công cụ/báo cáo; không chứng nhận nhãn là bản phát hành chính thức và không cấp phép training dài.

Review độc lập, chỉ đọc snapshot và dữ liệu trong workspace. Không sửa repository Windows, nhập nhãn, chạy training, SSH Nano, commit hoặc push.

## 1. Snapshot được review

- Base: `bcb92f86824812ceea3dee56dabe88ef802b1a25`.
- Target: `4e71d3217aed3b0757136b4c388abd3e3ed4da4a`.
- ZIP: **1.156.972 byte**; SHA256 `a34037a13ad0c4272e7a260e82b9e67b21f4fb814cc5382d0f6a9130a72a3a05`.
- ZIP đọc được toàn bộ, `testzip` đạt.
- **301/301 payload** và **27/27 alias trùng nội dung** khớp SHA256/số byte trong manifest.
- Reverse patch khôi phục 7 preimage có hash base khớp manifest; áp forward patch cho **328/328 đường dẫn** đúng hash target. Phạm vi patch là `handoff/` và `CODEX_HANDOFF.md`, không có thay đổi `src/`, `configs/` hoặc các đường dẫn riêng bị cấm.
- Giới hạn: không có checkout Git đầy đủ của base trong workspace; reviewer xác nhận 328 đường dẫn thay đổi, **không tự xác nhận toàn cây 3.708 file** hoặc lịch sử commit trên máy Windows.

## 2. R32-F1 — ĐÓNG

Đã đọc bản đính chính tại `handoff/logs/review32/14_nguon_nhan_vong32.md`, `BDD100K_INVENTORY.md`, brief và `REVIEW33_CHI_TIET.md`.

Các tài liệu hiện phân biệt đúng:

- 465.512 cặp annotation khớp box và song ánh ID trong từng video.
- **183 annotation khác lớp**, thuộc **46 track, 40 video**.
- **50 ngoại lệ ignore** là phép kiểm riêng; không cộng với 183 thành tổng annotation khác nhau.
- Nguyên nhân UNKNOWN; không xác định native hay COCO đúng hơn.
- So với bài báo chỉ là tương đồng thống kê tổng hợp, không xác thực từng nhãn.

Reviewer tải lại đủ 1.400 JSON native train + 200 JSON native val từ HF revision `694197099692fe1254e93215e73e68ece8b1f41f`; **1.600/1.600 SHA256 khớp manifest**. File COCO val dùng lại từ dữ liệu người dùng đã cung cấp, SHA256 `074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d`.

Chạy hàm đối chiếu đã sửa trên toàn bộ native val và COCO này:

| Kiểm | Kết quả độc lập |
|:--|:--|
| Annotation ghép được | 465.512 |
| Annotation dư một phía | 0 |
| Box lệch | 0 |
| Xung đột ID hai chiều | 0 |
| Lệch lớp | 183 |
| Ngoại lệ ignore | 50 |
| iscrowd ngoài quy tắc đang kiểm | 0 |
| Khóa annotation / image_id COCO trùng | 0 |
| Trạng thái đối chiếu | **DIFFERENCES** |

Hai CSV sinh mới **trùng từng byte** với bằng chứng review `c4d64a1`. CSV Claude chạy lại trên PC cũng trùng các file đó.

Quan sát về hai COCO JSON do người đóng gói HF cung cấp được ghi đúng là quan sát nội bộ, không dùng chứng minh nguồn độc lập. Reviewer không tải lại hai COCO phụ này trong vòng hiện tại; phần đó được đọc từ bằng chứng bàn giao.

## 3. R32-F2 — ĐÓNG

Đã đọc toàn bộ bản sửa `kiem_nhan_ung_vien.py`, chạy nguyên script tái hiện Codex và regression mới.

| Kiểm reviewer tự chạy | Kết quả |
|:--|:--|
| `REPRO_REVIEW_c4d64a1.py` trên target | **exit 0, bug_count=0**; đối chứng sạch đạt |
| Thiếu root / thiếu train | CLI trả **2** |
| NaN / `labels={}` / ID trùng trong frame | CLI trả **1** |
| File COCO được yêu cầu nhưng thiếu | CLI trả **2** |
| COCO lệch box / đổi lớp / gộp hai native ID | CLI trả **3**; gộp ID được phát hiện |
| Regression mới trên target | **30/30 PASS**, exit 0 |
| Cùng regression trên bản c4d64a1 được lưu nguyên byte | **3/30 PASS**, exit 1 |
| Parser target trên 1.600 JSON native thật | Không lỗi schema, đúng số frame/annotation |

27 ca regression không đạt trên mã cũ gồm cả lỗi cũ và tính năng/trạng thái mới chưa tồn tại; không gọi đó là 27 lỗi độc lập.

Các yêu cầu chính đã được đáp ứng: chặn thiếu đầu vào trong phạm vi yêu cầu; kiểm tọa độ hữu hạn và loại boolean; phân biệt kiểu `labels`; bắt ID/frame/khóa COCO trùng; kiểm song ánh ID hai chiều theo video; trạng thái schema tách khỏi COCO.

Quy ước mã thoát cần đọc đầy đủ:

- **0:** các split được chọn đạt, COCO MATCH; hoặc COCO NOT_RUN vì người chạy đã chọn rõ phạm vi/bỏ đối chiếu.
- **1:** lỗi schema hoặc ảnh–nhãn thiếu/thừa.
- **2:** chưa kiểm đủ vì thiếu/sai đầu vào.
- **3:** schema đạt nhưng đối chiếu COCO có DIFFERENCES.

Vì vậy **exit 0 không luôn đồng nghĩa đã đối chiếu COCO**. JSON và docstring hiện thể hiện rõ ngoại lệ NOT_RUN; đây là hành vi được yêu cầu, không phải finding còn mở.

`labels=null`/thiếu khóa được đếm riêng. Chấp nhận về cấu trúc không chứng minh một frame là ảnh âm tính hợp lệ cho training. Với snapshot đang review, **cả train và val đều có 0 trường hợp này**, nên không có vướng mắc thực tế mới từ quy ước đó.

## 4. Bằng chứng nhãn thật và giới hạn kiểm

| Split | JSON | Frame | Annotation |
|:--|--:|--:|--:|
| Train | 1.400 | 278.079 | 2.886.916 |
| Val | 200 | 39.973 | 465.512 |

Reviewer gọi trực tiếp **`kiem_file()` và `doi_chieu_coco()` của target** trên dữ liệu thật, không dựng ảnh giả để tuyên bố kiểm ảnh–nhãn. Không thấy lỗi schema theo bộ kiểm, NaN/Inf, ID trùng cùng frame, frame trùng hoặc sai kiểu thuộc tính; kết quả COCO như mục 2.

**Không có ảnh train/val trên workspace để tự chạy toàn bộ CLI kiểm độ đầy đủ ảnh.** Lượt CLI trên PC với schema/độ đầy đủ DAT và exit 3, cùng `kiem-nguon` cách ly exit 0 / đường chính exit 2, là bằng chứng Claude bàn giao được đối chiếu về nội dung; không ghi là reviewer đã chạy lại trên máy Windows.

Không chạy full suite tracking/Nano vì delta không đổi đường chạy đó. Không có GPU training trong review này.

Giới hạn nguồn vẫn giữ:

- Nhãn là snapshot bên thứ ba đã ghim revision và hash, chưa xác thực bằng archive gốc khớp MD5 chính thức.
- Train chưa có một bản tham chiếu annotation độc lập theo từng nhãn.
- COCO val và native khác lớp/ignore; chưa xác định lịch sử hoặc bên nào đúng hơn.
- Hash khớp chứng minh đang dùng đúng snapshot, không tự chứng minh chất lượng hoặc nguồn phát hành.

## 5. Quyết định và bước tiếp theo

**Hai finding đủ điều kiện đóng.** Có thể ghi nhận review này và chuyển base review tới `4e71d32` sau khi lưu/kiểm đúng ZIP đã nghiệm thu; không dùng một HEAD khác làm base thay thế.

**Đề xuất phương án C đủ cơ sở về mặt kỹ thuật để người dùng chốt.** Việc nhập nhãn vẫn chờ người dùng đồng ý, đúng trạng thái bàn giao. Khi được duyệt:

1. Nhập nguyên byte **1.400 JSON train**; giữ bản cách ly và manifest nguồn.
2. Giữ nguyên COCO val đang dùng và các kết quả/GT đã khóa; không tự thay bằng native val hay COCO của mirror.
3. Kiểm hash sau nhập và chạy `kiem-nguon`.
4. Tiếp tục khóa split, xuất và smoke trên dữ liệu thật theo phạm vi đã duyệt; nếu có đầu vào hoặc tiêu chí cần quyết định thì trình trước khi mở rộng.

Review này không duyệt training dài, ngân sách epoch mới, thêm dataset/model hay tối ưu Nano. Nếu về sau có archive gốc, đối chiếu theo đường dẫn + hash từng file và ghi lịch sử; không âm thầm thay snapshot đã khóa.

## 6. Gói bằng chứng

`CODEX_REVIEW_4e71d32_EVIDENCE.zip` chứa report, log test độc lập, output script Codex, kiểm package, script kiểm nhãn thật, hash của 1.600 file native và CSV kết quả. Không chứa dataset, ảnh, weights hoặc archive nguồn.

Script `check_real_target.py` là harness reviewer gọi mã target và nguồn HF ghim; layout chạy được ghi trong README. Nó kiểm parser/đối chiếu, không thay cho kiểm ảnh thật trên PC.
