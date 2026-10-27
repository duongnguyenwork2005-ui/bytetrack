# BÁO CÁO TIẾN ĐỘ BYTETRACK — chuẩn bị báo cáo thầy ngày 17/10/2026

- **Thời điểm chốt:** 13/10/2026, 12:38:55, UTC+07:00 (Asia/Saigon).
- **Repo:** `D:\LABOTARY\bytetrack`; **branch:** `phase2-mixed-traffic`.
- **HEAD:** `117d62da027e8ce3e4c9f455d576a924b4d82bbb` (12/10/2026).
- **Thay đổi chưa commit trước khi viết báo cáo:** sửa `handoff/PLAN_DA_CHOT.md`, sửa `handoff/logs/review48/09_thiet_ke_phan_tich_deta_pf_ekf.md`, thêm `handoff/logs/review48/10_test11_k1_va_trang_thai_de_xuat.md`. Đây là hồ sơ R50 chưa nghiệm thu, ảnh hưởng cách diễn giải trạng thái; Git không ghi nhận thay đổi mã/cấu hình/kết quả. Không đồng nhất điều này với kiểm hash toàn bộ dữ liệu bị Git bỏ qua.
- **Phương pháp:** đọc AGENTS, mục 0 PLAN, plan/state/log gần nhất; đối chiếu CSV/JSON và biên bản. Mốc so sánh là báo cáo R48 `07_tong_hop_bao_cao_17_10.md`, bản đã được R49 đồng bộ. Phiên này chỉ đọc và tạo báo cáo mới; không chạy lại thí nghiệm, sửa code, commit/push hay mở automation.

## 1. TÓM TẮT HIỆN TẠI

**Đã đủ bằng chứng để trình bày tiến bộ chính:** R48A–E đã nghiệm thu lý thuyết, audit dữ liệu, giao thức/test PF, mô phỏng, 26 run tracking dev12 và kiểm pixel ảnh tô xám. R49 đã nghiệm thu việc đồng bộ hồ sơ. K1 cải thiện PF, nhưng chưa giúp PF vượt EKF trong các so sánh đã đo. Augmentation mới chỉ có thiết kế. [A–D]

**Automation đã dừng và đang tạm dừng theo người dùng, không phải đang chờ quota để tự chạy.** Run gần nhất: `bytetrack_r50_codex_20261012_0738`; mốc dở: `R50A_TEST11_THIET_KE_DETA_TRANG_THAI`. State giữ `phase=error`, 2/3 vòng, 3 lời gọi. Orchestrator kết thúc 12/10 lúc 13:49:37; yêu cầu tạm dừng ghi lúc 19:34:32. Các PID orchestrator/worker đã ghi không còn tồn tại khi kiểm tra. [G]

Backend thực tế trong plan là **Codex `gpt-6-astra`, effort `high`**, coder và reviewer ở hai phiên riêng; đây không phải kiểm chứng chéo hai mô hình. Quota gần nhất **được lưu** lúc 12/10 13:36:00: **đã dùng 21% cửa sổ 5 giờ, 60% tuần**; không phải quota hiện tại ngày 13/10. Chưa xác nhận Claude reset. Quy tắc giữ dự phòng 20% tuần còn hiệu lực; trước khi tiếp tục phải đọc quota mới, hoặc kiểm tra thủ công nếu không đọc được. [G]

**Còn chặn:** R50 chưa có approve sau lần sửa cuối; coder báo thiếu đầu vào nghiệm thu, orchestrator phân loại lỗi sandbox dù tiến trình coder exit 0. H2/H3 và nguyên nhân DetA thấp chưa được phân tích; P487415eb chỉ cho phép thiết kế. Không có quyền tự mở pha thực thi hoặc tự tiếp tục sau lệnh tạm dừng.

## 2. TIẾN ĐỘ THEO GÓP Ý CỦA THẦY

| Yêu cầu | Đã làm | Kết quả chính | Còn thiếu |
|:--|:--|:--|:--|
| Lý thuyết KF/EKF/PF, SIS/SIR, nguồn | R48A đã nghiệm thu phương trình, đơn vị, ESS/resampling và đối chiếu mã | Phân biệt KF–CV, EKF–CTRV, PF–CTRV; K1 là optimal proposal theo giả định Q/R chéo, không phải lý thuyết mới | Arulampalam 2002 chưa đối chiếu toàn văn; chú thích đơn vị trong một số file mã còn khác cách diễn giải tài liệu [B] |
| Phân bố, độ đủ và chất lượng dữ liệu | R48B/E đã nghiệm thu audit split, lớp, cỡ, quỹ đạo, nguồn và mẫu ảnh | Số video/ảnh khớp; 320/320 mẫu YOLO khớp nhãn; pixel tô xám 131 đạt/26 chưa kết luận/3 không đạt trên 160 ảnh | Chưa chứng minh dữ liệu đủ; BDD dev có phụ thuộc train; tầng hiếm nhỏ; chưa kiểm hết 18.600 ảnh tô xám [C] |
| Làm giàu/augmentation | Đã kiểm augmentation hiệu lực của hai lần fine-tune; đã thiết kế M5 | Mosaic, dịch/co giãn, HSV, lật ngang đã dùng; đề xuất đổi `scale` 0,5→0,9 và đường cong lượng dữ liệu | M5 chưa chạy, chưa có hiệu quả augmentation mới; cần duyệt huấn luyện/ngân sách [D] |
| Suy biến hạt, xe nhỏ cạnh xe lớn | Kiểm sở hữu hạt; mô phỏng và phân tầng sự kiện thật đã nghiệm thu | Mỗi track có hạt riêng: không có cơ chế xe lớn lấy hạt từ quỹ chung. K1 giảm nghèo hạt khi bám; chưa hết sụp ESS sau mất đo dài | H2 association và H3 likelihood dữ liệu thật chưa kiểm; 42 sự kiện car nhỏ không đại diện xe máy/xe đạp [B, D, E] |
| PF so với EKF khi cua/che khuất | Mô phỏng có liên kết biết trước và pipeline dev12 đã nghiệm thu | PF vẫn kém EKF; K1 thu hẹp khoảng cách. Nhóm cua chỉ 32 track/5 video; che khuất đủ điều kiện 184 sự kiện/12 video | Không có bằng chứng “PF tốt hơn khi cua”; K1 chưa được đánh giá cuối trên UA-DETRAC test/BDD trong đợt này [D] |
| Jetson Nano | **Hoãn** theo quyết định hiện hành | Chưa có kết quả tối ưu hoặc tốc độ trên Nano để báo cáo | Chỉ mở lại khi người dùng quyết định; không đổi ưu tiên trước 17/10 [A] |

## 3. PHẦN MỚI SAU BÁO CÁO R48

| Hạng mục | Trạng thái rõ ràng | Thay đổi so với file 07 |
|:--|:--|:--|
| Test 11 sau khi mã có K1 — P64dfb0d8 | **Đã chạy, chưa nghiệm thu R50** | JSON ngày 12/10 ghi 5/5 PASS, exit 0; fixture tổng hợp 40 frame, seed 0/1, chạy lại seed 0 trùng byte. **K1 tắt trong YAML**: không được gọi là kiểm chứng runner với K1 bật. Sáu hash mã hiện tại khớp bằng chứng [F] |
| Kiểm kê và thiết kế DetA — P487415eb | **Đã duyệt thiết kế; đã viết, chưa nghiệm thu** | Có kiểm kê đầu vào lúc 04:06 ngày 12/10 và bản 09 sửa sau review. Xác định thiếu log nội bộ KC48 và thiếu predictive likelihood/NIS đầy đủ [E] |
| Thực thi H2/H3, phân rã nguyên nhân DetA | **Chưa làm; chưa duyệt thực thi** | Chưa tính bộ M1–M8 mới, chưa kiểm A0 tương đương log, chưa instrumentation/tracking mới. Chênh lệch DetA trong bảng R48 là kết quả cũ, không phải đã giải thích nguyên nhân |
| Đồng bộ R49, lưu Git | **Đã nghiệm thu R49; đã commit** | R49 approve; `a7a6961` lưu tài liệu, `117d62d` snapshot dự án. File 07 hiện đã chứa sửa R49; không có số thực nghiệm mới do đồng bộ/commit. Ba tài liệu R50 hiện vẫn chưa commit [A, H] |
| Chuyển Claude→Codex, sửa hồ sơ R50 | **Đã thực hiện chuyển backend; hồ sơ chưa nghiệm thu, lượt đã dừng** | Lượt Claude cũ và checkpoint được bảo toàn. Review Codex yêu cầu sửa F1–F6; đã có bản sửa nhưng chưa có review tiếp theo approve [G] |
| Nghiên cứu/thay đổi kết quả sau đó | **Chưa ghi nhận trong hồ sơ đã đối chiếu; việc mở rộng hoãn** | Không thấy run mới hơn R50 trong hai kho run đã kiểm. Sửa chẩn đoán ESS, bootstrap, ghép phân tầng và kiểm PF mốc 2 đã nằm trong file 07, không tính là kết quả mới sau R48 |

## 4. KẾT QUẢ CÓ THỂ ĐƯA VÀO SLIDE

**Dữ liệu và giới hạn mẫu.** BDD: train-fit **1.261 video/50.487 ảnh**, dev nội bộ **139/5.559**, val **200/39.973**; train/dev lấy mỗi 5 frame có nhãn. UA-DETRAC: train 60 video, dev12 là tập con train, test 40. BDD có **70/139 video dev chung tiền tố ghi hình với train-fit**, một cặp gần trùng xuyên hai split; chưa thể coi dev độc lập. Kiểm pixel chỉ kết luận trên mẫu 160 ảnh, không quy chắc 3 ảnh lỗi cho JPEG. [C]

**Bảng chính nên dùng: UA-DETRAC dev12, không phải test.** 12 video, 26 run, điều kiện A `conf_min=0,25`, cache YOLOv8n–COCO `conf0.05_all`, 25 fps, `elapsed_mode=skip`, `max_det=300`; PF seed **0/1/2**, EKF tất định. Tính COMBINED từng seed rồi trung bình. Chỉ số dưới đây ở thang **0–1**; IDSW là tổng số lần đổi ID, trung bình 3 seed và làm tròn. HOTA/AssA/DetA trung bình qua 19 ngưỡng alpha; Identity/CLEAR dùng IoU 0,5. [D]

| Biến thể | HOTA ↑ | AssA ↑ | DetA ↑ | IDF1 ↑ | IDSW ↓ |
|:--|--:|--:|--:|--:|--:|
| EKF | 0,5925 | 0,6838 | 0,5207 | 0,7245 | 472 |
| PF100 | 0,4933 | 0,5383 | 0,4692 | 0,5509 | 2.069 |
| PF100 + K1 | 0,5377 | 0,6203 | 0,4802 | 0,6334 | 1.094 |
| PF100 + K1 + K2 | 0,5496 | 0,6272 | 0,4934 | 0,6505 | 980 |
| PF500 + K1 | 0,5548 | 0,6469 | 0,4883 | 0,6583 | 836 |

**Kết luận được phép:** K1 tăng HOTA của PF100 **+0,0444**, CI 95% **[+0,0184; +0,0867]**. PF500+K1 vẫn kém EKF **−0,0378**, CI **[−0,0650; −0,0207]**. CI bootstrap ghép cặp theo 12 video, 5.000 mẫu, seed bootstrap 0; không dùng min–max giữa seed thay CI. Chưa biết nguyên nhân chênh DetA; cùng detector không có nghĩa DetA tracking phải bằng nhau.

**Cua/che khuất/xe nhỏ:** hiệu tỉ lệ giữ ID **PF100+K1 − EKF**, thang 0–1; CI 95% bootstrap theo video, 5.000 mẫu, seed 0. PF trung bình 3 seed trước tổng hợp tầng. [D]

| Tầng và mẫu số | Hiệu [CI 95%] | Được nói |
|:--|:--|:--|
| Cua: 32 track/5 video; mỗi track = frame giữ ID chính/frame GT | −0,040 [−0,050; −0,019] | PF vẫn kém; “cua” là hướng quỹ đạo trên ảnh |
| Che khuất: 184 sự kiện đủ điều kiện/12 video/137 track; ID trước = ID sau | −0,098 [−0,169; −0,050] | Chưa chứng minh lợi thế PF khi che khuất |
| Car nhỏ cạnh xe lớn: 42 sự kiện/7 video/29 track nhỏ; ID trước = ID sau | −0,024 [−0,037; 0,000] | Chưa phân biệt được; CI chạm 0 không chứng minh tương đương |

**Minh họa mô phỏng, tách khỏi dữ liệu thật:** ô `cua_deu`, mất đo L=20 bước, κ=1, h₀=24 px; nhiễu d=0–29, PF seed 0/1/2 (90 quỹ đạo/biến thể PF; EKF 30). Sai số tâm dự đoán khi đo trở lại, chuẩn hóa theo chiều cao box: EKF **0,23h**, PF100 **5,49h**, PF100+K1 **2,64h**. PF100 tái khởi tạo 43/90, K1 3/90; ESS hợp lệ sau mất đo vẫn khoảng 1% N, loại tái khởi tạo khỏi mẫu số ESS. Đây là một ô mô tả, không có CI kèm các số minh họa này. Không suy rộng mô phỏng sang dữ liệu thật, car nhỏ sang motorcycle/bicycle, hoặc thời gian harness Python sang FPS/Nano. [D]

## 5. VIỆC TIẾP THEO

Đề xuất ưu tiên trước 17/10; **báo cáo này không cấp quyền thực thi và không hủy lệnh tạm dừng**.

| Ưu tiên / câu hỏi | Đầu ra cụ thể | Tracking/huấn luyện? | Phê duyệt và chi phí |
|:--|:--|:--|:--|
| 1. R50 đã đáp ứng F1–F6 chưa? | Reviewer riêng đọc diff, bản đầu mốc và bằng chứng; biên bản approve/revise, trạng thái thống nhất | Không; không làm lại test 11 đã chạy | Nằm trong phạm vi R50 đã duyệt, nhưng cần người dùng cho tiếp tục sau pause; chưa có dự toán quota đáng tin |
| 2. Slide trả lời đúng góp ý thầy chưa? | Hoàn thiện từ đề cương 08: lý thuyết → dữ liệu → PF/K1 → giới hạn → quyết định cần xin | Không | Nằm trong mục tiêu báo cáo đã chốt; chỉnh slide là tác vụ tiếp theo, chưa làm trong phiên này; chưa định lượng thời gian |
| 3. Các khẳng định lý thuyết đã đủ nguồn chưa? | Bảng đối chiếu toàn văn Arulampalam và tài liệu/mã; giữ nhãn chưa xác minh nếu thiếu nguồn | Không | Theo mục tiêu M1; việc tra cứu bổ sung cần đưa vào phạm vi lượt tiếp theo, không tự sửa mã |
| 4. DetA thấp vì thiếu đầu ra hay định vị/ghép? | Nếu được duyệt: phân tích pha A trên kết quả cũ; pha B chỉ sau kiểm A0 và nguồn/ngữ nghĩa log | A/B không cần tracking hay train; C/D cần tracking mới, chưa mở | **Cần quyết định mới cho A/B**. Bản 09 ước tính A+B: 5–7 giờ viết/test, khoảng 30–40 phút CPU cộng A0; chưa đo, chưa được nghiệm thu, không phải cam kết |
| 5. Có lỗi đánh giá gây nhiễu diễn giải dữ liệu không? | Đặc tả kiểm 572 ảnh dev12 không có `<frame>` và quy tắc lớp/vùng bỏ qua trước đề xuất train mới | Đọc evaluator trước; có thể cần chấm lại sau duyệt, không mặc nhiên tracking/train | **Cần duyệt riêng**; chưa đủ căn cứ ước lượng. Giữ M5 huấn luyện, K1 trên tập cuối và tối ưu Nano ở trạng thái chưa chạy/hoãn |

---

## PHỤ LỤC — ĐƯỜNG DẪN VÀ BẰNG CHỨNG

Các đường dẫn tương đối dưới đây tính từ **`D:\LABOTARY\bytetrack\`**. Đây là chỉ mục để ChatGPT/người đọc truy lại, không phải yêu cầu chạy lệnh. Số liệu được đọc từ kết quả có sẵn; phiên lập báo cáo không tái thực nghiệm và không thay thế cổng nghiệm thu.

**[A] Trạng thái và mốc so sánh**

- `AGENTS.md`; `handoff/PLAN_DA_CHOT.md`, mục 0.
- `handoff/logs/review48/07_tong_hop_bao_cao_17_10.md`: bản R48 đã sửa chú thích/nguồn qua R49. Câu “test 11 chưa chạy lại” tại mục 8 là lịch sử, được bổ sung bởi bằng chứng R50; không sửa file cũ trong nhiệm vụ này.
- `handoff/logs/review48/08_de_cuong_slide.md`: khung slide có sẵn, chưa được hoàn thiện trong phiên báo cáo này.
- Nền dự án trước R48: PLAN dẫn kết quả tracking UA-DETRAC test 40 video, BDD transfer và đánh giá detector R47. Báo cáo này tập trung phần chuẩn bị 17/10; không tái nghiệm thu hoặc tái tính các mốc cũ.

**[B] Lý thuyết, mã và nghiệm thu R48A**

- `handoff/logs/review48/01_ly_thuyet_kf_ekf_pf.md` và `handoff/logs/review48/02_doi_chieu_code_pf.md`.
- `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_002.json`: `approve`, issues rỗng.
- Theo hồ sơ lý thuyết: ByteTrack đã đối chiếu toàn văn; Arulampalam et al. 2002 mới có thông tin nguồn, chưa đối chiếu toàn văn. K2 roughening đã tồn tại trong mã, không trình bày như thuật toán mới.

**[C] Audit dữ liệu**

- `handoff/logs/review48/03_audit_du_lieu.md`; `results/phase2/kc48_audit/moc_doi_chieu.csv`; `augmentation_hieu_luc.csv` cùng thư mục.
- `results/phase2/kc48_audit/pixel_to_xam_tom_tat.json`, `pixel_to_xam_tung_anh.csv`, `tieu_chi_pixel_to_xam.json`.
- R48B: `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_007.json` approve. R48E: `handoff/logs/review48/nghiem_thu_r48cde/0015_review_answer.json` approve sau revise `0013`.
- Audit ghi 458.183 ảnh qua hash/header/SOI/EOI không lỗi, 6.783 ảnh giải mã mẫu không lỗi; không phải đã giải mã toàn bộ. Nhãn BDD train 1.400/1.400 JSON khớp manifest nhập, chưa xác thực bằng archive gốc. Mẫu YOLO 320 ảnh: 8 tầng × 40, seed 48; pixel là 160 ảnh tô xám thuộc mẫu này.
- 88 sự kiện xe nhỏ/65 track/9 video là tập audit; sau điều kiện trước–sau chỉ còn 42/29/7 dùng chấm giữ ID. 717 sự kiện che khuất ban đầu loại 369 thiếu GT trước và 164 thiếu GT sau, còn 184. Không dùng các mẫu số này thay nhau.

**[D] Giao thức, số liệu slide, augmentation và nghiệm thu R48C/D**

- `handoff/logs/review48/04_giao_thuc_kiem_chung_pf.md`; `06_thiet_ke_augmentation.md`; báo cáo 07 mục 5–8, 11–12.
- `results/phase2/kc48_pf_dev/experiment.yaml`, `runs.csv`; manifest `results/phase2/manifests/detrac_dev.json`.
- Bảng chính và CI: `results/phase2/kc48_pf_dev/phan_tich/bang_chinh.csv`, `doi_chieu_bootstrap.csv`; phân tầng: `phan_tang.csv`, `giu_id_theo_track.csv` cùng thư mục.
- Sự kiện: `results/phase2/kc48_pf_dev/su_kien/bang_su_kien.csv`, `co_mau_va_loai.csv`, `kiem_su_kien_xe_nho.json`.
- Mô phỏng: `results/phase2/kc48_mo_phong/phan_tich/tom_tat_o.csv`, `doi_chieu.csv`, `mau_so_chan_doan.csv`; `chon.json` ở thư mục cha. 8 kịch bản gồm 4 lệch mô hình, L∈{0,5,10,20}, κ∈{0,5;1;2}, h₀∈{24;72}px. Seed tinh chỉnh d=1000–1009 tách d đánh giá 0–29. Khi suy luận, bootstrap ghép cặp theo d, lấy trung bình 3 seed PF trong mỗi d; 5.000 mẫu, seed 0.
- Hình có sẵn: `results/phase2/kc48_mo_phong/hinh_hat_so_huu.png`, `phan_tich/hinh_c1_e_r.png`; biểu đồ audit ở `results/phase2/kc48_audit/bieu_do_phan_bo_lop_kich_thuoc.png`, `bieu_do_quy_dao_cua.png`. Hình hạt sở hữu là mô phỏng minh họa.
- Nghiệm thu: `handoff/logs/review48/nghiem_thu_r48cde/0002_review_answer.json` (C), `0009_review_answer.json` (D, sau revise `0006`). Review D đã đọc CSV và log, không chạy lại thí nghiệm; lỗi dẫn nguồn nhỏ đã được R49 sửa.
- Các sửa R48D đã có trong file 07: phạt cặp IoU dưới ngưỡng trước Hungarian; loại tái khởi tạo khỏi ESS/resampling; COMBINED trước trung bình seed. Đã chạy bổ sung 14.073 quỹ đạo mô phỏng để sửa chẩn đoán, không phải nghiên cứu mới sau báo cáo 07; không làm lại trong phiên này.
- M5 chỉ thiết kế: `scale` 0,5→0,9, tập con video lồng nhau 25/50/100%, kiểm soát ngân sách tối ưu. Không có số huấn luyện mới để khẳng định augmentation cải thiện kết quả.

**[E] Thiết kế DetA/H2/H3**

- `handoff/logs/review48/09_thiet_ke_phan_tich_deta_pf_ekf.md` (bản chưa commit/chưa nghiệm thu); `results/phase2/kc48_pf_dev/thiet_ke_deta/kiem_ke.json` (12/10 04:06:00 +07, 5,61 giây).
- Kiểm kê xác nhận 26 run KC48 có MOT/timing/raw_fields, thiếu `assoc_log`/`pf_diag`; log mốc 2 có một phần cho base, không đủ kết luận cho K1. Không có predictive likelihood/NIS đầy đủ của cả EKF/PF.
- Pha A/B: phân tích dữ liệu cũ, vẫn cần duyệt thực thi. A0 phải kiểm cả tính tương đương đầu ra và nguồn/ngữ nghĩa log; trùng byte MOT không chứng minh prior/association/trọng số trùng. Pha C chạy lại có log; pha D instrumentation likelihood, đều cần duyệt riêng. Không dùng phần dư GT như NIS đã hiệu chuẩn hoặc cận trên sai số detector.
- Các chi phí trong bản 09 là ước tính chưa đo/chưa nghiệm thu, không quy đổi sang token hay quota thật.

**[F] Test 11 và bản R50**

- `results/phase2/kc48_pf_code/test11_k1/ket_qua.json`: bắt đầu 12/10 04:01:02 +07, 6,72 giây test/24,77 giây wrapper; 5/5 PASS, exit 0. N=100, `pf_roughening=0`, YAML thiếu `pf_optimal_proposal` nên K1 mặc định tắt.
- HEAD khi chạy `a7a6961`; K1 lúc đó nằm trong cây làm việc chưa commit, không được suy cấu hình/mã chỉ từ HEAD. Phiên lập báo cáo đã đối chiếu SHA256 sáu file mã với `sha256_cay_lam_viec` trong JSON: đều khớp. Đây là kiểm đọc hash, không chạy test.
- `handoff/logs/review48/10_test11_k1_va_trang_thai_de_xuat.md`: bản chưa nghiệm thu; kết quả fixture không xác nhận toàn pipeline `run_experiment.py` trên dev12 hoặc runner với K1 bật.
- Đề xuất trong bản R50: P64dfb0d8 đã chạy; P487415eb thiết kế; P2542b6c1 đã commit. P9a6c2078 được bản nháp ghi đóng vì trùng việc R49, nhưng bản cập nhật R50 vẫn chờ review. Pee184526/Pf350787b/P3c58df9b/Paa38fde5 giữ HOÃN; không suy thành quyền thực thi.

**[G] Automation: trạng thái thực tế, log và quota**

- Gốc run hiện tại: `D:\LABOTARY\automation\runs_local\bytetrack_r50_codex_20261012_0738\`.
- Đã đọc `plan.json`, các trường trạng thái/lịch sử trong `state.json`, `orchestrator.json`, `quota_latest.json`; `reports/review_001.json`; `calls/0003_coder/{answer.json,classification.json,process.json}`. Log orchestrator: `logs/orchestrator_20261012_133517.log`.
- Review cuối là **revise F1–F6**; thư mục reports chỉ có `review_001.json` và `round_001.md`. Lần coder 0003 có exit 0 nhưng answer là `blocked`, classification là `error/sandbox`; không có kết quả review approve tiếp theo. Orchestrator exit 2. Việc đã ghi được ba tài liệu không đủ để đóng mốc; cũng không suy mọi thao tác đọc của coder đều thất bại chỉ từ nhãn sandbox.
- `state.active` còn descriptor coder PID 20612; đó không phải bằng chứng đang chạy. PID này và orchestrator PID 11104 không tồn tại lúc kiểm. State history ghi `user_pause_requested` ngày 12/10 19:34:32, giữ phase error.
- `quota_latest.json`: nguồn `account/rateLimits/read`, quan sát **12/10/2026 13:36:00 +07**, dùng 21%/5 giờ và 60%/tuần. Không gọi API quota mới trong nhiệm vụ đọc hồ sơ; không suy số còn lại hiện tại từ snapshot này. Trước khi resume cần quan sát mới; nếu không đọc được phải kiểm thủ công, không bật nạp tiền/ngoài gói.
- Lượt Claude cũ: `D:\LABOTARY\automation\runs\bytetrack_r50_test11_thiet_ke_deta_20261012_0356\`; đã stop, trạng thái lưu quota_wait. Checkpoint: `D:\LABOTARY\automation\checkpoints\r50_codex_switch_20261012_073357\`. Claude từng báo hết tuần; reset 16/10 05:00 chỉ là dự kiến. Cơ chế chờ reset không hủy yêu cầu tạm dừng; quay lại Claude-first chỉ sau xác nhận reset ở ranh giới mốc an toàn.
- Hai kho `runs`/`runs_local` không có thư mục run mới hơn R50 trong lần kiểm này. Đây là phạm vi hồ sơ đã kiểm, không phải giám sát mọi công việc có thể chạy ngoài orchestrator.

**[H] R49 và Git**

- `handoff/logs/review48/nghiem_thu_r49/0004_review_answer.json`: approve, issues rỗng; review tĩnh xác nhận đồng bộ và giữ nguyên bảng số liệu. `MANIFEST.md` cùng thư mục ghi nguồn/hash biên bản.
- `a7a6961` ngày 12/10 03:54:41 +07: commit tài liệu R48/R49; `117d62d` lúc 07:30:35: snapshot dự án. Push trước đó được ghi trong mục 0 PLAN; phiên báo cáo không push và không kiểm remote qua mạng. Lưu Git không biến hồ sơ R50 thành đã nghiệm thu.
