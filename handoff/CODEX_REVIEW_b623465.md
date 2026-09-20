**Review độc lập — `ca93c1e` → `b623465`**

**Kết luận: PASS WITH LIMITATIONS. Đóng ba finding kỹ thuật P1/P2/P3 của review `ca93c1e` trong phạm vi đường chạy và dữ liệu đã kiểm. Không phát hiện lỗi chặn mới.** Các yêu cầu chính về diễn giải đã được xử lý; còn một dòng tóm tắt cũ trong handoff, có thể đồng bộ trong lần cập nhật tài liệu tiếp theo.

Gói đã đọc: `review_delta_latest.zip`, 1.360.258 byte, 46 thành viên ZIP. Base `ca93c1e20be60512ca2d9579954253a4dbea99af`; target `b6234653f616e8059825b59b4da70262e1f98dde`. SHA256 ZIP: `5cf9133be5c0e96ed2abb3a42d56e2ea483b0f29c047aee7017cb493faf7b9bc`.

**1. Kết quả chạy độc lập**

| Kiểm tra | Kết quả thực tế tại môi trường review |
|---|---|
| `test_eval_guard.py` | **11/11 PASS**, exit 0 |
| `test_report_stats.py` | **8/8 PASS**, exit 0 |
| `test_evaluator_choice.py` | **7/7 PASS**, exit 0 |
| Probe bổ sung: GT rỗng qua evaluator thật | Exit 0, `GT_dets=0`, `FP=3`; GT rỗng được chấp nhận |
| Probe bổ sung: xóa GT rồi gọi CLI thật | Lần đầu exit 0; sau khi xóa GT exit **1**, có `ThieuDauVaoDanhGia`, không in `TAI SU DUNG` |

Các lệnh test được chạy trên bản sao review, với `BYTETRACK_BACKEND=np`, Python 3.12 và TrackEval 1.3.0. Những kiểm tra này không cần detector/GPU hay ảnh BDD thật. Không sửa source để làm test PASS.

**P1 — guard GT: đã đóng.** Khi GT thay đổi tại cùng đường dẫn, runner chạy lại và đổi metric đúng như fixture trước đây: `GT_dets 1→2`, `IDF1 0,6666667→0,8`, `HOTA 0,7071068→0,6832349`. Không đổi đầu vào thì tái sử dụng. Đổi `seqLength` làm mất hiệu lực evaluation cũ; thiếu GT/seqinfo báo lỗi trước khi xét tái dùng. Fingerprint đã chứa GT, seqinfo, timeline, nguồn segment cho occlusion và module `run_state.py`.

Giới hạn phép kiểm occlusion: đã xác nhận thay đổi GT/CSV nguồn làm fingerprint đổi và đối chiếu cách phân giải đường dẫn với `occlusion_eval.py`; không chạy lại thí nghiệm occlusion đầy đủ trong đợt này.

**P2 — helper báo cáo: đã đóng.** Đếm detection qua `open_cache(...).frame(i)`, giới hạn sequence bằng manifest và áp `max_det` từng frame. Fixture có video ngoài manifest và frame vượt giới hạn nay khớp đầu vào tracker. Sidecar thiếu `seconds` hoặc thiếu hẳn được báo riêng; báo cáo từ chối đưa ra tổng đầy đủ khi thiếu số đo.

**P3 — test evaluator: đã đóng.** Test mới gọi `stage_eval()` thật, đọc `eval_meta.json` và kiểm metric. Trên cùng fixture, lựa chọn tự động `mot_crowd` cho FP=1, ép `mot` cho FP=2. Fixture GT hợp lệ chồng vùng ignore giữ đủ hai GT, FN=0, FP=0. Điều này bảo vệ hành vi runner thay vì chỉ kiểm biểu thức chọn lớp.

**2. Evaluation và provenance**

Tôi dùng GT đã tái tạo từ file nhãn thật ở vòng trước để tính lại fingerprint bằng mã mới, rồi đối chiếu toàn bộ 12 `eval_meta.json`:

- **12/12 fingerprint GT đầy đủ khớp**, gồm hash GT, hash seqinfo và hash timeline; mỗi run có đúng 200 sequence, không thiếu GT/seqinfo.
- `gt_per_video` khớp **200/200 video** với bản khóa. `seqmap_used=False` phù hợp với đường chạy truyền danh sách video trực tiếp.
- Các trường cấu hình, cache, danh sách sequence và evaluator giữ nguyên so với metadata của `ca93c1e`.
- Cả 12 metadata có thời điểm evaluation mới, từ `06:42:59` đến `06:48:36` ngày `2026-09-20`, muộn hơn lượt đánh giá cũ theo hồ sơ được cung cấp.
- **20/20 module được liệt kê trong fingerprint** khớp source bàn giao sau khi đối chiếu LF/CRLF; không tìm thấy module lệch trong 12 metadata.

**Về 2.400 file tracking:** bảng `tracking_hash_truoc.json` đã sử dụng quy ước LF mới. Hash tổng hợp từ toàn bộ SHA256 trước chạy khớp fingerprint tracking sau chạy ở **12/12 run**, tương ứng **2.400/2.400 file**. Đây là bằng chứng nội dung tracking giữ nguyên **sau chuẩn hóa LF**. Không gọi phép đối chiếu này là so byte thô, vì CRLF và LF được coi là cùng nội dung theo quy ước mới.

Quy ước chuẩn hóa mới trong `sha256_file()` áp dụng cho các đuôi **`.txt`, `.ini`, `.csv`**. Các đuôi khác như `.py`, `.json`, `.yaml` không tự động được chuẩn hóa bởi hàm đó. Khi mô tả khả năng ổn định giữa Windows/Nano, cần nêu đúng phạm vi này.

**Về metric:** delta không thay các file metric/`raw_fields.json`/bảng bootstrap chính của 12 run; chúng được kế thừa từ base đã kiểm ở vòng trước. Log evaluation mới ghi đủ 12 lượt, exit 0; tôi đối chiếu **48 giá trị in trong log** (HOTA, IDF1, AssA, IDSW cho từng run), tất cả khớp base ở độ chính xác được in. Log aggregate cũng kết thúc exit 0.

Do đó, hồ sơ hiện có nhất quán với kết luận: lỗi guard trước đây có thật nhưng việc sửa và đánh giá lại **không làm thay đổi các kết luận BDD đã bàn giao**. Không có cơ sở yêu cầu chạy lại detector/tracker hoặc UA-DETRAC A/B/C/GĐ1 từ review này.

**3. Gói delta và bản lưu `_PRE_*`**

| Đối chiếu | Kết quả |
|---|---:|
| Payload vật lý được manifest khai báo | **40/40 khớp SHA256** |
| Đường dẫn thay đổi trong patch | **41**; gồm một log dedup vẫn có thay đổi trong patch |
| Patch reverse/forward và hash preimage/target | Thành công trong phạm vi gói |
| File riêng bị cấm trong ZIP/đường dẫn patch | **0** |
| File `_PRE_P1_ca93c1e` bị loại khỏi gói | **69/69** khớp byte với file tương ứng đã có ở base |

Vì 69 file archive chỉ là bản sao nội dung cũ, việc loại chúng khỏi delta lần này hợp lý và không làm thiếu thay đổi cần review. `tests_latest.log` trùng nội dung với log theo ngày, vẫn được thể hiện trong patch dù chỉ lưu một payload.

Một phụ thuộc `make_review_delta.py` trong bản sao base cục bộ của môi trường review còn là phiên bản cũ hơn. Tôi tái dựng preimage từ patch rồi đối chiếu đúng hash base được manifest cung cấp, sau đó áp lại và kiểm hash target. Vì không có checkout Git đầy đủ trên máy Windows, không mô tả đây là kiểm chứng toàn bộ Git tree độc lập.

**4. Diễn giải và ghi chú tài liệu**

Nguồn sinh báo cáo và các báo cáo mới đã:

- Gọi tổng detector + tracker là **ước tính cộng hai pha đo riêng**, bỏ tuyên bố cận trên và bảo đảm tốc độ triển khai.
- Nêu khoảng FPS suy ra và mức chênh theo số liệu, thay câu chung “gần như không đổi FPS”.
- Ghi SD PF riêng theo metric và giới hạn kết luận trong implementation/cấu hình/ba seed đã thử.
- Nêu rõ ngoại lệ PF giữa trung bình metric và tái tổ hợp thống kê.
- Đính chính DETRAC c10/c25, bỏ quy kết nguyên nhân CI hẹp chỉ do cỡ mẫu.
- Rút lại câu FP ngoài timeline tự biến mất; mô tả đúng khả năng TrackEval từ chối timestep ngoài phạm vi.

**Ghi chú nhỏ, không chặn:** trong bảng trạng thái của `CODEX_HANDOFF.md`, hàng mục 6 vẫn còn câu “12 lượt × 2 điều kiện” và “SD theo seed 0,00005–0,00095” chưa ghi rõ metric. Đồng bộ thành **6 cấu hình/seed × 2 điều kiện = 12 run tổng cộng** và **SD HOTA** như brief/phụ lục đã sửa. Có thể gộp chỉnh sửa này vào lần cập nhật tài liệu kế tiếp; không cần một vòng thí nghiệm hay review kỹ thuật riêng.

**5. Giới hạn của kết luận PASS**

- Tôi chạy độc lập ba suite nêu ở mục 1 và các probe bổ sung. **28/28 suite PASS** là kết quả tôi đối chiếu trong log/CSV do tác giả bàn giao, không phải toàn bộ suite đã được chạy lại ở môi trường review.
- Không có ảnh/cache/tracking thật trong delta để chạy lại toàn pipeline. Không tự đo lại detector 22,9 ms/frame hoặc số detection cực đại 25/59. Helper đã được kiểm bằng fixture; các số trên dữ liệu thật được ghi nhận từ bàn giao.
- Không có bản `run_meta.json` sau chạy để tự so byte với 12 hash trước chạy. Phạm vi xác minh tracking độc lập là các hash chuẩn LF nói trên và đường gọi `--stage eval` không đi vào `stage_track`.
- Kiểm chứng số học đầy đủ 96 metric tổng hợp, 19.200 giá trị theo video và 100 dòng bootstrap thuộc review `ca93c1e`; không chạy lại các phép tính không thay đổi chỉ để tăng số lượt kiểm tra trong vòng này.
- Kết quả vẫn giới hạn ở protocol gộp phương tiện trên nhãn BDD COCO-video chuyển đổi. Chưa có bằng chứng mới cho phục hồi che khuất trên BDD, metric riêng xe máy hoặc tốc độ Jetson Nano. Chưa có split train/dev phù hợp để tiến hành tuning BDD theo kế hoạch.

**Có thể chốt đợt sửa và tiếp tục mốc tiếp theo.** Giữ nguyên cấu hình/khóa và kết quả nghiên cứu hiện có. Snapshot `b623465` đã được đọc; có thể dùng làm base cho lần bàn giao delta sau, đồng thời lưu báo cáo này và gói tương ứng.

Review chỉ tạo bản sao phục vụ kiểm tra, fixture/probe tạm và báo cáo; không sửa source dự án gốc, index, commit hay ZIP đính kèm.
