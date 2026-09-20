**Review độc lập — target `ca93c1e`**

**Kết luận: REQUEST CHANGES đối với runner và công cụ báo cáo. Số liệu tracking chính và bootstrap đã vượt qua kiểm tra độc lập.** Không có bằng chứng trong đợt review này buộc chạy lại tracking UA-DETRAC hoặc BDD100K. Không mở thêm tuning/ablation.

Đã đọc gói `review_delta_latest.zip`, base `a8ecaf6`, target `ca93c1e20be60512ca2d9579954253a4dbea99af`. Kích thước 5.710.502 byte; SHA256 `0cd6f46d318255058cddf7b656a3d28d02c453db435f4c84a1c79a877963fc22`.

**1. [P1] Đổi ground truth nhưng evaluation vẫn tái sử dụng kết quả cũ**

Vị trí: `src/run_fingerprint.py:110`, `src/run_experiment.py:242`.

`run_fingerprint()` có tracking, cấu hình, cache, danh sách video, code và evaluator nhưng không có nội dung GT hoặc `seqinfo.ini`. Hash GT trong bản khóa không được `stage_eval()` dùng để quyết định tái sử dụng. `occl_fingerprint()` cũng chưa chứa nguồn annotation của bước đánh giá che khuất.

Tôi đã tái hiện qua **`stage_eval()` thật + TrackEval 1.3.0**, không mock metric:

- Một frame có GT hợp lệ `(100,100,50,50)` và vùng crowd `(400,100,80,80)`.
- Tracking có một detection khớp GT, một detection `(410,110,60,60)` nằm trong crowd, một FP thông thường `(800,400,50,50)`.
- Chạy lần đầu bằng `mot_crowd`.
- Chỉ đổi dòng crowd từ `conf=0,class=13` thành GT hợp lệ `conf=1,class=1`; giữ nguyên tracking và các đầu vào khác.
- Chạy lại bình thường, rồi chạy lại với `force=True`.

| Kết quả | GT_dets | IDF1 | HOTA |
|---|---:|---:|---:|
| Trước khi đổi GT | 1 | 0.6666667 | 0.7071068 |
| Đổi GT, chạy bình thường | **1 — metric cũ** | **0.6666667** | **0.7071068** |
| Đổi GT, ép evaluation chạy lại | **2** | **0.8000000** | **0.6832349** |

Cả ba lời gọi exit 0. Lần thứ hai in `TAI SU DUNG`; `eval_meta.json` không đổi dù hash GT đã đổi.

**Cách sửa và điều kiện đóng:** đưa nội dung `gt.txt`, `seqinfo.ini` và những đầu vào thực tế quyết định tập đánh giá vào fingerprint. Với occlusion, hash nguồn annotation thực sự được đọc. GT thiếu phải báo lỗi, không tái dùng metric trước khi thử đọc GT. Test phải chạy qua `stage_eval()`: không đổi thì tái dùng; đổi GT tại cùng đường dẫn thì tự đánh giá lại và metric thay đổi; đổi độ dài timeline cũng phải vô hiệu kết quả cũ. Không bổ sung hash hồi tố vào metadata của evaluation cũ.

Khi chỉnh danh sách module, bổ sung `run_state.py` đang được cả CLI và runner dùng nhưng còn thiếu trong `CODE_FILES`. Việc đổi commit hiện vẫn làm fingerprint đổi; phần thiếu này liên quan các sửa đổi chưa commit của module đó.

**Phạm vi ảnh hưởng:** lỗi có thật trong cơ chế tái sử dụng; chưa chứng minh metric BDD vừa bàn giao bị lỗi này. Tôi đã tái tạo GT từ nhãn thật và khớp hash bản khóa ở **200/200 video**. Sau khi sửa, có thể đánh giá lại **chỉ 12 run BDD đã có tracking** để ghi bằng chứng GT mới, rồi so metric với bản hiện tại. Giữ kết quả cũ để đối chiếu; không cần chạy detector hoặc tracker chỉ để sửa metadata evaluation.

**2. [P2] Thống kê detection và thời gian không theo đúng manifest/đầu vào tracker**

Vị trí: `src/report_bdd_transfer.py:174` và `src/report_bdd_transfer.py:222`.

`_det_counts()` quét mọi NPZ trong cache và chỉ lọc lớp/confidence. Nó không áp dụng `max_det` theo từng frame như `VideoDetections.frame()`, cũng không giới hạn theo manifest thí nghiệm. `_detector_seconds()` tương tự quét mọi JSON trong cache.

Phép tái hiện độc lập:

| Thiết lập | Đầu vào tracker thật | Công cụ báo cáo |
|---|---:|---:|
| Manifest chọn một video, một frame có 3 detection, `max_det=1` | 1 detection | **3 detection** |
| Thêm một video ngoài manifest, có 2 detection | Vẫn 1 detection | **5 detection** |
| Sidecar video được chọn ghi 1 giây; video ngoài manifest ghi 20 giây | 1 giây / 1 frame | **21 giây / 2 frame** |

**Cách sửa:** đọc đúng danh sách sequence của thí nghiệm; dùng cùng đường đọc/lọc cache với runner để thống kê detection. Tính thời gian trên đúng sidecar của các sequence đó, báo rõ nếu thiếu số đo. Thêm regression với frame vượt `max_det` và video thừa ngoài manifest.

Tôi không có cache BDD thật trong gói nên **chưa xác định các con số detection/22,9 ms hiện tại có sai số bao nhiêu hoặc có bị ảnh hưởng không**. Kiểm lại hai bảng này trên máy; chỉ sinh lại báo cáo khi cần. Lỗi helper này không thay đổi các file HOTA/IDF1 đã tái tính.

**3. [P2] Test tuyên bố chạy qua runner nhưng chưa gọi runner**

Vị trí: `src/test_evaluator_choice.py:125`.

Phép kiểm số 5 ghi là chạy `stage_eval()` thật và kiểm metadata, nhưng thực tế chỉ tính lại biểu thức chọn evaluator, rồi kiểm `chon == 'mot_crowd' and rex is not None`. Không gọi `stage_eval()`, không đọc `eval_meta.json`. Test vẫn có thể PASS nếu đường gọi evaluator trong runner bị sửa sai.

Tôi đã gọi `stage_eval()` thật trong phép tái hiện mục 1: nó hiện chọn đúng `mot_crowd`, FP=1, GT_dets=1. Vì vậy **bản sửa F1 hoạt động**, nhưng regression chưa bảo vệ đúng đường chạy như mô tả.

**Cách sửa:** thay phép kiểm số 5 bằng lời gọi thực với fixture, kiểm metric và `eval_meta.json`. Giữ đủ các tình huống ignore, FP thông thường và GT hợp lệ chồng vùng ignore; không chỉ kiểm lựa chọn lớp.

**4. [P2] Một số kết luận trong báo cáo cần thu hẹp**

Sửa nguồn sinh báo cáo cùng các tài liệu tương ứng để lần sinh tiếp theo không đưa lỗi trở lại.

- **Tốc độ:** `src/report_bdd_transfer.py:247` khẳng định triển khai 5 lớp/max_det 300 “sẽ nhanh hơn” và tổng thời gian là “cận trên”. Hai pha đo riêng, không lặp có kiểm soát và không đo toàn bộ đường chạy không chứng minh được cận trên. Đặt tên là **ước tính cộng hai pha đo riêng**; giữ thời gian tracker đo được riêng. Bỏ bảo đảm nhanh hơn/cận trên và giới hạn câu “gần như không đổi FPS” theo đúng cặp được nói đến. Bảng hiện cho CV 41,8 và PF500 35,6 FPS suy ra, nên không thể dùng câu đó chung cho mọi bộ lọc.
- **So với DETRAC:** `handoff/REVIEW_BRIEF.md` viết “DETRAC p=0,22; BDD CI hẹp vì có 200 video”. Cần nêu rõ điều kiện. DETRAC c10 đã có CI HOTA không chứa 0; c25 chưa có. Không suy ra nguyên nhân CI hẹp chỉ từ cỡ mẫu, vì phân bố video, dữ liệu và hiệu ứng cũng khác. Báo Δ và CI riêng cho từng dataset/điều kiện; bỏ diễn giải nhân quả này và dẫn nguồn nếu giữ p-value.
- **PF:** khoảng SD `0,00005–0,00095` đúng khi nói về **HOTA**. SD AssA tại c10 là **0,00295458**; SD IDF1 tại c25 là **0,00109525**. Viết “trong ba seed đã thử, chênh lệch HOTA PF–CV lớn hơn nhiều biến thiên giữa seed”, không kết luận loại trừ hoàn toàn ảnh hưởng seed hay khái quát cho mọi PF.
- **Hậu quả lỗi timeline cũ:** không khẳng định mọi FP sau frame 94 “biến mất” hoặc “làm đẹp số liệu”. Loader TrackEval 1.3.0 có kiểm tra timestep ngoài `seqLength` và ném lỗi. Nêu đúng điều đã chứng minh: độ dài 94 sai; cần giữ timeline đến 202; nếu có tracking ngoài độ dài cũ, evaluator có thể từ chối dữ liệu.
- **Quy mô ma trận:** là **6 cấu hình/seed × 2 điều kiện = 12 run tổng cộng**, mỗi run 200 video; không phải “12 lượt × 2 điều kiện”. Pilot có 12 run riêng trên 3 video.
- **Mức kết luận:** dùng “CI 95% của phép so sánh đã định trước không chứa 0” thay cho “lợi thế là thật”. Đây là CI riêng cho từng phép so sánh, chưa phải bảo đảm đồng thời cho toàn bộ nhiều cặp. BDD ở đây là protocol gộp phương tiện trên nhãn COCO-video chuyển đổi, không phải kết quả benchmark native hay bằng chứng riêng về phục hồi che khuất.

**Những phần đã kiểm chứng độc lập**

| Kiểm tra | Kết quả |
|---|---|
| Gói delta | 249/249 mục logic khớp hash; không thấy đường dẫn riêng bị cấm; patch roundtrip thành công trong phạm vi gói |
| Nhãn gốc | SHA256 khớp `074ff795…`; 200 video, 39.973 bản ghi ảnh, 465.512 annotation |
| Tái tạo GT toàn bộ val | **200/200 hash GT + seqmap khớp bản khóa**, theo quy ước LF/16 ký tự hash đã ghi; 382.903 dòng GT đánh giá, 22.998 dòng ignore |
| Timeline video `b23b2649-8349d2a1` | JSON có 202 frame ảnh; annotation cuối 94; export ghi `seqLength=202` |
| YAML và manifest thí nghiệm | Hash khớp bản khóa |
| Mã nguồn so với metadata evaluation | 19/19 module khớp sau khi đối chiếu biến thể LF/CRLF; cả 12 run có cùng bảng hash module |
| Metric tổng hợp | **96/96 giá trị**, từ 12 run × 8 metric, sai lệch tối đa `2,22e-16` |
| Metric từng video | **19.200 giá trị** từ 12 run × 200 video × 8 metric khớp trong `1e-9` |
| Bootstrap | **100/100 dòng** của `paired_pairs.csv`, gồm hai cách tổng hợp; 5.000 mẫu ghép cặp, seed 0; sai lệch tối đa `1,82e-12` |
| CLI resume | Cùng đường YAML đổi ngưỡng 0,25→0,95: 4→0 dòng, chạy lại; chạy lần nữa không đổi: tái dùng |
| Backend metadata/audit | Run mới ghi `np` và được audit nhận diện; metadata cũ thiếu bằng chứng vẫn UNKNOWN/LEGACY |
| Test dự án chạy tại môi trường review | `test_bdd100k_coco`: 11/11; `test_evaluator_choice`: 5/5, với giới hạn coverage ở mục 3 |

Phần tính lại metric dùng công thức độc lập/vector hóa NumPy, không gọi `boot_stats.combine()` hay `paired_bootstrap()` của dự án. Nguồn vẫn là thống kê `raw_fields.json` do lần đánh giá của dự án tạo ra: đây là xác nhận số học và tính nhất quán, không phải chạy lại detector/tracker trên ảnh thật.

**Kết quả EKF–CV được xác nhận**

| Điều kiện | ΔHOTA [CI 95%] | ΔIDF1 [CI 95%] | ΔIDSW [CI 95%] |
|---|---|---|---|
| c25 | +0,0014965 [+0,0004847; +0,0025902] | +0,0045889 [+0,0028079; +0,0063655] | −797 [−1055,025; −557,975] |
| c10 | +0,0024926 [+0,0011966; +0,0038573] | +0,0060451 [+0,0039691; +0,0081640] | −825 [−1066,025; −601] |

AssA cũng có CI chênh lệch dương ở cả hai điều kiện; DetA có CI chứa 0 ở cả hai. HOTA của UKF–CV có CI chứa 0. PF kém trong implementation/cấu hình và ba seed đã thử.

Lưu ý PF: bảng chính lấy trung bình metric theo seed, còn bootstrap lấy trung bình thống kê trước khi tái tổ hợp. Bản khóa đã khai báo sự khác nhau này. Tôi tái tính đúng cả hai; không đổi cách tổng hợp sau khi xem kết quả. Tuy vậy, lời giới thiệu `cap_so_sanh.md` cần nhắc ngoại lệ PF thay vì nói tất cả CI cùng một đại lượng với bảng chính.

**Khóa cấu hình và giới hạn kiểm chứng**

Theo timestamp lưu trong hồ sơ: khóa `2026-09-20T03:49:13`; pilot sớm nhất `03:49:59`; evaluation full sớm nhất `04:21:33`, muộn nhất `04:26:56`. Cả 24 metadata pilot/full đều sau mốc khóa, chọn `mot_crowd`, ghi commit thực nghiệm `d1f6d62`. Đây là đối chiếu hồ sơ được cung cấp, không phải chứng thực độc lập đồng hồ hay lịch sử máy. Ba sequence smoke đã xem trước vẫn thuộc val như đã công bố.

Log được bàn giao có 26/26 suite PASS và exit code 0. Tôi **không chạy lại toàn bộ 26 suite**, không có GPU/ảnh/cache/tracking thật để làm điều đó. Môi trường review dùng Python 3.12, backend NumPy, TrackEval 1.3.0 và lap 0.5.13. Các probe trong tiến trình review chặn torch/Ultralytics; CLI chạy trong môi trường không cài hai thư viện này.

Review tree được dựng từ các snapshot đã có và payload delta. Không có checkout Git đầy đủ của máy Windows; việc áp patch/đối chiếu hash giới hạn ở các đường dẫn và preimage được gói cung cấp. Không xác nhận lại sự tồn tại hay khả năng đọc 39.973 JPG trên máy người dùng. Nano chưa có kết quả đo được cung cấp; không suy ra tình trạng kết nối chỉ từ việc thiếu SSH config/thư mục kết quả.

**Phạm vi công việc tiếp theo**

1. Sửa guard GT và regression thật qua runner; giữ nguyên mọi cấu hình nghiên cứu đã khóa.
2. Sửa helper báo cáo để dùng manifest và cùng bộ lọc cache, kiểm lại hai bảng detection/timing.
3. Chỉnh các câu diễn giải ở mục 4, đồng bộ brief/handoff và nguồn sinh báo cáo.
4. Để có provenance GT trực tiếp, lưu bản evaluation hiện tại rồi đánh giá lại từ **tracking hiện có** cho 12 run BDD. So metric trước/sau; nếu có lệch thì điều tra nguyên nhân trước khi thay kết luận. Không ghi fingerprint hồi tố cho bản cũ.
5. Bàn giao delta nhỏ, log các phép kiểm liên quan và bảng đối chiếu metric. Không cần mở thí nghiệm mới hoặc chạy lại A/B/C/GĐ1 chỉ vì review này.

Review này không sửa source dự án, index, commit hay ZIP gốc. Chỉ tạo bản sao phục vụ kiểm tra, fixture/probe tạm và báo cáo.
