# Vòng 45 — ghi nhận nghiệm thu `3e94b21`; chuẩn bị quyết định D1–D5 cho đánh giá detector trên UA-DETRAC (chi tiết)

**Phạm vi** (người dùng duyệt 2026-10-07):
- Ghi nhận review `3e94b21`.
- Chuẩn bị năm quyết định quy ước lớp / ignore cho UA-DETRAC test, kèm dự thảo giao thức.
- **Không** chạy đánh giá GPU, không viết bộ đọc GT, không sửa mã lõi, dataset hay checkpoint.
- Base review: **`3e94b21`**. Vòng 45 **chưa** được nghiệm thu.

> **Đính chính vòng 46** (review `d4715aa`: REQUEST CHANGES; chi tiết `REVIEW46_CHI_TIET.md`). Số đếm car / bus / van /
> others, danh sách 173 ảnh và các phương án loại 173 ảnh không đổi.
> - **R45-F1**: lớp N/A ở D3 **không** phải "không FP". Scorer giữ nguyên vẫn đếm TP / FP thô và precision cho lớp
>   không có GT; chỉ AP / recall / F1 = null và macro loại lớp đó. Báo N/A cần bước chuẩn hoá báo cáo phải triển
>   khai sau khi duyệt D3.
> - **R45-F2**: bỏ các kết luận một chiều về recall / AP ở D5. Phát biểu về toolkit giới hạn theo phần mã đã đọc.
>   40.835 chỉ là số GT XML car / bus đạt ngưỡng trong thống kê này, không phải số GT toolkit chắc chắn loại.
> - **R45-F3**: phương án giữ 173 ảnh thiếu 265 lượt vùng ignore cấp sequence; đã tính lại ở
>   `logs/review46/phan_tich/dem_theo_phuong_an.json`.
> - **R45-F4**: cột camera / thời tiết của `theo_video.csv` vòng 45 bị bỏ trống do lỗi parser. Bản đúng:
>   `logs/review46/phan_tich/theo_video.csv`; cả 40 video đều có đủ hai thuộc tính.

Mọi số đo về UA-DETRAC ở đây đến từ thao tác **chỉ đọc** XML và ảnh (`logs/review45/01`–`05`).

## 0. Trạng thái đầu vòng

Log `logs/review45/00_trang_thai_dau.log`:
- HEAD `3e94b21`, **0 commit** sau target được review.
- Working tree chỉ có file riêng của người dùng.
- Đang cắm sạc, gói High performance.

## 1. Ghi nhận nghiệm thu `3e94b21` (commit `d3b5460`)

- **Báo cáo** `handoff/CODEX_REVIEW_3e94b21.md`: trùng byte bản trong `D:\LABOTARY\review_codex` (8.566 B, sha256
  `26f0d0d0…`).
- **Kết luận**: **PASS WITH LIMITATIONS**, đóng **R43-B1** và **R43-N1**. Phạm vi gồm bản sửa metadata, support code,
  đường đọc GT / chấm điểm kiểm bằng fixture, và hồ sơ BDD vòng 43 được giữ nguyên.
- **Reviewer đã chạy độc lập** (Linux, Python 3.12):
  - test BDD **22/22**;
  - cùng test trên entry point `a03f6e1`: **20/22**;
  - script reviewer cũ: `bug_count 0`;
  - nhóm E và E7–E9 của `test_detector_5lop.py`: **13/13 PASS + 1 SKIP** (E6). AP khớp **pycocotools 2.0.11** thật
    trên fixture.
- **Giới hạn ghi đúng như review**:
  - reviewer **không** chạy lại inference trên dữ liệu thật, **không** kiểm trực tiếp checkpoint hay 9 file lớn;
  - không xác nhận "4.697 file toàn cây";
  - **92/92 của tác giả vẫn là đối chiếu log**;
  - mảng `ign` dùng chung mọi lớp là quy tắc của dự án, không đồng nhất với benchmark COCO / BDD;
  - review **không** duyệt ánh xạ UA-DETRAC.
- **Bằng chứng**: `CODEX_REVIEW_3e94b21_EVIDENCE.zip` 176.037 B, sha256 `72f5f61a…`. Giải nén vào thư mục mới, ZIP gốc
  giữ nguyên. `SHA256SUMS` khớp **43/43** trên đĩa và trên blob đã stage. Lưu ở `handoff/review_evidence/CODEX_3e94b21/`.
- **Gói đã nghiệm thu**: 600.666 B, `9bb1bbab…`, khớp mục 1 của review. Lưu ở `handoff/archive/review_delta_3e94b21.zip`
  **trước** khi đổi base.
- **Base review** `4d54744` → **`3e94b21`** (vòng 44); giữ đủ 13 khai báo đầu vào không bàn giao.
- **Không** chạy lại ba lượt BDD: bản sửa metadata không kích hoạt ở ngưỡng 0,25. Kết quả, checkpoint và log giữ nguyên.

## 2. Dữ kiện UA-DETRAC test (chỉ đọc)

**Đếm lại** (`01_phan_tich_detrac_test.log`, `phan_tich/tong_hop.json`, `theo_video.csv`). Mọi số **khớp** mục 8 của
`REVIEW43_CHI_TIET.md`.
- 40 video (manifest `ff2b4b9c…` = `testlist-full.txt` của toolkit).
- 56.167 `<frame>`, 56.340 ảnh, **173 ảnh không có `<frame>`**, 0 `<frame>` không có ảnh. Mọi ảnh 960×540.
- 675.774 box: car 548.555, bus 71.785, van 38.519, others 16.915.

| `vehicle_type` | box | track | video | rộng × cao trung vị (px) | GT có IoA ≥ 0,5 với vùng bỏ qua (max từng vùng / hợp pixel) |
|:--|--:|--:|--:|:--|:--|
| car | 548.555 | 1.975 | 40 | 77,8 × 43,7 | 35.248 / 35.757 |
| bus | 71.785 | 205 | 37 | 173,7 × 92,7 | 5.587 / 6.454 |
| van | 38.519 | 126 | 35 | 86,3 × 59,7 | 2.844 / 3.035 |
| others | 16.915 | 31 | 21 | 153,5 × 129,6 | 401 / 667 |

**Dữ kiện mới so với vòng 43**:
- **Không có** nhãn `truck`. Mã toolkit có nhận lớp này (`struct2gt.m`), nhưng XML test không dùng.
- `density` bằng số `<target>` ở **mọi** frame. Có **0** frame với `target_list` rỗng, tức XML **không có quy ước "frame
  rỗng đã xác nhận"**.
- 163 `ignored_region` trùng file `igrs/*_IgR.txt` của toolkit ở 40/40 video; 4 cặp vùng chồng nhau.
- 7.779 box car nằm **trọn** trong vùng bỏ qua.

**Xem bằng mắt** (`02_mau_theo_track.log`; ảnh crop **không** commit; CSV ghi đủ video / track / frame / box để tái lập).
Phân loại là **chủ quan** của tác giả, mỗi track một crop ở frame giữa, chưa kiểm chéo.
- `others`, đủ 31 track: 21 xe tải chở hàng (ben, thùng, sàn, xe bưu chính; 34,9 % box), **7 xe cứu hộ / kéo xe
  (54,2 % box**, vì là các track rất dài), 1 trông giống xe buýt (2,3 %), 2 không rõ (8,6 %).
- `van`, 48/126 track ngẫu nhiên seed 0: 45 van chở người / minivan / MPV kiểu xe con (91,7 % box của mẫu), **2 minibus
  kiểu xe buýt** (7,4 %), 1 van chở hàng.
- `bus`, 24 track mẫu: đều là xe buýt thành phố (kể cả xe nối toa).
- `car`, 24 track mẫu: xe con / taxi. Vài crop bị vật che (xe buýt, xe tải) lấn vào khung — đây là che khuất, không phải
  sai lớp.

**Toolkit DETRAC chính thức** (`data/extracted/DETRAC-MOT-toolkit`, chỉ đọc mã):
- `dropDetections.m`, được gọi trong bước tiền xử lý `preProcessDetectionResults.m` **trước** khi gọi
  `AP_DET_EVAL.exe`, loại detection có tỉ lệ pixel nằm trong **hợp** các vùng ≥ 0,5.
- `parseGT.m` (**chỉ đường MOT**) cắt box GT có IoA ≥ 0,5 ở hai đầu quỹ đạo.
- Đầu vào detection có 7 cột, không có trường lớp (`checkResultFormat.m`). `README_curve.txt` mô tả PR theo ngưỡng score
  0:0,05:1.
- Phần chấm detection, gồm cả cách xử lý GT trong vùng bỏ qua và cách dùng `vehicle_type`, nằm trong `AP_DET_EVAL.exe`
  (nhị phân): **chưa xác minh** *(đính chính vòng 46; bản trước viết "toolkit chấm một lớp vehicle" như đã biết)*.

## 3. Năm quyết định cần người dùng duyệt

| # | Dữ liệu quan sát | Lựa chọn | **Đề xuất** | Lý do | Ảnh hưởng TP / FP / FN / AP |
|:--|:--|:--|:--|:--|:--|
| **D1** `van` | 38.519 box, 126 track, 35 video. Mẫu: phần lớn van / MPV kiểu xe con, ~2/48 track là minibus. Repo **không có** bằng chứng BDD gán van vào lớp nào | (a) vùng bỏ qua cho mọi lớp; (b) GT car | **(a)** | `van` là lớp riêng của DETRAC, trộn van kiểu xe con với minibus. Gán vào car là giả định taxonomy chưa kiểm cho model học BDD; (a) trung tính | (a) car GT 548.555 (55.764 ảnh). Detection bất kỳ lớp trên van **không** FP nếu IoA ≥ ngưỡng; bỏ sót van **không** FN. (b) car GT +38.519 (+7,0 %) → 587.074: detect van là car → TP, bỏ sót → FN car; detection bus trên minibus → FP bus + FN car |
| **D2** `others` | 16.915 box, chỉ 31 track / 21 video. Mẫu: 21 xe tải chở hàng, 7 xe cứu hộ (54,2 % box), 1 giống xe buýt, 2 không rõ | (a) vùng bỏ qua; (b) GT truck | **(a)** | Không đủ bằng chứng mọi `others` là truck; AP truck sẽ bị vài track dài chi phối (31 track, 7 xe cứu hộ chiếm hơn nửa box) | (a) không có AP truck; detection không ghép GT mà IoA với box `others` ≥ ngưỡng thì không FP. (b) truck GT 16.915 box / 15.957 ảnh; detection bus đúng trên track giống xe buýt → FP bus + FN truck; mAP thành 3 lớp |
| **D3** lớp báo AP | GT tương thích: car (xe con / taxi), bus (xe buýt thành phố). Không có GT truck, xe máy, xe đạp | car, bus; thêm truck nếu D2 = (b) | **car, bus; "mAP (car, bus)" = trung bình 2 lớp; truck / motorcycle / bicycle = N/A** | Chỉ báo lớp có GT tương thích. Evaluator trả AP `null` (không phải 0) cho lớp không GT, và macro chỉ lấy lớp có GT | *(Đính chính vòng 46)* Scorer giữ nguyên **vẫn** đếm TP / FP thô và precision (= 0 nếu có detection lớp đó với conf ≥ 0,25) cho lớp không GT; AP / recall / F1 = null; lớp không vào mAP / macro. Các số thô này **không** phải metric. Để bảng báo cáo hiển thị N/A (JSON và CSV), adapter cần bước chuẩn hoá D3: **thay đổi phải triển khai sau khi D3 được duyệt**, chưa có sẵn. mAP 2 lớp **không** so với mAP 5 lớp của BDD |
| **D4** 173 ảnh không có `<frame>` | 6 video. Khe giữa: MVI_39051 992–1060 (69 ảnh), MVI_39211 477–570 (94), MVI_39271 77–80 (4); frame có nhãn liền kề đều 1 xe. Đuôi video: MVI_40743 1630, MVI_40763 1742–1745, MVI_40793 1960; frame liền trước có 8 / 4 / 2 xe. XML không có frame rỗng nào. Xem ảnh: MVI_39271 f78 có xe **ngoài** vùng bỏ qua (chính chiếc xe có nhãn ở f76) mà không có nhãn; ảnh đuôi thấy xe trên đường (chưa đối chiếu từng xe với vùng bỏ qua); MVI_39051 f1025 trông trống | (a) loại khỏi đánh giá, ghi danh sách; (b) giữ làm ảnh âm tính | **(a)** — đề xuất, chưa áp dụng | Đây là "**không có annotation**", không phải "đã xác nhận không có xe". Ở BDD, ảnh không box vẫn được **liệt kê** trong file nhãn nên được giữ; ở đây XML hoàn toàn **không liệt kê** 173 ảnh này. Tập chấm = đúng các frame nguồn nhãn liệt kê | (a) 56.167 ảnh; GT không đổi. (b) +173 ảnh (0,31 %) không GT, vẫn chịu `ignored_region` cấp sequence (265 lượt vùng): mọi detection ngoài vùng bỏ qua là **FP**, kể cả trên xe có thật nhưng không gán nhãn (ít nhất MVI_39271 f77–80) → chỉ thêm FP, không thêm TP / FN, nên P và AP không thể tăng; FN không đổi |
| **D5** `ignored_region` | 163 vùng, 4 cặp chồng nhau. GT car / bus có IoA ≥ 0,5 với vùng: 35.248 / 5.587 (max từng vùng); 7.779 car nằm trọn trong vùng. Pipeline tracking tô xám vùng này trên ảnh trước khi detect | (a) quy tắc BDD hiện dùng, không đổi; (b) dựa trên phần mã toolkit đã đọc (hợp vùng, ngưỡng cố định 0,5, loại detection trước khi chấm; xử lý GT của đường DET chưa xác minh) — cần sửa evaluator, ngoài kế hoạch; (c) **tự định nghĩa**: giữ evaluator nhưng chuyển GT có IoA ≥ 0,5 thành vùng bỏ qua — quy tắc GT mới, khác BDD | **(a)** | Giữ một quy tắc chấm cho BDD và DETRAC; không tự đổi ngưỡng hay thuật toán | Chi tiết quy tắc ở mục 4. (a) vùng chỉ áp cho detection **không** ghép được GT; GT trong vùng vẫn là GT (detect → TP, sót → FN). Không tô xám ảnh (khác pipeline tracking). *(Đính chính vòng 46)* (b)/(c) thay đổi tập GT và cách tính TP / FP / FN; recall / AP có thể **tăng hoặc giảm** (phản ví dụ fixture: AP50 0,5050 → 0); chiều và mức trên dữ liệu thật **chưa biết** khi chưa có prediction. 40.835 (= 35.248 + 5.587) chỉ là GT XML gốc đạt ngưỡng trong thống kê này, không phải số GT toolkit chắc chắn loại |

## 4. Quy tắc ignore hiện dùng (D5 = (a)) và khác biệt với toolkit

`eval_detector_5lop.ghep_anh_lop` (sha256 `30a1fd5a…`, đã nghiệm thu từ vòng 28). Với mỗi ảnh, mỗi lớp, mỗi ngưỡng IoU
t ∈ {0,50, 0,55, …, 0,95}:
1. Xếp detection theo conf giảm dần (mergesort).
2. Detection ghép với GT **hợp lệ** chưa ghép có IoU cao nhất ≥ t; hòa IoU thì chọn GT sau. Ghép được là **TP**.
3. Không ghép được: tính `IoA = diện tích(det ∩ vùng_k) / diện tích(det)`, lấy **max theo từng vùng k** (không phải hợp
   vùng), bằng hình học số thực. Nếu IoA ≥ t thì detection **bị bỏ qua** (không TP, không FP); ngược lại là **FP**.
4. Mảng vùng bỏ qua dùng chung cho mọi lớp của ảnh, gồm `ignored_region` và các box van / others nếu D1 / D2 = (a).
5. AP bỏ các detection bị bỏ qua. Điểm vận hành (conf ≥ 0,25, IoU 0,5) cũng không tính chúng là FP.

Khác biệt với toolkit: bảng ở `logs/review45/06_du_thao_giao_thuc_detrac.md` mục 10. Ở ngưỡng 0,5, hai cách tính IoA chỉ
lệch ở một phần nhỏ GT: với car, 699 box đạt theo hợp pixel nhưng không đạt theo max từng vùng, và 190 box theo chiều
ngược lại.

## 5. Dự thảo giao thức

`logs/review45/06_du_thao_giao_thuc_detrac.md` — **dự thảo, chưa khóa, chưa chạy**:
- ba checkpoint vòng 43, hash đã kiểm lại (`05`);
- cùng tập ảnh và quy tắc chấm cho cả ba model;
- tham số suy luận kế thừa nguyên vòng 43. Khác biệt bắt buộc: ảnh 960×540; không tô xám vùng bỏ qua;
- lưu NPZ dự đoán + `ket_qua.json` đủ để tính lại;
- bảng hash nội dung ảnh dựng **trước** khi khóa;
- báo cáo tách khỏi BDD.

Sau khi duyệt *(trình tự sửa ở vòng 46, làm rõ 1)*:
1. fingerprint dữ liệu: bảng hash mọi ảnh, w/h mọi ảnh, danh sách ảnh có thứ tự;
2. adapter: bộ đọc GT XML + chuẩn hoá báo cáo D3 + lưu danh sách `video/frame` có thứ tự; kèm fixture qua entry point thật, kiểm JSON lẫn CSV;
3. khóa kế hoạch mới tham chiếu đúng các hash, commit trước suy luận;
4. chạy tuần tự.

## 6. Giới hạn

- Phân loại van / others bằng mắt là **chủ quan**, trên mẫu (van 48/126 track), mỗi track một crop. Không dùng để gán lại
  nhãn; chỉ để người dùng quyết D1 / D2.
- Chưa có dự đoán nào trên UA-DETRAC, nên mọi "ảnh hưởng" ở mục 3 chỉ là phía GT / quy tắc, không phải số metric.
- Cách toolkit chấm detection (`AP_DET_EVAL.exe`) không đọc được; so sánh dựa trên mã MATLAB đọc được.
- UA-DETRAC test đã được dùng trong thí nghiệm tracking giai đoạn 2, nên không phải tập chưa từng xem.

## 7. Lệnh (từ gốc repo) và mã thoát

```text
python src/make_review_delta.py --set-base 3e94b21 --round 44 --base-snapshot handoff/archive/review_delta_3e94b21.zip --base-note ...  # ok
python handoff/logs/review45/phan_tich_detrac_test.py --out handoff/logs/review45/phan_tich --anh-out <scratchpad>  # exit 0
python handoff/logs/review45/mau_theo_track.py --loai {others|van|car|bus} --toi-da {0|48|24|24} --csv ... --png <scratchpad>  # exit 0
python handoff/logs/review45/ghi_phan_loai_bang_mat.py --thu-muc handoff/logs/review45/phan_tich  # exit 0
python handoff/logs/review45/phan_tich_vung_bo_qua.py --out handoff/logs/review45/phan_tich/vung_bo_qua.json  # exit 0
python handoff/logs/review45/dem_theo_phuong_an.py --out handoff/logs/review45/phan_tich/dem_theo_phuong_an.json  # exit 0
```

Không chạy test hay suite: vòng này không đổi mã dự án.

## 8. Commit vòng 45

| Commit | Nội dung |
|:--|:--|
| `d3b5460` | Ghi nhận review `3e94b21` + bằng chứng 43/43, archive gói, base → `3e94b21`, trạng thái đầu |
| `3c2958e` | Bằng chứng chỉ đọc D1–D5 + dự thảo giao thức |
| commit chứa phụ lục này (target delta) | Brief, phụ lục, handoff |
