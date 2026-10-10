# Review c4d64a1 — nhãn BDD100K MOT cách ly, vòng 30–32

**Kết luận: CHANGES REQUESTED đối với bộ kiểm và diễn giải bàn giao.** Tính toàn vẹn gói delta đạt trong phạm vi 204 đường dẫn thay đổi. Chưa tìm thấy bằng chứng bộ nhãn train bị hỏng. Có cơ sở tiếp tục theo phương án **C**, sau khi đóng hai finding dưới đây; đây không phải xác nhận bộ nhãn trùng bản phát hành chính thức.

Review chỉ đọc, chạy trong workspace riêng. Không sửa repository Windows, không nhập nhãn, không khóa split, không training và không SSH Nano. Base review nên giữ `bcb92f8` trong vòng sửa này.

## 1. Định danh đầu vào

- Base: `bcb92f86824812ceea3dee56dabe88ef802b1a25`.
- Target: `c4d64a136b75aa505881e07fb786333cc4744dcf`.
- ZIP: **788.495 byte**, SHA256 `4ca1b6dd8ad47c0c8736f156d5937909072469a72ebe1f8d97bd79d7ff5f378b`.
- Repo nhãn: `jaffe03195/bdd100k_sot`, revision `694197099692fe1254e93215e73e68ece8b1f41f`, member trong `bdd100k.zip`.
- Manifest nhãn bàn giao: SHA256 `a2e416f26d2ba44a50c7f3ec8578deec39f364cd2381c36dc12fe45a905fcf39`.
- COCO val dùng đối chiếu: file người dùng đã cung cấp, **117.094.032 byte**, SHA256 `074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d`.

## 2. R32-F1 — P2: kết luận ánh xạ lớp/ignore mạnh hơn bằng chứng

**Vị trí:** `handoff/logs/review32/14_nguon_nhan_vong32.md`, dòng 89; bảng nguồn trong `07_kiem_nhan_ung_vien.json`, trường `anh_xa_lop_scalabel_coco_ignore`.

Báo cáo nói ánh xạ lớp đúng kiểu chuyển đổi BDD, ba lớp phụ đều `ignore=1`. Tôi đọc lại **toàn bộ 200 JSON native val tải từ revision đã ghim** và file COCO val của dự án bằng mã đối chiếu riêng. Kết quả:

| Phép đối chiếu | Kết quả độc lập |
|:--|:--|
| Ghép theo video/frame/scalabel_id | 465.512/465.512, không annotation dư một phía |
| Bounding box, quy ước `w=x2-x1+1`, `h=y2-y1+1` | 0 sai khác |
| Song ánh `scalabel_id ↔ instance_id` trong từng video | 0 xung đột cả hai chiều |
| Lớp khác ngoài ba ánh xạ được báo cáo | **183 annotation, 46 track, 40 video** |
| Ngoại lệ so với mô tả `ignore` | **50 annotation** |

183 trường hợp gồm: truck→car 73, pedestrian→car 62, car→pedestrian 20, car→bus 14, bus→truck 4, pedestrian→rider 3, car→rider 2, bus→pedestrian 2, bicycle→rider 1, truck→pedestrian 1, truck→bus 1. Tất cả các hàng này có `ignore=0` trong COCO.

50 ngoại lệ ignore: 24 other person→pedestrian nhưng `ignore=0`; 5 other vehicle→car nhưng `ignore=0`; 21 pedestrian→pedestrian có `ignore=1`. Nhãn native không có thuộc tính `ignored`; con số này là ngoại lệ đối với **quy tắc được mô tả trong báo cáo**, không tự chứng minh lỗi annotation.

Ví dụ thực: video `b1c9c847-3bda4659`, ID `00114202`, frame 88–91 có native `truck`, COCO `car`; cùng track trong COCO có 4 annotation car và 32 truck. Bản native không có track đổi lớp, trong khi COCO có 53 track đổi tổ hợp lớp/ignore. Đây có thể là khác biệt phiên bản hoặc bước xử lý nhãn; **chưa xác định nguyên nhân, chưa xác định bên nào đúng hơn**.

**Ảnh hưởng:** không được lấy “box và ID khớp” để kết luận cả nhãn ngữ nghĩa khớp. Lớp/ignore liên quan trực tiếp tới detector 5 lớp và đánh giá sau training. Các kết quả BDD lịch sử vẫn thuộc bản COCO đã khóa; không thay nhãn và không chạy lại lịch sử chỉ từ finding này.

**Yêu cầu sửa có giới hạn:**

1. Dùng `class_mismatches.csv` và `ignore_exceptions.csv` trong gói bằng chứng để kiểm lại, ghi số lượng và ví dụ trong báo cáo/brief.
2. Ghi rõ phần đã xác nhận là box + ID; lớp/ignore còn khác. Nếu tìm được mã chuyển đổi hoặc lịch sử giải thích thì dẫn bằng chứng; nếu chưa có, giữ `UNKNOWN`, không bắt buộc tìm nguồn vô hạn.
3. Không tự sửa nhãn để ép khớp, không coi COCO hay native là chuẩn tuyệt đối. Giữ nguyên COCO val dùng cho các kết quả đã bàn giao và đề xuất rõ bản nhãn dùng ở từng bước tiếp theo.
4. Hạ câu “từng lớp khớp bài báo, chỉ train lệch” thành “thống kê tổng hợp tương đồng, có sai lệch”; thống kê làm tròn không xác thực từng annotation.

## 3. R32-F2 — P2: bộ kiểm có thể trả thành công khi chưa kiểm đủ hoặc đầu vào sai

**Vị trí:** `handoff/logs/review32/kiem_nhan_ung_vien.py`:

- Dòng 181–193 bỏ qua split không tồn tại rồi trả 0; root nhãn không tồn tại cho JSON `{}` và exit 0.
- Dòng 55 dùng `fr.get('labels') or []`, làm `labels: {}` thành danh sách rỗng hợp lệ trước kiểm kiểu.
- Dòng 63–72 chỉ kiểm `int/float` và bất đẳng thức; `NaN` qua được, báo mọi lỗi schema bằng 0.
- Không bắt ID trùng trong cùng frame. Dòng 157–159 chỉ kiểm chiều native→COCO, nên hai ID native cùng ánh xạ vào một instance COCO vẫn báo không bất nhất.
- Dòng 186 âm thầm bỏ đối chiếu nếu file `--coco-val` không tồn tại. Kết quả COCO ở dòng 201–202 chỉ được in, không tham gia trạng thái cuối.

**Tái hiện độc lập qua CLI thật**, bằng fixture tạm; không sửa dataset:

| Ca | Kết quả trên c4d64a1 |
|:--|:--|
| Đối chứng sạch | exit 0, box khớp, schema 0 lỗi |
| Không có root nhãn / không có train | exit 0; `{}` hoặc chỉ có val |
| `labels: {}` trong frame train | exit 0, `labels_khong_phai_list=0` |
| `x1=NaN` | exit 0, schema 0 lỗi |
| Hai nhãn cùng ID trong một frame | exit 0, schema 0 lỗi |
| Hai ID native gộp thành một instance COCO | exit 0, số ID bất nhất vẫn 0 |
| File COCO được chỉ định không tồn tại | exit 0, không có phần đối chiếu |
| COCO lệch bbox 5 px / đổi car thành truck | exit 0, sai lệch chỉ nằm trong JSON |

Script `REPRO_REVIEW_c4d64a1.py` exit **1**, `bug_count=5` cho năm ca bắt buộc: thiếu root, thiếu train, NaN, `labels={}`, ID trùng trong frame. Bốn ca COCO còn lại được ghi là **quan sát**, không cộng vào `bug_count`, vì lệnh chỉ báo cáo có thể chủ ý trả 0 khi có khác biệt. Hai ca bbox/lớp phản ánh chính sách report-only đã có trong docstring; không coi chúng là sai công thức. Tuy nhiên exit 0 này không được dùng như một chứng nhận đã đối chiếu thành công.

**Phạm vi ảnh hưởng:** công cụ bổ sung nằm trong `handoff/logs/review32`, không phải bằng chứng runner huấn luyện bị lỗi. Trên toàn bộ **1.600 nhãn native thật**, phép kiểm độc lập không thấy NaN/Inf, ID trùng cùng frame, frame trùng hoặc sai kiểu ba thuộc tính. Vì vậy chưa có bằng chứng cần thay nhãn train; finding yêu cầu sửa cách kiểm và cách báo trạng thái.

**Yêu cầu sửa:**

1. Bắt thiếu root/split bắt buộc và bộ nhãn rỗng. Nếu cho phép kiểm một phần, phải chọn rõ phạm vi và xuất `NOT_RUN` cho phần chưa kiểm.
2. Phân biệt `labels=[]` hợp lệ với kiểu sai; kiểm số hữu hạn, loại boolean làm tọa độ; kiểm trùng ID trong frame.
3. Kiểm ID song ánh hai chiều trong video; phát hiện khóa trùng trước khi đưa annotation vào dictionary.
4. File COCO được yêu cầu mà thiếu phải báo thiếu rõ. Có thể giữ chế độ chỉ báo cáo, nhưng xuất trạng thái đối chiếu riêng (`MATCH`, `DIFFERENCES`, `NOT_RUN` hoặc tương đương), không diễn giải exit 0 thành khớp toàn bộ. Không cần ép các sai lệch lớp đã biết thành lỗi dữ liệu.
5. Thêm regression từ các fixture này, chạy lại bộ kiểm trên bản cách ly. Không cần chạy full suite tracking/Nano nếu không sửa đường đó.

## 4. Những gì reviewer đã kiểm độc lập

### Gói delta

- Kích thước/SHA256 ZIP đúng; `testzip` đạt.
- **199/199 payload** đúng SHA256 và số byte; **5/5** file trùng nội dung có nguồn thay thế đúng hash.
- Khôi phục 7 preimage thay đổi từ payload + reverse patch, kiểm hash base khai báo, rồi áp forward patch: **204/204 đường dẫn** đúng hash target.
- Patch chỉ đổi `handoff/` và `CODEX_HANDOFF.md`; không đổi `src/`, `configs/` hay các đường dẫn riêng bị cấm.
- **Giới hạn:** reviewer không có checkout Git đầy đủ bcb92f8, nên không tự xác nhận tuyên bố “toàn cây 3.584 file”; kiểm này là roundtrip 204 đường dẫn thay đổi, không phải xác thực lịch sử commit bằng Git.

### Nguồn nhãn thật

- Đọc mục lục từ archive HF ở revision ghim: 5.352.373.031 byte; 1.602 hàng manifest khớp tên/kích thước/CRC của mục lục.
- Tải riêng và hash lại **1.400 native train + 200 native val**: **1.600/1.600 SHA256 khớp manifest**. Không tải ảnh hay cả archive 5,35 GB. Hai COCO JSON bổ sung của người đóng gói chỉ kiểm descriptor, không tải lại/hash nội dung.
- Train: **278.079 frame, 2.886.916 annotation, 112.819 ID theo video**, 3.930 frame có `labels=[]`.
- Val: **39.973 frame, 465.512 annotation, 18.842 ID theo video**, 615 frame có `labels=[]`.
- Không frame trùng, ID trùng trong frame, tọa độ không hữu hạn hoặc sai kiểu occluded/truncated/crowd; không track đổi lớp. Tên file/video/frameIndex nhất quán.
- Có thể xác nhận lại toàn bộ đối chiếu box/ID với COCO val như mục 2; các sai lệch lớp được lưu từng hàng để tái kiểm.
- Đã chạy test bàn giao **3/3** cho kiểm nhãn và **7/7** cho HTTP Range. Chỉ đổi đường dẫn Windows trong bản sao test để chạy trên Linux; mã được test giữ nguyên.

### Chưa tự xác nhận

- Không có archive nhãn gốc khớp MD5 chính thức. Hash HF ghim chỉ xác định bản đang dùng, không chứng minh nguồn phát hành ban đầu.
- Không có 278.079 ảnh train trong workspace để lặp lại kiểm ảnh–nhãn trên máy Windows. Con số “khớp ảnh chính thức 1:1” ở mức này vẫn dựa vào log bàn giao.
- So ảnh val của Claude là tên/kích thước/**CRC32 của ZIP**; không diễn giải thành so SHA256 hay đọc từng byte ảnh độc lập.
- Không tự chạy `kiem-nguon` chính thức trong Windows; log chính thức exit 2 và cách ly exit 0 được đọc, không nhận là lệnh reviewer đã chạy.
- Không xác minh từng nhãn train với một nguồn annotation độc lập; không dùng thống kê bài báo thay cho phép xác thực này.

## 5. Quyết định và phạm vi tiếp theo

**Đề xuất giữ phương án C.** Không có lý do từ review này để vứt bộ nhãn hoặc tải lại 7 gói ảnh. Trước khi nhập, sửa hai finding, cập nhật trạng thái nguồn và nộp delta nhỏ. Nhãn có thể được dùng có điều kiện như một snapshot bên thứ ba đã ghim, nếu người dùng duyệt rủi ro nguồn; không gọi là bản gốc đã được xác thực.

Sau khi được chấp thuận: chỉ nhập 1.400 JSON train nguyên byte, giữ bản cách ly và manifest; không ghi đè COCO val đang dùng. Chạy `kiem-nguon`, rồi thực hiện các bước chuẩn bị/smoke đã được duyệt. **Training dài, ngân sách epoch, thay dataset/model, và tối ưu Nano vẫn cần theo plan đã chốt; review này không cấp thêm phạm vi.**

Nếu sau này có archive gốc, so theo **đường dẫn và SHA256 từng file**, ghi rõ khác biệt; không tự thay snapshot sau khi đã khóa split hoặc training.

## 6. Bằng chứng giao kèm

`CODEX_REVIEW_c4d64a1_EVIDENCE.zip` gồm report, script tái hiện, script đối chiếu thật, các JSON/CSV kết quả, log hai bộ test, kiểm gói và manifest hash lại 1.600 file native. Không chứa dataset/ảnh/weights, ZIP nguồn, URL tải có chữ ký hay thông tin đăng nhập.

Chạy fixture từ repository trên PC:

```powershell
python REPRO_REVIEW_c4d64a1.py --repo D:\LABOTARY\bytetrack --out D:\LABOTARY\bytetrack\handoff\logs\review33\codex_repro
```

Script chạy CLI trong thư mục tạm. Trên mã được review, exit 1 là các ca mô tả vẫn tái hiện; đọc `summary.json` để phân biệt lỗi schema và quan sát về phạm vi đối chiếu.
