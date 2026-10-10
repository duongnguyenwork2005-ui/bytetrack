# Kết quả — đánh giá 3 detector trên UA-DETRAC test (bước 4)

**Người chạy**: Claude, 2026-10-10. Suy luận và chấm chạy từ 06:34:01 tới 07:27:41.
**Kế hoạch khóa**: `handoff/logs/review47/03_ke_hoach_khoa_danh_gia_detrac_test.md`, sha256 `7028a291d2595ee55ead9b055fa1001225733ff8075eec0773680178842d4dfd`.
- Gmail nhận email dấu thời gian lúc 06:34:06.
- Lượt dự đoán đầu tiên bắt đầu sau 06:35:09: thư mục kết quả được tạo sau khi kiểm dữ liệu 66 s.

**Trạng thái**: cả 3 lượt hợp lệ theo mục 5 của kế hoạch khóa. Chưa commit (người dùng quyết định).

## 1. Tóm tắt

- Theo `mAP (car, bus)`:

  | Model | Loại | AP50 | AP50–95 |
  |:--|:--|--:|--:|
  | `yolov8n_coco` | baseline COCO | **0,870** | **0,600** |
  | `yolov8n_ft` | fine-tune trên BDD100K | 0,586 | 0,338 |
  | `yolo11n_ft` | fine-tune trên BDD100K | 0,564 | 0,318 |

- Trên tập này, baseline cao hơn cả hai checkpoint fine-tune:
  - ở cả car lẫn bus;
  - ở cả AP50, AP75 và AP50–95.
- Chênh lệch lớn nhất là ở bus. Recall ở điểm vận hành là 0,34 với cả hai model fine-tune, so với 0,85 của baseline.
- Chiều này ngược với BDD100K val ở vòng 43, nơi fine-tune cao hơn baseline. Hai kết quả không so trực tiếp được (mục 4).
- Chấm lại cả 3 model từ dự đoán đã lưu (NPZ) cho số liệu trùng khớp tuyệt đối với lượt gốc. Bước này ngoài kế hoạch (mục 5).
- Giới hạn:
  - mỗi model chạy một lượt, không có khoảng tin cậy;
  - kết quả chỉ mô tả, không phải kết luận nhân quả;
  - tập này không dùng để chọn checkpoint.

## 2. Kết quả theo D3 (car, bus)

Quy ước trong các bảng:
- Δ = model − baseline `yolov8n_coco`, tính bằng điểm AP (×100).
- AP nội suy 101 điểm. AP50–95 là trung bình 10 ngưỡng IoU 0,50:0,05:0,95.
- Precision và recall tính ở điểm vận hành conf ≥ 0,25, IoU 0,5.
- truck, motorcycle, bicycle là N/A theo D3.

Số trong bảng sinh từ `ket_qua.json` của từng lượt, không chép tay.

**mAP (car, bus)**: trung bình 2 lớp

| Model | mAP50 | mAP75 | mAP50–95 | ΔmAP50 | ΔmAP50–95 | P macro @0,25 | R macro @0,25 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `yolov8n_coco` | 0,870 | 0,703 | 0,600 | — | — | 0,863 | 0,818 |
| `yolov8n_ft` | 0,586 | 0,368 | 0,338 | −28,4 | −26,2 | 0,798 | 0,500 |
| `yolo11n_ft` | 0,564 | 0,337 | 0,318 | −30,6 | −28,1 | 0,756 | 0,500 |

**AP theo lớp**

| Lớp | Số GT | Số ảnh có lớp | Model | AP50 | AP75 | AP50–95 | ΔAP50 | ΔAP50–95 |
|:--|--:|--:|:--|--:|--:|--:|--:|--:|
| car | 548.555 | 55.764 | `yolov8n_coco` | 0,858 | 0,664 | 0,570 | — | — |
| car | 548.555 | 55.764 | `yolov8n_ft` | 0,719 | 0,460 | 0,417 | −13,9 | −15,3 |
| car | 548.555 | 55.764 | `yolo11n_ft` | 0,716 | 0,448 | 0,411 | −14,2 | −15,9 |
| bus | 71.785 | 30.922 | `yolov8n_coco` | 0,883 | 0,741 | 0,629 | — | — |
| bus | 71.785 | 30.922 | `yolov8n_ft` | 0,454 | 0,275 | 0,258 | −42,9 | −37,1 |
| bus | 71.785 | 30.922 | `yolo11n_ft` | 0,411 | 0,226 | 0,226 | −47,1 | −40,3 |

**Điểm vận hành** conf ≥ 0,25, IoU 0,5. Detection chưa ghép có IoA với vùng bỏ qua ≥ 0,5 không tính là FP (D5).

| Lớp | Model | TP | FP | FN | Precision | Recall | F1 |
|:--|:--|--:|--:|--:|--:|--:|--:|
| car | `yolov8n_coco` | 430.081 | 53.536 | 118.474 | 0,889 | 0,784 | 0,833 |
| car | `yolov8n_ft` | 364.116 | 91.695 | 184.439 | 0,799 | 0,664 | 0,725 |
| car | `yolo11n_ft` | 362.420 | 103.743 | 186.135 | 0,777 | 0,661 | 0,714 |
| bus | `yolov8n_coco` | 61.090 | 11.871 | 10.695 | 0,837 | 0,851 | 0,844 |
| bus | `yolov8n_ft` | 24.119 | 6.130 | 47.666 | 0,797 | 0,336 | 0,473 |
| bus | `yolo11n_ft` | 24.291 | 8.791 | 47.494 | 0,734 | 0,338 | 0,463 |

## 3. Số liệu thô (không phải metric)

UA-DETRAC test không có GT cho truck, motorcycle và bicycle. Các bộ đếm dưới đây chỉ cho biết model sinh bao nhiêu box của các lớp đó. Không dùng chúng để đánh giá chất lượng detector (D3). Nguồn: `so_lieu_tho_khong_phai_metric.csv`.

| Lớp (không có GT) | Model | Detection conf ≥ 0,001 sau NMS | Detection conf ≥ 0,25, IoA < 0,5 với mọi vùng bỏ qua (scorer ghi FP) |
|:--|:--|--:|--:|
| truck | `yolov8n_coco` | 1.166.523 | 8.864 |
| truck | `yolov8n_ft` | 685.509 | 1.317 |
| truck | `yolo11n_ft` | 561.569 | 921 |
| motorcycle | `yolov8n_coco` | 244.458 | 13.854 |
| motorcycle | `yolov8n_ft` | 252.987 | 18.240 |
| motorcycle | `yolo11n_ft` | 197.068 | 13.661 |
| bicycle | `yolov8n_coco` | 73.795 | 1.219 |
| bicycle | `yolov8n_ft` | 406.905 | 6.982 |
| bicycle | `yolo11n_ft` | 594.499 | 9.884 |

**Chẩn đoán mô tả** (không làm đổi metric)

| Model | Ảnh đã dự đoán | Ảnh không có GT car/bus | Detection conf ≥ 0,25 trên các ảnh đó | Thời gian lượt (s) |
|:--|--:|--:|--:|--:|
| `yolov8n_coco` | 56.167/56.167 | 72 | 176 | 1047,0 |
| `yolov8n_ft` | 56.167/56.167 | 72 | 292 | 982,6 |
| `yolo11n_ft` | 56.167/56.167 | 72 | 305 | 975,2 |

## 4. Đọc kết quả (mô tả, không suy nhân quả)

**Quan sát**
1. **Car**: hai model fine-tune thấp hơn baseline khoảng 14 điểm AP50 và 15–16 điểm AP50–95. Ở điểm vận hành:
   - recall 0,66 so với 0,78;
   - FP 91.695 và 103.743 so với 53.536.
2. **Bus**: chênh lệch lớn nhất, 43–47 điểm AP50. Ở điểm vận hành:
   - recall 0,336 và 0,338 so với 0,851;
   - FN 47.666 và 47.494 so với 10.695;
   - precision 0,80 và 0,73 so với 0,84.
3. **Hai checkpoint fine-tune gần nhau**: mAP50 chênh 2,3 điểm. Mỗi model chỉ có một lượt và không có khoảng tin cậy, nên không xếp hạng hai model này với nhau.
4. **Bộ đếm truck nhỏ, nhưng chưa xác định được nhầm sang truck đóng góp bao nhiêu** (đính chính theo Codex nghiệm thu, P2).
   - Hai model fine-tune có 1.317 và 921 detection truck với conf ≥ 0,25 và IoA < 0,5 với mọi vùng bỏ qua. Baseline có 8.864.
   - Mức FN bus tăng thêm khoảng 37.000.
   - Bộ đếm này **không** gồm detection có IoA ≥ 0,5 với vùng bỏ qua. Theo D5, GT bus nằm trong vùng bỏ qua vẫn được chấm.
   - Vì vậy, bus trong vùng bỏ qua bị nhận thành truck có thể tạo FN bus mà không xuất hiện trong bộ đếm truck.
   - Số liệu hiện có chưa xác định được mức đóng góp của việc nhầm sang truck.
5. **Bus có thể bị gán nhãn car, nhưng chưa xác minh.**
   - FP car của hai model fine-tune tăng khoảng 38.000 và 50.000 so với baseline. Mức tăng này cùng cỡ với mức FN bus tăng thêm.
   - Bus bị gán nhãn car là **một khả năng**. Tuy vậy, FN car cũng tăng khoảng 66.000 và 68.000.
   - Kết quả hiện có không tách được ba trường hợp: bus không được phát hiện, bus bị gán sang lớp khác, hay box lệch ở IoU 0,5.
   - **Nguyên nhân chưa xác định.**

**So với BDD100K val** (vòng 43, `handoff/logs/review43/ket_qua/bang_ket_qua.md`)
- Ở BDD100K val, fine-tune cao hơn baseline:
  - mAP50 trên 5 lớp: +14,0 và +13,2 điểm;
  - car AP50: +22,1 và +21,8;
  - bus AP50: +14,7 và +11,7.
- Hai tập khác nhau ở các điểm sau, nên đây không phải phép so sánh trực tiếp:
  - số lớp: 5 lớp so với chỉ car, bus;
  - quy tắc GT: D1–D5 biến van và others thành vùng bỏ qua;
  - quy ước gán nhãn của hai bộ dữ liệu;
  - loại camera: camera hành trình so với camera giám sát cố định nhìn từ trên cao.
- "Lệch miền giữa BDD và UA-DETRAC" và "fine-tune làm mất đặc trưng học từ COCO" chỉ là **giả thuyết**. Lượt chạy này không kiểm các giả thuyết đó.

**Giới hạn**
- Mỗi model chạy một lượt. Không có khoảng tin cậy hay kiểm định thống kê.
- Tập UA-DETRAC test đã được dùng trong các thí nghiệm tracking trước đây, nên không mô tả là "chưa từng xem". Tập này không dùng để chọn checkpoint.
- Kết quả chỉ đúng với quy tắc đã khóa:
  - quy tắc GT D1–D5;
  - `imgsz` 640, conf 0,001, NMS 0,7;
  - ảnh gốc 960×540.

## 5. Tính hợp lệ (đối chiếu mục 5 kế hoạch khóa)

| Model | Bắt đầu | Kết thúc | exit | Kiểm dữ liệu trước dự đoán | sha256 checkpoint = kế hoạch | Ảnh đã dự đoán | Đủ 6 file |
|:--|:--|:--|:--|:--|:--|--:|:--|
| `yolov8n_coco` | 06:34:01 | 06:52:39 | 0 | đạt (66 s) | đúng | 56.167 | có |
| `yolov8n_ft` | 06:52:39 | 07:10:11 | 0 | đạt (66 s) | đúng | 56.167 | có |
| `yolo11n_ft` | 07:10:11 | 07:27:41 | 0 | đạt (71 s) | đúng | 56.167 | có |

- Mỗi lượt kiểm lại danh tính dữ liệu trước khi dự đoán, và đều khớp kế hoạch khóa:
  - danh sách có thứ tự: `46204978…`;
  - tổng hợp nội dung ảnh: `e4fff621…`.
- Cả 3 lượt chạy bằng adapter đã khóa `6540832e…`, có hai bằng chứng:
  - ba thư mục kết quả không có file `nguon_du_doan_*.json`. Bản sửa luôn ghi file này trước khi chấm, nên lượt nào chạy bằng bản sửa đều phải có;
  - bản sửa được chép vào repo lúc 07:39:35, sau khi lượt cuối kết thúc. Lệnh dùng `cp -p` nên mtime của file trong repo là giờ sửa bản sao, không phải giờ chép.
- Log có cảnh báo `'half' is deprecated` của ultralytics 8.4.141.
  - Evaluator truyền `half=False`, tức chạy FP32 đúng kế hoạch.
  - Log vòng 43 cũng có cảnh báo này.

**Chấm lại kiểm chéo** (ngoài kế hoạch; dùng NPZ đã lưu, không suy luận lại)

**Lần 1** (07:39:54, adapter v2 `971f50a9…`, chỉ `yolov8n_coco`): dừng giữa chừng vì chạy quá chậm.
- Nguyên nhân: `doc_du_doan` đọc `z["xyxy"]`, `z["conf"]`, `z["cls"]` bên trong vòng lặp theo ảnh. Mỗi lần đọc, `NpzFile` giải nén lại cả mảng; riêng `xyxy` (13,5 triệu box) mất 1,23 s. Ước tính cả lượt mất khoảng 24 giờ.
- Lỗi đã có từ bản khóa, nhưng chỉ ảnh hưởng `--cham-lai`. Ba lượt chính chấm ngay trong bộ nhớ nên không bị ảnh hưởng.
- Claude dừng lượt này lúc khoảng 07:48. Thư mục `20261010_yolov8n_coco_kiem_cham_lai/` (chỉ có `danh_sach_anh.json`) và log `20261010_logs/kiem_cham_lai_yolov8n_coco.log` được giữ nguyên.

**Bản sửa v3** (`bb4b497b…`, test 14/14): đọc mỗi mảng một lần. Chi tiết ở `02_adapter_detrac_test.md`.

**Lần 2** (07:49:27–07:59:32, adapter v3, cả 3 model, vào `20261010_<tên>_kiem_cham_lai_2/`): mỗi lượt kiểm lại dữ liệu (64–66 s), exit 0, mất khoảng 3,5 phút.

| Model | `ket_qua_D3` + chẩn đoán thô + danh tính dữ liệu | 3 CSV |
|:--|:--|:--|
| `yolov8n_coco` | 0 chỗ khác (so tuyệt đối, kể cả số thực) | trùng từng byte |
| `yolov8n_ft` | 0 chỗ khác | trùng từng byte |
| `yolo11n_ft` | 0 chỗ khác | trùng từng byte |

Sau khi chấm lại:
- sha256 của 3 NPZ và 3 `ket_qua.json` gốc không đổi.
- NPZ đã lưu tái tạo đúng số liệu của lượt gốc.
- Hai bản sửa v2 và v3 không làm đổi phần chấm.

Script so khớp nằm ở scratchpad của phiên Claude, không ở repo. Có thể làm lại bằng cách so `mo_hinh.<tên>.ket_qua_D3` của hai `ket_qua.json`.

## 6. Review và chỗ khác giao thức

**Chỗ khác giao thức** (kế hoạch khóa, mục 6):
- Chưa commit trước suy luận. Thay vào đó, email dấu thời gian được gửi trước lượt dự đoán đầu tiên.
- Codex review chạy song song với suy luận, không làm trước.

**Codex review lần 1** (06:35:44–06:39:25): **revise**, có 2 issue P2 về `--cham-lai`.
1. `--cham-lai` chưa ràng buộc NPZ với danh tính dữ liệu của lượt gốc.
2. Nếu bước chấm lỗi sau khi NPZ đã lưu thì không phục hồi được, vì `--cham-lai` cần `ket_qua.json`.

Cả hai issue không ảnh hưởng tới dự đoán, ghép hay tính AP của lượt thường.

**Bản sửa**: Claude sửa trên bản sao. Bản trong repo giữ nguyên khi 3 lượt đang chạy. Nội dung sửa:
- ghi `nguon_du_doan_<tên>.json` trước khi chấm, gồm sha256 của NPZ, danh tính dữ liệu và quy tắc dự đoán;
- `--cham-lai` từ chối (exit 2) khi dữ liệu, cấu hình dự đoán hoặc NPZ khác lúc dự đoán;
- lượt chạy bằng bản cũ thì dùng `ket_qua.json`;
- thêm test cho:
  - biên cắt khung;
  - D5 với IoA 0,50 và 0,49;
  - phục hồi khi bước chấm lỗi;
  - NPZ bị đổi;
  - lượt chạy bằng bản cũ.

Kết quả: 14/14 PASS.

**Codex review lần 2** (06:43:33–06:44:46): **approve**. Codex ghi các giới hạn sau:
- review tĩnh, Codex không tự chạy test;
- lượt chạy bằng bản cũ không lưu hash NPZ, nên không xác minh được NPZ còn nguyên từ lúc dự đoán;
- test cho nhánh bản cũ chỉ kiểm đường thành công;
- chưa có test từ chối cấu hình dự đoán khác;
- test cắt khung chưa assert tọa độ sau cắt;
- test D5 chỉ kiểm tổng TP/FP ở IoU 0,5;
- docstring test mô tả coverage quá rộng.

**Đưa bản sửa vào repo** lúc 07:39:35, sau khi 3 lượt xong:
- adapter `971f50a9…`, test `2945cd70…`, 14/14 PASS khi chạy trong repo;
- so với bản Codex đã duyệt (test `a1632335…`), test chỉ đổi docstring để thu hẹp mô tả coverage theo giới hạn cuối ở trên.

Bản sửa không đổi đường dự đoán, ghép hay AP, nên không cần chạy lại 3 lượt.

**Bản sửa v3** (khoảng 07:49, sau khi Codex đã approve v2): sửa lỗi hiệu năng của `doc_du_doan` (mục 5).
- Adapter `bb4b497b…`, test giữ `2945cd70…`, 14/14 PASS.
- Thay đổi chỉ là đọc mỗi mảng NPZ một lần. Lượt chấm lại 3 model cho số liệu trùng khớp tuyệt đối.

**Codex nghiệm thu bước 4** (chỉ đọc; phạm vi: số liệu, tính hợp lệ, diễn giải, v3, phụ lục kế hoạch khóa)

Lần 1 (08:01:10–08:03:12): **revise**.
- Codex tự băm và so artifact, tự đọc JSON/CSV. Không có lỗi số liệu nào.
- [P2] Mục 4, điểm 4 kết luận quá mức rằng nhầm bus→truck không giải thích được phần lớn số bus bỏ sót. Lý do: bộ đếm truck loại các detection có IoA ≥ 0,5 với vùng bỏ qua, còn GT bus trong vùng đó vẫn được chấm. Đã sửa điểm 4: chỉ giữ quan sát số đếm và ghi "chưa xác định được mức đóng góp".
- [P3] Danh sách trường JSON khác nhau giữa lượt gốc và lượt chấm lại thiếu `lenh`. Đã bổ sung (mục 7). Claude so lại: đúng 5 trường.

Lần 2, xác nhận phạm vi hẹp (08:04:30–08:05:21): **approve**.

Giới hạn Codex ghi lại:
- Codex không chạy test, suy luận hay chấm lại, và không băm lại toàn bộ ảnh/XML.
- Codex không tự xác minh các mốc: chép v2 lúc 07:39:35, dừng lượt chậm lúc khoảng 07:48, test 14/14.
- Hash NPZ được ghi sau khi chạy, nên không chứng minh NPZ nguyên vẹn từ thời điểm suy luận.
- Danh tính mã đã chạy trong quá khứ dựa trên hồ sơ.

## 7. File và hash

Thư mục gốc: `results/phase2/detector_eval/detrac_test/`.

| File | sha256 | Kích thước |
|:--|:--|--:|
| `20261010_yolov8n_coco/du_doan_yolov8n_coco.npz` | `599cdec3500377e8236add99642a69e658b19beb807fb44237601babff2e46f1` | 247 MB |
| `20261010_yolov8n_ft/du_doan_yolov8n_ft.npz` | `82fee2885cf93b3e93ac85660a5cc72da068cdb763246a19eadff8b73b019c5f` | 284 MB |
| `20261010_yolo11n_ft/du_doan_yolo11n_ft.npz` | `960fd880c082e504106056c715ab288f9d012f6262664f06eae2c4ead2eb6fac` | 288 MB |
| `20261010_yolov8n_coco/ket_qua.json` | `e1f3518bd01c331b9eab41206909ace8957150ea380722919880e38f86bd3d43` | |
| `20261010_yolov8n_ft/ket_qua.json` | `66e2f9cfcb0d3b342acf05fcc034fdf0ef0340c3303d55f9224716f8a7aa40fe` | |
| `20261010_yolo11n_ft/ket_qua.json` | `5df949480098a3bce7b52643f8ad6077e2994ac4f9f928b930bc8fe4442c9507` | |
| `20261010_logs/chay_3_model.py` (runner) | `3dec4ab6011e5d4367e3c06121db46c7b25ba5e9e4748476dfd76ced16223be1` | |

Hash NPZ được ghi sau khi chạy xong, vì bản adapter đã khóa chưa ghi hash này.

Mã:

| File | sha256 |
|:--|:--|
| `src/danh_gia_detector_detrac_test.py` dùng cho 3 lượt (đã khóa) | `6540832e8cae42d1b129f3c4aaed65422196ffdb502f0db19d183b640709100b` |
| `src/danh_gia_detector_detrac_test.py` v2 (Codex approve) | `971f50a9a22f912e4741f48b203867f831b8f30ec18d6807331673bcb0da2a1d` |
| `src/danh_gia_detector_detrac_test.py` v3, đang ở repo (dùng cho chấm lại lần 2) | `bb4b497b4f89a4464777144a35d66cbd109d03d3c485c4176cde0b67e358ccb4` |
| `src/test_danh_gia_detector_detrac_test.py` đang ở repo | `2945cd709a05c45a22a660a462cff5a09d8f1b2528e71a8d0c5eab17a55370d8` |

Kết quả chấm lại `ket_qua.json`. Số liệu trùng với lượt gốc. File chỉ khác ở các trường sau (bổ sung `lenh` theo Codex nghiệm thu, P3):
- `tao_luc`;
- `lenh` (dòng lệnh `--cham-lai` thay cho `--model`);
- `cham_lai_tu`;
- `mo_hinh.<tên>.thoi_gian_s` (thời gian chấm lại);
- `mo_hinh.<tên>.nguon_du_doan`.

| File | sha256 |
|:--|:--|
| `20261010_yolov8n_coco_kiem_cham_lai_2/ket_qua.json` | `19ee0d4e8467ee3e30b41bfc38ee7a98ab37740380dc6ef2f97efcaf327e4975` |
| `20261010_yolov8n_ft_kiem_cham_lai_2/ket_qua.json` | `808537cc4c2039e09eccce64ae4fba4d731f3a3a6b97a8064af2719ec23cdd91` |
| `20261010_yolo11n_ft_kiem_cham_lai_2/ket_qua.json` | `9d29cb55d06aec5ee0a7c53da281b71774b7bc1aa9df784628a435d3485fe757` |

Các file khác có sha256 trùng mục 1–3 kế hoạch khóa. Đã kiểm lại lúc khoảng 08:01, 12/12 khớp:
- 5 file mã: evaluator, adapter BDD, fingerprint, exporter, `detector_classes`;
- manifest, fingerprint, bảng hash, danh sách D4;
- 3 checkpoint.

Review Codex lưu ở `D:\LABOTARY\automation\runs\bytetrack_r47_fingerprint_adapter_20261010_021706\`:
- `codex_review_buoc1_3\`;
- `codex_review_sua_cham_lai\`;
- `codex_nghiem_thu_buoc4\`;
- `codex_nghiem_thu_buoc4_xac_nhan\`.

## 8. Việc tiếp theo (chờ người dùng quyết định)

1. **Commit**:
   - mã: adapter, test, script fingerprint;
   - bảng hash và fingerprint;
   - kế hoạch khóa và các báo cáo `review47`;
   - `ket_qua.json` và CSV.

   NPZ mỗi file 247–288 MB, nên cân nhắc giữ ngoài Git và chỉ ghi hash.
2. **Tìm nguyên nhân bus recall thấp** (hậu kiểm, ngoài kế hoạch khóa):
   - chạy trên NPZ đã lưu, không suy luận lại;
   - ví dụ: với mỗi GT bus bị bỏ sót, xem có detection lớp nào có IoU ≥ 0,5 và conf ≥ 0,25 không;
   - xem phân bố kích thước box;
   - tách kết quả theo từng video.
3. **Khoảng tin cậy bootstrap theo video** (40 video), nếu cần so sánh có kiểm định. Đây cũng là hậu kiểm.

Codex đã nghiệm thu báo cáo này (mục 6).
