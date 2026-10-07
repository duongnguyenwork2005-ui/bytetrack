# Vòng 43 — kế hoạch đánh giá detector (KHÓA trước khi xem bất kỳ metric test nào)

Ghi và commit **trước** lượt đánh giá đầu tiên trên dữ liệu thật. Sau khi đã xem kết quả, không đổi ngưỡng, quy tắc hay
tham số; mọi sửa đổi cần thiết chỉ ghi vào phụ lục mục 10.

## 1. Phạm vi (người dùng duyệt, 2026-10-06)

- Đánh giá detector trên PC local cho **ba checkpoint** (mục 2), trên **BDD100K MOT val** (mục 3–7) và **UA-DETRAC test**
  (mục 8). Báo cáo rồi dừng.
- Không train thêm, không tuning theo test, không SSH Nano. Giữ nguyên split, dataset, checkpoint, cấu hình tracking và
  kết quả lịch sử. Chưa so sánh bộ lọc, thuật toán tracking hay tối ưu Nano.
- Không chạy đánh giá dev nội bộ: ngoài yêu cầu vòng này. Dev đã được dùng để chọn `best.pt`; kết quả dev của Ultralytics
  ở vòng 42 tách riêng khỏi kết quả vòng này.

## 2. Checkpoint (đã kiểm trực tiếp: `01_kiem_checkpoint.{log,json}`)

| Tên trong kết quả | Đường dẫn | sha256 | Lớp |
|:--|:--|:--|:--|
| `yolov8n_coco` (baseline) | `models/yolov8n.pt` | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | COCO 80 lớp |
| `yolov8n_ft` | `models/finetune_phase2/train_bdd5_v1_yolov8n_day_du/weights/best.pt` | `5f66776d3202389ca484f3d9e2f107dddb354d21a9f1762d19401d690a4a0d8a` | 5 lớp |
| `yolo11n_ft` | `models/finetune_phase2/train_bdd5_v1_yolo11n_day_du/weights/best.pt` | `d7d1c7e86d03c69d913febe79e4cbb6c2e1bcf9f17f034724af4de2ccfba7923` | 5 lớp |

Kiểm trực tiếp:
- sha256 tính từ file thực đều khớp;
- nạp bằng `tiep_tuc_huan_luyen.doc_checkpoint`, đều là `DetectionModel` FP16;
- mọi tensor số thực hữu hạn (298/298, 298/298, 418/418);
- tên lớp: hai checkpoint fine-tune đúng `car, truck, bus, motorcycle, bicycle` (ID 0..4); baseline là COCO 80 lớp.

Không dùng `last.pt` hay checkpoint khác. Lượt đánh giá tự kiểm lại sha256 trước khi nạp (`--sha256`, lệch thì exit 2).

## 3. BDD100K MOT val — dữ liệu (định danh đã cố định của dự án)

- **Manifest khóa** `results/phase2/manifests/bdd100k_transfer_cocovid_val.json` (sha256 `2071c15b…`): **toàn bộ 200 video
  val**, không loại video nào.
- **Đã được xem trước đây**:
  - BDD val đã dùng trong thí nghiệm transfer (tracking) và smoke;
  - 3 sequence `b1c66a42-6f7d68ca`, `b1c81faa-3df17267`, `b1c81faa-c80764c5` đã xem metric ở smoke (vòng 5–6).

  Vì vậy **không** mô tả đây là tập chưa từng xem. Tập này cũng không được dùng để chọn checkpoint.
- **Nhãn**: bản **COCO-video** của dự án `data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json`, 117.094.032 B,
  sha256 `074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d` (= manifest).
  - Không dùng native val.
  - Không sửa 183 khác biệt lớp / 50 ngoại lệ ignore so với native (nguồn gốc UNKNOWN, đã ghi ở vòng 33–34).
- **Ảnh**: `data/raw/bdd100k/images/track/val/<video>/`, 1280×720.
  - Mỗi lượt băm lại **nội dung** từng ảnh, không dùng lối tắt mtime, rồi đối chiếu với bảng đã lưu
    `data/interim/hash_tap_danh_gia/bdd100k_mot_val_sha256.csv` (sha256 `60fea2a9…`, 39.973 dòng).
  - Lệch bất kỳ ảnh nào thì exit 2.
- **Tập ảnh**: mọi ảnh mà file nhãn liệt kê cho 200 video, tức **39.973 ảnh**.
  - Gồm cả **615 ảnh không có annotation nào**: 488 ảnh nằm trong khoảng có nhãn của video, 127 ảnh ở đuôi video.
  - Danh sách phải trùng khớp tập `.jpg` trên đĩa; thiếu hoặc thừa ảnh thì exit 2.
  - Không bỏ ảnh âm tính, ảnh khó hay frame không annotation.
- **Thống kê nhãn** (đếm từ file nhãn trước khi chạy; **không phải metric**):
  - GT 5 lớp (không crowd, không ignore, trước khi cắt khung): car 341.436, truck 27.343, bus 9.087, motorcycle 902,
    bicycle 4.135.
  - Vùng bỏ qua: 20.848, gồm crowd/ignore của car 20.079, truck 481, bus 4, motorcycle 13, bicycle 271.
  - Số chính xác sau khi cắt khung do lượt chạy ghi lại.

## 4. Quy tắc GT, ánh xạ lớp, crowd / ignore (kế thừa quy tắc đã nghiệm thu)

- **GT** theo đúng quy tắc của exporter 5 lớp (`src/bdd_yolo_export.py`, sha256 `1ca964b0…`; dùng cho train/dev, đã nghiệm
  thu vòng 27–29, 38). Nhãn đọc bằng `NguonCocoVid`; giữ nguyên `iscrowd` / `ignore` của file.
  - `vai_tro`: car / truck / bus / motorcycle / bicycle không crowd, không ignore → **GT** lớp 0..4.
  - crowd/iscrowd thuộc 5 lớp, và distractor nhóm xe (`ignore=1` với car / truck) → **vùng bỏ qua**.
  - pedestrian, rider, train, other person → không GT, không vùng bỏ qua.
  - `_hop_le_box`: cắt box vào khung ảnh; box nhỏ hơn 1 px sau khi cắt thì bỏ và đếm riêng. Toạ độ pixel theo kích thước
    ảnh thật; khác w/h khai báo trong nhãn thì exit 2.
- **Không tô xám ảnh val**: ảnh giữ nguyên byte. Vùng bỏ qua chỉ dùng khi chấm: detection không ghép được mà IoA ≥ ngưỡng
  thì bị bỏ qua, như crowd của COCO.
  - Khác với dev, nơi exporter đã tô xám các vùng này ngay trong ảnh. Đây là khác biệt đã biết giữa hai phép đo.
- **Ánh xạ lớp của model → 5 lớp theo TÊN** (`detector_classes.anh_xa_model_sang_lop5`):
  - baseline COCO: 2 car, 7 truck, 5 bus, 3 motorcycle, 1 bicycle;
  - model fine-tune: 0..4.

  Không dùng chung chỉ số lớp giữa hai loại checkpoint. Lớp khác của baseline bị lọc **trước** NMS.
- **Lớp được đánh giá**: cả 5 lớp, vì đều có GT trong BDD val. **mAP** = trung bình 5 lớp (macro).

## 5. Thiết lập dự đoán và chấm điểm (kế thừa `src/eval_detector_5lop.py`, sha256 `30a1fd5a…`, không đổi từ vòng 28)

- **Dự đoán**: hàm `du_doan_ultralytics` của evaluator, gọi với cùng tham số tường minh cho cả ba model; không lấy thiết
  lập riêng từ checkpoint.
  - `imgsz 640`, `conf 0.001`, NMS IoU `0.7` theo lớp (không agnostic), `max_det 300`/ảnh, FP32 (`half False`);
  - batch 16, GPU `device 0`;
  - ultralytics 8.4.141, torch 2.6.0+cu124.
- **Chấm**: hàm `danh_gia` của evaluator.
  - Ghép như `COCOeval.evaluateImg`; IoU như `bbIou`; hòa IoU chọn GT sau.
  - Vùng bỏ qua theo IoA ≥ ngưỡng.
  - AP nội suy 101 điểm recall ở 10 ngưỡng IoU 0.50:0.05:0.95. Báo **AP50**, **AP75**, **AP50–95**; lớp không có GT → null.
- **Điểm vận hành**: precision / recall / F1 theo lớp tại **conf ≥ 0,25 và IoU 0,5**, tức ngưỡng tracker c25 đã nghiệm
  thu. Macro là trung bình các lớp có GT.
- Ba model dùng **cùng ảnh, cùng GT, cùng thiết lập**.

## 6. Tổng hợp và báo cáo

- **Bảng theo lớp**: AP50, AP50–95, số GT, số ảnh có lớp; mAP50 / mAP50–95 trên 5 lớp; chênh lệch của từng model fine-tune
  so với baseline, tính bằng điểm AP. P / R / F1 tại điểm vận hành ở mục 5.
- **Chẩn đoán mô tả** (không đổi metric): số detection conf ≥ 0,25 trên ảnh không có GT 5 lớp.
- **Không**:
  - so trực tiếp mAP giữa các dataset có tập lớp khác nhau;
  - kết luận "cải thiện có ý nghĩa thống kê", vì mỗi model chỉ có một lượt / một seed;
  - trộn kết quả dev nội bộ (dùng để chọn `best.pt`) với kết quả lần này;
  - đổi checkpoint hay quay lại train dựa trên test.
- Mọi bảng được đối chiếu lại với `ket_qua.json` / `theo_lop.csv` gốc trước khi bàn giao.

## 7. Cách chạy

- **Điểm vào** `src/danh_gia_detector_bdd_val.py` (mới, vòng 43; sha256 `d1e2fd22…`).
  - Lý do: evaluator đã nghiệm thu không có đường đọc BDD val (`02_tai_hien_chan_bdd_val.log`: `--split test` exit 2, chỉ có
    `data.yaml` của train / dev).
  - File mới **chỉ thêm phần đọc GT**. Dự đoán, ghép, AP, điểm vận hành và lưu dự đoán dùng nguyên hàm của evaluator.
  - Test hồi quy `src/test_danh_gia_bdd_val.py` 19/19 (`03_test_danh_gia_bdd_val.log`). Trong đó B2 chứng minh GT đọc trực
    tiếp **trùng** đường exporter → evaluator đã nghiệm thu trên cùng frame.
  - Suite liên quan `test_detector_5lop`: **92/92**, exit 0 (`03_test_detector_5lop.log`). Mã cũ không bị sửa.
- **Lệnh**: mỗi lượt một model, tuần tự baseline → `yolov8n_ft` → `yolo11n_ft`, tránh thiếu RAM.

  ```text
  python -u src/danh_gia_detector_bdd_val.py --model <ten>=<duong dan> --sha256 <ten>=<sha256 muc 2> \
      --out results/phase2/detector_eval/bdd_val_cocovid/r43_<ten>
  ```

  Dữ liệu dùng giá trị mặc định của điểm vào, đúng các đường dẫn ở mục 3.
- **Khởi chạy độc lập** với phiên Claude: WMI `Win32_Process.Create` → `src/chay_giam_sat.py chay` (log, RAM đúng PID và
  tiến trình con, exit code thật) → lệnh trên.
  - Thư mục giám sát: `results/phase2/detector_eval/giam_sat/r43_<ten>/`.
  - Thư mục kết quả `results/phase2/detector_eval/` là **mới**; không ghi đè kết quả cũ.
- **Mỗi lượt ghi lại**:
  - lệnh, thời điểm, exit code;
  - số ảnh dự kiến 39.973 và số đã dự đoán;
  - fingerprint của model (sha256), của GT (sha256 nhãn, sha256 danh sách ảnh, sha256 tổng hợp nội dung ảnh, đối chiếu
    bảng hash), của cấu hình (`quy_tac`, `quy_tac_gt`) và của evaluator (sha256 mã ở mục 4–5, ghi trong log);
  - dự đoán `du_doan_<ten>.npz` đủ để tái tính metric; `ket_qua.json`, `theo_lop.csv`.
- **Không tái dùng** dự đoán hay cache cũ: chưa có cache detection 5 lớp nào cho BDD val; cache tracking cũ khác conf,
  khác định dạng.
- **Không chạy smoke trên dữ liệu thật**: đường chạy đã kiểm bằng fixture; lượt đầu tiên là lượt thật.
- **Khi lỗi**: giữ thư mục và log. Lỗi kỹ thuật thì sửa, có tái hiện và test hồi quy, rồi chạy lại vào thư mục mới
  (`…_lan2`). Không đổi tham số sau khi xem kết quả, không chạy biến thể.

## 8. UA-DETRAC test — CHƯA chạy, chờ quyết định ánh xạ lớp của người dùng

- **Manifest khóa** `results/phase2/manifests/detrac_test.json` (sha256 `ff2b4b9c…`): 40 video test.
- **GT** XML `data/extracted/DETRAC-Test-Annotations-XML/` (thống kê, không phải metric):
  - 56.167 frame có trong XML, nhưng có 56.340 ảnh trên đĩa (**173 ảnh không có `<frame>`**);
  - `vehicle_type`: car 548.555, **van 38.519**, bus 71.785, **others 16.915**;
  - 163 vùng `ignored_region`;
  - **không có nhãn xe máy / xe đạp**.
- **Chưa có ánh xạ đã chốt**. Ánh xạ duy nhất trong mã (`prepare_mixed_yolo_dataset.DETRAC_MAP`: van→car, others→truck) tự
  ghi "GIẢ ĐỊNH", và cấu hình chứa nó (`mixed5.yaml`) ghi "không dùng trong plan đã chốt". Không tự đồng nhất van với
  truck, không tạo ánh xạ mới.
- **Quyết định cần người dùng**:
  1. `van`: vùng bỏ qua cho mọi lớp / GT lớp car / cách khác?
  2. `others`: vùng bỏ qua / GT lớp truck / cách khác?
  3. Lớp báo AP: car, bus (thêm truck nếu others→truck). motorcycle và bicycle là **N/A**; dự đoán của các lớp này không
     tính FP.
  4. 173 ảnh không có `<frame>` trong XML: tính là ảnh âm tính, hay loại và ghi rõ?
  5. `ignored_region`: đề xuất dùng làm vùng bỏ qua theo IoA như evaluator.
- Đọc GT UA-DETRAC cũng cần một bộ đọc tương tự mục 7. Bộ đọc này chỉ viết **sau** khi có quyết định, kèm test, rồi khóa
  phụ lục kế hoạch trước khi chạy.

## 9. Tiêu chí lượt chạy hợp lệ (kỹ thuật, không phải chất lượng model)

- exit 0;
- 39.973 ảnh đã dự đoán;
- kiểm dữ liệu đạt: sha256 nhãn, tập video, tập ảnh, w/h, nội dung 39.973/39.973;
- sha256 checkpoint khớp.

## 10. Phụ lục sửa đổi sau khi khóa

(trống lúc khóa)

- **Vòng 46 (đính chính mô tả, không đổi giao thức BDD đã chạy)**: mục 8, ý 3, câu "dự đoán của các lớp này không
  tính FP" không đúng với scorer `danh_gia`. Scorer vẫn đếm TP / FP thô và precision cho lớp không có GT; chỉ AP /
  recall / F1 = null và macro loại lớp đó (review `d4715aa`, R45-F1; fixture `logs/review46/test_cong_cu_detrac.py`
  E1). Mục 8 là phần chờ quyết định UA-DETRAC, chưa khóa hay chạy. Phần BDD (mục 1–7, 9) không đổi; kết quả BDD
  không bị ảnh hưởng vì cả 5 lớp đều có GT.
