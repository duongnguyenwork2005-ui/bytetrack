# Review nội dung delta a8ecaf6 → 5c170b3

Ngày review: 2026-09-17. Kết luận: **CẦN SỬA TÀI LIỆU VÀ GIAO THỨC ĐO TRƯỚC KHI CHỐT**.

Diff không thay đổi code tracking, evaluator hay cấu hình thí nghiệm. Các kết luận chính về A/B/C vẫn phù hợp với số liệu đã đối chiếu. Chưa có căn cứ từ đợt này để yêu cầu chạy lại UA-DETRAC. Có năm nhóm vấn đề cần chỉnh trong phần bàn giao và hướng dẫn mốc 3 dưới đây.

## Phạm vi và bằng chứng

- Đầu vào mới: `delta_a8ecaf6..5c170b3.diff`, 845 dòng; SHA256 `11f01aacbb04b523d25ab5b41dd75bc89618f428fc81df2910d516bf6c02a7ef`.
- `git apply --stat` / `--numstat`: 9 file, 647 dòng thêm, 68 dòng xóa. Gồm tài liệu, `review_base.json` và hai CSV audit.
- Đối chiếu với nguồn đã có từ gói snapshot `7341812` và delta target `9a1c404`: code CLI/timer, các CSV A/B/C, JSON profile. Riêng bảng B `_fix1` đã đối chiếu cả bản trong delta `9a1c404`.
- Ba tài liệu mới dựng được hoàn chỉnh trong bộ nhớ từ diff, git blob khớp header: inventory `a27479a`, review trước `748aa23`, final summary `02c1215`.
- **Không có ZIP delta mới, MANIFEST hay cây Git đầy đủ của a8ecaf6/5c170b3 trong lần gửi này.** Vì vậy đây là review nội dung diff và đối chiếu với nguồn đã bàn giao, chưa phải xác nhận hash/toàn bộ snapshot target. Nếu nguồn hiện tại khác các file đối chiếu, cần cung cấp đúng file và hash đó.
- Không sửa source, kết quả thí nghiệm, index hay commit. Không chạy lại full suite, detector hoặc benchmark.

## 1. [P2] Hướng dẫn BDD có lệnh không được runner hỗ trợ và tên manifest sai

Vị trí: `handoff/BDD100K_INVENTORY.md`, mục 3, dòng 63 và 71; tên manifest lặp lại ở mục 4.

Lệnh smoke mới ghi:

```text
python src/run_experiment.py --exp configs/phase2/experiments/bdd_transfer_val.yaml --videos <v1> <v2> <v3>
```

Trong code runner đã bàn giao, argparse chỉ có `--exp`, `--stage`, `--dry-run`, `--no-skip`, `--force`, `--force-eval`. Tái hiện bằng chính phần argparse trích từ source, thay placeholder bằng `v1 v2 v3`, trả **exit 2: unrecognized arguments: --videos v1 v2 v3**. Đây là kiểm tra parser độc lập, không phải chạy toàn bộ runner trên dataset thật.

Ngoài ra, `check_bdd100k.py --part train --make-manifest` ghi `results/phase2/manifests/bdd100k_dev.json`; tài liệu lại hứa tạo `bdd100k_dev_train.json`. Hai tên cần thống nhất.

Cách sửa gọn: dùng YAML smoke riêng với manifest 1–3 sequence và tên experiment/output riêng, rồi chạy bằng giao diện runner đang hỗ trợ. Giữ manifest đánh giá đầy đủ riêng biệt. Kiểm tra tên file thực tế và kế hoạch chạy trước khi ghi rằng phần mềm đã sẵn sàng. Không cần mở rộng CLI nếu sửa hướng dẫn đã đủ.

Nguồn tải trong inventory cũng cần cập nhật: cổng Berkeley đã lỗi trong lần người dùng thử. Ghi rõ tình trạng ảnh và nhãn tracking riêng; chỉ có ảnh validation chưa đủ để chạy đánh giá có GT.

## 2. [P2] Giao thức Nano hiểu sai các trường thời gian đang được đo

Vị trí: `scripts/nano/README.md`, mục 4c, điểm 5; `handoff/MOC2_FINAL_SUMMARY.md`, mục 6a.

README yêu cầu tách `predict_ms`, `assoc_ms`, `update_ms` như ba pha độc lập và cho rằng `assoc` phải gần bằng nhau giữa CV/EKF. Source hiện có đo:

```text
t_update = toàn bộ lời gọi tracker.update(...)
t_predict = thời gian tracker.multi_predict(...)
t_assoc = t_update - t_predict
```

Do đó `t_assoc` bao gồm cả cập nhật bộ lọc theo detection, quản lý track, association và phần việc còn lại trong update. Nó không phải thời gian ghép thuần túy. Trong `profile_pipeline.py`, tổng được đặt tên `tracker_ms`, không có trường thời gian correction riêng.

Cùng detection cache cũng không bảo đảm cùng số track hay cùng ma trận ghép: dự đoán khác nhau có thể làm thay đổi quyết định ghép và vòng đời. Chênh lệch `assoc_ms` không tự chứng minh phép đo sai.

Sửa định nghĩa trong tài liệu cho đúng timer hiện tại. Nếu cần đo riêng prediction/correction/association/lifecycle, bổ sung instrumentation ở mốc triển khai sau và xác minh hành vi tracking không đổi. Phân biệt đo toàn tracker với microbenchmark bộ lọc trên workload được kiểm soát.

## 3. [P2] Loại lượt throttling và dùng timing có tải nền để quy lợi ích cho bộ lọc có thể làm lệch kết luận

Vị trí: `scripts/nano/README.md`, mục 4c, điểm 2–3; `handoff/MOC2_FINAL_SUMMARY.md`, mục 6a–6b; brief mô tả cùng giao thức.

- Quy tắc “đánh dấu và loại” mọi lượt throttling bỏ mất một phần hiệu năng vận hành thực. Nếu một cấu hình gây nóng/throttling nhiều hơn, loại các lượt đó riêng lẻ có thể thiên lệch phép so sánh. Giữ dữ liệu mọi lượt, báo kết quả chạy liên tục, và có thể thêm bảng không throttling theo tiêu chí công bố trước. Nếu một cặp không đủ điều kiện cho bảng phụ, xử lý cả cặp và ghi số lượt bị loại.
- A/B/A/B luôn để B đứng sau A. Nên cân bằng AB/BA hoặc ngẫu nhiên hóa thứ tự có seed, cùng warm-up và trạng thái bắt đầu. Ghi cả nhiệt độ/tần số và tiêu chí phát hiện throttling cụ thể.
- Mục 6a thừa nhận máy chạy việc khác nhưng vẫn cho phép “so tương đối giữa các bộ lọc”; mức kiểm soát đó chưa đủ để quy chênh lệch tốc độ cho motion model. Ngay trong bảng, EKF của A khoảng 376 fps còn `bt-ekf` của B khoảng 502 fps, cho thấy cần xét điều kiện từng lượt.
- Khoảng chênh `detector_wall` 14,9–17,1 ms là chênh giữa các lần profile quan sát được, chưa phải ước lượng nhiễu từ phép đo lặp có kiểm soát. Viết rằng nó góp phần giải thích thứ tự fps quan sát được; không dùng nó để khẳng định quan hệ nhân quả hoặc mức nhiễu Nano khi chưa đo thiết bị.

Có thể sửa cách diễn giải ngay. Benchmark ghép cặp thực hiện khi có môi trường phù hợp, không cần chạy lại các thí nghiệm độ chính xác A/B/C vì lỗi tài liệu này.

## 4. [P2] Một số số liệu chốt lệch file nguồn hoặc bị làm tròn hai lần

Vị trí: `handoff/MOC2_FINAL_SUMMARY.md`, mục 3, 4, 5, 6 và 8.

| Mục | Giá trị nguồn đã đối chiếu | Cách ghi cần sửa |
| --- | --- | --- |
| B: HOTA `sort-cv30 − sort-orig30` | `-0.013249990950779833` | Làm tròn trực tiếp 4 chữ số: **−0.0132**, không phải −0.0133 |
| C: HOTA `ekf-maha − ekf` | `-0.00344992609689454` | Làm tròn trực tiếp 4 chữ số: **−0.0034**, không phải −0.0035 |
| PF500: SD HOTA theo seed trên test | c25 `0.0005402005941070576`; c10 `0.0004294957774909704` | Bỏ khẳng định **≤0,0004**; ghi hai giá trị, hoặc khoảng 0,00043–0,00054 |
| Tỷ trọng `detector_wall_ms / total_ms` trong bốn JSON profile | CV 76,56%; EKF 77,85%; UKF 74,04%; PF300 73,60% | Nếu dùng `detector_wall`, ghi khoảng **74–78%** thay cho 70–75%; nếu cộng các stage nội bộ detector, nêu rõ cách tính khác |

Nguồn hai dòng đầu: `expB_detrac_test_fix1/c25/paired_pairs.csv` và `expC_detrac_test/c25/paired_pairs.csv`, chọn `method=combined_resample_videos`. Nguồn SD: `expA_detrac_test/{c25,c10}/summary.csv`. Nguồn profile: `results/phase2/profile/Duong_yolov8n_bytetrack*_gpu.json`.

Hai sai lệch làm tròn có dấu hiệu làm tròn qua giá trị trung gian: `-0.01324999095 → -0.013250 → -0.0133` và `-0.003449926 → -0.00345 → -0.0035`. Phải định dạng một lần từ số đầy đủ. Các sửa này không đổi dấu CI hay kết luận định tính.

## 5. [P2] Một số câu kết luận vượt mức bằng chứng

Vị trí: `handoff/MOC2_FINAL_SUMMARY.md`, mục 2, 5 và 8.

- “UKF không hơn CV” nên thành **“chưa có bằng chứng UKF hơn CV có ý nghĩa trong các điều kiện đã thử”**. CI chứa 0 không chứng minh bằng nhau hay không có lợi ích.
- “Tăng theo N ⇒ suy giảm là hệ quả của số hạt” mạnh hơn đối chứng đã có. Viết **“tăng số hạt cải thiện HOTA trong implementation đã thử, nhưng chưa xác định toàn bộ nguyên nhân PF kém”**. Giữ giới hạn log ESS hiện đã giải thích đúng ở mục 5.
- Ba seed có biến thiên nhỏ cho thấy khoảng cách với EKF lớn hơn biến thiên đã quan sát, không chứng minh mọi ảnh hưởng ngẫu nhiên đều đã bị loại trừ.
- “Rào cản không nằm ở motion model” không suy ra từ việc ba bộ lọc đều đạt khoảng 6% ở tầng full. Kết luận phù hợp là **“thay CV bằng các bộ lọc đã thử chưa giải quyết được mất ID ở tầng che nặng; prediction_drift là một dạng thất bại nổi bật trong phân tích dev”**. Đây cũng nhất quán với các nhãn nguyên nhân có quy tắc mà báo cáo đang dùng.
- Nếu muốn khẳng định cải thiện C2 có ý nghĩa thống kê riêng ở tầng `full`, dẫn CI theo video của đúng tầng và đúng cặp. Bảng `occlusion_strata.csv` hiện được dẫn chỉ thể hiện tỷ lệ/số sự kiện, không tự cung cấp CI. Có thể chỉ ghi tỷ lệ tăng 11/182 → 22/182 nếu chưa bổ sung bằng chứng này.

## Những phần có thể giữ

- Nhóm A: các hiệu số và CI chính theo c10/c25 khớp nguồn; EKF hơn CV có ý nghĩa ở c10 và hiệu ứng nhỏ.
- Nhóm B dùng `_fix1`, mô tả cặp đổi cả biểu diễn và Q/R/P0; không diễn giải các hiệu số như một phép phân rã cộng.
- Nhóm C báo cáo cả preserved tăng và HOTA giảm, dùng đúng cặp đối chứng; kết quả âm được giữ.
- Audit mới có đúng **134 dòng = 21 OK + 113 LEGACY**. Không suy diễn thiếu fingerprint thành đã được xác minh. Có **5 dòng với video_meta > 0**; nếu giữ số 0/129 từ gói cũ, cần ghi rõ snapshot cũ để người đọc không nhầm trạng thái hiện tại.
- Quy ước BDD `occlusion_ratio=0.5` khớp adapter hiện có, với điều kiện ghi rõ đây là giá trị trung gian cho cờ nhị phân, **không phải đo được che 50%**. Không dùng để so mức che định lượng với UA-DETRAC. Việc giữ gap là `unknown`/`occluded_adjacent` thay vì mặc định che hoàn toàn là phù hợp.

## Bàn giao tiếp theo nên gọn

Sửa các tài liệu liên quan và đồng bộ brief/handoff. Kiểm tra lại tên manifest, giao diện CLI và cách làm tròn từ CSV đầy đủ. Nếu không đổi code, không cần chạy full suite hay benchmark để xác nhận sửa câu chữ. Nếu bổ sung khả năng CLI, kiểm tra riêng phạm vi đó và việc output smoke không ghi vào kết quả final.

Lần tới gửi nguyên `review_delta_latest.zip` nhỏ, có MANIFEST và các nguồn cần đối chiếu. Không cần gửi lại ZIP đầy đủ. Review này xác nhận đã đọc nội dung diff 5c170b3; **chưa xác nhận toàn bộ snapshot 5c170b3 để tự động đổi review base**.
