# Audit demo `track_buffer=90` với xe di chuyển rõ ràng

## Kết luận

**Không tìm được demo xe đang di chuyển đủ tin cậy với buffer=90.**

Không ứng viên nào vượt qua đồng thời toàn bộ tiêu chí cứng. Vì vậy không có video hoặc ba keyframe thay thế được dựng, và không có ứng viên nào được replay/xếp hạng theo fused similarity. Demo cũ `MVI_40871`, GT 1 không được tái sử dụng: xe chỉ đạt 0,871 px/frame, path 10,455 px và net/median bbox diagonal 0,057.

## Xác nhận nguồn và cấu hình

Ba thư mục kết quả `sweep-cv-buf90`, `sweep-ekf_ctrv-buf90`, `sweep-ukf_ctrv-buf90` đều có đủ 60 file không rỗng trên `DETRAC-all/train`. Cả ba dùng cùng detector YOLOv8n, weights `models/yolov8n.pt` (SHA-256 `F59B3D833E2FF32E194B5BB8E08D211DC7C5BDF144B90D2C8412C47CCFC83B36`), `conf=0.25`, classes `[2, 3, 5, 7]`, seed mặc định Ultralytics 0 và `deterministic=true`.

| Thuộc tính | KF/CV | EKF/CTRV | UKF/CTRV |
|---|---:|---:|---:|
| YAML | `cv_buf90.yaml` | `ekf_ctrv_buf90.yaml` | `ukf_ctrv_buf90.yaml` |
| `tracker_type` | `bytetrack` | `bytetrack_ekf_ctrv` | `bytetrack_ukf_ctrv` |
| `track_high_thresh` | 0.25 | 0.25 | 0.25 |
| `track_low_thresh` | 0.10 | 0.10 | 0.10 |
| `new_track_thresh` | 0.25 | 0.25 | 0.25 |
| `track_buffer` | 90 | 90 | 90 |
| `match_thresh` | 0.80 | 0.80 | 0.80 |
| `fuse_score` | true | true | true |

Ba YAML chỉ khác `tracker_type`, tức motion model. Đây là cấu hình Stage E mở rộng, không phải A/B nguyên bản: `conf=0.25`, `track_buffer=90`.

## Kết quả lọc

Đánh giá hiện tại tạo 3.349 đoạn che khuất thô và gộp 583 đoạn trùng thành 2.766 sự kiện. Có 1.003 sự kiện đủ GT trước/sau để đánh giá.

Luồng lọc tuần tự:

1. Tổng sự kiện gộp: **2.766**.
2. Đủ điều kiện đánh giá: **1.003**.
3. EKF `id_match=1` và CV `id_match=0`: **13**.
4. Đồng thời đạt `speed≥3`, `path≥30`, `net/median_bbox_diagonal≥0,5` và GT đầy đủ: **5**.
5. Lần tái ghép đầu tiên liên quan trực tiếp tới sự kiện, nằm trong `end+1…end+5`: **1**.
6. Đồng thời ổn định EKF trong ±25 frame và có đủ dữ liệu association: **0**.

Không có ứng viên cuối nên thứ tự xếp hạng `EKF−CV`, normalized net displacement, thời lượng che khuất không được áp dụng. UKF không tham gia bất kỳ điều kiện lọc nào.

## Năm trường hợp vượt ngưỡng chuyển động

| Video / GT | Sự kiện | Speed | Path | Net/diag | Tái ghép EKF | Kết quả |
|---|---:|---:|---:|---:|---:|---|
| `MVI_39801` / 37 | 701–709 | 13,345 | 106,764 | 0,839 | 712 | Loại: chỉ có 2 frame EKF được ghép sau tái xuất trong cửa sổ +25, không đủ chứng minh ổn định |
| `MVI_40152` / 15 | 306–323 | 5,053 | 85,906 | 1,094 | 314 | Loại: tái ghép nằm trong đoạn che khuất |
| `MVI_40201` / 47 | 283–308 | 11,035 | 275,880 | 2,495 | 284 | Loại: tái ghép nằm trong đoạn che khuất |
| `MVI_40241` / 325 | 2255–2291 | 10,399 | 374,360 | 3,794 | 2280 | Loại: tái ghép trong đoạn che khuất và chuỗi ID EKF ±25 đổi `2008→1991` |
| `MVI_40752` / 10 | 122–128 | 7,858 | 47,147 | 0,612 | không có lần hợp lệ | Loại: lần tái ghép gần nhất không bắt nguồn từ sự kiện này |

Trường hợp gần nhất là `MVI_39801`, GT 37: tái ghép frame 712, tức `end+3`, và đạt đủ ba ngưỡng chuyển động. Tuy nhiên EKF chỉ được quan sát đúng ID 357 ở 2 frame sau tái ghép trong cửa sổ +25; đây có thể là một ghép ngắn, nên không đạt điều kiện ổn định. Outcome đã lưu là CV switched 310→374, EKF preserved 357→357, UKF switched 358→389; cả ba đều `id_continuous=0`. Không replay hay dựng video từ trường hợp này.

## Ý nghĩa các trường đánh giá

- `id_match=1`: ID đại diện trước và sau sự kiện giống nhau.
- `id_continuous=1`: cùng ID tồn tại tại mọi frame có GT trong sự kiện.
- Một ID trước/sau giống nhau nhưng chỉ xuất hiện rất ngắn sau tái ghép có thể đạt `id_match` mà vẫn không đủ ổn định để làm demo.

Góc rẽ chỉ được tính khi net displacement đạt ít nhất 0,5 median bbox diagonal; các hàng còn lại ghi `khong dang tin` và để trống góc. Góc không được dùng để lọc hoặc xếp hạng.

## Artefact

- `candidates_buffer90_moving.csv`: đủ 1.003 sự kiện, các đại lượng chuyển động, từng điều kiện pass/fail và lý do loại.
- `evaluation/`: dữ liệu sự kiện gộp và paired outcomes dùng để lập bảng.
- `metadata.json`: cấu hình, số lượng và kết luận không có ứng viên.
- `match_cost_at_reacquisition.csv`: chỉ có header; không có hàng vì không có ứng viên đủ điều kiện để replay.

Không tạo `demo_ekf_buffer90_moving.mp4`, `before_occlusion.png`, `during_occlusion.png` hoặc `reappearance.png`, vì tạo chúng sẽ ngụ ý tồn tại một demo hợp lệ. Không file nào trong `outputs/thesis_demos_buffer90/` bị sửa hoặc ghi đè.
