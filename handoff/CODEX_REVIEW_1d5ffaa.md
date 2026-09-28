# Review vòng 23 — 1d5ffaa

**Kết luận: PASS WITH LIMITATIONS cho phần tracking đủ 1.470 frame; REQUEST CHANGES cho bộ kiểm render mới (một finding P2).**

Bốn lượt Nano và các cặp live/replay có bằng chứng đạt. Lỗi mới nằm ở bộ kiểm video, không ảnh hưởng số liệu tracking hoặc benchmark. Chưa chốt toàn bộ target `1d5ffaa`; giữ base review `c4b39fe` đến khi đóng N23-F1.

## 1. N23-F1 — P2: video sai số frame vẫn được bộ kiểm báo ĐẠT

**Vị trí:** `src/kiem_video_render.py`, khối tạo `ky_vong`, `nen` và `mask` (dòng 73–85 trên target).

Cả `ky_vong` và `nen` đều đã được vẽ cùng số frame. Khi có bbox, `mask = np.any(ky_vong != nen, axis=2)` chọn vùng khác nhau của bbox/nhãn, loại vùng số frame chung giữa hai ảnh. Phép kiểm riêng vùng số frame chỉ chạy ở nhánh `else`, khi không có vùng bbox/nhãn khác nhau. Vì vậy tuy kết quả ghi “mọi frame: khung bbox/màu ID/số frame đúng chỗ”, số frame không được kiểm độc lập trên các frame có bbox.

**Tái hiện độc lập trên mã target, qua CLI thật:**

- Dùng fixture 30 frame của `test_render_mot_video.py`, có 27 frame chứa bbox và 3 frame không có bbox.
- Render một video đúng làm đối chứng.
- Tạo video thứ hai với cùng ảnh, bbox, ID, lớp, codec, FPS và số frame; chỉ thay chữ số frame thành **`f999` trên 27 frame có bbox**. Ba frame không có bbox giữ nguyên số đúng.
- Chạy `kiem_video_render.py` với MOT/lớp đúng.

| Đầu ra kiểm | Video đúng | Video sai số frame |
|:--|--:|--:|
| Số frame cố ý ghi sai số | 0 | **27/30** |
| Exit code của bộ kiểm | 0 | **0** |
| `dat` | true | **true** |
| Phép “mọi frame … số frame đúng chỗ” | true | **true** |
| `ti_le_loi_max` | 0,28000705 | **0,28000705** |

Script kèm báo cáo: `CODEX_REPRO_1d5ffaa_render_counter.py`. Chạy từ gốc repository bằng Python có OpenCV:

```bash
python /duong/dan/CODEX_REPRO_1d5ffaa_render_counter.py
```

Script tạo ảnh/video tổng hợp trong thư mục tạm, lưu JSON/log, không cần dataset hoặc Nano. Trên `1d5ffaa`, regression **exit 1** vì bộ kiểm đã chấp nhận video sai. Sau sửa, video đúng phải được chấp nhận và video sai phải bị từ chối.

**Ảnh hưởng tới bàn giao:** cả hai JSON kiểm video thật ghi `n_frame_co_box = 300`. Nhánh kiểm riêng số frame chưa được thực hiện cho các frame này. Điều đó không chứng minh video Nano thực tế sai số frame; nó chứng minh kết luận kiểm chứng hiện tại chưa đủ để bảo đảm số frame đúng.

**Sửa cần thiết:** kiểm số frame riêng trên mọi frame, kể cả frame có bbox; thêm negative test cho sai/mất số frame trong khi bbox/ID/lớp vẫn đúng. Giữ test đối chứng video đúng. Sau đó chạy lại bộ kiểm trên hai MP4 Nano đã có và cập nhật kết luận. Không cần chạy lại bốn lượt tracking 1.470 frame hoặc build lại engine để sửa finding này.

## 2. Gói và phạm vi source

- Base: `c4b39fe3861baac76211b2cb82c62d970b719767`.
- Target: `1d5ffaa3e69a6d863289821148f8a2d3689b4144`.
- ZIP: **5.896.919 byte**, 98 thành viên.
- SHA256: `82be887b7cb91eecb77b6e2322e8d4c3b75a496406e983c67bcc9eb43d29c7c9`.
- `testzip` đạt; **92/92 payload** khớp hash và số byte. Áp patch lên snapshot base đã lưu: **107/107 đường dẫn** khớp hash target.
- Không thấy đường dẫn riêng bị cấm hoặc video/weights/engine/ZIP triển khai trong payload và patch.
- 4 NPZ mới bị loại khỏi delta; 9 NPZ thiếu từ các snapshot trước tiếp tục được khai báo. Reviewer không có 13 NPZ này.

Đối chiếu 790 file hiện có trong các phạm vi `src/`, configs, vendor, kết quả và log vòng 20–22: chỉ hai file cũ thay đổi là `render_mot_video.py` và `run_all_tests.py`. Hai module mới là `kiem_video_render.py`, `test_render_mot_video.py`. Luồng live, detector, tracker, runner và các bằng chứng cũ không đổi.

## 3. Phần tracking 1.470 frame — kiểm độc lập đạt

Nguồn: `handoff/logs/review23/nano_dai_v23/results_nano/dai_v23/`.

Reviewer so trực tiếp file gốc `data/MVI_39031.txt` và `data_cls/MVI_39031.csv` giữa mỗi lượt live và replay, không chuẩn hóa xuống dòng hoặc ánh xạ lại ID.

| Lượt | Frame xử lý | Hàng MOT | ID phân biệt | MOT + lớp live/replay |
|:--|--:|--:|--:|:--|
| 1 CV | 1.470 | 8.780 | 74 | **Byte-identical** |
| 2 EKF | 1.470 | 8.765 | 78 | **Byte-identical** |
| 3 EKF | 1.470 | 8.765 | 78 | **Byte-identical** |
| 4 CV | 1.470 | 8.780 | 74 | **Byte-identical** |

Hai lượt cùng cấu hình cũng có MOT giống từng byte:

```text
CV  5017f37bf2837e14fff6feb325ba5cf386ceff6700ff9fb9544acd3952cc4f06
EKF 645bba3a31905d875f952bfb7a03aa9f874006a0a74737c1fbb5e11e2a71bf56
```

Kiểm từng CSV xác nhận frame 1–1470 xuất hiện đúng một lần trong cả pipeline và timing của mỗi lượt. Mỗi lượt có **11 frame không còn detection sau lọc**, 0 frame rỗng ở mức detection thô; các frame này vẫn được xử lý. Số nhận/xử lý/EOS đều là 1.470; exit code live và child đều 0. Cả **5.880 frame** nằm trong cửa sổ đo tương ứng, số frame đang bay tối đa 2 với K=4.

**24/24 hash module** trong provenance của bốn lượt khớp source target theo LF. Trong danh sách 77 file nguồn Nano, **73/73 file hiện diện** khớp SHA256; 4 file thiếu là NPZ đã khai báo. Log ghi ảnh đầu vào khớp PC 1470/1470; reviewer không trực tiếp hash ảnh vì ảnh không nằm trong delta.

Tính xác định của MOT đã được xác nhận trực tiếp. Tính xác định của detection cache giữa bốn lượt hiện được hỗ trợ bởi hash/log nguồn bàn giao; reviewer không tự so nội dung NPZ vắng mặt.

## 4. Timing và telemetry

Tính lại từ CSV và mốc gốc, không gọi script phân tích của tác giả. FPS xử lý = 1470 / `xu_ly_s`; FPS ổn định dùng timestamp kết thúc frame 25 và 1470. Trung vị stage bỏ 25 frame đầu, còn 1.445 frame/lượt. Các số khớp metadata trong sai số số thực.

| Lượt | FPS xử lý | FPS ổn định | Decode trung vị (ms) | Update trung vị (ms) | Trễ trung vị (ms) |
|:--|--:|--:|--:|--:|--:|
| CV 1 | 9,3103 | 9,3106 | 11,464 | 6,735 | 115,253 |
| EKF 2 | 9,0488 | 9,0451 | 16,315 | 7,929 | 120,023 |
| EKF 3 | 9,0449 | 9,0480 | 16,255 | 7,999 | 120,334 |
| CV 4 | 9,1367 | 9,1411 | 11,867 | 6,780 | 116,916 |

Theo cấu hình, trung vị của hai lượt: CV **9,2235 FPS / 6,7573 ms update**; EKF **9,0469 FPS / 7,9637 ms update**. Đây là các thống kê mô tả, chưa đủ để xếp hạng hiệu năng toàn luồng; thời gian decode khác nhau rõ giữa các lượt. FPS xử lý không gồm khởi tạo, kết thúc, ghi file hoặc render.

Đọc lại tegrastats thô: RAM tối đa **2.667 MB**, swap tối đa **16 MB**, nhiệt **CPU tối đa 38 °C**, **GPU tối đa 35,5 °C**. Không nên viết “mọi nhiệt độ ≤38 °C” vì các cảm biến khác có giá trị cao hơn. Bảng tài nguyên ghi boot không đổi, ext4 errors 0 và không còn child sau mỗi lượt.

MemAvailable trước lượt đầu đến sau lượt cuối tăng **2.676 kB**; so các mốc trước lượt 1 và trước lượt 4 tăng **3.008 kB**, giải thích số “+3 MB” trong báo cáo. Không thấy tích lũy bộ nhớ hệ thống giữa các lượt này; không suy ra đã loại trừ mọi rò rỉ bộ nhớ.

**Cần diễn đạt chính xác phạm vi ổn định:** đây là bốn lượt, mỗi lượt khoảng **158–163 giây xử lý**, trong một phiên khoảng **13 phút 33 giây**, có nghỉ và replay giữa lượt. Không phải một tiến trình tracking chạy liên tục 14 phút, cũng chưa phải kiểm chạy nhiều giờ.

## 5. Render và mức bằng chứng kiểm thử

- Reviewer tự chạy `test_render_mot_video.py`: **7/7 PASS, exit 0**, Python 3.12.14 và OpenCV headless 4.11.0.86. Bộ test hiện có chưa bao phủ N23-F1.
- Regression bổ sung N23-F1 xác nhận false pass như mục 1. Đây là lỗi bộ kiểm, không phải bằng chứng tracker sai hoặc hai video thật bị hỏng.
- Full suite **35/35 PASS** được đối chiếu từ log bàn giao `tests_20260928-1349.log`, HEAD `cf8b760`; không phải 35 suite do reviewer chạy lại.
- Log Nano ghi render 300 frame, 960×540, 25 fps bằng OpenCV 3.2 trên Python hệ thống. Hash MOT/lớp trong hai sidecar render khớp chính output live đã kiểm. Có bằng chứng công cụ render đã chạy trên Nano.
- Hai MP4 không nằm trong delta, nên reviewer chưa xem trực tiếp video thật. Kết luận “kiểm mọi số frame đúng” cần đợi sửa N23-F1 và kiểm lại hai file hiện có. Kết quả lấy mẫu tên lớp được giữ đúng phạm vi 255 box/video, không mở rộng thành kiểm mọi nhãn.

## 6. Yêu cầu đóng vòng

1. Sửa N23-F1 và đưa fixture âm tính vào suite render; giữ nguyên ngưỡng codec trừ khi có bằng chứng riêng cần thay đổi.
2. Chạy lại bộ kiểm trên hai MP4 Nano đã tạo, lưu JSON/log mới; nếu phát hiện sai mới sửa/render lại video. Không chạy lại tracking chỉ vì thay bộ kiểm render.
3. Cập nhật cách diễn đạt thời gian vận hành và nhiệt độ như mục 4. Giữ nguyên số liệu bốn lượt Nano và các kết quả nghiên cứu đã khóa.
4. Bàn giao delta sửa nhỏ từ `c4b39fe`; chỉ chuyển base sau review tiếp theo. Hiển thị trực tiếp/camera vẫn là phần việc tiếp theo, chưa được nghiệm thu trong vòng này.

Reviewer làm việc trên bản sao trong workspace, không sửa source triển khai, commit, index hoặc Nano. Không yêu cầu mở lại tuning hay các benchmark đã đạt để xử lý finding này.
