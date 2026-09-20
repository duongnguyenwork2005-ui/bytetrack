**CODEX REVIEW — `074ceca` — 2026-09-20**

**Kết luận: PASS WITH LIMITATIONS cho mốc hệ thống PC Windows.** Hai ca còn lại của review `923305d` đã được đóng bằng kiểm tra độc lập. Có thể chốt phiên bản PC và chuyển sang launcher/giao diện sử dụng trên pipeline hiện có.

Phạm vi kết luận: sửa timing khi dừng sớm, replay tên có dấu cách, bảo toàn hành vi runner và toàn vẹn gói. Đây không phải xác nhận triển khai/tốc độ Nano; Nano F1/F3 vẫn hoãn. Full suite và smoke YOLO/GPU của tác giả được đối chiếu từ log, không được mô tả là reviewer đã tự chạy lại trên GPU.

**1. Snapshot được nghiệm thu**

| Mục | Kết quả kiểm độc lập |
|---|---|
| Base gói | `b6234653f616e8059825b59b4da70262e1f98dde` |
| Target | `074ceca1ba36777cabc02a54d6699e890b19c51b` |
| ZIP | **6.360.571 byte**, 83 thành viên; `testzip` OK |
| SHA256 ZIP | `fa03d7ea02628a9618063ec09dcdeaa72bf4fe5b29a509a9e26e46bece07dbbd` |
| Payload/dedup | **77/77** payload đúng hash; **13** đường dẫn dedup có nội dung thay thế đúng hash |
| Patch | Kiểm và áp thật exit 0; **90/90** đường dẫn target khớp hash |
| File riêng/dữ liệu lớn bị loại | Không có file thuộc đường dẫn cấm trong thành viên ZIP hoặc đường dẫn đích của patch |
| Kết thúc review | 90/90 hash target giữ nguyên; không sửa source sản phẩm, commit, push hoặc index |

Reviewer dựng bản sao từ snapshot base đã lưu, không phải checkout Git đầy đủ của máy tác giả. README gốc được phục hồi bằng reverse patch rồi kiểm hash base; README Nano lấy từ snapshot cũ với đúng hash. Các fixture và file kết quả kiểm tra được tạo riêng trong môi trường review.

**2. Đóng ca dừng sớm bằng `q`**

Đã kiểm mã: `track_video()` giữ iterator và gọi `close()` trong `finally`; adapter chốt timing trong `finally` quanh `yield`. Thân vòng lặp được tách ra nhưng giữ cách dự đoán, cập nhật và ghi hàng tracking.

Kiểm độc lập qua `track_video_file.main()` thật, tracker NumPy/cache thật. Image reader, detector và GUI là test doubles; bộ đọc mất khoảng 20 ms/frame, `waitKey` có độ trễ kiểm soát và trả `q` ở frame 1 hoặc 2. Không tuyên bố đây là thao tác tay trên cửa sổ thật.

| Ca | Frame xử lý | Timing không hữu hạn | Show frame cuối | Tổng frame cuối | Wall ngoài frame | Replay |
|---|---:|---:|---:|---:|---:|---|
| Chạy hết 3 frame | 3 | 0 | 4,096855 ms | 24,970706 ms | 1,428218 ms | PASS, lệch 0 |
| `q` ở frame 1 | 1 | **0** | **4,086556 ms** | **24,862594 ms** | **0,825246 ms** | PASS, lệch 0 |
| `q` ở frame 2 | 2 | **0** | **4,095803 ms** | **25,067796 ms** | **0,614634 ms** | PASS, lệch 0 |

Các giá trị trên là phép kiểm vị trí/hoàn tất timer bằng độ trễ giả lập, không phải benchmark hiệu năng PC.

Các điều kiện đã xác nhận:

- `frame_total_ms`, `overhead_ms` và các stage đều hữu hạn; số dòng tổng hợp bằng số frame đã xử lý.
- Tổng stage + overhead khớp tổng mỗi frame; sai số tối đa trong bốn fixture tích hợp là **7,11e-15 ms**. Các trung bình tính lại từ CSV khớp metadata.
- Đã giữ **tham chiếu mạnh tới generator** sau khi hàm trả về; generator vẫn đóng ngay. Do đó việc chốt timing không dựa vào GC hay xóa biến.
- Không chạy detector cho frame tiếp theo khi dừng: số lần detector là 1 hoặc 2 đúng như yêu cầu. Bộ đọc có thêm đúng một lần đọc đầu để xác định kích thước nguồn, ngoài các frame thực xử lý.
- Tracking/cache đủ frame cuối; replay bằng subprocess cho cùng ID, bbox/conf lệch 0.
- Manifest phân biệt `nguon` có 3 ảnh và `da_xu_ly` có 1/2 frame, cùng trạng thái dừng sớm.
- Kiểm thêm callback ném lỗi có kiểm soát: lỗi được truyền ra, generator đóng ngay, chỉ đọc 1 frame. Không có thao tác nuốt lỗi hoặc tiếp tục đọc thêm.

**Kết luận: phần còn lại của PC-F1 đã đóng.**

**3. Đóng ca tên nguồn/YAML có dấu cách**

Đã chạy live với:

- Nguồn ảnh `traffic clip`.
- YAML nằm tại `cfg dir/ekf ctrv.yaml`.
- Khi replay, tên split là `REVIEW FINAL`, tên tracker cũng có dấu cách.

Replay dùng **chính `replay_argv` trong metadata**, chỉ thay hai phần tử placeholder đã được hướng dẫn, rồi gọi **`subprocess.run(argv, shell=False)` thật**. Không dựng lại lệnh khác và không dùng `.split()`.

| Ca | Live | Replay subprocess | Tracking |
|---|---:|---:|---|
| Chạy hết, tên nguồn/YAML có dấu cách | exit 0 | exit 0 | 3/3 hàng, ID khớp, bbox/conf lệch 0 |
| Dừng ở frame 1 | exit 0 | exit 0 | 1/1 hàng, lệch 0 |
| Dừng ở frame 2 | exit 0 | exit 0 | 2/2 hàng, lệch 0 |
| Tên có dấu cách + `--allow-truncated` | exit 0 | exit 0 | 3/3 hàng, flag được giữ, lệch 0 |

Trên môi trường POSIX của reviewer, chuỗi hiển thị được parse lại bằng `shlex.split` cũng khôi phục đúng toàn bộ `replay_argv`. Chưa thực thi chuỗi hiển thị qua cmd/PowerShell của Windows; đường argv không qua shell đã được kiểm độc lập.

**Kết luận: phần còn lại của PC-F2 đã đóng.**

**4. Runner sau khi tách vòng lặp giữ hành vi trong các ca đã kiểm**

Reviewer nạp hai bản `run_tracking.py` của `923305d` và `074ceca`, chạy trên cùng dữ liệu tổng hợp và cùng backend NumPy. Chuỗi kiểm có detection rỗng, frame không đọc được và khoảng nhảy chỉ số frame. Không so bằng nhau các cột thời gian thực thi vốn biến động.

| Cặp kiểm | Hàng tracking mỗi bên | Kết quả |
|---|---:|---|
| CV, `elapsed_mode=skip` | 50 | Bit-identical |
| CV, `elapsed_mode=steps` | 50 | Bit-identical |
| EKF, `elapsed_mode=skip` | 50 | Bit-identical |
| EKF, `elapsed_mode=steps` | 50 | Bit-identical |

So byte mảng tracking: **4/4 khớp**, mỗi ca 2 track. Bảng lớp và các cột `frame/n_steps/n_tracks/n_dets` cũng khớp tuyệt đối. Đây là bằng chứng trên fixture dùng tracker thật, không thay thế phép so toàn bộ dataset thật.

Log bàn giao bổ sung bằng chứng dữ liệu thật: `test_run_tracking_replay` **10/10**, `test_bytetrack_np_parity` **10/10**, full suite **30/30 PASS**. Reviewer đã đối chiếu CSV tổng kết và log được cung cấp; không tự chạy lại các test có detector/dataset thật này.

**5. Kết quả, tài liệu và provenance**

- **44 file artifact PC**, gồm bốn run vòng trước và bốn run `v2`, giữ nguyên byte so với snapshot `923305d`. Các kết quả audit timing `v2` ở review trước vẫn áp dụng; không cần đo lại chỉ vì đợt sửa này.
- Delta không có thay đổi ở đường dẫn kết quả nghiên cứu UA-DETRAC/BDD, GT hoặc lock. Đây là kết luận về phạm vi gói được cung cấp, không phải kiểm trực tiếp toàn ổ đĩa Windows.
- `nano_parity.py` chỉ sửa docstring trong vòng này; phần AST thực thi không đổi. Reviewer chạy độc lập **20/20 test parity PASS**, exit 0.
- Các diễn giải confidence, `wall_ghi_s`, 403 ms và nguyên nhân biến động detector đã được chỉnh theo mức bằng chứng. Phép nhân decode mới với số frame là đối chiếu số học, không phải đo hồi tố bản cũ.
- Log smoke của tác giả ghi cửa sổ `--show` thật, YOLO/GPU, lưu video/cache, 60 frame và replay subprocess **95/95 hàng lệch 0**. Reviewer xác nhận nội dung log; không có ảnh, checkpoint, GUI/GPU tương ứng để chạy lại độc lập.

**6. Giới hạn và một ghi chú test không chặn nghiệm thu PC Windows**

1. Reviewer dùng Python 3.12 và backend NumPy; môi trường không có OpenCV, torch, Ultralytics hoặc video/checkpoint gốc. Fixture thay rõ ràng detector/I/O/GUI. Full suite 30/30 và `test_track_video_file` 27/27 là bằng chứng từ log của tác giả.
2. Bấm `q` thật bằng tay trên cửa sổ chưa được smoke; thao tác này được giả lập ở kiểm thử. Cửa sổ thật chạy hết video đã có log riêng. Các phép kiểm độc lập đã bao phủ đường callback → break → close → chốt timing.
3. Camera trực tiếp, video FPS biến thiên và triển khai/tốc độ Nano vẫn ngoài phạm vi nghiệm thu này. Hai finding Nano F1/F3 trước đây chưa được đánh dấu đóng.
4. **Assertion S1 còn phụ thuộc Windows:** trong `test_track_video_file.py`, điều kiện `'"traffic clip"' in replay` yêu cầu dấu nháy kép. Trên POSIX, `shlex.join` sinh `'traffic clip'` bằng nháy đơn hợp lệ, nên điều kiện ấy trả False dù argv/replay đúng. Reviewer đã kiểm chuỗi POSIX và roundtrip đúng. Đây là lỗi tính di động của assertion, **không chặn hệ thống PC Windows**, nhưng cần sửa trước khi tuyên bố suite chạy được trên Linux/Nano. Có thể kiểm ranh giới đối số/roundtrip theo nền tảng thay vì kiểu ký tự quote; không cần mở một vòng benchmark mới.

**7. Chốt mốc và bước tiếp theo**

Codex xác nhận đã đọc và nghiệm thu snapshot **`074ceca1ba36777cabc02a54d6699e890b19c51b`** trong phạm vi PC ở mức **PASS WITH LIMITATIONS**. PC-F1–PC-F4 cùng hai ca bổ sung của review `923305d` được đóng trong phạm vi đã nêu.

Có thể lưu gói đã duyệt thành `handoff/archive/review_delta_074ceca.zip` với đúng SHA256 ở mục 1, rồi chuyển base review sang `074ceca` theo công cụ hiện có. Reviewer không tự sửa base trên máy Windows; đây là xác nhận để tác giả cập nhật. Gói tiếp theo chỉ cần chứa thay đổi sau snapshot này, giúp giảm phần dữ liệu cũ gửi lại.

Bước triển khai tiếp theo: **launcher/giao diện PC dùng lại pipeline đã kiểm chứng**, cho chọn nguồn, bộ lọc, cấu hình và nơi lưu kết quả; hiển thị tiến trình, dừng có kiểm soát và mở đầu ra. Giữ phiên bản này làm mốc tham chiếu, tiếp tục bảo toàn số liệu nghiên cứu và hoãn phần Nano theo kế hoạch của người dùng.
