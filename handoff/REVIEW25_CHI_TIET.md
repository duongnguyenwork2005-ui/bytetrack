# Phụ lục vòng 25 — dùng tại chỗ trên Jetson Nano: launcher HDMI + hiển thị trực tiếp bbox/ID/lớp

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Hướng dẫn người dùng: **`handoff/NANO_LOCAL_UI.md`**. Review trước: `handoff/CODEX_REVIEW_f86732b.md`
(PASS WITH LIMITATIONS, đóng N23-F1). Delta đã review lưu `handoff/archive/review_delta_f86732b.zip` (6.308.100 B, `71fddf70…`, khớp
báo cáo); base review **chuyển sang `f86732b`** (`7cfd24d`), khai báo tiếp 13 NPZ không bàn giao. Giới hạn vòng 24 giữ nguyên (kiểm tên
lớp mọi box còn một nhãn khó phân biệt mỗi video; bộ kiểm render chỉ hợp lệ với cùng OpenCV đã render).

Đầu vòng (`logs/review25/00_trang_thai_dau.log`): HEAD `f86732b`, 0 commit sau đó, working tree chỉ có file riêng. Nano vừa mất nguồn,
người dùng khởi động lại (boot 09:34:10): ext4 `errors_count` 0, engine `c32690cd…`, gói v11 manifest 556/556, 1.470 ảnh còn đủ.
Không đổi detector, ngưỡng, YAML khóa, tracker, kết quả UA-DETRAC/BDD, benchmark hay bằng chứng Nano cũ; không cài gói, không sudo.

## 1. Khả năng GUI trên thiết bị (kiểm trước khi chọn kiến trúc)

| | Python 3.6.9 hệ thống | env `nano` Python 3.8.20 |
|:--|:--|:--|
| Tk | 8.6, mở được trên `:0` | 8.6, mở được trên `:0` |
| OpenCV | 3.2.0, highgui **GTK+ 3** | **không có** |
| khác | GTK 3.22 (gi), PIL 8.4, TensorRT 8.2.1.9 | tracker, pandas, lap |

Phiên desktop `jetson` trên `:0` (gdm3, Unity/compiz), HDMI 1920×1080; có `xdotool`, `scrot`/`import`, `zenity`, `gtk-launch`,
`gnome-session-inhibit`.

## 2. Kiến trúc (dùng chung mã đã nghiệm thu, không thuật toán riêng)

```
launcher Tk (env nano, src/nano_launcher.py) --argv danh sach--> nano_live.py (env nano: loc + tracker, run_tracking.track_video)
      ^ progress.json / stop-file                                   |  lenh R(i, hang MOT/lop cua frame i) thay credit "+"
      |                                                             v
      +---- q/Esc/dong cua so -> stop-file <---- nano_detect_stream.py (Python 3.6: doc frame -> TensorRT -> gui FRAME i)
                                                  + luong hien thi: anh GOC frame i (khoa frame_idx) + R(i) -> ve_frame -> cua so GTK
```

- **Hiển thị ở tiến trình detector**: là tiến trình duy nhất có ảnh đã giải mã và OpenCV GTK; không truyền ảnh qua pipe, không giải
  mã lần hai. Ảnh gốc của frame *i* được lưu theo `frame_idx` **trước** khi gửi FRAME *i*.
- **Giao thức 3** (`nano_stream_proto`): lệnh `R` = credit + hàng kết quả của frame *i* (id, x, y, w, h, conf, cls). Hàng được tính
  bằng `nano_live.hang_hien_thi(out)` — **cùng phép tính** với hàng MOT/lớp trong `run_tracking._vong_lap_tracker`. `BoDocLenh` tách lệnh
  kể cả khi bị cắt giữa hai lần đọc. Tắt hiển thị: gửi `+` như vòng 24.
- Vẽ bằng **chính `render_mot_video.ve_frame`** (hàm render hậu kỳ); dòng trạng thái ở góc dưới.
- **Hàng đợi hiển thị có giới hạn** (2 frame): đầy thì bỏ frame xem trước cũ nhất và đếm (`hien_thi.dem.n_bo_hien_thi`); vòng lặp
  detector/tracker không bao giờ chờ hiển thị. Sau EOS, con nhận nốt `R` của frame còn đang bay để frame cuối cũng được vẽ.
- **Dừng**: nút Dừng, `q`/`Esc` trong cửa sổ xem, đóng cửa sổ xem, đóng launcher → cùng một stop-file → `nano_live` dừng có kiểm soát
  (mã 4, `da_dung`), ghi đủ phần đã xử lý, dừng/reap con, đóng cửa sổ.
- `nano_live`: `--hien-thi`; kiểm màn hình (`DISPLAY` + `xprop -root`) **trước** khi spawn (thiếu → mã 2); `progress.json`;
  `hien_thi.json` → `run_meta.hien_thi`; `thoi_gian.tong_s`, `thoi_gian.fps_hien_thi`; định nghĩa ghi rõ **không có tạm dừng**.
- Launcher: Tk, tái sử dụng `pc_launcher.LauncherController` (tách `_ket_luan_cuoi`, hành vi PC không đổi — `test_pc_launcher` 17/17);
  cấu hình triển khai `configs/nano_ui.json` (+ `~/.config/nano_tracking/nano_ui.json`); kiểm engine, Python detector (import
  tensorrt/cv2), LD_PRELOAD, ONNX, quyền ghi; hộp chọn file zenity; mỗi lượt thư mục mới; `launcher.json`, `launcher_nhat_ky.txt`;
  trong lúc chạy tạm chặn khóa màn hình khi rảnh (`gnome-session-inhibit`, tự thoát khi launcher chết).
- Lệnh mở `scripts/nano/07_giao_dien.sh`; lối tắt desktop + menu `scripts/nano/08_tao_loi_tat.sh` (không sudo).

## 3. Kiểm thử hồi quy

`src/test_nano_hien_thi.py` (đăng ký `run_all_tests`): giao thức 3 (cắt từng byte); hàng hiển thị == hàng MOT (CV, EKF); bật/tắt hiển
thị cùng detection → MOT + lớp giống từng byte; ảnh xem trước == `ve_frame(ảnh gốc, MOT đã ghi)` từng pixel; phím q giả lập → mã 4;
frame hỏng; không màn hình → mã 2; launcher: thiếu engine, đường dẫn có dấu cách, cấu hình sai, hoàn tất / Dừng / worker lỗi / đóng
launcher, dọn tiến trình chặn khóa; GUI Tk.

| Nơi | Kết quả | Log |
|:--|:--|:--|
| PC | `test_nano_hien_thi` **16/16** (+1 SKIP chỉ POSIX), `test_nano_stream` 20/20, `test_pc_launcher` 17/17, render 13/13, nano_detector 6/6 | commit `a7b50e2`, `71160d0` |
| Nano v12/v13 | `test_nano_hien_thi` **17/17** (gồm GUI Tk trên HDMI), `test_nano_stream` 20/20, `--kiem-tra` exit 0 | `v25_test_v12/`, `v25_test_v13/` |
| Nano v14 (gói cuối) | `test_nano_hien_thi` 17/17, `test_nano_stream` 20/20, `--kiem-tra` exit 0; launcher K1 hai lượt `hoan_tat`, nhật ký 0 dòng `fatal` | `v25_test_v14/`, `v25_kich_ban_v14/`, `08_trien_khai_v14.log` |
| PC full suite | **36/36 PASS** trên `91c95c7` (mã cuối) | `tests_20260929-1021.log` |

## 4. TensorRT thật trên Nano

Driver `logs/review25/02_hien_thi_trt.sh`: `nano_live` c25, K = 4, `--save-cache`; sau mỗi lượt replay bằng chính `replay_argv` →
`nano_parity` → so byte; lượt bật hiển thị có thể lưu ảnh xem trước để so với `so_mau_hien_thi.py`.

| Bộ | Điều kiện | Kết quả |
|:--|:--|:--|
| `v25_parity/` (4 × 300 ảnh) | màn hình đang khóa | detection 4 lượt **giống hệt**; bật/tắt: MOT + lớp **giống byte** (CV `15ab38b4…`, EKF `238dbb2a…`); live == replay, parity ĐẠT; ảnh xem trước 6/6 khớp pixel |
| `v25_hdmi/D_anh_huong_300/` (8 × 300 ảnh, ABBA) | HDMI **mở khóa** | 8/8 hoàn tất, live == replay, parity ĐẠT; detection và MOT mỗi tracker **giống nhau ở cả 4 lượt** bật/tắt |
| `v25_hdmi/E_day_du_1470/` (CV + EKF, 1.470 ảnh, hiển thị bật) | HDMI mở khóa | hoàn tất 1470/1470, **hiện 1470/1470, bỏ 0**; live == replay; ảnh xem trước 14/14 khớp; MOT + lớp **giống byte kết quả vòng 23** (CV `5017f37b…`/`a14e06bf…`, EKF `645bba3a…`/`a22bbf7d…`) |

**Ảnh hưởng của hiển thị** (`09_phan_tich_anh_huong_hien_thi.log`; D, 300 ảnh, 2 lượt/nhóm, thứ tự CV bật-tắt-tắt-bật, EKF tắt-bật-bật-tắt; mô tả, không kiểm định):

| Nhóm | FPS xử lý (từng lượt) | FPS hiển thị | hiện / bỏ | decode tv | infer tv | RAM max | CPU tb |
|:--|:--|:--|:--|:--|:--|:--|:--|
| CV bật | 9,21 (9,214; 9,209) | 9,26 | 300 / 0 | 11,5 ms | 75,0 ms | 2.528 MB | 24,1 % |
| CV tắt | 9,39 (9,382; 9,399) | — | — | 11,4 ms | 74,0 ms | 2.508 MB | 18,0 % |
| EKF bật | 9,18 (9,164; 9,201) | 9,23 | 300 / 0 | 11,4 ms | 75,0 ms | 2.529 MB | 24,4 % |
| EKF tắt | 9,38 (9,400; 9,355) | — | — | 11,5 ms | 74,1 ms | 2.506 MB | 18,0 % |

Hiển thị giảm FPS xử lý khoảng 0,2 (≈ 2 %); infer trung vị tăng ≈ 0,9 ms (GPU dùng chung với X/compiz); vẽ ≈ 1,8 ms, imshow ≈ 2,2 ms
ở luồng riêng. Tám lượt cùng mức decode (≈ 11,4–11,5 ms) nên so sánh bật/tắt cùng điều kiện; không dùng để xếp hạng CV/EKF.
Lượt 1.470 ảnh: CV 8,66 FPS (decode rơi vào mức chậm 16,6 ms), EKF 9,10 FPS (11,5 ms) — cùng hiện tượng hai mức đã ghi từ vòng 21.
Tài nguyên (tegrastats): RAM ≤ 2.530 MB, swap ≤ 92 MB, CPU tối đa 38,5 °C, GPU tối đa 35 °C; MemAvailable trước/sau mỗi lượt lệch
≤ 10 MB; sau mỗi lượt 0 tiến trình detector, 0 cửa sổ xem; ext4 0 lỗi; boot không đổi. Khởi tạo 8,0–9,3 s (lần đầu sau khi bật máy
62,7 s — nạp engine khi bộ nhớ đệm file còn lạnh).

## 5. Launcher và vòng đời chương trình

| Kịch bản | Cách thao tác | Kết quả |
|:--|:--|:--|
| K1 hai lượt liên tiếp một phiên (mở lượt mới) | tự động (`--tu-dong-*`, gọi đúng hàm của nút) | 2 thư mục mới, cả hai `hoan_tat` 300/300, hiện 300 bỏ 0 |
| K2 EKF từ **mp4**, Dừng sau 12 s rồi lượt mới | tự động | `da_dung` mã 4 ở frame 122 → lượt mới `hoan_tat` 300/300 |
| K3 đóng launcher khi đang chạy | tự động (`WM_DELETE_WINDOW`) | `da_dung` mã 4 ở frame 114, launcher tự đóng sau khi ghi xong |
| K4 phím **q** trong cửa sổ xem | xdotool (XTEST) trên HDMI mở khóa | `da_dung` mã 4 ở frame 81, lý do `phim_q_esc` |
| K5 **đóng cửa sổ xem** (Alt+F4) | xdotool trên HDMI mở khóa | `da_dung` mã 4 ở frame 82, lý do `dong_cua_so` |
| B mở từ **biểu tượng desktop** | `gtk-launch nano-tracking` trong phiên desktop | launcher cha = `systemd` (PID 1), phiên riêng, không thuộc SSH; `hoan_tat` 300/300; launcher tự đóng |

Sau mỗi kịch bản: 0 detector, 0 `nano_live`, 0 launcher, 0 cửa sổ xem, tiến trình chặn khóa màn hình đã trả lại. K1–K3 chạy lúc màn
hình khóa (không cần nhập liệu); K4/K5/B sau khi người dùng mở khóa (`07_pha_hdmi_sau_mo_khoa.log`, `v25_hdmi/`). Gói v14 lặp lại K1
(`v25_kich_ban_v14/`). Ảnh chụp màn hình Nano `logs/review25/anh_man_hinh/` (nửa kích thước, JPEG; hash PNG gốc trong
`nguon_png_sha256.txt`): 01 biểu tượng desktop, 02 launcher mở từ biểu tượng, 03 đang chạy (launcher + cửa sổ xem), 04 hoàn tất +
tóm tắt, 05/06 lượt 1.470 ảnh CV/EKF.

## 6. Chặn thực tế đã gặp và cách xử lý

Màn hình Nano **khóa tự động** sau 5 phút rảnh (Unity `IsLocked = true`, `idle-delay 300`, `lock-enabled true`): phím xdotool không
vào cửa sổ nào (kể cả zenity), ảnh chụp là màn hình khóa (`03_man_hinh_khoa.log`). Không dùng mật khẩu qua SSH: dựng bộ chờ
`05_cho_mo_khoa.sh`; người dùng mở khóa lúc 10:56:18 (giờ Nano), pha HDMI chạy tự động 10:56–11:13 dưới `gnome-session-inhibit`
(`07_pha_hdmi_sau_mo_khoa.log`). Rút kinh nghiệm cho người dùng: launcher tạm chặn khóa khi rảnh **trong lúc chạy**.
Khi màn hình khóa, luồng hiển thị vẫn chạy (không lỗi), chỉ không nhìn thấy.

Sửa nhỏ sau khi xem trên HDMI (gói v14): `det_cache.git_commit` bỏ stderr (gói Nano không có `.git` → dòng `fatal:` trong nhật ký
launcher); `08_tao_loi_tat.sh` `touch` sau khi đánh dấu tin cậy (Nautilus 3.26 mới hiện tên + icon), `Categories` một nhóm chính.

## 7. Thao tác người dùng trực tiếp

Người dùng: mở khóa màn hình Nano. Mọi thao tác khác trong vòng này là tự động hoặc qua xdotool/gtk-launch (ghi ở mục 5).
Checklist thao tác tay bằng chuột/bàn phím (5 bước, `NANO_LOCAL_UI.md` mục 8): **chưa thực hiện bởi người** trong vòng này; các bước tương ứng đã chạy bằng tự động hóa (xdotool XTEST cho phím q/Alt+F4, gtk-launch cho biểu tượng, tham số `--tu-dong-*` cho nút). Khác biệt còn lại so với thao tác tay: chuột bấm nút thật và hộp chọn file zenity chưa được bấm bởi người.

## 8. Giới hạn

Chưa có camera; không tạm dừng; một lượt tại một thời điểm. Cửa sổ xem có thể bỏ frame xem trước khi máy bận (đếm trong
`hien_thi.json`; các lượt đo ở đây bỏ 0). FPS phụ thuộc hai mức decode của OpenCV 3.2. Tự động hóa dùng tham số `--tu-dong-*` /
`NANO_UI_TU_DONG` (chỉ kiểm thử) và xdotool; chưa có đo trên nhiều video khác MVI_39031. Khi màn hình khóa, người dùng không thấy cửa
sổ xem dù luồng vẫn chạy.

## 9. File và commit

Mã: `src/nano_stream_proto.py`, `src/nano_detect_stream.py`, `src/nano_hien_thi.py` (mới), `src/nano_live.py`, `src/pc_launcher.py`,
`src/nano_launcher.py` (mới), `src/det_cache.py`, `src/test_nano_hien_thi.py` (mới), `src/run_all_tests.py`, `configs/nano_ui.json`
(mới), `scripts/nano/07_giao_dien.sh`, `scripts/nano/08_tao_loi_tat.sh` (mới). Bằng chứng `handoff/logs/review25/`. Gói Nano
v12 `f2b13fa8…`, v13 `ffbe9081…`, **v14 `315407c5…`** (đang dùng, `~/nano_run/v14`). Commit: `f0f092b` (review + trạng thái đầu), `7cfd24d` (base), `a7b50e2` (hiển thị + launcher + test), `71160d0` (chặn khóa khi
chạy, nhật ký, script thiết bị), `4dca738` (bằng chứng lúc màn hình khóa + hướng dẫn), `bbae485` (script pha HDMI), `91c95c7`
(det_cache stderr, lối tắt), `30891c9` (bằng chứng HDMI, v14, ảnh, full suite), rồi commit tài liệu vòng 25.
