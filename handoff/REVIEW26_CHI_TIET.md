# Phụ lục vòng 26 — sửa N25-F1 (đếm frame hiển thị sau khi đóng cửa sổ) và N25-F2 (tiến trình chọn file khi đóng launcher)

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Review: `handoff/CODEX_REVIEW_20e66e9.md` (REQUEST CHANGES phần giao diện, hai finding P2);
script tái hiện của Codex lưu nguyên: `handoff/CODEX_REPRO_20e66e9_ui.py` (sha256 `e8172102…`). **Base review giữ `f86732b`**; chưa
đánh dấu vòng 25 PASS. Đầu vòng (`logs/review26/00_trang_thai_dau.log`): HEAD `20e66e9`, 0 commit sau đó, working tree chỉ có file
riêng; Nano boot 09:34:10, ext4 0 lỗi, đang dùng gói v14 (`src/nano_hien_thi.py`, `src/nano_launcher.py` trùng hash `20e66e9`).
Không đổi detector, ngưỡng, YAML khóa, tracker, giao thức, kết quả UA-DETRAC/BDD hay số benchmark; không build engine, không chạy
lại lượt 1.470 frame.

## 1. Tái hiện trên mã chưa sửa (script Codex, không sửa script)

| Nơi chạy | Lệnh | Kết quả | exit |
|:--|:--|:--|:--|
| PC Windows, Python 3.11.9, OpenCV 5.0.0 | `python handoff/CODEX_REPRO_20e66e9_ui.py --repo .` | F1: `imshow` 1 lần nhưng `n_hien_thi` 2, frame cuối 2, `fps_hien_thi` 439,75; F2: SKIP (cần POSIX) | **1** |
| Nano, `/usr/bin/python3` 3.6.9, OpenCV 3.2.0 | `chay_repro_tung_phan.py … closed_preview` (gọi đúng hàm của script) | F1: `imshow` 1, `n_hien_thi` 2, cuối 2, fps 100,57 | **1** |
| Nano, env `nano` 3.8.20 (POSIX) | `chay_repro_tung_phan.py … picker_close` | F2: launcher đã destroy, **tiến trình chọn file còn sống**, 1 callback chờ | **1** |

Trên Nano không có interpreter nào vừa có OpenCV vừa ≥ 3.7, nên hai hàm của script được gọi riêng bằng `chay_repro_tung_phan.py`
(`logs/review26/01_*`, `02_*`, `repro_truoc/`). Nguyên nhân xác nhận trên mã: **F1** `_ve()` bỏ `imshow` khi `_cua_so_da_dong` nhưng
vẫn tăng `n_hien_thi`, thêm mốc thời gian, cập nhật frame cuối; **F2** zenity chạy bằng `subprocess.run` trong thread daemon, launcher
không giữ handle; `_dong()` destroy ngay (hoặc chỉ chờ tracker), không kết thúc/thu hồi picker, không hủy callback chờ.
Dòng “còn 1 tiến trình picker” khi kiểm bằng `pgrep -f` là **chính shell đang tìm** (cùng lỗi helper Codex nêu); kiểm theo executable
(`tim_tien_trinh.py`, `/proc/<pid>/exe` + argv): **0** — script tự dọn fixture của nó.

## 2. Sửa

**N25-F1** (`src/nano_hien_thi.py`; commit `64152f2`):

- `n_hien_thi`, mốc `t_hien_thi`, `frame_hien_thi_cuoi` chỉ tăng khi frame **thực sự được trình**: cửa sổ còn mở (kiểm `getWindowProperty`
  **trước mỗi** `imshow`, không chỉ ở `_su_kien`) và `cv2.imshow` chạy xong không lỗi. Cửa sổ đã đóng → không vẽ, không `imshow`
  (không tạo lại cửa sổ), đếm riêng `n_khong_hien_thi_cua_so_dong` + danh sách frame.
- Phân biệt: `n_ghep_duoc` (kết quả tracker ghép được ảnh) = `n_hien_thi` + `n_bo_hien_thi` + `n_khong_hien_thi_cua_so_dong`;
  `n_da_ve` = đã vẽ. Tracker không bị ảnh hưởng: MOT/lớp của mọi frame đã xử lý vẫn ghi đủ.
- `fps_hien_thi = (N − 1) / (mốc cuối − mốc đầu)`, N = `n_hien_thi`; **N < 2 → null**. Định nghĩa đồng bộ ở `nano_hien_thi`,
  `nano_live` (`run_meta.thoi_gian.dinh_nghia`), tóm tắt launcher (“dưới 2 frame: không có FPS hiển thị”) và `NANO_LOCAL_UI.md`. Công
  thức cũ vốn đã là (N − 1)/khoảng; chỉ câu chữ và việc đếm sau khi đóng thay đổi → **số benchmark vòng 25 giữ nguyên** (các lượt đó
  không đóng cửa sổ giữa chừng).
- Chế độ ẩn (kiểm thử): không có cửa sổ, “trình” là **giả lập** (đã vẽ xong), ghi rõ trong `hien_thi.che_do`; hook
  `--gia-lap-dong-cua-so-o-frame N` giả lập người dùng đóng cửa sổ sau frame N (chỉ kiểm thử).

**N25-F2** (`src/nano_launcher.py`; commit `64152f2`, `a370c57`):

- Zenity chạy bằng `subprocess.Popen`, launcher **giữ handle**; chờ kết quả bằng `after()` trên luồng Tk (bỏ thread daemon).
- Đóng launcher: hủy callback chờ kết quả, gửi SIGTERM tới **đúng PID của handle** (không diệt theo tên), chờ tối đa 0,3 s rồi thu
  hồi; chưa thoát thì `_cho_dong` tiếp tục chờ bằng `after()` (giao diện vẫn phản hồi), quá 2 s mới SIGKILL. Picker vừa tự thoát đúng
  lúc đóng → chỉ thu hồi, **bỏ kết quả**, không cập nhật widget. Tracker đang chạy → vẫn dừng có kiểm soát như trước.
- Mọi `after()` của launcher đi qua `_hen()` (ghi id); trước `destroy()` hủy hết; callback còn sót thành no-op; sau destroy không hẹn
  thêm. Nhật ký cả phiên ghi vào `~/.cache/nano_tracking/launcher_phien_*.log` khi đóng (có sự kiện hộp chọn file).
- Hook kiểm thử (chỉ tự động hóa, gọi đúng hàm của nút): `--tu-dong-hop-chon`, `--tu-dong-dong-khi-hop-chon-mo-sau`.

Helper kiểm tiến trình trong test: `tien_trinh_linux()` đọc `/proc/<pid>/exe` + argv thay `pgrep -f` (hai FAIL helper trên máy
reviewer); script thiết bị dùng `tien_trinh.py` cùng cách.

## 3. Trước / sau

| Kiểm | Mã cũ `20e66e9` | Mã sửa | Log |
|:--|:--|:--|:--|
| Script Codex, PC | exit 1 (F1 FAIL, F2 SKIP) | **exit 0** (F1 PASS: imshow 1, n 1, cuối 1, fps null) | `01_*`, `05_*` |
| Script Codex F1, Nano (3.6 + OpenCV 3.2) | exit 1 | **exit 0** | `02_*`, `06_*` mục A |
| Script Codex F2, Nano (env nano) | exit 1 (picker còn sống) | **exit 0** (picker đã thoát) | `02_*`, `06_*` mục A |
| 10 — frame 1 hiện → đóng → frame 2 tới muộn | FAIL (n 2, cuối 2, fps có số) | PASS (imshow 1, n 1, cuối 1, fps null, 1 không hiện) | `03c_*`, `04_*` |
| 10b — đóng giữa hai frame | FAIL (imshow 3 lên cửa sổ đã đóng) | PASS (không imshow, không tạo lại cửa sổ) | nt |
| 10c — fps = (N−1)/khoảng | FAIL chỉ vì câu định nghĩa (số vốn đúng) | PASS | nt |
| 11 — đầu-cuối đóng cửa sổ (giả lập) | FAIL (nano_live cũ không có hook → mã 2) | PASS: MOT + lớp mọi frame đã xử lý == tiền tố lượt đủ | nt |
| 12a — mở picker → đóng launcher (chưa chạy) | FAIL (picker còn sống, 1 callback) | PASS | `v26_test_lai/` |
| 12b — hủy picker → mở lại → chọn | PASS (chức năng cũ vốn chạy) | PASS | nt |
| 12c — picker vừa tự thoát đúng lúc đóng | FAIL (lỗi: mã cũ thiếu thuộc tính — không phải bằng chứng trực tiếp) | PASS (thu hồi, bỏ kết quả) | nt |
| 12d — picker mở + tracker chạy → đóng | FAIL (tracker dừng đúng nhưng picker không bị kết thúc, 1 callback) | PASS (picker thoát trước, tracker DA_DUNG, worker/detector thoát) | nt |
| 12e — cửa sổ Tk thật | FAIL (cửa sổ hủy, picker còn sống) | PASS (0 lỗi callback) | nt |
| Tổng `test_nano_hien_thi` | PC 16/20; Nano 18/23 | **PC 20/20** (+2 SKIP POSIX); **Nano 23/23** (+1 SKIP: 10 cần cv2, env nano không có) | |
| Suite liên quan (PC) | | `test_nano_stream` 20/20, `test_pc_launcher` 17/17, `test_render_mot_video` 13/13 | |
| Full suite PC | | **36/36 PASS** trên `6dbf40f` (mã cuối) | `tests_20260929-1401.log` |

Trong harness của Codex, `scheduled_callbacks` sau sửa vẫn là 1: stub `after()` của harness trả id `None` và không có
`after_cancel`, nên không hủy được; callback đó là no-op (launcher đã đóng). Harness `_Lich` của test 12a–d và cửa sổ Tk thật (12e)
xác nhận 0 callback còn lại.

## 4. Nano thật (gói **v16** `302f6af7…`, `~/nano_run/v16`; v14 giữ nguyên để đối chiếu; v15 bị thay trước khi dùng)

Script `logs/review26/06_kiem_thiet_bi_v26.sh` chạy dưới `gnome-session-inhibit`, màn hình đã mở khóa; thao tác cửa sổ **tự động**
bằng xdotool (XTEST) và tham số `--tu-dong-*` (gọi đúng hàm của nút), không phải người bấm chuột. Kiểm tiến trình bằng executable.

| Ca | Kết quả |
|:--|:--|
| D — TensorRT thật CV 300 ảnh, hiển thị HDMI, **Alt+F4 đóng cửa sổ xem** giữa chừng | `da_dung` mã 4, lý do `dong_cua_so`; xử lý 181 frame; **hiện 180, cuối 180, 1 frame (181) không hiện vì cửa sổ đã đóng**, bỏ 0; fps hiển thị 9,35; live == replay giống byte, `nano_parity` ĐẠT; MOT + lớp 181 frame (554 hàng) **giống tiền tố lượt đủ không hiển thị** (v25 D); detector đã thoát |
| E1 — zenity thật: mở → Esc → mở lại → gõ đường dẫn mp4 + Enter | lần 1 mã 1 “không chọn”, lần 2 mã 0 trả đúng đường dẫn; đóng launcher (Alt+F4): 0 zenity, launcher thoát |
| E2 — mở hộp chọn → đóng launcher (chưa chạy) | nhật ký: SIGTERM tới đúng PID zenity, thu hồi mã −15; launcher thoát; 0 zenity |
| E3 — tracker đang chạy + hộp chọn mở → đóng launcher | zenity kết thúc + thu hồi trước; lượt `da_dung` mã 4 (57 frame); detector, `nano_live`, launcher đã thoát |
| F — mở từ biểu tượng desktop (trỏ v16, `gtk-launch`), 2 lượt EKF | launcher cha `systemd`, phiên riêng; 2 lượt `hoan_tat` 300/300 (hiện 299 + bỏ 1; 300); thoát sạch |

Sau mỗi ca: 0 detector, 0 `nano_live`, 0 launcher, 0 zenity, 0 cửa sổ xem; ext4 0 lỗi. Ảnh màn hình Nano:
`logs/review26/anh_man_hinh/` (D trước/sau khi đóng, E1 zenity, E2, E3 launcher + zenity + cửa sổ xem, F).

## 5. Metadata hiển thị (mới)

`hien_thi.json` / `run_meta.hien_thi`: `che_do`, `dem.n_da_ve`, `dem.n_hien_thi`, `dem.n_khong_hien_thi_cua_so_dong`,
`frame_khong_hien_thi_cua_so_dong_dau`, `fps_hien_thi` (null khi < 2 frame), `dinh_nghia` từng trường.

## 6. Giới hạn và phần chưa xác định

- Test 10 không chạy được trong env `nano` (không có cv2); được phủ bởi PC và script Codex F1 trên Python 3.6 của Nano.
- Nhánh SIGKILL sau 2 s (picker không thoát khi nhận SIGTERM) chưa được kiểm với tiến trình treo thật; zenity thật thoát ngay.
- Thao tác cửa sổ là tự động (xdotool/gtk-launch/hook); chưa có người bấm chuột thật trong vòng này.
- 12c trên mã cũ lỗi do thiếu thuộc tính, không phải tái hiện trực tiếp điều kiện đua; điều kiện đó chỉ được kiểm trên mã mới.
- Camera ngoài phạm vi.

## 7. File và commit

Mã: `src/nano_hien_thi.py`, `src/nano_detect_stream.py` (hook), `src/nano_live.py` (hook, định nghĩa), `src/nano_launcher.py`,
`src/test_nano_hien_thi.py`. Bằng chứng `handoff/logs/review26/`. Commit: `e242353` (review + tái hiện), `64152f2` (sửa F1 + F2 +
test), `a370c57` (hook hộp chọn, nhật ký phiên), `ada9027` (test 12e, script thiết bị), `6dbf40f` (bằng chứng), rồi commit tài liệu vòng 26.
