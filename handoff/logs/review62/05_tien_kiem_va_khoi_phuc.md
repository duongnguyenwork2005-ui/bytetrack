# R62 — tiền kiểm thiết bị và nguồn khôi phục

Ngày 28/10/2026 (giờ Việt Nam). Chưa có benchmark mới, chưa sửa mã pipeline.

> Cập nhật cuối ngày: trạng thái tiền kiểm ban đầu bên dưới là lịch sử. Baseline ngắn đã hoàn tất,
> ứng viên đã sửa và kiểm; xem `20_tien_do_ung_vien.md`. Lượt dài đầu bị gián đoạn do reboot,
> đang chờ người dùng xác nhận nguyên nhân trước khi chạy tiếp GPU.

## Cập nhật sau khi người dùng đổi Jetson (thay trạng thái chặn SD bên dưới)

Người dùng xác nhận đã đổi sang Jetson khác và vẫn dùng thẻ SD cũ. Lệnh đọc lúc 08:43 UTC / 15:43 Việt Nam ghi `07_stack_jetson_moi.json`: root đã ở `/dev/mmcblk1p1`, còn 27.846.819.840 byte. Kết nối kế tiếp lúc 15:44 Việt Nam xác nhận thẻ 58,2 GiB, UUID `e63fc90f-5700-4a13-a49d-43dc08fead24`, v16 và thư mục ảnh 1.470 frame tồn tại; engine 12.626.138 byte, env `nano` có symlink Python 3.8. Máy vừa boot khoảng 4 phút. Đây là xác nhận tồn tại đường dẫn/kích thước, chưa phải xác minh đầy đủ nội dung.

Stack detector import được: NumPy 1.19.5, OpenCV 3.2.0, TensorRT 8.2.1.9, Python hệ thống 3.6; thư viện CUDA 10.2 và `trtexec` có sẵn. Device tree khai báo `p3449-0000-b00+p3448-0002-b00`; giá trị này mô tả cấu hình phần mềm, chưa xác nhận hãng carrier board thực tế. RAM 3.963 MiB, swap 6.077 MiB (đang dùng 0).

SSH mất kết nối giữa lệnh đọc hash engine, chưa trả hash hay phiên bản env `nano`. Kết nối mới timeout, ping host unreachable. Đã yêu cầu người dùng gửi `hostname -I` từ Nano. Không suy đoán nguyên nhân là thẻ hỏng hoặc máy tắt từ lỗi mạng này.

**Kết nối đã trở lại lúc 15:47 Việt Nam:** engine khớp hash cũ `c32690cd…`, Python env 3.8.20; manifest 562/562 file đạt, 300/1.470 JPG hiện diện, HDMI 1920x1080 hoạt động, ext4 errors 0 (`08_tien_kiem_v16.json`). Đã tạo bản sao độc lập từ đúng 562 file trong manifest + manifest ở `~/nano_run/r62_20261028/baseline/nano_transfer`; kiểm hash từng bản sao đạt. Mọi lượt mới chạy từ bản sao này, giữ v16 gốc.

**Đang đo:** `chay_mot_luot.py` gọi `nano_live`, replay bằng `replay_argv`, kiểm `nano_parity` và SHA256 MOT/lớp, từ chối ghi đè lượt hoặc replay có sẵn. Mỗi lượt lưu lệnh, boot/clock/RAM trước-sau và toàn bộ log. Lượt đầu CV bật hiển thị hoàn tất 300/300, 8,5934 FPS, live/replay trùng byte (`09_luot_dau_cv_on.json`). Khởi tạo 85,8 s; hiển thị 299/300, bỏ một frame xem trước lúc khởi tạo cửa sổ, tracking đủ. Giữ nguyên lượt này trong mẫu. Bảy lượt còn lại đang chạy theo lịch, nghỉ 20 s.

**Bước tiếp hiện tại:** hoàn tất baseline ngắn rồi kiểm nguồn ảnh đầy đủ và baseline dài; chọn tối ưu theo profile mới. Số FPS của Jetson cũ chỉ là tham khảo. Không cần dựng lại môi trường. Các bước chẩn đoán chân nhận thẻ bên dưới là lịch sử của trạng thái trước lần boot mới.

## Kết quả đã xác minh trước lần boot mới (lịch sử)

- SSH không tương tác từ Windows tới `jetson@10.11.61.127` qua alias `nano` đã thành công sau khi người dùng thêm khóa công khai. Không cần nhập mật khẩu cho các lệnh không sudo.
- Nano: Linux `4.9.337-tegra`, L4T R32.7.5, RAM khoảng 4 GiB. `/usr/bin/python3` là 3.6.9. Đồng hồ Nano dùng UTC+08; ngày/giờ trình bày cho người dùng dùng Asia/Saigon (UTC+07).
- Root hiện tại là `/dev/mmcblk0p1`, UUID `63067172-616c-4243-bf11-cece24ba1562`, 14 GiB, còn khoảng 2,2 GiB. Thiết bị có type `MMC`, model `DG4016`: eMMC 16 GB. `lsblk` và `/sys/bus/mmc/devices` chỉ nhận eMMC, không có thẻ SD `mmcblk1`.
- Người dùng xác nhận thẻ SD vẫn cắm. Kernel tạo host `mmc1` trên `sdhci-tegra.2`, nhưng chưa ghi nhận thẻ SD. Node `sdhci@700b0400` có `status=okay`, `cd-gpios` trỏ GPIO 194 và thuộc tính `cd-inverted`. Các dữ kiện này chưa đủ kết luận lỗi tiếp xúc, thẻ, hay cấu hình.
- Không có `~/nano_run/v16`, `~/nano_run/engine/yolov8n_fp16.engine`, `~/nano_run/data_full/MVI_39031` hoặc `~/miniforge3/envs/nano/bin/python`. Đã kiểm các thư mục dự kiến và tìm giới hạn độ sâu 3 tại `/home`, `/media`, `/mnt`; không quét toàn ổ/dataset. **Không có căn cứ kết luận dữ liệu trên thẻ đã mất.**
- `/boot/extlinux/extlinux.conf` hiện đặt `root=/dev/mmcblk0p1`, không có dòng FDT đang hoạt động. Hồ sơ R19 từng ghi trên cùng eMMC: `root=/dev/mmcblk1p1` và `FDT /boot/dtb/kernel_tegra210-p3448-0002-p3449-0000-b00.dtb`. Hai DTB hiện có ở `/boot/` và `/boot/dtb/` cùng hash `e6301b1f5305f53e99ff0693836a3582ac3f5edf71e80cb984c565d1c55a3e36`. Chưa có hash DTB cũ để khẳng định chúng giống/khác phiên baseline.
- `nvpmodel -q` báo MAXN, đồng thời báo không đủ quyền đọc một tham số EMC. `jetson_clocks --show` yêu cầu root. Các file debug GPIO/MMC cũng không đọc được bằng tài khoản hiện tại. Một lần thử `sudo -n cat ...` thất bại vì cần mật khẩu; không có thay đổi đặc quyền nào được thực hiện.

Bằng chứng máy đọc: `02_tien_kiem_luu_tru.json`, `03_kiem_bo_dieu_khien_sd.json`, `04_dtb_tren_emmc.json` trong cùng thư mục. Các lệnh chạy trên Nano chỉ đọc; log được lưu về PC.

Đối chiếu lịch sử: `handoff/logs/review19/08_xac_dinh_the_sd.log`, `handoff/logs/review20/07_tinh_trang_the_sd.log`, `handoff/logs/review20/07b_the_sd_sau_khoi_dong_lai.log`, `handoff/logs/review25/00_trang_thai_dau.log`. Thẻ cũ 58,2 GiB, tên `acer`, UUID từng ghi `e63fc90f-5700-4a13-a49d-43dc08fead24`; R25 root ở SD, còn khoảng 27 GiB.

## Nguồn khôi phục trên PC

Đã kiểm bằng tác vụ chỉ đọc trong cùng chat:

- Gói **v16 nguyên byte**: `handoff/nano_transfer_latest.zip`, 63.909.837 byte, SHA256 `302f6af73d97aea009066416aaaa202db81311e60121676a75dacbe204e9cbf1`. 563 entry, 68.933.753 byte giải nén. `PACKAGE_INFO.json` ghi commit nguồn `a370c57e2fdf1744d81c8cd2b22d18346b36bbd2`. Có mã, cấu hình, manifest, 300 JPG, ONNX, cache và tham chiếu; không chứa engine hoặc môi trường Python.
- Engine cũ theo `handoff/logs/review21/04_build_engine.log`: 12.626.138 byte, SHA256 `c32690cdb9f14d5f016212126621ec1fecef6047ebda82fb844530e9b7424a9a`. Chưa tìm thấy binary trong các vùng dự phòng đã kiểm (`models`, `handoff/archive`, log/backup R16/R20/R21 và manifest lưu trữ). Ưu tiên lấy lại engine nguyên bản từ SD.
- ONNX có trong v16: 12.824.061 byte, SHA256 `a66a5e03b6c6efec9b93d84cafb6ec119a853fbc5d7014f0640bb7ad069f6451`. Nếu phải build engine mới, cần ghi nhận baseline mới, không khẳng định hash/đầu ra trùng engine cũ trước khi kiểm.
- Danh sách môi trường: `handoff/logs/review20/sao_luu_nano/home/env_nano_goi.txt`, cùng `conda_create.log`, `dung_env.log`, `miniforge_install.log`. Python 3.8.20, NumPy 1.24.4, pandas 2.0.3, PyYAML 6.0.2, lap 0.5.13. Đây là inventory phiên bản/build, không phải bộ cài offline hay lockfile có đầy đủ URL/hash. Chưa tìm thấy installer Miniforge lịch sử trong vùng đã kiểm.

## Bước xử lý được đề ra trước lần boot mới (đã thay thế ở đầu file)

1. Người dùng đọc `sudo cat /sys/kernel/debug/mmc1/ios` và `sudo cat /sys/kernel/debug/gpio | grep -Ei 'sd|card|194'`, gửi kết quả để kiểm tín hiệu nhận thẻ. Không gửi mật khẩu.
2. Nếu vẫn không nhận thẻ, kiểm tiếp xúc bằng cách tắt máy sạch trước khi tháo/cắm lại; hoặc dùng đầu đọc ngoài để kiểm thẻ. Việc đụng phần cứng cần người dùng thực hiện.
3. Chỉ sau khi nhận được thẻ mới kiểm filesystem và thư mục v16/engine/env. Không sửa root boot sang thiết bị chưa nhận; không format, flash hay chạy sửa filesystem ở giai đoạn chẩn đoán này.
4. Khi khôi phục được baseline, tiếp tục tiền kiểm và lịch đo trong `01_ke_hoach_moc6.md`. Nếu không thể khôi phục, lập phương án môi trường mới với dung lượng và provenance rõ trước khi triển khai.
