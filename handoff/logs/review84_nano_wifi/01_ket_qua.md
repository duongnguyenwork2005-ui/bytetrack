# Nano USB Wi-Fi — 05/11/2026

Người dùng báo USB Wi-Fi lỗi và đang dùng LAN. SSH alias `nano` cũ trỏ 192.168.1.5 không kết nối; tìm đúng Nano qua LAN 192.168.1.4 bằng khóa SSH. USB `2357:011e` Realtek được `lsusb` nhận, `wlan0` dùng driver `rtl8821au`, không bị rfkill và quét thấy mạng VIETTEL_5G. Vì vậy không có bằng chứng adapter hoặc driver mất nhận dạng.

NetworkManager trước sửa báo `no-secrets`/`No agents were available`; hồ sơ VIETTEL_5G có `psk-flags=0` nhưng không có khóa đã lưu. SSH từ xa không được polkit cho phép sửa mạng và `sudo -n` yêu cầu mật khẩu. Người dùng nhập xác thực trong terminal Nano bằng `sudo nmcli --ask connection up uuid f6ce5c29-9a28-401d-bf3e-6dbaf549ef09 ifname wlan0`, sau đó chạy `bash ~/restore_wifi_20261105.sh`. Script yêu cầu mật khẩu Wi-Fi ở chế độ ẩn, lưu vào hồ sơ NetworkManager, ngắt/nối lại chính Wi-Fi; giữ LAN. Không có mật khẩu trong script, repo hoặc báo cáo.

Hậu kiểm: NetworkManager ghi keyfile được cập nhật và khi nối lại báo `secrets exist; no new secrets needed`; `wlan0` kết nối, IP `192.168.1.5/24`, SSH qua alias `nano` thành công. Ping qua `wlan0`: router 3/3, Internet 1.1.1.1 3/3. Route mặc định LAN metric100, Wi-Fi metric600, nên LAN vẫn dự phòng khi cắm dây. Chưa cần thay driver, cài gói hay khởi động lại Nano.
