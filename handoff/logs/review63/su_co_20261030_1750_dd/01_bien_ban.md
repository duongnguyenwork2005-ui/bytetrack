# R63 — Panic khi CHỈ ĐỌC thẻ SD, không tải GPU (30/10/2026 ~17:50)

Người dùng chạy bước chẩn đoán 3 (`sudo dd if=/dev/mmcblk1 of=/dev/null bs=4M status=progress`, chỉ đọc) và đồng thời
`sudo tegrastats --interval 200 --logfile ~/nguon_khi_tai.log`. Agent không chạy gì trên Nano trong lúc này. Máy tự tắt
giữa lúc `dd` chạy; boot mới `42a6b530-b472-40a1-b149-2bb518881bf1` (17:51). Bằng chứng nguyên byte trong thư mục này:
`dmesg-ramoops-0` (`50292897…`), `console-ramoops-0` (`1747a8f0…`), `nguon_khi_tai.log` (`ca0fb020…`).

Chuỗi trong ramoops (uptime giây): **545,42 `mmc1: Data CRC error`** → 551,04 `tegra-xusb … HC died` → 566,52
`rcu_preempt detected stalls` → **576,32 `Kernel panic - not syncing: Watchdog detected hard LOCKUP on cpu 0`**.
Giống hệt R62 và sự cố 17:10 (`../su_co_20261030_1710/01_bien_ban_su_co.md`), nhưng lần này **không có GPU/TensorRT**.

Nguồn trước khi sập (6 dòng log cuối, mỗi 200 ms): POM_5V_IN 2,27–2,39 W (tb 2,57 W), GR3D 0 %, CPU ≤ 9 %, CPU 34 °C,
PMIC 50 °C. Cực đại cả log 5,21 W. Lúc nhàn rỗi trước đó: 5,128 V / 568 mA (jack DC 5 V 4 A 20 W).

Ngoài ra máy còn khởi động lại một lần khoảng 17:30 không có ramoops (không ai thao tác; có thể cũng trong lúc `dd`).

## Kết luận (đo được / suy luận)

- Đo được: sự cố tái hiện **không cần GPU**, chỉ bằng đọc tuần tự thẻ SD; cả ba lần có ramoops đều mở đầu bằng
  `mmc1: Data CRC error`; công suất lúc sập thấp (~2,3 W), không có đỉnh trước sập ở độ phân giải 200 ms.
- Suy luận: nguyên nhân nằm ở **đường thẻ SD** (thẻ `acer` 58,2 GiB dùng chung qua hai board, khe/đế SD, hoặc chế độ
  UHS SDR104) chứ không phải TensorRT, GPU hay công suất bộ nguồn. Sụt áp tức thời < 200 ms chưa loại trừ tuyệt đối.
- Hệ quả: mọi phép đo tốc độ R62/R63 trên thẻ này có rủi ro sập ngẫu nhiên theo I/O; không đo tiếp cho tới khi đổi thẻ
  và chạy lại phép thử `dd` sạch.

## Phép thử thẻ hộp (30–31/10, boot từ eMMC, không Wi-Fi USB; người dùng chạy tại chỗ, bằng chứng = ảnh màn hình)

- Boot rootfs eMMC (`extlinux.conf` → `root=/dev/mmcblk0p1`, bản gốc giữ ở `extlinux.conf.r63_goc_the_sd`).
- Thẻ hộp (`mmc1:0001 SD 58.2 GiB`, p1): `dd` đọc toàn thẻ **xong, không sập**: 62.534.975.488 byte, 25.679,5 s,
  **2,4 MB/s**. `ios`: 204 MHz, SDR104, 4 bit, 1,8 V. Đo nhanh `iflag=direct` 188 MiB: 12,6 MB/s.
- `grep -niE "crc|error|retun|timeout|HC died"` trên dmesg toàn phiên: 25 dòng, **không dòng nào của mmc1**
  (Host read timeout lúc boot, imx219 không có camera, `tegradc dp: aux write got error` của màn hình).
- Diễn giải: khe/bus SD chạy SDR104 hơn 7 giờ không lỗi CRC ⇒ ít khả năng do khe; thẻ hộp chậm do bản thân thẻ.
  Còn chờ: đo nhanh + `ios` + `dd` toàn thẻ acer cùng điều kiện.
- Thẻ acer cùng điều kiện (31/10), đo nhanh `iflag=direct` 256 MiB: **19,1 MB/s**, không sập; `ios` y hệt (204 MHz,
  SDR104, 4 bit, 1,8 V). Người dùng hỏi lần sập 17:50 có phải chỉ là màn hình tự tắt: **không** — ramoops có
  `Kernel panic - not syncing: Watchdog detected hard LOCKUP` và boot_id đổi; riêng lần khởi động lại ~17:30 không
  có ramoops nên chưa xác định. Màn hình tự tắt tương ứng dòng `tegradc dp: aux write got error`, vô hại.
- Thẻ acer `dd` đọc toàn thẻ cùng điều kiện (boot eMMC, không Wi-Fi USB): **người dùng báo "Sập rồi"** (31/10).
  Chờ ramoops/boot_id để xác nhận chữ ký. Nếu chữ ký trùng: thẻ hộp sạch 7,1 h vs thẻ acer sập, cùng board/khe/nguồn
  ⇒ **thủ phạm là thẻ acer**; Wi-Fi USB và rootfs-trên-thẻ không cần thiết để tái hiện.
  **Đính chính ngay sau đó:** người dùng cho biết lượt acer này **có cắm Wi-Fi USB** ⇒ kết luận trên TẠM HOÃN cho tới
  khi biết lượt thẻ hộp có cắm Wi-Fi hay không (kiểm khách quan bằng `0bda|realtek|8821` trong hai log dmesg).
  Người dùng xác nhận: **lượt thẻ hộp cũng cắm Wi-Fi USB** ⇒ hai lượt chỉ khác thẻ ⇒ **thủ phạm là thẻ acer**
  (Wi-Fi có thể là yếu tố góp phần, chưa loại trừ, nhưng không cần thiết để giải quyết: bỏ thẻ acer).
- Sau lần sập thẻ acer: boot `7d1309ac-57ea-4e24-b6c6-ddf77583276e` (31/10 07:41), **`/sys/fs/pstore` rỗng** — lần
  này không có ramoops (khác ba lần trước). Chưa biết máy tự khởi động lại hay người dùng rút nguồn khi treo (rút nguồn
  mất RAM ⇒ mất ramoops). Bằng chứng còn lại: `~/dd_the_acer_dmesg.log`, `~/dd_the_acer.log` trên eMMC.
  Người dùng làm rõ: máy **tự khởi động lại** (giống watchdog panic các lần trước), sau đó người dùng **shutdown, rút
  thẻ, bật lại** ⇒ ramoops (chỉ sống qua warm reboot) bị mất do tắt nguồn, không phải do sự cố khác loại.
  Bài học: sau sự cố, chép `/sys/fs/pstore` TRƯỚC khi tắt máy.
- Log dmesg của lượt acer bị ghi nối vào `~/dd_the_hop_dmesg.log` (terminal 1 chưa đổi tên): 38276,8 s thẻ hộp
  removed → 38287,8 s `mmcblk1: mmc1:0001 acer 58.2 GiB` → 38288,5 s EXT4 mount p1; **sau đó không còn dòng nào**.
  `~/dd_the_acer.log` tạo 07:35, **0 byte**; boot mới 07:41:16 ⇒ sập trong ~5 phút đầu đọc toàn thẻ (ước ~5–6 GB ở
  19 MB/s). Dòng kernel cuối và tiến độ `dd` chưa kịp ghi xuống eMMC trước khi sập (page cache mất) ⇒ **lần này không
  có chữ ký log**; kết luận dựa trên so sánh hành vi cùng điều kiện (thẻ hộp sạch 7,1 h vs thẻ acer sập ~5 phút) cộng
  ba lần ramoops trước đều mở đầu bằng `mmc1: Data CRC error` trên chính thẻ acer.
- eMMC chưa đủ để chạy dự án: python3 (3.6) **không có module `tensorrt`** (thiếu `python3-libnvinfer`), **không có
  python3.8**; `wlan0` có driver (DOWN, chưa kết nối), `eth0` không cắm cáp.
- 31/10 ~08:30, SSH vào eMMC (Wi-Fi 192.168.1.5, khóa `id_ed25519_nano` người dùng thêm vào authorized_keys). Lệnh
  `apt-get install python3-libnvinfer python3-opencv python3-pip` của người dùng **thất bại nguyên khối** (không cài
  gì); `pip install --user numpy==1.19.5` (pip 9.0.1) build từ nguồn và lỗi với Cython 3.0.12 có sẵn trong `~/.local`.
  Kiểm chỉ đọc: hệ eMMC là JetPack **cài dở**: `libnvinfer*` trạng thái `iU` (unpacked, chưa configure),
  `libcudnn8` `rU` (deinstall nhưng file còn), `libc-bin` `it`, **không có `/usr/local/cuda*`** (không libcudart),
  apt hỏng (`libnvinfer-dev` cần `libcudnn8-dev` 830 MB); OpenCV hệ thống là NVIDIA 4.1.1 (thẻ acer: 3.2.0).
  ⇒ chạy từ eMMC đòi sửa apt + cài cudart/cudnn/python3-libnvinfer/libopencv-python, môi trường KHÁC v16/R62.
- Người dùng chọn B1 (ảnh thẻ acer → thẻ mới), muốn biết trước mẫu thẻ có hợp Nano không ⇒ phép thử hạ xung (tạm thời,
  mất khi rút thẻ/reboot): boot `7d1309ac…`, cắm lại thẻ acer (automount `EXT4-fs recovery complete` — journal replay
  sau lần sập), umount; `echo 100000000 > /sys/kernel/debug/mmc1/clock` ⇒ `ios` clock 100000000 Hz (từ 204000000).
  Đo nhanh 256 MiB `iflag=direct`: **18,7 MB/s** (204 MHz: 19,1) ⇒ thẻ là nút thắt, hạ xung gần như không mất tốc độ.
  Tiếp: `dd` toàn thẻ ở 100 MHz (log `~/dd_acer_100mhz.log`, `~/acer_100mhz_dmesg.log`, vòng `sync` mỗi 2 s).
- **Kết quả 100 MHz: `dd` toàn thẻ acer XONG, không sập**: 62.534.975.488 byte, 2.946,05 s, **21,2 MB/s**; cùng boot
  `7d1309ac…` (uptime 1 h 39 m), **0 dòng lỗi mmc1** trong dmesg; theo dõi SSH mỗi phút qua mốc ~6 GB (chỗ sập ở 204 MHz).
  ⇒ thẻ acer chỉ lỗi ở SDR104 **204 MHz**; ở 100 MHz đọc sạch, tốc độ không giảm (thẻ là nút thắt ~20 MB/s). Chưa
  phân biệt lỗi riêng chiếc thẻ hay cả mẫu thẻ, nhưng không cần: giới hạn xung là cách xử lý không tốn tiền.
- Device tree hiện hành (`/boot/dtb/kernel_tegra210-p3448-0002-p3449-0000-b00.dtb`, md5 `1bdd4238…`, extlinux `FDT`):
  khe SD = `sdhci@700b0400` (= `sdhci-tegra.2` = mmc1), `max-clk-limit` = 0x0c28cb00 (204 MHz), `uhs-mask` 0x0c;
  eMMC = `sdhci@700b0600`. Đề xuất cố định: bản sao DTB với `max-clk-limit` = 0x05f5e100 (100 MHz) + đổi dòng `FDT`
  (giữ bản gốc) — **chờ người dùng duyệt** (thay đổi hệ thống + reboot).
- **Người dùng duyệt (31/10)**; người dùng chép bằng sudo, agent kiểm hash qua SSH: DTB mới `3109b3a0…` (chỉ khác
  `max-clk-limit` 0xc28cb00 → 0x5f5e100 so với gốc `e6301b1f…`), `extlinux.conf` `9ccfb198…` (FDT → `_sd100.dtb`, thêm
  LABEL `goc204` dự phòng dùng DTB gốc, root eMMC), sao lưu `extlinux.conf.r63_truoc_sd100` `ab04f462…`; bản thời
  thẻ SD `extlinux.conf.r63_goc_the_sd` `f47da152…` giữ nguyên. Thư mục chuẩn bị: `~/r63_sd100/` (có `sha256.txt`).
- Reboot → boot `b1718903-ac6f-4918-8acb-20787a27c21e`; `/proc/device-tree/sdhci@700b0400/max-clk-limit` = 0x05f5e100
  (DTB mới được nạp). Nhưng thẻ vào **`new high speed SDXC card`** (không UHS): `ios` clock **50 MHz**, không phải
  SDR104@100 MHz như phép thử debugfs — có lẽ SDR104 bị loại vì trần 100 MHz và SDR50/DDR50 đã bị `uhs-mask` 0x0c che
  (suy luận). Chế độ này bảo thủ hơn (3,3 V, không tuning) nhưng là cấu hình KHÁC, cần đo tốc độ + đọc 15 GB lại.
  `ios`: clock 50000000 Hz, timing `2 (sd high-speed)`, signal 3,30 V. Đo nhanh 256 MiB `iflag=direct`: **14,1 MB/s**
  (SDR104@204: 19,1; SDR104@100 debugfs: 18,7) ⇒ chậm ~25 % khi đọc thẻ. Agent đề xuất giữ HS 50 MHz (biên tín hiệu
  rộng nhất; phép đo R63 lặp lại đọc từ page cache) thay vì bỏ che SDR50 trong `uhs-mask` (chế độ NVIDIA cố ý tắt,
  chưa thử).
- Đọc 15 GB ở HS 50 MHz (vượt xa mốc ~6 GB từng sập): **xong, sạch**: 15.099.494.400 byte, 1.312,69 s, **11,5 MB/s**
  trung bình; cùng boot `b1718903…`, 0 dòng lỗi mmc1. Đã chuẩn bị `~/r63_sd100/extlinux.conf.the_sd` (`42333ea6…`):
  chỉ đổi `root=/dev/mmcblk0p1` → `mmcblk1p1` ở LABEL primary, giữ FDT `_sd100` và LABEL `goc204` (eMMC, DTB gốc).
- Người dùng chép (lần đầu không ăn, chép lại), agent kiểm `extlinux.conf` = `42333ea6…`; sao lưu bản eMMC+sd100 là
  `extlinux.conf.r63_emmc_sd100` (`9ccfb198…`). Reboot → boot `5d3d12a9-ac4d-4d50-a769-718d19f8e405`, rootfs
  `/dev/mmcblk1p1` (thẻ acer, 58 G, trống 26 G), DT `max-clk-limit` 0x05f5e100, thẻ `new high speed SDXC`.
  Engine `c32690cd…` khớp; bản sao v16 `r63_20261030/baseline/nano_transfer` 562/562 hash khớp; env conda `nano` còn.
- **G0** `r63_20261030/g0_suc_khoe_05_the_sd100`: MAXN, CPU/GPU 33,5/34 °C, swap 0, không dấu hiệu panic, exit khác 0
  rỗng; tiến trình nền: `jtop --force` (dịch vụ), `nvpmodel_indicator`, `packagekitd`.
- Runner sửa thành `cong_cu_v4` (chỉ `chay_co_giam_sat.sh` đổi, `b487be48…`): `stdbuf -oL tegrastats` + vòng `sync`
  2 s để giữ telemetry khi sập; thử với `sleep 3` đạt.
- **Smoke 5 ảnh lặp lại đúng lượt từng gây panic 17:10** (`smoke_bench_5anh_the_sd100`, timeout 300 s): exit 0,
  không vi phạm cổng (CPU/GPU max 35,5/34 °C, swap +7 MB, 197 mẫu tegrastats, GR3D 99 %, dmesg mới không có dòng nghi
  vấn), boot_id không đổi. `khoi_tao_s` 51,2 s (lần đầu đọc từ thẻ chậm). TRT cảnh báo `Using an engine plan file
  across different models of devices` — engine `c32690cd…` được build ở R62 trên board trước. Đây là smoke chức năng,
  **không** phải số đo tốc độ.
- **Kết luận sự cố:** nguyên nhân panic = thẻ acer ở SDR104 204 MHz; đã xử lý bằng DTB giới hạn 100 MHz (thực tế
  HS 50 MHz, 3,3 V). Hoàn tác: chọn LABEL `goc204` trong menu U-Boot, hoặc chép lại `extlinux.conf.r63_goc_the_sd`.
- eMMC: `/dev/mmcblk0p1` 14 G, dùng 11 G, trống **2,4 G**; có `libnvinfer8 8.2.1-1+cuda10.2` (+ onnxparser, plugin).

## Phép thử phân biệt tiếp theo (người dùng)

1. Tắt Nano, đọc toàn bộ thẻ cũ trên đầu đọc thẻ của PC (tạo ảnh sao lưu) — đọc sạch trên PC ⇒ nghi khe/đế SD của
   Nano hoặc SDR104; lỗi trên PC ⇒ thẻ hỏng.
2. Ghi ảnh sang thẻ mới (dung lượng thực ≥ 58,2 GiB) hoặc cài JetPack 4.6.x mới, khởi động Nano bằng thẻ mới, chạy lại
   `dd` đọc toàn thẻ + G0. Không sập ⇒ lỗi do thẻ cũ. Vẫn sập ⇒ khe SD/board; cân nhắc giới hạn tốc độ SD hoặc chạy từ eMMC.
