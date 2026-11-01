# R63 — Kernel panic trên Nano lúc smoke công cụ (30/10/2026 17:10–17:11)

**Đã dừng mọi phép đo GPU** theo điều kiện dừng của kế hoạch (`01_ke_hoach_r63.md` mục 5) và review GPT
(`D:/LABOTARY/automation/runs_local/hoi_codex_R63_duyet_ke_hoach_20261030/answer.json`: không tự reboot/chạy lại sau panic).
Không có số đo FPS hay thời gian công đoạn nào từ lượt này.

## Diễn biến

- Boot `3cf3e6a1…` (khởi động khoảng 16:26). G0 lúc 16:54 và 17:09 sạch: nhiệt CPU/GPU 33,5 °C, swap 0, không dòng
  kernel nghi vấn, pstore rỗng.
- 17:10:21 chạy smoke chức năng `bench_engine_pipeline.py` (5 ảnh, 2 pass) trên bản sao v16
  `~/nano_run/r63_20261030/baseline/nano_transfer` (562/562 hash), qua `chay_co_giam_sat.sh` (timeout 300 s).
  Lượt không ghi được file nào (kể cả `moi_truong.json`): máy sập ở giai đoạn nạp engine/khởi tạo; `tegrastats.log` rỗng
  (đầu ra bị đệm, chưa kịp ghi). Máy khởi động lại 17:11, boot mới `6244300d-9848-4870-b8dd-6e1ac323f7b6`.
- `/sys/fs/pstore` sau boot có `dmesg-ramoops-0` (SHA256 `9f2d805a…`) và `console-ramoops-0` (`4ef11bff…`), lưu
  nguyên byte trong thư mục này cùng `smoke_bench_5anh/`.

## Chuỗi trong ramoops (uptime giây)

| t | Sự kiện |
|:--|:--|
| 26, 361 | WARNING `cfg80211_ch_switch_notify` từ driver Wi-Fi USB `rtl8821au` (lúc nối mạng; có trước tải) |
| 2645,90 | `mmc1: Data CRC error` + register dump sdhci (thẻ SD gốc) |
| 2656,58 | `tegra-xusb: xHCI host not responding … HC died`, ngắt toàn bộ USB (gồm Wi-Fi) |
| 2667,00 | `rcu_preempt detected stalls` (CPU0) |
| 2670,74 | nvgpu `Job on channel … timed out`, `fake mmu fault` |
| 2675,78 | `tegra-i2c 7000d000: i2c transfer timed out addr: 0x3c`, `max77620-gpio … read failed: -110` (PMIC) |
| 2676,28 | **`Kernel panic - not syncing: Watchdog detected hard LOCKUP on cpu 0`** |

**Trùng chữ ký R62** (`review62/22_dmesg_ramoops.txt`): `mmc1: Data CRC error` (3016,7) → `xusb HC died` (3023,5) →
I2C timeout 0x3c / max77620 (3032,7) → RCU stall (3037,8) → `Kernel panic - not syncing: softlockup: hung tasks` (3064,0).
Hai lần, hai Jetson khác nhau (R62 đã đổi máy) nhưng **cùng thẻ SD, cùng nguồn/phụ kiện (chưa xác minh)**.

## Diễn giải (suy luận, chưa kiểm chứng)

Đồng thời hỏng ba bus độc lập (SDMMC, USB xHCI, I2C tới PMIC) ngay khi GPU bắt đầu tải phù hợp với sụt áp/nguồn không
đủ hoặc lỗi phần cứng dùng chung (nguồn, thẻ SD, đế/cáp), hơn là lỗi mã TensorRT. Không có bằng chứng trực tiếp về điện
áp (không đọc được INA3221 trong G0; không có log undervoltage). Lần này sập ngay khi nạp engine với chỉ 5 ảnh, nên
**không thể** đo baseline/ứng viên một cách tin cậy cho tới khi xử lý nguyên nhân. Không quy cho R63 tools: G0 và runner
`sleep` không gây sự cố; trước đó R62 đã panic với runner khác.

## Cần người dùng (phần cứng, agent không tự làm)

1. Nguồn đang dùng: micro-USB 5 V/2 A hay jack DC 5 V/4 A (có cắm jumper J48)? Khuyến nghị kiểm/đổi sang jack DC 5 V ≥ 4 A.
2. Thẻ SD: thẻ cũ dùng lại qua hai máy và cả hai lần đều có `mmc1: Data CRC error` trước panic — cân nhắc thẻ khác
   (sao lưu `~/nano_run` trước) hoặc kiểm thẻ.
3. Wi-Fi USB `rtl8821au` có WARNING driver; nếu được, dùng cáp Ethernet, hoặc cắm qua hub có nguồn.
4. Sau thay đổi: chạy lại G0 (`cong_cu_v3/kiem_suc_khoe_nano.sh`), rồi một lượt ngắn có giám sát trước mọi phép đo.

## Sửa công cụ rút ra

`chay_co_giam_sat.sh`: tegrastats ghi qua đệm nên mất dữ liệu khi máy sập — cần ghi trực tiếp/flush từng dòng (vd
`tegrastats --logfile` hoặc `stdbuf -oL`) để giữ telemetry tới thời điểm sập.
