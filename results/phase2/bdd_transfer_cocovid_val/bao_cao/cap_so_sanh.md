# Cặp so sánh — chênh lệch ghép cặp theo video

Bootstrap **5000 lần, seed 0**, ghép cặp **theo video**; COMBINED tái tổ hợp bằng `src/boot_stats.py`. Các cặp đã **khóa trước** trong YAML.

> **Ngoại lệ PF — hai bảng là hai đại lượng khác nhau.** Với `pf500`, **bảng chính** lấy trung bình **metric** theo ba seed, còn **bootstrap ở đây** lấy trung bình **thống kê đủ** theo seed rồi mới tái tổ hợp. Với metric phi tuyến, hai cách cho kết quả **khác nhau**, nên Δ của `pf500` dưới đây **không** đúng bằng hiệu hai ô trong bảng chính. Cách tổng hợp này đã khai báo trong bản khóa từ trước, **không** chọn sau khi xem kết quả (`PHASE2_REPORT.md` mục 15.2).

## `c25`

| cặp | metric | Δ (COMBINED) | CI 95 % | CI chứa 0 |
|:--|:--|--:|:--|:--|
| ekf − cv | HOTA | 0.0015 | [0.0005, 0.0026] | không |
| ekf − cv | IDF1 | 0.0046 | [0.0028, 0.0064] | không |
| ekf − cv | AssA | 0.0038 | [0.0010, 0.0068] | không |
| ekf − cv | IDSW | -797 | [-1,055, -558] | không |
| ukf − cv | HOTA | 0.0009 | [-0.0003, 0.0022] | **có** |
| ukf − cv | IDF1 | 0.0038 | [0.0016, 0.0059] | không |
| ukf − cv | AssA | 0.0019 | [-0.0014, 0.0054] | **có** |
| ukf − cv | IDSW | -690 | [-923, -473] | không |
| ukf − ekf | HOTA | -0.0006 | [-0.0013, 0.0002] | **có** |
| ukf − ekf | IDF1 | -0.0008 | [-0.0021, 0.0004] | **có** |
| ukf − ekf | AssA | -0.0019 | [-0.0039, 0.0001] | **có** |
| ukf − ekf | IDSW | 107 | [31.0000, 186] | không |
| pf500 − cv | HOTA | -0.0402 | [-0.0443, -0.0362] | không |
| pf500 − cv | IDF1 | -0.0614 | [-0.0679, -0.0552] | không |
| pf500 − cv | AssA | -0.0644 | [-0.0724, -0.0565] | không |
| pf500 − cv | IDSW | 1,695 | [1,334, 2,073] | không |
| pf500 − ekf | HOTA | -0.0417 | [-0.0457, -0.0378] | không |
| pf500 − ekf | IDF1 | -0.0660 | [-0.0724, -0.0598] | không |
| pf500 − ekf | AssA | -0.0681 | [-0.0758, -0.0607] | không |
| pf500 − ekf | IDSW | 2,492 | [2,089, 2,901] | không |

## `c10`

| cặp | metric | Δ (COMBINED) | CI 95 % | CI chứa 0 |
|:--|:--|--:|:--|:--|
| ekf − cv | HOTA | 0.0025 | [0.0012, 0.0039] | không |
| ekf − cv | IDF1 | 0.0060 | [0.0040, 0.0082] | không |
| ekf − cv | AssA | 0.0062 | [0.0027, 0.0098] | không |
| ekf − cv | IDSW | -825 | [-1,066, -601] | không |
| ukf − cv | HOTA | 0.0011 | [-0.0003, 0.0025] | **có** |
| ukf − cv | IDF1 | 0.0042 | [0.0020, 0.0064] | không |
| ukf − cv | AssA | 0.0023 | [-0.0014, 0.0062] | **có** |
| ukf − cv | IDSW | -704 | [-934, -490] | không |
| ukf − ekf | HOTA | -0.0014 | [-0.0023, -0.0005] | không |
| ukf − ekf | IDF1 | -0.0018 | [-0.0033, -0.0003] | không |
| ukf − ekf | AssA | -0.0039 | [-0.0062, -0.0016] | không |
| ukf − ekf | IDSW | 121 | [43.0000, 202] | không |
| pf500 − cv | HOTA | -0.0399 | [-0.0437, -0.0361] | không |
| pf500 − cv | IDF1 | -0.0607 | [-0.0667, -0.0550] | không |
| pf500 − cv | AssA | -0.0629 | [-0.0705, -0.0558] | không |
| pf500 − cv | IDSW | 1,670 | [1,323, 2,042] | không |
| pf500 − ekf | HOTA | -0.0424 | [-0.0461, -0.0386] | không |
| pf500 − ekf | IDF1 | -0.0668 | [-0.0726, -0.0611] | không |
| pf500 − ekf | AssA | -0.0691 | [-0.0762, -0.0623] | không |
| pf500 − ekf | IDSW | 2,495 | [2,119, 2,888] | không |


**Đọc đúng mức bằng chứng**

- CI **chứa 0** ⇒ *không đủ bằng chứng về khác biệt*, **không** phải bằng chứng về sự tương đương.
- CI **không chứa 0** ⇒ phát biểu đúng là *"CI 95 % của phép so sánh đã định trước không chứa 0"*, **không** phải "lợi thế là thật".
- Đây là CI **riêng cho từng phép so sánh**. Bảng có nhiều cặp × nhiều metric, nên **chưa** có bảo đảm đồng thời (không hiệu chỉnh đa so sánh).
- Phạm vi: protocol **gộp phương tiện thành một lớp** trên nhãn **COCO-video chuyển đổi**. Đây **không** phải kết quả benchmark BDD100K native, và **không** phải bằng chứng về khả năng phục hồi sau che khuất (nguồn nhãn không có thuộc tính che khuất).

