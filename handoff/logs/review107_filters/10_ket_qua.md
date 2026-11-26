# R107 — ứng viên PF sau ghép lại không đạt cổng

Đã hoàn tất thí nghiệm và Tier 1; **không chọn ứng viên**. IDSW giảm 3,35%, chưa đạt cổng 10%; HOTA và IDF1 giảm. Không mở tập chuyển miền theo đúng quy tắc dừng. Không đổi tracker hay cấu hình luận văn. Review Codex phiên khác còn chờ orchestrator.

## Cơ chế và phạm vi

Giao thức `00_protocol.md` được ghi trước khi mở điểm R107. Plugin `pf_postgap.py` chỉ được cài tạm trong tiến trình replay rồi phục hồi method gốc trong `finally`. Sau khoảng không cập nhật, chờ lần ghép lại và một quan sát liên tiếp kế tiếp. Từ hai tâm detection thật, dịch vận tốc Cartesian trung bình có trọng số của hạt; giữ phân bố dư vận tốc, logw, omega, hộp hạt và RNG. Không đổi association, không nội suy, không dùng thêm likelihood giả, không sửa output quá khứ.

Căn cứ gần nhất được kiểm lại từ các dòng chẩn đoán R102: 13 IDSW rẽ ở camera2 seed0, 7 track cũ Lost, 4 ca Lost có detection IoU≥0,5 nhưng cost>0,8. Đây là tái tính trên log đã lưu, **không** chạy lại replay R102 hay xác nhận nhân quả. Cơ chế thử chỉ có thể giúp sau khi đã ghép lại được; không cứu lần ghép đầu bị từ chối.

Phạm vi: UA-DETRAC **train dev12** theo manifest có sẵn, cache YOLOv8n COCO `yolov8n-coco_conf0.05_all`, cutoff>0,25, classes 2/3/5/7, max_det300, buffer30, 25 fps, elapsed skip; PF500 K1K2 motion2, seed0/1/2. Cùng detector/cache/GT/cấu hình đã khóa. CPU replay, không inference/GPU/train. Chạy đủ 36 clip×seed, 16.919 frame/seed = 50.757 frame; 3.398 lần kích hoạt heuristic.

## Kết quả và cổng

Mỗi seed được TrackEval chấm toàn bộ 12 video, tổ hợp COMBINED đúng theo raw fields; các số PF dưới đây là trung bình của ba **COMBINED từng seed**, không trung bình HOTA video.

- IDSW: nền **697,00** → ứng viên **673,67**, giảm **23,33 (3,35%)**. Cổng yêu cầu ≤627,30: **trượt**.
- HOTA: **57,0057% → 56,7771%**, giảm 0,2286 điểm phần trăm; qua giới hạn không giảm quá 0,5 điểm.
- FN: **39.395,67 → 39.545,00**, tăng 149,33 (0,379%); qua giới hạn +0,5% (≤39.592,645).
- IDF1: **68,3610% → 67,9307%**, giảm 0,4303 điểm; FP **13.485,00 → 13.463,67**. MOTA **54,1464% → 54,0568%**.
- IDSW seed0/1/2: **689/702/700 → 687/660/674**, chênh **−2/−42/−26**. Cả ba không tăng, nhưng mức giảm khác nhau đáng kể.
- Trung bình seed theo video: **6 giảm, 6 tăng**, không đạt yêu cầu số giảm > số tăng. Bootstrap ghép cặp 12 video, 5.000 mẫu, seed107: khoảng percentile95 của tổng chênh IDSW **[−56;+5]**, đi qua 0. Đây là mô tả thăm dò, không chứng minh lợi ích tổng quát.

Tham chiếu KF/EKF/UKF được tái hợp trực tiếp từ raw R70 cùng dev: HOTA **59,0002/59,2521/58,7862%**, IDSW **489/472/474**, FN **37.564/37.448/37.458**. Ứng viên PF vẫn cách các bộ lọc này đáng kể về IDSW; không tuyên bố hội tụ về mức KF/EKF/UKF.

Do trượt cổng, **không chạy RoundaboutHD chuyển miền**, không BDD val/UA test, không tune hay thử thêm scalar/biến thể. Không có kết luận mới trên dataset khác hay Jetson Nano.

## Kiểm tra thực đã chạy

Lệnh dùng Python được chỉ định. Cú pháp lặp cho các stage là:

```powershell
& 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe' -B handoff/logs/review107_filters/tier1.py <stage> [label]
```

`tier1_checks.json` ghi argv con đầy đủ, cwd, thời gian, exit, log và môi trường CPU; mọi subprocess đều được chờ kết thúc.

- `unit`: exit1, lỗi adapter fixture gọi `.numpy()` trên ndarray. `unit unit_fixed`: exit1, assertion fixture đòi Lost track phải hết K1 prior dù vừa predict mà chưa update. Hai lỗi fixture đã sửa, giữ cả log; không đổi plugin giữa các lượt. `unit unit_fixed2`: **exit0, 4/4 test**. Kiểm vận tốc có trọng số không đều, phân bố dư/hộp/omega, covariance hữu hạn/PSD, thứ tự gap, không tiêu thụ thêm RNG hay đổi logw, chỉ một update, bỏ qua dừng/chậm, phục hồi monkeypatch khi exception, fixture tracker thực và tái lập.
- `regression`: **exit0**, 10 hàm/21 assertion PF gốc. Không gọi test11 CLI vì nó ghi vào data ngoài quyền R107; đường CLI đó **chưa kiểm** trong lượt này. Runner mới gọi `track_video` trực tiếp.
- `preflight`: **exit0**, cấu hình và cache-generation spec khớp metadata baseline R95; raw R72 ba seed tái hợp khớp CSV; ghi hash trước dev trong `inputs.json`.
- `smoke`: **exit0**, plugin tắt prefix128 của MVI_20052/20062 ×3 seed khớp **mảng MOT chính xác 6/6** với R95; ứng viên seed0 lặp khớp mảng MOT và event **2/2 clip**. Không gọi đây là MOT byte parity vì định dạng text xuất có thể khác.
- `dev`: **exit0**, 36/36 đơn vị, 366,31 giây chạy tracking; không phải phép đo Nano hay so tốc độ ghép cặp.
- `eval`: **exit0**, ba seed TrackEval HOTA/CLEAR/Identity chấm đủ và raw COMBINED khớp; cổng false được giữ.
- `integrity`: **exit0**, kiểm 36 MOT hữu hạn, hộp dương, không trùng frame/ID, đủ số frame; **91 hash** nguồn/config/cache/GT/baseline/HEAD/index đã chụp không đổi sau chạy.
- `verify`: **exit0**, dùng trực tiếp `HOTA.combine_sequences` và `Identity.combine_sequences` của TrackEval để đối chiếu sáu tổ hợp nền/ứng viên; kiểm cổng, provenance và output. Raw không lưu LocA nên fill trung tính chỉ cho field không dùng đó, không đưa ra kết luận LocA.
- Lệnh `...python.exe -B handoff/logs/review107_filters/diagnose_inputs.py`: **exit0**, kiểm lại phép đếm R102 và raw KF/EKF/UKF; bằng chứng `diagnosis_verified.json`. Một lệnh truy vấn phụ `python -c` tính thống kê tệp bị PowerShell làm hỏng quote (NameError, exit1); đã dùng `ConvertFrom-Json` để đọc lại, exit0, không thay dữ liệu hay kết quả thử.

## Giới hạn và ngoại lệ

`01_method_limit.md` ghi giới hạn phát hiện lúc dev còn chạy, trước mở điểm: PF nền báo mean/cov trước resampling; plugin tính lại sau resampling khi kích hoạt. Vì vậy hộp báo ra tại frame ấy cũng có thể đổi dù tọa độ hộp của mỗi hạt giữ nguyên. Đây là heuristic có hai tác động liên quan, **không phải ablation thuần vận tốc**; không quy chênh lệch IDSW cho cơ chế vận tốc riêng. Không thực hiện thêm ablation để tách yếu tố này.

Baseline full dùng raw R72 đã có, không replay lại; parity thực chỉ ở hai prefix128. Dev và nguồn chẩn đoán đã được xem nhiều lần, không phải test độc lập. Tái lập thực của ứng viên được kiểm ở smoke/fixture; **chưa lặp full dev**. Không có bằng chứng ưu thế chuyển miền hay FPS Nano.

**Ngoại lệ ghi ngoài write_paths:** lần unit đầu thư viện Ultralytics tự tạo `D:/LABOTARY/bytetrack/Ultralytics/settings.json` vì cha thư mục settings thử chưa tồn tại. `tier1_unit.log` ghi fallback và `diagnosis_verified.json` ghi hash tệp. Đã tạo sẵn settings trong R107 cho mọi lượt sau; không sửa/xóa tệp ngoài phạm vi đó. Vì thế không tuyên bố toàn lượt chỉ ghi trong write_paths. Các nguồn tracker/config/HEAD/index được chụp hash vẫn giữ nguyên.

Không SSH, Git mutation, cài gói hoặc agent con; mọi job tự mở đã được chờ exit. Kiểm kê tiến trình bổ sung bằng `Get-CimInstance Win32_Process` bị sandbox từ chối (`Access denied`); con số 0 do PowerShell in sau lỗi **không phải bằng chứng danh sách rỗng**. Không xác minh inventory toàn máy; bằng chứng kết thúc job của lượt này là trạng thái hoàn tất của các exec session và `subprocess.run` đã chờ trong `tier1_checks.json` (xem `process_check.json`). Mục0 PLAN đã cập nhật trước tệp này. Mốc R106 lúc tiếp nhận vẫn ghi chờ review riêng; phiên này không tự xác nhận review R106 hay R107 đã đạt.

Các tệp trọng yếu: `00_protocol.md`, `pf_postgap.py`, `run_r107.py`, `test_postgap.py`, `inputs.json`, `smoke_checks.json`, `dev_gate.json`, `integrity_verified.json`, `verification.json`, `diagnosis_verified.json`, `tier1_checks.json`, `dev/tracking/`, `dev/evaluation/`.
