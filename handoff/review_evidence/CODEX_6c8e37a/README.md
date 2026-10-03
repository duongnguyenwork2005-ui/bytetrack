# Bằng chứng review 6c8e37a — 2026-10-03

Đọc CODEX_REVIEW_6c8e37a.md trước. Đây là bộ bằng chứng có phạm vi, không phải repository đầy đủ để train.

- check_csv_recovery.py: 69/69; hàm thật và writer batch thật, marker checkpoint chỉ để sao chép; không torch/train.
- check_journal_recovery.py: 54/54; hồi quy cũ, fixture batch cập nhật đúng 20 cột. Không thay tiêu chí nhật ký.
- observe_csv_risks.py: bản nguyên vòng 40; 3 trên base, 1 trên target. Exit 0 không có nghĩa hết rủi ro.
- tree/handoff/logs/review41/06b_quan_sat_qua_buoc_chuan_bi.py: đối chứng tác giả, reviewer chạy lại được 0.
- evidence/evidence_audit.json: so base 9/9, bằng chứng cũ 52/52, đếm log tác giả; không gọi log tác giả là test reviewer chạy.
- audit_evidence.py ghi cách đối chiếu trong workspace review: cần gói đã tái dựng và snapshot review_421dd53 bên cạnh. Không tự đủ trong ZIP rút gọn này.
- check_package.py cần ZIP delta đầu vào đúng hash. Nó dựng các file thay đổi và base_preimages vào thư mục cha của thư mục chứa --out; hãy dùng thư mục kiểm mới.
- tree/src và base_preimages/src chỉ gồm ba file liên quan, không đủ phụ thuộc cho train_detector CLI.
- Thư mục evidence là kết quả reviewer; log dưới tree/handoff/logs/review41 là tác giả bàn giao.

## Tái lập độc lập (không train)

```text
python check_csv_recovery.py --repo tree --out csv_recovery_rerun.json
python check_journal_recovery.py --repo tree --out journal_recovery_rerun.json
python observe_csv_risks.py --repo base_preimages --out observer_base_rerun.json
python observe_csv_risks.py --repo tree --out observer_target_rerun.json
python tree/handoff/logs/review41/06b_quan_sat_qua_buoc_chuan_bi.py --out counter_rerun.json
```

Trong các phép trên, việc os.replace thất bại được giả lập; không kiểm mất điện vật lý.
SHA256SUMS.txt bao phủ mọi file ngoài chính nó. Không có dataset, weights, video, ZIP nguồn hoặc cache NPZ.
