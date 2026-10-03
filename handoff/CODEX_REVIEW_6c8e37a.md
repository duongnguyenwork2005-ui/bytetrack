# Review vòng 41 — `6c8e37a`

**Kết luận: PASS WITH LIMITATIONS. Đóng R40-N1 (`results.csv`) và R40-N2 (`theo_batch.csv`) trong phạm vi chuẩn bị resume thật.** Không phát hiện lỗi chặn trong phần sửa được kiểm. Có một đính chính mô tả mức thấp, R41-N1 dưới đây; không yêu cầu mở thêm một vòng sửa mã.

Review này không phê duyệt training dài, không xác nhận chịu được mọi dạng mất điện và không nghiệm thu hai đề xuất guard ghi nối / khóa lượt chạy.

Ngày review: 2026-10-03. Reviewer: Codex.

## 1. Snapshot và phạm vi

| Mục | Giá trị |
|:--|:--|
| Gói nhận | `review_delta_latest.zip` |
| Kích thước | 690.663 byte |
| SHA256 | `8920742ecfe6c3978175855746130339a9b36fefe74439fa69cfb98a8cc6d1dc` |
| Base | `421dd53ecbe171ff11219a7efba900cdbbe5a34e` |
| Target | `6c8e37aca2326451f0a46f3a8838b6171fe66b3d` |
| Nguồn chính được kiểm | `src/tiep_tuc_huan_luyen.py`, `src/train_detector.py`, test resume và bằng chứng vòng 41 |
| SHA256 module resume target | `44045186de71dbb720d8d19d07f00f898576c3eb41cef950d3943546accbecc0` |

Đã kiểm ZIP, payload, dedup và patch: **84 thành viên, 78 payload + 5 đường dẫn dùng lại payload, 83/83 đường dẫn roundtrip khớp**, 0 đường dẫn cấm. Chín file base khôi phục từ patch khớp hash manifest và **9/9 trùng byte với snapshot `421dd53` đã review trước đó**. Bằng chứng review cũ được nhúng trong delta khớp `SHA256SUMS.txt` **52/52**.

Đây là kiểm **các đường dẫn thay đổi**, có đối chiếu base đã lưu. Reviewer **không có toàn bộ Git repository**, không tự kiểm toàn cây clone như tác giả và không xác minh HEAD/working tree trên máy Windows. Tuyên bố “kiểm độc lập toàn cây đạt” của tác giả không được đổi thành kết quả reviewer tự chạy.

## 2. Kết quả độc lập và đối chiếu log

| Phép kiểm | Kết quả | Phạm vi |
|:--|:--|:--|
| `check_csv_recovery.py` của reviewer | **69/69**, exit 0 | Hàm lập kế hoạch, sao lưu, xử lý CSV và writer batch thật trên thư mục tạm |
| `check_journal_recovery.py` của reviewer | **54/54**, exit 0 | Hồi quy nhật ký vòng 40; chỉ cập nhật fixture batch thành schema 20 cột hợp lệ |
| `observe_csv_risks.py` cũ trên base / target | **3 → 1** | Tái chạy nguyên script quan sát; exit 0 là hoàn tất quan sát, không phải mọi rủi ro đã đóng |
| Đối chứng `06b_quan_sat_qua_buoc_chuan_bi.py` của tác giả, reviewer chạy lại | **0** rủi ro quan sát được | Batch đi qua bước chuẩn bị trước writer |
| Suite resume tác giả, lần 2 | **138/138**, exit 0 | **Đối chiếu log**, không lặp lại training CPU |
| Nhóm C mới trên mã cũ | **11/31**, exit 1 | **Đối chiếu log**: 20 FAIL, phù hợp dự đoán và lỗi cần bắt |
| Detector / pilot | **92/92 / 34/34**, exit 0 | **Đối chiếu log**, không tự chạy lại hai suite |

Môi trường review không có torch/Ultralytics. Các checkpoint của phép kiểm độc lập là file marker **chỉ được sao chép**, không được nạp như model. Writer batch `GhiLuotDayDu.het_epoch_train()` chạy thật. Phép ghi thêm `results.csv` của reviewer dùng hàng CSV fixture tường minh, **không phải** `BaseTrainer.save_metrics` thật; việc nối qua Ultralytics và entry point `train_detector.py` thật được đối chiếu từ mã test và log tác giả.

Không có training GPU/CPU, SSH Nano, sửa mã dự án, commit, push hay thay dataset trong review này.

## 3. Vì sao đóng hai finding

**R40-N1 — `results.csv`: ĐÓNG.** Các đuôi NUL và trường epoch đang ghi dở được loại đúng; hàng hoàn chỉnh thiếu newline được giữ. LF, CRLF và CRLF bị cắt sau `\r` đều được kiểm. Khi checkpoint ở epoch 2, hàng hoàn chỉnh epoch 3 bị loại theo quy tắc đã có; hàng hợp lệ của epoch 1–2 không đổi. Không còn lỗi chuyển NUL sang số hoặc giữ đuôi `1` để nó dính vào hàng epoch kế tiếp trong các ca đã tái hiện.

**R40-N2 — `theo_batch.csv`: ĐÓNG.** Phần đuôi được xử lý trước lần append đầu tiên của resume. Lịch sử hợp lệ giữ nguyên, gồm `lan`, ô số tùy chọn trống và hàng có epoch lớn hơn checkpoint được chọn. Writer thật nối thêm một hàng đúng 20 cột; không dính hàng. Kiểm độc lập còn chạy bước chuẩn bị lần tiếp theo để xác nhận file đã sửa tiếp tục dùng được.

Cả hai file đều có bản sao nguyên byte và biên nhận khớp hash/hành động. Thay đổi sau preflight, thiếu hoặc sai bản sao, file biến mất đều bị từ chối trong phép kiểm độc lập. Lỗi `os.replace` được **giả lập riêng cho từng CSV**: file đích tương ứng và bản sao giữ nguyên; lượt thử tiếp theo dùng số lần mới khôi phục được, giữ bản sao lượt thất bại.

Đường CLI đã đọc đặt phân tích CSV trong preflight, trước nhánh dry-run trả về và trước khi nạp model để train. Header sai / hàng giữa sai schema được chuyển thành `LoiTiepTuc`; runner bắt lỗi và trả 2. Log nhóm C chứng minh cả dry-run lẫn resume thật từ chối hai ca này, thư mục lượt không đổi. Phép kiểm độc lập xác nhận hàm phân tích chỉ đọc và từ chối đúng các fixture header, hàng giữa, file rỗng; không gán phạm vi đó thành một lần tự chạy CLI.

**Vì sao script cũ vẫn còn 1:** ca batch gọi thẳng writer trên file hỏng, không qua bước chuẩn bị. Writer không thuộc phần được sửa. Đối chứng 06b và 69 kiểm độc lập đi đúng bước chuẩn bị nên không còn nối dính. Hàng batch tối giản ba trường của script cũ cũng không hợp lệ theo schema thật; đối chứng ghi rõ việc từ chối hàng lịch sử ấy, rồi kiểm thêm hàng đủ schema. Không dùng số 1 này để kết luận bản sửa resume thất bại, cũng không dùng số 0 của đối chứng để tuyên bố mọi đường ghi đã được bảo vệ.

## 4. R41-N1 — đính chính mô tả, không chặn nghiệm thu

Phụ lục vòng 41 mục 3, kế hoạch kiểm mục 2 và docstring `sua_duoi_csv()` nói **“đọc lại” sau khi ghi**. Mã hiện tại:

1. Phân tích và so hàng của biến `moi` trong bộ nhớ.
2. Ghi file tạm, flush + fsync, đọc/băm **file tạm**.
3. `os.replace` vào file đích.
4. Ghi biên nhận từ nội dung đã tính.

Không có bước đọc lại CSV **đích** sau `os.replace` trong hàm này. Điều đó khác với hàm sửa nhật ký có đọc lại file đích. Các test có đọc output để kiểm, nhưng không biến phép kiểm test thành hành vi runtime.

**Cách xử lý đủ cho lần cập nhật bàn giao kế tiếp:** đổi mô tả thành “kiểm nội dung mới và hash file tạm trước khi thay file đích; test đọc lại output xác nhận”. Không cần thêm mã đọc lại, mở test mới hay hoãn chốt vòng 41 chỉ vì câu chữ này. Đây không phải bằng chứng file đích đang bị ghi sai.

## 5. Giới hạn và lịch sử thất bại

- Không phân biệt được một trường cuối bị mất chữ số nhưng vẫn là số hợp lệ với hàng hoàn chỉnh thiếu newline. Chấp nhận giới hạn đã công khai; không tuyên bố phát hiện mọi ghi dở.
- Hàng bị mất hoàn toàn không được khôi phục; vòng này không dùng `train_results` trong checkpoint để tái dựng CSV.
- Không có giao dịch nguyên tử chung cho cả nhật ký và hai CSV. Có thể một file đã sửa trước khi thay file sau thất bại; các bản sao và lượt thử tiếp theo đã được kiểm, không đồng nghĩa khôi phục sau mọi sự cố nguồn điện.
- Giả lập lỗi thay file và kill tiến trình không phải thử mất điện thật. Không thêm khóa chống chạy đồng thời; writer gọi trực tiếp vẫn chưa tự kiểm đuôi.
- Lần suite đầu chưa hoàn tất: đếm được **107 PASS, 1 FAIL**, không có dòng tổng kết. Log nguồn Windows được bàn giao ghi Critical Battery / Sleep và khoảng ngủ xấp xỉ 16:54–17:53; phù hợp giải thích test bị gián đoạn. Reviewer đọc bản log này, không truy vấn trực tiếp Windows. Lần 2 có đủ **138 PASS, 0 FAIL, exit 0**; không bỏ qua lịch sử lần đầu.
- Giới hạn resume trước đây giữ nguyên: không bit-reproducible, chưa thử Ctrl+C thật trong training thật, rủi ro RAM thấp từng quan sát trên GPU smoke. Vòng này không đo lại các mục đó.

## 6. Bước tiếp theo đúng phạm vi đã duyệt

Có thể ghi nhận review này, lưu gói `6c8e37a` cùng bằng chứng và chuyển base review sang **`6c8e37a`** khi làm bàn giao kế tiếp. Đính chính R41-N1 bằng câu chữ.

Guard ở hàm append và khóa thư mục lượt là **đề xuất riêng, chưa được triển khai hoặc mặc nhiên phê duyệt**. Review này không bắt buộc thêm chúng để đóng R40-N1/N2.

Training dài vẫn cần người dùng chốt số epoch, ngân sách và thời điểm bắt đầu. Các kịch bản 15 epoch / 56,64 giờ / 73,95 giờ không tự trở thành lệnh chạy hay thời gian bảo đảm. Không thay split, dataset, bước frame, batch/workers hoặc môi trường Nano theo review này.

## 7. Bằng chứng đi kèm

`CODEX_REVIEW_6c8e37a_EVIDENCE.zip` gồm các script reviewer, JSON kết quả độc lập, đối chiếu log, nguồn liên quan ở base/target, log vòng 41 cần thiết, README lệnh tái lập và `SHA256SUMS.txt`. Không nhúng lặp toàn bộ gói bằng chứng các vòng cũ, dataset, checkpoint hay ZIP delta.

Lệnh tái lập từ thư mục giải nén bằng chứng:

```text
python check_csv_recovery.py --repo tree --out csv_recovery_rerun.json
python check_journal_recovery.py --repo tree --out journal_recovery_rerun.json
python observe_csv_risks.py --repo base_preimages --out observer_base_rerun.json
python observe_csv_risks.py --repo tree --out observer_target_rerun.json
python tree/handoff/logs/review41/06b_quan_sat_qua_buoc_chuan_bi.py --out counter_rerun.json
```

Các lệnh trên chỉ dùng fixture ngắn, không train. Nếu kiểm lại delta bằng `check_package.py`, dùng đúng ZIP có SHA256 ghi ở mục 1; phạm vi vẫn là các đường dẫn thay đổi.
