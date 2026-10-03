# Review vòng 40 — `421dd53`

**Kết luận: PASS WITH LIMITATIONS cho bản sửa R39-F1. Đóng R39-F1.**

Bước chuẩn bị ghi của resume đã sao lưu và xử lý đúng đuôi nhật ký trước khi nối sự kiện mới. Kiểm độc lập của reviewer đạt **54/54 điều kiện**; đối chứng bốn ca của tác giả chạy lại đạt `bug_count=0`. Không thấy finding mới chặn nghiệm thu phạm vi sửa nhật ký.

**Phạm vi kết luận:** đóng lỗi R39-F1, không xác nhận toàn bộ cơ chế phục hồi chịu được mọi dạng mất điện và không phải phê duyệt training dài. Hai lỗi CSV do tác giả công khai đã được reviewer tái hiện, được ghi riêng ở mục 5 để người dùng quyết định phạm vi tiếp theo.

## 1. Danh tính gói và tính liên tục

| Mục | Kết quả |
|:--|:--|
| Ngày review | 2026-10-03 |
| Base | `b83dc1295cad821b55c60ae15855331b870fa98f` |
| Target | `421dd53ecbe171ff11219a7efba900cdbbe5a34e` |
| ZIP | `review_delta_latest.zip`, **2.177.781 byte** |
| SHA256 | `806b1e013c499118778f774c29c91d80aab4e76975408f2892816275979cf3a0` |
| Thành viên ZIP | 254, `testzip` đạt |
| Payload / alias trùng nội dung | 248 / 84, đủ số byte và SHA256 |
| Reverse/forward patch | **332/332 đường dẫn thay đổi** khớp target |
| Preimage | **8/8** khớp snapshot base đã kiểm ở vòng trước |
| Bằng chứng Codex vòng 39 được chép vào gói | **97/97** khớp `SHA256SUMS.txt` |
| Log vòng 39 | **61/61 file** nguyên byte so với gói đã review |

`chay_giam_sat.py`, `test_chay_giam_sat.py`, `run_all_tests.py` không đổi byte so với `1454b3d`. Không chạy lại các test monitor vì vòng này không đổi đường đó.

Không thấy dataset, weights, video, NPZ hay đường dẫn riêng bị cấm trong các payload đã kiểm. Có đủ khai báo 13 NPZ không bàn giao. Tôi không có checkout Git đầy đủ độc lập; **332 đường dẫn không được diễn giải thành kiểm toàn cây repo**. Phép kiểm clone toàn cây do tác giả báo cáo chưa được reviewer lặp lại.

## 2. Vì sao đóng R39-F1

Mã liên quan: `src/tiep_tuc_huan_luyen.py` — `phan_tich_duoi_nhat_ky`, `sua_duoi_nhat_ky`, `sao_luu_truoc_tiep_tuc`; `src/train_detector.py` — nhánh dry-run và nhánh chuẩn bị resume.

Đường chạy đã được đối chiếu trong mã:

1. Preflight đọc và tính kế hoạch xử lý, có hash nhật ký. Dry-run in kế hoạch rồi trả về trước bước sao lưu/ghi.
2. Resume thật gọi bước sao lưu trước khi nạp và gọi train từ checkpoint tiếp tục.
3. Nhật ký phải khớp hash của kế hoạch; bản gốc được sao lưu nguyên byte vào thư mục của lần tiếp tục.
4. Đuôi JSON ghi dở bị cắt từ đầu dòng hỏng; JSON cuối hợp lệ nhưng thiếu newline được giữ lại và thêm newline. Các byte hợp lệ trước đó giữ nguyên.
5. Ghi qua file tạm, `fsync`, `os.replace`; đọc lại để kiểm sự kiện, hash và newline. Biên nhận được lưu riêng và đưa vào provenance của lượt resume.
6. Callback `bat_dau`, các sự kiện epoch và `ket_thuc` của lần resume được ghi sau bước chuẩn bị này. Lỗi ở giữa nhật ký vẫn bị từ chối.

Sửa tại đây phù hợp yêu cầu vòng trước. **Không bắt buộc sửa trực tiếp hai hàm cấp thấp `doc_nhat_ky`/`ghi_nhat_ky`.**

## 3. Kiểm độc lập reviewer đã chạy

### 3.1 Chuẩn bị ghi, bảo toàn và đọc lại — 54/54

`check_journal_recovery.py` chạy các hàm thật trên thư mục tạm, gồm toàn bộ `sao_luu_truoc_tiep_tuc`, thao tác sao lưu/sửa file và append/read. Có bảy dạng đầu vào:

- Nhật ký sạch.
- JSON cuối ghi dở không newline.
- JSON cuối ghi dở có newline.
- JSON cuối hợp lệ thiếu newline, có chữ tiếng Việt.
- Đuôi byte NUL.
- JSON ghi dở kèm các dòng trắng phía sau.
- Lịch sử CRLF và JSON cuối hợp lệ thiếu newline.

Ở mỗi ca đều kiểm: phân tích không ghi; byte sau sửa đúng; sự kiện cũ còn đủ; bản sao nguyên byte; biên nhận và hash đúng; bằng chứng cũ được giữ; append/read thành công; lần chuẩn bị và append thứ hai vẫn đọc lại được, bản sao lần đầu còn nguyên.

Các ca bổ sung đều đạt:

- Dòng hỏng ở giữa bị từ chối bởi hai đường đọc/phân tích, không sửa file.
- Nhật ký thay đổi sau lúc lập kế hoạch bị từ chối trước khi sửa nhật ký hoặc ghi bản sao nhật ký.
- Giả lập `os.replace` thất bại ngay trước khi thay nhật ký: bản cũ và bản sao vẫn nguyên byte.
- Thử lại bằng số lần tiếp tục mới sau lỗi đó thành công.
- Không ghi đè thư mục sao lưu bằng cách dùng lại số lần tiếp tục cũ.

**Giới hạn của phép kiểm này:** file checkpoint chỉ chứa byte đánh dấu để kiểm sao chép, không được nạp như model. Không có torch, không train và không kiểm optimizer bằng phép này. Lỗi `os.replace` được chủ động giả lập; không mô phỏng mất điện vật lý.

### 3.2 Hai script đối chứng

| Script chạy lại trên target | Kết quả | Ý nghĩa |
|:--|:--|:--|
| `REPRO_REVIEW_1454b3d.py` nguyên bản | exit 1, `bug_count=3` | Gọi thẳng reader/writer, bỏ qua bước chuẩn bị mới; đúng dự đoán đã ghi từ trước |
| `06b_repro_qua_buoc_chuan_bi.py` | exit 0, `bug_count=0` | Cùng bốn ca, có bước chuẩn bị ghi; bản sao nguyên byte 4/4 |

Script nguyên bản còn báo ba lỗi **không được dùng để mở lại R39-F1**, vì review trước đã xác định test tích hợp là tiêu chí nếu sửa ở đường chuẩn bị riêng. Bằng chứng đóng finding là mã, phép kiểm 54 điều kiện và test tích hợp được đối chiếu ở mục 4.

## 4. Test tích hợp do tác giả chạy — đã đọc mã và đối chiếu log

Nhóm N gọi `train_detector.py` qua subprocess, CPU, YOLOv8n thật trên fixture nhỏ. Theo mã và log, mỗi biến thể được kiểm qua: dry-run → resume, ngắt sau epoch 2 → đọc/kiểm sao lưu → dry-run tiếp → resume tới hết epoch 3. Ca lỗi giữa file bị từ chối ở cả dry-run và resume thật.

| Log | PASS / FAIL / SKIP | Exit |
|:--|:--|--:|
| Resume lần 1 | 98 / 1 / 0 | 1 |
| Resume lần 2, HEAD ghi trong log `074c486` | **102 / 0 / 0** | 0 |
| Nhóm N mới trên mã cũ `1454b3d` | **12 / 18 / 0** | 1 |
| Detector 5 lớp | **92 / 0 / 0** | 0 |
| Pilot chẩn đoán | **34 / 0 / 0** | 0 |

Tôi đếm lại các dòng kết quả và đối chiếu tổng kết/exit code. Lần 1 bị lỗi ở setup U8 thiếu nhật ký; bản test cuối thêm nhật ký vào fixture, giữ phép kiểm bảo toàn CSV cũ. Log thất bại được giữ.

Trong 18 ca không đạt trên mã cũ, một phần kiểm tính năng mới như biên nhận/sao lưu. Bằng chứng nhạy với **lỗi cũ** là A và D thất bại ở đúng ba biến thể, trong khi ca sạch đạt. Không coi cả 18 lỗi là 18 bug khác nhau.

Reviewer **không lặp lại 102 test CPU, 92/92, 34/34 hoặc training GPU** vì môi trường review không có torch/Ultralytics/checkpoint. Các con số đó là log tác giả đã đối chiếu, không phải lượt training reviewer tự chạy. Các giới hạn về tính không trùng bit và Ctrl+C thật của vòng 39 vẫn giữ nguyên.

## 5. Hai rủi ro CSV đã xác nhận — phạm vi tiếp theo cần người dùng duyệt

Đây là vấn đề tác giả đã công khai và chưa sửa. Không phải hồi quy mới do xử lý nhật ký vòng 40. Chúng không ngăn việc đóng **R39-F1**, nhưng ngăn kết luận rộng rằng mọi file phục hồi sau gián đoạn đều đã an toàn.

### R40-N1 — `results.csv` có thể dừng resume hoặc giữ dòng dở

Tái hiện bằng `sao_luu_truoc_tiep_tuc()` thật, checkpoint byte đánh dấu, CSV nhỏ trong thư mục tạm:

- Đuôi bốn byte NUL: `ValueError` thô, không phải `LoiTiepTuc`; bản sao CSV vẫn nguyên byte.
- Sau các dòng epoch 1–9, đuôi chỉ có chữ số `1` — có thể là phần đầu của dòng epoch 10 — được giữ lại. Hàm lọc đọc nó thành epoch 1 và không kiểm đủ cột. CSV sau chuẩn bị vẫn có một dòng thiếu các trường còn lại.

Không có bằng chứng rằng CSV của các lượt kiểm đã bàn giao bị lỗi này. Ca trên là kiểm gián đoạn bằng fixture, không phải sự cố OOM/mất điện thực tế trên PC.

### R40-N2 — `theo_batch.csv` có thể nối dính dòng sau gián đoạn

Tôi không chỉ ghép chuỗi như ví dụ của tác giả: dùng trực tiếp `GhiLuotDayDu.het_epoch_train()` với một hàng được đặt sẵn trong `_rows`, không train.

Sau đuôi `0,1,`, writer nối hàng mới `(lan=1, epoch=2, i=0)`. Khi đọc lại, ba trường đầu thành `0,1,1` và xuất hiện hai trường dư ngoài 20 cột khai báo. Đây là mất tính đúng đắn của log batch; file này dùng chẩn đoán, không tự quyết định checkpoint resume.

**Đề xuất phạm vi sửa tiếp:** chỉ xử lý đuôi hai CSV trong bước chuẩn bị resume, có bản sao nguyên byte, kiểm schema, giữ hàng hoàn chỉnh kể cả thiếu newline và từ chối lỗi giữa file. Kiểm bằng fixture ngắn qua đường thật. Đây mới là đề xuất; **chưa được phê duyệt hoặc triển khai trong review này**.

Guard trong hàm append và khóa chạy đồng thời là đề xuất riêng. Review này không kiểm concurrency và không yêu cầu tự thêm chúng vào vòng sửa CSV.

## 6. Giới hạn và quyết định

- `os.replace` và file tạm giúp tránh sửa dở ngay trong file đích. Phép kiểm chỉ xác nhận lỗi trước replace giữ bản cũ, rồi thử lại được. Sau replace thành công, file đích là bản mới. Không diễn giải thành bảo đảm giữ bản cũ ở mọi thời điểm mất điện hoặc đã kiểm độ bền của thiết bị lưu trữ.
- Không có kiểm độc lập dataset đầy đủ, checkpoint tensor hoặc training dài trong vòng này.
- RAM available tối thiểu 17 MiB ở GPU smoke vòng 39 vẫn là giới hạn đã biết; vòng này không đo lại và không tự đổi worker/batch hay đóng ứng dụng.
- Có thể ghi nhận `421dd53` là snapshot đã review **PASS WITH LIMITATIONS**, đóng R39-F1, và dùng làm base cho vòng tiếp sau khi lưu trữ đúng gói/hash này. Việc đó không phê duyệt sửa CSV, thêm tính năng hoặc bắt đầu training dài.
- Người dùng cần quyết định có sửa hai CSV trước hay không. Số epoch, ngân sách và thời điểm training dài vẫn chờ duyệt.

## 7. Bằng chứng bàn giao

`CODEX_REVIEW_421dd53_EVIDENCE.zip` có báo cáo này, script checker, kết quả chạy reviewer, mã target liên quan, log vòng 40 và `SHA256SUMS.txt`. Không chứa dataset, weights, checkpoint thật hay môi trường cài đặt.

Sau khi giải nén, kiểm nhật ký không cần torch:

```text
python check_journal_recovery.py --repo tree --out journal_recovery_local.json
```

Để quan sát các giới hạn CSV (không sửa chúng):

```text
python observe_csv_risks.py --repo tree --out csv_risks_local.json
```

Các script chỉ dùng thư mục tạm và file JSON đầu ra. `tree/` trong gói bằng chứng không phải toàn repo; test CLI training phải dùng repo và môi trường của tác giả.
