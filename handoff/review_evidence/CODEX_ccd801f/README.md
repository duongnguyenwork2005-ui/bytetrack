# Bằng chứng review ccd801f

Kết luận và giới hạn: CODEX_REVIEW_ccd801f.md.
Không chứa dataset; không cần SSH Nano.

## Chạy lại test nhập nhãn

    python import_tool/test_nhap_nhan_train.py

Test chỉ dùng thư mục tạm và dữ liệu giả; không chạm data/ của dự án.

## Chạy lại kiểm gói và manifest

Đặt delta gốc 805.621 byte (SHA256 trong báo cáo) thành review_delta_latest.zip cạnh README này, giải nén vào package/. Không dùng một delta mới hơn cùng tên.

    python check_handoff.py --package package --zip review_delta_latest.zip --base-tree base_preimages --old-manifest package/files/handoff/review_evidence/CODEX_4e71d32/tree/handoff/logs/review32/09_manifest_nhan_ung_vien.tsv --old-native-hashes package/files/handoff/review_evidence/CODEX_4e71d32/evidence/native_hashes.json --out rerun_evidence

Cần Python 3 và Git. Script tạo repository tạm để áp patch; không sửa Git config của dự án.
base_preimages/ chỉ có 5 file đúng hash base, không phải checkout đầy đủ.

evidence/ là output reviewer đã chạy. Dữ liệu nhập vật lý ở ổ D: không nằm trong gói, nên không tuyên bố tự chạy kiem-nguon trên Windows hoặc tự xác nhận toàn cây 3.780 file.

SHA256SUMS.txt dùng hash byte thô, bao gồm cả báo cáo và script.
