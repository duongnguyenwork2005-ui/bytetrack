# R48C — Truy nguyên "EKF tại `eaf2e87` cho file tracking khác run mốc 2"

Ngày 2026-10-11, HEAD `eaf2e87`, nhánh `phase2-mixed-traffic`. Việc chỉ đọc: không checkout, không chạy lại kết quả cũ,
không sửa file nào của run mốc 2. Nhãn: [QS] quan sát có file / lệnh dẫn, [GT] giả thuyết chưa kiểm.

## 1. Kết luận

**Hai file tracking không khác nhau.** Nhận định ở `05_dung_dan_pf_va_smoke.md` mục 6 (bản trước) là **sai**, nay đính
chính. File tracking EKF của smoke và của run mốc 2 **trùng từng byte** trên cả 2 video. Hai giá trị sha256[:16] khác nhau
trong `eval_meta.json` đến từ hai **quy ước hash** khác nhau của `src/run_fingerprint.sha256_file`, không phải từ nội
dung tracking:

- Run mốc 2 (`1c7ce8c`) hash **byte thô**.
- Từ commit `935a445` (2026-09-20), file `.txt` / `.ini` / `.csv` được hash sau khi **chuẩn hóa CRLF → LF**.

Không có frame hay track nào lệch, nên không có "frame / track đầu tiên lệch" và không có commit tracking nào gây lệch
trên ca này.

## 2. Lệnh và đầu ra

| Lệnh | Exit | Đầu ra |
|:--|:--|:--|
| `python src/kc48/so_sanh_ekf_moc2.py` | 0 | `results/phase2/kc48_ekf_tai_lap/so_sanh_dong.json` (thời điểm `2026-10-11T15:10:30+0700`), `frame_MVI_20052.csv`, `frame_MVI_20062.csv` |
| `git show 1c7ce8c:src/run_fingerprint.py` | 0 | quy ước hash thời mốc 2 (mục 4) |
| `git show 935a445 -U2 -- src/run_fingerprint.py` | 0 | diff đổi quy ước (mục 4) |
| `git log --format=... 1c7ce8c..eaf2e87 -- <mã tracking>` | 0 | 12 commit (mục 5) |

Script đọc:
- `data/processed/trackers/DETRAC-all/p2-expB_detrac_dev_fix1-A-bt-ekf/{data/*.txt, data_cls/*.csv}` (mốc 2);
- `results/phase2/kc48_pf_smoke/_processed/trackers/DETRAC-all/p2-kc48_pf_smoke-A-ekf/{data/*.txt, data_cls/*.csv}`
  (smoke);
- hai `eval_meta.json` tương ứng trong `results/trackeval/DETRAC-all/`.

Script so ở 3 cấp:
1. **Byte**: sha256 thô và sha256 sau chuẩn hóa LF.
2. **Dòng theo vị trí**, kèm so dạng multiset.
3. **Theo frame**: khóa (frame, id), độ lệch từng cột, frame / track đầu tiên lệch.

Sau đó script đối chiếu từng `per_video` đã ghi trong `eval_meta.json` với hai quy ước hash.

## 3. Kết quả [QS]

Nguồn: `so_sanh_dong.json`.

| Video | Dòng mốc 2 / smoke | Dòng trùng theo vị trí | Frame lệch | Khóa (frame, id) khác số | Khóa chỉ một bên | sha256 thô (cả hai file) | sha256 sau LF (cả hai file) |
|:--|--:|--:|--:|--:|--:|:--|:--|
| `MVI_20052` | 8.106 / 8.106 | 8.106 | 0 / 694 | 0 / 8.106 | 0 / 0 | `42befd96aa984031…` | `4bb3dda8edc3b6fb…` |
| `MVI_20062` | 5.990 / 5.990 | 5.990 | 0 / 800 | 0 / 5.990 | 0 / 0 | `2d4a45b822812e47…` | `cf97601fb5d5d4f3…` |

- Độ lệch lớn nhất trên x, y, w, h, score đều bằng 0.
- `byte_trung_khit: true` ở cả 2 video. Hai file cùng số byte (801.218; 594.378) và đều dùng CRLF ở mọi dòng (8.106;
  5.990 dòng).
- `data_cls/*.csv` cũng trùng byte: `5d728067…`, `9fad8bfc…`.

Đối chiếu `per_video` trong `eval_meta.json`, trường `eval_meta_quy_uoc_hash`:

| Run | `git_commit` / `evaluated_at` trong eval_meta | `MVI_20052` đã ghi | khớp | `MVI_20062` đã ghi | khớp |
|:--|:--|:--|:--|:--|:--|
| mốc 2 `p2-expB_detrac_dev_fix1-A-bt-ekf` | `1c7ce8c` / 2026-09-16T12:34:19 | `42befd96aa984031` | byte thô của **cả hai** file | `2d4a45b822812e47` | byte thô của cả hai file |
| smoke `p2-kc48_pf_smoke-A-ekf` | `eaf2e87` / 2026-10-11T06:53:10 | `4bb3dda8edc3b6fb` | sau LF của **cả hai** file | `cf97601fb5d5d4f3` | sau LF của cả hai file |

## 4. Commit đổi quy ước hash, bằng chứng diff [QS]

Tại `1c7ce8c` (`git show 1c7ce8c:src/run_fingerprint.py`):

```text
Luu y: sha256 file .txt/.npz duoc tinh tren BYTE nguyen ban (khong chuan hoa xuong dong) ...
def sha256_file(p: Path, missing: str = "MISSING") -> str:
    if not p.exists():
        return missing
    h = hashlib.sha256()
    with p.open("rb") as f: ...
```

`935a445` "P1: dua noi dung ground truth vao dau van tay danh gia" (2026-09-20), trích diff
(`git show 935a445 -U2 -- src/run_fingerprint.py`):

```text
-Luu y: sha256 file .txt/.npz duoc tinh tren BYTE nguyen ban (khong chuan hoa xuong dong) ...
+QUY UOC HASH (mot quy tac duy nhat, cong bo o day va o `HASH_CONVENTION`):
+    - file VAN BAN (.txt / .ini / .csv): sha256 cua noi dung da chuan hoa xuong dong ve LF.
+TEXT_SUFFIXES = (".txt", ".ini", ".csv")
     if not p.exists():
         return missing
+    if p.suffix.lower() in TEXT_SUFFIXES:
+        return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
```

`tracking_fingerprint()` gọi chính `sha256_file` cho `data/<video>.txt`. Vì file tracking trên Windows dùng CRLF, mọi
`eval_meta.json` tạo trước `935a445` và sau `935a445` đều cho `tracking.per_video` / `tracking.sha256` khác nhau, dù
nội dung tracking trùng byte. HEAD vẫn dùng quy ước này (`src/run_fingerprint.py:71-85`).

## 5. Các commit mã tracking giữa hai run

`git log 1c7ce8c..eaf2e87` trên `src/{ekf_ctrv,tracker_ctrv,run_tracking,config,det_cache,time_units,ul_backend,
bytetrack_np,run_fingerprint}.py` và `configs/bytetrack_ekf_ctrv.yaml` liệt kê 12 commit, từ `9ec7d8e` (2026-09-16)
đến `91c95c7` (2026-09-29). Các commit này gồm `11603cb` (backend ByteTrack không phụ thuộc torch), `9dcd2d6`, `5e28f37`
và `f29f1b8`.

Ở cấp đầu ra, chúng **không đổi** file tracking EKF điều kiện A trên 2 video smoke. Smoke dùng backend `ultralytics`
8.4.141, theo `eval_meta.json` của smoke. Tôi **không** đọc từng diff của các commit này: khi đầu ra đã trùng byte thì
không còn lệch nào cần quy cho một commit. Mốc 2 không ghi `backend` trong `eval_meta.json` (định dạng cũ), nên không
so được phiên bản thư viện qua meta.

## 6. Hệ quả với bảng kế thừa mốc 2 (`00_tiep_nhan_va_ke_thua.md`)

- [QS] Ở `eaf2e87`, EKF ByteTrack điều kiện A (`configs/bytetrack_ekf_ctrv.yaml`, cache `yolov8n-coco_conf0.05_all`)
  tái tạo **trùng byte** file tracking của run mốc 2 trên 2/12 video dev (`MVI_20052`, `MVI_20062`). Bằng chứng này
  ủng hộ việc kế thừa số EKF của mốc 2 cho cùng cấu hình, ít nhất trên 2 video đó.
- **Phạm vi chưa kiểm**: 10 video dev còn lại; các bộ lọc khác (CV, UKF, PF; mốc 2 dev không có run PF trong
  `data/processed/trackers/DETRAC-all/p2-expB_detrac_dev_fix1-A-*`); điều kiện khác A; số liệu TrackEval của mốc 2
  (smoke chấm lại trên bản sao GT, và giao thức 04 mục 10 không cho dùng số smoke làm kết luận). Không suy rộng ra
  các ô đó.
- Bỏ giả thuyết [GT] ở `05` mục 6 cũ, "có thể là thay đổi mã giữa hai commit, hoặc phiên bản thư viện", vì sai với dữ
  liệu này. Quy tắc "mọi so sánh R48D chỉ dùng run tạo tại cùng commit" vẫn giữ, vì đó là quy định của giao thức. Lý do
  giữ không còn là "EKF đã lệch".
- **Quy tắc đối chiếu từ nay**: so dấu vân tay tracking qua mốc `935a445` phải tính lại cả hai bên theo **cùng** quy
  ước (byte thô, hoặc sau LF) trên chính file `data/*.txt`. Không so trực tiếp `tracking.per_video` / `tracking.sha256`
  của hai `eval_meta.json` khác đời.

## 7. Bài học

Một chênh lệch lấy từ dấu vân tay đã ghi chưa phải là chênh lệch nội dung. Trước khi kết luận "kết quả khác", phải so
chính file nguồn ở cấp byte / dòng và kiểm quy ước hash của từng đời mã.
