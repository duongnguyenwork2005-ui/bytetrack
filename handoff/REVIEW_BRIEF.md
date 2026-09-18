# REVIEW_BRIEF — review vòng 4, kiểm kê BDD100K, backend không phụ thuộc torch

<!-- BASE_TARGET -->

Base giữ nguyên `a8ecaf6`. Chi tiết: `handoff/MOC2_FINAL_SUMMARY.md`, `handoff/BDD100K_INVENTORY.md`,
`scripts/nano/README.md`, `third_party/ultralytics_trackers/PROVENANCE.json`.

## 1. Sửa tài liệu — 6 điểm, đã đối chiếu source trước khi sửa

| Điểm | Bằng chứng | Đã sửa |
|:--|:--|:--|
| `--videos` | `run_experiment.py` **không có** cờ đó | YAML smoke riêng `bdd_smoke_val.yaml` (tên riêng ở **cả 4 chỗ**) |
| tên manifest | script tạo `bdd100k_dev.json` | sửa ở INVENTORY + `bdd_tuned_val.yaml` |
| timer | `run_tracking.py:177–183`: `update` = **tổng**; `assoc = update − predict` **gồm cả** filter-update + vòng đời | mô tả lại; **bỏ** yêu cầu sai "assoc CV/EKF phải bằng nhau" |
| làm tròn 2 lần | thô −0.01324999 ⇒ **−0.0132**; `ekf-maha − ekf` −0.00344993 ⇒ **−0.0034** | định dạng **một lần** từ float đầy đủ |
| SD seed PF | thật: HOTA **0,00054**, AssA **0,00090** | thay "≤ 0,0004" (sai) |
| tỷ trọng detector | thật **73,6–77,9 %**, tracker **4,0–10,2 %** | thay "~70–75 %" / "~4–11 %" (sai) |
| vượt bằng chứng | — | CI chứa 0 = **không đủ bằng chứng về khác biệt**, không phải tương đương; ba bộ lọc cùng kém **không loại trừ** motion model (chung họ giả định + chung khung association); PF: số hạt là **một phần** |
| `occlusion_ratio=0.5` | — | **quy ước cho cờ nhị phân**, không phải 50 % đo được |

Giao thức Nano: **lưu mọi lượt kể cả throttling**, số chính thức là chạy liên tục; bảng "không throttling"
là phân tích bổ sung với tiêu chí loại công bố trước; cân bằng **AB và BA**.

## 2. BDD100K — có ảnh, **chưa có nhãn**

Đã giải nén đúng vị trí: `data/raw/bdd100k/images/track/val/` = **200 video / 39.973 jpg**, không lồng thư
mục. **Không có** `labels/box_track_20/` (0 JSON) và **không có** split `train`. Adapter lấy danh sách
video/frame/`timestamp` **từ nhãn JSON** ⇒ chặn cứng: `check_bdd100k` exit 1, `det_cache` và evaluator đều
không chạy. **Smoke BDD chưa chạy**, không dùng fixture thay thế. Còn thiếu **MOT 2020 Labels** (~50 MB).
Chỉ có `val` ⇒ chưa được tuning trên BDD.

## 3. Backend không phụ thuộc torch (CV + EKF)

5 module tracker của ultralytics **không import torch dòng nào** ⇒ **vendor** (AGPL-3.0, sha256 upstream +
vendor trong `PROVENANCE.json`), chỉ đổi **5 dòng import** sang shim NumPy — thuật toán ghép **giữ nguyên
100 %**. `ul_backend.py` chọn nguồn lớp nền ⇒ **một** bản mã CTRV/CV cho cả hai backend. Dependencies:
`numpy`, `scipy`, `lap`. Backend nằm trong **fingerprint** ⇒ đổi backend thì runner **phải chạy lại**.

**Parity, tolerance công bố trước = bit-identical**, 3 sequence dev chọn trước: CV **17.348 hàng** và EKF
**17.333 hàng** bit-identical; song ánh ID **đồng nhất** (469 track); chuỗi kiểm có 177 frame không track và
116 track tái xuất hiện sau gap; chạy trong tiến trình **chặn** `torch` và `ultralytics`.

## Test thực tế

`run_all_tests.py --out-dir handoff/logs/backend_np` → **23 PASS, 0 FAIL, 0 SKIPPED, 0 NOT_RUN** (exit 0),
gồm `test_bytetrack_np_parity` 8/8 và `test_run_experiment_guard` 9/9.

## Chưa xác minh

**Nhãn BDD100K** (chặn smoke). **Nano chưa kết nối** — parity trên PC **không** chứng minh tương thích hay
tốc độ Nano; bản vendor cần **Python ≥ 3.7** còn JetPack 4.6 có **3.6**; `scipy`/`lap` aarch64 chưa thử;
đường detector trên Nano chưa nối vào backend mới.

## Cần bạn kiểm kỹ

1. Bản vendor có thật sự chỉ khác upstream ở 5 dòng import không; phép so có tự phát hiện nếu upstream đổi.
2. Còn đường nào **không** qua `ul_backend` mà vẫn kéo ultralytics vào backend `np` không.
3. Hợp đồng `Detections` đã đủ so với `Boxes` mà `BYTETracker` thực dùng chưa.
