# R66 — kiểm kê các tập dữ liệu mới tải (02/11/2026, chỉ đọc, chưa giải nén)

**Yêu cầu người dùng (02/11):** "tôi mới tải thêm mấy bộ dataset này về, hãy chạy test tiếp", và "bảo chat gpt tìm cách
đi" (tìm cách cải thiện PF; xem câu trả lời về PF ngày 02/11). Các file nằm tại `D:\LABOTARY\`.

**Bằng chứng:**
- sha256 từng zip: `sha256_zip.txt`.
- Kết quả kiểm kê: `kiem_ke_zip.json`, sinh bằng `kiem_ke_zip.py`. Script dùng `zipfile` để đếm file theo đuôi, liệt kê
  thư mục, đọc README/text nhỏ, rồi `testzip` toàn bộ để kiểm CRC.

| Tập (file) | Trạng thái | Nội dung chính |
|:--|:--|:--|
| **Sherbrooke** (`sherbrooke_frames.zip` 234 MB, `sherbrooke_annotations.zip`) | CRC OK | Bộ UrbanTracker (Jodoin và cs., WACV 2014), giao thông hỗn hợp đô thị.<br>Ảnh: 4.000 JPG `00000001…00004000`.<br>GT định dạng Polytrack (`.sqlite`): xe, người đi bộ, tất cả, trong vùng mask; chỉ gán nhãn **frame 2754–3754**.<br>README không ghi FPS hay giấy phép. Công cụ gốc dùng ngưỡng ABSDIST 90. |
| **TrafficMOT** (`TrafficMOT-…zip` 0,97 GB) | CRC OK | Giao thông Ấn Độ, 1280×720, 10 lớp: Motor_Bike, Bus, LMV, Auto, Bike, Pedestrian, LCV, E-rickshaw, Tractor, Truck.<br>Gói này có 27 video gán nhãn đủ (`Fully_annotate`) và 27 video chỉ gán frame đầu.<br>Mỗi video có **30 frame lấy cách 1** (frame 0, 2, …, 58), CSV có `trackID`. Tập test 51 video **không có** trong gói.<br>**Giấy phép CC BY-NC-ND 4.0** (ghi trong README). FPS gốc không ghi. |
| **VisDrone2019-MOT val** (1,6 GB) | CRC OK | 7 chuỗi drone, 2.846 ảnh. Nhãn MOT: frame, id, x, y, w, h, cờ score, category, truncation, occlusion. Lớp 0 là vùng bỏ qua. |
| **VisDrone2019-MOT test-dev** (2,3 GB) | CRC OK | 17 chuỗi drone, 6.635 ảnh, cùng định dạng nhãn. |
| **HELMET** (`4pwj8-osfstorage-archive.zip`, OSF 4pwj8) | **HỎNG — tải thiếu** | File 3.328.049.152 byte, không có end-of-central-directory (EOCD/EOCD64). Entry đầu là `HELMET_DATASET/annotation.zip`. **Cần tải lại**; không dùng được cho tới khi có bản đủ. |

**Ghi chú cho giao thức (để GPT quyết):**
- Chưa giải nén. Theo `handoff/LUU_TRU.md` mục 5, ảnh giải nén (sinh lại được) có thể đặt trên ổ rời F:. Cache
  detection và kết quả để trên D: (còn khoảng 47 GB trống).
- VisDrone là ảnh drone, giống UTUAV: pilot UTUAV cho thấy detector hiện tại khó chuyển miền.
- TrafficMOT chỉ có 30 frame lấy cách 1 cho mỗi video; khoảng thời gian thật giữa hai frame phụ thuộc FPS gốc, hiện
  chưa biết.
- Sherbrooke cần chuyển nhãn từ SQLite Polytrack sang MOT, và xử lý vùng mask.

## Nguồn FPS / giấy phép (tra cứu 02/11 14:3x, agent web-fetch chỉ đọc; trích nguyên văn)

| Tập | FPS gốc | Nguồn | Giấy phép | Ghi chú giao thức |
|:--|:--|:--|:--|:--|
| **Sherbrooke** | **30 fps**, 800×600 | https://www.jpjodoin.com/urbantracker/dataset.html: "The video resolution is 800x600. For the evaluation, a 1001 frames (30 fps) part of the video was chosen with 15 cars and 5 pedestrians." | Chưa đọc được: thông báo bản quyền là popup javascript. Trang yêu cầu trích dẫn bài WACV 2014. | 1001 frame khớp README GT (2754–3754). Trang có file "Annotated zone mask" nhưng không giải thích ý nghĩa. |
| **HELMET** | **10 fps**, 100 frame/clip, 1920×1080 | `HELMET_DATASET/Readme.txt` trong bản phát hành OSF (sha256 khớp API OSF): "each video clip has 100 frames (10 second x 10 fps) with 1920 x 1080 resolution" | **CC BY 4.0** (`node_license` trong API OSF) | Có `track_id` cho xe máy, `frame_id` 1–100, nhãn mũ bảo hiểm; có `data_split.csv` chia train/validation/test. Trích dẫn OSF: Lin, Deng, Albers, Siebert, IEEE Access 2020. |
| **VisDrone2019-MOT** | **Chưa tìm thấy** | README dataset, README toolkit MOT, arXiv 1804.07437 | Không có giấy phép; toolkit ghi "for research purpose only" | Toolkit: "we only consider five object categories in multi-object tracking, i.e., car, bus, truck, pedestrian, and van"; GT score 0 = bỏ qua; lớp 0 = vùng ignore (`dropObjects`). |
| **TrafficMOT** | **Chưa tìm thấy** | arXiv 2311.18839 (HTML) | CC BY-NC-ND 4.0 (README gói) | Bài báo: 78 video gán nhãn đủ (51 test), mỗi video 30 frame; metric MOTA, IDF1. Không nói frame 0, 2, …, 58 có phải lấy cách 1 hay không. |

HELMET: bản "tải cả project" của OSF bị cắt hai lần (`4pwj8-osfstorage-archive.zip` trên D: hỏng, 1,8 GB). Đang tải
**từng file** qua API OSF sang `F:\LABOTARY_data\goc_tai_ve\HELMET\`, kiểm sha256 với OSF (`tai_helmet.sh`,
`osf_helmet_*_listing.json`, log `01_tai_helmet.log`): 7 gói ảnh `part_1…7.zip` tổng 28,9 GB, cùng annotation, split,
Readme và F-measure.
