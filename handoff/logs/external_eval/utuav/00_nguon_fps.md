# UTUAV — xác minh FPS nguồn (Q3, 02/11/2026): CHƯA XÁC MINH ĐƯỢC ⇒ giữ pilot

Quyết định áp dụng: GPT 02/11 `D:/LABOTARY/automation/runs_local/hoi_codex_R64_viec_tiep_20261102/answer.json` Q3
("tối đa 30 phút tìm bằng chứng FPS…; nếu chưa xác minh được: giữ UTUAV ở pilot, ghi FPS chưa biết; không tạo benchmark
chính thức với FPS giả định và không chạy ma trận dài; có thể chuẩn bị nội dung hỏi tác giả để người dùng tự gửi").

Thời điểm tìm: 02/11/2026 khoảng 06:50–07:05 (+07), do agent web-fetch thực hiện. Agent chỉ đọc trang, không tải
archive.

## Nguồn đã kiểm

| Nguồn | Kết quả | Nội dung liên quan (trích nguyên văn) |
|:--|:--|:--|
| https://videodatasets.org/UTUAV/ (v2, "Last updated: August 2026") | HTTP 200 | B: "6,500 labelled images with a resolution of 3840x2160 (4k) pixels"; C: "a sequence of 10,560 frames, of which 6,600 are annotated, with a resolution of 3840x2160 (4k) pixels"; Sequential_Gap300 dùng "a gap of 300 (dropped) images". **Không ghi FPS/stride.** |
| https://videodatasets.org/UTUAV/index_v1.html (v1, lịch sử) | HTTP 200 | Cùng số khung và độ phân giải; "the numbers are indicative only". **Không ghi FPS.** |
| Bài Drones 2025, doi 10.3390/drones10010015 → https://www.mdpi.com/2504-446X/10/1/15 | **HTTP 403** (không đọc được toàn văn; các URL /htm, /pdf cùng máy chủ) | — |
| Semantic Scholar API (DOI trên) | HTTP 200, chỉ có abstract | "The images are recorded from a semi-static position at two different altitudes (100 and 120 m)…". **Không ghi FPS.** |
| Europe PMC (DOI) | HTTP 200, 0 kết quả | — |
| arXiv (`all:UTUAV`) | HTTP 200, 0 kết quả | — |
| https://videodatasets.org/UTUAV/Instructions.odt | **HTTP 404** | — |

Chưa kiểm: ResearchGate và bản PDF từ Google Scholar (không có URL xác nhận).

## Kết luận

- **FPS nguồn và stride lấy mẫu của B/C: CHƯA BIẾT.** Không có nguồn chính thức nào đọc được ghi rõ. Trang v2 còn cho
  biết dữ liệu hiện tại khác số liệu trong bài Drones 2025, nên kể cả khi đọc được bài báo cũng cần đối chiếu đúng phiên
  bản dữ liệu.
- Không chấp nhận các nguồn thay thế: tên ảnh, mtime, seqinfo pilot tự tạo, giả định 30 fps.
- Theo Q3: **UTUAV giữ ở pilot** (`UTUAV.md`, `../external_mot_prep/02_pilot_utuav.md`). Không viết adapter, không chạy
  smoke hay ma trận `UTUAVSYS-validation`. Trong luận văn ghi "FPS nguồn chưa xác minh; pilot dùng giả định 30 fps cho
  detection tổng hợp từ GT".
- Mở lại khi người dùng có được bằng chứng: toàn văn bài báo có câu về FPS/stride đúng phiên bản dữ liệu, hoặc thư xác
  nhận của tác giả.

## Thư hỏi tác giả (bản nháp — người dùng tự gửi, Claude không gửi)

Người nhận: Jorge Espinosa và Sergio A. Velastin (địa chỉ liên hệ trên https://videodatasets.org/UTUAV/).

> Subject: UTUAV dataset – frame rate and sampling of sequences B and C
>
> Dear Dr. Espinosa and Prof. Velastin,
>
> I am a student using the UTUAV dataset (version 2, Sequential_Gap300 partitions) to evaluate multi-object tracking
> with motion models. To set the time step of the tracker correctly, I need to know:
>
> 1. the frame rate (fps) of the original videos for sequences B and C;
> 2. whether the released images are every consecutive frame of those videos, or were sub-sampled (and if so, the
>    stride), in particular for the validation ranges B 26341–27078 and C 20271–21021.
>
> If this is stated in your 2025 Drones paper, I would be grateful for the section reference.
>
> Thank you very much for making the dataset available.
>
> Best regards,
> [Họ tên, trường, email]
