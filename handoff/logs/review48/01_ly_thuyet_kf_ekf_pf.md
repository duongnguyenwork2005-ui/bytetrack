# R48A (M1) — Cơ sở lý thuyết KF–CV, EKF–CTRV, PF–CTRV; SIS / SIR

Ngày 2026-10-10. Mã đối chiếu tại commit `eaf2e877081938420508e696de22dbd2aa0bff97` (HEAD lúc viết). Các file bộ
lọc không sửa so với HEAD (`git status --porcelain` rỗng, ghi trong `results/phase2/kc48_pf_code/so_huu_hat.json`).
Bảng khái niệm → `file:hàm:dòng`, kiểm sở hữu trạng thái và danh sách giả thuyết: `02_doi_chieu_code_pf.md`.
Đã sửa một lần sau review Codex vòng 1 (R48A-F1…F4); danh sách chỗ sửa ở mục 10.

## 0. Quy ước

### 0.1 Nhãn khẳng định

| Nhãn | Nghĩa |
|:--|:--|
| **[LT]** | Lý thuyết: dẫn xuất toán học viết trong tài liệu này (kiểm được bằng đại số), hoặc lấy từ nguồn kèm trạng thái đối chiếu |
| **[MÃ]** | Triển khai: đọc trực tiếp mã tại commit trên, dẫn `file:dòng` |
| **[QS]** | Quan sát: số đo đã có, dẫn file kết quả |
| **[GT]** | Giả thuyết chưa kiểm |

### 0.2 Nguồn và trạng thái đối chiếu (theo `tai_lieu_nguon/README.md`)

| Ký hiệu | Nguồn | Trạng thái |
|:--|:--|:--|
| [BT] | Y. Zhang et al., "ByteTrack: Multi-object Tracking by Associating Every Detection Box", ECCV 2022, DOI 10.1007/978-3-031-20047-2_1; arXiv 2110.06864v3 | Toàn văn arXiv v3 có trong `tai_lieu_nguon/` (sha256 `772e6470…`). **Đã đối chiếu toàn văn** trang 2–7 trong R48A; chỉ những ý liệt kê dưới đây (kèm trang) là đã đối chiếu. Metadata ECCV khớp Crossref |
| [A02] | M. S. Arulampalam, S. Maskell, N. Gordon, T. Clapp, "A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking", IEEE TSP 50(2):174–188, 2002, DOI 10.1109/78.978374 | **Chỉ metadata đã xác minh (Crossref). Toàn văn không có ⇒ mọi ý ghi "[A02]" là "chưa đối chiếu toàn văn"** |

Không dẫn nguồn nào khác: agent không có mạng, nguồn ngoài README chưa xác minh được. Các phương trình ghi
**[LT, tự dẫn]** được chứng minh ngay trong tài liệu này, không dựa vào nguồn ngoài.

Ý đã đối chiếu với toàn văn [BT]:
- Tr. 2: "Similar to [6] (SORT), we adopt Kalman filter [29] to predict the location of the tracklets in the new frame".
- Tr. 4, Algorithm 1 và đoạn văn kèm theo: tách detection theo ngưỡng điểm τ thành D_high / D_low; dự đoán vị trí mới
  của mọi track bằng Kalman filter (dòng 14–16); ghép lần 1 giữa D_high và **mọi** track (kể cả T_lost) bằng IoU
  hoặc Re-ID, thuật toán Hungarian; ghép lần 2 giữa D_low và các track còn lại **chỉ bằng IoU**; detection điểm thấp
  không ghép được thì bỏ (coi là nền); track không ghép đưa vào T_lost, xóa khi mất quá một số frame ("i.e. 30").
- Tr. 5, Implementation details: τ mặc định 0,6; ghép bị từ chối khi IoU < 0,2; track mất giữ 30 frame.
- Tr. 5, Datasets: BDD100K MOT gồm 1.400 video train, 200 val, 400 test, 8 lớp, có chuyển động camera lớn.
- Tr. 5, Ablation: trên BDD100K "large camera motion and the annotations are in low frame rate, which causes failure
  of motion cues".
- Tr. 6: "Kalman filter fails in autonomous driving scenes … Thus, we do not use Kalman filter on BDD100K."
- Các trang đã đọc **không** mô tả vector trạng thái của KF. Vector 8-D XYAH và hệ số nhiễu ở mục 2 lấy từ **mã**
  ultralytics đang dùng, không phải từ bài báo. Cấu hình của repo cũng khác bài báo (ngưỡng điểm cao 0,25; ghép bằng
  `fuse_score`) — xem `02_doi_chieu_code_pf.md` mục 1.

### 0.3 Đơn vị thời gian (theo `src/time_units.py` mục A–D)

- Mọi bộ lọc dùng **đơn vị bước**: Δt = 1 bước = một lần predict = một frame được xử lý (`CTRV_DT = 1.0`,
  `src/config.py:272`; KF–CV có dt = 1,0 viết cứng, `third_party/ultralytics_trackers/utils/kalman_filter.py:45-50`).
  Không bộ lọc nào có tham số "theo giây" (`src/time_units.py:65-71`).
- Đơn vị biến trạng thái: v [px/bước], ω [rad/bước], v_a [1/bước], v_h [px/bước]. Độ lệch chuẩn nhiễu quá trình là
  độ lệch chuẩn của **gia số rời rạc** cộng vào biến ở mỗi lần predict, nên mang **đơn vị của chính biến đó**:
  σ_θ [rad], σ_ω [rad/bước] (bảng đơn vị từng thành phần Q ở mục 3). Chú thích trong mã ghi σ_θ [rad/bước], σ_ω
  [rad/bước²] (`src/config.py:304-305`, `src/time_units.py:33-34`, `src/ekf_ctrv.py:299-300`), tức đọc gia số chia
  cho Δt thành tốc độ thay đổi. Hai cách cho cùng giá trị số khi Δt = 1 bước, nhưng phần tử của Q là phương sai của
  gia số. Mã và chú thích không sửa ở R48A.
- Đổi sang giây **chỉ để báo cáo**: v·fps [px/s], ω·fps [rad/s]; hiệp phương sai đổi bằng P′ = S P Sᵀ, S = diag(s_i)
  (`src/time_units.py:74-84`, `convert_state_units`). UA-DETRAC 25 fps (`src/config.py:100`) ⇒ 1 bước = 0,04 s;
  BDD100K theo quy ước repo 5 fps ⇒ 1 bước = 0,2 s (`src/time_units.py:108-110`). Hệ quả: cùng một Q "theo bước" là
  hai giả thiết vật lý khác nhau ở 25 fps và 5 fps (`src/time_units.py:50-53`).
- Khi bỏ frame, `ElapsedStepsMixin` gọi predict k lần, k = max(1, round(Δt_thực / Δt_danh_nghĩa)), mỗi lần cộng Q một
  lần (`src/time_units.py:86-103, 163-204`).

## 1. Bài toán lọc Bayes [LT]

Mô hình không gian trạng thái Markov ẩn cho **một** track:

$$x_k = f(x_{k-1}) + w_k,\; w_k\sim\mathcal N(0,Q_k);\qquad z_k = h(x_k)+v_k,\; v_k\sim\mathcal N(0,R_k).$$

Đệ quy Bayes (x_k: trạng thái tại bước k; z_{1:k}: các quan sát đã ghép cho track đến bước k):

$$p(x_k\mid z_{1:k-1})=\int p(x_k\mid x_{k-1})\,p(x_{k-1}\mid z_{1:k-1})\,dx_{k-1},\qquad
p(x_k\mid z_{1:k})=\frac{p(z_k\mid x_k)\,p(x_k\mid z_{1:k-1})}{p(z_k\mid z_{1:k-1})}.$$

KF là nghiệm đúng khi f, h tuyến tính và nhiễu Gauss; EKF / UKF xấp xỉ hậu nghiệm bằng một Gauss; PF xấp xỉ bằng tập mẫu
có trọng số ([A02] trình bày khung này — chưa đối chiếu toàn văn; các phương trình ở đây là kết quả chuẩn, tự kiểm
được). Trong tracking nhiều đối tượng, **association** (ByteTrack) quyết định z_k nào thuộc track nào; bộ lọc mặc
nhiên coi phép ghép là đúng. Ghép sai thì mọi bộ lọc đều cập nhật bằng quan sát của đối tượng khác.

## 2. KF–CV, trạng thái XYAH 8-D (baseline ByteTrack / ultralytics) [MÃ + LT]

**Trạng thái** x = [x, y, a, h, ẋ, ẏ, ȧ, ḣ]ᵀ: (x, y) tâm bbox [px]; a = w/h [–]; h chiều cao [px]; ẋ, ẏ, ḣ [px/bước];
ȧ [1/bước]. **Quan sát** z = [x, y, a, h]ᵀ, đổi từ tlwh (`byte_tracker.py:179-185`). Ma trận đo H = [I₄ 0].

**Chuyển trạng thái** (vận tốc không đổi, tuyến tính):

$$F=\begin{bmatrix}I_4 & \Delta t\,I_4\\ 0 & I_4\end{bmatrix},\qquad x_k=F\,x_{k-1}+w_k .$$

**Nhiễu**, σ_p = 1/20, σ_v = 1/160 (`kalman_filter.py:54-55`):
- Q_k = diag((σ_p h)², (σ_p h)², 10⁻⁴, (σ_p h)², (σ_v h)², (σ_v h)², 10⁻¹⁰, (σ_v h)²), h của ước lượng **trước**
  predict (`kalman_filter.py:110-121`). Đơn vị độ lệch chuẩn gia số: x, y, h [px]; a [–]; ẋ, ẏ, ḣ [px/bước]; ȧ
  [1/bước] (σ_p không thứ nguyên, σ_v mang đơn vị 1/bước); phần tử Q có đơn vị bình phương tương ứng.
- R = diag((σ_p h)², (σ_p h)², 0,1², (σ_p h)²), h **dự đoán** (`kalman_filter.py:147-153`). Nếu truyền confidence c
  thì R ← R·max(1 − c, 0,05) (dòng 154-155), nhưng `STrack.update` / `re_activate` **không truyền** confidence
  (`byte_tracker.py:115-117, 147-149`; bản ultralytics 8.4.141 đã cài cũng vậy — `so_huu_hat.json`, khóa
  `moi_truong.STrack_goi_bo_loc`) ⇒ cơ chế này không hoạt động với mọi bộ lọc trong pipeline.
- P₀ = diag((2σ_p h)², (2σ_p h)², 10⁻⁴, (2σ_p h)², (10σ_v h)², (10σ_v h)², 10⁻¹⁰, (10σ_v h)²), x̂₀ = [z, 0, 0, 0, 0]
  (`kalman_filter.py:78-88`).

**Predict / update** (KF chuẩn):

$$\hat x^-=F\hat x,\quad P^-=FPF^\top+Q;\qquad S=HP^-H^\top+R,\quad K=P^-H^\top S^{-1},\quad
\hat x=\hat x^-+K(z-H\hat x^-),\quad P=P^--KSK^\top .$$

**Track mất detection**: trước predict đặt ḣ = 0 (`byte_tracker.py:81-82, 93-94`; `CVSTrack` của repo
`src/tracker_ctrv.py:454-456`).

**Phụ thuộc kích thước** [LT]: mọi độ lệch chuẩn vị trí / vận tốc ∝ h ⇒ Q, R ∝ h². Tính theo đơn vị chuẩn hóa x/h,
mô hình bất biến theo tỉ lệ: cùng sai số tương đối cho cùng Mahalanobis. Theo pixel: box h = 20 px có σ_R = 1 px,
box h = 120 px có σ_R = 6 px. Riêng a dùng độ lệch tuyệt đối (Q: 0,01; R: 0,1). Đây là thiết kế heuristic, không ước
lượng từ dữ liệu; nó ngầm giả định sai số detector tỉ lệ với h. [GT] Sai số thực có thể không tỉ lệ với h (lượng tử
pixel ở box nhỏ, cắt khung / che khuất ở box lớn) — giả thuyết H3 trong `02`.

## 3. EKF–CTRV, trạng thái 9-D [MÃ + LT]

**Trạng thái** x = [c_x, c_y, v, θ, ω, a, h, v_a, v_h]ᵀ (`src/ekf_ctrv.py:13-24`): (c_x, c_y) tâm [px]; v tốc độ dài
[px/bước]; θ hướng [rad]; ω tốc độ quay [rad/bước]; a, h, v_a, v_h như mục 2. θ đo từ trục +x của ảnh về phía +y; vì
trục y ảnh hướng xuống, θ tăng (ω > 0) là quay **theo chiều kim đồng hồ** trên màn hình [LT, hệ quả của
`atan2(dy, dx)` trong tọa độ ảnh, `src/ekf_ctrv.py:531`].

**Hàm chuyển trạng thái** (`src/ekf_ctrv.py:366-397`), s₀ = sin θ, c₀ = cos θ, s₁ = sin(θ + ωΔt), c₁ = cos(θ + ωΔt):

$$|\omega|>\varepsilon:\quad c_x'=c_x+\frac{v}{\omega}(s_1-s_0),\qquad c_y'=c_y+\frac{v}{\omega}(c_0-c_1);$$
$$|\omega|\le\varepsilon:\quad c_x'=c_x+v\,\Delta t\,c_0,\qquad c_y'=c_y+v\,\Delta t\,s_0;$$
$$v'=v,\quad \theta'=\mathrm{wrap}(\theta+\omega\Delta t),\quad \omega'=\omega,\quad a'=a+v_a\Delta t,\quad h'=h+v_h\Delta t,\quad v_a'=v_a,\quad v_h'=v_h .$$

Dẫn xuất [LT, tự dẫn]: vận tốc (v cos θ(t), v sin θ(t)), θ(t) = θ + ωt, tích phân từ 0 đến Δt cho đúng hai biểu thức
trên (`src/ekf_ctrv.py:58-65`). CTRV giả định v và ω **không đổi trong một bước** ⇒ quỹ đạo là cung tròn bán kính v/ω.

**Jacobian** F = ∂f/∂x tại x̂ (`src/ekf_ctrv.py:399-432`), khối chéo [F_CTRV (5×5), F_size (4×4)]. Với |ω| > ε:
∂c_x′/∂v = (s₁ − s₀)/ω; ∂c_x′/∂θ = (v/ω)(c₁ − c₀); ∂c_x′/∂ω = −(v/ω²)(s₁ − s₀) + (v/ω)Δt c₁;
∂c_y′/∂v = (c₀ − c₁)/ω; ∂c_y′/∂θ = (v/ω)(s₁ − s₀); ∂c_y′/∂ω = −(v/ω²)(c₀ − c₁) + (v/ω)Δt s₁; ∂θ′/∂ω = Δt;
∂a′/∂v_a = ∂h′/∂v_h = Δt. Với |ω| ≤ ε (giới hạn Taylor): ∂c_x′/∂v = c₀Δt, ∂c_x′/∂θ = −v s₀Δt,
∂c_x′/∂ω = −½ v Δt² s₀; ∂c_y′/∂v = s₀Δt, ∂c_y′/∂θ = v c₀Δt, ∂c_y′/∂ω = ½ v Δt² c₀.

**Predict**: x̂⁻ = f(x̂), P⁻ = F P Fᵀ + Q(h) với (`src/ekf_ctrv.py:437-453, 559-566`)

$$Q=\mathrm{diag}\big((\sigma_p h)^2,(\sigma_p h)^2,(\sigma_v h)^2,\sigma_\theta^2,\sigma_\omega^2,10^{-4},(\sigma_p h)^2,10^{-10},(\sigma_v h)^2\big),$$

σ_θ = 0,010 rad và σ_ω = 0,005 rad/bước là độ lệch chuẩn của gia số cộng vào θ và ω ở **mỗi lần predict**
(`src/config.py:304-305`; `src/ekf_ctrv.py:447-448`; PF dùng cùng giá trị, `src/pf_ctrv.py:188-189`). Q là nhiễu rời
rạc theo bước, cộng **trực tiếp** vào c_x, c_y (giống KF–CV) và vào v, θ, ω; không suy từ một mô hình nhiễu liên tục
của CTRV. Mục đích ghi trong mã: giữ hệ số baseline để chỉ mô hình chuyển động là biến nghiên cứu
(`src/ekf_ctrv.py:309-316`).

**Kiểm tĩnh đơn vị từng thành phần Q** [MÃ + LT]. Vì x_k = f(x_{k−1}) + w_k, mỗi thành phần của w_k có đơn vị của
biến trạng thái tương ứng và Q = Cov(w_k) có đơn vị bình phương. "Theo bước" nghĩa là Q được cộng **một lần mỗi
predict** (bỏ k frame ⇒ cộng k lần, mục 0.3); nó không phải một thừa số trong đơn vị. Độ lệch chuẩn đọc tại
`src/ekf_ctrv.py:444-452`, trùng `src/pf_ctrv.py:185-193`:

| Chỉ số | Biến [đơn vị] | Độ lệch chuẩn gia số mỗi predict | Đơn vị độ lệch chuẩn | Đơn vị phần tử Q | s_i khi đổi sang giây (`src/time_units.py:120`) |
|:--|:--|:--|:--|:--|:--|
| 0, 1 | c_x, c_y [px] | h/20 | px | px² | 1 |
| 2 | v [px/bước] | h/160 | px/bước (hệ số 1/160 mang 1/bước) | (px/bước)² | fps |
| 3 | θ [rad] | σ_θ = 0,010 | **rad** | rad² | 1 |
| 4 | ω [rad/bước] | σ_ω = 0,005 | **rad/bước** | (rad/bước)² | fps |
| 5 | a [–] | 10⁻² | – | – | 1 |
| 6 | h [px] | h/20 | px | px² | 1 |
| 7 | v_a [1/bước] | 10⁻⁵ | 1/bước | (1/bước)² | fps |
| 8 | v_h [px/bước] | h/160 | px/bước | (px/bước)² | fps |

Kiểm chéo [LT, tự dẫn]: nếu đổi đơn vị trạng thái sang giây bằng cùng phép S mà `convert_state_units` áp cho P
(`src/time_units.py:143-157`; S = diag(s_i), `RATE_INDICES` dòng 119-122), nhiễu w thành S w nên Q′ = S Q Sᵀ.
Q_θθ giữ nguyên vì θ không thuộc `RATE_INDICES`, còn Q_ωω nhân fps². Vậy gia số θ mỗi bước vẫn có độ lệch chuẩn 0,010
rad, gia số ω mỗi bước có độ lệch chuẩn 0,005·fps rad/s (25 fps: 0,125 rad/s trên một bước 0,04 s). Kết quả này khớp
với đơn vị [rad] và [rad/bước] trong bảng. Cách viết của chú thích mã (σ_θ [rad/bước] là "nhiễu hướng", σ_ω
[rad/bước²] là "mức thay đổi của turn rate mỗi frame", `src/config.py:304-305`) chia gia số cho Δt = 1 bước để đọc
thành tốc độ thay đổi. Giá trị số không đổi, nhưng đó không phải đơn vị của phần tử Q. Không sửa mã ở R48A.

**Update**: h(x) = Hx tuyến tính (H chọn c_x, c_y, a, h; `src/ekf_ctrv.py:326-331`) ⇒ bước update **trùng KF chuẩn**,
không xấp xỉ (`src/ekf_ctrv.py:610-633`); R như mục 2, tính tại h dự đoán (`src/ekf_ctrv.py:584-608`). Innovation
không chứa θ nên không cần quấn góc trong innovation; θ được quấn sau update (`src/ekf_ctrv.py:630`).

**Bản chất xấp xỉ** [LT]: EKF truyền **điểm trung bình** qua f phi tuyến đầy đủ, còn hiệp phương sai qua tuyến tính
hóa bậc 1; bỏ qua E[f(x)] ≠ f(E[x]) khi độ bất định lớn so với độ cong của f. Ví dụ tự dẫn: nếu θ ~ N(θ̄, σ_θ²) độc
lập với v thì E[v cos θ] = v̄ cos θ̄ · e^{−σ_θ²/2} (hàm đặc trưng của phân bố chuẩn): độ dời trung bình **thật** của
mô hình ngắn hơn độ dời của f(x̂) theo hệ số e^{−σ_θ²/2} (σ_θ = 0,3 → 0,956; σ_θ = 1,0 → 0,607). Ở ví dụ này σ_θ
là độ lệch chuẩn **trạng thái** của θ [rad], không phải độ lệch chuẩn nhiễu quá trình 0,010 rad. PF với ước lượng
`mean` biểu diễn được hiệu ứng này; nó có lợi hay hại cho IoU association là câu hỏi mở (H6 trong `02`).

**Khởi tạo hai khung hình** (`src/ekf_ctrv.py:493-543`; `src/tracker_ctrv.py:177-206`): ở quan sát ghép đầu tiên sau
khi sinh track, tốc độ = ‖Δp‖ / (Δt · n_frames) tính từ **vị trí quan sát** trước; nếu ≥ 0,5 px/bước thì đặt
v = tốc độ, θ = atan2(Δy, Δx), đặt độ lệch chuẩn biên duyên σ_θ = 0,30 rad, σ_v = 20h/160 bằng phép biến đổi tương
hợp giữ hệ số tương quan (`src/ekf_ctrv.py:202-256`); chưa đủ tốc độ thì thử lại ở quan sát sau. Trước đó P₀ có
σ_θ = 1,5 rad, σ_ω = 0,10 rad/bước (`src/config.py:310-311`). Đổi đơn vị [LT]: 0,10 rad/bước ở 25 fps là 2,5 rad/s
(≈ 143°/s) — một prior rất rộng cho ω.

**Giới hạn ω ≈ 0**: ngưỡng ε = 10⁻⁴ rad/bước (`src/config.py:278`). [LT, tự dẫn] Khai triển
(v/ω)(s₁ − s₀) = vΔt c₀ − ½ v ω Δt² s₀ + O(ω²) ⇒ tại |ω| = ε hai nhánh lệch nhau cỡ ½ v ε Δt² (v = 10 px/bước:
5·10⁻⁴ px), không đáng kể; Jacobian giữ số hạng ±½vΔt² theo ω để EKF vẫn học được ω khi xe đi thẳng
(`src/ekf_ctrv.py:114-127`). PF vector hóa bằng `np.where` với ω_safe = 1 ở nhánh thẳng để tránh chia cho 0
(`src/pf_ctrv.py:165-175`).

**Xử lý góc**: wrap(θ) = ((θ + π) mod 2π) − π ∈ [−π, π) (`src/ekf_ctrv.py:193-199`), áp dụng sau f
(`src/ekf_ctrv.py:390`; `src/pf_ctrv.py:176`), sau update EKF (`src/ekf_ctrv.py:630`), sau mọi bước lấy mẫu của PF
(`src/pf_ctrv.py:203, 211, 251, 305`; `src/tracker_ctrv.py:319`). Trung bình θ của PF / UKF là **trung bình vòng**
atan2(Σw sin θ, Σw cos θ) và phần dư θ được quấn trước khi tính hiệp phương sai (`src/pf_ctrv.py:227-231`;
`src/ukf_ctrv.py:191-205`). Lý do [LT]: trung bình số học của π − 0,05 và −π + 0,05 là 0 (ngược hướng), trung bình vòng
là ±π.

**Track mất detection**: chỉ đặt v_h = 0, **không** xóa ω — cố ý để CTRV tiếp tục quay khi bị che giữa khúc cua
(`src/tracker_ctrv.py:26-35, 98-104`).

## 4. UKF–CTRV (để đối chiếu) [MÃ]

Cùng f, Q, R, P₀ và khởi tạo hai khung hình với EKF (`src/time_units.py:55-58`). Thay Jacobian bằng 2n + 1 = 19 sigma
point, α = 1, β = 2, κ = 0 ⇒ λ = 0 (`src/config.py:337-339`; `src/ukf_ctrv.py:113-153`); trung bình θ dạng vòng, phần
dư quấn góc (`src/ukf_ctrv.py:191-205`). UKF vẫn biểu diễn hậu nghiệm bằng **một** Gauss; chỉ cách lan truyền trung bình
/ hiệp phương sai khác EKF.

## 5. PF–CTRV (bootstrap / SIR) [MÃ + LT]

Mỗi track j giữ một đám mây {X_k^{(i)}, w_k^{(i)}}, i = 1..N, N ∈ {100, 300, 500} theo cấu hình
`configs/phase2/trackers/bytetrack_pf_ctrv_n{100,300,500}.yaml:12`.

**Mô hình lý tưởng chung với EKF** [LT]. Hai bộ lọc nhắm cùng một mô hình: x_k = f(x_{k−1}) + w_k,
w_k ~ N(0, Q(h_{k−1})); z_k = H x_k + v_k, v_k ~ N(0, R(h_k)); θ trên đường tròn; x₀ ~ N(x̂₀, P₀); f, Q(·), R(·), P₀
như mục 2–3 (Q, R phụ thuộc trạng thái qua h). Docstring `src/pf_ctrv.py:4-11` (và `src/time_units.py:60-63`) nói hai
bộ lọc chỉ khác cách biểu diễn hậu nghiệm. Câu đó chỉ đúng **ở mức mô hình này**: một Gauss so với một đám mây có
trọng số. **Triển khai** còn các xấp xỉ và heuristic khác nhau giữa hai bộ lọc, liệt kê ngay dưới bảng (heuristic
1–6 ứng với ý 1, 3, 7, 4, 5, 8 của `02` mục 5). Vì vậy so sánh EKF / PF trong pipeline đo gộp cả khác biệt biểu diễn
lẫn các heuristic này.

| Bước | Công thức | Mã |
|:--|:--|:--|
| Khởi tạo | X₀^{(i)} ~ N(x̂₀, P₀) (P₀ chéo), quấn θ, chặn h ≥ 10⁻³ (heuristic 3); w = 1/N | `src/pf_ctrv.py:199-205`; `src/tracker_ctrv.py:282-284` |
| Khởi tạo 2 khung (heuristic 4) | quyết định theo luật EKF trên (mean, cov) tổng hợp; rồi v^{(i)} ~ N(tốc độ, (20h/160)²), θ^{(i)} ~ N(hướng, 0,3²); ω, c_x, c_y, a, h giữ nguyên; w = 1/N | `src/pf_ctrv.py:294-306, 350-366` |
| Đề xuất (= prior) | X_k^{(i)} = f(X_{k−1}^{(i)}) + ε^{(i)}, ε^{(i)} ~ N(0, Q(h_{k−1}^{(i)})) — Q tại h của **từng** particle (heuristic 2); quấn θ; chặn h ≥ 10⁻³ (heuristic 3); trọng số **không đổi** | `src/pf_ctrv.py:207-213`; `src/tracker_ctrv.py:304-323` |
| Likelihood | ℓ^{(i)} = −½ Σ_{d∈{c_x,c_y,a,h}} (z_d − X_d^{(i)})² / R_dd, R = diag((h_z/20)², (h_z/20)², 0,1², (h_z/20)²) với h_z = h **quan sát** (heuristic 1) | `src/pf_ctrv.py:215-219, 262-264` |
| Cập nhật trọng số | log w^{(i)} ← log w^{(i)} + ℓ^{(i)}, chuẩn hóa bằng log-sum-exp | `src/pf_ctrv.py:282-283` |
| Ước lượng điểm + ma trận | `mean`: Σ w̃^{(i)} X^{(i)} (θ trung bình vòng). `map`: particle có trọng số lớn nhất X^{(argmax w̃)} — tên tùy chọn, **không** phải MAP của mật độ hậu nghiệm (heuristic 6). Ma trận = Σ w̃^{(i)} d^{(i)} d^{(i)ᵀ}, d^{(i)} = X^{(i)} − ước lượng điểm, phần dư θ quấn: hiệp phương sai khi dùng `mean`, mô-men bậc hai quanh particle được chọn khi dùng `map`. Trong update tính **trước** resample; sau predict tính lại với trọng số hiện có | `src/pf_ctrv.py:221-234, 285`; `src/tracker_ctrv.py:322`; `src/pf_ctrv.py:336` |
| Resample | ESS = 1/Σ w̃² trên trọng số đã nhân likelihood; nếu ESS < 0,5 N: systematic resampling, w = 1/N; roughening tùy chọn (mặc định 0 = tắt) | `src/pf_ctrv.py:74-80, 240-253` |
| Luật suy biến (heuristic 5) | nếu min_i Mahalanobis² (chỉ theo R) > 10⁴: đặt lại c_x, c_y, a, h quanh z với độ lệch P₀, giữ v, θ, ω, w đều | `src/pf_ctrv.py:265-281` |
| Mất detection | chỉ propagate; v_h = 0 cho mọi particle trước propagate; trọng số không đổi | `src/pf_ctrv.py:323-336`; `src/tracker_ctrv.py:306-311` |
| Bbox cho association | tlwh từ track.mean (c_x, c_y, a, h) = ước lượng ở trên ⇒ IoU | `src/tracker_ctrv.py:158-172` |

**Xấp xỉ và heuristic của triển khai** (không suy ra từ mô hình lý tưởng ở đầu mục) [MÃ + LT]:
1. **R theo h quan sát.** Mô hình có R(h_k) phụ thuộc trạng thái. PF dùng R(h_z), h_z là h **quan sát**, cho mọi
   particle (`src/pf_ctrv.py:262-263`); EKF dùng R(ĥ⁻) tại h **dự đoán** (`src/ekf_ctrv.py:604`). Đây là hai phép thế
   khác nhau cho cùng một R(h). Với R(h_z), hằng số chuẩn hóa −½ log|2πR| như nhau cho mọi particle nên bỏ được
   (`src/pf_ctrv.py:215-219` bỏ hằng số). Nếu dùng đúng R(h_k^{(i)}) của mô hình thì hằng số này phụ thuộc particle và
   không bỏ được. Mức ảnh hưởng chưa đo.
2. **Q theo h của từng particle.** PF lấy Q tại h_{k−1}^{(i)} (`src/pf_ctrv.py:209`; `src/tracker_ctrv.py:314`), khớp
   với Q(h_{k−1}) của mô hình; EKF / UKF lấy Q tại h ước lượng (`src/ekf_ctrv.py:562`; UKF theo
   `src/time_units.py:56`), tức phép thế tại trung bình.
3. **Chặn h ≥ 10⁻³** bằng `np.maximum` sau lấy mẫu khởi tạo, sau propagate và trong nhánh suy biến
   (`src/pf_ctrv.py:204, 212, 274`; `src/tracker_ctrv.py:320`), cùng với h dùng để tính độ lệch chuẩn Q
   (`src/pf_ctrv.py:209, 249`; `src/tracker_ctrv.py:314`). EKF không chặn h (`src/ekf_ctrv.py` không có `1e-3` hay
   `maximum`). Đây là phép **chặn**: mẫu có h < 10⁻³ bị đặt đúng bằng 10⁻³, nên phân bố của h sau bước này có một
   **khối xác suất dồn tại biên** 10⁻³. Nó **không** phải phân bố Gauss cắt cụt (Gauss có điều kiện trên miền
   h > 10⁻³ rồi chuẩn hóa lại, không có khối tại biên). Mô hình lý tưởng với nhiễu Gauss cũng không bảo đảm h > 0;
   phép chặn là bảo vệ số học của triển khai, và trọng số không được hiệu chỉnh cho phần đề xuất thực tế lệch khỏi
   N(f, Q) ở thành phần h. [LT] Khi v_h ≈ 0, một bước cần nhiễu cỡ −20σ (σ = h/20) mới chạm biên, nên xác suất không
   đáng kể. [GT] Chỉ particle có v_h âm lớn kéo dài nhiều bước mới chạm biên; tần suất chưa đo.
4. **Khởi tạo hai khung.** PF lấy mẫu lại v, θ của từng particle và giữ ω (`src/pf_ctrv.py:294-306`); EKF đặt lại
   phương sai biên, giữ hệ số tương quan (`src/ekf_ctrv.py:202-256, 493-543`).
5. **Luật suy biến** (chỉ PF; `src/pf_ctrv.py:265-281`) là **khởi tạo lại**, không phải một phép cập nhật Bayes; ngưỡng
   10⁴ ứng với 100σ_R. [QS] Tỉ lệ kích hoạt trên dev = 0,0000 (`handoff/logs/pf_diag_report.log`, cột
   `frac_degenerate`).
6. **Tùy chọn `pf_estimate: map`** (`src/pf_ctrv.py:224-225`) chọn particle có trọng số chuẩn hóa lớn nhất,
   X^{(i*)} với i* = argmax_i w̃^{(i)}. Đây **không** phải MAP (mode) của mật độ hậu nghiệm liên tục, vì trọng số không
   phải giá trị mật độ. Với đề xuất prior, w̃^{(i)} ∝ w_{k−1}^{(i)} p(z_k | X^{(i)}); mật độ dự đoán chỉ thể hiện qua mật
   độ của mẫu, không nằm trong trọng số. Vì thế argmax w̃ là particle hợp quan sát nhất trong mẫu, không phải mode của
   p(x_k | z_{1:k}). Muốn tìm mode phải ước lượng mật độ (ví dụ bằng kernel); mã không làm việc đó. Hai hệ quả [MÃ]:
   (a) `np.argmax` trả chỉ số **đầu tiên** khi các trọng số bằng nhau. Trọng số đều ngay trong nhánh suy biến
   (`src/pf_ctrv.py:275-276`) và sau reseed hai khung (`src/pf_ctrv.py:359-360`); chúng cũng đều ở mọi lần predict sau
   một update có resample (w = 1/N; [QS] 98,4–99,98 % số lần update trên dev, mục 7.6), cho tới update kế tiếp. Ở các
   lần gọi đó `estimate_state` trả particle hàng 0, tức một particle tùy ý. Trong số đó có lần gọi **sau predict**
   (`src/tracker_ctrv.py:322`; `src/pf_ctrv.py:336`), là lần tạo bbox cho association ở frame sau. Chú thích
   "'map' có nghĩa" ở `src/pf_ctrv.py:285` chỉ đúng cho lần gọi trong update bình thường, trước resample. (b) Với `map`, ma trận trả về Σ_i w̃^{(i)} (X^{(i)} − X^{(i*)})(X^{(i)} − X^{(i*)})ᵀ là mô-men bậc hai quanh
   particle được chọn. [LT, tự dẫn] Với các thành phần không quấn góc, nó bằng Σ_w + (μ_w − X^{(i*)})(μ_w − X^{(i*)})ᵀ
   (μ_w, Σ_w là trung bình và hiệp phương sai có trọng số), không phải hiệp phương sai quanh trung bình.
   Cả ba cấu hình PF đặt `pf_estimate: mean` (`configs/phase2/trackers/bytetrack_pf_ctrv_n{100,300,500}.yaml:15`; grep
   `pf_estimate` trong `configs/` chỉ ra ba dòng này) và mặc định của `PFCTRV` là `mean` (`src/pf_ctrv.py:88`), nên
   không cấu hình PF nào tại HEAD dùng `map`. Chỉ đính chính tài liệu; thuật toán giữ nguyên.

Nhận xét [LT]: mật độ chuyển trạng thái của mô hình lý tưởng là p(x_k | x_{k−1}) = N(f(x_{k−1}), Q(h_{k−1})); θ sau
khi quấn có phân bố chuẩn quấn trên đường tròn (quấn không làm mất xác suất). Với track đang bám, đề xuất thực tế chỉ
khác mật độ này ở phép chặn h (heuristic 3). Với track LOST, v_h bị đặt 0 trước propagate; đây là quy tắc của
ByteTrack, áp như nhau cho mọi bộ lọc (dòng "Mất detection" trong bảng).

## 6. Particle khác random seed [LT + MÃ]

| | Particle | Random seed |
|:--|:--|:--|
| Là gì | Một mẫu X^{(i)} ∈ ℝ⁹ — một giả thuyết trạng thái của **một** track — kèm trọng số w^{(i)} | Số nguyên khởi tạo bộ sinh số ngẫu nhiên; quyết định **một thực hiện** của mọi phép ngẫu nhiên |
| Vai trò | N particle / track xấp xỉ hậu nghiệm; N lớn hơn ⇒ sai số Monte Carlo của **một** lần chạy nhỏ hơn | Đổi seed ⇒ cùng thuật toán, cùng N, khác mẫu ban đầu, nhiễu quá trình, số u của systematic resampling, roughening, reseed |
| Trong mã | `num_particles` (`src/pf_ctrv.py:95`; yaml `pf_num_particles`) | rng của track = `default_rng(SeedSequence([pf_seed, video_salt, track_id]))` (`src/pf_ctrv.py:149-154`; `src/tracker_ctrv.py:282`); runner ghi đè `pf_seed` bằng `--seeds` (`src/run_tracking.py:198-199`) |

Hệ quả:
- Nhiều seed **không** phải nhiều particle: 3 seed × N = 300 là ba lần chạy độc lập, mỗi lần 300 particle/track,
  không phải một bộ lọc 900 particle. Độ phân tán giữa các seed đo **biến thiên Monte Carlo** của bộ lọc (và của các
  quyết định ghép rời rạc mà nó kéo theo), không đo độ bất định của dữ liệu.
- Cùng seed + cùng video ⇒ kết quả bit-identical; kết quả không phụ thuộc thứ tự track trong pool
  (`src/test_pf_ctrv.py` test 6, 7 — chạy lại trong R48A: 26/26 PASS, exit 0).
- [LT, hệ quả của mã] rng của track phụ thuộc `track_id`, mà `track_id` phụ thuộc lịch sử association. Hai biến thể PF
  chạy cùng seed chỉ dùng chung số ngẫu nhiên cho tới khi lịch sử ghép (và do đó track_id) tách nhau ⇒ "ghép cặp seed"
  trong pipeline thật không phải common random numbers hoàn toàn; trong mô phỏng biết liên kết thì ghép cặp được trọn vẹn.

## 7. Importance sampling, SIS, SIR, suy biến, ESS [LT]

Khung chung theo [A02] (chưa đối chiếu toàn văn). Mọi phương trình dưới đây được dẫn lại trong tài liệu này.

### 7.1 Importance sampling
Với mật độ đích π (chỉ cần biết sai khác một hằng số) và mật độ đề xuất q có giá phủ π:
E_π[g] = ∫ g (π/q) q dx ≈ Σ_i w̃^{(i)} g(x^{(i)}), x^{(i)} ~ q, w^{(i)} ∝ π(x^{(i)})/q(x^{(i)}), w̃ = w / Σ w.

### 7.2 SIS — cập nhật trọng số tổng quát [LT, tự dẫn]
Đích trên không gian đường đi: π_k(x_{0:k}) = p(x_{0:k} | z_{1:k}) ∝ p(z_k | x_k) p(x_k | x_{k−1}) p(x_{0:k−1} | z_{1:k−1})
(tính Markov và z_k độc lập có điều kiện khi biết x_k). Đề xuất tuần tự q(x_{0:k} | z_{1:k}) = q(x_k | x_{k−1}, z_k) ·
q(x_{0:k−1} | z_{1:k−1}). Lấy tỉ số:

$$w_k^{(i)}\;\propto\;w_{k-1}^{(i)}\;\frac{p(z_k\mid x_k^{(i)})\;p(x_k^{(i)}\mid x_{k-1}^{(i)})}{q(x_k^{(i)}\mid x_{k-1}^{(i)},z_k)},\qquad
p(x_k\mid z_{1:k})\approx\sum_i \tilde w_k^{(i)}\,\delta\big(x_k-x_k^{(i)}\big).$$

### 7.3 Đề xuất bằng prior (bootstrap)
q = p(x_k | x_{k−1}) ⇒ tỉ số chuyển trạng thái / đề xuất bằng 1:

$$w_k^{(i)}\propto w_{k-1}^{(i)}\,p(z_k\mid x_k^{(i)});\qquad\text{nếu đã resample ở bước }k-1\;(w_{k-1}=1/N):\; w_k^{(i)}\propto p(z_k\mid x_k^{(i)}).$$

Ưu: chỉ cần lấy mẫu từ động học và tính likelihood. Nhược: đề xuất bỏ qua z_k; khi likelihood hẹp so với đám mây dự
đoán (sau mất dấu dài, hoặc R nhỏ) phần lớn particle rơi vào vùng likelihood thấp ⇒ ESS thấp. **Mã hiện tại là đúng
trường hợp này** (`src/pf_ctrv.py:17-18, 282`). Nếu đổi đề xuất (ví dụ dùng bước update của EKF/UKF làm q) thì phải
nhân đúng tỉ số p(z|x) p(x|x′)/q như 7.2; bỏ tỉ số sẽ làm ước lượng lệch.

### 7.4 Suy biến trọng số
Với SIS không resample, sau vài bước gần như toàn bộ trọng số dồn vào một particle; [A02] nêu phương sai của trọng số
quan trọng chỉ có thể tăng theo thời gian (chưa đối chiếu toàn văn). Đo mức suy biến bằng ESS.

### 7.5 ESS [LT, tự dẫn]
$$\widehat{N}_{\rm eff}=\frac{1}{\sum_i(\tilde w^{(i)})^2},\qquad 1\le \widehat N_{\rm eff}\le N .$$
Chứng minh: Cauchy–Schwarz cho Σ w̃² ≥ (Σ w̃)²/N = 1/N; và Σ w̃² ≤ max w̃ · Σ w̃ ≤ 1. Bằng N khi và chỉ khi trọng số đều;
bằng 1 khi một particle giữ toàn bộ trọng số. Khi N lớn, N̂_eff / N → (E_q[w])² / E_q[w²] (w chưa chuẩn hóa).

**Ví dụ giải tích** (1 chiều, innovation bằng 0): đám mây dự đoán N(0, σ_p²), likelihood exp(−x²/(2σ_r²)). Tích phân
Gauss cho E[w] = σ_r/√(σ_r² + σ_p²), E[w²] = σ_r/√(σ_r² + 2σ_p²), nên

$$\frac{N_{\rm eff}}{N}\to\rho(r)=\frac{\sqrt{1+2r^2}}{1+r^2},\qquad r=\sigma_p/\sigma_r;\qquad
\rho(1)=0{,}866,\;\rho(2)=0{,}600,\;\rho(5)=0{,}275,\;\rho(r)\approx\sqrt2/r\;(r\gg1).$$

Với d chiều độc lập mang thông tin: ρ^d. Innovation khác 0 (lệch mô hình, lệch detector) làm ESS thấp hơn nữa.

### 7.6 Resampling, SIR, systematic resampling [LT, tự dẫn]
Rút N chỉ số a^{(i)} với P(a = j) = w̃^{(j)}, đặt X^{(i)} ← X^{(a^{(i)})}, w = 1/N; không chệch: E[số bản sao của j] =
N w̃^{(j)}. Systematic: một u ~ U[0, 1), vị trí (m + u)/N, m = 0..N−1; số bản sao của j là số vị trí rơi vào
[C_{j−1}, C_j) (C là tổng tích lũy), đoạn dài w̃^{(j)} với bước 1/N ⇒ số bản sao ∈ {⌊N w̃^{(j)}⌋, ⌈N w̃^{(j)}⌉}.
Mã: `src/pf_ctrv.py:74-80` (`searchsorted(..., side="right")`, chặn sai số làm tròn bằng `cumsum[-1] = 1`). SIR =
SIS với đề xuất prior + resampling; [A02] gọi bộ lọc resample mỗi bước với đề xuất prior là bootstrap filter (chưa đối
chiếu toàn văn). Mã resample **theo ngưỡng** ESS < 0,5 N; [QS] trên dev, resample xảy ra ở 98,4–99,98 % số lần update
(`handoff/logs/pf_diag_report.log`, cột `frac_resampled`) ⇒ thực tế gần như SIR mỗi bước.

### 7.7 Mất đa dạng sau resampling (sample impoverishment)
Resampling nhân bản particle trọng số lớn, bỏ particle trọng số nhỏ ⇒ số giá trị khác nhau giảm. ESS tính **sau**
resample luôn bằng N (trọng số vừa đặt đều) nên **không** chứng minh đám mây đa dạng; phải đo số tổ tiên khác nhau / N,
số giá trị khác nhau (`n_unique` của c_x trong chẩn đoán, `src/pf_ctrv.py:116-119, 289`), độ phân tán. Đa dạng chỉ được
bù lại bằng nhiễu quá trình ở predict kế tiếp; thành phần có nhiễu nhỏ so với áp lực chọn lọc (ở đây độ lệch chuẩn
gia số mỗi predict: θ 0,01 rad; ω 0,005 rad/bước) hồi phục chậm ⇒ có thể trôi ngẫu nhiên theo các lần chọn lọc (định tính [LT]; mức độ trong pipeline
là [GT] H5). Biện pháp đã biết (liệt kê, **chưa chọn**): roughening / regularization (mã đã có `pf_roughening`, mặc định
tắt), bước MCMC, đề xuất có dùng z_k, tăng N, đổi ngưỡng resample ([A02] bàn về regularized PF / MCMC và chọn mật độ
đề xuất — chưa đối chiếu toàn văn).

[QS] Số đo đã có trên dev UA-DETRAC (`handoff/logs/pf_diag_report.log`, ghi tới `results/phase2/pf_diag/expA_detrac_dev`;
c10 / c25, N 100 / 300 / 500, 3 seed): khi đang bám (n_lost = 0) ESS/N trước resample ≈ 0,25–0,26 và n_unique/N sau
resample ≈ 0,38–0,39; sau 11–30 bước mất dấu ESS/N ≈ 0,007–0,017 và n_unique/N ≈ 0,027–0,035.

[GT, kiểm tra nhất quán, không phải chứng minh] Theo 7.5 với 3 chiều mang thông tin (c_x, c_y, h; chiều a có
σ_R = 0,1 rộng so với đám mây nên ρ ≈ 1) và innovation 0: ESS/N ≈ 0,25 ứng với r ≈ 1,9 (ρ(1,9)³ ≈ 0,24). Sau k = 20
bước mất, riêng nhiễu cộng vị trí đã cho σ_p ≥ √20 σ_Q = 4,5 σ_R (vì σ_Q = σ_R = h/20) ⇒ ρ³ ≈ 0,03 — cùng bậc với số đo,
số đo thấp hơn (phù hợp với phần phân tán thêm từ v, θ, ω và innovation lệch). Chưa kiểm định định lượng.

### 7.8 Chuẩn hóa log-weight (log-sum-exp) [LT, tự dẫn]
$$\log\tilde w^{(i)}=\ell^{(i)}-\mathrm{LSE}(\ell),\qquad \mathrm{LSE}(\ell)=m+\log\sum_j e^{\ell^{(j)}-m},\quad m=\max_j\ell^{(j)} .$$
Đẳng thức đúng tuyệt đối về đại số. Lý do dùng: ℓ có thể xuống −10³…−10⁴ (Mahalanobis² tới 10⁴ trước luật suy biến),
mà exp(x) trong float64 về 0 khi x dưới khoảng −745 ⇒ Σ exp(ℓ) = 0 ⇒ 0/0. Sau khi trừ m, số hạng lớn nhất bằng
e⁰ = 1 nên tổng thuộc [1, N], log hữu hạn. Mã: `_logsumexp` (`src/pf_ctrv.py:67-71`), dùng trong update (283), ước lượng
(223), ESS (237), resample (243). Trường hợp mọi ℓ = −∞ không xảy ra với likelihood Gauss và đầu vào hữu hạn; nếu xảy ra
thì update rơi vào nhánh suy biến (`src/pf_ctrv.py:267`).

## 8. Khi nào PF có thể có lợi — giả thuyết và giới hạn

[LT] PF xấp xỉ được hậu nghiệm bất kỳ (lệch Gauss, nhiều mode) khi N → ∞; EKF / UKF luôn ép về một Gauss.
[GT] Lợi thế chỉ kỳ vọng khi đồng thời: (a) hậu nghiệm thật sự lệch Gauss — ví dụ hướng chưa rõ lúc sinh track, mất dấu
dài **trong khi cua** với ω và v bất định (mật độ dự đoán cong hình "quả chuối"); (b) phi tuyến mạnh so với độ bất định
(độ cong của f trên vùng hiệp phương sai, cỡ v·Δt·σ_θ², v·Δt²·σ_ω, với σ_θ, σ_ω ở đây là độ lệch chuẩn **trạng
thái** của θ, ω, không phải của nhiễu quá trình).

Giới hạn trong bài toán này:
1. Quan sát mỗi frame ở 25 fps giữ hậu nghiệm gọn ⇒ gần Gauss ⇒ PF ≈ EKF cộng nhiễu Monte Carlo.
2. Mô hình CTRV (v, ω hằng trong một bước; chuyển động trên mặt phẳng ảnh có phối cảnh) sai thì cả hai bộ lọc cùng
   chịu; PF không sửa được sai mô hình.
3. N = 100–500 cho trạng thái 9 chiều với đề xuất prior ⇒ dễ nghèo hạt (mục 7.7).
4. Association IoU chỉ dùng **một** bbox (`mean`, hoặc particle trọng số lớn nhất nếu đặt `map`) ⇒ thông tin lệch
   Gauss của đám mây bị bỏ ở bước ghép.
5. Q, R được chọn cho EKF ở Giai đoạn 1 và dùng chung cho PF để cô lập cách biểu diễn **ở mức mô hình**. Cách này công
   bằng về giả thiết nhưng không tối ưu cho PF; ở mức triển khai vẫn còn các heuristic khác nhau giữa hai bộ lọc
   (mục 5).

**Không** khẳng định PF luôn thắng EKF khi cua. [QS] Ở mốc 2 trên 40 video UA-DETRAC test, PF–CTRV kém rõ rệt so với
EKF (`handoff/PLAN_DA_CHOT.md` mục 0; `CODEX_HANDOFF.md` mục 5, 8). Dự đoán kiểm được (để giao thức R48C quyết định,
**chưa chạy**): trong mô phỏng biết liên kết, nếu PF có lợi thì lợi thế phải xuất hiện ở dự đoán nhiều bước khi mất đo
dài trên quỹ đạo cua, tăng theo N, và biến mất ở chuyển động thẳng / khoảng mất ngắn.

## 9. Đã kiểm trong R48A

| Kiểm | Lệnh | Kết quả |
|:--|:--|:--|
| Test PF có sẵn (f_batch = f, ω ≈ 0, trung bình vòng, mất đo, suy biến, seed, thứ tự pool, resample, nhất quán với EKF ở N = 3000, vòng đời, runner) | `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/test_pf_ctrv.py` | exit 0, 26/26 PASS. **Lệch phạm vi**: test 11 sẵn có ghi file ngoài `write_paths` của R48A — xem `02` mục 6.1 |
| Sở hữu trạng thái particle | `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/kiem_so_huu_hat.py` | exit 0 — chi tiết `02` mục 3 |
| Đơn vị từng thành phần Q (vòng sửa 1, kiểm tĩnh) | không chạy lệnh; đọc `src/ekf_ctrv.py:444-452`, `src/pf_ctrv.py:185-193`, `src/time_units.py:119-122, 143-157` | bảng ở mục 3 |

Chưa kiểm: nội dung [A02] (không có toàn văn); các [GT] ở mục 5, 7.7, 8 và trong `02` mục 4.

## 10. Đính chính sau review Codex vòng 1 (R48A-F1…F4)

Review: `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_001.json` (verdict `revise`).
Vòng sửa chỉ đổi tài liệu (01, 02, mục 0 của `handoff/PLAN_DA_CHOT.md`); không sửa mã, cấu hình hay bằng chứng sở hữu,
không chạy lại test (đúng nhiệm vụ của vòng sửa).

| Mã | Nội dung review | Đã sửa ở |
|:--|:--|:--|
| F1 | σ_θ, σ_ω là độ lệch chuẩn của gia số rời rạc ([rad], [rad/bước]), không phải [rad/bước], [rad/bước²] | mục 0.3; mục 2 (đơn vị Q của KF–CV); mục 3 (bảng đơn vị từng thành phần Q, kiểm chéo bằng S Q Sᵀ, ví dụ co trung bình dùng σ trạng thái); mục 7.7; mục 8 (σ trạng thái khác σ nhiễu); `02` H6 |
| F2 | "PF chỉ khác EKF ở biểu diễn hậu nghiệm" mâu thuẫn với các khác biệt triển khai; phép chặn h không phải cắt cụt | mục 5: tách mô hình lý tưởng khỏi 6 heuristic triển khai; heuristic 3 (phép chặn tạo khối xác suất tại biên); mục 8 ý 5; `02` mục 5 |
| F3 | Tùy chọn `map` = particle trọng số lớn nhất, không phải MAP; argmax khi trọng số đều; ma trận quanh particle được chọn | bảng mục 5 (dòng ước lượng); heuristic 6; mục 8 ý 4; `02` mục 2, mục 4 (H2), mục 5 |
| F4 | Test 11 ghi file ngoài `write_paths` R48A, chưa ghi rõ là lệch phạm vi | mục 9; `02` mục 6.1; mục 0 của `handoff/PLAN_DA_CHOT.md` |
